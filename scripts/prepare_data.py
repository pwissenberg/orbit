#!/usr/bin/env python
"""Data preparation pipeline for SPACE alignment.

Usage:
    uv run python scripts/prepare_data.py --all
    uv run python scripts/prepare_data.py --preprocess
    uv run python scripts/prepare_data.py --convert
    uv run python scripts/prepare_data.py --node2vec
    uv run python scripts/prepare_data.py --orthologs
    uv run python scripts/prepare_data.py --config
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

from loguru import logger

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Default paths
RAW_NETWORKS_DIR = ROOT / "data" / "networks_raw" / "tsv_files"
CLEAN_NETWORKS_DIR = ROOT / "data" / "networks"
SPACE_NETWORKS_DIR = ROOT / "data" / "networks_space"
NODE2VEC_DIR = ROOT / "data" / "node2vec"
ORTHOLOGS_DIR = ROOT / "data" / "orthologs"
TRANSCRIPTS_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"
SPACE_CONFIG_DIR = ROOT / "data"


def load_seed_info() -> dict:
    """Load seed selection results."""
    with open(SEED_RESULTS) as f:
        return json.load(f)


def get_valid_species() -> list[str]:
    """Get species codes that have both networks and OrthoFinder data."""
    from orbit.data_prep import (
        list_species_with_networks,
        list_species_with_orthogroups,
    )

    net_species = set(list_species_with_networks(RAW_NETWORKS_DIR))
    og_species = set(list_species_with_orthogroups(TRANSCRIPTS_DIR))
    valid = sorted(net_species & og_species)
    logger.info(
        f"Species: {len(net_species)} with networks, {len(og_species)} with OrthoFinder, "
        f"{len(valid)} overlap"
    )
    return valid


def step_preprocess(species: list[str]) -> None:
    """Step 1a: Preprocess raw networks to clean TSV."""
    from orbit.data_prep import preprocess_network

    logger.info(f"Preprocessing {len(species)} networks...")
    for code in species:
        # Find raw file (has full species name in filename)
        matches = list(RAW_NETWORKS_DIR.glob(f"{code}_*_Top50EdgesPerGene_ProteinID.tsv"))
        if not matches:
            logger.warning(f"  No raw network for {code}, skipping")
            continue
        raw_path = matches[0]
        out_path = CLEAN_NETWORKS_DIR / f"{code}.tsv"
        preprocess_network(raw_path, out_path)
    logger.info(f"Done: {len(list(CLEAN_NETWORKS_DIR.glob('*.tsv')))} clean networks")


def step_convert(species: list[str]) -> None:
    """Step 1b: Convert clean TSV to SPACE gzipped format."""
    from orbit.data_prep import convert_to_space_format

    logger.info(f"Converting {len(species)} networks to SPACE format...")
    for code in species:
        net_path = CLEAN_NETWORKS_DIR / f"{code}.tsv"
        if not net_path.exists():
            logger.warning(f"  No clean network for {code}, skipping")
            continue
        out_path = SPACE_NETWORKS_DIR / f"{code}.txt.gz"
        convert_to_space_format(net_path, out_path)
    logger.info(f"Done: {len(list(SPACE_NETWORKS_DIR.glob('*.txt.gz')))} SPACE networks")


def _generate_one(args: tuple) -> str:
    """Worker function for parallel node2vec generation."""
    from orbit.data_prep import generate_embeddings

    code, gz_path, h5_path = args
    generate_embeddings(gz_path, h5_path, workers=8)
    return code


def step_node2vec(species: list[str]) -> None:
    """Step 1c: Generate Node2Vec embeddings (parallel, 6 workers × 8 cores)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(f"Generating Node2Vec embeddings for {len(species)} species...")
    tasks = []
    for code in species:
        gz_path = SPACE_NETWORKS_DIR / f"{code}.txt.gz"
        if not gz_path.exists():
            logger.warning(f"  No SPACE network for {code}, skipping")
            continue
        h5_path = NODE2VEC_DIR / f"{code}.h5"
        if h5_path.exists():
            logger.info(f"  {code}: already exists, skipping")
            continue
        tasks.append((code, gz_path, h5_path))

    if not tasks:
        logger.info("  All embeddings already exist.")
        return

    logger.info(f"  Running {len(tasks)} species with 6-way parallelism...")
    done = 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_generate_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                future.result()
                logger.info(f"  [{done}/{len(tasks)}] {code}: done")
            except Exception as e:
                logger.error(f"  [{done}/{len(tasks)}] {code}: FAILED — {e}")


def step_orthologs(species: list[str]) -> None:
    """Step 1d: Build ortholog pair files."""
    from orbit.data_prep import build_ortholog_pairs

    seed_info = load_seed_info()
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]

    # Seed pairs (C(5,2) = 10)
    seed_pairs = list(combinations(sorted(seeds), 2))
    logger.info(f"Building {len(seed_pairs)} seed-seed ortholog pairs...")
    seeds_dir = ORTHOLOGS_DIR / "seeds"
    for sp_a, sp_b in seed_pairs:
        out_path = seeds_dir / f"{sp_a}_{sp_b}.tsv"
        n = build_ortholog_pairs(sp_a, sp_b, NODE2VEC_DIR, TRANSCRIPTS_DIR, out_path)
        logger.info(f"  {sp_a}_{sp_b}: {n} pairs")

    # Non-seed pairs (each non-seed × all seeds in its group)
    nonseeds = [s for s in species if s not in seeds]
    logger.info(f"Building ortholog pairs for {len(nonseeds)} non-seed species...")
    for i, ns in enumerate(nonseeds):
        nearest_seed = groups.get(ns)
        if nearest_seed is None:
            logger.warning(f"  {ns}: no group assignment, skipping")
            continue
        ns_dir = ORTHOLOGS_DIR / "non_seeds" / ns
        # Build pairs with ALL seeds (SPACE needs non-seed aligned to seed group)
        for seed in sorted(seeds):
            pair_key = "_".join(sorted([ns, seed]))
            out_path = ns_dir / f"{pair_key}.tsv"
            n = build_ortholog_pairs(
                min(ns, seed), max(ns, seed),
                NODE2VEC_DIR, TRANSCRIPTS_DIR, out_path,
            )
        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(nonseeds)}] done")

    logger.info("Ortholog pair generation complete")


def step_config(species: list[str]) -> None:
    """Step 1e: Write SPACE config files."""
    from orbit.data_prep import write_space_config

    seed_info = load_seed_info()
    write_space_config(
        seeds=seed_info["seeds"],
        groups=seed_info["groups"],
        output_dir=SPACE_CONFIG_DIR,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare data for SPACE alignment")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Step 1a: preprocess networks")
    parser.add_argument("--convert", action="store_true", help="Step 1b: convert to SPACE format")
    parser.add_argument("--node2vec", action="store_true", help="Step 1c: generate embeddings")
    parser.add_argument("--orthologs", action="store_true", help="Step 1d: build ortholog pairs")
    parser.add_argument("--config", action="store_true", help="Step 1e: write SPACE config")
    args = parser.parse_args()

    if not any([args.all, args.preprocess, args.convert, args.node2vec, args.orthologs, args.config]):
        parser.print_help()
        sys.exit(1)

    species = get_valid_species()

    if args.all or args.preprocess:
        step_preprocess(species)
    if args.all or args.convert:
        step_convert(species)
    if args.all or args.node2vec:
        step_node2vec(species)
    if args.all or args.orthologs:
        step_orthologs(species)
    if args.all or args.config:
        step_config(species)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
