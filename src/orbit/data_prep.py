"""Data preparation for SPACE alignment pipeline.

Preprocessing raw TEA-GCN coexpression networks, generating Node2Vec
embeddings, building ortholog pairs, and writing SPACE config files.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# 1a. Network preprocessing
# ---------------------------------------------------------------------------


def preprocess_network(raw_path: Path, output_path: Path) -> int:
    """Convert raw TEA-GCN network to clean 3-col TSV.

    Input:  {CODE}_{Species}_Top50EdgesPerGene_ProteinID.tsv
            Header: Source_Protein_ID  Target_Protein_ID  zScore(Co-exp_Str_MR)

    Output: geneA\\tgeneB\\tweight  (no header, sorted edge key, max weight)

    Returns number of edges written.
    """
    edges: dict[tuple[str, str], float] = {}
    with open(raw_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            a, b = row[0], row[1]
            w = float(row[2])
            key = (min(a, b), max(a, b))
            if key not in edges or w > edges[key]:
                edges[key] = w

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for (a, b), w in sorted(edges.items()):
            fh.write(f"{a}\t{b}\t{w}\n")

    logger.info(f"  {raw_path.name} → {len(edges)} edges")
    return len(edges)


# ---------------------------------------------------------------------------
# 1b. Convert to SPACE gzipped format
# ---------------------------------------------------------------------------


def convert_to_space_format(network_path: Path, output_path: Path) -> None:
    """Convert clean TSV to SPACE node2vec input format.

    Input:  geneA\\tgeneB\\tweight  (TSV, float weights)
    Output: geneA geneB score  (space-separated, int 0-1000, gzipped, with header)
    """
    rows: list[tuple[str, str, float]] = []
    max_w = 0.0
    with open(network_path) as fh:
        for line in fh:
            a, b, w_str = line.strip().split("\t")
            w = float(w_str)
            rows.append((a, b, w))
            if w > max_w:
                max_w = w

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as fh:
        fh.write("protein1 protein2 score\n")
        for a, b, w in rows:
            score = int(w / max_w * 1000) if max_w > 0 else 0
            fh.write(f"{a} {b} {score}\n")


# ---------------------------------------------------------------------------
# 1c. Generate Node2Vec embeddings
# ---------------------------------------------------------------------------


def generate_embeddings(
    network_gz_path: Path,
    output_h5_path: Path,
    dimensions: int = 128,
    p: float = 1.0,
    q: float = 0.7,
    num_walks: int = 40,
    walk_length: int = 50,
    epochs: int = 1,
    window_size: int = 5,
    workers: int = -1,
    random_state: int = 1234,
) -> None:
    """Run SPACE's node2vec pipeline: PecanPy walks + gensim Word2Vec.

    Output H5 format (SPACE-compatible):
      metadata/n_proteins, metadata/embedding_dim, metadata/precision
      proteins: string array of gene IDs
      embeddings: (N, dim) float32 matrix
    """
    import orbit._compat  # noqa: F401 — patch numpy before SPACE
    from space.tools.data import GzipData, H5pyData
    from space.models.node2vec import PecanpyEmbedder

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_h5_path.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / (output_h5_path.stem + "_edges.tsv")

    # Convert string IDs to indices (SPACE convention)
    nodes = GzipData.string2idx(str(network_gz_path), str(temp_path))

    logger.info(f"  {output_h5_path.stem}: {len(nodes)} nodes, running node2vec...")

    embedder = PecanpyEmbedder(
        str(temp_path), p=p, q=q, workers=workers,
        weighted=True, directed=False, extend=False,
        gamma=0, random_state=random_state, delimiter="\t",
    )

    walks = embedder.generate_walks(num_walks=num_walks, walk_length=walk_length)
    model = embedder.learn_embeddings(
        walks, epochs=epochs, dimensions=dimensions,
        window_size=window_size, workers=workers,
        negative=5, hs=0, sg=1, random_state=random_state,
    )

    emb = model.wv.vectors
    index = model.wv.index_to_key
    proteins = list(nodes.keys())
    map_proteins = [proteins[int(i)] for i in index]

    H5pyData.write(map_proteins, emb, str(output_h5_path), 32)

    # Clean up temp file
    temp_path.unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    logger.info(f"  {output_h5_path.stem}: saved {len(map_proteins)}×{dimensions} embeddings")


# ---------------------------------------------------------------------------
# 1d. Build ortholog pair files
# ---------------------------------------------------------------------------


def build_ortholog_pairs(
    species_a: str,
    species_b: str,
    h5_dir: Path,
    transcripts_dir: Path,
    output_path: Path,
) -> int:
    """Build index-based ortholog pair TSV for one species pair.

    1. Load H5 protein arrays → build protein_id → index maps
    2. Load OrthoFinder transcripts → build protein_id → orthogroup maps
    3. For each shared orthogroup: generate all cross-species pairs
    4. Weight: 1/(n_A * n_B) per OG (SPACE default)
    5. Write: idx_A\\tidx_B\\tweight

    Returns number of pairs written.
    """
    # Load protein lists from H5 files
    proteins_a = _read_h5_proteins(h5_dir / f"{species_a}.h5")
    proteins_b = _read_h5_proteins(h5_dir / f"{species_b}.h5")

    idx_a = {p: i for i, p in enumerate(proteins_a)}
    idx_b = {p: i for i, p in enumerate(proteins_b)}

    # Load OrthoFinder mappings: protein_id → orthogroup
    og_a = _load_protein_to_og(transcripts_dir, species_a, set(idx_a.keys()))
    og_b = _load_protein_to_og(transcripts_dir, species_b, set(idx_b.keys()))

    # Group by orthogroup
    og_genes_a: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_a.items():
        og_genes_a[og].append(pid)

    og_genes_b: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_b.items():
        og_genes_b[og].append(pid)

    # Generate pairs for shared OGs
    shared_ogs = set(og_genes_a.keys()) & set(og_genes_b.keys())
    pairs: list[tuple[int, int, float]] = []
    for og in sorted(shared_ogs):
        genes_a = og_genes_a[og]
        genes_b = og_genes_b[og]
        w = 1.0 / (len(genes_a) * len(genes_b))
        for ga in genes_a:
            for gb in genes_b:
                pairs.append((idx_a[ga], idx_b[gb], w))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for ia, ib, w in pairs:
            fh.write(f"{ia}\t{ib}\t{w}\n")

    logger.debug(
        f"  {species_a}_{species_b}: {len(pairs)} pairs from {len(shared_ogs)} shared OGs"
    )
    return len(pairs)


def _read_h5_proteins(h5_path: Path) -> list[str]:
    """Read protein names from an H5 embedding file."""
    with h5py.File(h5_path, "r") as f:
        proteins = f["proteins"][:]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in proteins]


def _load_protein_to_og(
    transcripts_dir: Path, species: str, valid_proteins: set[str]
) -> dict[str, str]:
    """Load protein_id → orthogroup mapping from OrthoFinder transcripts.

    Tries exact Protein_ID match first. If coverage is below 50%,
    falls back to stripping version suffixes from both sides.
    """
    tsv_path = transcripts_dir / f"{species}_transcripts_to_OG.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["Orthogroup"])

    # Try exact Protein_ID match
    exact = {
        row.Protein_ID: row.Orthogroup
        for row in df.itertuples()
        if row.Protein_ID in valid_proteins
    }
    coverage = len(exact) / len(valid_proteins) if valid_proteins else 0

    if coverage >= 0.5:
        logger.debug(f"  {species}: exact Protein_ID match coverage {coverage:.1%}")
        return exact

    # Fallback: strip version suffixes (everything after first '.')
    logger.info(
        f"  {species}: exact match coverage only {coverage:.1%}, "
        "trying stripped IDs"
    )
    strip = lambda s: s.split(".")[0]
    of_stripped: dict[str, str] = {}
    for row in df.itertuples():
        of_stripped[strip(row.Protein_ID)] = row.Orthogroup

    result: dict[str, str] = {}
    for pid in valid_proteins:
        key = strip(pid)
        if key in of_stripped:
            result[pid] = of_stripped[key]

    stripped_coverage = len(result) / len(valid_proteins) if valid_proteins else 0
    logger.info(f"  {species}: stripped match coverage {stripped_coverage:.1%}")
    return result


# ---------------------------------------------------------------------------
# 1e-bis. Build Jaccard-weighted ortholog pairs (improvement over vanilla)
# ---------------------------------------------------------------------------


def build_jaccard_ortholog_pairs(
    species_a: str,
    species_b: str,
    h5_dir: Path,
    transcripts_dir: Path,
    network_dir: Path,
    output_path: Path,
) -> int:
    """Build Jaccard-weighted ortholog pairs from coexpression neighborhoods.

    Instead of uniform 1/(n_A * n_B) weights, this computes the Jaccard index
    of each ortholog pair's coexpression neighborhoods (converted to OGs).
    Pairs with Jaccard > 0 are written.

    Returns number of pairs written.
    """
    from orbit.evaluate import build_jaccard_ground_truth

    pairs = build_jaccard_ground_truth(
        species_a, species_b, network_dir, transcripts_dir, h5_dir
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for ia, ib, w in pairs:
            fh.write(f"{ia}\t{ib}\t{w}\n")

    logger.debug(
        f"  {species_a}_{species_b}: {len(pairs)} Jaccard-weighted pairs"
    )
    return len(pairs)


def build_hybrid_ortholog_pairs(
    species_a: str,
    species_b: str,
    h5_dir: Path,
    transcripts_dir: Path,
    network_dir: Path,
    output_path: Path,
    alpha: float = 5.0,
) -> int:
    """Build hybrid-weighted ortholog pairs: vanilla base + Jaccard boost.

    Weight = (1/(n_A * n_B)) * (1 + alpha * jaccard)

    This keeps ALL ortholog pairs (preserving alignment signal for retrieval)
    but upweights pairs with high coexpression neighborhood similarity.

    Returns number of pairs written.
    """
    from orbit.evaluate import build_jaccard_ground_truth

    # Get Jaccard scores as a lookup
    jaccard_pairs = build_jaccard_ground_truth(
        species_a, species_b, network_dir, transcripts_dir, h5_dir
    )
    jaccard_map = {(ia, ib): w for ia, ib, w in jaccard_pairs}

    # Build vanilla pairs (all orthologs with 1/(nA*nB) weights)
    proteins_a = _read_h5_proteins(h5_dir / f"{species_a}.h5")
    proteins_b = _read_h5_proteins(h5_dir / f"{species_b}.h5")

    idx_a = {p: i for i, p in enumerate(proteins_a)}
    idx_b = {p: i for i, p in enumerate(proteins_b)}

    og_a = _load_protein_to_og(transcripts_dir, species_a, set(idx_a.keys()))
    og_b = _load_protein_to_og(transcripts_dir, species_b, set(idx_b.keys()))

    og_genes_a: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_a.items():
        og_genes_a[og].append(pid)

    og_genes_b: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_b.items():
        og_genes_b[og].append(pid)

    shared_ogs = set(og_genes_a.keys()) & set(og_genes_b.keys())
    pairs: list[tuple[int, int, float]] = []
    n_boosted = 0
    for og in sorted(shared_ogs):
        genes_a = og_genes_a[og]
        genes_b = og_genes_b[og]
        base_w = 1.0 / (len(genes_a) * len(genes_b))
        for ga in genes_a:
            for gb in genes_b:
                ia, ib = idx_a[ga], idx_b[gb]
                jacc = jaccard_map.get((ia, ib), 0.0)
                w = base_w * (1.0 + alpha * jacc)
                if jacc > 0:
                    n_boosted += 1
                pairs.append((ia, ib, w))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for ia, ib, w in pairs:
            fh.write(f"{ia}\t{ib}\t{w}\n")

    logger.debug(
        f"  {species_a}_{species_b}: {len(pairs)} hybrid pairs "
        f"({n_boosted} Jaccard-boosted, alpha={alpha})"
    )
    return len(pairs)


# ---------------------------------------------------------------------------
# 1e. Write SPACE config files
# ---------------------------------------------------------------------------


def write_space_config(
    seeds: list[str],
    groups: dict[str, str],
    output_dir: Path,
) -> None:
    """Generate SPACE input files.

    - seeds.txt: one species code per line
    - seed_groups.json: {"plants": [seeds]}
    - tax_group.tsv: species_code\\tgroup  (for all species)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # seeds.txt
    seeds_path = output_dir / "seeds.txt"
    seeds_path.write_text("\n".join(seeds) + "\n")

    # seed_groups.json — single group "plants" with all seeds
    groups_path = output_dir / "seed_groups.json"
    groups_path.write_text(json.dumps({"plants": seeds}, indent=2) + "\n")

    # tax_group.tsv — all species with their group name
    tax_path = output_dir / "tax_group.tsv"
    with open(tax_path, "w") as fh:
        fh.write("taxid\tgroup\n")
        for sp in seeds:
            fh.write(f"{sp}\tplants\n")
        for sp in sorted(groups.keys()):
            fh.write(f"{sp}\tplants\n")

    logger.info(
        f"SPACE config: {len(seeds)} seeds, "
        f"{len(groups)} non-seeds → {output_dir}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_species_with_networks(raw_dir: Path) -> list[str]:
    """Extract species codes from raw network filenames.

    Real filenames: {CODE}_{GenusSpecies}_Top50EdgesPerGene_ProteinID.tsv
    The code is a 5-letter uppercase string (e.g., ABROB).
    """
    suffix = "_Top50EdgesPerGene_ProteinID.tsv"
    codes = []
    for f in sorted(raw_dir.glob(f"*{suffix}")):
        # Strip everything from first underscore that starts the species name
        # Code is chars before the second token (genus name)
        name = f.name.removesuffix(suffix)
        # Code is the first underscore-delimited token
        code = name.split("_")[0]
        codes.append(code)
    return codes


def list_species_with_orthogroups(transcripts_dir: Path) -> list[str]:
    """Extract species codes from OrthoFinder transcript filenames."""
    codes = []
    for f in sorted(transcripts_dir.glob("*_transcripts_to_OG.tsv")):
        code = f.name.replace("_transcripts_to_OG.tsv", "")
        codes.append(code)
    return codes
