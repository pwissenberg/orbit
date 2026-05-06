#!/usr/bin/env python3
"""Two-stage Procrustes alignment of Node2Vec embeddings.

Stage 1: Align each non-ARATH seed to ARATH (the reference species).
Stage 2: Align each non-seed species to its nearest seed (already aligned).

Uses scipy.linalg.orthogonal_procrustes to find the optimal rotation matrix R
that minimises ||X @ R - Y||_F over the ortholog-pair correspondence.

Rotation is isometric: within-species pairwise distances are perfectly preserved.

Input:
    data/node2vec/{species}.h5             -- Node2Vec embeddings (128D)
    data/orthologs/seeds/                  -- seed-seed ortholog pairs (indices into N2V)
    data/orthologs/non_seeds/              -- non-seed-seed ortholog pairs (indices into N2V)
    results/seed_selection.json            -- seed list + nearest-seed group assignments

Output:
    results/aligned_embeddings_procrustes_n2v/{species}.h5

Usage:
    uv run python scripts/run_procrustes_alignment.py
    uv run python scripts/run_procrustes_alignment.py --no-eval
    uv run python scripts/run_procrustes_alignment.py --workers 4
    uv run python scripts/run_procrustes_alignment.py --svd   # use SVD embeddings (legacy)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from scipy.linalg import orthogonal_procrustes

ROOT = Path(__file__).resolve().parent.parent

SVD_DIR = ROOT / "data" / "procrustes_svd"
NODE2VEC_DIR = ROOT / "data" / "node2vec"
ORTHOLOGS_SEEDS_DIR = ROOT / "data" / "orthologs" / "seeds"
ORTHOLOGS_NONSEEDS_DIR = ROOT / "data" / "orthologs" / "non_seeds"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"
OUTPUT_DIR_N2V = ROOT / "results" / "aligned_embeddings_procrustes_n2v"
OUTPUT_DIR_SVD = ROOT / "results" / "aligned_embeddings_procrustes"
NETWORK_DIR = ROOT / "data" / "networks"
TRANSCRIPTS_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_embeddings(species: str, emb_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and protein list for a species from an H5 directory."""
    path = emb_dir / f"{species}.h5"
    with h5py.File(path, "r") as fh:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in fh["proteins"][:]]
        embeddings = fh["embeddings"][:]
    return embeddings, proteins


def _load_ortholog_pairs(path: Path) -> np.ndarray:
    """Load ortholog pair file -> integer index array, shape (n_pairs, 2)."""
    return np.loadtxt(path, usecols=(0, 1), dtype=np.int64)


def _save_aligned(species: str, embeddings: np.ndarray, proteins: list[str],
                  output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{species}.h5"
    with h5py.File(out_path, "w") as fh:
        fh.create_dataset(
            "proteins",
            data=np.array([p.encode() for p in proteins], dtype=object),
            dtype=h5py.special_dtype(vlen=bytes),
        )
        fh.create_dataset("embeddings", data=embeddings.astype(np.float32), dtype=np.float32)


# ---------------------------------------------------------------------------
# Procrustes helper
# ---------------------------------------------------------------------------

def _find_ortholog_path(sp_a: str, sp_b: str) -> Path | None:
    """Find the ortholog pair file for any two species (seed or non-seed)."""
    sorted_pair = "_".join(sorted([sp_a, sp_b]))

    # Check seeds directory
    p = ORTHOLOGS_SEEDS_DIR / f"{sorted_pair}.tsv"
    if p.exists():
        return p

    # Check non-seeds directories for both species
    for ns in [sp_a, sp_b]:
        p = ORTHOLOGS_NONSEEDS_DIR / ns / f"{sorted_pair}.tsv"
        if p.exists():
            return p

    return None


def _procrustes_align_n2v(
    sp_source: str,
    sp_ref: str,
    ref_emb: np.ndarray,
    ref_proteins: list[str],
) -> tuple[np.ndarray, list[str], int]:
    """Align sp_source Node2Vec embeddings to sp_ref aligned space.

    Pair file indices map directly to Node2Vec embedding rows, so no
    intermediate index translation is needed.

    Returns (aligned_embeddings, source_proteins, n_pairs_used).
    """
    pair_path = _find_ortholog_path(sp_source, sp_ref)
    if pair_path is None:
        raise FileNotFoundError(f"No ortholog pairs for {sp_source} vs {sp_ref}")

    # Determine column order from filename (alphabetically sorted species names)
    stem = pair_path.stem
    parts = stem.split("_")
    sp_first = parts[0]

    if sp_source == sp_first:
        col_source, col_ref = 0, 1
    else:
        col_source, col_ref = 1, 0

    pairs = _load_ortholog_pairs(pair_path)

    # Load source Node2Vec embeddings (full matrix)
    emb_source, proteins_source = _load_embeddings(sp_source, NODE2VEC_DIR)

    # Build ref protein -> index map (ref_emb may be already-aligned, not raw N2V)
    ref_idx_map = {g: i for i, g in enumerate(ref_proteins)}

    # Pair indices point directly into the N2V protein arrays.
    # For the source, these are indices into emb_source (same order).
    # For the ref, we need to map N2V gene names to the ref_emb index
    # (which may differ if ref_emb was aligned from SVD — but for N2V mode,
    # ref_emb also comes from N2V so the indices are the same).
    n2v_ref_proteins = None  # lazy load only if needed

    rows_source, rows_ref = [], []
    for idx_s, idx_r in zip(pairs[:, col_source], pairs[:, col_ref]):
        # Source index maps directly
        if idx_s >= len(proteins_source):
            continue

        # For ref, check if the ref_emb index matches directly
        if idx_r < len(ref_proteins):
            # Try direct index first (fast path for N2V mode)
            ir = ref_idx_map.get(ref_proteins[idx_r])
            if ir == idx_r:
                rows_source.append(int(idx_s))
                rows_ref.append(int(idx_r))
                continue

        # Fallback: load N2V protein list for ref and translate
        if n2v_ref_proteins is None:
            _, n2v_ref_proteins_list = _load_embeddings(sp_ref, NODE2VEC_DIR)
            n2v_ref_proteins = n2v_ref_proteins_list

        if idx_r >= len(n2v_ref_proteins):
            continue
        gene_ref = n2v_ref_proteins[idx_r]
        ir = ref_idx_map.get(gene_ref)
        if ir is not None:
            rows_source.append(int(idx_s))
            rows_ref.append(int(ir))

    if len(rows_source) < 2:
        raise ValueError(f"Too few ortholog pairs ({len(rows_source)}) for {sp_source}-{sp_ref}")

    X_src = emb_source[rows_source]   # n_pairs x 128
    X_ref = ref_emb[rows_ref]          # n_pairs x 128

    # orthogonal_procrustes(A, B) finds R that minimises ||A @ R - B||_F
    R, _ = orthogonal_procrustes(X_src, X_ref)

    aligned = emb_source @ R  # (n_genes, 128)

    # Validate orthogonality: ||R^T R - I||_F should be < 0.01
    ortho_err = np.linalg.norm(R.T @ R - np.eye(R.shape[0]))
    if ortho_err > 0.01:
        logger.warning(f"  {sp_source}: orthogonality error {ortho_err:.4f} > 0.01")

    return aligned, proteins_source, len(rows_source)


def _procrustes_align_svd(
    sp_source: str,
    sp_ref: str,
    ref_emb: np.ndarray,
    ref_proteins: list[str],
) -> tuple[np.ndarray, list[str], int]:
    """Legacy: align sp_source SVD embeddings to sp_ref aligned space.

    Requires N2V protein lists for index translation (pair indices -> gene names -> SVD rows).
    """
    pair_path = _find_ortholog_path(sp_source, sp_ref)
    if pair_path is None:
        raise FileNotFoundError(f"No ortholog pairs for {sp_source} vs {sp_ref}")

    stem = pair_path.stem
    parts = stem.split("_")
    sp_first = parts[0]

    if sp_source == sp_first:
        col_source, col_ref = 0, 1
    else:
        col_source, col_ref = 1, 0

    pairs = _load_ortholog_pairs(pair_path)

    # Load N2V protein lists for index translation
    _, n2v_proteins_source = _load_embeddings(sp_source, NODE2VEC_DIR)
    _, n2v_proteins_ref = _load_embeddings(sp_ref, NODE2VEC_DIR)

    emb_source, proteins_source = _load_embeddings(sp_source, SVD_DIR)

    svd_idx_source = {g: i for i, g in enumerate(proteins_source)}
    ref_idx_map = {g: i for i, g in enumerate(ref_proteins)}

    rows_source, rows_ref = [], []
    for idx_s, idx_r in zip(pairs[:, col_source], pairs[:, col_ref]):
        gs = n2v_proteins_source[idx_s]
        gr = n2v_proteins_ref[idx_r]
        is_ = svd_idx_source.get(gs)
        ir = ref_idx_map.get(gr)
        if is_ is not None and ir is not None:
            rows_source.append(is_)
            rows_ref.append(ir)

    if len(rows_source) < 2:
        raise ValueError(f"Too few ortholog pairs ({len(rows_source)}) for {sp_source}-{sp_ref}")

    X_src = emb_source[rows_source]
    X_ref = ref_emb[rows_ref]

    R, _ = orthogonal_procrustes(X_src, X_ref)
    aligned = emb_source @ R

    ortho_err = np.linalg.norm(R.T @ R - np.eye(R.shape[0]))
    if ortho_err > 0.01:
        logger.warning(f"  {sp_source}: orthogonality error {ortho_err:.4f} > 0.01")

    return aligned, proteins_source, len(rows_source)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Two-stage Procrustes alignment")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--workers", type=int, default=None, help="Workers for evaluation")
    parser.add_argument(
        "--svd", action="store_true",
        help="Use SVD embeddings (64D) instead of Node2Vec (128D)",
    )
    args = parser.parse_args()

    use_svd = args.svd
    emb_dir = SVD_DIR if use_svd else NODE2VEC_DIR
    output_dir = OUTPUT_DIR_SVD if use_svd else OUTPUT_DIR_N2V
    align_fn = _procrustes_align_svd if use_svd else _procrustes_align_n2v
    mode_label = "SVD (64D)" if use_svd else "Node2Vec (128D)"
    eval_prefix = "evaluation_procrustes" if use_svd else "evaluation_procrustes_n2v"
    within_prefix = "evaluation_within_species_procrustes" if use_svd else "evaluation_within_species_procrustes_n2v"

    logger.info(f"Mode: {mode_label}")
    logger.info(f"Input: {emb_dir}")
    logger.info(f"Output: {output_dir}")

    with open(SEED_RESULTS) as fh:
        seed_info = json.load(fh)
    seeds = seed_info["seeds"]   # ['ARATH', 'ORYSA', 'PICAB', 'SELMO', 'MARPO']
    groups = seed_info["groups"]  # non-seed -> nearest-seed mapping

    REFERENCE = "ARATH"
    non_ref_seeds = [s for s in seeds if s != REFERENCE]

    # -----------------------------------------------------------------------
    # Stage 1: Align seeds to reference (ARATH)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Stage 1: aligning seeds to reference ({REFERENCE})")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    # Reference just copies its embeddings
    ref_emb, ref_proteins = _load_embeddings(REFERENCE, emb_dir)
    _save_aligned(REFERENCE, ref_emb, ref_proteins, output_dir)
    logger.info(f"  {REFERENCE} -- reference (copied), {len(ref_proteins)} genes, dim={ref_emb.shape[1]}")

    # Cache of aligned seed embeddings so Stage 2 can look them up
    aligned_seeds: dict[str, tuple[np.ndarray, list[str]]] = {
        REFERENCE: (ref_emb, ref_proteins),
    }

    for seed in non_ref_seeds:
        try:
            aligned_emb, src_proteins, n_pairs = align_fn(
                seed, REFERENCE, ref_emb, ref_proteins
            )
            _save_aligned(seed, aligned_emb, src_proteins, output_dir)
            aligned_seeds[seed] = (aligned_emb, src_proteins)
            logger.info(f"  {seed} -> {REFERENCE}: {n_pairs} pairs, {len(src_proteins)} genes")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  {seed}: FAILED -- {exc}")
            return 1

    seed_time = time.perf_counter() - t0
    logger.info(f"Stage 1 complete in {seed_time:.1f}s")

    # -----------------------------------------------------------------------
    # Stage 2: Align non-seeds to nearest seed
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: aligning non-seeds to nearest seed")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    failed: list[str] = []
    success = 0

    non_seeds = sorted(groups.keys())
    for ns in non_seeds:
        nearest_seed = groups[ns]
        seed_emb, seed_prots = aligned_seeds[nearest_seed]

        try:
            aligned_emb, ns_proteins, n_pairs = align_fn(
                ns, nearest_seed, seed_emb, seed_prots
            )
            _save_aligned(ns, aligned_emb, ns_proteins, output_dir)
            success += 1
            if success % 20 == 0:
                logger.info(f"  [{success}/{len(non_seeds)}] non-seeds aligned")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  {ns} -> {nearest_seed}: FAILED -- {exc}")
            failed.append(ns)

    nonseed_time = time.perf_counter() - t0
    logger.info(f"Stage 2 complete in {nonseed_time:.1f}s -- {success} ok, {len(failed)} failed")

    if failed:
        logger.warning(f"Failed species: {', '.join(failed)}")
        return 1

    total_aligned = len(seeds) + len(non_seeds)
    logger.info(f"Total aligned: {total_aligned} species -> {output_dir}")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    if args.no_eval:
        logger.info("Skipping evaluation (--no-eval)")
        return 0

    from orbit.evaluate import evaluate_all_pairs, evaluate_all_within_species

    # Cross-species evaluation
    logger.info("=" * 60)
    logger.info("Running cross-species evaluation")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    results = evaluate_all_pairs(
        aligned_dir=output_dir,
        network_dir=NETWORK_DIR,
        transcripts_dir=TRANSCRIPTS_DIR,
        seed_results_path=SEED_RESULTS,
        k=50,
        top_m=10,
        eval_mode="all",
        n_workers=args.workers,
    )

    cross_path = ROOT / "results" / f"{eval_prefix}.json"
    with open(cross_path, "w") as fh:
        json.dump(results, fh, indent=2)

    valid = [r for r in results if "error" not in r]
    avg_rho = np.mean([r["spearman_rho"] for r in valid]) if valid else 0.0
    avg_hits = np.mean([r["hits_at_k"] for r in valid]) if valid else 0.0

    eval_time = time.perf_counter() - t0
    logger.info(f"Cross-species eval complete in {eval_time:.1f}s")
    logger.info(f"  {len(valid)} pairs | avg Spearman={avg_rho:.4f} | avg Hits@50={avg_hits:.4f}")
    logger.info(f"  Saved -> {cross_path}")

    # Within-species evaluation
    logger.info("=" * 60)
    logger.info("Running within-species evaluation")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    within_results = evaluate_all_within_species(
        aligned_dir=output_dir,
        network_dir=NETWORK_DIR,
        k=50,
        n_workers=args.workers,
    )

    within_path = ROOT / "results" / f"{within_prefix}.json"
    with open(within_path, "w") as fh:
        json.dump(within_results, fh, indent=2)

    valid_within = [r for r in within_results if "error" not in r]
    avg_prec = np.mean([r["precision_at_k"] for r in valid_within]) if valid_within else 0.0
    avg_rec = np.mean([r["recall_at_k"] for r in valid_within]) if valid_within else 0.0

    within_time = time.perf_counter() - t0
    logger.info(f"Within-species eval complete in {within_time:.1f}s")
    logger.info(f"  {len(valid_within)} species | avg P@50={avg_prec:.4f} | avg R@50={avg_rec:.4f}")
    logger.info(f"  Saved -> {within_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
