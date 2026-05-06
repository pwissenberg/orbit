#!/usr/bin/env python3
"""Improved two-stage Procrustes alignment of Node2Vec embeddings.

Extends the baseline Procrustes pipeline with three improvements:
  1. Weighted Procrustes — Jaccard/hybrid weights in the rotation fit
  2. Iterative refinement — discover new anchor pairs from aligned space, re-fit R
  3. CSLS scoring — anti-hubness scoring for better anchor discovery

Stage 1: Align each non-ARATH seed to ARATH (the reference species).
Stage 2: Align each non-seed species to its nearest seed (already aligned).

Usage:
    # Build weighted pair files
    uv run python scripts/run_improved_procrustes.py --stage pairs --weighting jaccard
    uv run python scripts/run_improved_procrustes.py --stage pairs --weighting hybrid

    # Run alignment variants
    uv run python scripts/run_improved_procrustes.py --stage align --weighting jaccard
    uv run python scripts/run_improved_procrustes.py --stage align --weighting vanilla --iterative --csls
    uv run python scripts/run_improved_procrustes.py --stage align --weighting hybrid --iterative --csls

    # Evaluate
    uv run python scripts/run_improved_procrustes.py --stage eval --weighting jaccard

    # Full pipeline
    uv run python scripts/run_improved_procrustes.py --stage all --weighting hybrid --iterative --csls
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

NODE2VEC_DIR = ROOT / "data" / "node2vec"
ORTHOLOGS_VANILLA_DIR = ROOT / "data" / "orthologs"
ORTHOLOGS_JACCARD_DIR = ROOT / "data" / "orthologs_jaccard"
ORTHOLOGS_HYBRID_DIR = ROOT / "data" / "orthologs_hybrid"
NETWORK_DIR = ROOT / "data" / "networks"
TRANSCRIPTS_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"

# Module-level globals for ProcessPoolExecutor workers (avoids pickling large data)
_PRELOADED_EMBEDDINGS: dict[str, tuple[np.ndarray, list[str]]] | None = None


# ---------------------------------------------------------------------------
# I/O helpers (from run_procrustes_alignment.py)
# ---------------------------------------------------------------------------

def _load_embeddings(species: str, emb_dir: Path) -> tuple[np.ndarray, list[str]]:
    path = emb_dir / f"{species}.h5"
    with h5py.File(path, "r") as fh:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in fh["proteins"][:]]
        embeddings = fh["embeddings"][:]
    return embeddings, proteins


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


def _find_ortholog_path(sp_a: str, sp_b: str, orthologs_dir: Path) -> Path | None:
    sorted_pair = "_".join(sorted([sp_a, sp_b]))
    seeds_dir = orthologs_dir / "seeds"
    p = seeds_dir / f"{sorted_pair}.tsv"
    if p.exists():
        return p
    nonseeds_dir = orthologs_dir / "non_seeds"
    for ns in [sp_a, sp_b]:
        p = nonseeds_dir / ns / f"{sorted_pair}.tsv"
        if p.exists():
            return p
    return None


def _load_ortholog_pairs_weighted(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ortholog pair file -> (indices array shape (n,2), weights array shape (n,)).

    If the file has 3 columns, the third is the weight.
    If only 2 columns, weights default to 1.0.
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    indices = data[:, :2].astype(np.int64)
    if data.shape[1] >= 3:
        weights = data[:, 2].astype(np.float64)
    else:
        weights = np.ones(len(indices), dtype=np.float64)
    return indices, weights


# ---------------------------------------------------------------------------
# Weighted Procrustes solver
# ---------------------------------------------------------------------------

def _weighted_procrustes(X_src: np.ndarray, X_ref: np.ndarray,
                         weights: np.ndarray | None = None,
                         translate: bool = False,
                         scale: bool = False,
                         ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """Find orthogonal R (+ optional translation/scaling) minimising
    weighted ||s * (X_src - mu_src) @ R + mu_ref - X_ref||_F.

    Returns (R, mu_src, mu_ref, s):
      - R: (d, d) orthogonal rotation matrix
      - mu_src: (d,) source centroid (None if translate=False)
      - mu_ref: (d,) reference centroid (None if translate=False)
      - s: scaling factor (1.0 if scale=False)
    """
    # Compute weighted means for centering
    if translate:
        if weights is not None and not np.allclose(weights, weights[0]):
            w_norm = weights / weights.sum()
            mu_src = (X_src * w_norm[:, None]).sum(axis=0)
            mu_ref = (X_ref * w_norm[:, None]).sum(axis=0)
        else:
            mu_src = X_src.mean(axis=0)
            mu_ref = X_ref.mean(axis=0)
        X_src_c = X_src - mu_src
        X_ref_c = X_ref - mu_ref
    else:
        mu_src = None
        mu_ref = None
        X_src_c = X_src
        X_ref_c = X_ref

    # Weighted cross-covariance
    if weights is None or np.allclose(weights, weights[0]):
        M = X_src_c.T @ X_ref_c
    else:
        w_sqrt = np.sqrt(weights / weights.sum())
        X_src_w = X_src_c * w_sqrt[:, None]
        X_ref_w = X_ref_c * w_sqrt[:, None]
        M = X_src_w.T @ X_ref_w

    U, _, Vt = np.linalg.svd(M)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(U @ Vt)
    D = np.diag([*np.ones(len(U) - 1), d])
    R = (U @ D @ Vt).astype(np.float32)

    # Optimal scaling: s = trace(X_ref_c^T @ X_src_c @ R) / ||X_src_c||^2
    if scale:
        if weights is not None and not np.allclose(weights, weights[0]):
            w_norm = weights / weights.sum()
            numerator = np.sum(w_norm[:, None] * X_ref_c * (X_src_c @ R))
            denominator = np.sum(w_norm[:, None] * X_src_c * X_src_c)
        else:
            numerator = np.trace(X_ref_c.T @ (X_src_c @ R))
            denominator = np.trace(X_src_c.T @ X_src_c)
        s = float(numerator / (denominator + 1e-12))
    else:
        s = 1.0

    return R, mu_src, mu_ref, s


# ---------------------------------------------------------------------------
# CSLS scoring
# ---------------------------------------------------------------------------

def _build_faiss_index(X: np.ndarray, use_gpu: bool = True, gpu_id: int | None = None):
    """Build a faiss IndexFlatIP on L2-normalized embeddings.

    If gpu_id is specified, places the index on that specific GPU (allows
    round-robin assignment across workers). Otherwise uses all GPUs.

    Returns (index, gpu_used).
    """
    import faiss

    X = np.ascontiguousarray(X, dtype=np.float32)
    index = faiss.IndexFlatIP(X.shape[1])

    gpu_used = False
    if use_gpu:
        try:
            if gpu_id is not None:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            else:
                index = faiss.index_cpu_to_all_gpus(index)
            gpu_used = True
        except Exception:
            pass  # CPU fallback

    index.add(X)
    return index, gpu_used


def _csls_knn(X_src: np.ndarray, X_ref: np.ndarray, k: int = 10,
              use_gpu: bool = True,
              gpu_id: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CSLS-corrected nearest neighbors.

    Returns:
        nn_src2ref: (n_src,) indices of NN in ref for each src
        nn_ref2src: (n_ref,) indices of NN in src for each ref
        csls_scores_src2ref: (n_src,) CSLS scores for the best match
        csls_scores_ref2src: (n_ref,) CSLS scores for the best match
    """
    import faiss

    n_src, dim = X_src.shape
    n_ref = X_ref.shape[0]

    # L2-normalize
    eps = 1e-12
    X_src_n = X_src / (np.linalg.norm(X_src, axis=1, keepdims=True) + eps)
    X_ref_n = X_ref / (np.linalg.norm(X_ref, axis=1, keepdims=True) + eps)
    X_src_n = np.ascontiguousarray(X_src_n, dtype=np.float32)
    X_ref_n = np.ascontiguousarray(X_ref_n, dtype=np.float32)

    # Build indices
    index_ref, _ = _build_faiss_index(X_ref_n, use_gpu=use_gpu, gpu_id=gpu_id)
    index_src, _ = _build_faiss_index(X_src_n, use_gpu=use_gpu, gpu_id=gpu_id)

    # k-NN searches for mean similarity penalties
    # Use max(k, 1) so top-1 NN is included in the result (avoids redundant search)
    k_actual = max(min(k, min(n_src, n_ref)), 1)
    sim_src2ref, idx_src2ref = index_ref.search(X_src_n, k_actual)  # (n_src, k)
    sim_ref2src, idx_ref2src = index_src.search(X_ref_n, k_actual)  # (n_ref, k)

    r_src = sim_src2ref.mean(axis=1)  # (n_src,) mean k-NN similarity penalty
    r_ref = sim_ref2src.mean(axis=1)  # (n_ref,) mean k-NN similarity penalty

    # Extract top-1 NN from the k-NN results (first column)
    idx_top1_s2r = idx_src2ref[:, 0]    # (n_src,)
    sim_top1_s2r = sim_src2ref[:, 0]    # (n_src,)
    idx_top1_r2s = idx_ref2src[:, 0]    # (n_ref,)
    sim_top1_r2s = sim_ref2src[:, 0]    # (n_ref,)

    # CSLS scores: 2*cos(x,y) - r_x - r_y
    csls_s2r = 2 * sim_top1_s2r - r_src - r_ref[idx_top1_s2r]
    csls_r2s = 2 * sim_top1_r2s - r_ref - r_src[idx_top1_r2s]

    return idx_top1_s2r, idx_top1_r2s, csls_s2r, csls_r2s


def _cosine_knn(X_src: np.ndarray, X_ref: np.ndarray,
                use_gpu: bool = True,
                gpu_id: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute cosine nearest neighbors (no CSLS correction).

    Returns same format as _csls_knn.
    """
    import faiss

    eps = 1e-12
    X_src_n = X_src / (np.linalg.norm(X_src, axis=1, keepdims=True) + eps)
    X_ref_n = X_ref / (np.linalg.norm(X_ref, axis=1, keepdims=True) + eps)
    X_src_n = np.ascontiguousarray(X_src_n, dtype=np.float32)
    X_ref_n = np.ascontiguousarray(X_ref_n, dtype=np.float32)

    index_ref, _ = _build_faiss_index(X_ref_n, use_gpu=use_gpu, gpu_id=gpu_id)
    index_src, _ = _build_faiss_index(X_src_n, use_gpu=use_gpu, gpu_id=gpu_id)

    sim_s2r, idx_s2r = index_ref.search(X_src_n, 1)
    sim_r2s, idx_r2s = index_src.search(X_ref_n, 1)

    return idx_s2r.ravel(), idx_r2s.ravel(), sim_s2r.ravel(), sim_r2s.ravel()


# ---------------------------------------------------------------------------
# Iterative Procrustes refinement
# ---------------------------------------------------------------------------

def _iterative_procrustes(
    emb_source: np.ndarray,
    emb_ref: np.ndarray,
    anchor_src_idx: np.ndarray,
    anchor_ref_idx: np.ndarray,
    anchor_weights: np.ndarray | None,
    n_iters: int = 5,
    use_csls: bool = False,
    csls_k: int = 10,
    csls_threshold: float = 0.5,
    use_gpu: bool = True,
    gpu_id: int | None = None,
    translate: bool = False,
    scale: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """Iterative Procrustes refinement with optional translation/scaling.

    1. Fit initial R (+ mu_src, mu_ref, s) from ortholog anchors.
    2. For each iteration:
       a. Apply transform to source embeddings
       b. Find mutual nearest neighbors (MNN) using CSLS or cosine
       c. Combine original anchors + new MNN pairs
       d. Re-fit R (+ translation/scaling) via weighted Procrustes
       e. Check convergence

    Returns (R, mu_src, mu_ref, s).
    """
    X_src_anchors = emb_source[anchor_src_idx]
    X_ref_anchors = emb_ref[anchor_ref_idx]

    # Initial fit
    R, mu_src, mu_ref, s = _weighted_procrustes(
        X_src_anchors, X_ref_anchors, anchor_weights,
        translate=translate, scale=scale,
    )

    anchor_set = set(zip(anchor_src_idx.tolist(), anchor_ref_idx.tolist()))
    mean_anchor_weight = float(anchor_weights.mean()) if anchor_weights is not None else 1.0

    for it in range(n_iters):
        R_old = R.copy()

        # Apply current transform to full source embeddings
        X_src_aligned = emb_source.copy()
        if mu_src is not None:
            X_src_aligned = X_src_aligned - mu_src
        X_src_aligned = X_src_aligned @ R * s
        if mu_ref is not None:
            X_src_aligned = X_src_aligned + mu_ref

        # Find MNN
        if use_csls:
            nn_s2r, nn_r2s, scores_s2r, scores_r2s = _csls_knn(
                X_src_aligned, emb_ref, k=csls_k, use_gpu=use_gpu, gpu_id=gpu_id
            )
        else:
            nn_s2r, nn_r2s, scores_s2r, scores_r2s = _cosine_knn(
                X_src_aligned, emb_ref, use_gpu=use_gpu, gpu_id=gpu_id
            )

        # Mutual nearest neighbors: both directions agree
        n_src = len(emb_source)
        is_mutual = nn_r2s[nn_s2r] == np.arange(n_src)
        above_threshold = scores_s2r >= csls_threshold

        mnn_mask = is_mutual & above_threshold
        mnn_src_idx = np.where(mnn_mask)[0]
        mnn_ref_idx = nn_s2r[mnn_mask]

        # Filter out pairs already in original anchor set
        new_pairs = [
            (s_idx, r_idx) for s_idx, r_idx in zip(mnn_src_idx, mnn_ref_idx)
            if (int(s_idx), int(r_idx)) not in anchor_set
        ]

        if not new_pairs:
            logger.debug(f"    iter {it+1}: no new MNN pairs, stopping")
            break

        # Combine anchors + new MNN pairs
        new_src = np.array([p[0] for p in new_pairs], dtype=np.int64)
        new_ref = np.array([p[1] for p in new_pairs], dtype=np.int64)

        all_src = np.concatenate([anchor_src_idx, new_src])
        all_ref = np.concatenate([anchor_ref_idx, new_ref])
        if anchor_weights is not None:
            new_weights = np.full(len(new_pairs), mean_anchor_weight, dtype=np.float64)
            all_weights = np.concatenate([anchor_weights, new_weights])
        else:
            all_weights = None

        # Re-fit
        R, mu_src, mu_ref, s = _weighted_procrustes(
            emb_source[all_src], emb_ref[all_ref], all_weights,
            translate=translate, scale=scale,
        )

        # Convergence check
        delta = np.linalg.norm(R - R_old, ord="fro")
        logger.debug(
            f"    iter {it+1}: {len(new_pairs)} new MNN pairs "
            f"(total {len(all_src)}), ||dR||_F={delta:.6f}"
        )
        if delta < 1e-5:
            logger.debug(f"    converged at iter {it+1}")
            break

    return R, mu_src, mu_ref, s


# ---------------------------------------------------------------------------
# Core alignment function
# ---------------------------------------------------------------------------

def _align_one_species(
    sp_source: str,
    sp_ref: str,
    ref_emb: np.ndarray,
    ref_proteins: list[str],
    orthologs_dir: Path,
    *,
    weighting: str,
    iterative: bool,
    use_csls: bool,
    n_iters: int,
    csls_k: int,
    csls_threshold: float,
    use_gpu: bool,
    gpu_id: int | None = None,
    emb_source: np.ndarray | None = None,
    proteins_source: list[str] | None = None,
    translate: bool = False,
    scale: bool = False,
) -> tuple[np.ndarray, list[str], int]:
    """Align sp_source to sp_ref using improved Procrustes.

    Returns (aligned_embeddings, source_proteins, n_pairs_used).
    """
    pair_path = _find_ortholog_path(sp_source, sp_ref, orthologs_dir)
    if pair_path is None:
        raise FileNotFoundError(f"No ortholog pairs for {sp_source} vs {sp_ref}")

    # Determine column order from filename
    stem = pair_path.stem
    sp_first = stem.split("_")[0]
    if sp_source == sp_first:
        col_source, col_ref = 0, 1
    else:
        col_source, col_ref = 1, 0

    pairs, weights = _load_ortholog_pairs_weighted(pair_path)

    # Load source embeddings if not provided (preloaded)
    if emb_source is None or proteins_source is None:
        emb_source, proteins_source = _load_embeddings(sp_source, NODE2VEC_DIR)

    ref_idx_map = {g: i for i, g in enumerate(ref_proteins)}

    # Translate pair indices to embedding row indices
    n2v_ref_proteins = None
    rows_source, rows_ref, pair_weights = [], [], []

    for k_idx in range(len(pairs)):
        idx_s = pairs[k_idx, col_source]
        idx_r = pairs[k_idx, col_ref]
        w = weights[k_idx]

        if idx_s >= len(proteins_source):
            continue

        # Try direct index (fast path)
        if idx_r < len(ref_proteins):
            ir = ref_idx_map.get(ref_proteins[idx_r])
            if ir == idx_r:
                rows_source.append(int(idx_s))
                rows_ref.append(int(idx_r))
                pair_weights.append(w)
                continue

        # Fallback: load N2V protein list for ref and translate
        if n2v_ref_proteins is None:
            _, n2v_ref_proteins = _load_embeddings(sp_ref, NODE2VEC_DIR)

        if idx_r >= len(n2v_ref_proteins):
            continue
        gene_ref = n2v_ref_proteins[idx_r]
        ir = ref_idx_map.get(gene_ref)
        if ir is not None:
            rows_source.append(int(idx_s))
            rows_ref.append(int(ir))
            pair_weights.append(w)

    if len(rows_source) < 2:
        raise ValueError(f"Too few ortholog pairs ({len(rows_source)}) for {sp_source}-{sp_ref}")

    anchor_src = np.array(rows_source, dtype=np.int64)
    anchor_ref = np.array(rows_ref, dtype=np.int64)
    anchor_weights = np.array(pair_weights, dtype=np.float64)

    # Use uniform weights for vanilla weighting
    if weighting == "vanilla":
        anchor_weights = None

    if iterative:
        R, mu_src, mu_ref, s = _iterative_procrustes(
            emb_source, ref_emb,
            anchor_src, anchor_ref, anchor_weights,
            n_iters=n_iters,
            use_csls=use_csls,
            csls_k=csls_k,
            csls_threshold=csls_threshold,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            translate=translate,
            scale=scale,
        )
    else:
        X_src = emb_source[anchor_src]
        X_ref = ref_emb[anchor_ref]
        R, mu_src, mu_ref, s = _weighted_procrustes(
            X_src, X_ref, anchor_weights,
            translate=translate, scale=scale,
        )

    # Apply full Procrustes transform: aligned = s * (X - mu_src) @ R + mu_ref
    aligned = emb_source.copy()
    if mu_src is not None:
        aligned = aligned - mu_src
    aligned = aligned @ R * s
    if mu_ref is not None:
        aligned = aligned + mu_ref

    # Validate orthogonality
    ortho_err = np.linalg.norm(R.T @ R - np.eye(R.shape[0]))
    if ortho_err > 0.01:
        logger.warning(f"  {sp_source}: orthogonality error {ortho_err:.4f} > 0.01")

    if translate or scale:
        logger.debug(f"  {sp_source}: translate={translate} scale={scale} s={s:.4f}")

    return aligned, proteins_source, len(rows_source)


# ---------------------------------------------------------------------------
# Parallel worker for Stage 2
# ---------------------------------------------------------------------------

def _align_worker(args: tuple) -> tuple[str, str, int, str | None]:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    (sp_source, sp_ref, orthologs_dir_str, output_dir_str,
     weighting, iterative, use_csls, n_iters, csls_k, csls_threshold,
     use_gpu, gpu_id, translate, scale) = args

    orthologs_dir = Path(orthologs_dir_str)
    output_dir = Path(output_dir_str)

    try:
        # Get ref embeddings from preloaded globals
        global _PRELOADED_EMBEDDINGS
        if _PRELOADED_EMBEDDINGS is not None and sp_ref in _PRELOADED_EMBEDDINGS:
            ref_emb, ref_proteins = _PRELOADED_EMBEDDINGS[sp_ref]
        else:
            ref_emb, ref_proteins = _load_embeddings(sp_ref, NODE2VEC_DIR)

        # Get source embeddings from preloaded globals
        if _PRELOADED_EMBEDDINGS is not None and sp_source in _PRELOADED_EMBEDDINGS:
            emb_source, proteins_source = _PRELOADED_EMBEDDINGS[sp_source]
        else:
            emb_source, proteins_source = _load_embeddings(sp_source, NODE2VEC_DIR)

        aligned_emb, src_proteins, n_pairs = _align_one_species(
            sp_source, sp_ref, ref_emb, ref_proteins, orthologs_dir,
            weighting=weighting,
            iterative=iterative,
            use_csls=use_csls,
            n_iters=n_iters,
            csls_k=csls_k,
            csls_threshold=csls_threshold,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            emb_source=emb_source,
            proteins_source=proteins_source,
            translate=translate,
            scale=scale,
        )

        _save_aligned(sp_source, aligned_emb, src_proteins, output_dir)
        return sp_source, sp_ref, n_pairs, None

    except Exception as exc:
        return sp_source, sp_ref, 0, str(exc)


def _init_worker(preloaded: dict | None, _unused=None):
    """Initializer for ProcessPoolExecutor workers — sets module-level globals."""
    global _PRELOADED_EMBEDDINGS
    _PRELOADED_EMBEDDINGS = preloaded
    # Prevent BLAS thread oversubscription: each forked worker should use 1 thread
    # since parallelism comes from multiple workers, not intra-worker threading
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Pair building (reuses existing functions)
# ---------------------------------------------------------------------------

def _build_one_jaccard_pair(args: tuple) -> tuple[str, str, int, str | None]:
    """Worker for parallel Jaccard pair building."""
    sp_a, sp_b, h5_dir, transcripts_dir, network_dir, out_path = args
    from orbit.data_prep import build_jaccard_ortholog_pairs
    try:
        n = build_jaccard_ortholog_pairs(
            sp_a, sp_b, Path(h5_dir), Path(transcripts_dir), Path(network_dir), Path(out_path)
        )
        return sp_a, sp_b, n, None
    except Exception as e:
        return sp_a, sp_b, 0, str(e)


def _build_one_hybrid_pair(args: tuple) -> tuple[str, str, int, str | None]:
    """Worker for parallel hybrid pair building."""
    sp_a, sp_b, h5_dir, transcripts_dir, network_dir, out_path, alpha = args
    from orbit.data_prep import build_hybrid_ortholog_pairs
    try:
        n = build_hybrid_ortholog_pairs(
            sp_a, sp_b, Path(h5_dir), Path(transcripts_dir), Path(network_dir), Path(out_path),
            alpha=alpha,
        )
        return sp_a, sp_b, n, None
    except Exception as e:
        return sp_a, sp_b, 0, str(e)


def build_pairs(weighting: str, workers: int, alpha: float):
    """Build weighted ortholog pairs for all species pairs."""
    with open(SEED_RESULTS) as f:
        seed_info = json.load(f)
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]

    if weighting == "jaccard":
        out_dir = ORTHOLOGS_JACCARD_DIR
        worker_fn = _build_one_jaccard_pair
    elif weighting == "hybrid":
        out_dir = ORTHOLOGS_HYBRID_DIR
        worker_fn = _build_one_hybrid_pair
    else:
        logger.info("Vanilla pairs already exist in data/orthologs/, nothing to build")
        return

    work_items = []

    # Seed-seed pairs
    seed_dir = out_dir / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for sp_a, sp_b in combinations(sorted(seeds), 2):
        out_path = seed_dir / f"{sp_a}_{sp_b}.tsv"
        if out_path.exists():
            continue
        item = (sp_a, sp_b, str(NODE2VEC_DIR), str(TRANSCRIPTS_DIR), str(NETWORK_DIR), str(out_path))
        if weighting == "hybrid":
            item = item + (alpha,)
        work_items.append(item)

    # Non-seed pairs
    for ns in sorted(groups.keys()):
        ns_dir = out_dir / "non_seeds" / ns
        ns_dir.mkdir(parents=True, exist_ok=True)
        for seed in sorted(seeds):
            sp_a, sp_b = sorted([ns, seed])
            out_path = ns_dir / f"{sp_a}_{sp_b}.tsv"
            if out_path.exists():
                continue
            item = (sp_a, sp_b, str(NODE2VEC_DIR), str(TRANSCRIPTS_DIR), str(NETWORK_DIR), str(out_path))
            if weighting == "hybrid":
                item = item + (alpha,)
            work_items.append(item)

    if not work_items:
        logger.info(f"All {weighting} pairs already exist, nothing to do")
        return

    logger.info(f"Building {len(work_items)} {weighting} pairs with {workers} workers")
    t0 = time.perf_counter()

    done, failed = 0, []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in work_items}
        for future in as_completed(futures):
            sp_a, sp_b, n, err = future.result()
            done += 1
            if err:
                logger.error(f"  {sp_a}_{sp_b}: FAILED -- {err}")
                failed.append(f"{sp_a}_{sp_b}")
            elif done % 50 == 0 or done == len(work_items):
                elapsed = time.perf_counter() - t0
                logger.info(f"  [{done}/{len(work_items)}] pairs done ({elapsed:.1f}s)")

    if failed:
        logger.warning(f"Failed pairs ({len(failed)}): {failed}")

    elapsed = time.perf_counter() - t0
    logger.info(f"Pair building complete: {done} pairs in {elapsed:.1f}s ({elapsed/60:.1f}min)")


# ---------------------------------------------------------------------------
# Output directory selection
# ---------------------------------------------------------------------------

def _get_output_dir(weighting: str, iterative: bool, use_csls: bool,
                    translate: bool = False, scale: bool = False) -> Path:
    parts = ["aligned_embeddings_proc"]
    if weighting != "vanilla":
        parts.append(weighting)
    if iterative:
        parts.append("iter")
    if use_csls:
        parts.append("csls")
    if translate:
        parts.append("trans")
    if scale:
        parts.append("scale")
    return ROOT / "results" / "_".join(parts)


def _get_orthologs_dir(weighting: str) -> Path:
    if weighting == "jaccard":
        return ORTHOLOGS_JACCARD_DIR
    elif weighting == "hybrid":
        return ORTHOLOGS_HYBRID_DIR
    return ORTHOLOGS_VANILLA_DIR


def _get_eval_prefix(weighting: str, iterative: bool, use_csls: bool,
                     translate: bool = False, scale: bool = False) -> str:
    parts = ["evaluation_proc"]
    if weighting != "vanilla":
        parts.append(weighting)
    if iterative:
        parts.append("iter")
    if use_csls:
        parts.append("csls")
    if translate:
        parts.append("trans")
    if scale:
        parts.append("scale")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Preload embeddings
# ---------------------------------------------------------------------------

def _preload_all_embeddings() -> dict[str, tuple[np.ndarray, list[str]]]:
    """Preload all Node2Vec embeddings into memory (~2.3GB for 153 species)."""
    t0 = time.perf_counter()
    h5_files = sorted(NODE2VEC_DIR.glob("*.h5"))
    preloaded = {}
    for h5_path in h5_files:
        species = h5_path.stem
        emb, proteins = _load_embeddings(species, NODE2VEC_DIR)
        preloaded[species] = (emb, proteins)
    elapsed = time.perf_counter() - t0
    total_genes = sum(e.shape[0] for e, _ in preloaded.values())
    total_mb = sum(e.nbytes for e, _ in preloaded.values()) / 1e6
    logger.info(
        f"Preloaded {len(preloaded)} species ({total_genes:,} genes, "
        f"{total_mb:.0f}MB) in {elapsed:.1f}s"
    )
    return preloaded


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_alignment(args):
    """Run the two-stage Procrustes alignment."""
    weighting = args.weighting
    iterative = args.iterative
    use_csls = args.csls
    n_iters = args.n_iters
    csls_k = args.csls_k
    csls_threshold = args.csls_threshold
    workers = args.workers
    translate = args.translate
    scale = args.scale

    output_dir = _get_output_dir(weighting, iterative, use_csls, translate, scale)
    orthologs_dir = _get_orthologs_dir(weighting)

    # Check if GPU is available (for CSLS)
    use_gpu = False
    if use_csls:
        try:
            import faiss
            if faiss.get_num_gpus() > 0:
                use_gpu = True
                logger.info(f"FAISS GPU available: {faiss.get_num_gpus()} GPUs")
            else:
                logger.info("FAISS GPU not available, using CPU")
        except ImportError:
            logger.info("FAISS not installed with GPU support, using CPU")

    variant = f"{weighting}"
    if iterative:
        variant += "+iter"
    if use_csls:
        variant += "+csls"
    if translate:
        variant += "+trans"
    if scale:
        variant += "+scale"
    logger.info(f"Variant: {variant}")
    logger.info(f"Orthologs: {orthologs_dir}")
    logger.info(f"Output: {output_dir}")

    with open(SEED_RESULTS) as fh:
        seed_info = json.load(fh)
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]

    REFERENCE = "ARATH"
    non_ref_seeds = [s for s in seeds if s != REFERENCE]

    # -------------------------------------------------------------------
    # Preload embeddings
    # -------------------------------------------------------------------
    logger.info("Preloading all embeddings into memory...")
    preloaded = _preload_all_embeddings()

    # -------------------------------------------------------------------
    # Stage 1: Align seeds to reference (ARATH) — sequential
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Stage 1: aligning seeds to reference ({REFERENCE})")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    ref_emb, ref_proteins = preloaded[REFERENCE]
    _save_aligned(REFERENCE, ref_emb, ref_proteins, output_dir)
    logger.info(f"  {REFERENCE} -- reference (copied), {len(ref_proteins)} genes")

    aligned_seeds: dict[str, tuple[np.ndarray, list[str]]] = {
        REFERENCE: (ref_emb, ref_proteins),
    }

    for seed in non_ref_seeds:
        try:
            emb_seed, proteins_seed = preloaded[seed]
            aligned_emb, src_proteins, n_pairs = _align_one_species(
                seed, REFERENCE, ref_emb, ref_proteins, orthologs_dir,
                weighting=weighting,
                iterative=iterative,
                use_csls=use_csls,
                n_iters=n_iters,
                csls_k=csls_k,
                csls_threshold=csls_threshold,
                use_gpu=use_gpu,
                gpu_id=0,  # Stage 1: pin to GPU 0 (only 4 seeds, sequential)
                emb_source=emb_seed,
                proteins_source=proteins_seed,
                translate=translate,
                scale=scale,
            )
            _save_aligned(seed, aligned_emb, src_proteins, output_dir)
            aligned_seeds[seed] = (aligned_emb, src_proteins)
            logger.info(f"  {seed} -> {REFERENCE}: {n_pairs} pairs, {len(src_proteins)} genes")
        except Exception as exc:
            logger.error(f"  {seed}: FAILED -- {exc}")
            return 1

    seed_time = time.perf_counter() - t0
    logger.info(f"Stage 1 complete in {seed_time:.1f}s")

    # -------------------------------------------------------------------
    # Stage 2: Align non-seeds to nearest seed — parallel
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: aligning non-seeds to nearest seed")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    # Update preloaded with aligned seed embeddings (workers need aligned refs)
    for seed_name, (aligned_emb, aligned_prots) in aligned_seeds.items():
        preloaded[seed_name] = (aligned_emb, aligned_prots)

    # Detect GPUs for round-robin assignment
    n_gpus = 0
    if use_gpu:
        try:
            import faiss
            n_gpus = faiss.get_num_gpus()
        except Exception:
            pass

    non_seeds = sorted(groups.keys())
    work_items = []
    for i, ns in enumerate(non_seeds):
        nearest_seed = groups[ns]
        # Round-robin GPU assignment: each worker gets a specific GPU
        worker_gpu_id = i % n_gpus if n_gpus > 0 and use_gpu else None
        work_items.append((
            ns, nearest_seed, str(orthologs_dir), str(output_dir),
            weighting, iterative, use_csls, n_iters, csls_k, csls_threshold,
            use_gpu, worker_gpu_id, translate, scale,
        ))

    # Determine effective worker count
    effective_workers = min(workers, len(work_items))
    if use_gpu and n_gpus > 0:
        logger.info(f"Round-robin GPU assignment: {n_gpus} GPUs for FAISS CSLS")

    logger.info(f"Aligning {len(work_items)} non-seeds with {effective_workers} workers")

    failed = []
    success = 0

    # CUDA contexts don't survive fork(). For Stage 2 alignment, FAISS GPU
    # gives negligible benefit (sub-ms searches on ~30k×128 embeddings, iterative
    # refinement converges in 1 iteration). Use fork + CPU-only FAISS for maximum
    # parallelism with COW memory sharing. GPU resources are better spent on
    # the evaluation phase which does heavy kNN over all species.
    if sys.platform == "linux":
        mp_context = multiprocessing.get_context("fork")
    else:
        mp_context = multiprocessing.get_context("spawn")
    # Force CPU-only FAISS in workers to avoid CUDA fork issues
    for item_idx in range(len(work_items)):
        item = list(work_items[item_idx])
        item[10] = False   # use_gpu = False
        item[11] = None    # gpu_id = None
        work_items[item_idx] = tuple(item)

    if effective_workers <= 1:
        # Sequential fallback — set globals directly
        global _PRELOADED_EMBEDDINGS
        _PRELOADED_EMBEDDINGS = preloaded
        for item in work_items:
            sp_source, sp_ref, n_pairs, err = _align_worker(item)
            if err:
                logger.error(f"  {sp_source} -> {sp_ref}: FAILED -- {err}")
                failed.append(sp_source)
            else:
                success += 1
                if success % 20 == 0:
                    logger.info(f"  [{success}/{len(work_items)}] non-seeds aligned")
        _PRELOADED_EMBEDDINGS = None
    else:
        # Set module globals before fork so workers inherit via COW
        _PRELOADED_EMBEDDINGS = preloaded

        executor_kwargs = dict(
            max_workers=effective_workers,
            initializer=_init_worker,
            initargs=(preloaded, None),
            mp_context=mp_context,
        )

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            futures = {executor.submit(_align_worker, item): item for item in work_items}
            for future in as_completed(futures):
                sp_source, sp_ref, n_pairs, err = future.result()
                if err:
                    logger.error(f"  {sp_source} -> {sp_ref}: FAILED -- {err}")
                    failed.append(sp_source)
                else:
                    success += 1
                    if success % 20 == 0 or success == len(work_items):
                        elapsed = time.perf_counter() - t0
                        logger.info(
                            f"  [{success}/{len(work_items)}] non-seeds aligned ({elapsed:.1f}s)"
                        )

    # Clean up globals
    _PRELOADED_EMBEDDINGS = None

    nonseed_time = time.perf_counter() - t0
    logger.info(f"Stage 2 complete in {nonseed_time:.1f}s -- {success} ok, {len(failed)} failed")

    if failed:
        logger.warning(f"Failed species: {', '.join(failed)}")

    total_aligned = len(seeds) + success
    logger.info(f"Total aligned: {total_aligned} species -> {output_dir}")
    return 0


def run_evaluation(args):
    """Run evaluation on aligned embeddings."""
    from orbit.evaluate import evaluate_all_pairs, evaluate_all_within_species

    output_dir = _get_output_dir(args.weighting, args.iterative, args.csls,
                                  args.translate, args.scale)
    eval_prefix = _get_eval_prefix(args.weighting, args.iterative, args.csls,
                                    args.translate, args.scale)
    workers = args.workers

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
        n_workers=workers,
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
        n_workers=workers,
    )

    within_path = ROOT / "results" / f"{eval_prefix}_within.json"
    with open(within_path, "w") as fh:
        json.dump(within_results, fh, indent=2)

    valid_within = [r for r in within_results if "error" not in r]
    avg_prec = np.mean([r["precision_at_k"] for r in valid_within]) if valid_within else 0.0
    avg_rec = np.mean([r["recall_at_k"] for r in valid_within]) if valid_within else 0.0

    within_time = time.perf_counter() - t0
    logger.info(f"Within-species eval complete in {within_time:.1f}s")
    logger.info(f"  {len(valid_within)} species | avg P@50={avg_prec:.4f} | avg R@50={avg_rec:.4f}")
    logger.info(f"  Saved -> {within_path}")

    # Print comparison with baseline if available
    baseline_path = ROOT / "results" / "evaluation_procrustes_n2v.json"
    if baseline_path.exists():
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        valid_base = [r for r in baseline if "error" not in r]
        if valid_base:
            base_rho = np.mean([r["spearman_rho"] for r in valid_base])
            base_hits = np.mean([r["hits_at_k"] for r in valid_base])
            print("\n" + "=" * 70)
            print(f"{'Metric':>20s}  {'Baseline':>12s}  {'This variant':>12s}  {'Change':>10s}")
            print("-" * 70)
            print(f"{'avg Hits@50':>20s}  {base_hits:12.4f}  {avg_hits:12.4f}  {avg_hits/base_hits:.2f}x" if base_hits > 0 else f"{'avg Hits@50':>20s}  {base_hits:12.4f}  {avg_hits:12.4f}")
            print(f"{'avg Spearman':>20s}  {base_rho:12.4f}  {avg_rho:12.4f}  {avg_rho/base_rho:.2f}x" if base_rho > 0 else f"{'avg Spearman':>20s}  {base_rho:12.4f}  {avg_rho:12.4f}")
            print("=" * 70)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Improved Procrustes alignment with weighting, iteration, and CSLS"
    )
    parser.add_argument(
        "--stage", required=True, choices=["pairs", "align", "eval", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--weighting", default="vanilla", choices=["vanilla", "jaccard", "hybrid"],
        help="Ortholog pair weighting scheme (default: vanilla)",
    )
    parser.add_argument(
        "--iterative", action="store_true",
        help="Enable iterative Procrustes refinement",
    )
    parser.add_argument(
        "--csls", action="store_true",
        help="Use CSLS for MNN discovery (requires --iterative)",
    )
    parser.add_argument(
        "--n-iters", type=int, default=5,
        help="Number of refinement iterations (default: 5)",
    )
    parser.add_argument(
        "--alpha", type=float, default=5.0,
        help="Hybrid boost factor (default: 5.0)",
    )
    parser.add_argument(
        "--csls-threshold", type=float, default=0.5,
        help="MNN similarity threshold (default: 0.5)",
    )
    parser.add_argument(
        "--csls-k", type=int, default=10,
        help="k for CSLS neighborhood mean (default: 10)",
    )
    parser.add_argument(
        "--translate", action="store_true",
        help="Enable translation (center on anchor means before rotation)",
    )
    parser.add_argument(
        "--scale", action="store_true",
        help="Enable optimal uniform scaling after rotation",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help=f"Parallelism (default: {os.cpu_count()})",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation after alignment",
    )
    args = parser.parse_args()

    if args.csls and not args.iterative:
        logger.warning("--csls requires --iterative, enabling --iterative")
        args.iterative = True

    timings = {}
    pipeline_start = time.perf_counter()

    if args.stage in ("pairs", "all"):
        t0 = time.perf_counter()
        build_pairs(args.weighting, args.workers, args.alpha)
        timings["pair_building"] = time.perf_counter() - t0

    if args.stage in ("align", "all"):
        t0 = time.perf_counter()
        rc = run_alignment(args)
        timings["alignment"] = time.perf_counter() - t0
        if rc != 0:
            return rc

    if args.stage in ("eval", "all"):
        if args.stage == "all" and args.no_eval:
            logger.info("Skipping evaluation (--no-eval)")
        else:
            t0 = time.perf_counter()
            run_evaluation(args)
            timings["evaluation"] = time.perf_counter() - t0

    timings["pipeline_total"] = time.perf_counter() - pipeline_start

    # Timing summary
    logger.info("=" * 60)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 60)
    for step, secs in timings.items():
        mins = secs / 60
        if secs < 120:
            logger.info(f"  {step:.<30s} {secs:>8.1f}s")
        elif secs < 7200:
            logger.info(f"  {step:.<30s} {mins:>8.1f}min ({secs:.0f}s)")
        else:
            hrs = secs / 3600
            logger.info(f"  {step:.<30s} {hrs:>8.2f}h ({mins:.1f}min)")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
