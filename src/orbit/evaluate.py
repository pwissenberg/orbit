"""Evaluate cross-species alignment quality.

Metrics:
  - Hits@K: fraction of genes whose best Jaccard partner is in embedding KNN
  - MRR@K: Mean Reciprocal Rank of best Jaccard partner in embedding KNN
  - TopM-Hits@K: fraction of genes with any top-M Jaccard partner in KNN
  - Spearman correlation: Jaccard weight vs cosine similarity in aligned space
  - Shuffle baselines for all metrics
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from orbit.data_prep import _load_protein_to_og


# ---------------------------------------------------------------------------
# 1. Build Jaccard ground truth from coexpression neighborhoods
# ---------------------------------------------------------------------------


def _load_network_neighborhoods(network_path: Path) -> dict[str, set[str]]:
    """Load coexpression network and return gene -> set of neighbor genes."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    with open(network_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            neighbors[a].add(b)
            neighbors[b].add(a)
    return dict(neighbors)


def _convert_neighborhoods_to_ogs(
    neighbors: dict[str, set[str]],
    protein_to_og: dict[str, str],
) -> dict[str, set[str]]:
    """Convert coexpression neighborhoods from gene IDs to orthogroup IDs."""
    og_neighbors: dict[str, set[str]] = {}
    for gene, nbrs in neighbors.items():
        if gene not in protein_to_og:
            continue
        og_nbrs = set()
        for n in nbrs:
            if n in protein_to_og:
                og_nbrs.add(protein_to_og[n])
        if og_nbrs:
            og_neighbors[gene] = og_nbrs
    return og_neighbors


def build_jaccard_ground_truth(
    species_a: str,
    species_b: str,
    network_dir: Path,
    transcripts_dir: Path,
    h5_dir: Path,
) -> list[tuple[int, int, float]]:
    """Compute Jaccard similarity of coexpression neighborhoods for ortholog pairs.

    For each pair of genes (u in A, v in B) in the same orthogroup:
      1. Get their coexpression neighborhoods from the network
      2. Convert neighborhoods to orthogroup IDs
      3. Compute Jaccard(OG_neighbors_u, OG_neighbors_v)

    Returns list of (idx_a, idx_b, jaccard_score) tuples.
    """
    # Load H5 protein arrays for index mapping
    with h5py.File(h5_dir / f"{species_a}.h5", "r") as f:
        proteins_a = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]
    with h5py.File(h5_dir / f"{species_b}.h5", "r") as f:
        proteins_b = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]

    idx_a = {p: i for i, p in enumerate(proteins_a)}
    idx_b = {p: i for i, p in enumerate(proteins_b)}

    # Load OrthoFinder mappings
    og_a = _load_protein_to_og(transcripts_dir, species_a, set(idx_a.keys()))
    og_b = _load_protein_to_og(transcripts_dir, species_b, set(idx_b.keys()))

    # Load coexpression networks
    net_a = _load_network_neighborhoods(network_dir / f"{species_a}.tsv")
    net_b = _load_network_neighborhoods(network_dir / f"{species_b}.tsv")

    # Convert neighborhoods to OGs
    og_nbrs_a = _convert_neighborhoods_to_ogs(net_a, og_a)
    og_nbrs_b = _convert_neighborhoods_to_ogs(net_b, og_b)

    # Group genes by orthogroup
    og_genes_a: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_a.items():
        if pid in og_nbrs_a:
            og_genes_a[og].append(pid)

    og_genes_b: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_b.items():
        if pid in og_nbrs_b:
            og_genes_b[og].append(pid)

    # Compute Jaccard for all ortholog pairs
    shared_ogs = set(og_genes_a.keys()) & set(og_genes_b.keys())
    pairs = []
    for og in shared_ogs:
        for ga in og_genes_a[og]:
            for gb in og_genes_b[og]:
                na = og_nbrs_a[ga]
                nb = og_nbrs_b[gb]
                intersection = len(na & nb)
                union = len(na | nb)
                if union > 0:
                    jacc = intersection / union
                    if jacc > 0:
                        pairs.append((idx_a[ga], idx_b[gb], jacc))

    logger.info(
        f"  {species_a}-{species_b}: {len(pairs)} Jaccard pairs "
        f"from {len(shared_ogs)} shared OGs"
    )
    return pairs


# ---------------------------------------------------------------------------
# 2. Evaluate aligned embeddings
# ---------------------------------------------------------------------------


def _load_aligned_embeddings(h5_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load aligned embeddings from H5 file."""
    with h5py.File(h5_path, "r") as f:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]
        embeddings = f["embeddings"][:]
    return embeddings, proteins


def _load_node2vec_embeddings(h5_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load Node2Vec embeddings from H5 file (before alignment)."""
    # Node2Vec H5 files have the same structure as aligned embeddings
    return _load_aligned_embeddings(h5_path)


def compute_norm_diagnostics(
    species: str,
    node2vec_dir: Path,
    aligned_dir: Path,
) -> dict:
    """Compare embedding norms before and after alignment.

    Returns diagnostics showing if SPACE preserves, collapses, or expands embeddings.
    """
    # Load Node2Vec embeddings (before alignment)
    n2v_path = node2vec_dir / f"{species}.h5"
    aligned_path = aligned_dir / f"{species}.h5"

    if not n2v_path.exists() or not aligned_path.exists():
        return {"error": "missing embeddings"}

    X_before, _ = _load_node2vec_embeddings(n2v_path)
    X_after, _ = _load_aligned_embeddings(aligned_path)

    # Compute norms
    norm_before = np.linalg.norm(X_before, axis=1)
    norm_after = np.linalg.norm(X_after, axis=1)

    # Statistics
    return {
        "species": species,
        "n_genes": len(X_before),
        # Before alignment (Node2Vec)
        "norm_before_min": float(norm_before.min()),
        "norm_before_median": float(np.median(norm_before)),
        "norm_before_max": float(norm_before.max()),
        "norm_before_mean": float(norm_before.mean()),
        "norm_before_std": float(norm_before.std()),
        # After alignment (SPACE)
        "norm_after_min": float(norm_after.min()),
        "norm_after_median": float(np.median(norm_after)),
        "norm_after_max": float(norm_after.max()),
        "norm_after_mean": float(norm_after.mean()),
        "norm_after_std": float(norm_after.std()),
        # Change ratio
        "norm_ratio_mean": float(norm_after.mean() / norm_before.mean()) if norm_before.mean() > 0 else 0.0,
        "norm_ratio_median": float(np.median(norm_after) / np.median(norm_before)) if np.median(norm_before) > 0 else 0.0,
    }


def evaluate_pair(
    species_a: str,
    species_b: str,
    aligned_dir: Path,
    jaccard_pairs: list[tuple[int, int, float]],
    k: int = 50,
    top_m: int = 10,
    n_shuffles: int = 5,
    sample_n: int = 300_000,
) -> dict:
    """Evaluate alignment quality for one species pair.

    Returns dict with Hits@K, MRR@K, TopM-Hits@K, Spearman rho, and baselines.
    """
    XA, _ = _load_aligned_embeddings(aligned_dir / f"{species_a}.h5")
    XB, _ = _load_aligned_embeddings(aligned_dir / f"{species_b}.h5")
    nA, nB = len(XA), len(XB)

    if not jaccard_pairs:
        return {"error": "no jaccard pairs", "species_a": species_a, "species_b": species_b}

    # Embedding norm diagnostics (check for embedding collapse)
    norm_a = np.linalg.norm(XA, axis=1)
    norm_b = np.linalg.norm(XB, axis=1)
    norm_a_stats = {
        "norm_a_min": float(norm_a.min()),
        "norm_a_median": float(np.median(norm_a)),
        "norm_a_max": float(norm_a.max()),
        "norm_a_mean": float(norm_a.mean()),
    }
    norm_b_stats = {
        "norm_b_min": float(norm_b.min()),
        "norm_b_median": float(np.median(norm_b)),
        "norm_b_max": float(norm_b.max()),
        "norm_b_mean": float(norm_b.mean()),
    }

    # Convert pairs to arrays
    u = np.array([p[0] for p in jaccard_pairs], dtype=np.int64)
    v = np.array([p[1] for p in jaccard_pairs], dtype=np.int64)
    s = np.array([p[2] for p in jaccard_pairs], dtype=np.float32)

    # Build ground truth: best Jaccard partner per node in A
    best_v = np.full(nA, -1, dtype=np.int64)
    best_s = np.full(nA, -np.inf, dtype=np.float32)
    second_s = np.full(nA, -np.inf, dtype=np.float32)

    # Sort by (u, -s) to process in order
    order = np.lexsort((-s, u))
    u_sorted, v_sorted, s_sorted = u[order], v[order], s[order]

    # TopM ground truth
    top_v = np.full((nA, top_m), -1, dtype=np.int64)
    cur_u = -1
    fill = 0
    for uu, vv, ss in zip(u_sorted, v_sorted, s_sorted):
        if uu != cur_u:
            cur_u = uu
            fill = 0
        if best_v[uu] == -1:
            best_v[uu] = vv
            best_s[uu] = ss
        elif second_s[uu] == -np.inf:
            second_s[uu] = ss
        if fill < top_m:
            top_v[uu, fill] = vv
            fill += 1

    valid_best = best_v >= 0
    n_valid = int(valid_best.sum())

    if n_valid == 0:
        return {"error": "no valid ground truth", "species_a": species_a, "species_b": species_b}

    # L2-normalize
    eps = 1e-12
    XA_l2 = XA / (np.linalg.norm(XA, axis=1, keepdims=True) + eps)
    XB_l2 = XB / (np.linalg.norm(XB, axis=1, keepdims=True) + eps)

    # KNN retrieval
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(XB_l2)
    knn = nn.kneighbors(XA_l2, return_distance=False)  # (nA, K)

    # Hits@K and MRR@K against best_v
    match = knn == best_v[:, None]
    present = match.any(axis=1) & valid_best
    ranks = np.zeros(nA, dtype=np.int32)
    ranks[present] = match[present].argmax(axis=1) + 1
    rr = np.zeros(nA, dtype=np.float32)
    rr[present] = 1.0 / ranks[present]
    hits = present.astype(np.float32)

    hits_k = float(hits[valid_best].mean())
    mrr_k = float(rr[valid_best].mean())

    # TopM-Hits@K
    valid_top = top_v[:, 0] >= 0
    top_hit = (knn[:, :, None] == top_v[:, None, :]).any(axis=(1, 2))
    top_hits_k = float(top_hit[valid_top].mean()) if valid_top.any() else 0.0

    # Shuffle baselines
    rng = np.random.default_rng(42)
    shuffle_hits_list = []
    shuffle_mrr_list = []
    for _ in range(n_shuffles):
        idx_valid = np.where(valid_best)[0]
        best_v_shuf = best_v.copy()
        best_v_shuf[idx_valid] = rng.permutation(best_v[idx_valid])

        match_s = knn == best_v_shuf[:, None]
        present_s = match_s.any(axis=1) & valid_best
        ranks_s = np.zeros(nA, dtype=np.int32)
        ranks_s[present_s] = match_s[present_s].argmax(axis=1) + 1
        rr_s = np.zeros(nA, dtype=np.float32)
        rr_s[present_s] = 1.0 / ranks_s[present_s]
        shuffle_hits_list.append(float(present_s[valid_best].astype(float).mean()))
        shuffle_mrr_list.append(float(rr_s[valid_best].mean()))

    # Spearman correlation: Jaccard vs cosine similarity
    m = min(sample_n, len(u))
    samp = rng.choice(len(u), size=m, replace=False) if len(u) > m else np.arange(len(u))
    uu, vv, ss = u[samp], v[samp], s[samp]
    cos_sim = (XA_l2[uu] * XB_l2[vv]).sum(axis=1).astype(np.float32)
    rho, p = spearmanr(ss, cos_sim)

    # Shuffle Spearman
    vv_sh = rng.permutation(vv)
    cos_sim_sh = (XA_l2[uu] * XB_l2[vv_sh]).sum(axis=1).astype(np.float32)
    rho_sh, p_sh = spearmanr(ss, cos_sim_sh)

    # Jaccard gap analysis
    gap = best_s - second_s
    gap_valid = gap[valid_best & (second_s > -np.inf)]
    mean_gap = float(gap_valid.mean()) if len(gap_valid) > 0 else 0.0

    result = {
        "species_a": species_a,
        "species_b": species_b,
        "n_genes_a": nA,
        "n_genes_b": nB,
        "n_jaccard_pairs": len(jaccard_pairs),
        "n_valid_ground_truth": n_valid,
        "hits_at_k": hits_k,
        "mrr_at_k": mrr_k,
        "top_m_hits_at_k": top_hits_k,
        "shuffle_hits_at_k": float(np.mean(shuffle_hits_list)),
        "shuffle_mrr_at_k": float(np.mean(shuffle_mrr_list)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "shuffle_spearman_rho": float(rho_sh),
        "jaccard_gap_mean": mean_gap,
        "k": k,
        "top_m": top_m,
        **norm_a_stats,  # Add embedding norm diagnostics for species A
        **norm_b_stats,  # Add embedding norm diagnostics for species B
    }
    return result


# ---------------------------------------------------------------------------
# 3. Full evaluation across all species pairs
# ---------------------------------------------------------------------------


def _evaluate_single_pair(args: tuple) -> dict:
    """Worker function for parallel evaluation of a single species pair."""
    sp_a, sp_b, pair_type, aligned_dir, network_dir, transcripts_dir, k, top_m = args
    logger.info(f"Evaluating {sp_a} vs {sp_b} ({pair_type})")
    pairs = build_jaccard_ground_truth(
        sp_a, sp_b, network_dir, transcripts_dir, aligned_dir
    )
    result = evaluate_pair(sp_a, sp_b, aligned_dir, pairs, k=k, top_m=top_m)
    result["pair_type"] = pair_type
    _log_result(result)
    return result


def evaluate_all_pairs(
    aligned_dir: Path,
    network_dir: Path,
    transcripts_dir: Path,
    seed_results_path: Path,
    k: int = 50,
    top_m: int = 10,
    eval_mode: str = "seeds",
    n_workers: int | None = None,
) -> list[dict]:
    """Evaluate all species pairs.

    eval_mode:
      - "seeds": evaluate only seed-seed pairs (10 pairs for 5 seeds)
      - "all": evaluate seed-seed + seed-nonseed pairs

    n_workers: number of parallel processes. Defaults to min(n_pairs, 32).
      Set to 1 to disable parallelism.
    """
    with open(seed_results_path) as f:
        seed_info = json.load(f)
    seeds = seed_info["seeds"]
    groups = seed_info.get("groups", {})

    # Build list of all jobs
    jobs = []

    if eval_mode in ("seeds", "all"):
        for sp_a, sp_b in combinations(seeds, 2):
            jobs.append((sp_a, sp_b, "seed-seed", aligned_dir, network_dir, transcripts_dir, k, top_m))

    if eval_mode == "all":
        for ns, nearest_seed in groups.items():
            aligned_path = aligned_dir / f"{ns}.h5"
            if not aligned_path.exists():
                continue
            sp_a, sp_b = sorted([ns, nearest_seed])
            jobs.append((sp_a, sp_b, "seed-nonseed", aligned_dir, network_dir, transcripts_dir, k, top_m))

    if not jobs:
        return []

    workers = n_workers if n_workers is not None else min(len(jobs), os.cpu_count() or 32)
    logger.info(f"Evaluating {len(jobs)} pairs with {workers} parallel workers")

    if workers <= 1:
        return [_evaluate_single_pair(job) for job in jobs]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_single_pair, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results: seed-seed first, then seed-nonseed, alphabetical within each
    results.sort(key=lambda r: (0 if r.get("pair_type") == "seed-seed" else 1, r.get("species_a", ""), r.get("species_b", "")))
    return results


def _log_result(r: dict) -> None:
    """Pretty-print one evaluation result."""
    if "error" in r:
        logger.warning(f"  {r['species_a']}-{r['species_b']}: {r['error']}")
        return
    logger.info(
        f"  {r['species_a']}-{r['species_b']}: "
        f"Hits@{r['k']}={r['hits_at_k']:.4f}  "
        f"MRR@{r['k']}={r['mrr_at_k']:.4f}  "
        f"Top{r['top_m']}-Hits@{r['k']}={r['top_m_hits_at_k']:.4f}  "
        f"Spearman={r['spearman_rho']:.4f}  "
        f"(shuffle: Hits={r['shuffle_hits_at_k']:.4f}  "
        f"Spearman={r['shuffle_spearman_rho']:.4f})"
    )


# ---------------------------------------------------------------------------
# 4. Within-species evaluation (precision@K / recall@K)
# ---------------------------------------------------------------------------


def evaluate_within_species(
    species: str,
    aligned_dir: Path,
    network_dir: Path,
    k: int = 50,
    n_shuffles: int = 5,
) -> dict:
    """Evaluate within-species embedding quality.

    For each gene, check whether its cosine kNN neighbors in embedding space
    match its coexpression network neighbors.

    Returns dict with precision@K, recall@K, shuffle baselines, etc.
    """
    h5_path = aligned_dir / f"{species}.h5"
    if not h5_path.exists():
        return {"species": species, "error": "no aligned embeddings"}

    net_path = network_dir / f"{species}.tsv"
    if not net_path.exists():
        return {"species": species, "error": "no network file"}

    embeddings, proteins = _load_aligned_embeddings(h5_path)
    neighbors = _load_network_neighborhoods(net_path)

    # Filter to genes present in both embeddings and network
    protein_set = set(proteins)
    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    common_genes = protein_set & set(neighbors.keys())

    if not common_genes:
        return {"species": species, "error": "no genes in both embeddings and network"}

    # L2-normalize embeddings
    eps = 1e-12
    X = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + eps)

    # Build kNN index and batch-query ALL genes at once (much faster than one-by-one)
    n_neighbors = min(k + 1, len(proteins))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X)
    all_knn = nn.kneighbors(X, return_distance=False)  # (N, k+1) — single batch call

    # Collect eval genes: those in both embeddings and network with neighbors
    eval_indices = []
    eval_nbr_sets = []
    for gene in common_genes:
        nbrs = neighbors[gene] & protein_set
        if not nbrs:
            continue
        eval_indices.append(protein_to_idx[gene])
        eval_nbr_sets.append(nbrs)

    if not eval_indices:
        return {"species": species, "error": "no genes with network neighbors in embeddings"}

    # Compute precision/recall from batch kNN results
    precisions = []
    recalls = []
    for idx, nbrs in zip(eval_indices, eval_nbr_sets):
        knn_indices = all_knn[idx]
        knn_indices = knn_indices[knn_indices != idx][:k]
        knn_genes = {proteins[i] for i in knn_indices}
        overlap = len(knn_genes & nbrs)
        precisions.append(overlap / k)
        recalls.append(overlap / min(len(nbrs), k))

    precision_at_k = float(np.mean(precisions))
    recall_at_k = float(np.mean(recalls))

    # Shuffle baselines: permute embedding rows, batch-query, recompute
    rng = np.random.default_rng(42)
    shuffle_prec_list = []
    shuffle_rec_list = []

    for _ in range(n_shuffles):
        perm = rng.permutation(len(proteins))
        X_shuf = X[perm]
        nn_shuf = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn_shuf.fit(X_shuf)
        all_knn_shuf = nn_shuf.kneighbors(X_shuf, return_distance=False)  # batch query

        s_prec = []
        s_rec = []
        for idx, nbrs in zip(eval_indices, eval_nbr_sets):
            knn_idx = all_knn_shuf[idx]
            knn_idx = knn_idx[knn_idx != idx][:k]
            knn_genes = {proteins[perm[i]] for i in knn_idx}
            overlap = len(knn_genes & nbrs)
            s_prec.append(overlap / k)
            s_rec.append(overlap / min(len(nbrs), k))

        if s_prec:
            shuffle_prec_list.append(float(np.mean(s_prec)))
            shuffle_rec_list.append(float(np.mean(s_rec)))

    result = {
        "species": species,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "n_eval": len(precisions),
        "n_genes": len(proteins),
        "shuffle_precision_mean": float(np.mean(shuffle_prec_list)) if shuffle_prec_list else 0.0,
        "shuffle_recall_mean": float(np.mean(shuffle_rec_list)) if shuffle_rec_list else 0.0,
        "k": k,
    }

    logger.info(
        f"  {species}: P@{k}={precision_at_k:.4f}  R@{k}={recall_at_k:.4f}  "
        f"n_eval={len(precisions)}  "
        f"(shuffle: P={result['shuffle_precision_mean']:.4f}  R={result['shuffle_recall_mean']:.4f})"
    )
    return result


def _evaluate_within_single(args: tuple) -> dict:
    """Worker function for parallel within-species evaluation."""
    species, aligned_dir, network_dir, k = args
    return evaluate_within_species(species, aligned_dir, network_dir, k=k)


def evaluate_all_within_species(
    aligned_dir: Path,
    network_dir: Path,
    k: int = 50,
    n_workers: int | None = None,
) -> list[dict]:
    """Evaluate within-species metrics for all species with aligned embeddings.

    Returns list of per-species result dicts.
    """
    h5_files = sorted(aligned_dir.glob("*.h5"))
    species_list = [f.stem for f in h5_files]

    if not species_list:
        logger.warning("No aligned embeddings found")
        return []

    jobs = [(sp, aligned_dir, network_dir, k) for sp in species_list]
    workers = n_workers if n_workers is not None else min(len(jobs), os.cpu_count() or 32)
    logger.info(f"Within-species eval: {len(jobs)} species with {workers} workers")

    if workers <= 1:
        return [_evaluate_within_single(job) for job in jobs]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_within_single, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r.get("species", ""))
    return results
