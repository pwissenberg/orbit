"""Compute within-species Jaccard similarity matrices for coexpression networks.

Three computation backends, selectable via the `method` argument:

  cpu (default)
    Vectorized scipy sparse matrix multiplication.
    Builds binary adjacency A, computes A @ A.T in C-level BLAS.
    Replaces the old Python-level pair loop — ~20-50x faster.

  gpu
    Same algorithm via cupy (GPU sparse matmul on CUDA).
    Requires: uv add cupy-cuda12x
    ~5-15x faster than cpu backend for large species.

  auto
    Uses gpu if cupy is importable, otherwise cpu.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix, save_npz

Method = Literal["auto", "cpu", "gpu"]


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


def _build_adjacency_lists(
    genes: list[str],
    neighbors: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) index arrays for the binary adjacency matrix."""
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows: list[int] = []
    cols: list[int] = []
    for i, gene in enumerate(genes):
        for nb in neighbors.get(gene, set()):
            j = gene_to_idx.get(nb)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


def _compute_jaccard_cpu(
    genes: list[str],
    neighbors: dict[str, set[str]],
    log_progress: bool = True,
) -> csr_matrix:
    """Vectorized Jaccard via scipy sparse matrix multiplication.

    Algorithm:
        A         = binary adjacency matrix  (n × n, sparse)
        inter     = A @ A.T                  (A@A.T)[i,j] = |N(i) ∩ N(j)|
        degrees   = A.sum(axis=1)            |N(i)|
        union[i,j]  = deg[i] + deg[j] - inter[i,j]
        jaccard     = inter / union

    One C-level sparse BLAS call replaces the O(E_candidates) Python loop.
    """
    n = len(genes)

    if log_progress:
        logger.info("  Building adjacency matrix (CPU)...")

    rows, cols = _build_adjacency_lists(genes, neighbors)
    ones = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((ones, (rows, cols)), shape=(n, n))

    if log_progress:
        logger.info(f"  Adjacency: {A.nnz} edges → computing A @ A.T ...")

    inter = A @ A.T  # sparse × sparse → sparse (scipy C-level BLAS)

    if log_progress:
        logger.info(f"  Intersection matrix: {inter.nnz} nonzero pairs")

    degrees = np.asarray(A.sum(axis=1)).ravel()

    # A @ A.T is symmetric, so inter already contains both (i,j) and (j,i).
    # Compute Jaccard on all off-diagonal entries directly — no concatenation needed.
    inter_coo = inter.tocoo()
    off_diag = inter_coo.row != inter_coo.col
    I = inter_coo.row[off_diag]
    J = inter_coo.col[off_diag]
    inter_vals = inter_coo.data[off_diag]

    union_vals = degrees[I] + degrees[J] - inter_vals
    jac_vals = np.where(union_vals > 0, inter_vals / union_vals, 0.0).astype(np.float32)

    return csr_matrix((jac_vals, (I, J)), shape=(n, n), dtype=np.float32)


def _compute_jaccard_gpu(
    genes: list[str],
    neighbors: dict[str, set[str]],
    log_progress: bool = True,
) -> csr_matrix:
    """Vectorized Jaccard via cupy GPU sparse matrix multiplication.

    Same algorithm as _compute_jaccard_cpu but runs on CUDA.
    Requires cupy: uv add cupy-cuda12x
    """
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as cp_csr
    except ImportError as exc:
        raise ImportError(
            "cupy is not installed. Install with: uv add cupy-cuda12x"
        ) from exc

    n = len(genes)

    if log_progress:
        logger.info("  Building adjacency matrix (GPU)...")

    rows_np, cols_np = _build_adjacency_lists(genes, neighbors)

    r_cp = cp.asarray(rows_np)
    c_cp = cp.asarray(cols_np)
    ones_cp = cp.ones(len(rows_np), dtype=cp.float32)
    A = cp_csr((ones_cp, (r_cp, c_cp)), shape=(n, n))

    if log_progress:
        logger.info(f"  GPU adjacency: {A.nnz} edges → computing A @ A.T ...")

    inter = A @ A.T

    # Guard against a known cupy/cuSPARSE silent bug: for certain matrix
    # structures the sparse matmul returns nnz=0 without raising an error.
    # Detected by checking that a non-empty A produces a non-empty result.
    if inter.nnz == 0 and A.nnz > 0:
        raise MemoryError(
            "cupy A@A.T returned nnz=0 for non-empty matrix "
            "(cuSPARSE silent bug) — will retry on CPU"
        )

    if log_progress:
        logger.info(f"  GPU intersection matrix: {inter.nnz} nonzero pairs")

    degrees_cp = cp.asarray(A.sum(axis=1)).ravel()

    # A @ A.T is symmetric — inter already has both (i,j) and (j,i).
    # Compute Jaccard on all off-diagonal entries on GPU, then transfer the
    # full CSR in one shot with .get() instead of per-array transfers + concatenation.
    inter_coo = inter.tocoo()
    off_diag = inter_coo.row != inter_coo.col
    I_cp = inter_coo.row[off_diag]
    J_cp = inter_coo.col[off_diag]
    inter_vals_cp = inter_coo.data[off_diag]

    union_vals_cp = degrees_cp[I_cp] + degrees_cp[J_cp] - inter_vals_cp
    jac_cp = cp.where(
        union_vals_cp > 0, inter_vals_cp / union_vals_cp, 0.0
    ).astype(cp.float32)

    # Build CSR on GPU, transfer entire matrix in one .get() call
    result_gpu = cp_csr((jac_cp, (I_cp, J_cp)), shape=(n, n))
    result = result_gpu.get()  # scipy csr_matrix

    # free_all_blocks() only releases UNREFERENCED blocks.
    # Explicitly delete every cupy array first so the GC drops all references
    # before we call free, otherwise 6+ GB stays pinned and causes OOM on
    # the next species this worker processes.
    del result_gpu, jac_cp, union_vals_cp, inter_vals_cp
    del I_cp, J_cp, off_diag, inter_coo, degrees_cp, inter, A
    del ones_cp, r_cp, c_cp
    cp.get_default_memory_pool().free_all_blocks()

    return result


def _resolve_method(method: Method) -> Literal["cpu", "gpu"]:
    if method == "gpu":
        return "gpu"
    if method == "cpu":
        return "cpu"
    # auto: use GPU if cupy available
    try:
        import cupy  # noqa: F401
        return "gpu"
    except ImportError:
        return "cpu"


def compute_jaccard_matrix(
    network_path: Path,
    output_path: Path,
    method: Method = "auto",
) -> dict[str, Any]:
    """Compute within-species Jaccard similarity matrix from coexpression network.

    Args:
        network_path: Path to TSV network file (geneA, geneB, weight)
        output_path: Path to save sparse matrix (.npz format)
        method: Computation backend — "cpu", "gpu", or "auto" (GPU if cupy available)

    Returns:
        Metadata dict: species, n_genes, density, size_mb, nnz, backend
    """
    species = network_path.stem
    resolved = _resolve_method(method)
    logger.info(f"Computing Jaccard for {species} [requested={resolved}]")

    neighbors = _load_network_neighborhoods(network_path)
    genes = sorted(neighbors.keys())
    n_genes = len(genes)
    logger.info(f"  {n_genes} genes loaded")

    if resolved == "gpu":
        try:
            J = _compute_jaccard_gpu(genes, neighbors, log_progress=True)
        except (MemoryError, Exception) as exc:
            # Catch both:
            #   cupy.cuda.memory.OutOfMemoryError  (VRAM exhausted)
            #   MemoryError we raise for cuSPARSE silent-zero bug
            # Also catches secondary thrust allocator crashes that surface as
            # generic exceptions with "out of memory" in the message.
            is_oom = (
                isinstance(exc, MemoryError)
                or "out of memory" in str(exc).lower()
                or "OutOfMemory" in type(exc).__name__
            )
            if is_oom:
                # Free whatever GPU memory is still held before falling back
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                logger.warning(
                    f"  GPU OOM for {species} ({n_genes} genes) — falling back to CPU"
                )
                resolved = "cpu"
                J = _compute_jaccard_cpu(genes, neighbors, log_progress=True)
            else:
                raise
    else:
        J = _compute_jaccard_cpu(genes, neighbors, log_progress=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, J, compressed=False)  # uncompressed: ~60x faster save

    density = J.nnz / (n_genes**2) if n_genes > 0 else 0.0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved {size_mb:.1f} MB, density={density:.2%}, nnz={J.nnz}")

    return {
        "species": species,
        "n_genes": n_genes,
        "density": density,
        "size_mb": size_mb,
        "nnz": J.nnz,
        "backend": resolved,
    }
