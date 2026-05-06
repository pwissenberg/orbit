#!/usr/bin/env python3
"""Runtime benchmark: SPACE autoencoder vs Procrustes vs Procrustes+Jaccard+iter+CSLS.

Times the alignment step only (not data prep or evaluation) for all 153 species.
Each method is run serially (no parallelism) for a fair comparison.
Outputs per-species times, totals, and hardware info to results/benchmark_runtime.json.

Usage:
    # Run all three methods (~5 hours, SPACE dominates)
    uv run python scripts/benchmark_runtime.py

    # Run a single method
    uv run python scripts/benchmark_runtime.py --method procrustes
    uv run python scripts/benchmark_runtime.py --method procrustes_jaccard
    uv run python scripts/benchmark_runtime.py --method space

    # Dry run (check data availability)
    uv run python scripts/benchmark_runtime.py --dry-run

    # Plot results (after benchmark completes)
    uv run python scripts/benchmark_runtime.py --plot
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from scipy.linalg import orthogonal_procrustes

ROOT = Path(__file__).resolve().parent.parent

NODE2VEC_DIR = ROOT / "data" / "node2vec"
ORTHOLOGS_DIR = ROOT / "data" / "orthologs"
ORTHOLOGS_JACCARD_DIR = ROOT / "data" / "orthologs_jaccard"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"
OUTPUT_DIR = ROOT / "results" / "benchmark_runtime"
RESULTS_PATH = ROOT / "results" / "benchmark_runtime.json"

REFERENCE = "ARATH"


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def _collect_hardware_info() -> dict:
    """Collect hardware specs for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    # CPU
    try:
        out = subprocess.check_output(
            ["grep", "-m1", "model name", "/proc/cpuinfo"],
            text=True, stderr=subprocess.DEVNULL,
        )
        info["cpu_model"] = out.split(":")[1].strip()
    except Exception:
        info["cpu_model"] = platform.processor() or "unknown"

    try:
        import os
        info["cpu_cores"] = os.cpu_count()
    except Exception:
        pass

    # RAM
    try:
        out = subprocess.check_output(
            ["grep", "MemTotal", "/proc/meminfo"],
            text=True, stderr=subprocess.DEVNULL,
        )
        kb = int(out.split()[1])
        info["ram_gb"] = round(kb / 1024 / 1024, 1)
    except Exception:
        pass

    # GPU
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        gpus = [line.strip() for line in out.strip().split("\n") if line.strip()]
        info["gpus"] = gpus
        info["gpu_count"] = len(gpus)
    except Exception:
        info["gpus"] = []
        info["gpu_count"] = 0

    return info


# ---------------------------------------------------------------------------
# I/O helpers
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


def _load_ortholog_pairs(path: Path) -> np.ndarray:
    return np.loadtxt(path, usecols=(0, 1), dtype=np.int64)


def _load_ortholog_pairs_weighted(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    indices = data[:, :2].astype(np.int64)
    if data.shape[1] >= 3:
        weights = data[:, 2].astype(np.float64)
    else:
        weights = np.ones(len(indices), dtype=np.float64)
    return indices, weights


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


def _load_seed_info() -> dict:
    with open(SEED_RESULTS) as f:
        return json.load(f)


def _make_result(method: str, total: float, stage1: float, stage2: float,
                 species_times: dict) -> dict:
    valid = [t for t in species_times.values() if t is not None]
    return {
        "method": method,
        "total_seconds": total,
        "stage1_seconds": stage1,
        "stage2_seconds": stage2,
        "n_species": len(species_times),
        "n_success": len(valid),
        "per_species_mean": float(np.mean(valid)),
        "per_species_std": float(np.std(valid)),
        "per_species_median": float(np.median(valid)),
        "per_species_min": float(np.min(valid)),
        "per_species_max": float(np.max(valid)),
        "per_species_times": species_times,
    }


# ---------------------------------------------------------------------------
# Method 1: Procrustes (baseline, unweighted)
# ---------------------------------------------------------------------------

def _procrustes_align_one(
    sp_source: str,
    sp_ref: str,
    ref_emb: np.ndarray,
    ref_proteins: list[str],
    orthologs_dir: Path,
) -> tuple[np.ndarray, list[str], int]:
    pair_path = _find_ortholog_path(sp_source, sp_ref, orthologs_dir)
    if pair_path is None:
        raise FileNotFoundError(f"No ortholog pairs for {sp_source} vs {sp_ref}")

    stem = pair_path.stem
    sp_first = stem.split("_")[0]
    col_source, col_ref = (0, 1) if sp_source == sp_first else (1, 0)

    pairs = _load_ortholog_pairs(pair_path)
    emb_source, proteins_source = _load_embeddings(sp_source, NODE2VEC_DIR)
    ref_idx_map = {g: i for i, g in enumerate(ref_proteins)}

    rows_source, rows_ref = [], []
    for idx_s, idx_r in zip(pairs[:, col_source], pairs[:, col_ref]):
        if idx_s >= len(proteins_source) or idx_r >= len(ref_proteins):
            continue
        ir = ref_idx_map.get(ref_proteins[idx_r])
        if ir == idx_r:
            rows_source.append(int(idx_s))
            rows_ref.append(int(idx_r))

    if len(rows_source) < 2:
        raise ValueError(f"Too few pairs ({len(rows_source)}) for {sp_source}-{sp_ref}")

    R, _ = orthogonal_procrustes(emb_source[rows_source], ref_emb[rows_ref])
    return emb_source @ R, proteins_source, len(rows_source)


def benchmark_procrustes(seed_info: dict, output_subdir: Path) -> dict:
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]
    non_ref_seeds = [s for s in seeds if s != REFERENCE]

    species_times = {}
    t_total = time.perf_counter()

    # Reference (copy)
    t0 = time.perf_counter()
    ref_emb, ref_proteins = _load_embeddings(REFERENCE, NODE2VEC_DIR)
    _save_aligned(REFERENCE, ref_emb, ref_proteins, output_subdir)
    species_times[REFERENCE] = time.perf_counter() - t0

    aligned_seeds = {REFERENCE: (ref_emb, ref_proteins)}

    # Stage 1: seeds -> ARATH
    for seed in non_ref_seeds:
        t0 = time.perf_counter()
        aligned_emb, src_proteins, _ = _procrustes_align_one(
            seed, REFERENCE, ref_emb, ref_proteins, ORTHOLOGS_DIR
        )
        _save_aligned(seed, aligned_emb, src_proteins, output_subdir)
        aligned_seeds[seed] = (aligned_emb, src_proteins)
        species_times[seed] = time.perf_counter() - t0

    stage1_time = time.perf_counter() - t_total

    # Stage 2: non-seeds -> nearest seed
    t_stage2 = time.perf_counter()
    for i, ns in enumerate(sorted(groups.keys())):
        nearest_seed = groups[ns]
        seed_emb, seed_prots = aligned_seeds[nearest_seed]
        t0 = time.perf_counter()
        try:
            aligned_emb, ns_proteins, _ = _procrustes_align_one(
                ns, nearest_seed, seed_emb, seed_prots, ORTHOLOGS_DIR
            )
            _save_aligned(ns, aligned_emb, ns_proteins, output_subdir)
            species_times[ns] = time.perf_counter() - t0
        except Exception as e:
            logger.error(f"  {ns}: FAILED -- {e}")
            species_times[ns] = None
        if (i + 1) % 50 == 0:
            logger.info(f"  [{i+1}/{len(groups)}] non-seeds done")

    stage2_time = time.perf_counter() - t_stage2
    total_time = time.perf_counter() - t_total

    return _make_result("procrustes", total_time, stage1_time, stage2_time, species_times)


# ---------------------------------------------------------------------------
# Method 2: Procrustes + Jaccard + iterative + CSLS
# ---------------------------------------------------------------------------

def _weighted_procrustes(X_src: np.ndarray, X_ref: np.ndarray,
                         weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None or np.allclose(weights, weights[0]):
        M = X_src.T @ X_ref
    else:
        w_sqrt = np.sqrt(weights / weights.sum())
        M = (X_src * w_sqrt[:, None]).T @ (X_ref * w_sqrt[:, None])

    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(U @ Vt)
    D = np.diag([*np.ones(len(U) - 1), d])
    return (U @ D @ Vt).astype(np.float32)


def _csls_knn(X_src: np.ndarray, X_ref: np.ndarray, k: int = 10,
              use_gpu: bool = True):
    import faiss

    eps = 1e-12
    X_src_n = np.ascontiguousarray(
        X_src / (np.linalg.norm(X_src, axis=1, keepdims=True) + eps), dtype=np.float32
    )
    X_ref_n = np.ascontiguousarray(
        X_ref / (np.linalg.norm(X_ref, axis=1, keepdims=True) + eps), dtype=np.float32
    )

    index_ref = faiss.IndexFlatIP(X_ref_n.shape[1])
    index_src = faiss.IndexFlatIP(X_src_n.shape[1])
    if use_gpu:
        try:
            index_ref = faiss.index_cpu_to_all_gpus(index_ref)
            index_src = faiss.index_cpu_to_all_gpus(index_src)
        except Exception:
            pass
    index_ref.add(X_ref_n)
    index_src.add(X_src_n)

    k_actual = max(min(k, min(len(X_src), len(X_ref))), 1)
    sim_s2r, idx_s2r = index_ref.search(X_src_n, k_actual)
    sim_r2s, idx_r2s = index_src.search(X_ref_n, k_actual)

    r_src = sim_s2r.mean(axis=1)
    r_ref = sim_r2s.mean(axis=1)

    idx_top1_s2r = idx_s2r[:, 0]
    csls_s2r = 2 * sim_s2r[:, 0] - r_src - r_ref[idx_top1_s2r]

    return idx_top1_s2r, idx_r2s[:, 0], csls_s2r


def _iterative_procrustes(
    emb_source: np.ndarray,
    emb_ref: np.ndarray,
    anchor_src_idx: np.ndarray,
    anchor_ref_idx: np.ndarray,
    anchor_weights: np.ndarray | None,
    n_iters: int = 5,
    csls_k: int = 10,
    csls_threshold: float = 0.5,
    use_gpu: bool = True,
) -> np.ndarray:
    R = _weighted_procrustes(
        emb_source[anchor_src_idx], emb_ref[anchor_ref_idx], anchor_weights
    )
    anchor_set = set(zip(anchor_src_idx.tolist(), anchor_ref_idx.tolist()))
    mean_w = float(anchor_weights.mean()) if anchor_weights is not None else 1.0

    for it in range(n_iters):
        R_old = R.copy()
        X_aligned = emb_source @ R

        nn_s2r, nn_r2s, scores_s2r = _csls_knn(
            X_aligned, emb_ref, k=csls_k, use_gpu=use_gpu
        )

        n_src = len(emb_source)
        mnn_mask = (nn_r2s[nn_s2r] == np.arange(n_src)) & (scores_s2r >= csls_threshold)
        new_pairs = [
            (int(s), int(nn_s2r[s]))
            for s in np.where(mnn_mask)[0]
            if (int(s), int(nn_s2r[s])) not in anchor_set
        ]

        if not new_pairs:
            break

        all_src = np.concatenate([anchor_src_idx, np.array([p[0] for p in new_pairs])])
        all_ref = np.concatenate([anchor_ref_idx, np.array([p[1] for p in new_pairs])])
        all_w = np.concatenate([
            anchor_weights if anchor_weights is not None else np.ones(len(anchor_src_idx)),
            np.full(len(new_pairs), mean_w * 0.5),
        ])

        R = _weighted_procrustes(emb_source[all_src], emb_ref[all_ref], all_w)

        if np.linalg.norm(R - R_old, ord="fro") < 1e-5:
            break

    return R


def _procrustes_jaccard_align_one(
    sp_source: str,
    sp_ref: str,
    ref_emb: np.ndarray,
    ref_proteins: list[str],
    orthologs_dir: Path,
    use_gpu: bool = True,
) -> tuple[np.ndarray, list[str], int]:
    pair_path = _find_ortholog_path(sp_source, sp_ref, orthologs_dir)
    if pair_path is None:
        raise FileNotFoundError(f"No ortholog pairs for {sp_source} vs {sp_ref}")

    stem = pair_path.stem
    sp_first = stem.split("_")[0]
    col_source, col_ref = (0, 1) if sp_source == sp_first else (1, 0)

    pairs, weights = _load_ortholog_pairs_weighted(pair_path)
    emb_source, proteins_source = _load_embeddings(sp_source, NODE2VEC_DIR)
    ref_idx_map = {g: i for i, g in enumerate(ref_proteins)}

    rows_source, rows_ref, row_weights = [], [], []
    for i, (idx_s, idx_r) in enumerate(zip(pairs[:, col_source], pairs[:, col_ref])):
        if idx_s >= len(proteins_source) or idx_r >= len(ref_proteins):
            continue
        ir = ref_idx_map.get(ref_proteins[idx_r])
        if ir == idx_r:
            rows_source.append(int(idx_s))
            rows_ref.append(int(idx_r))
            row_weights.append(weights[i])

    if len(rows_source) < 2:
        raise ValueError(f"Too few pairs ({len(rows_source)}) for {sp_source}-{sp_ref}")

    R = _iterative_procrustes(
        emb_source, ref_emb,
        np.array(rows_source), np.array(rows_ref), np.array(row_weights),
        n_iters=5, csls_k=10, csls_threshold=0.5, use_gpu=use_gpu,
    )
    return emb_source @ R, proteins_source, len(rows_source)


def benchmark_procrustes_jaccard(seed_info: dict, output_subdir: Path,
                                  use_gpu: bool = True) -> dict:
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]
    non_ref_seeds = [s for s in seeds if s != REFERENCE]

    species_times = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    ref_emb, ref_proteins = _load_embeddings(REFERENCE, NODE2VEC_DIR)
    _save_aligned(REFERENCE, ref_emb, ref_proteins, output_subdir)
    species_times[REFERENCE] = time.perf_counter() - t0

    aligned_seeds = {REFERENCE: (ref_emb, ref_proteins)}

    for seed in non_ref_seeds:
        t0 = time.perf_counter()
        aligned_emb, src_proteins, _ = _procrustes_jaccard_align_one(
            seed, REFERENCE, ref_emb, ref_proteins, ORTHOLOGS_JACCARD_DIR, use_gpu=use_gpu
        )
        _save_aligned(seed, aligned_emb, src_proteins, output_subdir)
        aligned_seeds[seed] = (aligned_emb, src_proteins)
        species_times[seed] = time.perf_counter() - t0

    stage1_time = time.perf_counter() - t_total

    t_stage2 = time.perf_counter()
    for i, ns in enumerate(sorted(groups.keys())):
        nearest_seed = groups[ns]
        seed_emb, seed_prots = aligned_seeds[nearest_seed]
        t0 = time.perf_counter()
        try:
            aligned_emb, ns_proteins, _ = _procrustes_jaccard_align_one(
                ns, nearest_seed, seed_emb, seed_prots, ORTHOLOGS_JACCARD_DIR, use_gpu=use_gpu
            )
            _save_aligned(ns, aligned_emb, ns_proteins, output_subdir)
            species_times[ns] = time.perf_counter() - t0
        except Exception as e:
            logger.error(f"  {ns}: FAILED -- {e}")
            species_times[ns] = None
        if (i + 1) % 50 == 0:
            logger.info(f"  [{i+1}/{len(groups)}] non-seeds done")

    stage2_time = time.perf_counter() - t_stage2
    total_time = time.perf_counter() - t_total

    return _make_result("procrustes_jaccard_iter_csls", total_time, stage1_time,
                        stage2_time, species_times)


# ---------------------------------------------------------------------------
# Method 3: SPACE autoencoder
# ---------------------------------------------------------------------------

def benchmark_space(seed_info: dict, output_subdir: Path, device: str = "cuda") -> dict:
    import orbit._compat  # noqa: F401

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_alignment import _patch_space_for_string_ids, SPACE_CONFIG_DIR
    _patch_space_for_string_ids()
    from space.models.fedcoder import FedCoder, FedCoderNonSeed

    seeds = seed_info["seeds"]
    groups = seed_info["groups"]

    species_times = {}
    aligned_dir = str(output_subdir)

    # Stage 1: joint seed training
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    coder = FedCoder(
        seed_species=str(SPACE_CONFIG_DIR / "seeds.txt"),
        node2vec_dir=str(NODE2VEC_DIR),
        ortholog_dir=str(ORTHOLOGS_DIR / "seeds"),
        aligned_embedding_save_dir=aligned_dir,
        input_dim=128, latent_dim=512, alpha=0.5, gamma=0.1,
        lr=0.01, epochs=500, device=device, patience=5, delta=0.01,
    )
    coder.fit()
    coder.save_embeddings()
    stage1_time = time.perf_counter() - t0

    # SPACE trains all seeds jointly, so split evenly
    for seed in seeds:
        species_times[seed] = stage1_time / len(seeds)

    # Stage 2: per-species autoencoder
    t_stage2 = time.perf_counter()
    config = {
        "seed_groups_file": str(SPACE_CONFIG_DIR / "seed_groups.json"),
        "tax_group_file": str(SPACE_CONFIG_DIR / "tax_group.tsv"),
        "node2vec_dir": str(NODE2VEC_DIR),
        "aligned_dir": aligned_dir,
        "ortholog_dir": str(ORTHOLOGS_DIR / "non_seeds"),
    }

    for i, ns in enumerate(sorted(groups.keys())):
        h5_path = Path(config["node2vec_dir"]) / f"{ns}.h5"
        if not h5_path.exists():
            species_times[ns] = None
            continue

        t0 = time.perf_counter()
        try:
            coder_ns = FedCoderNonSeed(
                seed_groups=config["seed_groups_file"],
                tax_group=config["tax_group_file"],
                non_seed_species=ns,
                node2vec_dir=config["node2vec_dir"],
                aligned_dir=config["aligned_dir"],
                ortholog_dir=config["ortholog_dir"],
                aligned_embedding_save_dir=aligned_dir,
                input_dim=128, latent_dim=512, alpha=0.5, gamma=0.1,
                lr=0.01, epochs=500, device=device, patience=5, delta=0.01,
            )
            coder_ns.fit()
            coder_ns.save_embeddings()
            species_times[ns] = time.perf_counter() - t0
        except Exception as e:
            logger.error(f"  {ns}: FAILED -- {e}")
            species_times[ns] = None

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_stage2
            logger.info(f"  [{i+1}/{len(groups)}] non-seeds done ({elapsed:.0f}s elapsed)")

    stage2_time = time.perf_counter() - t_stage2
    total_time = time.perf_counter() - t_total

    return _make_result("space_autoencoder", total_time, stage1_time, stage2_time, species_times)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_benchmark(results_path: Path) -> None:
    """Generate runtime comparison figure from benchmark results."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]

    # Method display names and colors
    method_config = {
        "space": ("SPACE\nautoencoder", "#1f77b4"),
        "procrustes": ("Procrustes", "#2e7d31"),
        "procrustes_jaccard": ("Procrustes +\nJaccard + CSLS", "#e7672a"),
    }

    methods = [m for m in ["space", "procrustes", "procrustes_jaccard"] if m in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})

    # --- Panel A: Total wall-clock time (log scale) ---
    names = []
    totals = []
    colors = []
    for m in methods:
        label, color = method_config[m]
        names.append(label)
        totals.append(results[m]["total_seconds"])
        colors.append(color)

    bars = ax1.bar(names, totals, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Total time (seconds, log scale)", fontsize=11)
    ax1.set_title("A) Total alignment time (153 species)", fontsize=12, fontweight="bold", loc="left")

    # Annotate bars with human-readable times
    for bar, total in zip(bars, totals):
        if total >= 3600:
            label = f"{total/3600:.1f} h"
        elif total >= 60:
            label = f"{total/60:.1f} min"
        else:
            label = f"{total:.0f} s"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                 label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/3600:.0f}h" if x >= 3600 else (f"{x/60:.0f}m" if x >= 60 else f"{x:.0f}s")
    ))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", labelsize=10)

    # --- Panel B: Per-species time distribution (box plot) ---
    box_data = []
    box_labels = []
    box_colors = []
    for m in methods:
        label, color = method_config[m]
        times = [t for t in results[m]["per_species_times"].values() if t is not None]
        box_data.append(times)
        box_labels.append(label)
        box_colors.append(color)

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5,
                     medianprops={"color": "black", "linewidth": 1.5},
                     flierprops={"markersize": 3, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_yscale("log")
    ax2.set_ylabel("Time per species (seconds, log scale)", fontsize=11)
    ax2.set_title("B) Per-species alignment time", fontsize=12, fontweight="bold", loc="left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", labelsize=10)

    # Add median annotation
    for i, (times, label) in enumerate(zip(box_data, box_labels)):
        median = np.median(times)
        if median >= 60:
            txt = f"{median:.0f}s"
        else:
            txt = f"{median:.1f}s"
        ax2.text(i + 1, median * 0.4, f"median: {txt}", ha="center", fontsize=9,
                 fontstyle="italic", color="gray")

    # Hardware info footnote
    hw = data.get("hardware", {})
    hw_text = (
        f"Hardware: {hw.get('cpu_model', '?')} ({hw.get('cpu_cores', '?')} cores), "
        f"{hw.get('ram_gb', '?')} GB RAM"
    )
    if hw.get("gpu_count", 0) > 0:
        gpu_name = hw["gpus"][0].split(",")[0] if hw["gpus"] else "?"
        hw_text += f", {hw['gpu_count']}× {gpu_name}"
    fig.text(0.5, -0.02, hw_text, ha="center", fontsize=8, color="gray")

    plt.tight_layout()

    out_png = ROOT / "results" / "benchmark_runtime.png"
    out_pdf = ROOT / "results" / "benchmark_runtime.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    logger.info(f"Plot saved -> {out_png}")
    logger.info(f"Plot saved -> {out_pdf}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime benchmark for alignment methods")
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["procrustes", "procrustes_jaccard", "space"],
        default=["procrustes", "procrustes_jaccard", "space"],
        help="Which methods to benchmark (default: all three)",
    )
    parser.add_argument("--device", default="cuda", help="Device for SPACE/FAISS (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Just check data availability")
    parser.add_argument("--plot", action="store_true", help="Plot from existing results (skip benchmark)")
    args = parser.parse_args()

    if args.plot:
        if not RESULTS_PATH.exists():
            logger.error(f"No results found at {RESULTS_PATH}")
            return 1
        plot_benchmark(RESULTS_PATH)
        return 0

    seed_info = _load_seed_info()
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]
    n_total = len(seeds) + len(groups)
    logger.info(f"Benchmark: {len(seeds)} seeds + {len(groups)} non-seeds = {n_total} species")
    logger.info(f"Methods: {args.method}")

    hw_info = _collect_hardware_info()
    logger.info(f"Hardware: {hw_info.get('cpu_model', '?')} ({hw_info.get('cpu_cores', '?')} cores), "
                f"{hw_info.get('ram_gb', '?')} GB RAM, {hw_info.get('gpu_count', 0)} GPUs")

    if args.dry_run:
        logger.info("Dry run — checking data availability:")
        for sp in seeds:
            p = NODE2VEC_DIR / f"{sp}.h5"
            logger.info(f"  {sp} N2V: {'OK' if p.exists() else 'MISSING'}")
        logger.info(f"  Jaccard orthologs dir: {'OK' if ORTHOLOGS_JACCARD_DIR.exists() else 'MISSING'}")
        n_n2v = sum(1 for sp in list(seeds) + list(groups.keys())
                     if (NODE2VEC_DIR / f"{sp}.h5").exists())
        logger.info(f"  Node2Vec embeddings: {n_n2v}/{n_total}")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for method in args.method:
        subdir = OUTPUT_DIR / method
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {method}")
        logger.info(f"{'='*60}")

        if method == "procrustes":
            result = benchmark_procrustes(seed_info, subdir)
        elif method == "procrustes_jaccard":
            result = benchmark_procrustes_jaccard(seed_info, subdir, use_gpu=(args.device != "cpu"))
        elif method == "space":
            result = benchmark_space(seed_info, subdir, device=args.device)

        all_results[method] = result
        logger.info(f"\n  {method}: {result['total_seconds']:.1f}s total "
                     f"({result['per_species_mean']:.2f}s ± {result['per_species_std']:.2f}s per species)")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for method, result in all_results.items():
        total = result["total_seconds"]
        if total >= 3600:
            total_str = f"{total/3600:.1f}h"
        elif total >= 60:
            total_str = f"{total/60:.1f}min"
        else:
            total_str = f"{total:.0f}s"
        logger.info(f"  {method:30s}: {total_str:>8s} "
                     f"(stage1: {result['stage1_seconds']:.1f}s, "
                     f"stage2: {result['stage2_seconds']:.1f}s)")

    # Save
    output = {"hardware": hw_info, "results": all_results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved -> {RESULTS_PATH}")

    # Auto-plot
    plot_benchmark(RESULTS_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
