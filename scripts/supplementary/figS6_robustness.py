#!/usr/bin/env python3
"""Figure S6: Robustness of Procrustes alignment to ortholog-anchor subsampling.

For each seed-seed pair (10 pairs over 5 seed species), we:
  1. Pre-compute the Jaccard ground truth (coexpression-neighborhood overlap)
     — independent of alignment, used as the held-out evaluation signal.
  2. Pre-load Node2Vec embeddings for all seeds (once).
  3. Sweep over subsample rates r in {1.0, 0.75, 0.5, 0.25, 0.1}.
     For each (rate, replicate) combo, randomly subsample the ortholog pairs,
     solve orthogonal Procrustes, transform the source embeddings, then
     compute the cross-species Spearman rho between Jaccard scores and cosine
     similarities of the aligned vectors.

Output:
    results/supplementary/figS6_data.csv
    results/supplementary/figS6_ortholog_robustness.png
    results/supplementary/figS6_ortholog_robustness.pdf
    results/supplementary/figS6_run.log

Run on the GCP VM (does NOT modify the original codebase).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr

# Project paths
ROOT = Path.home() / "PLANT-SPACE"
NODE2VEC_DIR = ROOT / "data" / "node2vec"
NETWORK_DIR = ROOT / "data" / "networks"
TRANSCRIPTS_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
ORTHOLOGS_SEEDS_DIR = ROOT / "data" / "orthologs" / "seeds"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"
OUT_DIR = ROOT / "results" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make src/ importable for _load_protein_to_og helper
sys.path.insert(0, str(ROOT / "src"))

# Experiment configuration
SUBSAMPLE_RATES = [1.0, 0.75, 0.5, 0.25, 0.1]
N_REPLICATES = 5
RNG_BASE_SEED = 20260425


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_embeddings(species: str) -> tuple[np.ndarray, list[str]]:
    path = NODE2VEC_DIR / f"{species}.h5"
    with h5py.File(path, "r") as fh:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in fh["proteins"][:]]
        embeddings = fh["embeddings"][:]
    return embeddings, proteins


def load_ortholog_pairs(sp_a: str, sp_b: str) -> tuple[np.ndarray, str, str]:
    """Returns (pairs, first_in_filename, second_in_filename)."""
    sorted_pair = sorted([sp_a, sp_b])
    fname = "_".join(sorted_pair) + ".tsv"
    path = ORTHOLOGS_SEEDS_DIR / fname
    pairs = np.loadtxt(path, usecols=(0, 1), dtype=np.int64)
    return pairs, sorted_pair[0], sorted_pair[1]


# ---------------------------------------------------------------------------
# Jaccard ground truth (network neighborhoods → orthogroup overlap)
# ---------------------------------------------------------------------------

def _load_network_neighborhoods(network_path: Path) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    with open(network_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
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
    out: dict[str, set[str]] = {}
    for gene, nbrs in neighbors.items():
        if gene not in protein_to_og:
            continue
        og_nbrs = {protein_to_og[n] for n in nbrs if n in protein_to_og}
        if og_nbrs:
            out[gene] = og_nbrs
    return out


def build_jaccard_ground_truth(
    species_a: str,
    species_b: str,
    proteins_a: list[str],
    proteins_b: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (u_idx, v_idx, jaccard_score) parallel arrays."""
    from orbit.data_prep import _load_protein_to_og

    idx_a = {p: i for i, p in enumerate(proteins_a)}
    idx_b = {p: i for i, p in enumerate(proteins_b)}

    og_a = _load_protein_to_og(TRANSCRIPTS_DIR, species_a, set(idx_a.keys()))
    og_b = _load_protein_to_og(TRANSCRIPTS_DIR, species_b, set(idx_b.keys()))

    net_a = _load_network_neighborhoods(NETWORK_DIR / f"{species_a}.tsv")
    net_b = _load_network_neighborhoods(NETWORK_DIR / f"{species_b}.tsv")

    og_nbrs_a = _convert_neighborhoods_to_ogs(net_a, og_a)
    og_nbrs_b = _convert_neighborhoods_to_ogs(net_b, og_b)

    og_genes_a: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_a.items():
        if pid in og_nbrs_a:
            og_genes_a[og].append(pid)
    og_genes_b: dict[str, list[str]] = defaultdict(list)
    for pid, og in og_b.items():
        if pid in og_nbrs_b:
            og_genes_b[og].append(pid)

    shared_ogs = set(og_genes_a.keys()) & set(og_genes_b.keys())
    u_list, v_list, s_list = [], [], []
    for og in shared_ogs:
        for ga in og_genes_a[og]:
            na = og_nbrs_a[ga]
            for gb in og_genes_b[og]:
                nb = og_nbrs_b[gb]
                inter = len(na & nb)
                union = len(na | nb)
                if union > 0:
                    jacc = inter / union
                    if jacc > 0:
                        u_list.append(idx_a[ga])
                        v_list.append(idx_b[gb])
                        s_list.append(jacc)
    return (
        np.asarray(u_list, dtype=np.int64),
        np.asarray(v_list, dtype=np.int64),
        np.asarray(s_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Procrustes + evaluation
# ---------------------------------------------------------------------------

def procrustes_align(
    pairs: np.ndarray,            # (n_pairs, 2)
    col_source: int,
    col_ref: int,
    emb_source: np.ndarray,       # (n_genes_source, d)
    emb_ref: np.ndarray,           # (n_genes_ref, d)
) -> np.ndarray:
    """Solve orthogonal Procrustes on the given pairs. Returns the rotated
    full source-embedding matrix."""
    src_idx = pairs[:, col_source]
    ref_idx = pairs[:, col_ref]

    # Filter out-of-range indices defensively
    valid = (src_idx < emb_source.shape[0]) & (ref_idx < emb_ref.shape[0])
    src_idx = src_idx[valid]
    ref_idx = ref_idx[valid]

    X_src = emb_source[src_idx]
    X_ref = emb_ref[ref_idx]
    R, _ = orthogonal_procrustes(X_src, X_ref)
    return emb_source @ R


def cross_species_spearman(
    XA: np.ndarray,
    XB: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    sample_n: int = 300_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Spearman rho between Jaccard score and cosine similarity in aligned space."""
    if rng is None:
        rng = np.random.default_rng(0)
    eps = 1e-12
    XA_l2 = XA / (np.linalg.norm(XA, axis=1, keepdims=True) + eps)
    XB_l2 = XB / (np.linalg.norm(XB, axis=1, keepdims=True) + eps)
    n = len(u)
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        u, v, s = u[idx], v[idx], s[idx]
    cos_sim = (XA_l2[u] * XB_l2[v]).sum(axis=1).astype(np.float32)
    rho, _ = spearmanr(s, cos_sim)
    return float(rho)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log_path = OUT_DIR / "figS6_run.log"
    # tee log to file in addition to stdout
    log_fh = open(log_path, "w", buffering=1)

    def tee(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    t_start = time.perf_counter()

    with open(SEED_RESULTS) as f:
        seeds = json.load(f)["seeds"]
    tee(f"Seeds: {seeds}")

    # 1. Pre-load embeddings for all seed species
    tee("Loading Node2Vec embeddings for seed species...")
    embs: dict[str, np.ndarray] = {}
    prots: dict[str, list[str]] = {}
    for sp in seeds:
        e, p = load_embeddings(sp)
        embs[sp] = e
        prots[sp] = p
        tee(f"  {sp}: {e.shape[0]} genes x {e.shape[1]} dim")

    # 2. Pre-compute Jaccard ground truth for all 10 seed pairs
    seed_pairs = list(combinations(sorted(seeds), 2))  # alphabetical
    tee(f"Pre-computing Jaccard ground truth for {len(seed_pairs)} seed pairs...")
    gt: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sp_a, sp_b in seed_pairs:
        t0 = time.perf_counter()
        u, v, s = build_jaccard_ground_truth(sp_a, sp_b, prots[sp_a], prots[sp_b])
        gt[(sp_a, sp_b)] = (u, v, s)
        tee(f"  {sp_a}-{sp_b}: {len(u)} Jaccard pairs ({time.perf_counter() - t0:.1f}s)")

    # 3. Pre-load ortholog pairs for all 10 seed pairs
    tee("Loading ortholog pair files...")
    ortho_cache: dict[tuple[str, str], tuple[np.ndarray, str, str]] = {}
    for sp_a, sp_b in seed_pairs:
        pairs, first, second = load_ortholog_pairs(sp_a, sp_b)
        ortho_cache[(sp_a, sp_b)] = (pairs, first, second)
        tee(f"  {sp_a}-{sp_b}: {len(pairs)} ortholog pairs (file order: {first}|{second})")

    # 4. Sweep
    tee("=" * 70)
    tee(f"Subsample rates: {SUBSAMPLE_RATES}")
    tee(f"Replicates per rate: {N_REPLICATES}")
    tee(f"Total Procrustes solves: "
        f"{len(seed_pairs)} pairs * {len(SUBSAMPLE_RATES)} rates * {N_REPLICATES} reps "
        f"= {len(seed_pairs) * len(SUBSAMPLE_RATES) * N_REPLICATES}")
    tee("=" * 70)

    rows: list[dict] = []
    spearman_rng = np.random.default_rng(99)  # for the sample_n subsample in spearman

    total_combos = len(seed_pairs) * len(SUBSAMPLE_RATES) * N_REPLICATES
    combo_i = 0

    for sp_a, sp_b in seed_pairs:
        pairs, first, second = ortho_cache[(sp_a, sp_b)]
        # We align sp_b -> sp_a (sp_a is the reference). Determine columns.
        # File columns are (first, second). first is alphabetically smaller.
        if sp_a == first:
            col_ref, col_source = 0, 1  # ref=sp_a is col 0
        else:
            col_ref, col_source = 1, 0
        # Note: in our seed_pairs loop (sp_a, sp_b) is alphabetical from combinations(sorted(seeds), 2)
        # so sp_a == first always. But keep defensive code.

        emb_ref = embs[sp_a]
        emb_source = embs[sp_b]
        u, v, s = gt[(sp_a, sp_b)]
        # Note: u indexes into sp_a (ref), v indexes into sp_b (source).

        n_pairs_total = len(pairs)
        for rate in SUBSAMPLE_RATES:
            n_keep = max(2, int(round(rate * n_pairs_total)))
            for rep in range(N_REPLICATES):
                combo_i += 1
                seed = RNG_BASE_SEED + 1000 * rep + hash((sp_a, sp_b, rate)) % 1000
                rng = np.random.default_rng(seed)
                if rate < 1.0:
                    sel = rng.choice(n_pairs_total, size=n_keep, replace=False)
                    sub_pairs = pairs[sel]
                else:
                    sub_pairs = pairs
                t0 = time.perf_counter()
                # Align source -> ref
                aligned_source = procrustes_align(
                    sub_pairs, col_source, col_ref, emb_source, emb_ref
                )
                rho = cross_species_spearman(
                    emb_ref, aligned_source, u, v, s, rng=spearman_rng
                )
                dt = time.perf_counter() - t0
                row = {
                    "species_a": sp_a,
                    "species_b": sp_b,
                    "subsample_rate": rate,
                    "replicate": rep,
                    "n_anchors_used": int(len(sub_pairs)),
                    "n_anchors_total": int(n_pairs_total),
                    "spearman_rho": rho,
                    "elapsed_s": dt,
                }
                rows.append(row)
                tee(f"  [{combo_i}/{total_combos}] {sp_a}-{sp_b} rate={rate:.2f} "
                    f"rep={rep} n={len(sub_pairs)} rho={rho:.4f} ({dt:.1f}s)")

    # 5. Save raw results
    csv_path = OUT_DIR / "figS6_data.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tee(f"Saved raw results -> {csv_path}")

    # 6. Plot
    tee("Generating plot...")
    plot_results(rows, OUT_DIR)

    total_min = (time.perf_counter() - t_start) / 60.0
    tee(f"DONE in {total_min:.1f} minutes")
    log_fh.close()
    return 0


def plot_results(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Group by (species_a, species_b)
    pair_data: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["species_a"], r["species_b"])
        pair_data[key][r["subsample_rate"]].append(r["spearman_rho"])
        overall[r["subsample_rate"]].append(r["spearman_rho"])

    PROC_ORANGE = "#e7672a"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Left panel: per-pair lines
    ax = axes[0]
    rates_sorted = sorted(SUBSAMPLE_RATES)
    cmap = plt.get_cmap("tab10")
    for i, (key, by_rate) in enumerate(sorted(pair_data.items())):
        means = [np.mean(by_rate[r]) for r in rates_sorted]
        stds = [np.std(by_rate[r]) for r in rates_sorted]
        label = f"{key[0]}-{key[1]}"
        ax.errorbar(
            [r * 100 for r in rates_sorted], means, yerr=stds,
            label=label, marker="o", markersize=4, linewidth=1.2,
            color=cmap(i % 10), capsize=2, alpha=0.85,
        )
    ax.set_xlabel("Ortholog anchor subsample rate (%)")
    ax.set_ylabel(r"Cross-species Spearman $\rho$")
    ax.set_title("Per seed pair (mean $\\pm$ SD over 5 replicates)")
    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 25, 50, 75, 100])

    # Right panel: aggregated mean +- SD across all pairs and replicates
    ax = axes[1]
    pct = [r * 100 for r in rates_sorted]
    means = [np.mean(overall[r]) for r in rates_sorted]
    stds = [np.std(overall[r]) for r in rates_sorted]
    ax.plot(pct, means, color=PROC_ORANGE, marker="o", linewidth=2.2, label="Procrustes")
    ax.fill_between(
        pct,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        color=PROC_ORANGE, alpha=0.2, label="$\\pm$ 1 SD",
    )
    ax.set_xlabel("Ortholog anchor subsample rate (%)")
    ax.set_ylabel(r"Cross-species Spearman $\rho$")
    ax.set_title("Aggregated across 10 seed pairs $\\times$ 5 replicates")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 25, 50, 75, 100])

    fig.tight_layout()
    png_path = out_dir / "figS6_ortholog_robustness.png"
    pdf_path = out_dir / "figS6_ortholog_robustness.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png_path}", flush=True)
    print(f"  -> {pdf_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
