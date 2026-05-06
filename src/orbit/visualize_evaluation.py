"""Visualizations for evaluation results.

Produces:
- Per-pair Spearman bar chart (sorted, colored by seed vs nonseed pair type)
- Spearman distribution histogram
- Within-species P@50 bar chart for all species
- Aggregate summary (cross-species + within-species metrics)

Supports single-method runs (e.g. vanilla only) as well as multi-method
comparisons (vanilla vs jaccard).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _short_name(code: str) -> str:
    """Abbreviate species code, using 5 chars if needed to disambiguate."""
    if code.startswith("CYC"):
        return code[:5]
    return code[:3]


def _pair_label(r: dict) -> str:
    """Short label like 'ALS-ARA' from species codes."""
    return f"{_short_name(r['species_a'])}-{_short_name(r['species_b'])}"


# ---------------------------------------------------------------------------
# 1. Cross-species: per-pair Spearman bar chart (sorted)
# ---------------------------------------------------------------------------


def plot_spearman_bars(
    results: list[dict],
    output_path: str | Path | None = None,
    title: str = "Per-Pair Spearman Correlation (sorted)",
) -> plt.Figure:
    """Sorted bar chart of Spearman rho per pair, colored by pair type."""
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["spearman_rho"], reverse=True)

    labels = [_pair_label(r) for r in valid]
    spearman = [r["spearman_rho"] for r in valid]
    pair_types = [r.get("pair_type", "seed-seed") for r in valid]
    colors = ["coral" if pt == "seed-seed" else "steelblue" for pt in pair_types]

    n = len(labels)
    fig_width = max(16, n * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(range(n), spearman, color=colors, edgecolor="none", width=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Shuffle baseline
    shuffle_vals = [r.get("shuffle_spearman_rho", 0) for r in valid if "shuffle_spearman_rho" in r]
    if shuffle_vals:
        ax.axhline(np.mean(shuffle_vals), color="gray", linestyle="--",
                    linewidth=1, label=f"Shuffle baseline ({np.mean(shuffle_vals):.4f})")

    # Average line
    avg = np.mean(spearman)
    ax.axhline(avg, color="darkred", linestyle="-", linewidth=1.2, alpha=0.7,
               label=f"Mean ({avg:.4f})")

    # Show every Nth label to avoid clutter
    step = max(1, n // 40)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=90,
                       ha="center", fontsize=6)

    # Legend for pair types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="coral", label="Seed–seed"),
        Patch(facecolor="steelblue", label="Seed–nonseed"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0],
              loc="upper right", fontsize=8)

    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"{title}  (n={n})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 2. Spearman distribution histogram
# ---------------------------------------------------------------------------


def plot_spearman_distribution(
    results: list[dict],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of Spearman rho across all pairs, split by pair type."""
    valid = [r for r in results if "error" not in r]
    seed_seed = [r["spearman_rho"] for r in valid if r.get("pair_type") == "seed-seed"]
    seed_nonseed = [r["spearman_rho"] for r in valid if r.get("pair_type") == "seed-nonseed"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(-0.15, 0.35, 30)
    if seed_seed:
        ax.hist(seed_seed, bins=bins, color="coral", alpha=0.7,
                label=f"Seed–seed (n={len(seed_seed)}, mean={np.mean(seed_seed):.3f})")
    if seed_nonseed:
        ax.hist(seed_nonseed, bins=bins, color="steelblue", alpha=0.7,
                label=f"Seed–nonseed (n={len(seed_nonseed)}, mean={np.mean(seed_nonseed):.3f})")

    avg = np.mean([r["spearman_rho"] for r in valid])
    ax.axvline(avg, color="darkred", linestyle="-", linewidth=1.5,
               label=f"Overall mean ({avg:.4f})")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Cross-Species Spearman ρ  (n={len(valid)})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 3. Within-species P@50 bar chart
# ---------------------------------------------------------------------------


def plot_within_species(
    within_results: list[dict],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Sorted bar chart of within-species precision@50 for all species."""
    valid = [r for r in within_results if "error" not in r]
    valid.sort(key=lambda r: r["precision_at_k"], reverse=True)

    species = [r["species"] for r in valid]
    prec = [r["precision_at_k"] for r in valid]
    shuf = [r.get("shuffle_precision_mean", 0) for r in valid]

    n = len(species)
    fig_width = max(16, n * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(range(n), prec, color="steelblue", edgecolor="none", width=0.8, label="P@50")

    # Shuffle baseline
    if shuf:
        avg_shuf = np.mean(shuf)
        ax.axhline(avg_shuf, color="gray", linestyle="--", linewidth=1,
                    label=f"Shuffle baseline ({avg_shuf:.4f})")

    avg_prec = np.mean(prec)
    ax.axhline(avg_prec, color="darkred", linestyle="-", linewidth=1.2, alpha=0.7,
               label=f"Mean P@50 ({avg_prec:.4f})")

    step = max(1, n // 40)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([species[i] for i in range(0, n, step)], rotation=90,
                       ha="center", fontsize=6)

    ax.set_ylabel("Precision@50")
    ax.set_title(f"Within-Species Embedding Quality (P@50)  (n={n})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.0, max(prec) * 1.15))
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 4. Aggregate summary
# ---------------------------------------------------------------------------


def plot_aggregate_summary(
    results: list[dict],
    within_results: list[dict] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Summary panel: cross-species metrics + within-species metrics."""
    valid = [r for r in results if "error" not in r]
    n_panels = 3 if within_results else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Panel 1: Spearman box plot by pair type
    ax = axes[0]
    seed_sp = [r["spearman_rho"] for r in valid if r.get("pair_type") == "seed-seed"]
    nonseed_sp = [r["spearman_rho"] for r in valid if r.get("pair_type") == "seed-nonseed"]
    data = []
    labels = []
    colors = []
    if seed_sp:
        data.append(seed_sp)
        labels.append(f"Seed–seed\n(n={len(seed_sp)})")
        colors.append("coral")
    if nonseed_sp:
        data.append(nonseed_sp)
        labels.append(f"Seed–nonseed\n(n={len(nonseed_sp)})")
        colors.append("steelblue")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Cross-Species Spearman")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Hits@50 box plot
    ax = axes[1]
    seed_hits = [r["hits_at_k"] for r in valid if r.get("pair_type") == "seed-seed"]
    nonseed_hits = [r["hits_at_k"] for r in valid if r.get("pair_type") == "seed-nonseed"]
    data2 = []
    labels2 = []
    if seed_hits:
        data2.append(seed_hits)
        labels2.append(f"Seed–seed\n(n={len(seed_hits)})")
    if nonseed_hits:
        data2.append(nonseed_hits)
        labels2.append(f"Seed–nonseed\n(n={len(nonseed_hits)})")

    bp2 = ax.boxplot(data2, labels=labels2, patch_artist=True, widths=0.5)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Hits@50")
    ax.set_title("Cross-Species Hits@50")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Within-species (if available)
    if within_results:
        ax = axes[2]
        valid_w = [r for r in within_results if "error" not in r]
        prec = [r["precision_at_k"] for r in valid_w]
        shuf = [r.get("shuffle_precision_mean", 0) for r in valid_w]

        bp3 = ax.boxplot([prec, shuf], labels=[f"Real\n(n={len(prec)})", "Shuffle"],
                         patch_artist=True, widths=0.5)
        bp3["boxes"][0].set_facecolor("steelblue")
        bp3["boxes"][0].set_alpha(0.7)
        bp3["boxes"][1].set_facecolor("lightgray")
        bp3["boxes"][1].set_alpha(0.7)
        ax.set_ylabel("Precision@50")
        ax.set_title("Within-Species P@50")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Evaluation Summary: 153 Species, Vanilla SPACE Alignment", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Legacy: multi-method comparison (kept for backward compatibility)
# ---------------------------------------------------------------------------


def plot_spearman_comparison(
    results_vanilla: list[dict],
    results_jaccard: list[dict],
    results_hybrid: list[dict] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: per-pair Spearman rho for vanilla vs jaccard."""
    labels = [_pair_label(r) for r in results_vanilla]
    n = len(labels)

    sp_vanilla = [r["spearman_rho"] for r in results_vanilla]
    sp_jaccard = [r["spearman_rho"] for r in results_jaccard]

    x = np.arange(n)
    n_methods = 3 if results_hybrid else 2
    width = 0.8 / n_methods
    fig_width = max(14, n * 0.55)
    fontsize = 7 if n > 20 else 9

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(x - width / 2, sp_vanilla, width, label="Vanilla", color="steelblue")
    ax.bar(x + width / 2, sp_jaccard, width, label="Jaccard", color="coral")
    if results_hybrid:
        sp_hybrid = [r["spearman_rho"] for r in results_hybrid]
        ax.bar(x + width * 1.5, sp_hybrid, width, label="Hybrid", color="mediumpurple")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=fontsize)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-Pair Spearman Correlation")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_pairwise_scatter(
    results_vanilla: list[dict],
    results_jaccard: list[dict],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter: vanilla vs Jaccard Spearman per pair, with y=x diagonal."""
    sp_vanilla = np.array([r["spearman_rho"] for r in results_vanilla])
    sp_jaccard = np.array([r["spearman_rho"] for r in results_jaccard])
    labels = [_pair_label(r) for r in results_vanilla]
    pair_types = [r.get("pair_type", "seed-seed") for r in results_vanilla]

    fig, ax = plt.subplots(figsize=(7, 7))

    for pt, color, marker, label in [
        ("seed-seed", "coral", "o", "Seed–seed"),
        ("seed-nonseed", "steelblue", "s", "Seed–nonseed"),
    ]:
        mask = [t == pt for t in pair_types]
        if any(mask):
            ax.scatter(sp_vanilla[mask], sp_jaccard[mask], s=50, color=color,
                       edgecolor="black", linewidth=0.5, zorder=5, marker=marker,
                       label=label)

    n = len(labels)
    if n <= 15:
        to_label = range(n)
    else:
        to_label = np.argsort(sp_jaccard)[-5:]
    for i in to_label:
        ax.annotate(labels[i], (sp_vanilla[i], sp_jaccard[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)

    all_vals = list(sp_vanilla) + list(sp_jaccard)
    lo = min(all_vals) - 0.02
    hi = max(all_vals) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)

    above = int((sp_jaccard > sp_vanilla).sum())
    ax.set_xlabel("Vanilla Spearman ρ")
    ax.set_ylabel("Jaccard Spearman ρ")
    ax.set_title(f"Per-Pair Improvement: Vanilla vs Jaccard ({above}/{n} above diagonal)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig
