"""Visualizations for seed species selection.

Produces:
- Dendrogram with seeds highlighted
- Ortholog density heatmap
- k-sensitivity plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from orbit.seed_selection import (
    evaluate_seed_set,
    select_seeds,
)


def plot_dendrogram(
    dist_matrix: pd.DataFrame,
    seeds: list[str],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot hierarchical clustering dendrogram with seed species highlighted in red."""
    species = list(dist_matrix.index)
    condensed = squareform(dist_matrix.values, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(16, 8))
    dn = dendrogram(Z, labels=species, ax=ax, leaf_rotation=90, leaf_font_size=8)

    # Color seed labels red
    xlabels = ax.get_xticklabels()
    for label in xlabels:
        if label.get_text() in seeds:
            label.set_color("red")
            label.set_fontweight("bold")

    ax.set_title("Species Dendrogram (Jaccard Distance on Orthogroup Presence)")
    ax.set_ylabel("Jaccard Distance")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved dendrogram to {output_path}")

    return fig


def plot_ortholog_density_heatmap(
    density_matrix: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    seeds: list[str],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot heatmap of shared orthogroup counts, ordered by dendrogram."""
    species = list(dist_matrix.index)
    condensed = squareform(dist_matrix.values, checks=False)
    Z = linkage(condensed, method="average")
    dn = dendrogram(Z, labels=species, no_plot=True)
    order = dn["ivl"]

    reordered = density_matrix.loc[order, order]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(reordered.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Shared Orthogroups")

    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)

    # Highlight seed labels
    for label in ax.get_xticklabels():
        if label.get_text() in seeds:
            label.set_color("red")
            label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        if label.get_text() in seeds:
            label.set_color("red")
            label.set_fontweight("bold")

    ax.set_title("Pairwise Shared Orthogroup Counts (ordered by dendrogram)")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved heatmap to {output_path}")

    return fig


def plot_k_sensitivity(
    dist_matrix: pd.DataFrame,
    og_matrix: pd.DataFrame,
    density_matrix: pd.DataFrame,
    k_range: tuple[int, int] = (2, 15),
    min_shared_ogs: int = 1000,
    candidates: list[str] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot seed selection metrics across different k values.

    Two subplots:
    - Max non-seed-to-nearest-seed distance vs k (coverage)
    - Min shared OGs between any seed pair vs k (anchor quality)
    """
    ks = list(range(k_range[0], k_range[1] + 1))
    max_nonseed_dists = []
    min_seed_shared = []
    mean_nonseed_dists = []
    og_coverages = []

    for k in ks:
        try:
            seeds = select_seeds(
                dist_matrix, density_matrix, k=k,
                min_shared_ogs=min_shared_ogs, candidates=candidates,
            )
            metrics = evaluate_seed_set(seeds, dist_matrix, og_matrix, density_matrix)
            max_nonseed_dists.append(metrics["nonseed_max_dist"])
            mean_nonseed_dists.append(metrics["nonseed_mean_dist"])
            min_seed_shared.append(metrics["seed_min_shared_ogs"])
            og_coverages.append(metrics["og_coverage"])
        except ValueError as e:
            logger.warning(f"k={k}: {e}")
            max_nonseed_dists.append(np.nan)
            mean_nonseed_dists.append(np.nan)
            min_seed_shared.append(np.nan)
            og_coverages.append(np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Coverage: max distance
    axes[0, 0].plot(ks, max_nonseed_dists, "o-", color="steelblue")
    axes[0, 0].set_xlabel("k (number of seeds)")
    axes[0, 0].set_ylabel("Jaccard Distance")
    axes[0, 0].set_title("Max Non-Seed to Nearest Seed Distance")
    axes[0, 0].grid(True, alpha=0.3)

    # Coverage: mean distance
    axes[0, 1].plot(ks, mean_nonseed_dists, "s-", color="teal")
    axes[0, 1].set_xlabel("k (number of seeds)")
    axes[0, 1].set_ylabel("Jaccard Distance")
    axes[0, 1].set_title("Mean Non-Seed to Nearest Seed Distance")
    axes[0, 1].grid(True, alpha=0.3)

    # Anchor quality: min shared OGs
    axes[1, 0].plot(ks, min_seed_shared, "^-", color="coral")
    axes[1, 0].set_xlabel("k (number of seeds)")
    axes[1, 0].set_ylabel("Shared Orthogroups")
    axes[1, 0].set_title("Min Shared OGs Between Any Seed Pair")
    axes[1, 0].grid(True, alpha=0.3)

    # OG coverage
    axes[1, 1].plot(ks, og_coverages, "D-", color="mediumpurple")
    axes[1, 1].set_xlabel("k (number of seeds)")
    axes[1, 1].set_ylabel("Fraction of OGs Covered")
    axes[1, 1].set_title("Orthogroup Coverage by Seeds")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Seed Selection: k-Sensitivity Analysis", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved k-sensitivity plot to {output_path}")

    return fig
