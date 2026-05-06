"""Figure 3: All-species UMAP with k-means clusters + zoom-in on ARATH vs CUCME."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

OUT_DIR = Path("results/plots_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAXONOMIC_GROUPS = {
    "Eudicots": [
        "ARATH", "ARALY", "ARAHA", "ARAAL", "ALSLA", "BRAJU", "BRANA", "BRAOL", "BRARA", "CAPRU",
        "CARPA", "CHEQU", "BETVU", "GLYMA", "MEDTR", "PHAVU", "CAJCA", "CICAR", "VIGRA", "LOTJA",
        "TRIPR", "LUPAN", "POPTR", "RICCO", "MANES", "JATCU", "HEVBR", "LINBE", "EUCGR",
        "CITSI", "CITCL", "CITMA", "GOSBA", "GOSAR", "THECC", "CAMSA", "CAPAN", "SOLLY", "SOLTU",
        "NICTA", "NICBE", "NICSY", "PETHY", "COFCA", "FRAVE", "MALDO", "PRUPE", "PYRBR", "ROSCH",
        "CUCSA", "CUCME", "CITLA", "DAUCA", "ERYGU", "HELAN", "LACSA", "TABDI", "PHLPH",
        "VITIS", "NELNU", "AQUCO", "PAPSO", "KALFE", "SALMO",
        "DICCU", "DIOVI", "FRAEX", "EPIPI", "HACVI", "HYDVE",
        "ILEVE", "PANHA", "ABROB",
    ],
    "Monocots": [
        "ORYSA", "ZEAMA", "SORBI", "HORVU", "BRADI", "TRIAE", "AEGTA", "SETIT", "PANVI",
        "SACOF", "MUSAC", "PHODA", "ELAGU", "ASPOF", "ANACO", "ANGEV",
    ],
    "Gymnosperms": [
        "PICAB", "PISSA", "GINBI", "GNEGN", "CYCBI", "CYCED",
    ],
    "Ferns & allies": [
        "SELMO", "EQUHY", "CIBBA", "LYGFL", "CENAM", "DAVDE", "SALCU",
    ],
    "Bryophytes": [
        "MARPO", "PHYPA", "PHYBR", "PLEIR", "PLEVA", "SYNXX",
    ],
    "Basal angiosperms": [
        "AMBTR", "AMBOP", "NUPAD",
    ],
}

SPECIES_TO_GROUP = {}
for group, species_list in TAXONOMIC_GROUPS.items():
    for sp in species_list:
        SPECIES_TO_GROUP[sp] = group

GROUP_COLORS = {
    "Eudicots": "#2ca02c",
    "Monocots": "#ff7f0e",
    "Gymnosperms": "#1f77b4",
    "Ferns & allies": "#9467bd",
    "Bryophytes": "#d62728",
    "Basal angiosperms": "#8c564b",
    "Other": "#7f7f7f",
}


def main():
    # Load cached UMAP + species labels
    coords = np.load("results/umap_cache/umap_Jaccard_Procrustes_iter__CSLS.npy")
    species = np.load("results/umap_cache/species_labels.npy")
    print(f"Loaded {len(coords)} points")

    # Taxonomic group per gene
    groups = np.array([SPECIES_TO_GROUP.get(sp, "Other") for sp in species])

    # K-means clustering on UMAP coordinates
    print("Running k-means...")
    kmeans = MiniBatchKMeans(n_clusters=12, random_state=42, batch_size=50_000)
    cluster_labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    print("K-means done.")

    # Find cluster(s) where ARATH is most concentrated
    at_mask = species == "ARATH"
    cm_mask = species == "CUCME"
    at_clusters = cluster_labels[at_mask]
    at_cluster_counts = np.bincount(at_clusters, minlength=12)
    main_cluster = np.argmax(at_cluster_counts)
    print(f"ARATH main cluster: {main_cluster} ({at_cluster_counts[main_cluster]} / {at_mask.sum()} genes)")
    cm_in_cluster = np.sum((cluster_labels == main_cluster) & cm_mask)
    print(f"CUCME in that cluster: {cm_in_cluster}")

    # Zoom bounds: bounding box of the main cluster with some padding
    in_cluster = cluster_labels == main_cluster
    cluster_coords = coords[in_cluster]
    pad = 1.0
    zoom_xmin, zoom_xmax = cluster_coords[:, 0].min() - pad, cluster_coords[:, 0].max() + pad
    zoom_ymin, zoom_ymax = cluster_coords[:, 1].min() - pad, cluster_coords[:, 1].max() + pad

    # =========================================================
    # Panel A: All-species UMAP, colored by taxonomic group,
    #          ARATH highlighted with edge glow
    # =========================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Shuffle for fair overdraw
    rng = np.random.default_rng(42)
    order = rng.permutation(len(coords))

    # Draw all points by taxonomic group
    for group_name in ["Other", "Basal angiosperms", "Bryophytes", "Ferns & allies",
                       "Gymnosperms", "Monocots", "Eudicots"]:
        mask = groups[order] == group_name
        if not mask.any():
            continue
        ax1.scatter(
            coords[order[mask], 0], coords[order[mask], 1],
            c=GROUP_COLORS.get(group_name, "#7f7f7f"),
            s=0.3, alpha=0.08, rasterized=True,
        )

    # Highlight ARATH on top
    at_ordered = at_mask[order]
    ax1.scatter(
        coords[order[at_ordered], 0], coords[order[at_ordered], 1],
        c="#E8B931", s=2, alpha=0.6, rasterized=True, zorder=5,
    )

    # Draw cluster boundaries as convex hulls or just label centers
    for i, center in enumerate(centers):
        ax1.annotate(
            str(i), xy=center, fontsize=7, fontweight="bold",
            color="black", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )

    # Draw zoom rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (zoom_xmin, zoom_ymin), zoom_xmax - zoom_xmin, zoom_ymax - zoom_ymin,
        linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--", zorder=10,
    )
    ax1.add_patch(rect)

    ax1.set_xlabel("UMAP 1", fontsize=11)
    ax1.set_ylabel("UMAP 2", fontsize=11)
    ax1.set_facecolor("white")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("A", fontsize=14, fontweight="bold", loc="left")

    # Legend for Panel A
    tax_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=6, label=g)
        for g, c in GROUP_COLORS.items()
        if g != "Other"
    ]
    tax_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8B931",
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label=r"$\it{A.\ thaliana}$")
    )
    ax1.legend(handles=tax_handles, fontsize=7, loc="lower left",
               frameon=True, framealpha=0.9, edgecolor="none")

    # =========================================================
    # Panel B: Zoom into ARATH's main cluster
    #          Color ARATH gold, CUCME red, rest gray
    # =========================================================

    # Only plot points in the zoom region
    in_zoom = (
        (coords[:, 0] >= zoom_xmin) & (coords[:, 0] <= zoom_xmax) &
        (coords[:, 1] >= zoom_ymin) & (coords[:, 1] <= zoom_ymax)
    )

    # Background: all other species in gray
    other_zoom = in_zoom & ~at_mask & ~cm_mask
    ax2.scatter(
        coords[other_zoom, 0], coords[other_zoom, 1],
        c="#DDDDDD", s=1, alpha=0.3, rasterized=True,
    )

    # CUCME
    cm_zoom = in_zoom & cm_mask
    ax2.scatter(
        coords[cm_zoom, 0], coords[cm_zoom, 1],
        c="#DC143C", s=4, alpha=0.7, rasterized=True, zorder=4,
    )

    # ARATH
    at_zoom = in_zoom & at_mask
    ax2.scatter(
        coords[at_zoom, 0], coords[at_zoom, 1],
        c="#E8B931", s=4, alpha=0.7, rasterized=True, zorder=5,
    )

    ax2.set_xlim(zoom_xmin, zoom_xmax)
    ax2.set_ylim(zoom_ymin, zoom_ymax)
    ax2.set_xlabel("UMAP 1", fontsize=11)
    ax2.set_ylabel("UMAP 2", fontsize=11)
    ax2.set_facecolor("white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("B", fontsize=14, fontweight="bold", loc="left")

    # Legend for Panel B
    zoom_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8B931",
               markersize=8, label=r"$\it{A.\ thaliana}$ (model)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DC143C",
               markersize=8, label=r"$\it{C.\ melo}$ (non-model)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DDDDDD",
               markersize=8, label="Other species"),
    ]
    ax2.legend(handles=zoom_handles, fontsize=8, loc="lower left",
               frameon=True, framealpha=0.9, edgecolor="none")

    # Count stats for subtitle
    n_at_zoom = at_zoom.sum()
    n_cm_zoom = cm_zoom.sum()
    n_other_zoom = other_zoom.sum()
    ax2.text(
        0.98, 0.98,
        f"Cluster {main_cluster}: {n_at_zoom:,} AT / {n_cm_zoom:,} CM / {n_other_zoom:,} other",
        transform=ax2.transAxes, fontsize=7, ha="right", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7),
    )

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure3_panels_AB.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure3_panels_AB.png", dpi=300, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / 'figure3_panels_AB.png'}")
    plt.close()


if __name__ == "__main__":
    main()
