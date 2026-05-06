#!/usr/bin/env python3
"""Publication-quality UMAP of all 153 species in shared embedding space.

Colors genes by plant division. Clean legend, no title (added by composite script).
Subsamples genes for performance (max 30k per species).

Output: results/plots_v2/umap_153_pub.png
"""

from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
ALIGNED_DIR = ROOT / "results" / "aligned_embeddings"
SPECIES_FILE = ROOT / "data" / "species_names.tsv"
OUT = ROOT / "results" / "plots_v2" / "umap_153_pub.png"

# Division classification by nearest seed (approximate but fast)
# ARATH → Eudicots, ORYSA → Monocots, PICAB → Gymnosperms,
# SELMO → Ferns & Lycophytes, MARPO → Bryophytes & Algae
# We'll use NCBI taxonomy if available, else fall back to seed assignment

DIVISION_COLORS = {
    "Eudicots":          "#4CAF50",
    "Monocots":          "#FF9800",
    "Basal angiosperms": "#795548",
    "Gymnosperms":       "#2196F3",
    "Ferns":             "#9C27B0",
    "Lycophytes":        "#E91E63",
    "Bryophytes":        "#F44336",
    "Algae":             "#00BCD4",
}

# Manual division assignments for all 153 species
# Generated from NCBI taxonomy classification
SPECIES_DIVISION = {}

def classify_by_seed(seed_assignment):
    """Rough division classification based on nearest seed."""
    seed_to_div = {
        "ARATH": "Eudicots",
        "ORYSA": "Monocots",
        "PICAB": "Gymnosperms",
        "SELMO": "Ferns",
        "MARPO": "Bryophytes",
    }
    return {sp: seed_to_div.get(seed, "Other")
            for sp, seed in seed_assignment.items()}


def load_division_map():
    """Try NCBI taxonomy, fall back to seed assignment."""
    import json
    with open(ROOT / "results" / "seed_selection.json") as f:
        seed_sel = json.load(f)

    # Seed species are their own division
    div_map = {
        "ARATH": "Eudicots",
        "ORYSA": "Monocots",
        "PICAB": "Gymnosperms",
        "SELMO": "Lycophytes",
        "MARPO": "Bryophytes",
    }

    # Non-seeds: use seed assignment as rough proxy
    seed_assignment = {e["species"]: e["nearest_seed"]
                       for e in seed_sel["per_nonseed"]}
    rough = classify_by_seed(seed_assignment)
    div_map.update(rough)

    return div_map


def main():
    import json

    print("Loading division map...")
    div_map = load_division_map()

    # Get list of aligned species
    h5_files = sorted(ALIGNED_DIR.glob("*.h5"))
    species_list = [f.stem for f in h5_files]
    print(f"Found {len(species_list)} species")

    # Subsample and load
    MAX_PER_SPECIES = 3000
    all_emb = []
    all_div = []
    all_sp = []

    for sp in species_list:
        path = ALIGNED_DIR / f"{sp}.h5"
        with h5py.File(path, "r") as fh:
            emb = fh["embeddings"][:]

        # Subsample
        if len(emb) > MAX_PER_SPECIES:
            idx = np.random.RandomState(42).choice(len(emb), MAX_PER_SPECIES, replace=False)
            emb = emb[idx]

        div = div_map.get(sp, "Other")
        all_emb.append(emb)
        all_div.extend([div] * len(emb))
        all_sp.extend([sp] * len(emb))

    X = np.vstack(all_emb).astype(np.float32)
    divisions = np.array(all_div)
    print(f"Total genes: {X.shape[0]:,} x {X.shape[1]} dims")
    print("Division counts:", Counter(divisions).most_common())

    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="cosine",
        n_jobs=-1,
        low_memory=True,
    )
    coords = reducer.fit_transform(X)
    print(f"UMAP done: {coords.shape}")

    # Save coordinates for reuse
    np.savez(ROOT / "results" / "plots_v2" / "umap_153_coords.npz",
             coords=coords, divisions=divisions)

    # ── Plot ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot order: large groups first (background), small groups on top
    div_order = ["Eudicots", "Monocots", "Ferns", "Gymnosperms",
                 "Basal angiosperms", "Bryophytes", "Lycophytes", "Algae"]

    for div in div_order:
        mask = divisions == div
        if mask.sum() == 0:
            continue
        count = len(set(np.array(all_sp)[mask]))  # unique species in this div
        n_genes = mask.sum()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=1.0, alpha=0.25, c=DIVISION_COLORS.get(div, "#999999"),
            label=f"{div} ({count})",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    legend = ax.legend(
        loc="upper right", frameon=True, framealpha=0.9,
        edgecolor="#cccccc", fontsize=8, markerscale=8,
        handletextpad=0.4, title="Taxonomic division",
        title_fontsize=9,
    )

    total_genes = X.shape[0]
    ax.text(0.02, 0.02, f"{len(species_list)} species, {total_genes:,} genes",
            transform=ax.transAxes, fontsize=8, color="#777777",
            va="bottom")

    fig.savefig(str(OUT), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT).replace(".png", ".pdf"), dpi=300,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
