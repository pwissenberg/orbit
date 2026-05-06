#!/usr/bin/env python3
"""Figure 2D: Conserved coexpression neighborhoods between Arabidopsis and Rice.

Shows the ego-network of a LHCB (light-harvesting complex) gene in each species,
with shared orthogroup members colored and connected by dashed lines.
Demonstrates that orthologous genes share coexpression partners across species —
the structural basis for Procrustes alignment.

Usage:
    python scripts/plot_panel_d.py
    python scripts/plot_panel_d.py --orthogroup OG0000484 --sp-left ARATH --sp-right ORYSA
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parent.parent
OG_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
NET_DIR = ROOT / "data" / "networks"
OUT_DIR = ROOT / "report_figures"
OUT_DIR.mkdir(exist_ok=True)

SPECIES_COMMON = {
    "CITLA": "Watermelon", "CUCSA": "Cucumber", "CUCME": "Melon",
    "MALDO": "Apple", "PRUPE": "Peach", "ROSCH": "Rose",
    "ARATH": "A. thaliana", "ORYSA": "O. sativa",
    "ZEAMA": "Maize", "GLYMA": "Soybean", "MEDTR": "Medicago",
}

# Colorblind-friendly palette for shared orthogroups
OG_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#377eb8", "#e41a1c", "#984ea3",
]

SP_COLORS = {"ARATH": "#2e7d32", "ORYSA": "#e65100"}


# ---- Data loading ----

def load_orthogroup_genes(orthogroup):
    og_genes = defaultdict(list)
    for tsv in sorted(OG_DIR.glob("*_transcripts_to_OG.tsv")):
        sp = tsv.stem.replace("_transcripts_to_OG", "")
        with open(tsv) as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] == orthogroup:
                    og_genes[sp].append(parts[0])
    return dict(og_genes)


def load_gene_to_orthogroup(species_list):
    g2og = {}
    for sp in species_list:
        tsv = OG_DIR / ("%s_transcripts_to_OG.tsv" % sp)
        if not tsv.exists():
            continue
        with open(tsv) as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    g2og[(sp, parts[0])] = parts[2]
    return g2og


def og_for_node(sp, node, g2og):
    og = g2og.get((sp, node))
    if not og:
        stripped = node.split("|")[0]
        if stripped.endswith(".p"):
            stripped = stripped[:-2]
        og = g2og.get((sp, stripped))
    if not og:
        base = re.sub(r'\.\d+$', '', node.split("|")[0])
        og = g2og.get((sp, base))
    return og


def _resolve_gene(species, gene):
    net_path = NET_DIR / ("%s.tsv" % species)
    if not net_path.exists():
        return gene
    with open(net_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            for g in (parts[0], parts[1]):
                if g == gene or g.startswith(gene):
                    return g
    return gene


def load_ego_subnetwork(species, gene, top_k=10):
    net_path = NET_DIR / ("%s.tsv" % species)
    if not net_path.exists():
        return nx.Graph(), gene
    resolved = _resolve_gene(species, gene)
    neighbors = []
    with open(net_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            g1, g2, w = parts[0], parts[1], float(parts[2])
            if g1 == resolved:
                neighbors.append((g2, w))
            elif g2 == resolved:
                neighbors.append((g1, w))
    neighbors.sort(key=lambda x: -x[1])
    top_set = set(n for n, _ in neighbors[:top_k]) | {resolved}
    G = nx.Graph()
    G.add_nodes_from(top_set)
    with open(net_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            g1, g2, w = parts[0], parts[1], float(parts[2])
            if g1 in top_set and g2 in top_set:
                G.add_edge(g1, g2, weight=w)
    return G, resolved


def shorten(gene_id):
    g = gene_id.split("|")[0]
    if g.endswith(".p"):
        g = g[:-2]
    # For ARATH: AT1G29910.1 -> AT1G29910
    g = re.sub(r'\.\d+$', '', g)
    return g


# ---- Panel D plotting ----

def plot_panel_d(ax, sp_left, sp_right, og_genes, gene_to_og, top_k=10):
    """Compact bipartite coexpression network for a figure panel."""

    col_L = SP_COLORS.get(sp_left, "#2e7d32")
    col_R = SP_COLORS.get(sp_right, "#e65100")

    # Load ego-networks
    G_L, ctr_L = load_ego_subnetwork(sp_left, og_genes[sp_left][0], top_k=top_k)
    G_R, ctr_R = load_ego_subnetwork(sp_right, og_genes[sp_right][0], top_k=top_k)

    # Map nodes to orthogroups
    og_L = {n: og_for_node(sp_left, n, gene_to_og) for n in G_L.nodes}
    og_R = {n: og_for_node(sp_right, n, gene_to_og) for n in G_R.nodes}

    # Shared orthogroups (exclude center gene's own OG for clarity)
    ogs_left = set(og for n, og in og_L.items() if og and n != ctr_L)
    ogs_right = set(og for n, og in og_R.items() if og and n != ctr_R)
    shared = sorted(ogs_left & ogs_right)
    og_colors = {og: OG_PALETTE[i % len(OG_PALETTE)] for i, og in enumerate(shared)}

    print("  Shared OGs: %d — %s" % (len(shared), ", ".join(shared[:8])))

    ax.axis("off")

    # --- Force-directed layout: reveals true network topology ---
    def force_layout(G, center, cx, cy, radius=1.7, seed=42):
        """Spring layout using ALL edge weights for positioning.
        Strongly connected nodes cluster together, showing real structure."""
        pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.2,
                               iterations=200, center=(0, 0))
        # Recenter on the focal gene
        c = pos[center]
        pos = {n: (x - c[0], y - c[1]) for n, (x, y) in pos.items()}
        # Normalize to fit within radius
        max_r = max(np.sqrt(x**2 + y**2) for x, y in pos.values()) or 1.0
        return {n: (cx + x / max_r * radius, cy + y / max_r * radius)
                for n, (x, y) in pos.items()}

    pos_L = force_layout(G_L, ctr_L, cx=2.5, cy=0, radius=1.7, seed=42)
    pos_R = force_layout(G_R, ctr_R, cx=7.5, cy=0, radius=1.7, seed=43)

    # Compute bounds for background boxes
    all_x_L = [p[0] for p in pos_L.values()]
    all_y_L = [p[1] for p in pos_L.values()]
    all_x_R = [p[0] for p in pos_R.values()]
    all_y_R = [p[1] for p in pos_R.values()]
    pad = 0.6
    y_min = min(min(all_y_L), min(all_y_R)) - pad
    y_max = max(max(all_y_L), max(all_y_R)) + pad

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(y_min - 1.5, y_max + 1.2)

    # --- Background panels ---
    bg_L = plt.Rectangle((min(all_x_L) - pad, y_min),
                          max(all_x_L) - min(all_x_L) + 2 * pad, y_max - y_min + pad,
                          fc=to_rgba(col_L, 0.04), ec=to_rgba(col_L, 0.2),
                          lw=1.2, zorder=0, clip_on=False)
    bg_R = plt.Rectangle((min(all_x_R) - pad, y_min),
                          max(all_x_R) - min(all_x_R) + 2 * pad, y_max - y_min + pad,
                          fc=to_rgba(col_R, 0.04), ec=to_rgba(col_R, 0.2),
                          lw=1.2, zorder=0, clip_on=False)
    ax.add_patch(bg_L)
    ax.add_patch(bg_R)

    # Species labels
    name_L = SPECIES_COMMON.get(sp_left, sp_left)
    name_R = SPECIES_COMMON.get(sp_right, sp_right)
    ax.text(2.5, y_max + 0.9, name_L, fontsize=10, fontweight="bold",
            fontstyle="italic", color=col_L, ha="center", va="bottom")
    ax.text(7.5, y_max + 0.9, name_R, fontsize=10, fontweight="bold",
            fontstyle="italic", color=col_R, ha="center", va="bottom")

    # --- Coexpression edges: only center-to-neighbor (clean) ---
    def draw_edges(G, pos, center, sp_col):
        # Collect center-to-neighbor weights for normalization
        ctr_weights = []
        for n in G.nodes:
            if n != center and G.has_edge(center, n):
                ctr_weights.append(G[center][n].get("weight", 1.0))
        if not ctr_weights:
            return
        w_min, w_max = min(ctr_weights), max(ctr_weights)
        w_range = w_max - w_min if w_max > w_min else 1.0

        for n in G.nodes:
            if n == center or not G.has_edge(center, n):
                continue
            w = G[center][n].get("weight", 1.0)
            w_norm = (w - w_min) / w_range
            x0, y0 = pos[center]
            x1, y1 = pos[n]
            ax.plot([x0, x1], [y0, y1], color=sp_col,
                    lw=0.6 + 1.2 * w_norm, alpha=0.2 + 0.3 * w_norm, zorder=1)

    draw_edges(G_L, pos_L, ctr_L, col_L)
    draw_edges(G_R, pos_R, ctr_R, col_R)

    # --- Nodes and labels (no orthogroup coloring — plain network) ---
    nodes_L = list(G_L.nodes)
    nodes_R = list(G_R.nodes)

    for nodes, pos, center, sp_col in [
        (nodes_L, pos_L, ctr_L, col_L),
        (nodes_R, pos_R, ctr_R, col_R),
    ]:
        for node in nodes:
            x, y = pos[node]
            label = shorten(node)

            if node == center:
                ax.plot(x, y, "*", ms=16, color=sp_col,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=10)
                ax.annotate(label, (x, y), fontsize=5.5, fontweight="bold",
                            color=sp_col, ha="center", va="bottom",
                            xytext=(0, 10), textcoords="offset points", zorder=6)
            else:
                ax.plot(x, y, "o", ms=7, color=sp_col,
                        markeredgecolor="white", markeredgewidth=0.6,
                        alpha=0.7, zorder=5)
                ax.annotate(label, (x, y), fontsize=4.5, color="#555",
                            ha="center", va="bottom",
                            xytext=(0, 7), textcoords="offset points", zorder=6)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orthogroup", default="OG0000484")
    parser.add_argument("--sp-left", default="ARATH")
    parser.add_argument("--sp-right", default="ORYSA")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    og_genes = load_orthogroup_genes(args.orthogroup)
    if not og_genes:
        print("ERROR: Orthogroup %s not found" % args.orthogroup)
        return
    print("Orthogroup %s: %d species" % (args.orthogroup, len(og_genes)))

    gene_to_og = load_gene_to_orthogroup([args.sp_left, args.sp_right])
    print("Loaded %d gene-to-OG mappings" % len(gene_to_og))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Single panel, sized for 1/4 of a figure (~half-column width)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    plot_panel_d(ax, args.sp_left, args.sp_right, og_genes, gene_to_og,
                 top_k=args.top_k)

    out = OUT_DIR / "panel_d_coexpression.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print("\nSaved: %s" % out)
    plt.close(fig)


if __name__ == "__main__":
    main()
