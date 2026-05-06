#!/usr/bin/env python3
"""Three-panel UMAP + orthogroup + cross-species network figure.

Panel A: Full 153-species UMAP overview with zoom box
Panel B: Zoomed-in UMAP showing orthogroup genes clustered by species
Panel C: Side-by-side cross-species coexpression networks (2 species)
         with orthogroup connections at the bottom

Usage:
    python scripts/plot_umap_orthogroup_composite.py
    python scripts/plot_umap_orthogroup_composite.py --orthogroup OG0040495
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches, matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parent.parent
OG_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
NET_DIR = ROOT / "data" / "networks"
OUT_DIR = ROOT / "report_figures"
OUT_DIR.mkdir(exist_ok=True)

DIVISION_COLORS = {
    "Eudicots": "#4CAF50", "Monocots": "#FF9800", "Basal angiosperms": "#795548",
    "Gymnosperms": "#2196F3", "Ferns": "#9C27B0", "Lycophytes": "#E91E63",
    "Bryophytes": "#F44336",
}
SPECIES_COMMON = {
    "CITLA": "Watermelon", "CUCSA": "Cucumber", "CUCME": "Melon",
    "MALDO": "Apple", "PRUPE": "Peach", "ROSCH": "Rose",
    "ZIZJU": "Jujube", "FRAVE": "Strawberry", "PRUAV": "Sweet cherry",
    "ARATH": "Arabidopsis", "ORYSA": "Rice",
}
SPECIES_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628",
    "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]
SHARED_OG_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666", "#e41a1c", "#377eb8",
]


# ---- Data loading (unchanged) ----

def load_division_map():
    with open(ROOT / "results" / "seed_selection.json") as f:
        seed_sel = json.load(f)
    div_map = {"ARATH": "Eudicots", "ORYSA": "Monocots", "PICAB": "Gymnosperms",
               "SELMO": "Lycophytes", "MARPO": "Bryophytes"}
    seed_to_div = {"ARATH": "Eudicots", "ORYSA": "Monocots",
                   "PICAB": "Gymnosperms", "SELMO": "Ferns", "MARPO": "Bryophytes"}
    for e in seed_sel["per_nonseed"]:
        div_map[e["species"]] = seed_to_div.get(e["nearest_seed"], "Other")
    return div_map


def load_orthogroup_genes(orthogroup):
    og_genes = defaultdict(list)
    for tsv_path in sorted(OG_DIR.glob("*_transcripts_to_OG.tsv")):
        sp = tsv_path.stem.replace("_transcripts_to_OG", "")
        with open(tsv_path) as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] == orthogroup:
                    og_genes[sp].append(parts[0])
    return dict(og_genes)


def load_gene_to_orthogroup(species_list):
    gene_to_og = {}
    for sp in species_list:
        tsv_path = OG_DIR / ("%s_transcripts_to_OG.tsv" % sp)
        if not tsv_path.exists():
            continue
        with open(tsv_path) as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    gene_to_og[(sp, parts[0])] = parts[2]
    return gene_to_og


def _resolve_gene_in_network(species, gene):
    net_path = NET_DIR / ("%s.tsv" % species)
    if not net_path.exists():
        return gene
    with open(net_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            if parts[0] == gene or parts[1] == gene:
                return gene
            for g in (parts[0], parts[1]):
                if g.startswith(gene):
                    return g
            break
    with open(net_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            for g in (parts[0], parts[1]):
                if g.startswith(gene):
                    return g
    return gene


def load_ego_subnetwork(species, gene, top_k=15):
    net_path = NET_DIR / ("%s.tsv" % species)
    if not net_path.exists():
        return nx.Graph(), gene
    resolved = _resolve_gene_in_network(species, gene)
    if resolved != gene:
        print("    %s: resolved %s -> %s" % (species, gene, resolved))
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
    top_neighbors = set(n for n, w in neighbors[:top_k])
    subgraph_nodes = top_neighbors | {resolved}
    G = nx.Graph()
    for n in subgraph_nodes:
        G.add_node(n)
    with open(net_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            g1, g2, w = parts[0], parts[1], float(parts[2])
            if g1 in subgraph_nodes and g2 in subgraph_nodes:
                G.add_edge(g1, g2, weight=w)
    return G, resolved


def load_all_embeddings(aligned_dir, max_per_species=3000, force_include=None):
    h5_files = sorted(aligned_dir.glob("*.h5"))
    div_map = load_division_map()
    force_include = force_include or {}
    all_emb, all_species, all_proteins, all_divisions = [], [], [], []
    for h5_path in h5_files:
        sp = h5_path.stem
        if sp.startswith("umap"):
            continue
        with h5py.File(h5_path, "r") as fh:
            proteins = [p.decode() if isinstance(p, bytes) else p for p in fh["proteins"][:]]
            emb = fh["embeddings"][:].astype(np.float32)
        if len(emb) > max_per_species:
            must_have = set()
            if sp in force_include:
                prot_to_idx = {p: i for i, p in enumerate(proteins)}
                for gene in force_include[sp]:
                    idx = prot_to_idx.get(gene)
                    if idx is None:
                        base = gene.rsplit(".", 1)[0] if "." in gene else gene
                        for p, i in prot_to_idx.items():
                            if p == base or p.startswith(base):
                                idx = i; break
                    if idx is not None:
                        must_have.add(idx)
            remaining = [i for i in range(len(emb)) if i not in must_have]
            n_random = max(0, max_per_species - len(must_have))
            rng = np.random.RandomState(42)
            random_idx = rng.choice(len(remaining), min(n_random, len(remaining)), replace=False)
            idx = sorted(list(must_have) + [remaining[i] for i in random_idx])
            emb = emb[idx]; proteins = [proteins[i] for i in idx]
        div = div_map.get(sp, "Other")
        all_emb.append(emb)
        all_species.extend([sp] * len(proteins))
        all_proteins.extend(proteins)
        all_divisions.extend([div] * len(proteins))
    return np.vstack(all_emb), np.array(all_species), np.array(all_proteins), np.array(all_divisions)


def find_og_indices(species_arr, proteins_arr, og_genes):
    og_indices, og_species_labels, og_protein_labels = [], [], []
    for sp, genes in og_genes.items():
        for gene in genes:
            mask = (species_arr == sp) & (proteins_arr == gene)
            if mask.sum() > 0:
                idx = np.where(mask)[0][0]
                og_indices.append(idx); og_species_labels.append(sp); og_protein_labels.append(gene)
                continue
            base = gene.rsplit(".", 1)[0] if "." in gene else gene
            for i, (s, p) in enumerate(zip(species_arr, proteins_arr)):
                if s == sp and (p == base or p.startswith(base)):
                    og_indices.append(i); og_species_labels.append(sp); og_protein_labels.append(gene)
                    break
    return og_indices, og_species_labels, og_protein_labels


def og_for_node(sp, node, gene_to_og):
    og = gene_to_og.get((sp, node))
    if not og:
        stripped = node.split("|")[0]
        if stripped.endswith(".p"):
            stripped = stripped[:-2]
        og = gene_to_og.get((sp, stripped))
    if not og:
        # Strip trailing version numbers (.1, .2, etc.)
        import re
        base = re.sub(r'\.\d+$', '', node.split("|")[0])
        og = gene_to_og.get((sp, base))
    return og


# ---- Panel A: Overview ----

def plot_panel_a(ax, coords, divisions, species_arr, zoom_box, og_coords=None):
    for div in ["Eudicots", "Monocots", "Ferns", "Gymnosperms",
                "Basal angiosperms", "Bryophytes", "Lycophytes"]:
        mask = divisions == div
        if mask.sum() == 0:
            continue
        n_sp = len(set(species_arr[mask]))
        ax.scatter(coords[mask, 0], coords[mask, 1], s=0.3, alpha=0.15,
                   c=DIVISION_COLORS.get(div, "#999"), label="%s (%d)" % (div, n_sp), rasterized=True)
    if zoom_box is not None:
        x0, x1, y0, y1 = zoom_box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, lw=2.5, ec="#e41a1c", fc="#e41a1c",
                                    alpha=0.08, ls="-", zorder=10))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, lw=2, ec="#e41a1c", fc="none",
                                    ls="--", zorder=11))
    ax.set_xlabel("UMAP 1", fontsize=10); ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("153 species in shared embedding space", fontsize=11, fontweight="bold", pad=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc",
              fontsize=7, markerscale=8, handletextpad=0.3, title="Division", title_fontsize=8)


# ---- Panel B: Zoom ----

def plot_panel_b(ax, coords, divisions, species_arr, og_coords, og_species_labels,
                 sp_color_map, zoom_box, featured_species=None, max_species=8):
    x0, x1, y0, y1 = zoom_box
    in_w = ((coords[:, 0] >= x0) & (coords[:, 0] <= x1) &
            (coords[:, 1] >= y0) & (coords[:, 1] <= y1))
    ax.scatter(coords[in_w, 0], coords[in_w, 1], s=2, alpha=0.18, c="#cccccc", rasterized=True, zorder=1)
    og_species_unique = sorted(set(og_species_labels))
    # Limit species shown if too many — prioritize featured + seeds
    if len(og_species_unique) > max_species:
        priority = list(featured_species or []) + ["ARATH", "ORYSA", "PICAB", "SELMO", "MARPO"]
        kept = [sp for sp in priority if sp in og_species_unique]
        remaining = [sp for sp in og_species_unique if sp not in kept]
        np.random.RandomState(42).shuffle(remaining)
        og_species_unique = kept + remaining
        og_species_unique = og_species_unique[:max_species]
        # Filter og data to only kept species
        keep_mask = [i for i, s in enumerate(og_species_labels) if s in og_species_unique]
        og_coords = og_coords[keep_mask]
        og_species_labels = [og_species_labels[i] for i in keep_mask]
    for sp in og_species_unique:
        sp_mask = [i for i, s in enumerate(og_species_labels) if s == sp]
        sc = np.array([og_coords[i] for i in sp_mask])
        common = SPECIES_COMMON.get(sp, sp)
        ax.scatter(sc[:, 0], sc[:, 1], s=600, c=[to_rgba(sp_color_map[sp], 0.2)], edgecolors="none", zorder=4)
        ax.scatter(sc[:, 0], sc[:, 1], s=250, alpha=0.95, c=sp_color_map[sp],
                   edgecolors="white", linewidths=2.0, label="%s (%s)" % (common, sp), zorder=5)
    # Radial labels
    center = og_coords.mean(axis=0)
    entries = []
    for sp in og_species_unique:
        sp_mask = [i for i, s in enumerate(og_species_labels) if s == sp]
        for idx in sp_mask:
            coord = og_coords[idx]
            angle = np.arctan2(coord[1] - center[1], coord[0] - center[0])
            entries.append((angle, sp, coord))
    entries.sort(key=lambda x: x[0])
    min_gap = 2 * np.pi / (len(entries) + 1) * 0.8
    angles = [e[0] for e in entries]
    for i in range(1, len(angles)):
        if angles[i] - angles[i - 1] < min_gap:
            angles[i] = angles[i - 1] + min_gap
    shift = entries[0][0] - angles[0]
    angles = [a + shift for a in angles]
    for i, (_, sp, coord) in enumerate(entries):
        a = angles[i]
        ox, oy = 55 * np.cos(a), 55 * np.sin(a)
        ha = "left" if ox >= 0 else "right"
        common = SPECIES_COMMON.get(sp, sp)
        ax.annotate(common, (coord[0], coord[1]), fontsize=8, fontweight="bold",
                    color=sp_color_map[sp], xytext=(ox, oy), textcoords="offset points",
                    ha=ha, va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=sp_color_map[sp], alpha=0.9, lw=0.8),
                    arrowprops=dict(arrowstyle="-", color=sp_color_map[sp], lw=0.8, alpha=0.6))
    if len(og_coords) >= 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(og_coords)
            pts = og_coords[hull.vertices]; pts = np.vstack([pts, pts[0]])
            ax.fill(pts[:, 0], pts[:, 1], alpha=0.10, color="#2e7d32", zorder=2)
            ax.plot(pts[:, 0], pts[:, 1], "--", color="#2e7d32", lw=2.0, alpha=0.5, zorder=3)
        except Exception:
            pass
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("UMAP 1", fontsize=10); ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Orthogroup genes cluster across species", fontsize=11, fontweight="bold", pad=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc",
              fontsize=7.5, markerscale=0.8, title="Species", title_fontsize=8)


# ---- Panel C: Cross-species bipartite network ----

def _shorten_gene(gene_id):
    """Shorten gene IDs for display."""
    # Strip |PACid_... suffixes
    g = gene_id.split("|")[0]
    # Strip trailing .p
    if g.endswith(".p"):
        g = g[:-2]
    return g


def plot_panel_c(ax, sp_left, sp_right, og_genes, gene_to_og,
                 sp_color_left, sp_color_right, top_k=12):
    """Bipartite layout: two vertical columns of genes with horizontal OG connections."""

    # Load subnetworks
    G_left, center_left = load_ego_subnetwork(sp_left, og_genes[sp_left][0], top_k=top_k)
    G_right, center_right = load_ego_subnetwork(sp_right, og_genes[sp_right][0], top_k=top_k)

    # Map nodes to orthogroups
    og_left = {n: og_for_node(sp_left, n, gene_to_og) for n in G_left.nodes}
    og_right = {n: og_for_node(sp_right, n, gene_to_og) for n in G_right.nodes}

    # Find shared orthogroups (excluding center gene's own OG)
    ogs_L = set(og for n, og in og_left.items() if og and n != center_left)
    ogs_R = set(og for n, og in og_right.items() if og and n != center_right)
    shared = sorted(ogs_L & ogs_R)
    og_colors = {og: SHARED_OG_PALETTE[i % len(SHARED_OG_PALETTE)] for i, og in enumerate(shared)}

    print("  Cross-species shared OGs: %d (%s)" % (len(shared), ", ".join(shared[:8])))

    ax.axis("off")

    # --- Arrange nodes with cross-side alignment ---
    # Coordinate y-positions so shared OG members sit at matching heights,
    # producing clean horizontal connection lines instead of a tangled web.

    # Classify nodes into shared-OG vs non-shared
    shared_L, shared_R = {}, {}  # og -> [nodes]
    nonshared_L, nonshared_R = [], []

    for n in G_left.nodes:
        if n == center_left:
            continue
        og = og_left.get(n)
        if og in og_colors:
            shared_L.setdefault(og, []).append(n)
        else:
            nonshared_L.append(n)

    for n in G_right.nodes:
        if n == center_right:
            continue
        og = og_right.get(n)
        if og in og_colors:
            shared_R.setdefault(og, []).append(n)
        else:
            nonshared_R.append(n)

    # Build aligned rows: (left_node_or_None, right_node_or_None)
    rows = [(center_left, center_right)]  # center genes at top

    for og in shared:
        l_nodes = shared_L.get(og, [])
        r_nodes = shared_R.get(og, [])
        for i in range(max(len(l_nodes), len(r_nodes))):
            rows.append((l_nodes[i] if i < len(l_nodes) else None,
                         r_nodes[i] if i < len(r_nodes) else None))

    # Non-shared nodes: keep only a few strongest for context (avoid overflow)
    max_nonshared = 4
    nonshared_L = nonshared_L[:max_nonshared]
    nonshared_R = nonshared_R[:max_nonshared]
    for i in range(max(len(nonshared_L), len(nonshared_R))):
        rows.append((nonshared_L[i] if i < len(nonshared_L) else None,
                     nonshared_R[i] if i < len(nonshared_R) else None))

    x_L, x_R = 1.5, 8.5
    label_x_L, label_x_R = 0.0, 10.0

    n_rows = len(rows)
    y_top, y_bot = n_rows * 0.5, -0.5

    pos_L, pos_R = {}, {}
    nodes_L, nodes_R = [], []
    for i, (ln, rn) in enumerate(rows):
        y = y_top - i * (y_top - y_bot) / max(n_rows - 1, 1)
        if ln is not None:
            pos_L[ln] = (x_L, y)
            nodes_L.append(ln)
        if rn is not None:
            pos_R[rn] = (x_R, y)
            nodes_R.append(rn)

    ax.set_xlim(-1.0, 11.0)
    ax.set_ylim(y_bot - 2.5, y_top + 1.5)

    # --- Background panels ---
    bg_L = plt.Rectangle((-0.5, y_bot - 0.3), 5.0, y_top - y_bot + 1.5,
                          fc=to_rgba(sp_color_left, 0.05), ec=to_rgba(sp_color_left, 0.25),
                          lw=1.5, zorder=0)
    bg_R = plt.Rectangle((5.5, y_bot - 0.3), 5.0, y_top - y_bot + 1.5,
                          fc=to_rgba(sp_color_right, 0.05), ec=to_rgba(sp_color_right, 0.25),
                          lw=1.5, zorder=0)
    ax.add_patch(bg_L)
    ax.add_patch(bg_R)

    # Species titles
    common_L = SPECIES_COMMON.get(sp_left, sp_left)
    common_R = SPECIES_COMMON.get(sp_right, sp_right)
    ax.text(2.0, y_top + 1.2, "%s (%s)" % (common_L, sp_left), fontsize=12, fontweight="bold",
            color=sp_color_left, ha="center", va="bottom")
    ax.text(8.0, y_top + 1.2, "%s (%s)" % (common_R, sp_right), fontsize=12, fontweight="bold",
            color=sp_color_right, ha="center", va="bottom")

    # --- Draw within-species coexpression edges ---
    def draw_intra_edges(G, pos, center, og_map, sp_color, side):
        weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
        if not weights:
            return
        w_min, w_max = min(weights), max(weights)
        w_range = w_max - w_min if w_max > w_min else 1.0

        for u, v, d in G.edges(data=True):
            if u not in pos or v not in pos:
                continue
            w_norm = (d.get("weight", 1.0) - w_min) / w_range
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            og_u, og_v = og_map.get(u), og_map.get(v)

            # Color edges by shared OG membership
            if og_u in og_colors and og_u == og_v:
                ec, ea, ew = og_colors[og_u], 0.25 + 0.2 * w_norm, 0.6 + 0.6 * w_norm
            elif u == center or v == center:
                ec, ea, ew = sp_color, 0.08 + 0.06 * w_norm, 0.3 + 0.3 * w_norm
            else:
                ec, ea, ew = "#d5d5d5", 0.05 + 0.05 * w_norm, 0.2 + 0.2 * w_norm

            # Gentle curve staying within the column
            rad = -0.3 if side == "left" else 0.3
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-", color=ec, lw=ew, alpha=ea,
                                        connectionstyle="arc3,rad=%.2f" % rad),
                        zorder=1)

    draw_intra_edges(G_left, pos_L, center_left, og_left, sp_color_left, "left")
    draw_intra_edges(G_right, pos_R, center_right, og_right, sp_color_right, "right")

    # --- Draw nodes and gene labels ---
    def draw_nodes_and_labels(nodes, pos, center, og_map, sp_color, label_x, ha):
        degrees = dict(nx.Graph([(u, v) for u, v in G_left.edges()]).degree()) if label_x < 5 else dict(nx.Graph([(u, v) for u, v in G_right.edges()]).degree())
        for node in nodes:
            x, y = pos[node]
            og = og_map.get(node)

            if node == center:
                # Center gene — star, larger
                ax.plot(x, y, "*", markersize=18, color=sp_color,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=10)
                label = _shorten_gene(node)
                ax.text(label_x, y, label, fontsize=6.5, fontweight="bold",
                        color=sp_color, ha=ha, va="center", zorder=6)
            elif og in og_colors:
                ms = 9
                ax.plot(x, y, "o", markersize=ms, color=og_colors[og],
                        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                label = _shorten_gene(node)
                ax.text(label_x, y, label, fontsize=5.5, color=og_colors[og],
                        ha=ha, va="center", zorder=6)
            else:
                ms = 5
                ax.plot(x, y, "o", markersize=ms, color="#d0d0d0",
                        markeredgecolor="#aaaaaa", markeredgewidth=0.3, zorder=4)
                label = _shorten_gene(node)
                ax.text(label_x, y, label, fontsize=4.5, color="#999999",
                        ha=ha, va="center", zorder=6)

    draw_nodes_and_labels(nodes_L, pos_L, center_left, og_left, sp_color_left, label_x_L, "right")
    draw_nodes_and_labels(nodes_R, pos_R, center_right, og_right, sp_color_right, label_x_R, "left")

    # --- Draw horizontal dashed lines between aligned shared OG pairs ---
    for og in shared:
        color = og_colors[og]
        left_nodes = [n for n in nodes_L if og_left.get(n) == og and n != center_left]
        right_nodes = [n for n in nodes_R if og_right.get(n) == og and n != center_right]

        # One line per aligned pair (not all-to-all)
        for ln, rn in zip(left_nodes, right_nodes):
            lx, ly = pos_L[ln]
            rx, ry = pos_R[rn]
            ax.plot([lx + 0.15, rx - 0.15], [ly, ry], "--",
                    color=color, lw=1.5, alpha=0.6, zorder=3)

    # --- Orthogroup boxes at bottom ---
    if shared:
        og_box_y = y_bot - 1.0
        box_width = 1.8
        box_height = 0.4
        n_og = min(len(shared), 8)
        total_width = n_og * (box_width + 0.3) - 0.3
        start_x = 5.0 - total_width / 2

        for i, og in enumerate(shared[:8]):
            bx = start_x + i * (box_width + 0.3)
            color = og_colors[og]
            ax.add_patch(plt.Rectangle((bx, og_box_y - box_height / 2), box_width, box_height,
                                        fc=to_rgba(color, 0.15), ec=color, lw=1.2, zorder=3))
            ax.text(bx + box_width / 2, og_box_y, og, fontsize=6, ha="center", va="center",
                    fontweight="bold", color=color, zorder=4)

        ax.text(5.0, og_box_y - box_height / 2 - 0.3, "Shared orthogroups",
                fontsize=9, ha="center", va="top", fontweight="bold", color="#555555")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-dir", type=Path,
                        default=ROOT / "results" / "aligned_embeddings_procrustes_n2v")
    parser.add_argument("--orthogroup", default="OG0036594",
                        help="Orthogroup for UMAP panels A & B")
    parser.add_argument("--orthogroup-c", default=None,
                        help="Separate orthogroup for Panel C (defaults to --orthogroup)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sp-left", default="ARATH")
    parser.add_argument("--sp-right", default="ORYSA")
    args = parser.parse_args()

    aligned_dir = args.aligned_dir
    og_id_umap = args.orthogroup
    og_id_c = args.orthogroup_c or og_id_umap

    # --- Load UMAP orthogroup (Panels A & B) ---
    print("Loading UMAP orthogroup %s..." % og_id_umap)
    og_genes_umap = load_orthogroup_genes(og_id_umap)
    if not og_genes_umap:
        print("ERROR: Orthogroup %s not found" % og_id_umap)
        return
    print("  Found in %d species" % len(og_genes_umap))

    # --- Load Panel C orthogroup (may be different) ---
    if og_id_c != og_id_umap:
        print("Loading Panel C orthogroup %s..." % og_id_c)
        og_genes_c = load_orthogroup_genes(og_id_c)
        print("  Found in %d species" % len(og_genes_c))
    else:
        og_genes_c = og_genes_umap

    print("Loading gene-to-orthogroup mappings...")
    gene_to_og = load_gene_to_orthogroup([args.sp_left, args.sp_right])
    print("  Loaded %d mappings" % len(gene_to_og))

    print("Loading embeddings...")
    X, species_arr, proteins_arr, divisions = load_all_embeddings(
        aligned_dir, force_include=og_genes_umap)
    print("  %d species, %d genes" % (len(set(species_arr)), X.shape[0]))

    cache_path = aligned_dir / "umap_cache_with_og.npz"
    if cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=True)
        coords = data["coords"]
        if len(coords) != len(species_arr):
            cache_path.unlink()
    if not cache_path.exists():
        import umap
        print("Running UMAP...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", n_jobs=-1)
        coords = reducer.fit_transform(X)
        np.savez(str(cache_path), coords=coords, species=species_arr,
                 proteins=proteins_arr, divisions=divisions)

    og_indices, og_species_labels, _ = find_og_indices(species_arr, proteins_arr, og_genes_umap)
    og_coords = coords[og_indices]
    print("Found %d/%d UMAP OG genes" % (len(og_indices), sum(len(g) for g in og_genes_umap.values())))

    og_species_unique = sorted(set(og_species_labels))
    sp_color_map = {sp: SPECIES_COLORS[i % len(SPECIES_COLORS)]
                    for i, sp in enumerate(og_species_unique)}

    # Zoom box
    center = og_coords.mean(axis=0)
    spread = max(og_coords[:, 0].max() - og_coords[:, 0].min(),
                 og_coords[:, 1].max() - og_coords[:, 1].min())
    pad = max(spread * 1.0, 0.8)
    zoom_box = (center[0] - pad, center[0] + pad, center[1] - pad, center[1] + pad)

    # Colors for the two featured species in Panel C
    sp_color_left = sp_color_map.get(args.sp_left, "#e41a1c")
    sp_color_right = sp_color_map.get(args.sp_right, "#4daf4a")

    # ---- Figure ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(16, 6))
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[0.4, 0.6], wspace=0.25)

    ax_a = fig.add_subplot(outer_gs[0, 0])
    ax_b = fig.add_subplot(outer_gs[0, 1])

    print("Plotting Panel A...")
    plot_panel_a(ax_a, coords, divisions, species_arr, zoom_box=None)

    print("Plotting Panel B...")
    plot_panel_c(ax_b, args.sp_left, args.sp_right, og_genes_c, gene_to_og,
                 sp_color_left, sp_color_right, top_k=args.top_k)

    # Panel labels
    ax_a.text(-0.06, 1.06, "A", transform=ax_a.transAxes, fontsize=16, fontweight="bold", va="top")
    ax_b.text(-0.02, 1.02, "B", transform=ax_b.transAxes, fontsize=16, fontweight="bold", va="top")

    out_path = OUT_DIR / "umap_orthogroup_composite.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print("\nSaved: %s" % out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
