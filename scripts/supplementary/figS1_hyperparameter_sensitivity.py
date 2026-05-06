#!/usr/bin/env python3
"""Figure S1 — Node2Vec hyperparameter sensitivity.

Two-panel figure:
  (A) Stage 1: Spearman rho across p x q grid (heatmap)
  (B) Stage 2: Spearman rho across num_walks x walk_length, faceted by epochs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path.home() / 'PLANT-SPACE'
RESULTS = PROJECT / 'results'
OUT = RESULTS / 'supplementary'
OUT.mkdir(parents=True, exist_ok=True)

STAGE1_PATH = RESULTS / 'grid_search_cross_species' / 'stage1_results.json'
STAGE2_PATH = RESULTS / 'grid_search_cross_species' / 'grid_search_results.json'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

PROC_ORANGE = '#e07b54'
SPACE_BLUE = '#7aaed4'
PROTT5_GREEN = '#5fb360'


def load_stage1() -> pd.DataFrame:
    with open(STAGE1_PATH) as f:
        d = json.load(f)
    rows = []
    for r in d['results']:
        rows.append({
            'p': r['p'], 'q': r['q'],
            'spearman': r['mean_spearman_rho'],
            'hits_at_k': r['mean_hits_at_k'],
        })
    return pd.DataFrame(rows)


def load_stage2() -> pd.DataFrame:
    with open(STAGE2_PATH) as f:
        d = json.load(f)
    rows = []
    for r in d['results']:
        if r.get('p') != 1.0 or r.get('q') != 0.7:
            continue
        rows.append({
            'num_walks': r['num_walks'],
            'walk_length': r['walk_length'],
            'epochs': r['epochs'],
            'spearman': r['mean_spearman_rho'],
            'hits_at_k': r['mean_hits_at_k'],
        })
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['num_walks', 'walk_length', 'epochs'], keep='first')
    return df


def plot_panel_a(ax, df: pd.DataFrame, fig) -> None:
    p_vals = sorted(df['p'].unique())
    q_vals = sorted(df['q'].unique())
    grid = np.full((len(p_vals), len(q_vals)), np.nan)
    for _, row in df.iterrows():
        i = p_vals.index(row['p'])
        j = q_vals.index(row['q'])
        grid[i, j] = row['spearman']

    im = ax.imshow(grid, cmap='viridis', aspect='auto',
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(range(len(q_vals)))
    ax.set_xticklabels([str(q) for q in q_vals])
    ax.set_yticks(range(len(p_vals)))
    ax.set_yticklabels([str(p) for p in p_vals])
    ax.set_xlabel('q (in-out parameter)')
    ax.set_ylabel('p (return parameter)')
    ax.set_title('(A) Stage 1: p x q grid\n(epochs=5, num_walks=20, walk_length=50)')

    for i in range(len(p_vals)):
        for j in range(len(q_vals)):
            v = grid[i, j]
            if not np.isnan(v):
                norm = (v - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid) + 1e-9)
                color = 'white' if norm < 0.5 else 'black'
                ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                        color=color, fontsize=8.5)

    best_idx = np.nanargmax(grid)
    bi, bj = np.unravel_index(best_idx, grid.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                               fill=False, edgecolor=PROC_ORANGE, lw=2.5))

    # Horizontal colorbar BELOW the heatmap (not to the right).
    # Use an inset axes anchored to ax so the bar tracks the panel position.
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.10, pos.width, 0.025])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Mean Spearman rho', labelpad=4)
    cbar.ax.tick_params(labelsize=8)


def plot_panel_b_faceted(axes, df: pd.DataFrame) -> None:
    """Three subplots, one per epochs value."""
    epochs_list = sorted(df['epochs'].unique())
    num_walks_list = sorted(df['num_walks'].unique())
    walk_length_list = sorted(df['walk_length'].unique())
    palette = [PROC_ORANGE, SPACE_BLUE, PROTT5_GREEN]
    wl_to_color = {wl: palette[i % len(palette)] for i, wl in enumerate(walk_length_list)}

    bar_w = 0.8 / len(walk_length_list)

    # Shared y-range (zoomed) for visibility
    ymin = max(0.0, df['spearman'].min() - 0.005)
    ymax = df['spearman'].max() + 0.005

    for ai, (ax, ep) in enumerate(zip(axes, epochs_list)):
        x = np.arange(len(num_walks_list))
        for k, wl in enumerate(walk_length_list):
            ys = []
            for nw in num_walks_list:
                row = df[(df['epochs'] == ep) & (df['num_walks'] == nw)
                         & (df['walk_length'] == wl)]
                ys.append(float(row['spearman'].iloc[0]) if len(row) else np.nan)
            pos = x + (k - (len(walk_length_list) - 1) / 2) * bar_w
            ax.bar(pos, ys, width=bar_w * 0.95,
                   color=wl_to_color[wl], edgecolor='black',
                   linewidth=0.4,
                   label=f'walk_length={int(wl)}' if ai == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f'nw={int(nw)}' for nw in num_walks_list], fontsize=9)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f'epochs={int(ep)}')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        if ai > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel('Mean Spearman rho')

    axes[0].legend(loc='lower right', frameon=True, fontsize=8)


def main() -> None:
    df1 = load_stage1()
    df2 = load_stage2()

    print(f'Stage 1: {len(df1)} configs')
    print(f'Stage 1 best: {df1.loc[df1["spearman"].idxmax()].to_dict()}')
    print(f'Stage 2: {len(df2)} configs after dedup')
    print(f'Stage 2 best: {df2.loc[df2["spearman"].idxmax()].to_dict()}')

    fig = plt.figure(figsize=(15, 5.4))
    # Layout: heatmap on left (1 col) + 3 stage-2 facets on right (3 cols)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.05, 0.05, 0.95, 0.95, 0.95],
                          wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b1 = fig.add_subplot(gs[0, 2])
    ax_b2 = fig.add_subplot(gs[0, 3], sharey=ax_b1)
    ax_b3 = fig.add_subplot(gs[0, 4], sharey=ax_b1)

    plot_panel_a(ax_a, df1, fig)
    plot_panel_b_faceted([ax_b1, ax_b2, ax_b3], df2)

    # Header above stage-2 facets
    fig.text(0.71, 0.97, '(B) Stage 2: num_walks x walk_length x epochs  [p=1.0, q=0.7]',
             ha='center', va='top', fontsize=11)

    fig.suptitle(
        'Figure S1 — Node2Vec hyperparameter sensitivity '
        '(mean Spearman rho across ARATH-ORYSA, ARATH-BRADI, ORYSA-BRADI)',
        fontsize=11.5, y=1.04,
    )

    out_png = OUT / 'figS1_hyperparameter_sensitivity.png'
    out_pdf = OUT / 'figS1_hyperparameter_sensitivity.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Saved {out_png}')
    print(f'Saved {out_pdf}')


if __name__ == '__main__':
    main()
