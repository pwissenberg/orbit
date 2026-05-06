"""Figure S4 — Per-species Fmax for cross-species GO transfer.

Faceted bar chart: 3 facets (MF/BP/CC), bars per species, grouped by method.
Methods: procrustes, prott5, procrustes_prott5.

NOTE on error bars: fmax_std in the source JSON is the std of per-GO-term Fmax
values (one Fmax per term, then mean and std taken across the n_terms terms in
that aspect). We display SE = SD / sqrt(n_terms) to convey uncertainty in the
mean Fmax. SE is then clipped so the displayed bars stay in [0, 1].
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.home() / 'PLANT-SPACE'
OUT = ROOT / 'results/supplementary'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'results/downstream/go_transfer_cafa.json'

with open(DATA) as fh:
    blob = json.load(fh)

df = pd.DataFrame(blob['results'])
PRIMARY = ['procrustes', 'prott5', 'procrustes_prott5']
df = df[df['method'].isin(PRIMARY)].copy()
df = df[df['classifier'] == 'logreg'].copy()

LABELS = {
    'procrustes': 'Procrustes',
    'prott5': 'ProtT5',
    'procrustes_prott5': 'Procrustes+ProtT5',
}
COLORS = {
    'Procrustes': '#e07b54',
    'ProtT5': '#5fb360',
    'Procrustes+ProtT5': '#9467bd',
}
df['method_label'] = df['method'].map(LABELS)

# SE = SD / sqrt(n_terms). fmax_std is std across GO terms; n_terms gives the
# count. SE measures uncertainty of the mean Fmax for that (species, aspect).
df['fmax_se'] = df['fmax_std'] / np.sqrt(df['n_terms'].clip(lower=1))

aspects = ['MF', 'BP', 'CC']
species = sorted(df['test_species'].unique())
print('species:', species)

# n_test per species for x-tick annotation
n_test_by_species = (
    df.groupby('test_species')['n_test'].first().to_dict()
)

# Save tidy CSV
csv_out = OUT / 'figS4_per_species_go_transfer.csv'
df_out = df[['test_species', 'aspect', 'method', 'method_label',
             'fmax', 'fmax_std', 'fmax_se',
             'auprc', 'auprc_std', 'n_terms', 'n_train', 'n_test']].sort_values(
    ['aspect', 'test_species', 'method'])
df_out.to_csv(csv_out, index=False)
print('CSV:', csv_out)

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica']

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=300, sharey=True)
methods_ordered = list(LABELS.values())
n_meth = len(methods_ordered)
bar_w = 0.8 / n_meth
x = np.arange(len(species))

for ax, asp in zip(axes, aspects):
    sub_a = df[df['aspect'] == asp]
    for i, m_lbl in enumerate(methods_ordered):
        sub_m = (sub_a[sub_a['method_label'] == m_lbl]
                 .set_index('test_species').reindex(species))
        means = sub_m['fmax'].to_numpy()
        ses = sub_m['fmax_se'].to_numpy()
        # Clip error bars so they remain in [0, 1] (Fmax range)
        lower = np.minimum(ses, means)            # don't go below 0
        upper = np.minimum(ses, 1.0 - means)       # don't go above 1
        yerr = np.vstack([lower, upper])
        pos = x + (i - (n_meth - 1) / 2) * bar_w
        ax.bar(pos, means, bar_w, yerr=yerr, label=m_lbl, color=COLORS[m_lbl],
               edgecolor='black', linewidth=0.4, capsize=2,
               error_kw={'linewidth': 0.7, 'alpha': 0.85})
    ax.set_xticks(x)
    xt = [f'{sp}\n(n={n_test_by_species[sp]})' for sp in species]
    ax.set_xticklabels(xt, rotation=0, ha='center', fontsize=9)
    ax.set_title(asp)
    ax.set_xlabel('Test species (n = # test proteins)')
    ax.set_ylim(0.0, 1.05)
    # thin dashed reference line at y=1
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Fmax')
axes[-1].legend(loc='upper right', frameon=False)
fig.suptitle(
    'Per-species cross-species GO transfer (Fmax). Error bars = SE across GO terms.',
    y=1.02,
)
fig.tight_layout()

png_path = OUT / 'figS4_per_species_go_transfer.png'
pdf_path = OUT / 'figS4_per_species_go_transfer.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print('PNG:', png_path)
print('PDF:', pdf_path)

# Spot-check
sub = df[(df['aspect'] == 'MF') & (df['test_species'] == 'ORYSA')][
    ['method_label', 'fmax', 'fmax_std', 'fmax_se', 'n_terms']]
print('\nSpot-check (ORYSA MF):')
print(sub.to_string(index=False))
