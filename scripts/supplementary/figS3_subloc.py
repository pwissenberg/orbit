"""Figure S3 — Per-localization MCC breakdown for subcellular prediction.

Compare per-compartment MCC for: Procrustes (proc_full = base Procrustes,
matches main results section), SPACE-v2, ProtT5, and Procrustes+ProtT5.

Error bars are SE (= SD / sqrt(n_species)) clipped to MCC range [-1, 1].
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.home() / 'PLANT-SPACE'
SUBDIR = ROOT / 'results/downstream/subloc'
OUT = ROOT / 'results/supplementary'
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    'Procrustes': 'proc_full_mcc_per_compartment.csv',
    'SPACE-v2': 'space_v2_mcc_per_compartment.csv',
    'ProtT5': 'prott5_mcc_per_compartment.csv',
    'Procrustes+ProtT5': 'proc_full_t5_mcc_per_compartment.csv',
}

COLORS = {
    'Procrustes': '#e07b54',
    'SPACE-v2': '#7aaed4',
    'ProtT5': '#5fb360',
    'Procrustes+ProtT5': '#9467bd',
}

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica']

records = []
for method, fname in METHODS.items():
    fp = SUBDIR / fname
    if not fp.exists():
        print(f'MISSING: {fp}', file=sys.stderr)
        continue
    df = pd.read_csv(fp)
    grp = df.groupby('compartment')['mcc'].agg(['mean', 'std', 'count']).reset_index()
    grp['method'] = method
    # SE = SD / sqrt(n_species). guard against count<=1 (then SE -> 0).
    grp['se'] = grp['std'] / np.sqrt(grp['count'].clip(lower=1))
    records.append(grp)

data = pd.concat(records, ignore_index=True)
comps = list(data[data['method'] == 'Procrustes']['compartment'])
data['compartment'] = pd.Categorical(data['compartment'], categories=comps, ordered=True)
data = data.sort_values(['compartment', 'method']).reset_index(drop=True)

csv_path = OUT / 'figS3_subloc_mcc_per_compartment.csv'
pivot = data.pivot(index='compartment', columns='method', values='mean').reindex(comps)
pivot.to_csv(csv_path)
print('CSV:', csv_path)

# Plot
fig, ax = plt.subplots(figsize=(12, 5.2), dpi=300)
methods = list(METHODS.keys())
n_methods = len(methods)
n_comp = len(comps)
bar_w = 0.8 / n_methods
x = np.arange(n_comp)

for i, m in enumerate(methods):
    sub = data[data['method'] == m].set_index('compartment').reindex(comps)
    means = sub['mean'].to_numpy()
    ses = sub['se'].to_numpy()
    # Clip SE bars to [-1, 1] (MCC range)
    lower = np.minimum(ses, means - (-1.0))
    upper = np.minimum(ses, 1.0 - means)
    yerr = np.vstack([lower, upper])
    pos = x + (i - (n_methods - 1) / 2) * bar_w
    ax.bar(pos, means, bar_w, yerr=yerr, label=m, color=COLORS[m],
           edgecolor='black', linewidth=0.4, capsize=2,
           error_kw={'linewidth': 0.7, 'alpha': 0.85})

ax.set_xticks(x)
ax.set_xticklabels(comps, rotation=30, ha='right')
ax.set_ylabel('MCC (mean across species)')
ax.set_xlabel('Subcellular compartment')
ax.set_title(
    'Per-compartment MCC for subcellular localization prediction '
    '(error bars = SE across species)'
)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylim(-0.1, 1.05)
ax.legend(loc='upper right', frameon=False, ncol=2)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
png_path = OUT / 'figS3_subloc_mcc_per_compartment.png'
pdf_path = OUT / 'figS3_subloc_mcc_per_compartment.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print('PNG:', png_path)
print('PDF:', pdf_path)

print('\nSpot-check (Procrustes Plastid mean):',
      float(pivot.loc['Plastid', 'Procrustes']))
print('Spot-check (ProtT5 Nucleus mean):',
      float(pivot.loc['Nucleus', 'ProtT5']))
print('Spot-check Procrustes Nucleus n_species/se:',
      data[(data['method']=='Procrustes') & (data['compartment']=='Nucleus')][['count','std','se']].to_string(index=False))
