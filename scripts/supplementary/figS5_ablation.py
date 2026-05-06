"""Figure S5 — Weighted (Jaccard + iter + CSLS) vs unweighted Procrustes ablation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "PLANT-SPACE"
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)


def load_list(p: Path):
    with open(p) as fh:
        return json.load(fh)


cross_full = load_list(ROOT / "results/evaluation_proc_jaccard_iter_csls.json")
cross_base = load_list(ROOT / "results/evaluation_procrustes_n2v.json")
within_full = load_list(ROOT / "results/evaluation_proc_jaccard_iter_csls_within.json")
within_base = load_list(ROOT / "results/evaluation_within_species_procrustes_n2v.json")

cross_metrics = ["hits_at_k", "mrr_at_k", "spearman_rho", "top_m_hits_at_k"]
cross_labels = ["Hits@50", "MRR@50", "Spearman ρ", "Top-M Hits@50"]
within_metrics = ["precision_at_k", "recall_at_k"]
within_labels = ["Precision@50 (within)", "Recall@50 (within)"]


def mean_metric(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


rows = []
for key, lbl in zip(cross_metrics, cross_labels):
    m_b, s_b, n_b = mean_metric(cross_base, key)
    m_f, s_f, n_f = mean_metric(cross_full, key)
    rows.append({
        "category": "cross-species (n=158 pairs)",
        "metric": lbl,
        "basic_mean": m_b, "basic_std": s_b, "basic_n": n_b,
        "full_mean": m_f, "full_std": s_f, "full_n": n_f,
        "improvement_pct": 100.0 * (m_f - m_b) / m_b if m_b != 0 else float("nan"),
    })
for key, lbl in zip(within_metrics, within_labels):
    m_b, s_b, n_b = mean_metric(within_base, key)
    m_f, s_f, n_f = mean_metric(within_full, key)
    rows.append({
        "category": "within-species (n=153)",
        "metric": lbl,
        "basic_mean": m_b, "basic_std": s_b, "basic_n": n_b,
        "full_mean": m_f, "full_std": s_f, "full_n": n_f,
        "improvement_pct": 100.0 * (m_f - m_b) / m_b if m_b != 0 else float("nan"),
    })

df = pd.DataFrame(rows)
csv_path = OUT / "figS5_jaccard_csls_ablation.csv"
df.to_csv(csv_path, index=False)
print("CSV:", csv_path)
print(df.to_string(index=False))

# Plot
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]
labels = df["metric"].tolist()
basic_means = df["basic_mean"].to_numpy()
basic_stds = df["basic_std"].to_numpy()
full_means = df["full_mean"].to_numpy()
full_stds = df["full_std"].to_numpy()

fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
x = np.arange(len(labels))
w = 0.38
b1 = ax.bar(x - w/2, basic_means, w, yerr=basic_stds,
            label="Procrustes (basic)", color="#bcbcbc", edgecolor="black",
            linewidth=0.4, capsize=3, error_kw={"linewidth": 0.7})
b2 = ax.bar(x + w/2, full_means, w, yerr=full_stds,
            label="Procrustes + Jaccard + iterative + CSLS", color="#e07b54",
            edgecolor="black", linewidth=0.4, capsize=3,
            error_kw={"linewidth": 0.7})

# Add value labels above bars
for bars, means in [(b1, basic_means), (b2, full_means)]:
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Score")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Vertical separator between cross and within
ax.axvline(len(cross_metrics) - 0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
ymax = max(full_means.max() + full_stds.max(), 0.7) * 1.05
ax.set_ylim(top=ymax)
ax.text(len(cross_metrics) / 2 - 0.5, ymax * 0.97, "Cross-species (n=158 pairs)",
        ha="center", fontsize=9, alpha=0.7)
ax.text(len(cross_metrics) + len(within_metrics) / 2 - 0.5, ymax * 0.97,
        "Within-species (n=153)", ha="center", fontsize=9, alpha=0.7)
# Legend in top-left, in-plot
ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=9)
fig.suptitle("Ablation: Jaccard weighting + iterative refinement + CSLS over basic Procrustes",
             y=1.0, fontsize=11)

fig.tight_layout()
png_path = OUT / "figS5_jaccard_csls_ablation.png"
pdf_path = OUT / "figS5_jaccard_csls_ablation.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print("PNG:", png_path)
print("PDF:", pdf_path)
