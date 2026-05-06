"""Table S2 — Full cross-species retrieval metrics per species pair.

Two methods (Procrustes+Jaccard+iter+CSLS, SPACE-v2) × 158 pairs each,
laid out as paired columns for compactness.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "PLANT-SPACE"
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)

proc = json.load(open(ROOT / "results/evaluation_proc_jaccard_iter_csls.json"))
space = json.load(open(ROOT / "results/evaluation_space_v2.json"))

KEEP = ["species_a", "species_b", "n_genes_a", "n_genes_b",
        "hits_at_k", "mrr_at_k", "top_m_hits_at_k", "spearman_rho"]


def to_df(rows, suffix):
    df = pd.DataFrame(rows)[KEEP].copy()
    rename = {c: f"{c}_{suffix}" for c in
              ["hits_at_k", "mrr_at_k", "top_m_hits_at_k", "spearman_rho"]}
    df = df.rename(columns=rename)
    return df


df_p = to_df(proc, "procrustes")
df_s = to_df(space, "space_v2")
merged = pd.merge(df_p, df_s, on=["species_a", "species_b", "n_genes_a", "n_genes_b"],
                  how="outer")
merged = merged.sort_values(["species_a", "species_b"]).reset_index(drop=True)

csv_path = OUT / "tableS2_cross_species_pair_metrics.csv"
merged.to_csv(csv_path, index=False, float_format="%.4f")
print("CSV:", csv_path, "rows:", len(merged))

# Markdown (compact: 4 score cols × 2 methods)
md_lines = ["# Table S2 — Full cross-species retrieval metrics per species pair",
            "",
            "Methods compared: **Procrustes** (Jaccard-weighted + iterative + CSLS) and **SPACE-v2** (autoencoder).",
            "",
            "| species_a | species_b | n_genes_a | n_genes_b | "
            "Hits@50 (Proc) | MRR@50 (Proc) | Top-M (Proc) | Spearman ρ (Proc) | "
            "Hits@50 (SPACE) | MRR@50 (SPACE) | Top-M (SPACE) | Spearman ρ (SPACE) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]


def f(x):
    if pd.isna(x):
        return ""
    return f"{x:.4f}"


for _, r in merged.iterrows():
    md_lines.append(
        f"| {r['species_a']} | {r['species_b']} | {int(r['n_genes_a'])} | {int(r['n_genes_b'])} | "
        f"{f(r['hits_at_k_procrustes'])} | {f(r['mrr_at_k_procrustes'])} | "
        f"{f(r['top_m_hits_at_k_procrustes'])} | {f(r['spearman_rho_procrustes'])} | "
        f"{f(r['hits_at_k_space_v2'])} | {f(r['mrr_at_k_space_v2'])} | "
        f"{f(r['top_m_hits_at_k_space_v2'])} | {f(r['spearman_rho_space_v2'])} |"
    )
md_lines.append("")
md_lines.append(f"_n = {len(merged)} species pairs. k = 50 for retrieval metrics._")

md_path = OUT / "tableS2_cross_species_pair_metrics.md"
md_path.write_text("\n".join(md_lines))
print("MD:", md_path)
print("Sample row:", merged.iloc[0].to_dict())
