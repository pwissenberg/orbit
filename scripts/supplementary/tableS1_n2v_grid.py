"""Table S1 — Node2Vec hyperparameter grid search (two-stage design).

Emits two CSVs, one per stage, so each table is a clean rectangle without
empty cells:

  tableS1a_link_prediction.csv  — Stage 1 link-prediction AUC across the
      full (p, q, num_walks, epochs) grid at fixed walk_length=50.

  tableS1b_cross_species.csv    — Stage 2 cross-species Spearman rho
      refinement at the Stage 1 winner (p=1.0, q=0.7), varying
      (num_walks, walk_length, epochs).

A markdown sidecar tableS1_n2v_grid_search.md documents both tables and the
two-stage rationale.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "PLANT-SPACE"
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)

n2v_lp     = json.load(open(ROOT / "results/grid_search_node2vec/stage1_results.json"))
n2v_lp_ext = json.load(open(ROOT / "results/grid_search_node2vec/stage1_extended_results.json"))
cs         = json.load(open(ROOT / "results/grid_search_cross_species/grid_search_results.json"))

# ----- Stage 1: link prediction -----
lp_rows = []
for r in n2v_lp["results"]:
    lp_rows.append({
        "p": r["p"], "q": r["q"],
        "num_walks": r["num_walks"],
        "walk_length": 50,
        "epochs": r["epochs"],
        "link_prediction_AUC": r["mean_auc"],
        "AUC_std": r["std_auc"],
    })
for r in n2v_lp_ext["results"]:
    lp_rows.append({
        "p": 1.0, "q": 1.0,
        "num_walks": r["num_walks"],
        "walk_length": 50,
        "epochs": r["epochs"],
        "link_prediction_AUC": r["mean_auc"],
        "AUC_std": r["std_auc"],
    })
lp_df = (pd.DataFrame(lp_rows)
           .drop_duplicates(subset=["p","q","num_walks","walk_length","epochs"], keep="first")
           .sort_values(["p","q","num_walks","walk_length","epochs"])
           .reset_index(drop=True))
lp_csv = OUT / "tableS1a_link_prediction.csv"
lp_df.to_csv(lp_csv, index=False, float_format="%.4f")
print("S1a:", lp_csv, "rows:", len(lp_df))

# ----- Stage 2: cross-species Spearman refinement -----
cs_rows = []
for r in cs["results"]:
    cs_rows.append({
        "p": r["p"], "q": r["q"],
        "num_walks": r["num_walks"],
        "walk_length": r.get("walk_length", 50),
        "epochs": r["epochs"],
        "cross_species_spearman": r["mean_spearman_rho"],
    })
cs_df = (pd.DataFrame(cs_rows)
           .drop_duplicates(subset=["p","q","num_walks","walk_length","epochs"], keep="first")
           .sort_values(["num_walks","walk_length","epochs"])
           .reset_index(drop=True))
cs_csv = OUT / "tableS1b_cross_species.csv"
cs_df.to_csv(cs_csv, index=False, float_format="%.4f")
print("S1b:", cs_csv, "rows:", len(cs_df))

# ----- Markdown sidecar -----
md = ["# Table S1 — Node2Vec hyperparameter grid search",
      "",
      "Two-stage design: a cheap link-prediction proxy (S1a) was evaluated across",
      "the full (p, q, num_walks, epochs) grid at walk_length=50; the (p, q) winner",
      "was then fixed and the more expensive cross-species Spearman rho metric (S1b)",
      "was swept over (num_walks, walk_length, epochs).",
      "",
      "## Table S1a — Link-prediction grid (Stage 1)",
      "",
      "| p | q | num_walks | walk_length | epochs | Link-prediction AUC | AUC sigma |",
      "|---|---|-----------|-------------|--------|---------------------|-----------|"]
for _, r in lp_df.iterrows():
    md.append(f"| {r.p:.4f} | {r.q:.4f} | {int(r.num_walks)} | {int(r.walk_length)} | "
              f"{int(r.epochs)} | {r.link_prediction_AUC:.4f} | {r.AUC_std:.4f} |")
md += ["",
       f"_Stage 1: {len(lp_df)} configurations. AUC averaged over {{ARATH, ORYSA, PICAB}}._",
       "",
       "## Table S1b — Cross-species Spearman refinement (Stage 2)",
       "",
       "| p | q | num_walks | walk_length | epochs | Cross-species Spearman rho |",
       "|---|---|-----------|-------------|--------|----------------------------|"]
for _, r in cs_df.iterrows():
    md.append(f"| {r.p:.4f} | {r.q:.4f} | {int(r.num_walks)} | {int(r.walk_length)} | "
              f"{int(r.epochs)} | {r.cross_species_spearman:.4f} |")
md += ["",
       f"_Stage 2: {len(cs_df)} configurations at the Stage 1 winner (p=1.0, q=0.7)._",
       "_Spearman rho averaged over pairs {ARATH-ORYSA, ARATH-BRADI, ORYSA-BRADI}._"]

md_path = OUT / "tableS1_n2v_grid_search.md"
md_path.write_text("\n".join(md))
print("MD:", md_path)
