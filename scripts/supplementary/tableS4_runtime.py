"""Table S4 — Runtime benchmarks (Procrustes / Procrustes+Jaccard / SPACE)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "PLANT-SPACE"
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "results/benchmark_runtime.json"

blob = json.load(open(DATA))
hw = blob["hardware"]
res = blob["results"]

LABELS = {
    "procrustes": "Procrustes (basic)",
    "procrustes_jaccard": "Procrustes + Jaccard + iter + CSLS",
    "space": "SPACE autoencoder",
}

rows = []
for key, label in LABELS.items():
    r = res[key]
    rows.append({
        "method": label,
        "stage1_seconds": r["stage1_seconds"],
        "stage2_seconds": r["stage2_seconds"],
        "total_seconds": r["total_seconds"],
        "n_species": r["n_species"],
        "n_success": r["n_success"],
        "per_species_mean_s": r["per_species_mean"],
        "per_species_median_s": r["per_species_median"],
        "per_species_min_s": r["per_species_min"],
        "per_species_max_s": r["per_species_max"],
        "per_species_std_s": r["per_species_std"],
    })

df = pd.DataFrame(rows)
csv_path = OUT / "tableS4_runtime_benchmarks.csv"
df.to_csv(csv_path, index=False, float_format="%.4f")
print("CSV:", csv_path)


def fmt_t(s):
    if s < 60:
        return f"{s:.2f} s"
    elif s < 3600:
        return f"{s/60:.2f} min"
    else:
        return f"{s/3600:.2f} h"


md_lines = ["# Table S4 — Runtime benchmarks for embedding alignment", "",
            "Wall-clock time on the GCP VM "
            f"({hw['cpu_cores']} CPU cores, {hw['ram_gb']:.0f} GB RAM, "
            f"{hw['gpu_count']}× NVIDIA L4). Stage 1 = seed-species joint alignment; "
            "Stage 2 = projecting all 153 species into the shared space.",
            "",
            "| Method | Stage 1 | Stage 2 | Total | Per-species mean | Median | Min | Max |",
            "|---|---|---|---|---|---|---|---|"]

for _, r in df.iterrows():
    md_lines.append(
        f"| **{r['method']}** | {fmt_t(r['stage1_seconds'])} | {fmt_t(r['stage2_seconds'])} | "
        f"{fmt_t(r['total_seconds'])} | {fmt_t(r['per_species_mean_s'])} | "
        f"{fmt_t(r['per_species_median_s'])} | {fmt_t(r['per_species_min_s'])} | "
        f"{fmt_t(r['per_species_max_s'])} |"
    )

# Speedup row
proc_total = df.loc[df["method"].str.startswith("Procrustes (basic)"), "total_seconds"].iloc[0]
pj_total = df.loc[df["method"].str.startswith("Procrustes + Jaccard"), "total_seconds"].iloc[0]
space_total = df.loc[df["method"].str.startswith("SPACE"), "total_seconds"].iloc[0]
md_lines.append("")
md_lines.append(
    f"_Procrustes is **{space_total / proc_total:.0f}×** faster than the SPACE autoencoder; "
    f"Procrustes+Jaccard is **{space_total / pj_total:.0f}×** faster._"
)
md_lines.append(f"_n = {df['n_species'].iloc[0]} species per method._")

md_path = OUT / "tableS4_runtime_benchmarks.md"
md_path.write_text("\n".join(md_lines))
print("MD:", md_path)
print(df.to_string(index=False))
