"""Table S6 — Full NetGO 2.0 metrics (Fmax + AUPRC for MF/BP/CC) per method."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "PLANT-SPACE"
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)
GO = ROOT / "results/downstream/go_pred"

METHOD_LABELS = {
    "procn2v": "Procrustes (N2V)",
    "procn2v_t5": "Procrustes (N2V) + ProtT5",
    "jaccard": "Jaccard",
    "jaccard_t5": "Jaccard + ProtT5",
    "space_v2": "SPACE-v2",
    "space_v2_t5": "SPACE-v2 + ProtT5",
    "space_pub": "SPACE-published",
    "space_pub_t5": "SPACE-published + ProtT5",
    "prott5": "ProtT5 (alone)",
    "proc_improved": "Procrustes (improved)",
    "proc_improved_t5": "Procrustes (improved) + ProtT5",
}

ASPECTS = ["MF", "BP", "CC"]

rows = []
for key, label in METHOD_LABELS.items():
    fp = GO / f"{key}_summary.json"
    if not fp.exists():
        print("MISSING", fp)
        continue
    blob = json.load(open(fp))
    rec = {"method": label}
    for a in ASPECTS:
        if a in blob.get("per_aspect", {}):
            rec[f"fmax_{a}"] = blob["per_aspect"][a]["fmax"]
            rec[f"auprc_{a}"] = blob["per_aspect"][a]["auprc"]
            rec[f"n_terms_{a}"] = blob["per_aspect"][a]["n_terms"]
        else:
            rec[f"fmax_{a}"] = float("nan")
            rec[f"auprc_{a}"] = float("nan")
            rec[f"n_terms_{a}"] = 0
    rows.append(rec)

df = pd.DataFrame(rows)
csv_path = OUT / "tableS6_netgo_metrics.csv"
df.to_csv(csv_path, index=False, float_format="%.4f")
print("CSV:", csv_path)

# Markdown
md_lines = ["# Table S6 — NetGO 2.0 GO term prediction metrics per method", "",
            "Fmax and AUPRC for Molecular Function (MF), Biological Process (BP), "
            "and Cellular Component (CC) aspects. Annotations: QuickGO experimental codes "
            "(EXP/IDA/IPI/IMP/IGI/IEP/TAS/IC).",
            "",
            "| Method | Fmax (MF) | AUPRC (MF) | Fmax (BP) | AUPRC (BP) | Fmax (CC) | AUPRC (CC) |",
            "|---|---|---|---|---|---|---|"]


def f(x):
    if pd.isna(x):
        return "—"
    return f"{x:.3f}"


for _, r in df.iterrows():
    md_lines.append(
        f"| {r['method']} | {f(r['fmax_MF'])} | {f(r['auprc_MF'])} | "
        f"{f(r['fmax_BP'])} | {f(r['auprc_BP'])} | "
        f"{f(r['fmax_CC'])} | {f(r['auprc_CC'])} |"
    )
md_lines.append("")
n_mf = int(df["n_terms_MF"].iloc[0]) if "n_terms_MF" in df.columns and len(df) else 0
n_bp = int(df["n_terms_BP"].iloc[0]) if len(df) else 0
n_cc = int(df["n_terms_CC"].iloc[0]) if len(df) else 0
md_lines.append(f"_Number of GO terms evaluated: MF={n_mf}, BP={n_bp}, CC={n_cc}._")

md_path = OUT / "tableS6_netgo_metrics.md"
md_path.write_text("\n".join(md_lines))
print("MD:", md_path)
print(df[["method", "fmax_MF", "fmax_BP", "fmax_CC"]].to_string(index=False))
