"""Table S5 — Full DeepLoc 2.0 metrics per method.

For each method, compile mean ± std of (f1_micro, f1_macro, accuracy,
jaccard_micro) from CV and LOSO scores.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "PLANT-SPACE"
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)
SUB = ROOT / "results/downstream/subloc"

# method_key -> display label
METHOD_LABELS = {
    "proc_full": "Procrustes",
    "proc_full_t5": "Procrustes + ProtT5",
    "jaccard": "Jaccard",
    "jaccard_t5": "Jaccard + ProtT5",
    "space_v2": "SPACE-v2",
    "space_v2_t5": "SPACE-v2 + ProtT5",
    "vanilla": "Vanilla",
    "vanilla_t5": "Vanilla + ProtT5",
    "prott5": "ProtT5 (alone)",
}

METRICS = ["f1_micro", "f1_macro", "accuracy", "jaccard_micro"]


def summary_for(method, kind):
    """kind in {'cv', 'loso'}"""
    fp = SUB / f"{method}_{'cv_scores' if kind == 'cv' else 'loso_scores'}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    out = {"method": METHOD_LABELS[method], "evaluation": kind, "n_rows": len(df)}
    for m in METRICS:
        if m in df.columns:
            out[f"{m}_mean"] = df[m].mean()
            out[f"{m}_std"] = df[m].std()
        else:
            out[f"{m}_mean"] = float("nan")
            out[f"{m}_std"] = float("nan")
    return out


rows = []
for method in METHOD_LABELS:
    for kind in ["cv", "loso"]:
        r = summary_for(method, kind)
        if r is not None:
            rows.append(r)

df = pd.DataFrame(rows)
csv_path = OUT / "tableS5_deeploc_metrics.csv"
df.to_csv(csv_path, index=False, float_format="%.4f")
print("CSV:", csv_path)

# Markdown — pivot so each method has CV row + LOSO row
md_lines = ["# Table S5 — DeepLoc 2.0 subcellular localization metrics per method", "",
            "Mean ± standard deviation across folds (CV = within-species 5-fold; "
            "LOSO = leave-one-species-out across {ARATH, GLYMA, ORYSA, ZEAMA}).",
            "",
            "| Method | Evaluation | F1-micro | F1-macro | Accuracy | Jaccard (micro) |",
            "|---|---|---|---|---|---|"]


def fmt(m, s):
    if pd.isna(m):
        return "—"
    return f"{m:.3f} ± {s:.3f}"


# Sort by method label, with CV before LOSO
df["_eval_order"] = df["evaluation"].map({"cv": 0, "loso": 1})
df = df.sort_values(["method", "_eval_order"])

for _, r in df.iterrows():
    eval_label = "5-fold CV" if r["evaluation"] == "cv" else "LOSO"
    md_lines.append(
        f"| {r['method']} | {eval_label} | "
        f"{fmt(r['f1_micro_mean'], r['f1_micro_std'])} | "
        f"{fmt(r['f1_macro_mean'], r['f1_macro_std'])} | "
        f"{fmt(r['accuracy_mean'], r['accuracy_std'])} | "
        f"{fmt(r['jaccard_micro_mean'], r['jaccard_micro_std'])} |"
    )
md_lines.append("")

md_path = OUT / "tableS5_deeploc_metrics.md"
md_path.write_text("\n".join(md_lines))
print("MD:", md_path)
print(df[["method", "evaluation", "f1_micro_mean", "f1_macro_mean", "accuracy_mean"]].to_string(index=False))
