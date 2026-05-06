"""Table S3 — Network statistics for all 153 species.

For each TSV in data/networks/ compute n_genes, n_edges, density, mean_degree.
Use joblib parallel since I/O bound.
"""
from __future__ import annotations
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd

ROOT = Path.home() / "PLANT-SPACE"
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)
NET_DIR = ROOT / "data/networks"


def stats_for(tsv: Path):
    df = pd.read_csv(tsv, sep="\t", header=None, names=["a", "b", "w"], dtype={"a": str, "b": str})
    nodes = set(df["a"]).union(df["b"])
    n = len(nodes)
    e = len(df)
    density = (2 * e) / (n * (n - 1)) if n > 1 else 0.0
    mean_degree = (2 * e) / n if n > 0 else 0.0
    return {
        "species": tsv.stem,
        "n_genes": n,
        "n_edges": e,
        "density": density,
        "mean_degree": mean_degree,
    }


tsvs = sorted(NET_DIR.glob("*.tsv"))
print("processing", len(tsvs), "networks")
records = Parallel(n_jobs=-1, verbose=5)(delayed(stats_for)(p) for p in tsvs)

df = pd.DataFrame(records).sort_values("species").reset_index(drop=True)
csv_path = OUT / "tableS3_network_stats.csv"
df.to_csv(csv_path, index=False, float_format="%.6g")
print("CSV:", csv_path, "rows:", len(df))

md_lines = ["# Table S3 — Network statistics for all 153 species", "",
            "Coexpression networks built from TEA-GCN. Density = 2E/(N(N-1)); "
            "mean degree = 2E/N.",
            "",
            "| Species | n_genes | n_edges | density | mean degree |",
            "|---|---|---|---|---|"]
for _, r in df.iterrows():
    md_lines.append(
        f"| {r['species']} | {int(r['n_genes'])} | {int(r['n_edges'])} | "
        f"{r['density']:.3g} | {r['mean_degree']:.3f} |"
    )
md_lines.append("")
md_lines.append(f"_n = {len(df)} species. "
                f"Mean genes per network: {df['n_genes'].mean():.0f}; "
                f"mean edges: {df['n_edges'].mean():.0f}._")

md_path = OUT / "tableS3_network_stats.md"
md_path.write_text("\n".join(md_lines))
print("MD:", md_path)
print(df.head().to_string(index=False))
