"""Generate evaluation comparison plots (vanilla vs jaccard vs hybrid).

Usage:
    uv run python scripts/visualize_evaluation.py                     # all plots
    uv run python scripts/visualize_evaluation.py --plots spearman    # specific
    uv run python scripts/visualize_evaluation.py --output-dir results/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from orbit.visualize_evaluation import (
    plot_aggregate_summary,
    plot_pairwise_scatter,
    plot_spearman_comparison,
)


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/plots"),
        help="Directory to save plots (default: results/plots)",
    )
    parser.add_argument(
        "--plots", nargs="*",
        choices=["spearman", "aggregate", "scatter"],
        default=["spearman", "aggregate", "scatter"],
        help="Which plots to generate (default: all)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory containing evaluation JSON files",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Suffix for evaluation files, e.g. '_all' for evaluation_vanilla_all.json",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix

    vanilla = load_results(args.results_dir / f"evaluation_vanilla{sfx}.json")
    jaccard = load_results(args.results_dir / f"evaluation_jaccard{sfx}.json")
    hybrid = load_results(args.results_dir / f"evaluation_hybrid{sfx}.json")

    if "spearman" in args.plots:
        out = args.output_dir / f"spearman_comparison{sfx}.png"
        fig = plot_spearman_comparison(vanilla, jaccard, hybrid, output_path=out)
        print(f"Saved: {out}")
        plt.close(fig)

    if "aggregate" in args.plots:
        out = args.output_dir / f"aggregate_summary{sfx}.png"
        fig = plot_aggregate_summary(vanilla, jaccard, hybrid, output_path=out)
        print(f"Saved: {out}")
        plt.close(fig)

    if "scatter" in args.plots:
        out = args.output_dir / f"pairwise_scatter{sfx}.png"
        fig = plot_pairwise_scatter(vanilla, jaccard, output_path=out)
        print(f"Saved: {out}")
        plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
