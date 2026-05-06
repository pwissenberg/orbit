#!/usr/bin/env python
"""Evaluate vanilla SPACE alignment quality.

Usage:
    uv run python scripts/evaluate.py --mode seeds
    uv run python scripts/evaluate.py --mode all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

ALIGNED_DIR = ROOT / "results" / "aligned_embeddings"
NETWORK_DIR = ROOT / "data" / "networks"
TRANSCRIPTS_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"
OUTPUT_DIR = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPACE alignment")
    parser.add_argument(
        "--mode",
        default="seeds",
        choices=["seeds", "all"],
        help="Evaluate seed-seed pairs only, or all pairs",
    )
    parser.add_argument("--k", type=int, default=50, help="K for Hits@K and MRR@K")
    parser.add_argument("--top-m", type=int, default=10, help="TopM ground truth size")
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ALIGNED_DIR,
        help="Directory with aligned H5 embeddings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "evaluation_vanilla.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(n_pairs, 32)). Set to 1 for sequential.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    # Load seed info for config
    with open(SEED_RESULTS) as f:
        seed_info = json.load(f)
    seeds = seed_info["seeds"]

    # Initialize wandb tracking
    if not args.no_wandb:
        from orbit.tracking import init_run
        init_run("vanilla", {
            "k": args.k,
            "top_m": args.top_m,
            "eval_mode": args.mode,
            "seeds": seeds,
        })

    from orbit.evaluate import evaluate_all_pairs

    results = evaluate_all_pairs(
        aligned_dir=args.aligned_dir,
        network_dir=NETWORK_DIR,
        transcripts_dir=TRANSCRIPTS_DIR,
        seed_results_path=SEED_RESULTS,
        k=args.k,
        top_m=args.top_m,
        eval_mode=args.mode,
        n_workers=args.workers,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Pair':>15s}  {'Type':>12s}  {'Hits@K':>8s}  {'MRR@K':>8s}  "
          f"{'TopM-H@K':>8s}  {'Spearman':>8s}  {'Shuf-H@K':>8s}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['species_a']}-{r['species_b']:>15s}  {'ERROR':>12s}  {r['error']}")
            continue
        pair = f"{r['species_a']}-{r['species_b']}"
        print(
            f"{pair:>15s}  {r['pair_type']:>12s}  "
            f"{r['hits_at_k']:8.4f}  {r['mrr_at_k']:8.4f}  "
            f"{r['top_m_hits_at_k']:8.4f}  {r['spearman_rho']:8.4f}  "
            f"{r['shuffle_hits_at_k']:8.4f}"
        )
    print("=" * 90)

    # Averages
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_hits = sum(r["hits_at_k"] for r in valid) / len(valid)
        avg_mrr = sum(r["mrr_at_k"] for r in valid) / len(valid)
        avg_rho = sum(r["spearman_rho"] for r in valid) / len(valid)
        avg_shuf = sum(r["shuffle_hits_at_k"] for r in valid) / len(valid)
        print(
            f"{'AVERAGE':>15s}  {'':>12s}  "
            f"{avg_hits:8.4f}  {avg_mrr:8.4f}  "
            f"{'':>8s}  {avg_rho:8.4f}  {avg_shuf:8.4f}"
        )

    # Within-species evaluation
    from orbit.evaluate import evaluate_all_within_species

    logger.info("Running within-species evaluation...")
    within_results = evaluate_all_within_species(
        aligned_dir=args.aligned_dir,
        network_dir=NETWORK_DIR,
        k=args.k,
        n_workers=args.workers,
    )

    # Print within-species summary
    valid_within = [r for r in within_results if "error" not in r]
    if valid_within:
        print("\n" + "=" * 80)
        print(f"{'Species':>10s}  {'P@K':>8s}  {'R@K':>8s}  "
              f"{'Shuf P@K':>8s}  {'Shuf R@K':>8s}  {'n_eval':>8s}")
        print("-" * 80)
        for r in valid_within:
            print(
                f"{r['species']:>10s}  {r['precision_at_k']:8.4f}  {r['recall_at_k']:8.4f}  "
                f"{r['shuffle_precision_mean']:8.4f}  {r['shuffle_recall_mean']:8.4f}  "
                f"{r['n_eval']:>8d}"
            )
        avg_p = sum(r["precision_at_k"] for r in valid_within) / len(valid_within)
        avg_r = sum(r["recall_at_k"] for r in valid_within) / len(valid_within)
        print("-" * 80)
        print(f"{'AVERAGE':>10s}  {avg_p:8.4f}  {avg_r:8.4f}")
        print("=" * 80)

    # Log to wandb
    if not args.no_wandb:
        from orbit.tracking import log_evaluation, log_plots, log_within_species, finish_run
        log_evaluation(results, output_path=args.output)
        log_within_species(within_results)
        log_plots(OUTPUT_DIR / "plots")
        finish_run()


if __name__ == "__main__":
    main()
