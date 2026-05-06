#!/usr/bin/env python
"""CLI for seed species selection from OrthoFinder data.

Usage:
    uv run python scripts/select_seeds.py --k 5 --plot
    uv run python scripts/select_seeds.py --k-range 2 15 --plot
    uv run python scripts/select_seeds.py --k 5 --no-filter-outliers  # skip outlier filtering
    uv run python scripts/select_seeds.py --seeds ARATH,ORYSA,PICAB,SELMO,MARPO  # manual seeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from orbit.seed_selection import (
    build_orthogroup_matrix,
    compute_ortholog_density,
    compute_species_distances,
    evaluate_seed_set,
    filter_outlier_species,
    select_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select seed species for SPACE alignment from OrthoFinder data"
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("data/orthofinder/transcripts_to_og"),
        help="Directory with {CODE}_transcripts_to_OG.tsv files",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of seed species")
    parser.add_argument(
        "--k-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Sweep k from MIN to MAX (overrides --k)",
    )
    parser.add_argument(
        "--min-shared-ogs",
        type=int,
        default=1000,
        help="Min shared orthogroups for seed pair validity",
    )
    parser.add_argument(
        "--no-filter-outliers",
        action="store_true",
        help="Disable automatic filtering of species with extreme OG counts",
    )
    parser.add_argument(
        "--max-iqr-factor",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    parser.add_argument(
        "--min-species",
        type=int,
        default=5,
        help="Only use OGs present in >= N species for distance computation (default: 5)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated species codes to use as seeds (bypasses p-dispersion)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for plots and results",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build matrices ---
    logger.info("Building orthogroup matrix...")
    og_matrix = build_orthogroup_matrix(args.transcripts_dir)

    logger.info("Computing species distances...")
    dist_matrix = compute_species_distances(og_matrix, min_species=args.min_species)

    logger.info("Computing ortholog density...")
    shared_ogs_matrix, pairs_matrix = compute_ortholog_density(args.transcripts_dir)

    # --- Manual seeds or automatic selection ---
    if args.seeds:
        # Manual seed mode: bypass outlier filtering and p-dispersion
        seeds = [s.strip() for s in args.seeds.split(",")]
        all_species = set(dist_matrix.index)
        missing = [s for s in seeds if s not in all_species]
        if missing:
            raise ValueError(f"Seeds not found in OrthoFinder data: {missing}")

        logger.info(f"Using {len(seeds)} manual seeds: {seeds}")
        metrics = evaluate_seed_set(seeds, dist_matrix, og_matrix, shared_ogs_matrix)

        # Print results
        print(f"\n{'='*60}")
        print(f"Manual seed species ({len(seeds)}):")
        print(f"{'='*60}")
        for i, s in enumerate(seeds, 1):
            print(f"  {i}. {s}")

        print(f"\nSeed set metrics:")
        print(f"  Min pairwise distance:    {metrics['seed_min_dist']:.4f}")
        print(f"  Mean pairwise distance:   {metrics['seed_mean_dist']:.4f}")
        print(f"  Min shared orthogroups:   {metrics['seed_min_shared_ogs']}")
        print(f"  Mean shared orthogroups:  {metrics['seed_mean_shared_ogs']}")
        print(f"  Max nonseed-to-seed dist: {metrics['nonseed_max_dist']:.4f}")
        print(f"  Mean nonseed-to-seed dist:{metrics['nonseed_mean_dist']:.4f}")
        print(f"  OG coverage:              {metrics['og_coverage']:.4f}")

        print(f"\nSpecies groupings (non-seed -> nearest seed):")
        groups = metrics["groups"]
        for seed in seeds:
            members = [sp for sp, s in groups.items() if s == seed]
            print(f"  {seed}: {len(members)} species")

        # Save results
        results_path = args.output_dir / "seed_selection.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved results to {results_path}")
    else:
        # --- Filter outliers ---
        candidates = None
        if not args.no_filter_outliers:
            candidates = filter_outlier_species(og_matrix, max_iqr_factor=args.max_iqr_factor)
            logger.info(f"{len(candidates)} candidate species after outlier filtering")

        # --- Select seeds ---
        def _select(k: int) -> list[str]:
            return select_seeds(
                dist_matrix, shared_ogs_matrix, k=k,
                min_shared_ogs=args.min_shared_ogs, candidates=candidates,
            )

        if args.k_range:
            k_min, k_max = args.k_range
            logger.info(f"Running k-sensitivity sweep from {k_min} to {k_max}")

            all_results = []
            for k in range(k_min, k_max + 1):
                try:
                    seeds = _select(k)
                    metrics = evaluate_seed_set(seeds, dist_matrix, og_matrix, shared_ogs_matrix)
                    all_results.append(metrics)

                    print(f"\n{'='*60}")
                    print(f"k={k}: {seeds}")
                    print(f"  Seed min dist:       {metrics['seed_min_dist']:.4f}")
                    print(f"  Seed mean dist:      {metrics['seed_mean_dist']:.4f}")
                    print(f"  Seed min shared OGs: {metrics['seed_min_shared_ogs']}")
                    print(f"  Max nonseed dist:    {metrics['nonseed_max_dist']:.4f}")
                    print(f"  OG coverage:         {metrics['og_coverage']:.4f}")
                except ValueError as e:
                    logger.warning(f"k={k}: {e}")

            # Save sweep results
            results_path = args.output_dir / "k_sweep_results.json"
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved sweep results to {results_path}")

            if args.plot:
                import matplotlib
                matplotlib.use("Agg")
                from orbit.visualize_seeds import plot_k_sensitivity

                plot_k_sensitivity(
                    dist_matrix,
                    og_matrix,
                    shared_ogs_matrix,
                    k_range=(k_min, k_max),
                    min_shared_ogs=args.min_shared_ogs,
                    candidates=candidates,
                    output_path=args.output_dir / "k_sensitivity.png",
                )
        else:
            seeds = _select(args.k)
            metrics = evaluate_seed_set(seeds, dist_matrix, og_matrix, shared_ogs_matrix)

            # Print results
            print(f"\n{'='*60}")
            print(f"Selected {args.k} seed species:")
            print(f"{'='*60}")
            for i, s in enumerate(seeds, 1):
                print(f"  {i}. {s}")

            print(f"\nSeed set metrics:")
            print(f"  Min pairwise distance:    {metrics['seed_min_dist']:.4f}")
            print(f"  Mean pairwise distance:   {metrics['seed_mean_dist']:.4f}")
            print(f"  Min shared orthogroups:   {metrics['seed_min_shared_ogs']}")
            print(f"  Mean shared orthogroups:  {metrics['seed_mean_shared_ogs']}")
            print(f"  Max nonseed-to-seed dist: {metrics['nonseed_max_dist']:.4f}")
            print(f"  Mean nonseed-to-seed dist:{metrics['nonseed_mean_dist']:.4f}")
            print(f"  OG coverage:              {metrics['og_coverage']:.4f}")

            print(f"\nSpecies groupings (non-seed -> nearest seed):")
            groups = metrics["groups"]
            for seed in seeds:
                members = [sp for sp, s in groups.items() if s == seed]
                print(f"  {seed}: {', '.join(sorted(members)) if members else '(no non-seeds)'}")

            # Save results
            results_path = args.output_dir / "seed_selection.json"
            with open(results_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved results to {results_path}")

    # --- Plots ---
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")

        from orbit.visualize_seeds import (
            plot_dendrogram,
            plot_ortholog_density_heatmap,
        )

        if not args.k_range:
            plot_dendrogram(
                dist_matrix, seeds, output_path=args.output_dir / "dendrogram.png"
            )
            plot_ortholog_density_heatmap(
                shared_ogs_matrix,
                dist_matrix,
                seeds,
                output_path=args.output_dir / "ortholog_density_heatmap.png",
            )

    plt_msg = " (with plots)" if args.plot else ""
    logger.info(f"Done{plt_msg}. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
