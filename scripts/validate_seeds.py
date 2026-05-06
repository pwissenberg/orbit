#!/usr/bin/env python3
"""Validate seed species quality for SPACE alignment pipeline.

Runs three validation checks on each seed species:
  1. Node2Vec stability (embedding variance across multiple runs)
  2. Network quality (density and degree metrics)
  3. Ortholog coverage (OrthoFinder assignment coverage)

Outputs validation reports to results/validation/ directory.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from orbit.seed_validation import (
    measure_node2vec_stability,
    compute_network_quality,
    check_ortholog_coverage,
    write_validation_report,
)


def load_seed_species(seed_selection_path: Path) -> list[str]:
    """Load seed species from seed_selection.json."""
    with open(seed_selection_path) as fh:
        data = json.load(fh)
    return data["seeds"]


def main():
    parser = argparse.ArgumentParser(
        description="Validate seed species for SPACE alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--network-dir",
        type=Path,
        default=Path("data/networks"),
        help="Directory containing cleaned network TSV files (default: data/networks)",
    )
    parser.add_argument(
        "--network-space-dir",
        type=Path,
        default=Path("data/networks_space"),
        help="Directory containing SPACE-format gzipped network files (default: data/networks_space)",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("data/orthofinder/transcripts_to_og"),
        help="Directory containing OrthoFinder transcripts (default: data/orthofinder/transcripts_to_og)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/validation"),
        help="Output directory for validation reports (default: results/validation)",
    )
    parser.add_argument(
        "--seed-selection",
        type=Path,
        default=Path("results/seed_selection.json"),
        help="Path to seed selection JSON (default: results/seed_selection.json)",
    )
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip Node2Vec stability test (faster, useful for quick checks)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load seed species
    try:
        seeds = load_seed_species(args.seed_selection)
        logger.info(f"Loaded {len(seeds)} seed species: {', '.join(seeds)}")
    except Exception as e:
        logger.error(f"Failed to load seed species: {e}")
        return 2

    # Get timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    # Initialize result containers
    stability_results = {}
    quality_results = {}

    # Run validation suite for each seed
    for seed in seeds:
        logger.info(f"\nValidating {seed}...")

        network_path = args.network_dir / f"{seed}.tsv"
        network_gz_path = args.network_space_dir / f"{seed}.txt.gz"

        # Check file existence
        if not network_path.exists():
            logger.error(f"  Network file not found: {network_path}")
            continue
        if not args.skip_stability and not network_gz_path.exists():
            logger.error(f"  SPACE network file not found: {network_gz_path}")
            continue

        # A. Node2Vec stability (unless --skip-stability)
        if not args.skip_stability:
            logger.info(f"  Running Node2Vec stability test (this may take several minutes)...")
            stability_result = measure_node2vec_stability(
                species=seed,
                network_gz=network_gz_path,
                network_dir=args.network_dir,
                output_dir=args.output_dir,
                n_runs=5,
                dimensions=128,
            )
            stability_results[seed] = stability_result
        else:
            logger.info(f"  Skipping Node2Vec stability test (--skip-stability)")

        # B. Network quality
        logger.info(f"  Checking network quality...")
        quality_result_network = compute_network_quality(
            species=seed,
            network_path=network_path,
        )

        # C. Ortholog coverage
        logger.info(f"  Checking ortholog coverage...")
        quality_result_ortholog = check_ortholog_coverage(
            species=seed,
            transcripts_dir=args.transcripts_dir,
            network_path=network_path,
        )

        # Combine quality results
        quality_results[seed] = {
            "species": seed,
            "network": quality_result_network,
            "ortholog": quality_result_ortholog,
            "pass": quality_result_network["pass"] and quality_result_ortholog["pass"],
        }

    # Write validation reports
    logger.info("\nWriting validation reports...")

    if not args.skip_stability:
        write_validation_report(
            output_path=args.output_dir / "seed_stability.json",
            phase="01-data-preparation-seed-validation",
            validation_type="stability",
            results=stability_results,
            timestamp=timestamp,
        )

    # Write quality report (network + ortholog combined)
    quality_summary = {}
    for seed, result in quality_results.items():
        quality_summary[seed] = {
            "species": seed,
            "network_quality": result["network"],
            "ortholog_coverage": result["ortholog"],
            "pass": result["pass"],
        }

    passed_quality = sum(1 for r in quality_results.values() if r["pass"])
    failed_quality = len(quality_results) - passed_quality
    quality_status = "PASS" if failed_quality == 0 else "FAIL"

    quality_report = {
        "phase": "01-data-preparation-seed-validation",
        "validation_type": "quality",
        "timestamp": timestamp,
        "status": quality_status,
        "summary": {
            "total_seeds": len(quality_results),
            "passed": passed_quality,
            "failed": failed_quality,
        },
        "thresholds": {
            "density_min": 0.001,
            "density_max": 0.5,
            "ortholog_coverage": 0.8,
        },
        "results": quality_summary,
    }

    quality_report_path = args.output_dir / "seed_quality.json"
    with open(quality_report_path, "w") as fh:
        json.dump(quality_report, fh, indent=2)
    logger.info(f"  Wrote quality report: {quality_report_path}")

    # Terminal summary
    logger.info("\n" + "=" * 70)
    if not args.skip_stability:
        passed_stability = sum(1 for r in stability_results.values() if r.get("pass", False))
        failed_stability = len(stability_results) - passed_stability
        stability_status = "PASS" if failed_stability == 0 else "FAIL"
        logger.info(f"Node2Vec Stability: {stability_status} ({passed_stability}/{len(seeds)} seeds passed)")

    logger.info(f"Network Quality + Ortholog Coverage: {quality_status} ({passed_quality}/{len(seeds)} seeds passed)")

    overall_pass = (args.skip_stability or failed_stability == 0) and failed_quality == 0
    overall_status = "PASS" if overall_pass else "FAIL"
    logger.info(f"\nSeed validation: {overall_status}")
    logger.info("=" * 70)

    # Exit codes
    if overall_pass:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
