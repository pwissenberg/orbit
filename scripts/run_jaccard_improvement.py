#!/usr/bin/env python
"""Run the Jaccard-weighted SPACE improvement pipeline.

1. Build Jaccard-weighted ortholog pairs (replaces vanilla 1/(nA*nB) weights)
2. Re-run SPACE alignment with Jaccard pairs
3. Evaluate and compare with vanilla

Usage:
    uv run python scripts/run_jaccard_improvement.py --stage pairs
    uv run python scripts/run_jaccard_improvement.py --stage align --workers 8 --device cuda
    uv run python scripts/run_jaccard_improvement.py --stage eval
    uv run python scripts/run_jaccard_improvement.py --stage all --workers 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations, product
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

NODE2VEC_DIR = ROOT / "data" / "node2vec"
NETWORK_DIR = ROOT / "data" / "networks"
TRANSCRIPTS_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"
SPACE_CONFIG_DIR = ROOT / "data"

# Jaccard-specific paths
JACCARD_ORTHOLOGS_DIR = ROOT / "data" / "orthologs_jaccard"
JACCARD_ALIGNED_DIR = ROOT / "results" / "aligned_embeddings_jaccard"


def load_seed_info() -> dict:
    with open(SEED_RESULTS) as f:
        return json.load(f)


def _build_one_pair(args):
    """Worker function for parallel pair building (must be top-level for pickling)."""
    sp_a, sp_b, h5_dir, transcripts_dir, network_dir, out_path = args
    from orbit.data_prep import build_jaccard_ortholog_pairs

    try:
        n = build_jaccard_ortholog_pairs(
            sp_a, sp_b, Path(h5_dir), Path(transcripts_dir), Path(network_dir), Path(out_path)
        )
        return sp_a, sp_b, n, None
    except Exception as e:
        return sp_a, sp_b, 0, str(e)


def build_jaccard_pairs(workers: int = 64):
    """Build Jaccard-weighted ortholog pairs for all species pairs."""
    seed_info = load_seed_info()
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]

    # Collect all work items (skip existing)
    work_items = []

    # Seed-seed pairs
    seed_dir = JACCARD_ORTHOLOGS_DIR / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for sp_a, sp_b in combinations(sorted(seeds), 2):
        out_path = seed_dir / f"{sp_a}_{sp_b}.tsv"
        if out_path.exists():
            logger.info(f"  {sp_a}_{sp_b}: already exists, skipping")
            continue
        work_items.append((sp_a, sp_b, str(NODE2VEC_DIR), str(TRANSCRIPTS_DIR), str(NETWORK_DIR), str(out_path)))

    # Non-seed pairs (each nonseed vs all 5 seeds)
    for ns in sorted(groups.keys()):
        ns_dir = JACCARD_ORTHOLOGS_DIR / "non_seeds" / ns
        ns_dir.mkdir(parents=True, exist_ok=True)
        for seed in sorted(seeds):
            sp_a, sp_b = sorted([ns, seed])
            out_path = ns_dir / f"{sp_a}_{sp_b}.tsv"
            if out_path.exists():
                continue
            work_items.append((sp_a, sp_b, str(NODE2VEC_DIR), str(TRANSCRIPTS_DIR), str(NETWORK_DIR), str(out_path)))

    if not work_items:
        logger.info("All Jaccard pairs already exist, nothing to do")
        return

    logger.info(f"Building {len(work_items)} Jaccard pairs with {workers} workers")
    t0 = time.perf_counter()

    if workers <= 1:
        # Serial fallback
        from orbit.data_prep import build_jaccard_ortholog_pairs
        for sp_a, sp_b, h5_dir, tdir, ndir, out_path in work_items:
            build_jaccard_ortholog_pairs(sp_a, sp_b, Path(h5_dir), Path(tdir), Path(ndir), Path(out_path))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        done, failed = 0, []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_build_one_pair, item): item for item in work_items}
            for future in as_completed(futures):
                sp_a, sp_b, n, err = future.result()
                done += 1
                if err:
                    logger.error(f"  {sp_a}_{sp_b}: FAILED — {err}")
                    failed.append(f"{sp_a}_{sp_b}")
                else:
                    if done % 50 == 0 or done == len(work_items):
                        elapsed = time.perf_counter() - t0
                        logger.info(f"  [{done}/{len(work_items)}] pairs done ({elapsed:.1f}s)")

        if failed:
            logger.warning(f"Failed pairs ({len(failed)}): {failed}")

    elapsed = time.perf_counter() - t0
    logger.info(f"Pair building complete: {len(work_items)} pairs in {elapsed:.1f}s ({elapsed/60:.1f}min)")


def run_alignment(device: str, workers: int, epochs: int, delta: float):
    """Run SPACE alignment with Jaccard-weighted pairs."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))

    import orbit._compat  # noqa: F401
    from run_alignment import (
        _patch_space_for_string_ids,
        run_seed_alignment,
        run_nonseed_alignment,
    )

    _patch_space_for_string_ids()

    extra = {
        "epochs": epochs,
        "delta": delta,
        "patience": 5,
    }

    t0 = time.perf_counter()
    run_seed_alignment(
        device=device,
        orthologs_dir=JACCARD_ORTHOLOGS_DIR,
        aligned_dir=JACCARD_ALIGNED_DIR,
        **extra,
    )
    seed_time = time.perf_counter() - t0
    logger.info(f"Seed alignment (FedCoder): {seed_time:.1f}s ({seed_time/60:.1f}min)")

    t0 = time.perf_counter()
    run_nonseed_alignment(
        device=device,
        workers=workers,
        orthologs_dir=JACCARD_ORTHOLOGS_DIR,
        aligned_dir=JACCARD_ALIGNED_DIR,
        **extra,
    )
    nonseed_time = time.perf_counter() - t0
    logger.info(f"Non-seed alignment: {nonseed_time:.1f}s ({nonseed_time/60:.1f}min)")


def run_evaluation(*, use_wandb: bool = False, epochs: int = 200, delta: float = 0.01):
    """Evaluate the Jaccard-weighted alignment and compare with vanilla."""
    seed_info = load_seed_info()
    seeds = seed_info["seeds"]

    if use_wandb:
        from orbit.tracking import init_run
        init_run("jaccard", {
            "epochs": epochs,
            "delta": delta,
            "k": 50,
            "top_m": 10,
            "eval_mode": "seeds",
            "seeds": seeds,
        })

    from orbit.evaluate import evaluate_all_pairs

    logger.info("Evaluating Jaccard-weighted alignment (seed pairs)...")
    results_jaccard = evaluate_all_pairs(
        aligned_dir=JACCARD_ALIGNED_DIR,
        network_dir=NETWORK_DIR,
        transcripts_dir=TRANSCRIPTS_DIR,
        seed_results_path=SEED_RESULTS,
        k=50,
        top_m=10,
        eval_mode="seeds",
    )

    out_path = ROOT / "results" / "evaluation_jaccard.json"
    with open(out_path, "w") as f:
        json.dump(results_jaccard, f, indent=2)

    # Load vanilla results for comparison
    vanilla_path = ROOT / "results" / "evaluation_vanilla.json"
    results_vanilla = []
    if vanilla_path.exists():
        with open(vanilla_path) as f:
            results_vanilla = json.load(f)

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Pair':>15s}  {'Vanilla H@50':>12s}  {'Jaccard H@50':>12s}  "
          f"{'Vanilla Spear':>13s}  {'Jaccard Spear':>13s}  {'Improvement':>11s}")
    print("-" * 100)

    vanilla_map = {
        f"{r['species_a']}-{r['species_b']}": r
        for r in results_vanilla if "error" not in r
    }

    for r in results_jaccard:
        if "error" in r:
            continue
        pair = f"{r['species_a']}-{r['species_b']}"
        v = vanilla_map.get(pair, {})
        v_hits = v.get("hits_at_k", 0)
        j_hits = r["hits_at_k"]
        v_rho = v.get("spearman_rho", 0)
        j_rho = r["spearman_rho"]
        improvement = j_hits / v_hits if v_hits > 0 else float("inf")
        print(
            f"{pair:>15s}  {v_hits:12.4f}  {j_hits:12.4f}  "
            f"{v_rho:13.4f}  {j_rho:13.4f}  {improvement:10.1f}x"
        )

    print("=" * 100)

    valid_j = [r for r in results_jaccard if "error" not in r]
    valid_v = [r for r in results_vanilla if "error" not in r]
    if valid_j and valid_v:
        avg_v = sum(r["hits_at_k"] for r in valid_v) / len(valid_v)
        avg_j = sum(r["hits_at_k"] for r in valid_j) / len(valid_j)
        avg_v_rho = sum(r["spearman_rho"] for r in valid_v) / len(valid_v)
        avg_j_rho = sum(r["spearman_rho"] for r in valid_j) / len(valid_j)
        print(
            f"{'AVERAGE':>15s}  {avg_v:12.4f}  {avg_j:12.4f}  "
            f"{avg_v_rho:13.4f}  {avg_j_rho:13.4f}  {avg_j/avg_v:.1f}x"
        )

    # Within-species evaluation
    from orbit.evaluate import evaluate_all_within_species

    logger.info("Running within-species evaluation...")
    within_results = evaluate_all_within_species(
        aligned_dir=JACCARD_ALIGNED_DIR,
        network_dir=NETWORK_DIR,
        k=50,
    )

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
    if use_wandb:
        from orbit.tracking import log_evaluation, log_plots, log_within_species, finish_run
        out_path = ROOT / "results" / "evaluation_jaccard.json"
        log_evaluation(results_jaccard, output_path=out_path)
        log_within_species(within_results)
        log_plots(ROOT / "results" / "plots")
        finish_run()


def main():
    parser = argparse.ArgumentParser(description="Run Jaccard improvement pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pairs", "align", "eval", "all"],
        help="Which stage to run",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pair-workers", type=int, default=64,
                        help="Number of parallel workers for pair building (default 64)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    timings = {}
    pipeline_start = time.perf_counter()

    if args.stage in ("pairs", "all"):
        t0 = time.perf_counter()
        build_jaccard_pairs(workers=args.pair_workers)
        timings["pair_building"] = time.perf_counter() - t0

    if args.stage in ("align", "all"):
        t0 = time.perf_counter()
        run_alignment(args.device, args.workers, args.epochs, args.delta)
        timings["alignment_total"] = time.perf_counter() - t0

    if args.stage in ("eval", "all"):
        t0 = time.perf_counter()
        run_evaluation(
            use_wandb=not args.no_wandb,
            epochs=args.epochs,
            delta=args.delta,
        )
        timings["evaluation"] = time.perf_counter() - t0

    timings["pipeline_total"] = time.perf_counter() - pipeline_start

    # Print timing summary
    logger.info("=" * 60)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 60)
    for step, secs in timings.items():
        mins = secs / 60
        hrs = secs / 3600
        if secs < 120:
            logger.info(f"  {step:.<30s} {secs:>8.1f}s")
        elif secs < 7200:
            logger.info(f"  {step:.<30s} {mins:>8.1f}min ({secs:.0f}s)")
        else:
            logger.info(f"  {step:.<30s} {hrs:>8.2f}h ({mins:.1f}min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
