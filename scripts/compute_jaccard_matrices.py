#!/usr/bin/env python3
"""Compute Jaccard similarity matrices for all species in the dataset.

Processes coexpression networks in data/networks/ and saves sparse .npz files.
Uses the same torch.multiprocessing + round-robin GPU pattern as run_alignment.py.

Hardware on this machine
  GPUs   : 4 × NVIDIA L4  (23 GB VRAM each)
  Cores  : 48
  RAM    : 188 GB

Sensible defaults (auto-detected at runtime):
  GPU mode  → 16 workers, round-robin across 4 GPUs (4 workers / GPU)
  CPU mode  → 40 workers (leaves ~8 cores for OS / IO on 48-core machine)

Usage examples:
    # Recommended: GPU mode, 4 GPUs × 4 workers = 16 parallel species
    uv run python scripts/compute_jaccard_matrices.py

    # CPU only, 40 workers
    uv run python scripts/compute_jaccard_matrices.py --method cpu --workers 40

    # Override GPU worker count (more workers = more GPU memory pressure)
    uv run python scripts/compute_jaccard_matrices.py --method gpu --workers 8

    # Specific species only
    uv run python scripts/compute_jaccard_matrices.py --species TRIAE BRANA

    # Recompute everything (ignore existing .npz files)
    uv run python scripts/compute_jaccard_matrices.py --recompute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from orbit.jaccard_computation import Method, compute_jaccard_matrix


def _setup_logger() -> None:
    """Route loguru through tqdm.write so the progress bar isn't overwritten."""
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )


# ---------------------------------------------------------------------------
# Worker — module-level so both mp.Pool and ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------

def _worker(
    args: tuple[Path, Path, Method, bool, int | None],
) -> tuple[str, dict[str, Any] | None, str]:
    """Process one species.

    args = (network_path, output_path, method, skip_existing, gpu_id)
    gpu_id is 0-3 for GPU workers, None for CPU workers.

    Returns (species_code, metadata | None, status_string).
    """
    network_path, output_path, method, skip_existing, gpu_id = args
    species = network_path.stem

    if skip_existing and output_path.exists():
        return species, None, "skipped"

    if not network_path.exists():
        return species, None, "not_found"

    # Pin this worker to its assigned GPU (mirrors run_alignment.py device routing)
    effective_method = method
    if gpu_id is not None:
        try:
            import cupy as cp
            cp.cuda.Device(gpu_id).use()
        except Exception:  # noqa: BLE001
            # GPU VRAM exhausted — fall back to CPU for this species
            effective_method = "cpu"
            gpu_id = None

    t0 = time.perf_counter()
    try:
        metadata = compute_jaccard_matrix(network_path, output_path, method=effective_method)
        metadata["wall_seconds"] = round(time.perf_counter() - t0, 1)
        metadata["gpu_id"] = gpu_id
        return species, metadata, "ok"
    except Exception as exc:  # noqa: BLE001
        return species, None, f"error: {exc}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _detect_gpus() -> int:
    """Return number of available CUDA GPUs (0 if none / torch not available)."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def _default_workers(method: str) -> int:
    """Sensible default worker count based on method and hardware."""
    n_gpus = _detect_gpus()
    if method != "cpu" and n_gpus > 0:
        return n_gpus * 4          # 4 workers per GPU  →  16 on this machine
    return min(40, os.cpu_count() or 1)   # leave ~8 cores headroom on 48-core machine


def parse_args() -> argparse.Namespace:
    # Parse method first so default workers can reference it
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", default="auto")
    pre_args, _ = pre.parse_known_args()
    default_w = _default_workers(pre_args.method)

    parser = argparse.ArgumentParser(
        description="Compute Jaccard similarity matrices for plant species",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=Path("data/networks"),
        help="Directory containing network TSV files (default: data/networks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/procrustes_jaccard"),
        help="Directory to save Jaccard matrices (default: data/procrustes_jaccard)",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Specific species codes to process (default: all .tsv files)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help=(
            "'auto' uses GPU if cupy is installed, else CPU (default); "
            "'cpu' forces scipy sparse matmul; "
            "'gpu' forces cupy (requires cupy-cuda12x)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_w,
        help=(
            f"Parallel workers (default: {default_w} — "
            "4 × n_gpus in GPU mode, min(40, cpu_count) in CPU mode). "
            "Use 1 to disable multiprocessing."
        ),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute even if output .npz already exists (default: skip existing)",
    )
    return parser.parse_args()


def _get_species_list(network_dir: Path, species_filter: list[str] | None) -> list[str]:
    if species_filter:
        return sorted(species_filter)
    return sorted(p.stem for p in network_dir.glob("*.tsv"))


def main() -> int:
    _setup_logger()
    args = parse_args()

    if not args.network_dir.exists():
        logger.error(f"Network directory not found: {args.network_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    species_list = _get_species_list(args.network_dir, args.species)
    if not species_list:
        logger.error(f"No species found in {args.network_dir}")
        return 1

    # --- Detect hardware and resolve GPU routing ----------------------------
    n_gpus = _detect_gpus()
    using_gpu = args.method != "cpu" and n_gpus > 0

    if using_gpu:
        gpu_ids = list(range(n_gpus))
        logger.info(f"Multi-GPU mode: {n_gpus} GPUs, {args.workers} workers total")
        logger.info(f"  Round-robin: worker i → GPU i % {n_gpus}")
    else:
        gpu_ids = [None]
        logger.info(f"CPU mode: {args.workers} workers")

    skip_existing = not args.recompute
    n_existing = (
        sum(1 for sp in species_list if (args.output_dir / f"{sp}.npz").exists())
        if skip_existing else 0
    )

    logger.info(f"Species total : {len(species_list)}")
    logger.info(f"Already done  : {n_existing} (will skip)")
    logger.info(f"To process    : {len(species_list) - n_existing}")
    logger.info(f"Backend       : {args.method}")
    logger.info(f"Workers       : {args.workers}")

    # Build work items — round-robin GPU assignment (same as run_alignment.py)
    work_items = [
        (
            args.network_dir / f"{sp}.tsv",
            args.output_dir / f"{sp}.npz",
            args.method,
            skip_existing,
            gpu_ids[i % len(gpu_ids)],
        )
        for i, sp in enumerate(species_list)
    ]

    results: dict[str, Any] = {}
    failures: list[str] = []
    skipped = 0
    total_size_mb = 0.0
    t_start = time.perf_counter()

    bar = tqdm(
        total=len(work_items),
        initial=n_existing,   # start at already-done count, not 0
        unit="sp",
        desc="Jaccard",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    def _handle_result(sp: str, metadata, status: str) -> None:
        nonlocal skipped, total_size_mb
        if status == "skipped":
            skipped += 1
            # Already counted in initial=n_existing — don't update bar again
        elif status == "ok":
            results[sp] = metadata
            total_size_mb += metadata["size_mb"]
            gpu_tag = (
                f"GPU:{metadata['gpu_id']}"
                if metadata.get("gpu_id") is not None
                else metadata.get("backend", "cpu")
            )
            bar.set_postfix_str(
                f"{sp} {metadata['wall_seconds']}s [{gpu_tag}]", refresh=True
            )
            bar.update(1)
            logger.info(
                f"{sp} done — {metadata['wall_seconds']}s, "
                f"{metadata['n_genes']} genes, {gpu_tag}"
            )
        else:
            bar.update(1)
            logger.error(f"{sp}: {status}")
            failures.append(sp)

    if args.workers == 1:
        for item in work_items:
            sp, metadata, status = _worker(item)
            _handle_result(sp, metadata, status)

    elif using_gpu:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        with mp.Pool(processes=args.workers) as pool:
            for sp, metadata, status in pool.imap_unordered(_worker, work_items):
                _handle_result(sp, metadata, status)

    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, item): item[0].stem for item in work_items}
            for future in as_completed(futures):
                sp, metadata, status = future.result()
                _handle_result(sp, metadata, status)

    bar.close()

    # Save summary JSON
    summary_path = args.output_dir / "jaccard_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)

    wall_total = time.perf_counter() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Completed : {len(results)}")
    logger.info(f"Skipped   : {skipped}")
    logger.info(f"Failed    : {len(failures)}")
    logger.info(f"Output    : {total_size_mb:.1f} MB total")
    logger.info(f"Wall time : {wall_total:.1f}s")

    if failures:
        logger.warning(f"Failed species: {', '.join(failures)}")
        return 1

    logger.info("All species processed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
