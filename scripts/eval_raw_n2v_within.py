#!/usr/bin/env python
"""Within-species evaluation on RAW (unaligned) Node2Vec embeddings.

Establishes a ceiling for how much within-species coexpression-network
structure is preserved before any cross-species alignment is applied.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

# numpy/nptyping compatibility shim — must be imported before SPACE-related modules
from orbit import _compat  # noqa: F401
from orbit.evaluate import evaluate_all_within_species

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_raw_n2v_within")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    n2v_dir = project_root / "data" / "node2vec"
    network_dir = project_root / "data" / "networks"
    out_path = project_root / "results" / "evaluation_raw_n2v_within.json"

    assert n2v_dir.exists(), f"missing {n2v_dir}"
    assert network_dir.exists(), f"missing {network_dir}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_files = len(list(n2v_dir.glob("*.h5")))
    logger.info(f"Found {n_files} N2V H5 files in {n2v_dir}")
    logger.info(f"Networks dir: {network_dir}")
    logger.info(f"Output: {out_path}")

    n_workers = min(48, os.cpu_count() or 32)
    logger.info(f"Using {n_workers} workers, k=50")

    t0 = time.time()
    results = evaluate_all_within_species(
        aligned_dir=n2v_dir,   # function only assumes <species>.h5 files; raw N2V layout matches
        network_dir=network_dir,
        k=50,
        n_workers=n_workers,
    )
    dt = time.time() - t0

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    if ok:
        mean_p = sum(r["precision_at_k"] for r in ok) / len(ok)
        mean_r = sum(r["recall_at_k"] for r in ok) / len(ok)
        mean_sp = sum(r["shuffle_precision_mean"] for r in ok) / len(ok)
        mean_sr = sum(r["shuffle_recall_mean"] for r in ok) / len(ok)
        logger.info(
            f"DONE in {dt/60:.1f} min  |  n_species={len(ok)} ok / {len(err)} err  |  "
            f"mean P@50={mean_p:.4f}  R@50={mean_r:.4f}  "
            f"(shuffle P={mean_sp:.4f}  R={mean_sr:.4f})"
        )
    else:
        logger.error("No successful evals")
    if err:
        logger.warning(f"{len(err)} species errored:")
        for r in err[:10]:
            logger.warning(f"  {r.get('species', '?')}: {r.get('error')}")


if __name__ == "__main__":
    main()
