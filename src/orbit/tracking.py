"""Weights & Biases experiment tracking for ORBIT.

Lightweight wrapper around wandb for logging alignment evaluation results.
All functions are no-ops if wandb is disabled (--no-wandb flag).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_run = None


def init_run(method: str, config: dict) -> None:
    """Initialize a wandb run.

    Args:
        method: Alignment method name (vanilla, jaccard, hybrid).
        config: Experiment configuration dict (delta, epochs, k, etc.).
    """
    global _run
    import wandb

    _run = wandb.init(
        project="orbit",
        entity="chl383-copenhagen-university",
        name=f"{method}-eval",
        config={"method": method, **config},
        tags=[method],
    )
    logger.info(f"wandb run initialized: {_run.url}")


def log_evaluation(results: list[dict], output_path: Path | None = None) -> None:
    """Log evaluation results to the active wandb run.

    Args:
        results: List of per-pair evaluation dicts from evaluate_all_pairs().
        output_path: Path to evaluation JSON file to log as artifact.
    """
    if _run is None:
        return

    import wandb

    valid = [r for r in results if "error" not in r]
    if not valid:
        logger.warning("No valid results to log")
        return

    # --- Per-pair table ---
    columns = [
        "species_a", "species_b", "pair_type",
        "hits_at_k", "mrr_at_k", "top_m_hits_at_k",
        "spearman_rho", "shuffle_hits_at_k",
    ]
    table = wandb.Table(columns=columns)
    for r in valid:
        table.add_data(*[r.get(c, "") for c in columns])
    _run.log({"evaluation_table": table})

    # --- Summary metrics ---
    avg = lambda key: sum(r[key] for r in valid) / len(valid)
    _run.summary["avg_hits_at_k"] = avg("hits_at_k")
    _run.summary["avg_mrr_at_k"] = avg("mrr_at_k")
    _run.summary["avg_spearman"] = avg("spearman_rho")
    _run.summary["avg_shuffle_hits_at_k"] = avg("shuffle_hits_at_k")
    _run.summary["n_pairs"] = len(valid)
    _run.summary["n_errors"] = len(results) - len(valid)

    best = max(valid, key=lambda r: r["spearman_rho"])
    worst = min(valid, key=lambda r: r["spearman_rho"])
    _run.summary["best_pair"] = f"{best['species_a']}-{best['species_b']}"
    _run.summary["best_spearman"] = best["spearman_rho"]
    _run.summary["worst_pair"] = f"{worst['species_a']}-{worst['species_b']}"
    _run.summary["worst_spearman"] = worst["spearman_rho"]

    # --- Artifact: evaluation JSON ---
    if output_path and Path(output_path).exists():
        artifact = wandb.Artifact(
            name=f"evaluation-{_run.config.get('method', 'unknown')}",
            type="evaluation",
        )
        artifact.add_file(str(output_path))
        _run.log_artifact(artifact)

    logger.info(
        f"Logged {len(valid)} pairs to wandb | "
        f"avg Spearman={avg('spearman_rho'):.4f}, avg H@K={avg('hits_at_k'):.4f}"
    )


def log_within_species(results: list[dict]) -> None:
    """Log within-species evaluation results to the active wandb run.

    Args:
        results: List of per-species dicts from evaluate_all_within_species().
    """
    if _run is None:
        return

    import wandb

    valid = [r for r in results if "error" not in r]
    if not valid:
        logger.warning("No valid within-species results to log")
        return

    columns = [
        "species", "precision_at_k", "recall_at_k",
        "n_eval", "n_genes", "shuffle_precision_mean", "shuffle_recall_mean", "k",
    ]
    table = wandb.Table(columns=columns)
    for r in valid:
        table.add_data(*[r.get(c, "") for c in columns])
    _run.log({"within_species_table": table})

    avg = lambda key: sum(r[key] for r in valid) / len(valid)
    _run.summary["avg_precision_at_k"] = avg("precision_at_k")
    _run.summary["avg_recall_at_k"] = avg("recall_at_k")
    _run.summary["n_within_species"] = len(valid)

    logger.info(
        f"Logged {len(valid)} within-species results to wandb | "
        f"avg P@K={avg('precision_at_k'):.4f}, avg R@K={avg('recall_at_k'):.4f}"
    )


def log_plots(plot_dir: Path) -> None:
    """Log all PNG plots from a directory to wandb.

    Args:
        plot_dir: Directory containing .png plot files.
    """
    if _run is None:
        return

    import wandb

    plot_dir = Path(plot_dir)
    if not plot_dir.is_dir():
        return

    for png in sorted(plot_dir.glob("*.png")):
        _run.log({png.stem: wandb.Image(str(png))})
    logger.info(f"Logged {len(list(plot_dir.glob('*.png')))} plots to wandb")


def finish_run() -> None:
    """Finish the active wandb run."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None
