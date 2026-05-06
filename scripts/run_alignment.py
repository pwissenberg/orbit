#!/usr/bin/env python
"""Run SPACE alignment on prepared data.

Usage:
    uv run python scripts/run_alignment.py --stage seeds --device cuda
    uv run python scripts/run_alignment.py --stage nonseeds --device cuda
    uv run python scripts/run_alignment.py --stage all --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

import orbit._compat  # noqa: F401 — patch numpy before SPACE

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Default paths
NODE2VEC_DIR = ROOT / "data" / "node2vec"
ORTHOLOGS_DIR = ROOT / "data" / "orthologs"
ALIGNED_DIR = ROOT / "results" / "aligned_embeddings"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"
SPACE_CONFIG_DIR = ROOT / "data"


def load_seed_info() -> dict:
    with open(SEED_RESULTS) as f:
        return json.load(f)


def _patch_space_for_string_ids():
    """Monkey-patch SPACE's FedCoder to accept string species IDs.

    SPACE assumes taxonomy IDs (integers). Our species use 5-letter codes.
    We patch the __init__ methods to keep species IDs as strings.
    """
    import space.models.fedcoder as fc
    import pandas as pd

    _orig_fedcoder_init = fc.FedCoder.__init__

    def _patched_fedcoder_init(self, seed_species, **kwargs):
        # Read seed file but keep as strings instead of int()
        species_list = open(seed_species).read().strip().split("\n")
        # Temporarily swap in a dummy file path that will be re-read
        _orig_fedcoder_init.__func__  # just to verify it exists
        # Bypass original __init__ and do it manually
        _manual_fedcoder_init(self, species_list, **kwargs)

    def _manual_fedcoder_init(self, seed_species_list, **kwargs):
        """Replicate FedCoder.__init__ but with string species IDs."""
        import os
        from datetime import datetime
        from uuid import uuid4

        self.seed_species = seed_species_list  # strings, not ints

        self.node2vec_dir = kwargs["node2vec_dir"]
        self.ortholog_dir = kwargs["ortholog_dir"]
        self.embedding_save_folder = kwargs["aligned_embedding_save_dir"]
        self.save_top_k = kwargs.get("save_top_k", 3)

        log_dir = kwargs.get("log_dir") or os.path.join(
            kwargs["aligned_embedding_save_dir"], "logs"
        )
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.log_dir = log_dir + "-" + str(uuid4())
        self.model_save_path = os.path.join(self.log_dir, "model.pth")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.embedding_save_folder, exist_ok=True)

        self.input_dim = kwargs.get("input_dim", 128)
        self.latent_dim = kwargs.get("latent_dim", 512)
        self.hidden_dims = kwargs.get("hidden_dims")
        self.activation_fn = kwargs.get("activation_fn")
        self.batch_norm = kwargs.get("batch_norm", False)
        self.number_iters = kwargs.get("number_iters", 10)
        self.autoencoder_type = kwargs.get("autoencoder_type", "naive")
        self.gamma = kwargs.get("gamma", 0.1)
        self.alpha = kwargs.get("alpha", 0.5)
        self.lr = kwargs.get("lr", 0.01)
        self.device = kwargs.get("device", "cpu")
        self.patience = kwargs.get("patience", 5)
        self.delta = kwargs.get("delta", 0.0001)
        self.epochs = kwargs.get("epochs", 600)
        self.from_pretrained = kwargs.get("from_pretrained")

    fc.FedCoder.__init__ = lambda self, seed_species, **kw: _manual_fedcoder_init(
        self, open(seed_species).read().strip().split("\n"), **kw
    )

    # Patch FedCoderNonSeed.__init__ similarly
    def _manual_nonseed_init(self, **kwargs):
        """Replicate FedCoderNonSeed.__init__ with string species IDs."""
        import os
        from datetime import datetime
        from uuid import uuid4

        self.non_seed_species = kwargs["non_seed_species"]  # string

        seed_groups = json.load(open(kwargs["seed_groups"]))
        tax_group = pd.read_csv(kwargs["tax_group"], sep="\t")

        # Look up which group this species belongs to
        group_name = tax_group[tax_group["taxid"] == self.non_seed_species][
            "group"
        ].values[0]
        self.seed_species = seed_groups[group_name]  # list of strings
        self.seed_groups = seed_groups

        self.node2vec_dir = kwargs["node2vec_dir"]
        self.aligned_dir = kwargs["aligned_dir"]
        self.ortholog_dir = f'{kwargs["ortholog_dir"]}/{self.non_seed_species}'
        self.embedding_save_folder = kwargs["aligned_embedding_save_dir"]
        self.save_top_k = kwargs.get("save_top_k", 3)

        log_dir = kwargs.get("log_dir") or os.path.join(
            kwargs["aligned_embedding_save_dir"], "logs"
        )
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_dir = log_dir + "-" + str(uuid4())
        self.log_dir = log_dir
        self.model_save_path = os.path.join(self.log_dir, "model.pth")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.embedding_save_folder, exist_ok=True)

        self.input_dim = kwargs.get("input_dim", 128)
        self.latent_dim = kwargs.get("latent_dim", 512)
        self.hidden_dims = kwargs.get("hidden_dims")
        self.activation_fn = kwargs.get("activation_fn")
        self.batch_norm = kwargs.get("batch_norm", False)
        self.number_iters = kwargs.get("number_iters", 10)
        self.autoencoder_type = kwargs.get("autoencoder_type", "naive")
        self.gamma = kwargs.get("gamma", 0.1)
        self.alpha = kwargs.get("alpha", 0.5)
        self.lr = kwargs.get("lr", 0.01)
        self.device = kwargs.get("device", "cpu")
        self.patience = kwargs.get("patience", 5)
        self.delta = kwargs.get("delta", 0.0001)
        self.epochs = kwargs.get("epochs", 600)
        self.from_pretrained = kwargs.get("from_pretrained")

    fc.FedCoderNonSeed.__init__ = lambda self, **kw: _manual_nonseed_init(self, **kw)

    # Patch alignment_loss to compare strings instead of int(src)
    import torch
    import torch.nn.functional as F

    def _patched_alignment_loss(self, pair_batches):
        loss = []
        for src_tgt, (src_index, tgt_index, weight) in pair_batches.items():
            src, tgt = src_tgt.split("_")
            src_batch = self.node2vec_embeddings[src][src_index].to(self.device)
            tgt_batch = self.node2vec_embeddings[tgt][tgt_index].to(self.device)
            # Compare as strings (original does int(src) == self.non_seed_species)
            if src == str(self.non_seed_species):
                src_latent = self.models[f"encoder_{self.non_seed_species}"](src_batch)
                tgt_latent = tgt_batch
            else:
                src_latent = src_batch
                tgt_latent = self.models[f"encoder_{self.non_seed_species}"](tgt_batch)
            src_tgt_loss = F.pairwise_distance(src_latent, tgt_latent, p=2)
            weight = weight.to(self.device)
            src_tgt_loss = -F.logsigmoid(self.gamma - src_tgt_loss) * weight
            loss.append(src_tgt_loss)
        return torch.cat(loss).mean().unsqueeze(0)

    fc.FedCoderNonSeed.alignment_loss = _patched_alignment_loss


def run_seed_alignment(
    device: str = "cuda",
    from_pretrained: str | None = None,
    orthologs_dir: Path | None = None,
    aligned_dir: Path | None = None,
    **kwargs,
) -> None:
    """Stage 1: Align seed species using FedCoder."""
    from space.models.fedcoder import FedCoder

    _orthologs_dir = orthologs_dir or ORTHOLOGS_DIR
    _aligned_dir = aligned_dir or ALIGNED_DIR

    seed_info = load_seed_info()
    seeds = seed_info["seeds"]

    seeds_file = str(SPACE_CONFIG_DIR / "seeds.txt")
    node2vec_dir = str(NODE2VEC_DIR)
    ortholog_dir = str(_orthologs_dir / "seeds")
    aligned_dir = str(_aligned_dir)

    # Check if seed embeddings already exist
    existing = [s for s in seeds if (Path(aligned_dir) / f"{s}.h5").exists()]
    if len(existing) == len(seeds):
        logger.info(f"Stage 1: All {len(seeds)} seed embeddings already exist, skipping")
        return

    logger.info(f"Stage 1: Aligning {len(seeds)} seed species on {device}")
    logger.info(f"  Seeds: {seeds}")
    if from_pretrained:
        logger.info(f"  Loading pretrained model: {from_pretrained}")

    coder = FedCoder(
        seed_species=seeds_file,
        node2vec_dir=node2vec_dir,
        ortholog_dir=ortholog_dir,
        aligned_embedding_save_dir=aligned_dir,
        input_dim=kwargs.get("input_dim", 128),
        latent_dim=kwargs.get("latent_dim", 512),
        alpha=kwargs.get("alpha", 0.5),
        gamma=kwargs.get("gamma", 0.1),
        lr=kwargs.get("lr", 0.01),
        epochs=kwargs.get("epochs", 500),
        device=device,
        patience=kwargs.get("patience", 5),
        delta=kwargs.get("delta", 0.01),
        from_pretrained=from_pretrained,
    )

    if from_pretrained:
        # Just init and save — no need to retrain
        coder.init_everything()
        logger.info("  Model loaded, saving seed embeddings...")
    else:
        coder.fit()

    coder.save_embeddings()
    logger.info("Stage 1 complete: seed embeddings saved")


def _align_one_nonseed(args_tuple):
    """Worker function to align a single non-seed species."""
    ns, device, config = args_tuple
    # Each worker must patch SPACE independently
    _patch_space_for_string_ids()
    from space.models.fedcoder import FedCoderNonSeed

    try:
        logger.info(f"  Aligning {ns} (nearest seed: {config['group']}) on {device}")
        coder = FedCoderNonSeed(
            seed_groups=config["seed_groups_file"],
            tax_group=config["tax_group_file"],
            non_seed_species=ns,
            node2vec_dir=config["node2vec_dir"],
            aligned_dir=config["aligned_dir"],
            ortholog_dir=config["ortholog_dir"],
            aligned_embedding_save_dir=config["aligned_dir"],
            input_dim=config.get("input_dim", 128),
            latent_dim=config.get("latent_dim", 512),
            alpha=config.get("alpha", 0.5),
            gamma=config.get("gamma", 0.1),
            lr=config.get("lr", 0.01),
            epochs=config.get("epochs", 500),
            device=device,
            patience=config.get("patience", 5),
            delta=config.get("delta", 0.01),
        )
        coder.fit()
        coder.save_embeddings()
        logger.info(f"  {ns}: alignment complete")
        return ns, True
    except Exception as e:
        logger.error(f"  {ns}: FAILED — {e}")
        return ns, False


def run_nonseed_alignment(
    device: str = "cuda",
    workers: int = 1,
    orthologs_dir: Path | None = None,
    aligned_dir: Path | None = None,
    **kwargs,
) -> None:
    """Stage 2: Align each non-seed species to the seed space."""
    _orthologs_dir = orthologs_dir or ORTHOLOGS_DIR
    _aligned_dir = aligned_dir or ALIGNED_DIR

    seed_info = load_seed_info()
    groups = seed_info["groups"]

    config = {
        "seed_groups_file": str(SPACE_CONFIG_DIR / "seed_groups.json"),
        "tax_group_file": str(SPACE_CONFIG_DIR / "tax_group.tsv"),
        "node2vec_dir": str(NODE2VEC_DIR),
        "aligned_dir": str(_aligned_dir),
        "ortholog_dir": str(_orthologs_dir / "non_seeds"),
        "input_dim": kwargs.get("input_dim", 128),
        "latent_dim": kwargs.get("latent_dim", 512),
        "alpha": kwargs.get("alpha", 0.5),
        "gamma": kwargs.get("gamma", 0.1),
        "lr": kwargs.get("lr", 0.01),
        "epochs": kwargs.get("epochs", 500),
        "patience": kwargs.get("patience", 5),
        "delta": kwargs.get("delta", 0.01),
    }

    # Build list of species that still need alignment
    todo = []
    for ns in sorted(groups.keys()):
        h5_path = Path(config["node2vec_dir"]) / f"{ns}.h5"
        if not h5_path.exists():
            logger.warning(f"  {ns}: no node2vec, skipping")
            continue
        out_path = Path(config["aligned_dir"]) / f"{ns}.h5"
        if out_path.exists():
            logger.info(f"  {ns}: already aligned, skipping")
            continue
        todo.append(ns)

    done_count = len(groups) - len(todo)
    logger.info(
        f"Stage 2: {len(todo)} species remaining ({done_count} already done), "
        f"{workers} workers on {device}"
    )

    if not todo:
        logger.info("Stage 2: nothing to do")
        return

    if workers <= 1:
        # Serial mode (original behavior)
        from space.models.fedcoder import FedCoderNonSeed

        for i, ns in enumerate(todo):
            logger.info(
                f"  [{done_count+i+1}/{len(groups)}] Aligning {ns} "
                f"(nearest seed: {groups[ns]})"
            )
            config["group"] = groups[ns]
            coder = FedCoderNonSeed(
                seed_groups=config["seed_groups_file"],
                tax_group=config["tax_group_file"],
                non_seed_species=ns,
                node2vec_dir=config["node2vec_dir"],
                aligned_dir=config["aligned_dir"],
                ortholog_dir=config["ortholog_dir"],
                aligned_embedding_save_dir=config["aligned_dir"],
                input_dim=config["input_dim"],
                latent_dim=config["latent_dim"],
                alpha=config["alpha"],
                gamma=config["gamma"],
                lr=config["lr"],
                epochs=config["epochs"],
                device=device,
                patience=config["patience"],
                delta=config["delta"],
            )
            coder.fit()
            coder.save_embeddings()
    else:
        # Parallel mode using torch.multiprocessing
        import torch
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        # Multi-GPU round-robin: if device is bare "cuda", spread across all GPUs
        if device == "cuda" and torch.cuda.device_count() > 1:
            n_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(n_gpus)]
            logger.info(f"  Multi-GPU mode: {n_gpus} GPUs, {workers} workers total")
        else:
            devices = [device]

        work_items = []
        for i, ns in enumerate(todo):
            cfg = {**config, "group": groups[ns]}
            work_items.append((ns, devices[i % len(devices)], cfg))

        with mp.Pool(processes=workers) as pool:
            results = pool.map(_align_one_nonseed, work_items)

        failed = [ns for ns, ok in results if not ok]
        if failed:
            logger.warning(f"Failed species: {failed}")

    logger.info("Stage 2 complete: all non-seed embeddings saved")


def main():
    parser = argparse.ArgumentParser(description="Run SPACE alignment")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["seeds", "nonseeds", "all"],
        help="Which alignment stage to run",
    )
    parser.add_argument("--aligned-dir", type=Path, default=None,
                        help="Output directory for aligned embeddings (default: results/aligned_embeddings)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel species to train (default 1)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Early stopping min improvement threshold (default 0.01)")
    parser.add_argument(
        "--from-pretrained",
        default=None,
        help="Path to pretrained model.pth to resume from (skips training)",
    )
    args = parser.parse_args()

    # Patch SPACE to handle string species IDs
    _patch_space_for_string_ids()

    extra = {
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "patience": args.patience,
        "delta": args.delta,
    }

    if args.stage in ("seeds", "all"):
        run_seed_alignment(device=args.device, from_pretrained=args.from_pretrained,
                           aligned_dir=args.aligned_dir, **extra)
    if args.stage in ("nonseeds", "all"):
        run_nonseed_alignment(device=args.device, workers=args.workers,
                              aligned_dir=args.aligned_dir, **extra)

    logger.info("Alignment complete!")


if __name__ == "__main__":
    main()
