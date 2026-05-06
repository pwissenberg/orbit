#!/usr/bin/env python3
"""Subcellular localization prediction evaluation.

Evaluates alignment quality by predicting subcellular compartment from embeddings.
Compares three conditions per alignment method:
  1. ProtT5 alone (1024-dim)
  2. Aligned coexpression embedding alone (128-dim)
  3. Concatenated (1152-dim)

Uses LogisticRegression with 5-fold stratified cross-validation.

Optionally runs cross-species transfer: train on ARATH, predict on ORYSA.

Input:
    data/annotations/subloc/{species}_subloc.tsv   -- compartment labels
    data/prott5/{species}.h5                        -- ProtT5 embeddings
    results/aligned_embeddings_*/{species}.h5       -- aligned coexpression embeddings

Output:
    results/downstream/subloc/{method}_cv_scores.csv
    results/downstream/subloc/{method}_cv_mccs.csv
    results/downstream/subloc/transfer_{train}_{test}.csv

Usage:
    uv run python scripts/evaluate_subloc.py --species ARATH
    uv run python scripts/evaluate_subloc.py --species ARATH --methods vanilla jaccard procrustes_n2v
    uv run python scripts/evaluate_subloc.py --species ARATH --transfer ORYSA
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

SUBLOC_DIR = ROOT / "data" / "annotations" / "subloc"
PROTT5_DIR = ROOT / "data" / "prott5"
OUTPUT_DIR = ROOT / "results" / "downstream" / "subloc"

COMPARTMENTS = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
    "Mitochondrion", "Plastid", "Endoplasmic reticulum",
    "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome",
]

ALIGNED_DIRS = {
    "vanilla": ROOT / "results" / "aligned_embeddings",
    "jaccard": ROOT / "results" / "aligned_embeddings_jaccard",
    "procrustes_svd": ROOT / "results" / "aligned_embeddings_procrustes",
    "procrustes_n2v": ROOT / "results" / "aligned_embeddings_procrustes_n2v",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_h5_embeddings(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """Load embeddings from H5 file. Returns (protein->embedding dict, dim)."""
    with h5py.File(path, "r") as fh:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in fh["proteins"][:]]
        embeddings = fh["embeddings"][:]
    emb_dict = {p: embeddings[i] for i, p in enumerate(proteins)}
    return emb_dict, embeddings.shape[1]


def _load_subloc_labels(species: str) -> dict[str, set[str]]:
    """Load subcellular localization labels. Returns gene -> set of compartments."""
    path = SUBLOC_DIR / f"{species}_subloc.tsv"
    if not path.exists():
        return {}
    labels: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row["teagcn_id"]
            comp = row["compartment"]
            if comp in COMPARTMENTS:
                labels[gene].add(comp)
    return dict(labels)


def _build_label_matrix(
    gene_ids: list[str],
    labels: dict[str, set[str]],
) -> np.ndarray:
    """Build binary label matrix (n_genes, n_compartments)."""
    mat = np.zeros((len(gene_ids), len(COMPARTMENTS)), dtype=np.int32)
    for i, gene in enumerate(gene_ids):
        for comp in labels.get(gene, []):
            j = COMPARTMENTS.index(comp)
            mat[i, j] = 1
    return mat


def _prepare_data(
    species: str,
    method: str | None,
) -> tuple[list[str], np.ndarray | None, np.ndarray | None, np.ndarray] | None:
    """Prepare aligned + ProtT5 embeddings and labels for a species.

    Returns (gene_ids, prott5_emb, aligned_emb, label_matrix) or None if data missing.
    """
    labels = _load_subloc_labels(species)
    if not labels:
        logger.warning(f"  {species}: no subloc labels found")
        return None

    # Load ProtT5 embeddings
    prott5_path = PROTT5_DIR / f"{species}.h5"
    prott5_dict = {}
    if prott5_path.exists():
        prott5_dict, _ = _load_h5_embeddings(prott5_path)

    # Load aligned embeddings
    aligned_dict = {}
    if method and method in ALIGNED_DIRS:
        aligned_path = ALIGNED_DIRS[method] / f"{species}.h5"
        if aligned_path.exists():
            aligned_dict, _ = _load_h5_embeddings(aligned_path)

    # Find common genes (at least in labels + one embedding type)
    labeled_genes = set(labels.keys())
    prott5_genes = set(prott5_dict.keys())
    aligned_genes = set(aligned_dict.keys())

    # Use genes that have labels and at least one embedding
    available_genes = labeled_genes & (prott5_genes | aligned_genes)
    if not available_genes:
        logger.warning(f"  {species}: no genes with both labels and embeddings")
        return None

    gene_ids = sorted(available_genes)
    label_mat = _build_label_matrix(gene_ids, labels)

    # Filter to genes with at least one label
    has_label = label_mat.sum(axis=1) > 0
    gene_ids = [g for g, h in zip(gene_ids, has_label) if h]
    label_mat = label_mat[has_label]

    if len(gene_ids) < 50:
        logger.warning(f"  {species}: only {len(gene_ids)} labeled genes, too few")
        return None

    # Build embedding matrices
    prott5_emb = None
    if prott5_dict:
        prott5_emb = np.array([prott5_dict.get(g, np.zeros(1024)) for g in gene_ids], dtype=np.float32)
        # Replace missing with zeros (will be handled by scaler)
        missing = np.array([g not in prott5_dict for g in gene_ids])
        if missing.any():
            logger.info(f"  {species}: {missing.sum()} genes missing ProtT5 embeddings")

    aligned_emb = None
    if aligned_dict:
        dim = next(iter(aligned_dict.values())).shape[0]
        aligned_emb = np.array([aligned_dict.get(g, np.zeros(dim)) for g in gene_ids], dtype=np.float32)

    return gene_ids, prott5_emb, aligned_emb, label_mat


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stratified k-fold CV for multi-label subcellular localization.

    Returns (scores_df, mcc_df) with per-fold results.
    """
    # Use dominant label for stratification
    dominant = y.argmax(axis=1)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scores = []
    mccs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, dominant)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=seed)
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        # Compute metrics
        f1_mi = f1_score(y_val, y_pred, average="micro", zero_division=0)
        f1_ma = f1_score(y_val, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_val, y_pred)

        scores.append({"fold": fold, "f1_micro": f1_mi, "f1_macro": f1_ma, "accuracy": acc})

        # Per-compartment MCC
        mcc_row = {"fold": fold}
        for j, comp in enumerate(COMPARTMENTS):
            if y_val[:, j].sum() > 0:
                mcc_row[comp] = matthews_corrcoef(y_val[:, j], y_pred[:, j])
            else:
                mcc_row[comp] = np.nan
        mccs.append(mcc_row)

    return pd.DataFrame(scores), pd.DataFrame(mccs)


def evaluate_species(
    species: str,
    method: str,
    seed: int = 42,
) -> dict:
    """Evaluate subcellular localization for one species + one alignment method.

    Returns dict with results for three conditions:
      - prott5_only, aligned_only, concatenated
    """
    data = _prepare_data(species, method)
    if data is None:
        return {"species": species, "method": method, "error": "data not available"}

    gene_ids, prott5_emb, aligned_emb, label_mat = data
    logger.info(f"  {species}/{method}: {len(gene_ids)} genes, {label_mat.sum()} total labels")

    result = {"species": species, "method": method, "n_genes": len(gene_ids)}

    conditions = {}
    if prott5_emb is not None:
        conditions["prott5_only"] = prott5_emb
    if aligned_emb is not None:
        conditions["aligned_only"] = aligned_emb
    if prott5_emb is not None and aligned_emb is not None:
        conditions["concatenated"] = np.concatenate([prott5_emb, aligned_emb], axis=1)

    for cond_name, X in conditions.items():
        logger.info(f"  {species}/{method}/{cond_name}: X.shape={X.shape}")
        scores_df, mccs_df = _evaluate_cv(X, label_mat, seed=seed)

        result[f"{cond_name}_f1_micro"] = float(scores_df["f1_micro"].mean())
        result[f"{cond_name}_f1_micro_std"] = float(scores_df["f1_micro"].std())
        result[f"{cond_name}_f1_macro"] = float(scores_df["f1_macro"].mean())
        result[f"{cond_name}_f1_macro_std"] = float(scores_df["f1_macro"].std())
        result[f"{cond_name}_accuracy"] = float(scores_df["accuracy"].mean())

        # Per-compartment MCC (mean across folds)
        for comp in COMPARTMENTS:
            if comp in mccs_df.columns:
                vals = mccs_df[comp].dropna()
                result[f"{cond_name}_mcc_{comp}"] = float(vals.mean()) if len(vals) > 0 else np.nan

    return result


def evaluate_transfer(
    train_species: str,
    test_species: str,
    method: str,
    seed: int = 42,
) -> dict:
    """Cross-species transfer: train on one species, test on another."""
    train_data = _prepare_data(train_species, method)
    test_data = _prepare_data(test_species, method)

    if train_data is None or test_data is None:
        return {
            "train_species": train_species,
            "test_species": test_species,
            "method": method,
            "error": "data not available",
        }

    _, train_prott5, train_aligned, train_labels = train_data
    _, test_prott5, test_aligned, test_labels = test_data

    result = {
        "train_species": train_species,
        "test_species": test_species,
        "method": method,
    }

    conditions = {}
    if train_prott5 is not None and test_prott5 is not None:
        conditions["prott5_only"] = (train_prott5, test_prott5)
    if train_aligned is not None and test_aligned is not None:
        conditions["aligned_only"] = (train_aligned, test_aligned)
    if (train_prott5 is not None and test_prott5 is not None and
            train_aligned is not None and test_aligned is not None):
        conditions["concatenated"] = (
            np.concatenate([train_prott5, train_aligned], axis=1),
            np.concatenate([test_prott5, test_aligned], axis=1),
        )

    for cond_name, (X_train, X_test) in conditions.items():
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=seed)
        )
        clf.fit(X_train_s, train_labels)
        y_pred = clf.predict(X_test_s)

        result[f"{cond_name}_f1_micro"] = float(f1_score(test_labels, y_pred, average="micro", zero_division=0))
        result[f"{cond_name}_f1_macro"] = float(f1_score(test_labels, y_pred, average="macro", zero_division=0))
        result[f"{cond_name}_accuracy"] = float(accuracy_score(test_labels, y_pred))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Subcellular localization prediction evaluation")
    parser.add_argument("--species", nargs="+", default=["ARATH"], help="Species to evaluate")
    parser.add_argument(
        "--methods", nargs="+",
        default=["vanilla", "jaccard", "procrustes_svd", "procrustes_n2v"],
        help="Alignment methods to compare",
    )
    parser.add_argument("--transfer", nargs="*", help="Test species for cross-species transfer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter methods to those with available data
    available_methods = [m for m in args.methods if ALIGNED_DIRS.get(m, Path("/none")).exists()]
    if not available_methods:
        logger.warning("No aligned embedding directories found. Checking availability...")
        for m, d in ALIGNED_DIRS.items():
            logger.info(f"  {m}: {d} {'EXISTS' if d.exists() else 'MISSING'}")
        available_methods = args.methods  # try anyway

    # Cross-validation evaluation
    all_results = []
    for species in args.species:
        for method in available_methods:
            logger.info(f"Evaluating {species} / {method}")
            result = evaluate_species(species, method, seed=args.seed)
            all_results.append(result)

            if "error" not in result:
                for cond in ["prott5_only", "aligned_only", "concatenated"]:
                    f1 = result.get(f"{cond}_f1_micro", "N/A")
                    logger.info(f"  {cond}: F1-micro = {f1}")

    # Save CV results
    cv_path = OUTPUT_DIR / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"CV results saved -> {cv_path}")

    # Cross-species transfer
    if args.transfer:
        transfer_results = []
        for train_sp in args.species:
            for test_sp in args.transfer:
                if train_sp == test_sp:
                    continue
                for method in available_methods:
                    logger.info(f"Transfer: {train_sp} -> {test_sp} / {method}")
                    result = evaluate_transfer(train_sp, test_sp, method, seed=args.seed)
                    transfer_results.append(result)

        transfer_path = OUTPUT_DIR / "transfer_results.json"
        with open(transfer_path, "w") as f:
            json.dump(transfer_results, f, indent=2, default=str)
        logger.info(f"Transfer results saved -> {transfer_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUBCELLULAR LOCALIZATION EVALUATION SUMMARY")
    print("=" * 80)
    for r in all_results:
        if "error" in r:
            print(f"  {r['species']}/{r['method']}: {r['error']}")
            continue
        print(f"\n  {r['species']} / {r['method']} ({r['n_genes']} genes):")
        for cond in ["prott5_only", "aligned_only", "concatenated"]:
            f1mi = r.get(f"{cond}_f1_micro", "N/A")
            f1ma = r.get(f"{cond}_f1_macro", "N/A")
            if isinstance(f1mi, float):
                print(f"    {cond:20s}  F1-micro={f1mi:.4f}  F1-macro={f1ma:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
