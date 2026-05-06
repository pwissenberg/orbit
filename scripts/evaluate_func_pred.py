#!/usr/bin/env python3
"""GO function prediction evaluation.

Evaluates alignment quality by predicting Gene Ontology terms from embeddings.
Compares three conditions per alignment method:
  1. ProtT5 alone (1024-dim)
  2. Aligned coexpression embedding alone (128-dim)
  3. Concatenated (1152-dim)

For each GO aspect (BP, CC, MF), trains per-term binary LogisticRegression classifiers
on terms with >= min_proteins annotated genes. Evaluation via Fmax, AUPRC.

Input:
    data/annotations/go/{species}_goa.tsv       -- GO annotations
    data/prott5/{species}.h5                     -- ProtT5 embeddings
    results/aligned_embeddings_*/{species}.h5    -- aligned coexpression embeddings

Output:
    results/downstream/func_pred/scores.json

Usage:
    uv run python scripts/evaluate_func_pred.py --species ARATH
    uv run python scripts/evaluate_func_pred.py --species ARATH ORYSA --methods vanilla procrustes_n2v
    uv run python scripts/evaluate_func_pred.py --species ARATH --min-proteins 20
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
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

GO_DIR = ROOT / "data" / "annotations" / "go"
PROTT5_DIR = ROOT / "data" / "prott5"
OUTPUT_DIR = ROOT / "results" / "downstream" / "func_pred"

ALIGNED_DIRS = {
    "vanilla": ROOT / "results" / "aligned_embeddings",
    "jaccard": ROOT / "results" / "aligned_embeddings_jaccard",
    "procrustes_svd": ROOT / "results" / "aligned_embeddings_procrustes",
    "procrustes_n2v": ROOT / "results" / "aligned_embeddings_procrustes_n2v",
}

ASPECTS = ["BP", "CC", "MF"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_h5_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load embeddings from H5 file. Returns protein->embedding dict."""
    with h5py.File(path, "r") as fh:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in fh["proteins"][:]]
        embeddings = fh["embeddings"][:]
    return {p: embeddings[i] for i, p in enumerate(proteins)}


def _load_go_annotations(species: str) -> dict[str, dict[str, set[str]]]:
    """Load GO annotations. Returns {aspect: {gene: set of GO terms}}."""
    path = GO_DIR / f"{species}_goa.tsv"
    if not path.exists():
        return {}

    annotations: dict[str, dict[str, set[str]]] = {a: defaultdict(set) for a in ASPECTS}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row["teagcn_id"]
            go_term = row["go_term"]
            aspect = row["aspect"]
            if aspect in annotations:
                annotations[aspect][gene].add(go_term)

    return {a: dict(v) for a, v in annotations.items()}


def _compute_fmax(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Compute Fmax (maximum F1 over all thresholds) and AUPRC."""
    # Binary classification: compute precision-recall
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Fmax: max harmonic mean of precision and recall
    f_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )
    fmax = float(f_scores.max())

    # AUPRC
    auprc = float(average_precision_score(y_true, y_scores))

    return fmax, auprc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_aspect(
    aspect: str,
    go_annotations: dict[str, set[str]],
    prott5_dict: dict[str, np.ndarray],
    aligned_dict: dict[str, np.ndarray],
    min_proteins: int = 10,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Evaluate GO term prediction for one aspect.

    Returns dict with per-condition Fmax and AUPRC (macro-averaged across terms).
    """
    # Find all GO terms with sufficient annotated genes
    term_genes: dict[str, list[str]] = defaultdict(list)
    for gene, terms in go_annotations.items():
        for term in terms:
            if gene in prott5_dict or gene in aligned_dict:
                term_genes[term].append(gene)

    valid_terms = {t: genes for t, genes in term_genes.items() if len(genes) >= min_proteins}
    if not valid_terms:
        return {"aspect": aspect, "error": "no valid terms", "n_terms": 0}

    # Get all unique genes
    all_genes = set()
    for genes in valid_terms.values():
        all_genes.update(genes)

    # Filter to genes with embeddings
    genes_with_emb = all_genes & (set(prott5_dict.keys()) | set(aligned_dict.keys()))
    gene_list = sorted(genes_with_emb)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    logger.info(f"  {aspect}: {len(valid_terms)} terms, {len(gene_list)} genes")

    # Build embedding matrices
    conditions = {}
    if prott5_dict:
        dim = next(iter(prott5_dict.values())).shape[0]
        prott5_emb = np.array([prott5_dict.get(g, np.zeros(dim)) for g in gene_list], dtype=np.float32)
        if not np.all(prott5_emb == 0):
            conditions["prott5_only"] = prott5_emb

    if aligned_dict:
        dim = next(iter(aligned_dict.values())).shape[0]
        aligned_emb = np.array([aligned_dict.get(g, np.zeros(dim)) for g in gene_list], dtype=np.float32)
        if not np.all(aligned_emb == 0):
            conditions["aligned_only"] = aligned_emb

    if "prott5_only" in conditions and "aligned_only" in conditions:
        conditions["concatenated"] = np.concatenate(
            [conditions["prott5_only"], conditions["aligned_only"]], axis=1
        )

    result = {"aspect": aspect, "n_terms": len(valid_terms), "n_genes": len(gene_list)}

    for cond_name, X in conditions.items():
        # Scale features once
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        term_fmax_list = []
        term_auprc_list = []

        for term, pos_genes in valid_terms.items():
            # Binary labels for this term
            y = np.zeros(len(gene_list), dtype=np.int32)
            for g in pos_genes:
                if g in gene_to_idx:
                    y[gene_to_idx[g]] = 1

            if y.sum() < 5:
                continue

            # Cross-validated prediction scores
            try:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                y_scores = np.zeros(len(gene_list), dtype=np.float64)

                for train_idx, val_idx in skf.split(X_scaled, y):
                    clf = LogisticRegression(max_iter=1000, random_state=seed)
                    clf.fit(X_scaled[train_idx], y[train_idx])
                    y_scores[val_idx] = clf.predict_proba(X_scaled[val_idx])[:, 1]

                fmax, auprc = _compute_fmax(y, y_scores)
                term_fmax_list.append(fmax)
                term_auprc_list.append(auprc)
            except Exception:
                continue

        if term_fmax_list:
            result[f"{cond_name}_fmax_mean"] = float(np.mean(term_fmax_list))
            result[f"{cond_name}_fmax_std"] = float(np.std(term_fmax_list))
            result[f"{cond_name}_auprc_mean"] = float(np.mean(term_auprc_list))
            result[f"{cond_name}_auprc_std"] = float(np.std(term_auprc_list))
            result[f"{cond_name}_n_terms_evaluated"] = len(term_fmax_list)

    return result


def evaluate_species(
    species: str,
    method: str,
    min_proteins: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Evaluate GO function prediction for one species + one alignment method."""
    go_annotations = _load_go_annotations(species)
    if not go_annotations:
        return [{"species": species, "method": method, "error": "no GO annotations"}]

    # Load embeddings
    prott5_dict = {}
    prott5_path = PROTT5_DIR / f"{species}.h5"
    if prott5_path.exists():
        prott5_dict = _load_h5_embeddings(prott5_path)

    aligned_dict = {}
    if method in ALIGNED_DIRS:
        aligned_path = ALIGNED_DIRS[method] / f"{species}.h5"
        if aligned_path.exists():
            aligned_dict = _load_h5_embeddings(aligned_path)

    if not prott5_dict and not aligned_dict:
        return [{"species": species, "method": method, "error": "no embeddings"}]

    results = []
    for aspect in ASPECTS:
        aspect_annot = go_annotations.get(aspect, {})
        if not aspect_annot:
            results.append({"species": species, "method": method, "aspect": aspect, "error": "no annotations"})
            continue

        logger.info(f"  {species}/{method}: evaluating {aspect}")
        result = evaluate_aspect(
            aspect, aspect_annot, prott5_dict, aligned_dict,
            min_proteins=min_proteins, seed=seed,
        )
        result["species"] = species
        result["method"] = method
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="GO function prediction evaluation")
    parser.add_argument("--species", nargs="+", default=["ARATH"], help="Species to evaluate")
    parser.add_argument(
        "--methods", nargs="+",
        default=["vanilla", "jaccard", "procrustes_svd", "procrustes_n2v"],
        help="Alignment methods to compare",
    )
    parser.add_argument("--min-proteins", type=int, default=10, help="Min proteins per GO term")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for species in args.species:
        for method in args.methods:
            logger.info(f"Evaluating {species} / {method}")
            results = evaluate_species(species, method, min_proteins=args.min_proteins, seed=args.seed)
            all_results.extend(results)

    # Save results
    out_path = OUTPUT_DIR / "scores.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved -> {out_path}")

    # Print summary
    print("\n" + "=" * 90)
    print("GO FUNCTION PREDICTION SUMMARY")
    print("=" * 90)
    for r in all_results:
        if "error" in r:
            print(f"  {r.get('species','?')}/{r.get('method','?')}/{r.get('aspect','?')}: {r['error']}")
            continue
        print(f"\n  {r['species']} / {r['method']} / {r['aspect']} ({r['n_terms']} terms, {r['n_genes']} genes):")
        for cond in ["prott5_only", "aligned_only", "concatenated"]:
            fmax = r.get(f"{cond}_fmax_mean")
            auprc = r.get(f"{cond}_auprc_mean")
            n = r.get(f"{cond}_n_terms_evaluated", 0)
            if fmax is not None:
                print(f"    {cond:20s}  Fmax={fmax:.4f}  AUPRC={auprc:.4f}  ({n} terms)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
