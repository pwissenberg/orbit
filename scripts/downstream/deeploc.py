#!/usr/bin/env python3
"""Subcellular localization from aligned embeddings (DeepLoc 2.0 protocol).

Reproduces Fig. 3B (plant subset, Section 2.7), Fig. 4A (STRING, Section 2.8) and, with
``--partition-by species``, the leave-one-species-out protocol of Fig. S4. The embedding arms
are the aligned embeddings produced before with ``orbit align`` (plant:
``results/plant/<SPECIES>.h5``; STRING: ``results/string/<taxid>.h5``, identical to the Zenodo
release), the ProtT5 sequence embeddings, the FedCoder/SPACE baseline, and their
concatenations and PCA controls (see ``_common.py`` for ``--arm/--pca/--concat``).

Labels come either as the DeepLoc 2.0 table (``Swissprot_Train_Validation_dataset.csv``
from SPACE's ``benchmarks.zip``: ``ACC``, ``Partition`` and one 0/1 column per compartment),
mapped onto the embedding ids with ``--idmap`` (``cv_idmapping.tsv``), or as a long table
with a protein column and a ``compartment`` column (the UniProt annotations of Fig. S4).

One multi-label logistic regression per arm is trained on all partitions but one and tested on
the held-out partition; out-of-fold probabilities give the precision-recall curve with a
bootstrap band over proteins. Every arm is scored on the proteins present in all arms.

Outputs under --out: ``<arm>_folds.csv`` (metrics per fold), ``summary.json``,
``pr_curves.json``.

    python scripts/downstream/deeploc.py --labels benchmarks/deeploc/Swissprot_Train_Validation_dataset.csv \\
        --idmap benchmarks/deeploc/cv_idmapping.tsv --arm orbit=results/string \\
        --arm space=space/functional_emb --arm prott5=uniprot/per-protein.h5 \\
        --pca space_pca128=space:128 --concat orbit_t5=orbit+prott5 --concat space_t5=space+prott5 \\
        --out results/downstream/string_deeploc
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, jaccard_score, matthews_corrcoef,
                             precision_recall_curve)
from sklearn.multioutput import MultiOutputClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (add_arm_arguments, feature_matrix, load_arms, log, read_idmap,  # noqa: E402
                     shared_proteins, write_json)

COMPARTMENTS = ["Cytoplasm", "Nucleus", "Cell membrane", "Plastid", "Extracellular",
                "Mitochondrion", "Endoplasmic reticulum", "Golgi apparatus",
                "Lysosome/Vacuole", "Peroxisome"]
SEED = 42


# --- labels -------------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t")


def read_labels(path: Path, idmap: dict[str, str] | None = None, *, id_column: str = "ACC",
                partition_column: str = "Partition", partition_by: str = "column") -> pd.DataFrame:
    """``protein``, ``partition`` and one 0/1 column per compartment, one row per protein.

    Wide input (DeepLoc): compartment columns present. Long input: a ``compartment`` column
    with one row per annotation; names outside ``COMPARTMENTS`` are ignored. ``idmap`` maps
    ``id_column`` onto the embedding ids (unmapped rows are dropped); without it the protein
    column is ``protein``, ``teagcn_id`` or ``id_column``, whichever exists. With
    ``partition_by="species"`` the partition is the ``species`` column or the id prefix
    before the first ``.``."""
    df = read_table(path)
    wide = any(c in df.columns for c in COMPARTMENTS)
    if idmap is not None:
        if id_column not in df.columns:
            raise SystemExit(f"error: {path} has no column {id_column!r}")
        df["protein"] = df[id_column].map(idmap)
        df = df[df["protein"].notna()]
    elif "protein" not in df.columns:
        for c in ("teagcn_id", id_column):
            if c in df.columns:
                df = df.rename(columns={c: "protein"})
                break
        else:
            raise SystemExit(f"error: {path} has no protein column (protein, teagcn_id or {id_column})")
    if partition_by == "species":
        df["partition"] = df["species"] if "species" in df.columns else df["protein"].str.split(".").str[0]
    elif partition_by == "none":
        df["partition"] = 0
    elif partition_column in df.columns:
        df["partition"] = df[partition_column]
    else:
        raise SystemExit(f"error: {path} has no column {partition_column!r}; "
                         f"use --partition-by species for a leave-one-species-out split")
    if wide:
        present = [c for c in COMPARTMENTS if c in df.columns]
        out = df[["protein", "partition"] + present].copy()
        for c in COMPARTMENTS:
            if c not in out.columns:
                out[c] = 0
    else:
        if "compartment" not in df.columns:
            raise SystemExit(f"error: {path} has neither compartment columns nor a 'compartment' column")
        df = df[df["compartment"].isin(COMPARTMENTS)]
        part = df.drop_duplicates("protein").set_index("protein")["partition"]
        wide_df = pd.crosstab(df["protein"], df["compartment"]).clip(upper=1)
        out = wide_df.reindex(columns=COMPARTMENTS, fill_value=0).reset_index()
        out["partition"] = out["protein"].map(part)
    out = out.drop_duplicates("protein")
    out[COMPARTMENTS] = out[COMPARTMENTS].fillna(0).astype(int)
    out = out[out[COMPARTMENTS].sum(axis=1) > 0]
    return out[["protein", "partition"] + COMPARTMENTS].reset_index(drop=True)


# --- classifier and metrics ---------------------------------------------------------------

class MultiLabelLR:
    """The classifier of Hu et al. (2025): one logistic regression per compartment
    (``MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=42))``). A
    compartment with a single class in the training fold gets a constant predictor instead of
    an error, which only matters for small leave-one-species-out folds."""

    def __init__(self, seed: int = SEED, n_jobs: int = -1):
        self.seed, self.n_jobs = seed, n_jobs
        self.estimators_: list = []

    def _fit_one(self, X: np.ndarray, y: np.ndarray):
        if y.min() == y.max():
            return float(y[0])
        return LogisticRegression(max_iter=1000, random_state=self.seed).fit(X, y)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiLabelLR":
        from joblib import Parallel, delayed

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(X, Y[:, j]) for j in range(Y.shape[1]))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cols = [np.full(len(X), e) if isinstance(e, float) else e.predict_proba(X)[:, 1]
                for e in self.estimators_]
        return np.column_stack(cols)


def fold_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    m = {
        "f1_micro": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(Y_true, Y_pred)),
        "jaccard_micro": float(jaccard_score(Y_true, Y_pred, average="micro", zero_division=0)),
    }
    for i, comp in enumerate(COMPARTMENTS):
        # MCC is undefined when the held-out fold has a single class for a compartment.
        both = Y_true[:, i].min() != Y_true[:, i].max()
        m[f"mcc_{comp}"] = float(matthews_corrcoef(Y_true[:, i], Y_pred[:, i])) if both else float("nan")
    return m


def evaluate_partitions(X: np.ndarray, Y: np.ndarray, partitions: np.ndarray, *,
                        seed: int = SEED, n_jobs: int = -1) -> tuple[list[dict], np.ndarray]:
    """Hold out each partition in turn. Returns the per-fold metrics and the out-of-fold
    probability matrix."""
    folds, P = [], np.zeros(Y.shape, dtype=float)
    for fold, held in enumerate(sorted(np.unique(partitions), key=str)):
        tr, te = partitions != held, partitions == held
        clf = MultiLabelLR(seed, n_jobs).fit(X[tr], Y[tr])
        P[te] = clf.predict_proba(X[te])
        row = {"fold": fold, "partition": held.item() if hasattr(held, "item") else held,
               "n_train": int(tr.sum()), "n_test": int(te.sum())}
        row.update(fold_metrics(Y[te], (P[te] >= 0.5).astype(int)))
        folds.append(row)
    return folds, P


def summarize(folds: list[dict], dim: int, n_proteins: int) -> dict:
    f1 = [f["f1_micro"] for f in folds]
    return {
        "dim": dim, "n_proteins": n_proteins, "n_folds": len(folds),
        "f1_micro_mean": float(np.mean(f1)),
        "f1_micro_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
        "f1_macro_mean": float(np.mean([f["f1_macro"] for f in folds])),
        "accuracy_mean": float(np.mean([f["accuracy"] for f in folds])),
        "jaccard_micro_mean": float(np.mean([f["jaccard_micro"] for f in folds])),
        "per_fold": folds,
    }


def _fmax(pr: np.ndarray, rc: np.ndarray) -> tuple[float, int]:
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    i = int(np.argmax(f))
    return float(f[i]), i


def pr_curve_with_band(Y: np.ndarray, P: np.ndarray, *, n_boot: int = 200, seed: int = 0,
                       max_points: int = 2000) -> dict:
    """Micro precision-recall over all protein x compartment cells, its Fmax, and a 95%
    bootstrap band over proteins on a common recall grid."""
    pr, rc, _ = precision_recall_curve(Y.ravel(), P.ravel())
    fmax, i = _fmax(pr, rc)
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(seed)
    band, fmaxes = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(Y), len(Y))
        p2, r2, _ = precision_recall_curve(Y[idx].ravel(), P[idx].ravel())
        o = np.argsort(r2)
        band.append(np.interp(grid, r2[o], p2[o]))
        fmaxes.append(_fmax(p2, r2)[0])
    band = np.vstack(band) if band else np.zeros((1, len(grid)))
    stride = max(1, len(pr) // max_points)
    return {"precision": pr[::stride].tolist(), "recall": rc[::stride].tolist(),
            "grid": grid.tolist(),
            "lo": np.percentile(band, 2.5, axis=0).tolist(),
            "hi": np.percentile(band, 97.5, axis=0).tolist(),
            "fmax": fmax, "fmax_precision": float(pr[i]), "fmax_recall": float(rc[i]),
            "fmax_ci": [float(np.percentile(fmaxes, 2.5)), float(np.percentile(fmaxes, 97.5))]
            if fmaxes else [fmax, fmax]}


def write_folds(path: Path, folds: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(folds[0].keys()))
        w.writeheader()
        w.writerows(folds)


# --- main ---------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True, metavar="FILE",
                   help="DeepLoc 2.0 table (CSV) or long protein/compartment table (TSV)")
    p.add_argument("--idmap", type=Path, metavar="TSV", help="maps --id-column onto the embedding ids")
    p.add_argument("--idmap-columns", default="From,To", metavar="SRC,DST",
                   help="columns of --idmap (default From,To)")
    p.add_argument("--id-column", default="ACC", help="label column mapped by --idmap (default ACC)")
    p.add_argument("--partition-column", default="Partition", help="fold column (default Partition)")
    p.add_argument("--partition-by", choices=["column", "species"], default="column",
                   help="'species': one fold per species (leave-one-species-out, Fig. S4)")
    add_arm_arguments(p)
    p.add_argument("--out", type=Path, required=True, metavar="DIR")
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap replicates for the PR band")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    idmap = read_idmap(a.idmap, tuple(a.idmap_columns.split(","))) if a.idmap else None
    labels = read_labels(a.labels, idmap, id_column=a.id_column, partition_column=a.partition_column,
                         partition_by=a.partition_by)
    log(f"{len(labels)} labelled proteins, {labels['partition'].nunique()} partitions")
    species = {p.split(".")[0] for p in labels["protein"]}
    arms = load_arms(a, species, rekey_map=idmap)
    keep = shared_proteins(arms, labels["protein"])
    if len(keep) < 2:
        raise SystemExit("error: fewer than two labelled proteins are present in every arm")
    sub = labels.set_index("protein").loc[keep].reset_index()
    Y = sub[COMPARTMENTS].to_numpy(dtype=int)
    parts = sub["partition"].to_numpy()
    log(f"{len(keep)} proteins shared by {len(arms)} arms")

    a.out.mkdir(parents=True, exist_ok=True)
    summary, curves = {}, {}
    for name, embs in arms.items():
        X = feature_matrix(embs, keep)
        folds, P = evaluate_partitions(X, Y, parts, seed=a.seed, n_jobs=a.n_jobs)
        write_folds(a.out / f"{name}_folds.csv", folds)
        summary[name] = summarize(folds, X.shape[1], len(keep))
        curves[name] = pr_curve_with_band(Y, P, n_boot=a.n_boot, seed=0)
        curves[name].update({"dim": int(X.shape[1]), "n_proteins": len(keep)})
        log(f"  {name:14s} dim={X.shape[1]:5d}  F1-micro={summary[name]['f1_micro_mean']:.4f} "
            f"+/- {summary[name]['f1_micro_sd']:.4f}  Fmax={curves[name]['fmax']:.4f}")
    write_json(a.out / "summary.json", summary)
    write_json(a.out / "pr_curves.json", curves)
    log(f"wrote {a.out / 'summary.json'} and {a.out / 'pr_curves.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
