#!/usr/bin/env python3
"""GO term prediction from aligned embeddings under the NetGO 2.0 protocol of Hu et al. (2025).

Reproduces Fig. 3A (plant subset of the benchmark, Section 2.7) and Fig. 4C / Fig. S6
(STRING at full scale, Section 2.8). The embedding arms are the aligned embeddings produced
before with ``orbit align`` (plant: ``results/plant/<SPECIES>.h5``; STRING:
``results/string/<taxid>.h5``, identical to the Zenodo release), the ProtT5 embeddings, the
FedCoder/SPACE baseline, and their concatenations and PCA controls (``_common.py``).

Inputs from SPACE's ``benchmarks.zip`` (``benchmarks/netgo/``): ``train.txt`` and
``test.txt`` (UniProt accession, GO term, aspect, taxon), ``train_idmapping_euk.tsv`` and
``test_idmapping_euk.tsv`` (accession -> STRING id), ``test_<aspect>_ground_truth.txt``,
``go_2020_10_09.obo`` and ``netgo_t5.h5``. The plant run keeps only the plant taxa
(``--taxa 3702 39947 4530 3847 3880 3694 4577``), maps accessions onto TEA-GCN ids with the
``<SPECIES>_to_uniprot.tsv`` tables (``--idmap-columns uniprot_accession,teagcn_id``), scores
against ``gene_ontology_edit.obo.2017-11-01`` and concatenates with ``--normalize-concat``.

Per aspect, one logistic regression per GO term with at least ``--min-positives`` annotated
training proteins is fitted on the proteins annotated in that aspect; test predictions above
``--keep-above`` are written in CAFA format and scored with CAFA-evaluator (protein-centric
Fmax, AUPRC, Smin). ``--n-boot`` resamples the ground-truth proteins, identically for every
arm, so paired comparisons between arms are possible.

Outputs under --out: ``<arm>/<aspect>_pred/<aspect>_pred.tsv``, ``ground_truth/<aspect>_gt.tsv``,
``manifest.json``, ``scores.json``.

    python scripts/downstream/netgo.py --train benchmarks/netgo/train.txt --test benchmarks/netgo/test.txt \\
        --train-idmap benchmarks/netgo/train_idmapping_euk.tsv --test-idmap benchmarks/netgo/test_idmapping_euk.tsv \\
        --ground-truth-dir benchmarks/netgo --obo benchmarks/netgo/go_2020_10_09.obo \\
        --arm orbit=results/string --arm space=space/functional_emb --arm prott5=benchmarks/netgo/netgo_t5.h5 \\
        --pca space_pca128=space:128 --concat orbit_t5=orbit+prott5 --concat space_t5=space+prott5 \\
        --n-boot 200 --out results/downstream/string_netgo
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (add_arm_arguments, derive_arms, feature_matrix, load_arms, log, read_idmap,  # noqa: E402
                     require_paper_extra, shared_proteins, write_json)

require_paper_extra()
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

NAMESPACE = {"mf": "molecular_function", "bp": "biological_process", "cc": "cellular_component"}
SEED = 42
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz


# --- inputs -------------------------------------------------------------------------------

def read_split(path: Path, idmap: dict[str, str], taxa: set[str] | None = None) -> pd.DataFrame:
    """``uniprot``, ``term``, ``aspect`` (lower case), ``taxon`` and the embedding id
    ``protein`` of every annotation whose accession has a mapping (and whose taxon is in
    ``taxa``, if given)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["uniprot", "term", "aspect", "taxon"], dtype=str)
    df["aspect"] = df["aspect"].str.lower()
    if taxa is not None:
        df = df[df["taxon"].isin({str(t) for t in taxa})]
    df["protein"] = df["uniprot"].map(idmap)
    return df[df["protein"].notna()].drop_duplicates().reset_index(drop=True)


# --- fitting ------------------------------------------------------------------------------

def fit_term(pos: set[str], train_ids: list[str], X_train: np.ndarray, X_test: np.ndarray,
             min_positives: int, seed: int) -> np.ndarray | None:
    y = np.fromiter((p in pos for p in train_ids), dtype=np.int8, count=len(train_ids))
    if y.sum() < min_positives or y.sum() == len(y):
        return None
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X_train, y)
    return clf.predict_proba(X_test)[:, 1].astype(np.float32)


def predict_aspect(train: pd.DataFrame, aspect: str, keep_train: list[str], X_train: np.ndarray,
                   X_test: np.ndarray, *, min_positives: int, seed: int, n_jobs: int,
                   ) -> tuple[list[str], list[np.ndarray]]:
    """Per-term probabilities on the test matrix for every term of ``aspect`` with enough
    positives; training uses only the proteins annotated in that aspect (SPACE's
    ``func_pred.py``)."""
    tr = train[train["aspect"] == aspect]
    term_pos = tr.groupby("term")["protein"].apply(set)
    terms = sorted(t for t, p in term_pos.items() if len(p & set(keep_train)) >= min_positives)
    annotated = set(tr["protein"])
    index = {p: i for i, p in enumerate(keep_train)}
    asp_ids = [p for p in keep_train if p in annotated]
    if not terms or not asp_ids:
        return [], []
    X_asp = X_train[[index[p] for p in asp_ids]]
    preds = Parallel(n_jobs=n_jobs)(
        delayed(fit_term)(term_pos[t], asp_ids, X_asp, X_test, min_positives, seed) for t in terms)
    kept = [(t, p) for t, p in zip(terms, preds) if p is not None]
    return [t for t, _ in kept], [p for _, p in kept]


def write_predictions(path: Path, terms: list[str], preds: list[np.ndarray], test_uniprot: list[str],
                      keep_above: float) -> int:
    """CAFA prediction file ``uniprot<TAB>term<TAB>score`` with scores above ``keep_above``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for term, p in zip(terms, preds):
            for i in np.flatnonzero(p > keep_above):
                f.write(f"{test_uniprot[i]}\t{term}\t{p[i]:.6f}\n")
                n += 1
    return n


# --- scoring ------------------------------------------------------------------------------

def score(obo: Path, pred_dir: Path, gt_file: Path, aspect: str, th_step: float, n_cpu: int = 1) -> dict:
    """Fmax, AUPRC and Smin of CAFA-evaluator for one aspect."""
    from cafaeval.evaluation import cafa_eval

    df, _ = cafa_eval(str(obo), str(pred_dir), str(gt_file), th_step=th_step, n_cpu=n_cpu)
    if df is None or df.empty:
        return {"fmax": 0.0}
    df = df.reset_index()
    if "ns" in df.columns and NAMESPACE.get(aspect) in set(df["ns"]):
        df = df[df["ns"] == NAMESPACE[aspect]]
    out = {"fmax": float(df["f"].max())}
    if "s" in df.columns:
        out["smin"] = float(df["s"].min())
    if {"rc", "pr"} <= set(df.columns):
        d = df[["rc", "pr"]].dropna().sort_values("rc")
        if len(d) > 1:
            out["auprc"] = float(_trapezoid(d["pr"].to_numpy(), d["rc"].to_numpy()))
    return out


def _replicate(seed: int, gt_by: dict, pred_by: dict, prots: np.ndarray, obo: Path, aspect: str,
               th_step: float) -> float | None:
    """One bootstrap draw. Proteins drawn k times appear k times under renamed ids in BOTH
    the ground truth and the predictions, which keeps the bootstrap weights exact."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(rng.choice(prots, size=len(prots), replace=True), return_counts=True)
    gt_lines, pred_lines = [], []
    for p, k in zip(uniq, counts):
        for r in range(int(k)):
            tag = f"{p}__{r}"
            gt_lines.extend(f"{tag}\t{t}" for t in gt_by.get(p, ()))
            pred_lines.extend(f"{tag}\t{rest}" for rest in pred_by.get(p, ()))
    if not pred_lines:
        return None
    tmp = Path(tempfile.mkdtemp(prefix=f"netgo_boot{seed}_"))
    try:
        (tmp / "pred").mkdir()
        (tmp / "pred" / "p.tsv").write_text("\n".join(pred_lines) + "\n")
        (tmp / "gt.tsv").write_text("\n".join(gt_lines) + "\n")
        return score(obo, tmp / "pred", tmp / "gt.tsv", aspect, th_step)["fmax"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def paired_bootstrap(pred_file: Path, gt_file: Path, obo: Path, aspect: str, *, n_boot: int,
                     th_step: float, n_jobs: int) -> dict:
    """Bootstrap Fmax over ground-truth proteins. The protein list is sorted and replicate
    ``i`` uses seed ``i``, so every arm sees the same resample in replicate ``i``."""
    gt = pd.read_csv(gt_file, sep="\t", header=None, names=["u", "t"], dtype=str)
    pred = pd.read_csv(pred_file, sep="\t", header=None, names=["u", "t", "s"], dtype=str)
    prots = np.sort(gt["u"].unique())
    pred = pred[pred["u"].isin(set(prots))]
    gt_by = {p: g["t"].tolist() for p, g in gt.groupby("u", sort=False)}
    pred_by = {p: (q["t"] + "\t" + q["s"]).tolist() for p, q in pred.groupby("u", sort=False)}
    vals = Parallel(n_jobs=n_jobs)(
        delayed(_replicate)(i, gt_by, pred_by, prots, obo, aspect, th_step) for i in range(n_boot))
    ok = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not ok:
        return {"n_boot": 0}
    v = np.array([x for _, x in ok])
    return {"boot_mean": float(v.mean()), "boot_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)),
            "n_boot": len(v), "replicate_seeds": [i for i, _ in ok], "replicates": [float(x) for x in v]}


# --- main ---------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, required=True, metavar="TXT", help="NetGO train.txt")
    p.add_argument("--test", type=Path, required=True, metavar="TXT", help="NetGO test.txt")
    p.add_argument("--train-idmap", type=Path, required=True, metavar="TSV", help="accession -> embedding id")
    p.add_argument("--test-idmap", type=Path, metavar="TSV", help="default: --train-idmap")
    p.add_argument("--idmap-columns", default="From,To", metavar="SRC,DST")
    p.add_argument("--taxa", nargs="*", metavar="TAXID", help="keep only these taxa (the plant subset)")
    p.add_argument("--obo", type=Path, required=True, metavar="OBO", help="ontology for CAFA-evaluator")
    p.add_argument("--ground-truth-dir", type=Path, metavar="DIR",
                   help="directory with test_<aspect>_ground_truth.txt (default: derived from --test)")
    p.add_argument("--aspects", nargs="+", default=["mf", "cc", "bp"])
    p.add_argument("--min-positives", type=int, default=10)
    p.add_argument("--keep-above", type=float, default=0.01)
    p.add_argument("--th-step", type=float, default=0.001, help="CAFA-evaluator threshold step")
    p.add_argument("--n-boot", type=int, default=0, help="paired bootstrap replicates (0: none)")
    p.add_argument("--th-boot", type=float, default=0.01, help="threshold step inside the bootstrap")
    add_arm_arguments(p)
    p.add_argument("--out", type=Path, required=True, metavar="DIR")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cols = tuple(a.idmap_columns.split(","))
    train_idmap = read_idmap(a.train_idmap, cols)
    test_idmap = read_idmap(a.test_idmap, cols) if a.test_idmap else train_idmap
    taxa = set(a.taxa) if a.taxa else None
    train, test = read_split(a.train, train_idmap, taxa), read_split(a.test, test_idmap, taxa)
    log(f"train: {len(train)} annotations, {train['protein'].nunique()} proteins; "
        f"test: {len(test)} annotations, {test['protein'].nunique()} proteins")

    species = {p.split(".")[0] for p in pd.concat([train["protein"], test["protein"]])}
    arms = load_arms(a, species, rekey_map={**test_idmap, **train_idmap})
    keep_train = shared_proteins(arms, train["protein"])
    keep_test = shared_proteins(arms, test["protein"])
    if len(keep_train) < a.min_positives or not keep_test:
        raise SystemExit("error: too few training or test proteins are present in every arm")
    arms = derive_arms(a, arms, keep_train + keep_test)
    train = train[train["protein"].isin(keep_train)]
    test = test[test["protein"].isin(keep_test)]
    test_uniprot = test.drop_duplicates("protein").set_index("protein")["uniprot"].loc[keep_test].tolist()
    log(f"{len(keep_train)} train and {len(keep_test)} test proteins shared by {len(arms)} arms")

    gt_dir = a.out / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_files = {}
    for aspect in a.aspects:
        gt_files[aspect] = gt_dir / f"{aspect}_gt.tsv"
        provided = a.ground_truth_dir / f"test_{aspect}_ground_truth.txt" if a.ground_truth_dir else None
        if provided is not None:
            if not provided.exists():
                raise SystemExit(f"error: {provided} not found")
            shutil.copyfile(provided, gt_files[aspect])
        else:
            sub = test[test["aspect"] == aspect]
            gt_files[aspect].write_text("".join(f"{u}\t{t}\n" for u, t in zip(sub["uniprot"], sub["term"])))

    manifest, scores = {}, {}
    for name, embs in arms.items():
        X_train, X_test = feature_matrix(embs, keep_train), feature_matrix(embs, keep_test)
        manifest[name] = {"dim": int(X_train.shape[1]), "n_train": len(keep_train), "n_test": len(keep_test), "aspects": {}}
        scores[name] = {"dim": int(X_train.shape[1]), "aspects": {}}
        for aspect in a.aspects:
            terms, preds = predict_aspect(train, aspect, keep_train, X_train, X_test, min_positives=a.min_positives,
                                          seed=a.seed, n_jobs=a.n_jobs)
            if not terms:
                log(f"  {name}/{aspect}: no term with {a.min_positives} positives, skipped")
                continue
            pred_dir = a.out / name / f"{aspect}_pred"
            n_pred = write_predictions(pred_dir / f"{aspect}_pred.tsv", terms, preds, test_uniprot, a.keep_above)
            manifest[name]["aspects"][aspect] = {"n_terms": len(terms), "n_predictions": n_pred}
            entry = score(a.obo, pred_dir, gt_files[aspect], aspect, a.th_step)
            entry["n_gt_proteins"] = int(pd.read_csv(gt_files[aspect], sep="\t", header=None, dtype=str)[0].nunique())
            if a.n_boot:
                entry.update(paired_bootstrap(pred_dir / f"{aspect}_pred.tsv", gt_files[aspect], a.obo, aspect,
                                              n_boot=a.n_boot, th_step=a.th_boot, n_jobs=a.n_jobs))
            scores[name]["aspects"][aspect] = entry
            ci = f"  95% CI [{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]" if "ci_low" in entry else ""
            log(f"  {name:14s} {aspect}: {len(terms)} terms  Fmax={entry['fmax']:.4f}{ci}")
        write_json(a.out / "manifest.json", manifest)
        write_json(a.out / "scores.json", scores)
    log(f"wrote {a.out / 'scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
