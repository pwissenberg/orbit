#!/usr/bin/env python3
"""Subcellular localization on the Human Protein Atlas hold-out set (Fig. 4B, Section 2.8).

The counterpart of Hu et al. (2025) Fig. 3b: classifiers are trained on the DeepLoc 2.0
Swiss-Prot table and tested on the independent HPA set, which ships with DeepLoc 2.0's own
predictions as an external baseline. The embedding arms are the aligned STRING embeddings
produced before with ``orbit align`` (``results/string/<taxid>.h5``, identical to the Zenodo
release), the SPACE embeddings, ProtT5, and their concatenations and PCA controls
(``_common.py``: ``--arm/--pca/--concat``; ``--rekey`` for an arm keyed by UniProt accession).

Inputs from SPACE's ``benchmarks.zip`` (``benchmarks/deeploc/``):
``Swissprot_Train_Validation_dataset.csv`` and ``cv_idmapping.tsv`` (training labels and
their STRING ids), ``hpa_testset.csv`` and ``hpa_headers.txt`` (test labels and the six
compartments), ``9606.protein.aliases.v12.0.txt.gz`` (HPA's Ensembl translation ids and
UniProt accessions resolve to STRING ids through this table), ``hpa-deeploc-predictions.csv``.

Output under --out: ``hpa_curves.json`` with one precision-recall curve, Fmax and bootstrap
interval per arm plus the ``deeploc2`` baseline.

    python scripts/downstream/hpa.py --train benchmarks/deeploc/Swissprot_Train_Validation_dataset.csv \\
        --train-idmap benchmarks/deeploc/cv_idmapping.tsv --test benchmarks/deeploc/hpa_testset.csv \\
        --headers benchmarks/deeploc/hpa_headers.txt --aliases benchmarks/deeploc/9606.protein.aliases.v12.0.txt.gz \\
        --baseline benchmarks/deeploc/hpa-deeploc-predictions.csv \\
        --arm orbit=results/string --arm space=space/functional_emb --arm prott5=uniprot/per-protein.h5 --rekey prott5 \\
        --pca space_pca128=space:128 --concat orbit_t5=orbit+prott5 --concat space_t5=space+prott5 \\
        --out results/downstream/string_hpa
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_arm_arguments, feature_matrix, load_arms, log, read_idmap, shared_proteins, write_json  # noqa: E402
from deeploc import MultiLabelLR, pr_curve_with_band, read_labels  # noqa: E402


def read_aliases(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """From the STRING alias table: ``{Ensembl translation id: STRING id}`` and
    ``{UniProt accession: STRING id}`` (a STRING protein carries several accessions, all of
    which map onto it)."""
    ensembl, uniprot = {}, {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, alias, source = parts[0], parts[1], parts[2]
            if "Ensembl_translation" in source:
                ensembl.setdefault(alias, sid)
            if "UniProt_AC" in source:
                uniprot.setdefault(alias, sid)
    return ensembl, uniprot


def read_hpa(path: Path, compartments: list[str], ensembl_to_sid: dict[str, str]) -> pd.DataFrame:
    """``sid``, ``protein`` (STRING id) and the compartment columns of the HPA test set."""
    df = pd.read_csv(path)
    missing = [c for c in compartments if c not in df.columns]
    if missing:
        raise SystemExit(f"error: {path} lacks compartment columns {missing}")
    df["protein"] = df["sid"].astype(str).map(ensembl_to_sid)
    df = df[df["protein"].notna()].drop_duplicates("protein")
    return df[["sid", "protein"] + compartments].reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, required=True, metavar="CSV", help="DeepLoc 2.0 Swiss-Prot table")
    p.add_argument("--train-idmap", type=Path, required=True, metavar="TSV", help="cv_idmapping.tsv (ACC -> STRING id)")
    p.add_argument("--test", type=Path, required=True, metavar="CSV", help="hpa_testset.csv")
    p.add_argument("--headers", type=Path, required=True, metavar="TXT", help="hpa_headers.txt, one compartment per line")
    p.add_argument("--aliases", type=Path, required=True, metavar="GZ", help="9606.protein.aliases.v12.0.txt.gz")
    p.add_argument("--baseline", type=Path, metavar="CSV", help="hpa-deeploc-predictions.csv (DeepLoc 2.0 baseline)")
    add_arm_arguments(p)
    p.add_argument("--out", type=Path, required=True, metavar="DIR")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    compartments = [l.strip() for l in a.headers.read_text().splitlines() if l.strip()]
    ensembl_to_sid, acc_to_sid = read_aliases(a.aliases)
    train_idmap = read_idmap(a.train_idmap)
    train = read_labels(a.train, train_idmap, id_column="ACC", partition_by="none")
    test = read_hpa(a.test, compartments, ensembl_to_sid)
    log(f"{len(train)} training proteins, {len(test)} HPA proteins mapped to STRING ids")

    rekey_map = {**acc_to_sid, **train_idmap}  # accession -> STRING id, for --rekey arms
    species = {p.split(".")[0] for p in pd.concat([train["protein"], test["protein"]])}
    arms = load_arms(a, species, rekey_map=rekey_map)
    keep_train = shared_proteins(arms, train["protein"])
    keep_test = shared_proteins(arms, test["protein"])
    if len(keep_train) < 2 or not keep_test:
        raise SystemExit("error: too few training or test proteins are present in every arm")
    tr = train.set_index("protein").loc[keep_train]
    te = test.set_index("protein").loc[keep_test]
    Y_tr = tr[compartments].to_numpy(dtype=int)
    Y_te = te[compartments].to_numpy(dtype=int)
    log(f"{len(keep_train)} train and {len(keep_test)} test proteins shared by {len(arms)} arms")

    curves = {}
    for name, embs in arms.items():
        X_tr, X_te = feature_matrix(embs, keep_train), feature_matrix(embs, keep_test)
        P = MultiLabelLR(a.seed, a.n_jobs).fit(X_tr, Y_tr).predict_proba(X_te)
        curves[name] = pr_curve_with_band(Y_te, P, n_boot=a.n_boot, seed=0)
        curves[name].update({"dim": int(X_tr.shape[1]), "n_train": len(keep_train), "n_test": len(keep_test)})
        log(f"  {name:14s} dim={X_tr.shape[1]:5d}  Fmax={curves[name]['fmax']:.4f} "
            f"[{curves[name]['fmax_ci'][0]:.4f}, {curves[name]['fmax_ci'][1]:.4f}]")

    if a.baseline is not None:
        base = pd.read_csv(a.baseline).set_index("Protein_ID")
        common = [s for s in te["sid"] if s in base.index]
        if common:
            sub = te[te["sid"].isin(common)]
            P = base.loc[sub["sid"], compartments].to_numpy(dtype=float)
            curves["deeploc2"] = pr_curve_with_band(sub[compartments].to_numpy(dtype=int), P,
                                                    n_boot=a.n_boot, seed=0)
            curves["deeploc2"].update({"dim": 0, "n_train": 0, "n_test": len(common)})
            log(f"  {'deeploc2':14s} baseline   Fmax={curves['deeploc2']['fmax']:.4f} over {len(common)} proteins")

    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out / "hpa_curves.json", curves)
    log(f"wrote {a.out / 'hpa_curves.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
