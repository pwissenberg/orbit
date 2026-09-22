#!/usr/bin/env python3
"""Cross-species transfer of GO terms or KEGG pathways from aligned embeddings.

Reproduces Fig. 3C and Fig. S5 (GO transfer from A. thaliana to rice, maize, soybean and
Medicago, Section 2.7) and Table S6 (KEGG pathway transfer, labels from ``kegg_labels.py``).
The embedding arms are the aligned plant embeddings produced before with ``orbit align``
(``results/plant/<SPECIES>.h5``; the five seed files equal the Zenodo release), ProtT5, the
FedCoder-plant baseline and their concatenations (``_common.py``). Unlike the other plant
benchmarks, this one joins the raw vectors and standardises the result, so do not pass
``--normalize-concat`` to reproduce Fig. 3C. Species membership of a gene comes from the
per-species files of ``--species-dir`` (default: the first ``--arm`` directory).

Labels: a TSV with a header. For GO, a table of UniProt accessions and GO ids
(``--label-columns accession,go_id``) holding the GOA annotations with experimental
evidence codes; the aspect of every term is taken from ``--obo`` (go-basic.obo), the
annotations are propagated to their ancestors along ``is_a`` and ``part_of`` (Section 2.7;
``--no-propagate`` to skip), and the accessions are mapped onto gene ids per species with
``--idmap-dir/<SPECIES>_to_uniprot.tsv``. For KEGG, the ``protein``, ``term``, ``aspect``
table written by ``kegg_labels.py``, already in gene-id space.

One classifier per term is trained on the training species (``StandardScaler`` fitted on it,
logistic regression, terms with at least ``--min-train-positives`` positives) and applied to
every test species where the term has a positive; the term-centric Fmax from the
precision-recall curve is averaged over terms, and species are combined weighted by their
number of test proteins.

Output under --out: ``transfer.json`` with every (arm, test species, aspect) result including
the per-term values, and the weighted summary.

    python scripts/downstream/go_transfer.py --labels data/go_labels_cafa_expanded.tsv \\
        --label-columns accession,go_id --obo data/go-basic.obo --idmap-dir data/id_mapping \\
        --train ARATH --test ORYSA ZEAMA GLYMA MEDTR \\
        --arm orbit=results/plant --arm fedcoder=results/fedcoder_plant --arm prott5=data/prott5 \\
        --concat orbit_t5=orbit+prott5 --concat fedcoder_t5=fedcoder+prott5 \\
        --out results/downstream/plant_go_transfer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (add_arm_arguments, derive_arms, feature_matrix, load_arms, log, read_idmap,  # noqa: E402
                     require_paper_extra, write_json)
from orbit.io import read_ids  # noqa: E402

require_paper_extra()
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 42
ASPECT_OF_NAMESPACE = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}


# --- inputs -------------------------------------------------------------------------------

def read_obo(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """``({GO id: MF|BP|CC}, {GO id: direct parents})`` from an OBO file; parents follow
    ``is_a`` and ``relationship: part_of``, the edges CAFA-evaluator propagates along."""
    aspects: dict[str, str] = {}
    parents: dict[str, set[str]] = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current = None
            elif line.startswith("id: GO:"):
                current = line.split(": ", 1)[1]
                parents.setdefault(current, set())
            elif current is None:
                continue
            elif line.startswith("namespace:"):
                ns = line.split(": ", 1)[1]
                if ns in ASPECT_OF_NAMESPACE:
                    aspects[current] = ASPECT_OF_NAMESPACE[ns]
            elif line.startswith("is_a:"):
                parents[current].add(line.split(": ", 1)[1].split()[0])
            elif line.startswith("relationship: part_of"):
                parents[current].add(line.split()[2])
            elif line.startswith("is_obsolete: true"):
                aspects.pop(current, None)
    return aspects, parents


def ancestors(parents: dict[str, set[str]]) -> dict[str, set[str]]:
    """Transitive closure of ``parents``, each term excluded from its own set."""
    memo: dict[str, set[str]] = {}

    def up(t: str) -> set[str]:
        if t not in memo:
            memo[t] = set()
            for p in parents.get(t, ()):
                memo[t] |= {p} | up(p)
        return memo[t]

    return {t: up(t) for t in parents}


def propagate(labels: pd.DataFrame, parents: dict[str, set[str]], aspects: dict[str, str]) -> pd.DataFrame:
    """Add, for every annotation, the ancestor terms of the same aspect."""
    anc = ancestors(parents)
    rows = {(p, t) for p, t in zip(labels["protein"], labels["term"])}
    for p, t in list(rows):
        rows.update((p, a) for a in anc.get(t, ()) if aspects.get(a) == aspects.get(t))
    out = pd.DataFrame(sorted(rows), columns=["protein", "term"])
    out["aspect"] = out["term"].map(aspects)
    return out


def read_labels(path: Path, columns: tuple[str, str], aspects: dict[str, str] | None) -> pd.DataFrame:
    """``protein``, ``term``, ``aspect`` rows. With ``aspects`` (from the OBO) the aspect of
    every term is looked up and unknown terms are dropped; otherwise the table's ``aspect``
    column is used."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    src, term = columns
    for c in (src, term):
        if c not in df.columns:
            raise SystemExit(f"error: {path} has no column {c!r}; found {list(df.columns)}")
    out = pd.DataFrame({"protein": df[src], "term": df[term]})
    if aspects is not None:
        out["aspect"] = out["term"].map(aspects)
        out = out[out["aspect"].notna()]
    elif "aspect" in df.columns:
        out["aspect"] = df["aspect"]
    else:
        raise SystemExit(f"error: {path} has no 'aspect' column and no --obo was given")
    return out.drop_duplicates().reset_index(drop=True)


def species_labels(labels: pd.DataFrame, species: str, gene_ids: list[str],
                   idmap: dict[str, str] | None) -> dict[str, dict[str, set[str]]]:
    """``{aspect: {gene: set(terms)}}`` for the genes of one species, after mapping the label
    ids onto gene ids when an id mapping is given."""
    df = labels
    if idmap is not None:
        df = df.assign(protein=df["protein"].map(idmap))
        df = df[df["protein"].notna()]
    genes = set(gene_ids)
    df = df[df["protein"].isin(genes)]
    out: dict[str, dict[str, set[str]]] = {}
    for aspect, protein, term in zip(df["aspect"], df["protein"], df["term"]):
        out.setdefault(aspect, {}).setdefault(protein, set()).add(term)
    return out


# --- evaluation ---------------------------------------------------------------------------

def _fit_one_term(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                  seed: int) -> tuple[float, float]:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed).fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    pr, rc, _ = precision_recall_curve(y_test, scores)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    return float(f.max()), float(average_precision_score(y_test, scores))


def transfer_terms(X_train: np.ndarray, train_ids: list[str], train_labels: dict[str, set[str]],
                   X_test: np.ndarray, test_ids: list[str], test_labels: dict[str, set[str]], *,
                   min_train_positives: int = 10, seed: int = SEED, n_jobs: int = -1) -> dict:
    """Term-centric transfer: every term with ``min_train_positives`` training positives and
    at least one test positive gets one classifier; returns the per-term Fmax and AUPRC and
    their means. A term carried by every training gene (the aspect root after propagation)
    has no negatives and is skipped. Features are standardised on the training species."""
    terms = sorted({t for p in train_ids for t in train_labels.get(p, ())})
    eligible = []
    for t in terms:
        n_tr = sum(t in train_labels.get(p, ()) for p in train_ids)
        n_te = sum(t in test_labels.get(p, ()) for p in test_ids)
        if n_tr >= min_train_positives and n_te >= 1 and n_tr < len(train_ids):
            eligible.append(t)
    result = {"n_terms": len(eligible), "n_train": len(train_ids), "n_test": len(test_ids),
              "fmax": 0.0, "fmax_std": 0.0, "auprc": 0.0, "auprc_std": 0.0,
              "terms": eligible, "fmax_per_term": [], "auprc_per_term": []}
    if not eligible:
        return result
    scaler = StandardScaler().fit(X_train)
    X_tr, X_te = scaler.transform(X_train), scaler.transform(X_test)
    ys = [(np.array([t in train_labels.get(p, ()) for p in train_ids], dtype=np.int8),
           np.array([t in test_labels.get(p, ()) for p in test_ids], dtype=np.int8)) for t in eligible]
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_one_term)(X_tr, X_te, y_tr, y_te, seed) for y_tr, y_te in ys)
    fm, ap = np.array([o[0] for o in out]), np.array([o[1] for o in out])
    result.update({"fmax": float(fm.mean()), "fmax_std": float(fm.std()), "auprc": float(ap.mean()),
                   "auprc_std": float(ap.std()), "fmax_per_term": fm.tolist(), "auprc_per_term": ap.tolist()})
    return result


def weighted_summary(results: list[dict]) -> dict:
    """Per arm and aspect: the mean over test species weighted by the number of test
    proteins (Section 2.7), plus the unweighted mean."""
    summary: dict = {}
    for arm in dict.fromkeys(r["arm"] for r in results):
        summary[arm] = {}
        for aspect in dict.fromkeys(r["aspect"] for r in results if r["arm"] == arm):
            rs = [r for r in results if r["arm"] == arm and r["aspect"] == aspect and r["n_terms"] > 0]
            if not rs:
                continue
            w = np.array([r["n_test"] for r in rs], dtype=float)
            summary[arm][aspect] = {
                "fmax_weighted": float(np.average([r["fmax"] for r in rs], weights=w)),
                "fmax_unweighted": float(np.mean([r["fmax"] for r in rs])),
                "auprc_weighted": float(np.average([r["auprc"] for r in rs], weights=w)),
                "n_species": len(rs), "n_terms_total": int(sum(r["n_terms"] for r in rs)),
            }
    return summary


# --- main ---------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True, metavar="TSV")
    p.add_argument("--label-columns", default="protein,term", metavar="ID,TERM",
                   help="columns of --labels holding the protein id and the term (default protein,term)")
    p.add_argument("--obo", type=Path, metavar="OBO",
                   help="assign GO aspects from this ontology and propagate annotations to ancestors")
    p.add_argument("--no-propagate", action="store_true", help="do not propagate to ancestor terms")
    p.add_argument("--idmap-dir", type=Path, metavar="DIR",
                   help="directory of <SPECIES>_to_uniprot.tsv tables mapping label ids onto gene ids")
    p.add_argument("--idmap-columns", default="uniprot_accession,teagcn_id", metavar="SRC,DST")
    p.add_argument("--species-dir", type=Path, metavar="DIR",
                   help="per-species <SPECIES>.h5 files that define species membership (default: first --arm)")
    p.add_argument("--train", required=True, metavar="SPECIES")
    p.add_argument("--test", nargs="+", required=True, metavar="SPECIES")
    p.add_argument("--min-train-positives", type=int, default=10)
    add_arm_arguments(p)
    p.add_argument("--out", type=Path, required=True, metavar="DIR")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    aspects, parents = read_obo(a.obo) if a.obo else (None, None)
    labels = read_labels(a.labels, tuple(a.label_columns.split(",")), aspects)
    if aspects is not None and not a.no_propagate:
        before = len(labels)
        labels = propagate(labels, parents, aspects)
        log(f"{before} annotations propagated to {len(labels)} along is_a and part_of")
    all_species = [a.train] + list(a.test)
    cols = tuple(a.idmap_columns.split(","))
    idmaps: dict[str, dict[str, str] | None] = {}
    for sp in all_species:
        idmaps[sp] = read_idmap(a.idmap_dir / f"{sp}_to_uniprot.tsv", cols) if a.idmap_dir else None
    rekey_map = {k: v for m in idmaps.values() if m for k, v in m.items()} if a.idmap_dir else None
    base = load_arms(a, set(all_species), rekey_map=rekey_map)
    species_dir = a.species_dir or Path(a.arm[0].split("=", 1)[1])
    if not species_dir.is_dir():
        raise SystemExit(f"error: {species_dir} is not a directory of <SPECIES>.h5 files (--species-dir)")
    genes = {sp: read_ids(species_dir / f"{sp}.h5") for sp in all_species}
    per_species = {}
    for sp in all_species:
        idmap = idmaps[sp]
        if idmap is not None:
            # prefer, for every accession, the isoform that has an embedding
            idmap = read_idmap(a.idmap_dir / f"{sp}_to_uniprot.tsv", cols, keep=genes[sp])
        per_species[sp] = species_labels(labels, sp, genes[sp], idmap)
        n = len({p for d in per_species[sp].values() for p in d})
        log(f"{sp}: {n} labelled genes over {sorted(per_species[sp])}")

    def labelled(sp, embs):
        have = {p for d in per_species[sp].values() for p in d}
        return [g for g in genes[sp] if g in have and g in embs]

    in_all = {g for sp in all_species for g in genes[sp] if all(g in arm for arm in base.values())}
    fit_ids = [g for sp in all_species for g in labelled(sp, in_all)]
    arms = derive_arms(a, base, fit_ids)
    results = []
    for name, embs in arms.items():
        train_ids = labelled(a.train, embs)
        if not train_ids:
            log(f"  {name}: no labelled training gene present, skipped")
            continue
        X_train = feature_matrix(embs, train_ids)
        for sp in a.test:
            test_ids = labelled(sp, embs)
            if not test_ids:
                log(f"  {name}/{sp}: no labelled test gene present, skipped")
                continue
            X_test = feature_matrix(embs, test_ids)
            for aspect in sorted(set(per_species[a.train]) & set(per_species[sp])):
                r = transfer_terms(X_train, train_ids, per_species[a.train][aspect], X_test, test_ids,
                                   per_species[sp][aspect], min_train_positives=a.min_train_positives,
                                   seed=a.seed, n_jobs=a.n_jobs)
                results.append({"arm": name, "test_species": sp, "aspect": aspect, **r})
                log(f"  {name:14s} {sp:6s} {aspect:4s} terms={r['n_terms']:4d} n_test={r['n_test']:5d} "
                    f"Fmax={r['fmax']:.3f}")
    write_json(a.out / "transfer.json", {
        "train_species": a.train, "test_species": list(a.test), "min_train_positives": a.min_train_positives,
        "results": results, "summary": weighted_summary(results)})
    log(f"wrote {a.out / 'transfer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
