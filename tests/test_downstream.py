"""One smoke test per script under scripts/downstream/: synthetic inputs in the benchmark
file formats, one run, output files present with sane values."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from orbit.io import read_embeddings, write_embeddings

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "downstream"
pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("sklearn", "pandas", "joblib", "cafaeval")),
    reason="the downstream scripts need `uv sync --extra paper`")
COMPARTMENTS = ["Cytoplasm", "Nucleus", "Cell membrane", "Plastid", "Extracellular", "Mitochondrion",
                "Endoplasmic reticulum", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"]
TERMS = ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"]


def run(name: str, argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.main(argv)


@pytest.fixture
def emb(tmp_path):
    """Two species (STRING-style ids), 8 dimensions; label k of a protein follows the sign of
    coordinate k, so classifiers have something to learn."""
    d = tmp_path / "emb"
    d.mkdir()
    rng = np.random.default_rng(0)
    for sp, n in (("9606", 60), ("10090", 40)):
        write_embeddings(d / f"{sp}.h5", rng.normal(size=(n, 8)).astype(np.float32),
                         [f"{sp}.p{i:03d}" for i in range(n)])
    return d


def labels_of(emb, sp):
    X, ids = read_embeddings(emb / f"{sp}.h5")
    return ids, (X[:, :4] > 0).astype(int)


def test_deeploc(tmp_path, emb):
    rows, idmap = ["ACC,Partition," + ",".join(COMPARTMENTS)], ["From\tTo"]
    for sp in ("9606", "10090"):
        for i, (pid, y) in enumerate(zip(*labels_of(emb, sp))):
            idmap.append(f"A{pid}\t{pid}")
            rows.append(f"A{pid},{i % 3}," + ",".join(str(v) for v in list(y) + [0] * 6))
    (tmp_path / "dl.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "map.tsv").write_text("\n".join(idmap) + "\n")
    out = tmp_path / "out"
    assert run("deeploc", ["--labels", str(tmp_path / "dl.csv"), "--idmap", str(tmp_path / "map.tsv"),
                           "--arm", f"orbit={emb}", "--pca", "pca=orbit:4", "--out", str(out), "--n-boot", "2"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["orbit"]["n_folds"] == 3 and summary["orbit"]["f1_micro_mean"] > 0.7
    assert summary["pca"]["dim"] == 4
    assert 0.5 < json.loads((out / "pr_curves.json").read_text())["orbit"]["fmax"] <= 1


def test_hpa(tmp_path, emb):
    comps = COMPARTMENTS[:6]
    train, idmap, test, aliases = ["ACC,Partition," + ",".join(comps)], ["From\tTo"], ["sid," + ",".join(comps)], []
    for sp in ("9606", "10090"):
        for i, (pid, y) in enumerate(zip(*labels_of(emb, sp))):
            y = list(y) + [0, 0]
            if sp == "10090" or i >= 25:
                idmap.append(f"A{pid}\t{pid}")
                train.append(f"A{pid},0," + ",".join(map(str, y)))
            else:
                aliases.append(f"{pid}\tENSP{i}\tEnsembl_translation")
                test.append(f"ENSP{i}," + ",".join(map(str, y)))
    for name, lines in (("train.csv", train), ("map.tsv", idmap), ("hpa.csv", test), ("headers.txt", comps)):
        (tmp_path / name).write_text("\n".join(lines) + "\n")
    with gzip.open(tmp_path / "aliases.txt.gz", "wt") as f:
        f.write("#id\talias\tsource\n" + "\n".join(aliases) + "\n")
    out = tmp_path / "out"
    assert run("hpa", ["--train", str(tmp_path / "train.csv"), "--train-idmap", str(tmp_path / "map.tsv"),
                       "--test", str(tmp_path / "hpa.csv"), "--headers", str(tmp_path / "headers.txt"),
                       "--aliases", str(tmp_path / "aliases.txt.gz"), "--arm", f"orbit={emb}",
                       "--out", str(out), "--n-boot", "2"]) == 0
    curves = json.loads((out / "hpa_curves.json").read_text())
    assert curves["orbit"]["n_test"] == 25 and curves["orbit"]["fmax"] > 0.5


def test_netgo(tmp_path, emb):
    obo = ["[Term]\nid: GO:0003674\nnamespace: molecular_function"] + [
        f"[Term]\nid: {t}\nnamespace: molecular_function\nis_a: GO:0003674" for t in TERMS]
    (tmp_path / "go.obo").write_text("\n\n".join(obo) + "\n")
    for split, sp in (("train", "9606"), ("test", "10090")):
        ann, idmap = [], ["From\tTo"]
        for pid, y in zip(*labels_of(emb, sp)):
            idmap.append(f"U{pid}\t{pid}")
            ann += [f"U{pid}\t{t}\tmf\t{sp}" for t, v in zip(TERMS, y) if v]
        (tmp_path / f"{split}.txt").write_text("\n".join(ann) + "\n")
        (tmp_path / f"{split}_map.tsv").write_text("\n".join(idmap) + "\n")
    out = tmp_path / "out"
    assert run("netgo", ["--train", str(tmp_path / "train.txt"), "--test", str(tmp_path / "test.txt"),
                         "--train-idmap", str(tmp_path / "train_map.tsv"), "--test-idmap", str(tmp_path / "test_map.tsv"),
                         "--obo", str(tmp_path / "go.obo"), "--aspects", "mf", "--arm", f"orbit={emb}",
                         "--n-boot", "2", "--out", str(out)]) == 0
    mf = json.loads((out / "scores.json").read_text())["orbit"]["aspects"]["mf"]
    assert 0.5 < mf["fmax"] <= 1 and len(mf["replicates"]) == 2
    assert (out / "orbit" / "mf_pred" / "mf_pred.tsv").read_text().count("\n") > 40


@pytest.fixture
def plant(tmp_path):
    """Gene ids without a species prefix, id-mapping tables per species, a mini ontology."""
    d, maps = tmp_path / "plant", tmp_path / "maps"
    d.mkdir(), maps.mkdir()
    rng = np.random.default_rng(1)
    for sp, fmt, n in (("ARATH", "AT1G{:05d}.1", 60), ("ORYSA", "LOC_Os01g{:05d}.1", 40)):
        ids = [fmt.format(i) for i in range(n)]
        write_embeddings(d / f"{sp}.h5", rng.normal(size=(n, 8)).astype(np.float32), ids)
        (maps / f"{sp}_to_uniprot.tsv").write_text("teagcn_id\tquery_id\tuniprot_accession\n"
                                                    + "".join(f"{g}\t{g}\tQ{g}\n" for g in ids))
    (tmp_path / "go.obo").write_text("\n\n".join(
        ["[Term]\nid: GO:0003674\nnamespace: molecular_function"]
        + [f"[Term]\nid: {t}\nnamespace: molecular_function\nis_a: GO:0003674" for t in TERMS]) + "\n")
    return d, maps


def test_go_transfer(tmp_path, plant):
    d, maps = plant
    rows = ["accession\tgo_id"]
    for sp in ("ARATH", "ORYSA"):
        for pid, y in zip(*labels_of(d, sp)):
            rows += [f"Q{pid}\t{t}" for t, v in zip(TERMS, y) if v]
    (tmp_path / "labels.tsv").write_text("\n".join(rows) + "\n")
    out = tmp_path / "out"
    assert run("go_transfer", ["--labels", str(tmp_path / "labels.tsv"), "--label-columns", "accession,go_id",
                               "--obo", str(tmp_path / "go.obo"), "--idmap-dir", str(maps), "--train", "ARATH",
                               "--test", "ORYSA", "--arm", f"orbit={d}", "--out", str(out)]) == 0
    res = json.loads((out / "transfer.json").read_text())
    r = res["results"][0]
    assert r["aspect"] == "MF" and r["n_terms"] == 4 and r["fmax"] > 0.6
    assert res["summary"]["orbit"]["MF"]["n_species"] == 1


def test_kegg_labels(tmp_path, plant):
    d, maps = plant
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kegg_ath_links.tsv").write_text("path:ath00010\tath:AT1G00001\npath:ath01100\tath:AT1G00001\n")
    (cache / "kegg_osa_links.tsv").write_text("path:osa00010\tosa:1\n")
    (cache / "kegg_osa_to_uniprot.tsv").write_text("osa:1\tup:QLOC_Os01g00005.1\n")
    out = tmp_path / "kegg.tsv"
    assert run("kegg_labels", ["--species", "ARATH=ath", "ORYSA=osa", "--embeddings", str(d), "--idmap-dir", str(maps),
                               "--cache", str(cache), "--out", str(out)]) == 0
    assert out.read_text() == "protein\tterm\taspect\nAT1G00001.1\t00010\tKEGG\nLOC_Os01g00005.1\t00010\tKEGG\n"
