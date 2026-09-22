"""The downstream benchmark scripts under scripts/downstream/, run on synthetic inputs that
follow the benchmark file formats (SPACE's benchmarks.zip, DeepLoc 2.0, NetGO 2.0, KEGG)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from orbit.io import write_embeddings

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "downstream"

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("sklearn", "pandas", "joblib", "cafaeval")),
    reason="the downstream scripts need `uv sync --extra paper`")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def common():
    return load_script("_common")


# --- fixtures ---------------------------------------------------------------------------

def make_species(rng, prefix, n, dim):
    X = rng.normal(size=(n, dim)).astype(np.float32)
    ids = [f"{prefix}.p{i:04d}" for i in range(n)]
    return X, ids


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def embedding_dir(tmp_path, rng):
    """Two species, 60 and 40 proteins, 8 dimensions, written as <species>.h5."""
    d = tmp_path / "emb"
    d.mkdir()
    for sp, n in (("9606", 60), ("10090", 40)):
        X, ids = make_species(rng, sp, n, 8)
        write_embeddings(d / f"{sp}.h5", X, ids)
    return d


@pytest.fixture
def flat_h5(tmp_path, rng):
    """ProtT5-style file: one dataset per protein, 6 dimensions, for 50 of the 100 proteins."""
    path = tmp_path / "t5.h5"
    with h5py.File(path, "w") as f:
        for sp, n in (("9606", 30), ("10090", 20)):
            for i in range(n):
                f.create_dataset(f"{sp}.p{i:04d}", data=rng.normal(size=6))
    return path


# --- _common: loading and combining arms ------------------------------------------------

def test_load_embedding_dir_keys_vectors_by_protein_id(common, embedding_dir):
    embs = common.load_embedding_dir(embedding_dir)
    assert len(embs) == 100
    assert embs["9606.p0003"].shape == (8,)
    assert embs["9606.p0003"].dtype == np.float32


def test_load_embedding_dir_restricted_to_species(common, embedding_dir):
    embs = common.load_embedding_dir(embedding_dir, species={"10090", "4932"})
    assert len(embs) == 40
    assert all(k.startswith("10090.") for k in embs)


def test_load_embedding_dir_loads_everything_when_no_species_matches_a_file(common, embedding_dir):
    """Plant gene ids (AT1G01010.1) have no species prefix, so a filter derived from them must
    not empty the arm."""
    embs = common.load_embedding_dir(embedding_dir, species={"AT1G01010", "LOC_Os01g01010"})
    assert len(embs) == 100


def test_load_arm_reads_a_directory_a_flat_h5_or_an_orbit_h5(common, embedding_dir, flat_h5):
    assert len(common.load_arm(embedding_dir)) == 100
    flat = common.load_arm(flat_h5)
    assert len(flat) == 50 and flat["9606.p0001"].dtype == np.float32
    single = common.load_arm(embedding_dir / "10090.h5")
    assert len(single) == 40


def test_concat_keeps_only_shared_proteins_and_stacks_blocks(common):
    a = {"x": np.array([1.0, 0.0], np.float32), "y": np.array([0.0, 2.0], np.float32)}
    b = {"y": np.array([3.0], np.float32), "z": np.array([4.0], np.float32)}
    c = common.concat(a, b)
    assert set(c) == {"y"}
    np.testing.assert_array_equal(c["y"], [0.0, 2.0, 3.0])


def test_concat_normalize_scales_each_block_to_unit_length(common):
    c = common.concat({"y": np.array([0.0, 2.0], np.float32)},
                      {"y": np.array([3.0, 4.0], np.float32)}, normalize=True)
    np.testing.assert_allclose(c["y"], [0.0, 1.0, 0.6, 0.8], atol=1e-6)


def test_pca_reduce_projects_to_requested_dimension_deterministically(common):
    rng = np.random.default_rng(1)
    embs = {f"p{i}": rng.normal(size=16).astype(np.float32) for i in range(50)}
    z1, z2 = common.pca_reduce(embs, 4, seed=42), common.pca_reduce(embs, 4, seed=42)
    assert set(z1) == set(embs) and z1["p7"].shape == (4,) and z1["p7"].dtype == np.float32
    np.testing.assert_allclose(z1["p7"], z2["p7"])


def test_feature_matrix_stacks_rows_in_the_given_order(common):
    X = common.feature_matrix({"a": np.array([1, 2], np.float32), "b": np.array([3, 4], np.float32)}, ["b", "a"])
    np.testing.assert_array_equal(X, [[3, 4], [1, 2]])


def test_arm_arguments_build_every_declared_arm(common, embedding_dir, flat_h5):
    import argparse
    p = argparse.ArgumentParser()
    common.add_arm_arguments(p)
    a = p.parse_args(["--arm", f"orbit={embedding_dir}", "--arm", f"prott5={flat_h5}",
                      "--pca", "orbit_pca=orbit:4", "--concat", "orbit_t5=orbit+prott5"])
    arms = common.load_arms(a)
    assert list(arms) == ["orbit", "prott5", "orbit_pca", "orbit_t5"]
    assert arms["orbit_pca"]["9606.p0001"].shape == (4,)
    assert arms["orbit_t5"]["9606.p0001"].shape == (14,)
    assert len(arms["orbit_t5"]) == 50


def test_arm_arguments_normalize_concat_gives_unit_blocks(common, embedding_dir, flat_h5):
    import argparse
    p = argparse.ArgumentParser()
    common.add_arm_arguments(p)
    a = p.parse_args(["--arm", f"orbit={embedding_dir}", "--arm", f"t5={flat_h5}",
                      "--concat", "both=orbit+t5", "--normalize-concat"])
    v = common.load_arms(a)["both"]["9606.p0002"]
    assert np.isclose(np.linalg.norm(v[:8]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(v[8:]), 1.0, atol=1e-5)


def test_arm_arguments_reject_unknown_reference_and_bad_syntax(common, embedding_dir):
    import argparse
    p = argparse.ArgumentParser()
    common.add_arm_arguments(p)
    with pytest.raises(SystemExit):
        common.load_arms(p.parse_args(["--arm", f"orbit={embedding_dir}", "--concat", "x=orbit+nope"]))
    with pytest.raises(SystemExit):
        common.load_arms(p.parse_args(["--arm", "orbit"]))


def test_shared_proteins_is_the_sorted_intersection_over_arms(common):
    arms = {"a": {"p2": 0, "p1": 0, "p3": 0}, "b": {"p1": 0, "p3": 0}}
    assert common.shared_proteins(arms) == ["p1", "p3"]
    assert common.shared_proteins(arms, ["p3", "p9"]) == ["p3"]


def test_read_idmap_keep_prefers_a_target_that_is_present(common, tmp_path):
    """An accession listed first with an isoform absent from the embedding must still map onto
    the isoform that is present."""
    f = tmp_path / "ORYSA_to_uniprot.tsv"
    f.write_text("teagcn_id\tquery_id\tuniprot_accession\nOs01g0100200.2\tx\tQ6ZL45\nOs01g0100200.1\tx\tQ6ZL45\n")
    cols = ("uniprot_accession", "teagcn_id")
    assert common.read_idmap(f, cols) == {"Q6ZL45": "Os01g0100200.2"}
    assert common.read_idmap(f, cols, keep={"Os01g0100200.1"}) == {"Q6ZL45": "Os01g0100200.1"}


def test_rekey_maps_arm_ids_through_a_mapping_and_drops_the_rest(common, embedding_dir, flat_h5):
    """ProtT5 files are keyed by UniProt accession; --rekey ARM maps them onto the embedding
    ids so the arm can intersect with the aligned embeddings."""
    import argparse
    p = argparse.ArgumentParser()
    common.add_arm_arguments(p)
    a = p.parse_args(["--arm", f"orbit={embedding_dir}", "--arm", f"t5={flat_h5}", "--rekey", "t5"])
    mapping = {"9606.p0001": "X1", "9606.p0002": "X2", "nope": "X3"}
    arms = common.load_arms(a, rekey_map=mapping)
    assert set(arms["t5"]) == {"X1", "X2"}
    assert len(arms["orbit"]) == 100


def test_rekey_of_an_undeclared_arm_is_an_error(common, embedding_dir):
    import argparse
    p = argparse.ArgumentParser()
    common.add_arm_arguments(p)
    with pytest.raises(SystemExit, match="unknown arm"):
        common.load_arms(p.parse_args(["--arm", f"orbit={embedding_dir}", "--rekey", "t5"]), rekey_map={})
    with pytest.raises(SystemExit, match="id mapping"):
        common.load_arms(p.parse_args(["--arm", f"orbit={embedding_dir}", "--rekey", "orbit"]))


def test_read_idmap_uses_from_to_columns_by_default(common, tmp_path):
    f = tmp_path / "map.tsv"
    f.write_text("From\tTo\nP1\t9606.a\nP2\t9606.b\n")
    assert common.read_idmap(f) == {"P1": "9606.a", "P2": "9606.b"}


def test_read_idmap_accepts_named_columns_for_the_plant_tables(common, tmp_path):
    f = tmp_path / "ARATH_to_uniprot.tsv"
    f.write_text("teagcn_id\tquery_id\tuniprot_accession\nAT1G01010.1\tx\tQ0WV96\n")
    assert common.read_idmap(f, columns=("uniprot_accession", "teagcn_id")) == {"Q0WV96": "AT1G01010.1"}


# --- deeploc.py: subcellular localization by held-out partition -------------------------

def read_csv_rows(path):
    import csv as _csv
    with open(path, newline="") as f:
        return list(_csv.DictReader(f))


def labelled_accessions(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return set(df.loc[df[COMPARTMENTS].sum(axis=1) > 0, "ACC"])


COMPARTMENTS = ["Cytoplasm", "Nucleus", "Cell membrane", "Plastid", "Extracellular",
                "Mitochondrion", "Endoplasmic reticulum", "Golgi apparatus",
                "Lysosome/Vacuole", "Peroxisome"]


def planted_labels(X: np.ndarray) -> np.ndarray:
    """Four compartments follow the sign of one coordinate each; six are never annotated."""
    Y = np.zeros((len(X), len(COMPARTMENTS)), dtype=int)
    for c in range(4):
        Y[:, c] = (X[:, c] > 0).astype(int)
    return Y


@pytest.fixture
def deeploc_inputs(tmp_path, embedding_dir):
    """DeepLoc-style CSV keyed by UniProt accession plus an id mapping onto the embedding
    ids, three homology partitions."""
    import pandas as pd
    from orbit.io import read_embeddings

    rows, idmap = [], []
    for sp in ("9606", "10090"):
        X, ids = read_embeddings(embedding_dir / f"{sp}.h5")
        Y = planted_labels(X)
        for i, pid in enumerate(ids):
            acc = f"ACC{sp}{i:04d}"
            idmap.append((acc, pid))
            rows.append({"ACC": acc, "Partition": i % 3, "Sequence": "M" * 5,
                         **{c: int(Y[i, j]) for j, c in enumerate(COMPARTMENTS)}})
    csv_path = tmp_path / "Swissprot_Train_Validation_dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    map_path = tmp_path / "cv_idmapping.tsv"
    map_path.write_text("From\tTo\n" + "".join(f"{a}\t{b}\n" for a, b in idmap))
    return csv_path, map_path


def test_deeploc_runs_one_fold_per_partition_and_beats_chance(tmp_path, embedding_dir, deeploc_inputs):
    import json
    deeploc = load_script("deeploc")
    labels, idmap = deeploc_inputs
    n_labelled = len(labelled_accessions(labels))
    assert 80 < n_labelled < 100  # a few proteins carry no compartment and must be dropped
    out = tmp_path / "out"
    rc = deeploc.main(["--labels", str(labels), "--idmap", str(idmap),
                       "--arm", f"orbit={embedding_dir}", "--out", str(out), "--n-boot", "5"])
    assert rc == 0
    folds = read_csv_rows(out / "orbit_folds.csv")
    assert len(folds) == 3
    assert sorted(int(f["partition"]) for f in folds) == [0, 1, 2]
    assert sum(int(f["n_test"]) for f in folds) == n_labelled
    expected = ["fold", "partition", "n_train", "n_test", "f1_micro", "f1_macro", "accuracy",
                "jaccard_micro"] + [f"mcc_{c}" for c in COMPARTMENTS]
    assert list(folds[0].keys()) == expected
    summary = json.loads((out / "summary.json").read_text())
    assert summary["orbit"]["n_folds"] == 3 and summary["orbit"]["dim"] == 8
    assert summary["orbit"]["n_proteins"] == n_labelled
    assert summary["orbit"]["f1_micro_mean"] > 0.75
    curves = json.loads((out / "pr_curves.json").read_text())
    assert 0.5 < curves["orbit"]["fmax"] <= 1.0
    lo, hi = curves["orbit"]["fmax_ci"]
    assert 0 <= lo <= hi <= 1
    assert len(curves["orbit"]["grid"]) == len(curves["orbit"]["lo"]) == len(curves["orbit"]["hi"]) == 101


def test_deeploc_scores_every_arm_on_the_shared_protein_set(tmp_path, embedding_dir, flat_h5, deeploc_inputs):
    import json
    deeploc = load_script("deeploc")
    labels, idmap = deeploc_inputs
    out = tmp_path / "out"
    deeploc.main(["--labels", str(labels), "--idmap", str(idmap), "--arm", f"orbit={embedding_dir}",
                  "--arm", f"prott5={flat_h5}", "--concat", "orbit_t5=orbit+prott5",
                  "--out", str(out), "--n-boot", "2"])
    summary = json.loads((out / "summary.json").read_text())
    acc_to_pid = dict(l.split("\t") for l in idmap.read_text().splitlines()[1:])
    with h5py.File(flat_h5) as f:
        in_t5 = set(f.keys())
    expected = len({acc_to_pid[a] for a in labelled_accessions(labels)} & in_t5)
    assert 30 < expected < 50
    assert {s["n_proteins"] for s in summary.values()} == {expected}
    assert summary["orbit_t5"]["dim"] == 14


def test_deeploc_perfectly_separable_features_reach_fmax_one(tmp_path, deeploc_inputs):
    import json
    import pandas as pd
    deeploc = load_script("deeploc")
    labels, idmap = deeploc_inputs
    df = pd.read_csv(labels)
    acc_to_pid = dict(l.split("\t") for l in idmap.read_text().splitlines()[1:])
    cheat = tmp_path / "cheat"
    cheat.mkdir()
    for sp in ("9606", "10090"):
        sub = df[df["ACC"].str.startswith(f"ACC{sp}")]
        Y = sub[COMPARTMENTS].to_numpy(dtype=np.float32) * 6 - 3
        write_embeddings(cheat / f"{sp}.h5", Y, [acc_to_pid[a] for a in sub["ACC"]])
    out = tmp_path / "out"
    deeploc.main(["--labels", str(labels), "--idmap", str(idmap), "--arm", f"cheat={cheat}",
                  "--out", str(out), "--n-boot", "3"])
    curves = json.loads((out / "pr_curves.json").read_text())
    assert curves["cheat"]["fmax"] == pytest.approx(1.0)


def test_deeploc_long_labels_partition_by_species(tmp_path, embedding_dir):
    """The leave-one-species-out protocol of Fig. S4: long-format annotations, one fold per
    species, unknown compartment names ignored."""
    from orbit.io import read_embeddings
    deeploc = load_script("deeploc")
    lines = ["protein\tcompartment"]
    for sp in ("9606", "10090"):
        X, ids = read_embeddings(embedding_dir / f"{sp}.h5")
        for i, pid in enumerate(ids):
            for c in range(3):
                if X[i, c] > 0:
                    lines.append(f"{pid}\t{COMPARTMENTS[c]}")
            lines.append(f"{pid}\tNot a compartment")
    labels = tmp_path / "subloc.tsv"
    labels.write_text("\n".join(lines) + "\n")
    out = tmp_path / "out"
    rc = deeploc.main(["--labels", str(labels), "--partition-by", "species",
                       "--arm", f"orbit={embedding_dir}", "--out", str(out), "--n-boot", "2"])
    assert rc == 0
    folds = read_csv_rows(out / "orbit_folds.csv")
    assert sorted(f["partition"] for f in folds) == ["10090", "9606"]
    assert all(0 < int(f["n_test"]) <= 60 for f in folds)


def test_deeploc_help_names_the_aligned_embeddings(capsys):
    deeploc = load_script("deeploc")
    with pytest.raises(SystemExit):
        deeploc.main(["--help"])
    assert "orbit align" in capsys.readouterr().out


# --- hpa.py: Human Protein Atlas hold-out ------------------------------------------------

HPA_COMPARTMENTS = COMPARTMENTS[:6]


@pytest.fixture
def hpa_inputs(tmp_path, embedding_dir, deeploc_inputs):
    """Test set keyed by Ensembl translation ids that resolve to STRING ids through the alias
    table; DeepLoc 2.0's own predictions as the external baseline."""
    import gzip
    import pandas as pd
    from orbit.io import read_embeddings

    X, ids = read_embeddings(embedding_dir / "9606.h5")
    Y = planted_labels(X)
    rows, alias_lines = [], []
    for i, pid in enumerate(ids[:25]):
        sid = f"ENSP{i:06d}"
        alias_lines.append(f"{pid}\t{sid}\tEnsembl_translation")
        alias_lines.append(f"{pid}\tACC9606{i:04d}\tUniProt_AC")
        rows.append({"sid": sid, **{c: int(Y[i, j]) for j, c in enumerate(HPA_COMPARTMENTS)}})
    alias_lines.append("9606.unrelated\tENSP999999\tEnsembl_translation")
    test_csv = tmp_path / "hpa_testset.csv"
    pd.DataFrame(rows).to_csv(test_csv, index=False)
    headers = tmp_path / "hpa_headers.txt"
    headers.write_text("\n".join(HPA_COMPARTMENTS) + "\n")
    aliases = tmp_path / "9606.protein.aliases.v12.0.txt.gz"
    with gzip.open(aliases, "wt") as f:
        f.write("#string_protein_id\talias\tsource\n" + "\n".join(alias_lines) + "\n")
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"Protein_ID": [r["sid"] for r in rows],
                         **{c: rng.uniform(size=len(rows)) for c in HPA_COMPARTMENTS}})
    baseline = tmp_path / "hpa-deeploc-predictions.csv"
    base.to_csv(baseline, index=False)
    train_csv, train_idmap = deeploc_inputs
    return dict(train=train_csv, train_idmap=train_idmap, test=test_csv, headers=headers,
                aliases=aliases, baseline=baseline)


def test_hpa_maps_the_test_set_through_the_alias_table_and_scores_every_arm(tmp_path, embedding_dir, hpa_inputs):
    import json
    hpa = load_script("hpa")
    out = tmp_path / "out"
    rc = hpa.main(["--train", str(hpa_inputs["train"]), "--train-idmap", str(hpa_inputs["train_idmap"]),
                   "--test", str(hpa_inputs["test"]), "--headers", str(hpa_inputs["headers"]),
                   "--aliases", str(hpa_inputs["aliases"]), "--baseline", str(hpa_inputs["baseline"]),
                   "--arm", f"orbit={embedding_dir}", "--out", str(out), "--n-boot", "4"])
    assert rc == 0
    curves = json.loads((out / "hpa_curves.json").read_text())
    assert set(curves) == {"orbit", "deeploc2"}
    assert curves["orbit"]["n_test"] == 25 and curves["orbit"]["dim"] == 8
    assert curves["orbit"]["fmax"] > 0.6
    lo, hi = curves["orbit"]["fmax_ci"]
    assert 0 <= lo <= hi <= 1
    assert curves["deeploc2"]["n_test"] == 25 and "fmax_ci" in curves["deeploc2"]
    assert curves["orbit"]["n_train"] < 100  # training proteins from the DeepLoc table only


def test_hpa_baseline_with_a_duplicated_protein_id_still_scores(tmp_path, embedding_dir, hpa_inputs):
    import json
    hpa = load_script("hpa")
    baseline = hpa_inputs["baseline"]
    lines = baseline.read_text().splitlines()
    baseline.write_text("\n".join(lines + [lines[1]]) + "\n")  # first protein listed twice
    out = tmp_path / "out"
    hpa.main(["--train", str(hpa_inputs["train"]), "--train-idmap", str(hpa_inputs["train_idmap"]),
              "--test", str(hpa_inputs["test"]), "--headers", str(hpa_inputs["headers"]),
              "--aliases", str(hpa_inputs["aliases"]), "--baseline", str(baseline),
              "--arm", f"orbit={embedding_dir}", "--out", str(out), "--n-boot", "2"])
    curves = json.loads((out / "hpa_curves.json").read_text())
    assert curves["deeploc2"]["n_test"] == 25


def test_hpa_rekeys_uniprot_keyed_arms_through_idmap_and_aliases(tmp_path, embedding_dir, hpa_inputs):
    """A ProtT5 file keyed by accession covers training proteins via cv_idmapping.tsv and test
    proteins via the UniProt_AC aliases; --rekey joins both."""
    import json
    hpa = load_script("hpa")
    t5 = tmp_path / "t5_by_accession.h5"
    rng = np.random.default_rng(5)
    with h5py.File(t5, "w") as f:
        for sp, n in (("9606", 60), ("10090", 40)):
            for i in range(n):
                f.create_dataset(f"ACC{sp}{i:04d}", data=rng.normal(size=6))
    out = tmp_path / "out"
    hpa.main(["--train", str(hpa_inputs["train"]), "--train-idmap", str(hpa_inputs["train_idmap"]),
              "--test", str(hpa_inputs["test"]), "--headers", str(hpa_inputs["headers"]),
              "--aliases", str(hpa_inputs["aliases"]), "--arm", f"orbit={embedding_dir}",
              "--arm", f"prott5={t5}", "--rekey", "prott5", "--concat", "orbit_t5=orbit+prott5",
              "--out", str(out), "--n-boot", "2"])
    curves = json.loads((out / "hpa_curves.json").read_text())
    assert curves["orbit_t5"]["dim"] == 14 and curves["orbit_t5"]["n_test"] == 25
    assert curves["prott5"]["n_train"] == curves["orbit"]["n_train"]


# --- netgo.py: GO term prediction under the NetGO 2.0 protocol ---------------------------

MF_ROOT = "GO:0003674"
MF_TERMS = ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"]
RARE_TERM = "GO:0000005"


def mini_obo(path):
    stanzas = [f"[Term]\nid: {MF_ROOT}\nname: molecular_function\nnamespace: molecular_function\n"]
    for t in MF_TERMS + [RARE_TERM]:
        stanzas.append(f"[Term]\nid: {t}\nname: term {t}\nnamespace: molecular_function\nis_a: {MF_ROOT} ! molecular_function\n")
    path.write_text("format-version: 1.2\n\n" + "\n".join(stanzas))


@pytest.fixture
def netgo_inputs(tmp_path, embedding_dir):
    """train.txt / test.txt (uniprot, term, aspect, taxon), id mappings onto the embedding
    ids, a five-term ontology. Four terms follow one coordinate each; the fifth has three
    positives and must be skipped. One training protein belongs to a foreign taxon."""
    from orbit.io import read_embeddings

    def rows(sp, taxon, rare):
        X, ids = read_embeddings(embedding_dir / f"{sp}.h5")
        out, mapping, unlabelled = [], [], []
        for i, pid in enumerate(ids):
            uni = f"U{sp}{i:04d}"
            mapping.append((uni, pid))
            if not (X[i, :4] > 0).any():
                unlabelled.append(pid)
            for k, t in enumerate(MF_TERMS):
                if X[i, k] > 0:
                    out.append((uni, t, "mf", taxon))
            if rare and i < 3:
                out.append((uni, RARE_TERM, "mf", taxon))
        return out, mapping, unlabelled

    train, train_map, unlabelled = rows("9606", 9606, rare=True)
    test, test_map, _ = rows("10090", 10090, rare=False)
    # A foreign-taxon annotation on a vector that carries no other label: --taxa drops it.
    train.append(("Uforeign", MF_TERMS[0], "mf", 9999))
    train_map.append(("Uforeign", unlabelled[0]))
    d = tmp_path / "netgo"
    d.mkdir()
    (d / "n_train.txt").write_text(str(len({u for u, *_ in train})))
    (d / "n_test.txt").write_text(str(len({u for u, *_ in test})))
    for name, data, mapping in (("train", train, train_map), ("test", test, test_map)):
        (d / f"{name}.txt").write_text("".join(f"{u}\t{t}\t{a}\t{x}\n" for u, t, a, x in data))
        (d / f"{name}_idmapping_euk.tsv").write_text("From\tTo\n" + "".join(f"{a}\t{b}\n" for a, b in mapping))
    obo = d / "go.obo"
    mini_obo(obo)
    return d


def netgo_args(d, out, *arms, extra=()):
    return ["--train", str(d / "train.txt"), "--test", str(d / "test.txt"),
            "--train-idmap", str(d / "train_idmapping_euk.tsv"), "--test-idmap", str(d / "test_idmapping_euk.tsv"),
            "--obo", str(d / "go.obo"), "--aspects", "mf", "--out", str(out), *arms, *extra]


def test_netgo_writes_cafa_predictions_and_skips_rare_terms(tmp_path, embedding_dir, netgo_inputs):
    import json
    netgo = load_script("netgo")
    out = tmp_path / "out"
    rc = netgo.main(netgo_args(netgo_inputs, out, "--arm", f"orbit={embedding_dir}", extra=["--n-boot", "0"]))
    assert rc == 0
    pred = (out / "orbit" / "mf_pred" / "mf_pred.tsv").read_text().splitlines()
    cols = [l.split("\t") for l in pred]
    assert all(len(c) == 3 for c in cols)
    assert all(c[0].startswith("U10090") for c in cols)          # test proteins by uniprot id
    assert all(float(c[2]) > 0.01 for c in cols)                 # SPACE keeps scores above 0.01
    assert RARE_TERM not in {c[1] for c in cols}                 # fewer than 10 positives
    assert {c[1] for c in cols} == set(MF_TERMS)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["orbit"]["aspects"]["mf"]["n_terms"] == 4
    n_train, n_test = int((netgo_inputs / "n_train.txt").read_text()), int((netgo_inputs / "n_test.txt").read_text())
    assert manifest["orbit"]["n_train"] == n_train and manifest["orbit"]["n_test"] == n_test
    scores = json.loads((out / "scores.json").read_text())
    assert 0.5 < scores["orbit"]["aspects"]["mf"]["fmax"] <= 1.0
    assert "auprc" in scores["orbit"]["aspects"]["mf"]
    gt = (out / "ground_truth" / "mf_gt.tsv").read_text().splitlines()
    assert all(len(l.split("\t")) == 2 for l in gt)


def test_netgo_taxa_filter_drops_foreign_training_proteins(tmp_path, embedding_dir, netgo_inputs):
    import json
    netgo = load_script("netgo")
    out = tmp_path / "out"
    netgo.main(netgo_args(netgo_inputs, out, "--arm", f"orbit={embedding_dir}",
                          extra=["--n-boot", "0", "--taxa", "9606", "10090"]))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["orbit"]["n_train"] == int((netgo_inputs / "n_train.txt").read_text()) - 1


def test_netgo_perfect_features_reach_fmax_one_and_bootstrap_is_paired(tmp_path, embedding_dir, netgo_inputs):
    import json
    from orbit.io import read_embeddings
    netgo = load_script("netgo")
    cheat = tmp_path / "cheat"
    cheat.mkdir()
    for sp in ("9606", "10090"):
        X, ids = read_embeddings(embedding_dir / f"{sp}.h5")
        Y = np.column_stack([(X[:, k] > 0) for k in range(4)]).astype(np.float32) * 6 - 3
        write_embeddings(cheat / f"{sp}.h5", Y, ids)
    out = tmp_path / "out"
    netgo.main(netgo_args(netgo_inputs, out, "--arm", f"cheat={cheat}", "--arm", f"orbit={embedding_dir}",
                          extra=["--n-boot", "3"]))
    scores = json.loads((out / "scores.json").read_text())
    assert scores["cheat"]["aspects"]["mf"]["fmax"] == pytest.approx(1.0, abs=1e-6)
    assert scores["orbit"]["aspects"]["mf"]["fmax"] < scores["cheat"]["aspects"]["mf"]["fmax"]
    for arm in ("cheat", "orbit"):
        e = scores[arm]["aspects"]["mf"]
        assert len(e["replicates"]) == 3 and e["ci_low"] <= e["ci_high"]
    assert scores["cheat"]["aspects"]["mf"]["replicate_seeds"] == scores["orbit"]["aspects"]["mf"]["replicate_seeds"]


def test_netgo_uses_provided_ground_truth_files(tmp_path, embedding_dir, netgo_inputs):
    """SPACE ships test_<aspect>_ground_truth.txt; when given, it replaces the ground truth
    derived from test.txt."""
    import json
    netgo = load_script("netgo")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "test_mf_ground_truth.txt").write_text("U100900001\tGO:0000001\n")
    out = tmp_path / "out"
    netgo.main(netgo_args(netgo_inputs, out, "--arm", f"orbit={embedding_dir}",
                          extra=["--n-boot", "0", "--ground-truth-dir", str(gt_dir)]))
    scores = json.loads((out / "scores.json").read_text())
    assert scores["orbit"]["aspects"]["mf"]["n_gt_proteins"] == 1


# --- go_transfer.py: term-centric cross-species transfer (GO or KEGG labels) --------------

PLANT = {"ARATH": ("AT1G{:05d}.1", "Qa{:04d}", 60), "ORYSA": ("LOC_Os01g{:05d}.1", "Qo{:04d}", 40),
         "ZEAMA": ("Zm00001d{:06d}_P001", "Qz{:04d}", 20)}


@pytest.fixture
def plant_inputs(tmp_path, rng):
    """Plant-style embeddings (gene ids without a species prefix), a CAFA-style label table
    keyed by UniProt accession, per-species id mappings and the mini ontology. Term k follows
    coordinate k; GO:0000004 is never annotated in the test species; the rare term has three
    training positives."""
    emb = tmp_path / "plant"
    emb.mkdir()
    idmaps = tmp_path / "idmaps"
    idmaps.mkdir()
    label_rows = ["accession\tgo_id"]
    for sp, (gene_fmt, acc_fmt, n) in PLANT.items():
        X = rng.normal(size=(n, 8)).astype(np.float32)
        ids = [gene_fmt.format(i) for i in range(n)]
        write_embeddings(emb / f"{sp}.h5", X, ids)
        lines = ["teagcn_id\tquery_id\tuniprot_accession"]
        for i, gid in enumerate(ids):
            acc = acc_fmt.format(i)
            lines.append(f"{gid}\t{gid}\t{acc}")
            for k, t in enumerate(MF_TERMS):
                if X[i, k] > 0 and not (sp != "ARATH" and t == "GO:0000004"):
                    label_rows.append(f"{acc}\t{t}")
            if sp == "ARATH" and i < 3:
                label_rows.append(f"{acc}\t{RARE_TERM}")
        (idmaps / f"{sp}_to_uniprot.tsv").write_text("\n".join(lines) + "\n")
    labels = tmp_path / "go_labels.tsv"
    labels.write_text("\n".join(label_rows) + "\n")
    obo = tmp_path / "go.obo"
    mini_obo(obo)
    return dict(emb=emb, idmaps=idmaps, labels=labels, obo=obo)


def transfer_args(inp, out, *arms, test=("ORYSA", "ZEAMA")):
    return ["--labels", str(inp["labels"]), "--label-columns", "accession,go_id", "--obo", str(inp["obo"]),
            "--idmap-dir", str(inp["idmaps"]), "--idmap-columns", "uniprot_accession,teagcn_id",
            "--train", "ARATH", "--test", *test, "--out", str(out), *arms]


def test_go_transfer_evaluates_eligible_terms_and_weights_species_by_test_size(tmp_path, plant_inputs):
    import json
    go_transfer = load_script("go_transfer")
    out = tmp_path / "out"
    rc = go_transfer.main(transfer_args(plant_inputs, out, "--arm", f"orbit={plant_inputs['emb']}"))
    assert rc == 0
    res = json.loads((out / "transfer.json").read_text())
    rows = [r for r in res["results"] if r["arm"] == "orbit"]
    assert {(r["test_species"], r["aspect"]) for r in rows} == {("ORYSA", "MF"), ("ZEAMA", "MF")}
    for r in rows:
        assert r["n_terms"] == 3                       # 4 planted terms, one absent from the test species
        assert set(r["terms"]) == set(MF_TERMS) - {"GO:0000004"}
        assert len(r["fmax_per_term"]) == len(r["auprc_per_term"]) == 3
        assert r["fmax"] > 0.6
        assert 40 < r["n_train"] < 60                  # only genes carrying a label take part
    n = {r["test_species"]: r["n_test"] for r in rows}
    assert 25 < n["ORYSA"] < 40 and 8 < n["ZEAMA"] < 20
    f = {r["test_species"]: r["fmax"] for r in rows}
    expected = (f["ORYSA"] * n["ORYSA"] + f["ZEAMA"] * n["ZEAMA"]) / (n["ORYSA"] + n["ZEAMA"])
    assert res["summary"]["orbit"]["MF"]["fmax_weighted"] == pytest.approx(expected)
    assert res["summary"]["orbit"]["MF"]["n_species"] == 2


def test_go_transfer_reports_bad_arm_syntax_and_missing_arms_clearly(tmp_path, plant_inputs):
    go_transfer = load_script("go_transfer")
    with pytest.raises(SystemExit, match="--arm must be written"):
        go_transfer.main(transfer_args(plant_inputs, tmp_path / "o", "--arm", "orbit"))
    with pytest.raises(SystemExit, match="at least one --arm"):
        go_transfer.main(transfer_args(plant_inputs, tmp_path / "o", "--species-dir", str(plant_inputs["emb"])))


def test_go_transfer_rekeys_an_accession_keyed_arm_through_the_idmap_dir(tmp_path, plant_inputs):
    """ProtT5 per-protein files are keyed by accession; --rekey maps them onto gene ids with
    the per-species tables so they can be concatenated with the aligned embeddings."""
    import json
    go_transfer = load_script("go_transfer")
    t5 = tmp_path / "t5.h5"
    rng = np.random.default_rng(7)
    with h5py.File(t5, "w") as f:
        for sp, (_, acc_fmt, n) in PLANT.items():
            for i in range(n):
                f.create_dataset(acc_fmt.format(i), data=rng.normal(size=5))
    out = tmp_path / "out"
    rc = go_transfer.main(transfer_args(plant_inputs, out, "--arm", f"orbit={plant_inputs['emb']}",
                                        "--arm", f"prott5={t5}", "--rekey", "prott5",
                                        "--concat", "orbit_t5=orbit+prott5", "--normalize-concat", test=("ORYSA",)))
    assert rc == 0
    res = json.loads((out / "transfer.json").read_text())
    by_arm = {r["arm"]: r for r in res["results"]}
    assert by_arm["prott5"]["n_train"] == by_arm["orbit"]["n_train"]
    assert by_arm["orbit_t5"]["n_test"] == by_arm["orbit"]["n_test"]


def test_go_transfer_perfect_features_reach_fmax_one(tmp_path, plant_inputs):
    import json
    from orbit.io import read_embeddings
    go_transfer = load_script("go_transfer")
    cheat = tmp_path / "cheat"
    cheat.mkdir()
    for sp in PLANT:
        X, ids = read_embeddings(plant_inputs["emb"] / f"{sp}.h5")
        Y = np.column_stack([(X[:, k] > 0) for k in range(4)]).astype(np.float32) * 6 - 3
        write_embeddings(cheat / f"{sp}.h5", Y, ids)
    out = tmp_path / "out"
    go_transfer.main(transfer_args(plant_inputs, out, "--arm", f"cheat={cheat}", test=("ORYSA",)))
    res = json.loads((out / "transfer.json").read_text())
    assert res["results"][0]["fmax"] == pytest.approx(1.0)


def test_go_transfer_accepts_a_prebuilt_label_table_in_embedding_id_space(tmp_path, plant_inputs):
    """The KEGG route (Table S6): protein, term, aspect already in embedding-id space, no
    ontology and no id mapping."""
    import json
    from orbit.io import read_embeddings
    go_transfer = load_script("go_transfer")
    lines = ["protein\tterm\taspect"]
    for sp in ("ARATH", "ORYSA"):
        X, ids = read_embeddings(plant_inputs["emb"] / f"{sp}.h5")
        for i, gid in enumerate(ids):
            for k in range(2):
                if X[i, k] > 0:
                    lines.append(f"{gid}\t000{k}0\tKEGG")
    labels = tmp_path / "kegg_labels.tsv"
    labels.write_text("\n".join(lines) + "\n")
    out = tmp_path / "out"
    rc = go_transfer.main(["--labels", str(labels), "--train", "ARATH", "--test", "ORYSA",
                           "--arm", f"orbit={plant_inputs['emb']}", "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "transfer.json").read_text())
    assert res["results"][0]["aspect"] == "KEGG" and res["results"][0]["n_terms"] == 2


# --- kegg_labels.py: KEGG pathway membership in embedding-id space -----------------------

def test_kegg_labels_builds_the_table_from_cached_kegg_files(tmp_path, plant_inputs):
    """ARATH genes map by locus (preferring the .1 isoform); other species go KEGG -> UniProt
    -> embedding id; overview maps 01100 and 01110 are excluded. No network access: the REST
    responses are pre-populated in the cache."""
    kegg_labels = load_script("kegg_labels")
    cache = tmp_path / "kegg_cache"
    cache.mkdir()
    (cache / "kegg_ath_links.tsv").write_text(
        "path:ath00010\tath:AT1G00001\npath:ath01100\tath:AT1G00001\npath:ath00020\tath:AT1G00002\n"
        "path:ath00020\tath:AT9G99999\n")
    (cache / "kegg_osa_links.tsv").write_text("path:osa00010\tosa:4326813\npath:osa01110\tosa:4326813\n")
    (cache / "kegg_osa_to_uniprot.tsv").write_text("osa:4326813\tup:Qo0005\n")
    out = tmp_path / "kegg_labels.tsv"
    rc = kegg_labels.main(["--species", "ARATH=ath", "ORYSA=osa", "--embeddings", str(plant_inputs["emb"]),
                           "--idmap-dir", str(plant_inputs["idmaps"]), "--idmap-columns", "uniprot_accession,teagcn_id",
                           "--cache", str(cache), "--out", str(out)])
    assert rc == 0
    rows = {tuple(l.split("\t")) for l in out.read_text().splitlines()[1:]}
    assert rows == {("AT1G00001.1", "00010", "KEGG"), ("AT1G00002.1", "00020", "KEGG"),
                    ("LOC_Os01g00005.1", "00010", "KEGG")}
    assert out.read_text().splitlines()[0] == "protein\tterm\taspect"


def test_kegg_labels_maps_an_accession_whose_first_isoform_is_absent(tmp_path, plant_inputs):
    kegg_labels = load_script("kegg_labels")
    idmap = plant_inputs["idmaps"] / "ORYSA_to_uniprot.tsv"
    idmap.write_text(idmap.read_text() + "LOC_Os01g00005.9\tx\tQo9999\nLOC_Os01g00005.1\tx\tQo9999\n")
    cache = tmp_path / "kegg_cache"
    cache.mkdir()
    (cache / "kegg_osa_links.tsv").write_text("path:osa00010\tosa:1\n")
    (cache / "kegg_osa_to_uniprot.tsv").write_text("osa:1\tup:Qo9999\n")
    out = tmp_path / "kegg_labels.tsv"
    kegg_labels.main(["--species", "ORYSA=osa", "--embeddings", str(plant_inputs["emb"]),
                      "--idmap-dir", str(plant_inputs["idmaps"]), "--cache", str(cache), "--out", str(out)])
    assert out.read_text().splitlines()[1:] == ["LOC_Os01g00005.1\t00010\tKEGG"]


def test_kegg_labels_prefers_the_first_isoform_for_arath(tmp_path):
    kegg_labels = load_script("kegg_labels")
    ids = ["AT1G00001.2", "AT1G00001.1", "AT1G00002.3"]
    assert kegg_labels.arath_locus_map(ids) == {"AT1G00001": "AT1G00001.1", "AT1G00002": "AT1G00002.3"}
