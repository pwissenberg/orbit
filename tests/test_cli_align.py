"""End-to-end tests of `orbit align` on small synthetic datasets of both tracks.

The plant-like dataset uses a directory of OrthoFinder tables (pair anchors), the STRING-like
dataset an eggNOG members file (centroid anchors). Every species is the same latent geometry
in a private orientation, so the rotation can recover the correspondence and the checks are
sharp.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from orbit.cli import main
from orbit.io import read_embeddings, write_embeddings

D = 16


def random_rotation(rng, d=D):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def report(out):
    lines = (out / "alignment_report.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    assert header == ["species", "role", "aligned_to", "n_anchors", "n_genes", "status"]
    return {l.split("\t")[0]: dict(zip(header, l.split("\t"))) for l in lines[1:]}


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


# --- plant-like: OrthoFinder tables, pair anchors ---------------------------------------

@pytest.fixture
def plant(tmp_path):
    """Seeds ARATH, ORYSA, PICAB; non-seeds ABROB, TRIAE. 60 latent orthogroups, one gene
    each. PICAB shares only 20 orthogroups with anyone, ABROB carries 20 shared and 10
    private ones."""
    rng = np.random.default_rng(0)
    emb = tmp_path / "node2vec"
    emb.mkdir()
    ogdir = tmp_path / "orthogroups"
    ogdir.mkdir()
    Z = rng.standard_normal((60, D))
    rows_of = {"ARATH": range(0, 50), "ORYSA": range(0, 50), "PICAB": range(0, 50),
               "ABROB": list(range(0, 20)) + list(range(50, 60)), "TRIAE": range(0, 50)}
    for s, rows in rows_of.items():
        rows = list(rows)
        Q = random_rotation(rng)
        X = (Z[rows] + 0.01 * rng.standard_normal((len(rows), D))) @ Q
        ids = [f"{s}_g{i}.1" for i in rows]
        write_embeddings(emb / f"{s}.h5", X.astype(np.float32), ids)
        with open(ogdir / f"{s}_transcripts_to_OG.tsv", "w") as f:
            f.write("Transcript_ID\tProtein_ID\tOrthogroup\n")
            for i, g in zip(rows, ids):
                og = f"OGPRIV{i}" if (s == "PICAB" and i >= 20) else f"OG{i:07d}"
                f.write(f"t\t{g}\t{og}\n")
    (tmp_path / "seeds.txt").write_text("ARATH\nORYSA\nPICAB\n")
    return tmp_path


def plant_args(base, out, *extra):
    return ["align", "--embeddings", str(base / "node2vec"), "--orthogroups", str(base / "orthogroups"),
            "--seeds", str(base / "seeds.txt"), "--reference", "ARATH", "--out", str(out), *map(str, extra)]


def test_plant_track_end_to_end(plant):
    out, rot = plant / "aligned", plant / "rotations"
    assert main(plant_args(plant, out, "--rotations", rot)) == 0
    for s in ["ARATH", "ORYSA", "PICAB", "ABROB", "TRIAE"]:
        assert (out / f"{s}.h5").exists(), s
        R = np.load(rot / f"{s}.npy")
        assert R.dtype == np.float32
        assert R.shape == (D, D) and np.allclose(R.T @ R, np.eye(D), atol=1e-5)
    # the reference is untouched
    Xa, ida = read_embeddings(plant / "node2vec" / "ARATH.h5")
    Ya, idb = read_embeddings(out / "ARATH.h5")
    assert ida == idb and np.array_equal(Xa, Ya)
    assert np.array_equal(np.load(rot / "ARATH.npy"), np.eye(D, dtype=np.float32))
    # within-species geometry is an isometry of the input
    Xo, _ = read_embeddings(plant / "node2vec" / "ORYSA.h5")
    Yo, _ = read_embeddings(out / "ORYSA.h5")
    assert np.allclose(np.linalg.norm(Xo, axis=1), np.linalg.norm(Yo, axis=1), rtol=1e-4)
    # orthologs co-localise: latent row i of every species lands on the same point
    Yt, idt = read_embeddings(out / "TRIAE.h5")
    by_row = lambda ids, Y: {i.split("_g")[1]: v for i, v in zip(ids, Y)}
    a, t = by_row(idb, Ya), by_row(idt, Yt)
    assert np.mean([cosine(a[k], t[k]) for k in sorted(set(a) & set(t))]) > 0.95
    rows = report(out)
    assert rows["ARATH"]["role"] == "reference" and rows["ARATH"]["aligned_to"] == ""
    assert rows["ORYSA"]["role"] == "seed" and rows["ORYSA"]["aligned_to"] == "ARATH"
    assert rows["TRIAE"]["role"] == "non-seed" and rows["TRIAE"]["aligned_to"] in {"ARATH", "ORYSA"}
    assert rows["ORYSA"]["n_anchors"] == "50" and rows["ORYSA"]["n_genes"] == "50"
    assert all(r["status"] == "ok" for r in rows.values())


def test_assignment_file_pins_the_seed_of_every_non_seed(plant):
    assignment = plant / "seed_assignment.json"
    assignment.write_text(json.dumps({"ABROB": "PICAB", "TRIAE": "PICAB"}))
    out = plant / "aligned2"
    assert main(plant_args(plant, out, "--assignment", assignment)) == 0
    rows = report(out)
    assert rows["ABROB"]["aligned_to"] == "PICAB" and rows["TRIAE"]["aligned_to"] == "PICAB"
    # the layout of the old seed_selection.json ({"groups": {...}}) is accepted too
    assignment.write_text(json.dumps({"seeds": ["ARATH"], "groups": {"ABROB": "ORYSA", "TRIAE": "ORYSA"}}))
    out = plant / "aligned3"
    assert main(plant_args(plant, out, "--assignment", assignment)) == 0
    assert report(out)["ABROB"]["aligned_to"] == "ORYSA"


def test_species_flag_restricts_stage_two_but_seeds_are_always_aligned(plant):
    out = plant / "aligned4"
    assert main(plant_args(plant, out, "--species", "TRIAE")) == 0
    assert (out / "TRIAE.h5").exists() and (out / "ARATH.h5").exists()
    assert not (out / "ABROB.h5").exists()


def test_anchor_kind_can_be_overridden(plant):
    out = plant / "aligned5"
    assert main(plant_args(plant, out, "--anchors", "centroids")) == 0
    assert report(out)["ORYSA"]["n_anchors"] == "50"  # one gene per orthogroup: same count


def test_reference_must_be_a_seed_and_seed_files_must_exist(plant):
    with pytest.raises(SystemExit) as e:
        main(plant_args(plant, plant / "x", "--reference", "TRIAE"))
    assert e.value.code not in (0, None)
    (plant / "seeds.txt").write_text("ARATH\nZEAMA\n")
    with pytest.raises(SystemExit) as e:
        main(plant_args(plant, plant / "y"))
    assert e.value.code not in (0, None)


def test_seeds_file_is_deduplicated_and_comments_are_ignored(plant):
    (plant / "seeds.txt").write_text("ARATH\nARATH\n  # a comment\nORYSA\nPICAB\n")
    out = plant / "dedup"
    assert main(plant_args(plant, out, "--species", "TRIAE", "TRIAE")) == 0
    assert list(report(out)) == ["ARATH", "ORYSA", "PICAB", "TRIAE"]


def test_species_sharing_no_orthogroup_with_any_seed_is_skipped(plant):
    rng = np.random.default_rng(5)
    ids = [f"LONER_g{i}.1" for i in range(10)]
    write_embeddings(plant / "node2vec" / "LONER.h5", rng.standard_normal((10, D)).astype(np.float32), ids)
    with open(plant / "orthogroups" / "LONER_transcripts_to_OG.tsv", "w") as f:
        f.write("Transcript_ID\tProtein_ID\tOrthogroup\n")
        for g in ids:
            f.write(f"t\t{g}\tOGONLY{g}\n")
    out = plant / "loner"
    assert main(plant_args(plant, out, "--species", "LONER", "TRIAE")) == 1
    rows = report(out)
    assert rows["LONER"] == {"species": "LONER", "role": "non-seed", "aligned_to": "", "n_anchors": "0",
                             "n_genes": "10", "status": "skipped"}
    assert rows["TRIAE"]["status"] == "ok" and not (out / "LONER.h5").exists()


def test_seeds_aligned_mode_adds_a_species_to_the_released_frame(plant):
    """Stage 2 against seeds that already share the reference frame (as the Zenodo files do)
    gives the same result as the full two-stage run, without rewriting the seeds."""
    out1 = plant / "full"
    assert main(plant_args(plant, out1)) == 0
    emb2 = plant / "released"
    emb2.mkdir()
    for s in ["ARATH", "ORYSA", "PICAB"]:
        (emb2 / f"{s}.h5").write_bytes((out1 / f"{s}.h5").read_bytes())
    (emb2 / "TRIAE.h5").write_bytes((plant / "node2vec" / "TRIAE.h5").read_bytes())
    out2 = plant / "added"
    args = plant_args(plant, out2, "--seeds-aligned", "--species", "TRIAE")
    args[args.index("--embeddings") + 1] = str(emb2)
    assert main(args) == 0
    Y1, id1 = read_embeddings(out1 / "TRIAE.h5")
    Y2, id2 = read_embeddings(out2 / "TRIAE.h5")
    assert id1 == id2 and np.allclose(Y1, Y2, atol=1e-4)
    assert not (out2 / "ARATH.h5").exists()
    rows = report(out2)
    assert rows["ORYSA"]["status"] == "unchanged" and rows["TRIAE"]["status"] == "ok"


# --- STRING-like: eggNOG members file, centroid anchors ---------------------------------

def make_string_dataset(base, *, mirror_10090=False):
    """Hub 9606, seeds 10090 and 4932, non-seeds 1450537 and 7227. Orthogroups have one to
    three members per species, so the centroid anchors matter. With ``mirror_10090`` the mouse
    embedding is an exact mirror image of the human one."""
    rng = np.random.default_rng(0)
    n2v = base / "node2vec"
    n2v.mkdir()
    (base / "eggnog").mkdir()
    taxa = ["9606", "10090", "4932", "1450537", "7227"]
    n_og = 40
    Z = rng.standard_normal((n_og, D))
    members = {t: {} for t in taxa}
    human = None
    for t in taxa:
        Q = random_rotation(rng)
        ogs = range(n_og) if t != "1450537" else range(0, 25)
        rows, ids = [], []
        for og in ogs:
            for j in range(int(rng.integers(1, 4))):
                pid = f"{t}.P{og}_{j}"
                ids.append(pid)
                rows.append(Z[og] + 0.02 * rng.standard_normal(D))
                members[t].setdefault(og, []).append(pid)
        X = np.array(rows) @ Q
        if t == "9606":
            human = (ids, X)
        if t == "10090" and mirror_10090:
            F = np.eye(D)
            F[0, 0] = -1
            ids = [i.replace("9606.", "10090.") for i in human[0]]
            X = human[1] @ F
            members[t] = {og: [p.replace("9606.", "10090.") for p in ps] for og, ps in members["9606"].items()}
        write_embeddings(n2v / f"{t}.h5", X.astype(np.float32), ids)
    with gzip.open(base / "eggnog" / "2759.tsv.gz", "wt") as f:
        for og in range(n_og):
            prots = [p for t in taxa for p in members[t].get(og, [])]
            f.write(f"2759\tOG{og}\t{len(prots)}\t{len(taxa)}\t{','.join(taxa)}\t{','.join(prots)}\n")
    (base / "seeds.txt").write_text("9606\n10090\n4932\n")
    return base


def string_args(base, out, *extra):
    return ["align", "--embeddings", str(base / "node2vec"), "--orthogroups", str(base / "eggnog" / "2759.tsv.gz"),
            "--seeds", str(base / "seeds.txt"), "--reference", "9606", "--out", str(out), *map(str, extra)]


def test_string_track_end_to_end(tmp_path):
    base = make_string_dataset(tmp_path)
    out, rot = base / "aligned", base / "rotations"
    assert main(string_args(base, out, "--allow-reflection", "--rotations", rot)) == 0
    for t in ["9606", "10090", "4932", "1450537", "7227"]:
        assert (out / f"{t}.h5").exists(), t
        R = np.load(rot / f"{t}.npy")
        assert np.allclose(R.T @ R, np.eye(D), atol=1e-5)
    Xh, idh = read_embeddings(base / "node2vec" / "9606.h5")
    Yh, idh2 = read_embeddings(out / "9606.h5")
    assert idh == idh2 and np.array_equal(Xh, Yh)
    # orthologs co-localise: the centroid of every orthogroup agrees between hub and non-seed
    Yn, idn = read_embeddings(out / "1450537.h5")

    def centroids(Y, ids):
        c = {}
        for v, i in zip(Y, ids):
            c.setdefault(i.split(".P")[1].split("_")[0], []).append(v)
        return {k: np.mean(v, axis=0) for k, v in c.items()}

    ch, cn = centroids(Yh, idh2), centroids(Yn, idn)
    assert np.mean([cosine(ch[k], cn[k]) for k in cn]) > 0.95
    rows = report(out)
    assert rows["9606"]["role"] == "reference"
    assert rows["10090"]["role"] == "seed" and rows["10090"]["aligned_to"] == "9606"
    assert rows["1450537"]["role"] == "non-seed" and rows["1450537"]["aligned_to"] in {"9606", "10090", "4932"}
    assert rows["1450537"]["n_anchors"] == "25"  # shared orthogroups, one centroid row each


def test_reflection_flag_reproduces_a_mirrored_seed(tmp_path):
    base = make_string_dataset(tmp_path, mirror_10090=True)
    out, rot = base / "refl", base / "refl_rot"
    assert main(string_args(base, out, "--allow-reflection", "--rotations", rot, "--species", "7227")) == 0
    R = np.load(rot / "10090.npy").astype(np.float64)
    assert np.isclose(np.linalg.det(R), -1.0, atol=1e-4)
    Yh, _ = read_embeddings(out / "9606.h5")
    Ym, _ = read_embeddings(out / "10090.h5")
    assert np.allclose(Yh, Ym, atol=1e-4)
    out2, rot2 = base / "proper", base / "proper_rot"
    assert main(string_args(base, out2, "--rotations", rot2, "--species", "7227")) == 0
    assert np.isclose(np.linalg.det(np.load(rot2 / "10090.npy").astype(np.float64)), 1.0, atol=1e-4)


def test_min_anchors_skips_a_species_and_sets_the_exit_code(tmp_path):
    base = make_string_dataset(tmp_path)
    out = base / "skip"
    assert main(string_args(base, out, "--allow-reflection", "--min-anchors", "30", "--species", "1450537")) == 1
    rows = report(out)
    assert rows["1450537"]["status"] == "skipped" and not (out / "1450537.h5").exists()
    assert rows["10090"]["status"] == "ok"
