"""Tests for orbit.anchors: ortholog anchors for the rotation."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from orbit.anchors import (
    load_orthogroups,
    nearest_seed,
    orthogroup_centroid_anchors,
    orthogroup_pair_anchors,
    read_eggnog_orthogroups,
    read_orthofinder_orthogroups,
)


def _write_orthofinder(path, rows):
    with open(path, "w") as f:
        f.write("Transcript_ID\tProtein_ID\tOrthogroup\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def _write_eggnog(path, lines):
    with gzip.open(path, "wt") as f:
        for og, taxa, members in lines:
            f.write(f"2759\t{og}\t{len(members)}\t{len(taxa)}\t{','.join(taxa)}\t{','.join(members)}\n")


# --- OrthoFinder (plants) -------------------------------------------------------------

def test_orthofinder_exact_protein_id_match(tmp_path):
    p = tmp_path / "ARATH_transcripts_to_OG.tsv"
    _write_orthofinder(p, [("t1", "AT1G01010.1", "OG1"), ("t2", "AT1G01020.1", "OG2"),
                           ("t3", "AT1G09999.1", "OG3")])
    m = read_orthofinder_orthogroups(p, ["AT1G01010.1", "AT1G01020.1", "AT5G00001.1"])
    assert m == {"AT1G01010.1": "OG1", "AT1G01020.1": "OG2"}


def test_orthofinder_falls_back_to_stripped_ids_when_coverage_is_low(tmp_path):
    p = tmp_path / "X_transcripts_to_OG.tsv"
    _write_orthofinder(p, [("t1", "g1.2", "OG1"), ("t2", "g2.1", "OG2"), ("t3", "g3.1", "OG3")])
    m = read_orthofinder_orthogroups(p, ["g1", "g2", "g4"])
    assert m == {"g1": "OG1", "g2": "OG2"}


def test_orthofinder_skips_rows_without_orthogroup(tmp_path):
    p = tmp_path / "X_transcripts_to_OG.tsv"
    with open(p, "w") as f:
        f.write("Transcript_ID\tProtein_ID\tOrthogroup\nt1\tg1\t\nt2\tg2\tOG2\n")
    assert read_orthofinder_orthogroups(p, ["g1", "g2"]) == {"g2": "OG2"}


# --- eggNOG (STRING) ------------------------------------------------------------------

def test_eggnog_members_are_split_by_taxon_and_first_orthogroup_wins(tmp_path):
    p = tmp_path / "2759.tsv.gz"
    _write_eggnog(p, [("OGA", ["9606", "10090"], ["9606.P1", "10090.Q1", "9606.P2"]),
                      ("OGB", ["9606", "4932"], ["9606.P1", "4932.Y1"])])
    m = read_eggnog_orthogroups(p, ["9606", "10090"])
    assert m == {"9606": {"9606.P1": "OGA", "9606.P2": "OGA"}, "10090": {"10090.Q1": "OGA"}}


def test_eggnog_members_table_may_be_plain_text_and_restricted_to_wanted_proteins(tmp_path):
    p = tmp_path / "2759.tsv"
    p.write_text("2759\tOGA\t3\t2\t9606,10090\t9606.P1,10090.Q1,9606.P2\n")
    assert read_eggnog_orthogroups(p, ["9606"]) == {"9606": {"9606.P1": "OGA", "9606.P2": "OGA"}}
    assert read_eggnog_orthogroups(p, ["9606"], only={"9606": {"9606.P1"}}) == {"9606": {"9606.P1": "OGA"}}


# --- the orthogroup source of `orbit align` --------------------------------------------

def test_load_orthogroups_from_an_orthofinder_directory(tmp_path):
    d = tmp_path / "orthogroups"
    d.mkdir()
    _write_orthofinder(d / "ARATH_transcripts_to_OG.tsv", [("t", "a1", "OG1"), ("t", "a2", "OG2")])
    _write_orthofinder(d / "ORYSA_transcripts_to_OG.tsv", [("t", "o1", "OG1")])
    og, kind = load_orthogroups(d, {"ARATH": ["a1", "a2", "a9"], "ORYSA": ["o1"]})
    assert kind == "orthofinder"
    assert og == {"ARATH": {"a1": "OG1", "a2": "OG2"}, "ORYSA": {"o1": "OG1"}}


def test_load_orthogroups_from_an_eggnog_members_file(tmp_path):
    p = tmp_path / "2759.tsv.gz"
    _write_eggnog(p, [("OGA", ["9606", "10090"], ["9606.P1", "10090.Q1", "10090.Q2"])])
    og, kind = load_orthogroups(p, {"9606": ["9606.P1"], "10090": ["10090.Q1"]})
    assert kind == "eggnog"
    # restricted to the identifiers of the embeddings: 10090.Q2 has no vector
    assert og == {"9606": {"9606.P1": "OGA"}, "10090": {"10090.Q1": "OGA"}}


def test_load_orthogroups_names_the_species_whose_table_is_missing(tmp_path):
    d = tmp_path / "orthogroups"
    d.mkdir()
    _write_orthofinder(d / "ARATH_transcripts_to_OG.tsv", [("t", "a1", "OG1")])
    with pytest.raises(FileNotFoundError, match="ORYSA"):
        load_orthogroups(d, {"ARATH": ["a1"], "ORYSA": ["o1"]})


# --- anchor matrices -------------------------------------------------------------------

def test_pair_anchors_enumerate_every_cross_species_pair_per_shared_orthogroup():
    Xa = np.arange(8, dtype=float).reshape(4, 2)
    Xb = -np.arange(6, dtype=float).reshape(3, 2)
    ids_a, ids_b = ["a0", "a1", "a2", "a3"], ["b0", "b1", "b2"]
    og_a = {"a0": "OG1", "a1": "OG1", "a2": "OG2", "a3": "OG9"}
    og_b = {"b0": "OG1", "b1": "OG2", "b2": "OG7"}
    A, B = orthogroup_pair_anchors(Xa, ids_a, og_a, Xb, ids_b, og_b)
    # OG1: (a0,b0), (a1,b0); OG2: (a2,b1); OG9 and OG7 are not shared
    assert A.shape == B.shape == (3, 2)
    assert np.array_equal(A, Xa[[0, 1, 2]])
    assert np.array_equal(B, Xb[[0, 0, 1]])


def test_pair_anchors_ignore_orthogroup_members_absent_from_the_embedding():
    Xa = np.ones((1, 2))
    Xb = np.ones((1, 2))
    A, B = orthogroup_pair_anchors(Xa, ["a0"], {"a0": "OG1", "ghost": "OG1"},
                                   Xb, ["b0"], {"b0": "OG1"})
    assert A.shape == (1, 2)


def test_centroid_anchors_use_one_row_per_shared_orthogroup():
    Xa = np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]])
    Xb = np.array([[1.0, 0.0], [3.0, 0.0]])
    og_a = {"a0": "OG1", "a1": "OG1", "a2": "OG2"}
    og_b = {"b0": "OG1", "b1": "OG3"}
    A, B = orthogroup_centroid_anchors(Xa, ["a0", "a1", "a2"], og_a, Xb, ["b0", "b1"], og_b)
    assert A.shape == B.shape == (1, 2)
    assert np.allclose(A[0], [1.0, 1.0])
    assert np.allclose(B[0], [1.0, 0.0])


def test_anchor_builders_return_empty_matrices_when_nothing_is_shared():
    A, B = orthogroup_pair_anchors(np.ones((1, 3)), ["a"], {"a": "OG1"},
                                   np.ones((1, 3)), ["b"], {"b": "OG2"})
    assert A.shape == (0, 3) and B.shape == (0, 3)


# --- seed assignment -------------------------------------------------------------------

def test_nearest_seed_is_the_seed_with_most_shared_orthogroups():
    seeds = {"ARATH": {"OG1", "OG2", "OG3"}, "ORYSA": {"OG1", "OG5"}, "MARPO": {"OG7"}}
    assert nearest_seed({"OG1", "OG5", "OG9"}, seeds) == "ORYSA"
    assert nearest_seed({"OG1", "OG2"}, seeds) == "ARATH"


def test_nearest_seed_breaks_ties_by_seed_order():
    seeds = {"ARATH": {"OG1"}, "ORYSA": {"OG1"}}
    assert nearest_seed({"OG1"}, seeds) == "ARATH"


def test_nearest_seed_with_no_overlap_raises():
    with pytest.raises(ValueError):
        nearest_seed({"OG9"}, {"ARATH": {"OG1"}})
