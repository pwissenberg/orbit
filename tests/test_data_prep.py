"""Tests for data preparation module."""

import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from orbit.data_prep import (
    build_ortholog_pairs,
    convert_to_space_format,
    list_species_with_networks,
    list_species_with_orthogroups,
    preprocess_network,
    write_space_config,
)


# --- Fixtures ---


@pytest.fixture
def raw_network(tmp_path):
    """Create a small raw TEA-GCN network file."""
    content = (
        "Source_Protein_ID\tTarget_Protein_ID\tzScore(Co-exp_Str_MR)\n"
        "gene_A.p1\tgene_B.p1\t2.5\n"
        "gene_B.p1\tgene_A.p1\t2.3\n"  # reverse of first edge (lower weight)
        "gene_A.p1\tgene_C.p1\t1.8\n"
        "gene_C.p1\tgene_D.p1\t0.5\n"
        "gene_D.p1\tgene_C.p1\t0.7\n"  # reverse, higher weight
    )
    raw_path = tmp_path / "SP_A_Species_one_Top50EdgesPerGene_ProteinID.tsv"
    raw_path.write_text(content)
    return raw_path


@pytest.fixture
def clean_network(tmp_path, raw_network):
    """Preprocess raw network and return the clean file path."""
    out_path = tmp_path / "networks" / "SP_A.tsv"
    preprocess_network(raw_network, out_path)
    return out_path


@pytest.fixture
def toy_h5_pair(tmp_path):
    """Create two small H5 embedding files and OrthoFinder data for testing ortholog pairs."""
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()

    # Species X: 4 proteins
    proteins_x = ["Xg1.p1", "Xg2.p1", "Xg3.p1", "Xg4.p1"]
    emb_x = np.random.randn(4, 8).astype(np.float32)
    _write_h5(h5_dir / "SP_X.h5", proteins_x, emb_x)

    # Species Y: 3 proteins
    proteins_y = ["Yg1.p1", "Yg2.p1", "Yg3.p1"]
    emb_y = np.random.randn(3, 8).astype(np.float32)
    _write_h5(h5_dir / "SP_Y.h5", proteins_y, emb_y)

    # OrthoFinder transcripts
    # SP_X: Xg1→OG0001, Xg2→OG0001, Xg3→OG0002, Xg4→OG0003
    df_x = pd.DataFrame({
        "Transcript_ID": ["Xg1", "Xg2", "Xg3", "Xg4"],
        "Protein_ID": ["Xg1.p1", "Xg2.p1", "Xg3.p1", "Xg4.p1"],
        "Orthogroup": ["OG0001", "OG0001", "OG0002", "OG0003"],
    })
    df_x.to_csv(transcripts_dir / "SP_X_transcripts_to_OG.tsv", sep="\t", index=False)

    # SP_Y: Yg1→OG0001, Yg2→OG0002, Yg3→OG0004
    df_y = pd.DataFrame({
        "Transcript_ID": ["Yg1", "Yg2", "Yg3"],
        "Protein_ID": ["Yg1.p1", "Yg2.p1", "Yg3.p1"],
        "Orthogroup": ["OG0001", "OG0002", "OG0004"],
    })
    df_y.to_csv(transcripts_dir / "SP_Y_transcripts_to_OG.tsv", sep="\t", index=False)

    return h5_dir, transcripts_dir


def _write_h5(path, proteins, embeddings):
    """Write a minimal SPACE-format H5 file."""
    with h5py.File(path, "w") as f:
        f.create_group("metadata")
        f["metadata"].attrs["n_proteins"] = len(proteins)
        f["metadata"].attrs["embedding_dim"] = embeddings.shape[1]
        f["metadata"].attrs["precision"] = 32
        f.create_dataset("proteins", data=np.array(proteins, dtype="S"))
        f.create_dataset("embeddings", data=embeddings)


# --- Tests ---


class TestPreprocessNetwork:
    def test_creates_output(self, raw_network, tmp_path):
        out_path = tmp_path / "out" / "net.tsv"
        preprocess_network(raw_network, out_path)
        assert out_path.exists()

    def test_deduplicates_edges(self, raw_network, tmp_path):
        out_path = tmp_path / "out" / "net.tsv"
        n = preprocess_network(raw_network, out_path)
        # Original: 5 rows, but 2 duplicate pairs → 3 unique edges
        assert n == 3

    def test_keeps_max_weight(self, raw_network, tmp_path):
        out_path = tmp_path / "out" / "net.tsv"
        preprocess_network(raw_network, out_path)
        lines = out_path.read_text().strip().split("\n")
        edge_weights = {}
        for line in lines:
            a, b, w = line.split("\t")
            edge_weights[(a, b)] = float(w)
        # gene_A, gene_B: max(2.5, 2.3) = 2.5
        assert edge_weights[("gene_A.p1", "gene_B.p1")] == 2.5
        # gene_C, gene_D: max(0.5, 0.7) = 0.7
        assert edge_weights[("gene_C.p1", "gene_D.p1")] == 0.7

    def test_no_header(self, raw_network, tmp_path):
        out_path = tmp_path / "out" / "net.tsv"
        preprocess_network(raw_network, out_path)
        first_line = out_path.read_text().split("\n")[0]
        assert "Source" not in first_line

    def test_sorted_edge_keys(self, raw_network, tmp_path):
        out_path = tmp_path / "out" / "net.tsv"
        preprocess_network(raw_network, out_path)
        for line in out_path.read_text().strip().split("\n"):
            a, b, _ = line.split("\t")
            assert a <= b


class TestConvertToSpaceFormat:
    def test_creates_gzipped(self, clean_network, tmp_path):
        out_path = tmp_path / "space" / "SP_A.txt.gz"
        convert_to_space_format(clean_network, out_path)
        assert out_path.exists()

    def test_has_header(self, clean_network, tmp_path):
        out_path = tmp_path / "space" / "SP_A.txt.gz"
        convert_to_space_format(clean_network, out_path)
        with gzip.open(out_path, "rt") as f:
            header = f.readline().strip()
        assert header == "protein1 protein2 score"

    def test_space_separated(self, clean_network, tmp_path):
        out_path = tmp_path / "space" / "SP_A.txt.gz"
        convert_to_space_format(clean_network, out_path)
        with gzip.open(out_path, "rt") as f:
            next(f)  # skip header
            line = f.readline().strip()
        parts = line.split(" ")
        assert len(parts) == 3

    def test_scores_are_integers(self, clean_network, tmp_path):
        out_path = tmp_path / "space" / "SP_A.txt.gz"
        convert_to_space_format(clean_network, out_path)
        with gzip.open(out_path, "rt") as f:
            next(f)  # skip header
            for line in f:
                _, _, score = line.strip().split(" ")
                assert score == str(int(score))

    def test_max_score_is_1000(self, clean_network, tmp_path):
        out_path = tmp_path / "space" / "SP_A.txt.gz"
        convert_to_space_format(clean_network, out_path)
        scores = []
        with gzip.open(out_path, "rt") as f:
            next(f)
            for line in f:
                _, _, score = line.strip().split(" ")
                scores.append(int(score))
        assert max(scores) == 1000


class TestBuildOrthologPairs:
    def test_pair_count(self, toy_h5_pair, tmp_path):
        h5_dir, transcripts_dir = toy_h5_pair
        out_path = tmp_path / "pairs" / "SP_X_SP_Y.tsv"
        n = build_ortholog_pairs("SP_X", "SP_Y", h5_dir, transcripts_dir, out_path)
        # Shared OGs: OG0001 (X:2 genes, Y:1 gene → 2 pairs), OG0002 (X:1, Y:1 → 1 pair)
        # OG0003 and OG0004 are not shared
        assert n == 3

    def test_weight_calculation(self, toy_h5_pair, tmp_path):
        h5_dir, transcripts_dir = toy_h5_pair
        out_path = tmp_path / "pairs" / "SP_X_SP_Y.tsv"
        build_ortholog_pairs("SP_X", "SP_Y", h5_dir, transcripts_dir, out_path)
        lines = out_path.read_text().strip().split("\n")
        weights = [float(line.split("\t")[2]) for line in lines]
        # OG0001: w = 1/(2*1) = 0.5, OG0002: w = 1/(1*1) = 1.0
        assert 0.5 in weights
        assert 1.0 in weights

    def test_output_format(self, toy_h5_pair, tmp_path):
        h5_dir, transcripts_dir = toy_h5_pair
        out_path = tmp_path / "pairs" / "SP_X_SP_Y.tsv"
        build_ortholog_pairs("SP_X", "SP_Y", h5_dir, transcripts_dir, out_path)
        for line in out_path.read_text().strip().split("\n"):
            parts = line.split("\t")
            assert len(parts) == 3
            int(parts[0])  # index A
            int(parts[1])  # index B
            float(parts[2])  # weight


class TestWriteSpaceConfig:
    def test_seeds_txt(self, tmp_path):
        seeds = ["ALSLA", "ARAAL", "CYCBI"]
        groups = {"SP_A": "ALSLA", "SP_B": "ARAAL"}
        write_space_config(seeds, groups, tmp_path)
        content = (tmp_path / "seeds.txt").read_text().strip().split("\n")
        assert content == seeds

    def test_seed_groups_json(self, tmp_path):
        seeds = ["ALSLA", "ARAAL", "CYCBI"]
        groups = {"SP_A": "ALSLA", "SP_B": "ARAAL"}
        write_space_config(seeds, groups, tmp_path)
        with open(tmp_path / "seed_groups.json") as f:
            data = json.load(f)
        assert data == {"plants": seeds}

    def test_tax_group_tsv(self, tmp_path):
        seeds = ["ALSLA", "ARAAL"]
        groups = {"SP_A": "ALSLA", "SP_B": "ARAAL"}
        write_space_config(seeds, groups, tmp_path)
        df = pd.read_csv(tmp_path / "tax_group.tsv", sep="\t")
        assert len(df) == 4  # 2 seeds + 2 nonseeds
        assert set(df["group"]) == {"plants"}
        assert set(df["taxid"]) == {"ALSLA", "ARAAL", "SP_A", "SP_B"}


class TestListHelpers:
    def test_list_species_with_networks(self, tmp_path):
        (tmp_path / "ARATH_Arabidopsis_thaliana_Top50EdgesPerGene_ProteinID.tsv").touch()
        (tmp_path / "BRADI_Brachypodium_distachyon_Top50EdgesPerGene_ProteinID.tsv").touch()
        codes = list_species_with_networks(tmp_path)
        assert codes == ["ARATH", "BRADI"]

    def test_list_species_with_orthogroups(self, tmp_path):
        (tmp_path / "SP_X_transcripts_to_OG.tsv").touch()
        (tmp_path / "SP_Y_transcripts_to_OG.tsv").touch()
        codes = list_species_with_orthogroups(tmp_path)
        assert codes == ["SP_X", "SP_Y"]
