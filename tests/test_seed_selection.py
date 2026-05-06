"""Tests for seed selection module using toy data."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orbit.seed_selection import (
    assign_nonseed_groups,
    build_orthogroup_matrix,
    compute_ortholog_density,
    compute_species_distances,
    evaluate_seed_set,
    select_seeds,
)


# --- Fixtures ---


@pytest.fixture
def toy_transcripts(tmp_path):
    """Create 6 toy species with known orthogroup memberships.

    Species layout (designed for predictable p-dispersion):
        A, B: share many OGs (close, same clade)
        C, D: share many OGs (close, same clade)
        E, F: share many OGs (close, same clade)
        A vs C/D: moderate overlap
        A vs E/F: low overlap (distant)
        C vs E/F: moderate overlap
    """
    species_ogs = {
        "SP_A": {"OG0001": ["g1", "g2"], "OG0002": ["g3"], "OG0003": ["g4"],
                 "OG0004": ["g5"], "OG0005": ["g6"], "OG0006": ["g7"],
                 "OG0007": ["g8"], "OG0008": ["g9"]},
        "SP_B": {"OG0001": ["g1"], "OG0002": ["g3", "g4"], "OG0003": ["g5"],
                 "OG0004": ["g6"], "OG0005": ["g7"], "OG0006": ["g8"],
                 "OG0007": ["g9"], "OG0009": ["g10"]},
        "SP_C": {"OG0001": ["g1"], "OG0003": ["g2"], "OG0004": ["g3"],
                 "OG0010": ["g4"], "OG0011": ["g5"], "OG0012": ["g6"],
                 "OG0013": ["g7"], "OG0014": ["g8"]},
        "SP_D": {"OG0001": ["g1"], "OG0003": ["g2"], "OG0010": ["g3"],
                 "OG0011": ["g4"], "OG0012": ["g5"], "OG0013": ["g6"],
                 "OG0014": ["g7"], "OG0015": ["g8"]},
        "SP_E": {"OG0001": ["g1"], "OG0016": ["g2"], "OG0017": ["g3"],
                 "OG0018": ["g4"], "OG0019": ["g5"], "OG0020": ["g6"],
                 "OG0021": ["g7"], "OG0022": ["g8"]},
        "SP_F": {"OG0001": ["g1"], "OG0016": ["g2"], "OG0017": ["g3"],
                 "OG0018": ["g4"], "OG0019": ["g5"], "OG0020": ["g6"],
                 "OG0022": ["g7"], "OG0023": ["g8"]},
    }

    for sp_code, ogs in species_ogs.items():
        rows = []
        for og, genes in ogs.items():
            for gene in genes:
                rows.append({"Transcript_ID": gene, "Protein_ID": f"{gene}.1", "Orthogroup": og})
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / f"{sp_code}_transcripts_to_OG.tsv", sep="\t", index=False)

    return tmp_path


@pytest.fixture
def toy_matrices(toy_transcripts):
    """Build all matrices from toy data."""
    og_matrix = build_orthogroup_matrix(toy_transcripts)
    # Use min_species=2 for toy data (only 6 species, most OGs in 1-2 species)
    dist_matrix = compute_species_distances(og_matrix, min_species=2)
    shared_ogs, pairs = compute_ortholog_density(toy_transcripts)
    return og_matrix, dist_matrix, shared_ogs, pairs


# --- Tests ---


class TestBuildOrthogroupMatrix:
    def test_shape(self, toy_transcripts):
        og_matrix = build_orthogroup_matrix(toy_transcripts)
        assert og_matrix.shape[0] == 6  # 6 species
        assert og_matrix.shape[1] > 0  # some orthogroups

    def test_binary(self, toy_transcripts):
        og_matrix = build_orthogroup_matrix(toy_transcripts)
        assert set(og_matrix.values.flatten()).issubset({0, 1})

    def test_species_present(self, toy_transcripts):
        og_matrix = build_orthogroup_matrix(toy_transcripts)
        assert "SP_A" in og_matrix.index
        assert "SP_F" in og_matrix.index

    def test_known_og_presence(self, toy_transcripts):
        og_matrix = build_orthogroup_matrix(toy_transcripts)
        # OG0001 is in all species
        assert og_matrix["OG0001"].sum() == 6
        # OG0009 is only in SP_B
        assert og_matrix["OG0009"].sum() == 1
        assert og_matrix.loc["SP_B", "OG0009"] == 1

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_orthogroup_matrix(tmp_path)


class TestComputeSpeciesDistances:
    def test_symmetric(self, toy_matrices):
        _, dist_matrix, _, _ = toy_matrices
        np.testing.assert_array_almost_equal(dist_matrix.values, dist_matrix.values.T)

    def test_diagonal_zero(self, toy_matrices):
        _, dist_matrix, _, _ = toy_matrices
        np.testing.assert_array_almost_equal(np.diag(dist_matrix.values), 0.0)

    def test_close_species_closer(self, toy_matrices):
        """SP_A and SP_B share many OGs, should be closer than SP_A and SP_E."""
        _, dist_matrix, _, _ = toy_matrices
        assert dist_matrix.loc["SP_A", "SP_B"] < dist_matrix.loc["SP_A", "SP_E"]

    def test_range(self, toy_matrices):
        _, dist_matrix, _, _ = toy_matrices
        vals = dist_matrix.values[np.triu_indices(6, k=1)]
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0


class TestComputeOrthologDensity:
    def test_shared_ogs_symmetric(self, toy_matrices):
        _, _, shared_ogs, _ = toy_matrices
        np.testing.assert_array_equal(shared_ogs.values, shared_ogs.values.T)

    def test_known_shared_count(self, toy_matrices):
        """SP_A and SP_B share OG0001-OG0007 = 7 orthogroups."""
        _, _, shared_ogs, _ = toy_matrices
        assert shared_ogs.loc["SP_A", "SP_B"] == 7

    def test_pairs_account_for_paralogs(self, toy_matrices):
        """SP_A has 2 genes in OG0001, SP_B has 1 → 2*1=2 pairs for that OG."""
        _, _, _, pairs = toy_matrices
        # SP_A x SP_B: OG0001(2*1) + OG0002(1*2) + OG0003(1*1) + OG0004(1*1)
        #            + OG0005(1*1) + OG0006(1*1) + OG0007(1*1) = 2+2+1+1+1+1+1 = 9
        assert pairs.loc["SP_A", "SP_B"] == 9


class TestSelectSeeds:
    def test_returns_k_seeds(self, toy_matrices):
        _, dist_matrix, shared_ogs, _ = toy_matrices
        seeds = select_seeds(dist_matrix, shared_ogs, k=3, min_shared_ogs=1)
        assert len(seeds) == 3

    def test_seeds_are_spread(self, toy_matrices):
        """With k=3 and 3 clades (AB, CD, EF), should pick one from each."""
        _, dist_matrix, shared_ogs, _ = toy_matrices
        seeds = select_seeds(dist_matrix, shared_ogs, k=3, min_shared_ogs=1)
        # Should have at most one of each pair
        clade_ab = set(seeds) & {"SP_A", "SP_B"}
        clade_cd = set(seeds) & {"SP_C", "SP_D"}
        clade_ef = set(seeds) & {"SP_E", "SP_F"}
        assert len(clade_ab) <= 1
        assert len(clade_cd) <= 1
        assert len(clade_ef) <= 1
        # And should span all three clades
        assert len(clade_ab) + len(clade_cd) + len(clade_ef) == 3

    def test_density_filter(self, toy_matrices):
        """With very high min_shared_ogs, should fail."""
        _, dist_matrix, shared_ogs, _ = toy_matrices
        with pytest.raises(ValueError, match="No species pair"):
            select_seeds(dist_matrix, shared_ogs, k=2, min_shared_ogs=99999)

    def test_k_too_large(self, toy_matrices):
        _, dist_matrix, shared_ogs, _ = toy_matrices
        with pytest.raises(ValueError, match="exceeds"):
            select_seeds(dist_matrix, shared_ogs, k=100, min_shared_ogs=1)

    def test_k_too_small(self, toy_matrices):
        _, dist_matrix, shared_ogs, _ = toy_matrices
        with pytest.raises(ValueError, match="must be >= 2"):
            select_seeds(dist_matrix, shared_ogs, k=1, min_shared_ogs=1)


class TestAssignNonseedGroups:
    def test_all_nonseeds_assigned(self, toy_matrices):
        _, dist_matrix, shared_ogs, _ = toy_matrices
        seeds = select_seeds(dist_matrix, shared_ogs, k=3, min_shared_ogs=1)
        groups = assign_nonseed_groups(seeds, dist_matrix)
        all_species = set(dist_matrix.index)
        assert set(groups.keys()) == all_species - set(seeds)

    def test_close_species_grouped(self, toy_matrices):
        """If SP_A is a seed, SP_B should be assigned to it (closest)."""
        _, dist_matrix, shared_ogs, _ = toy_matrices
        seeds = select_seeds(dist_matrix, shared_ogs, k=3, min_shared_ogs=1)
        groups = assign_nonseed_groups(seeds, dist_matrix)
        # For any non-seed, it should be closest to its assigned seed
        for sp, nearest_seed in groups.items():
            dist_to_assigned = dist_matrix.loc[sp, nearest_seed]
            for s in seeds:
                assert dist_matrix.loc[sp, s] >= dist_to_assigned - 1e-10


class TestEvaluateSeedSet:
    def test_returns_expected_keys(self, toy_matrices):
        og_matrix, dist_matrix, shared_ogs, _ = toy_matrices
        seeds = select_seeds(dist_matrix, shared_ogs, k=3, min_shared_ogs=1)
        metrics = evaluate_seed_set(seeds, dist_matrix, og_matrix, shared_ogs)

        expected_keys = {
            "seeds", "k", "seed_min_dist", "seed_mean_dist",
            "seed_min_shared_ogs", "seed_mean_shared_ogs",
            "nonseed_max_dist", "nonseed_mean_dist", "og_coverage",
            "per_nonseed", "groups",
        }
        assert set(metrics.keys()) == expected_keys

    def test_og_coverage_positive(self, toy_matrices):
        og_matrix, dist_matrix, shared_ogs, _ = toy_matrices
        seeds = select_seeds(dist_matrix, shared_ogs, k=3, min_shared_ogs=1)
        metrics = evaluate_seed_set(seeds, dist_matrix, og_matrix, shared_ogs)
        assert 0.0 < metrics["og_coverage"] <= 1.0

    def test_per_nonseed_count(self, toy_matrices):
        og_matrix, dist_matrix, shared_ogs, _ = toy_matrices
        seeds = select_seeds(dist_matrix, shared_ogs, k=3, min_shared_ogs=1)
        metrics = evaluate_seed_set(seeds, dist_matrix, og_matrix, shared_ogs)
        assert len(metrics["per_nonseed"]) == 3  # 6 species - 3 seeds
