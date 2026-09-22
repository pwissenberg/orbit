"""Tests for the network-free parts of orbit.fetch."""

from __future__ import annotations

import pytest

from orbit.fetch import plan_taxa, seed_list


def test_seed_list_follows_the_space_group_order():
    groups = {"protists": [5, 6], "metazoa": [9606, 10090], "fungi": [4932], "plants": [3702]}
    assert seed_list(groups) == ["9606", "10090", "4932", "3702", "5", "6"]


def test_plan_taxa_is_the_seeds_plus_the_requested_taxa_without_duplicates():
    seeds = ["9606", "10090"]
    euks = ["9606", "10090", "7227", "4932"]
    assert plan_taxa(seeds, euks, ["7227", "9606", "7227"], all_=False) == ["9606", "10090", "7227"]
    assert plan_taxa(seeds, euks, [], all_=True) == euks


def test_plan_taxa_rejects_taxa_outside_the_benchmark():
    with pytest.raises(ValueError, match="1234"):
        plan_taxa(["9606"], ["9606"], ["1234"], all_=False)
