"""Seed species selection from OrthoFinder evolutionary data.

Selects seed species that maximize phylogenetic coverage AND ortholog
connectivity for the SPACE two-stage alignment pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import pdist, squareform


def build_orthogroup_matrix(transcripts_dir: str | Path) -> pd.DataFrame:
    """Build binary species x orthogroup presence/absence matrix.

    Parameters
    ----------
    transcripts_dir : path to directory containing {CODE}_transcripts_to_OG.tsv files

    Returns
    -------
    DataFrame with species codes as rows, orthogroup IDs as columns, binary values.
    """
    transcripts_dir = Path(transcripts_dir)
    files = sorted(transcripts_dir.glob("*_transcripts_to_OG.tsv"))
    if not files:
        raise FileNotFoundError(f"No transcripts_to_OG files in {transcripts_dir}")

    species_ogs: dict[str, set[str]] = {}
    for f in files:
        code = f.name.replace("_transcripts_to_OG.tsv", "")
        df = pd.read_csv(f, sep="\t")
        species_ogs[code] = set(df["Orthogroup"].dropna().unique())
        logger.debug(f"{code}: {len(species_ogs[code])} orthogroups")

    all_ogs = sorted(set().union(*species_ogs.values()))
    logger.info(f"Built OG matrix: {len(species_ogs)} species x {len(all_ogs)} orthogroups")

    matrix = pd.DataFrame(0, index=sorted(species_ogs.keys()), columns=all_ogs, dtype=np.int8)
    for code, ogs in species_ogs.items():
        matrix.loc[code, list(ogs)] = 1

    return matrix


def compute_species_distances(
    og_matrix: pd.DataFrame, min_species: int = 5
) -> pd.DataFrame:
    """Compute pairwise Jaccard dissimilarity from orthogroup presence vectors.

    Parameters
    ----------
    og_matrix : binary species x orthogroup matrix
    min_species : only use orthogroups present in at least this many species.
        Filtering out rare/species-specific OGs removes noise from WGD and
        assembly artifacts, giving distances that better reflect phylogeny.

    Returns
    -------
    Symmetric DataFrame of Jaccard distances (0 = identical, 1 = disjoint).
    """
    if min_species > 1:
        og_species_count = og_matrix.sum(axis=0)
        keep = og_species_count >= min_species
        filtered = og_matrix.loc[:, keep]
        logger.info(
            f"Distance computation: using {keep.sum()} of {len(og_species_count)} OGs "
            f"(present in >= {min_species} species)"
        )
    else:
        filtered = og_matrix

    dists = squareform(pdist(filtered.values, metric="jaccard"))
    return pd.DataFrame(dists, index=og_matrix.index, columns=og_matrix.index)


def compute_ortholog_density(transcripts_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise ortholog density between all species pairs.

    For each species pair, computes:
    - shared_ogs: number of orthogroups both species participate in
    - ortholog_pairs: sum of fi*fj for each shared OG (total anchor pairs)

    Returns
    -------
    (shared_ogs_matrix, ortholog_pairs_matrix) — both symmetric DataFrames.
    """
    transcripts_dir = Path(transcripts_dir)
    files = sorted(transcripts_dir.glob("*_transcripts_to_OG.tsv"))

    # Build per-species: {orthogroup -> gene_count}
    species_og_counts: dict[str, dict[str, int]] = {}
    for f in files:
        code = f.name.replace("_transcripts_to_OG.tsv", "")
        df = pd.read_csv(f, sep="\t")
        counts = df.groupby("Orthogroup")["Transcript_ID"].count().to_dict()
        species_og_counts[code] = counts

    codes = sorted(species_og_counts.keys())
    n = len(codes)
    shared_ogs = np.zeros((n, n), dtype=np.int32)
    ortholog_pairs = np.zeros((n, n), dtype=np.int64)

    logger.info(f"Computing ortholog density for {n} species ({n*(n-1)//2} pairs)...")

    for i in range(n):
        ogs_i = species_og_counts[codes[i]]
        for j in range(i + 1, n):
            ogs_j = species_og_counts[codes[j]]
            common = set(ogs_i.keys()) & set(ogs_j.keys())
            shared_ogs[i, j] = shared_ogs[j, i] = len(common)
            pairs = sum(ogs_i[og] * ogs_j[og] for og in common)
            ortholog_pairs[i, j] = ortholog_pairs[j, i] = pairs

    # Fill diagonal with self-OG counts
    for i in range(n):
        shared_ogs[i, i] = len(species_og_counts[codes[i]])

    shared_df = pd.DataFrame(shared_ogs, index=codes, columns=codes)
    pairs_df = pd.DataFrame(ortholog_pairs, index=codes, columns=codes)

    logger.info(
        f"Ortholog density: shared OGs range [{shared_ogs[np.triu_indices(n, k=1)].min()}"
        f"–{shared_ogs[np.triu_indices(n, k=1)].max()}], "
        f"pairs range [{ortholog_pairs[np.triu_indices(n, k=1)].min()}"
        f"–{ortholog_pairs[np.triu_indices(n, k=1)].max()}]"
    )

    return shared_df, pairs_df


def filter_outlier_species(
    og_matrix: pd.DataFrame, max_iqr_factor: float = 1.5
) -> list[str]:
    """Identify species with extreme orthogroup counts (likely polyploid/assembly artifacts).

    Uses IQR-based outlier detection on per-species OG counts.

    Returns
    -------
    List of species codes that are NOT outliers (i.e., suitable candidates).
    """
    og_counts = og_matrix.sum(axis=1)
    q1, q3 = og_counts.quantile(0.25), og_counts.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + max_iqr_factor * iqr
    kept = og_counts[og_counts <= upper].index.tolist()
    removed = og_counts[og_counts > upper].index.tolist()
    if removed:
        logger.info(
            f"Filtered {len(removed)} outlier species (OG count > {upper:.0f}): {removed}"
        )
    return kept


def select_seeds(
    dist_matrix: pd.DataFrame,
    density_matrix: pd.DataFrame,
    k: int,
    min_shared_ogs: int = 1000,
    candidates: list[str] | None = None,
) -> list[str]:
    """Select k seed species via greedy p-dispersion with ortholog density constraint.

    Algorithm:
    1. Start with the two most distant species that share >= min_shared_ogs orthogroups
    2. Iteratively add the species maximizing min-distance to existing seeds,
       filtered to candidates sharing >= min_shared_ogs OGs with at least one seed
    3. Repeat until k seeds selected

    Parameters
    ----------
    dist_matrix : pairwise Jaccard distance matrix
    density_matrix : pairwise shared orthogroup count matrix
    k : number of seeds to select
    min_shared_ogs : minimum shared orthogroups required between a candidate and
        at least one existing seed
    candidates : restrict seed selection to these species (e.g., after outlier filtering).
        If None, all species in dist_matrix are considered.

    Returns
    -------
    List of k species codes selected as seeds.
    """
    if candidates is not None:
        species = [s for s in dist_matrix.index if s in candidates]
    else:
        species = list(dist_matrix.index)
    n = len(species)
    if k > n:
        raise ValueError(f"k={k} exceeds number of species ({n})")
    if k < 2:
        raise ValueError("k must be >= 2")

    # Work with sub-matrices restricted to candidate species
    dist = dist_matrix.loc[species, species].values.copy()
    density = density_matrix.loc[species, species].values.copy()

    # Step 1: Find the most distant valid pair
    best_dist = -1.0
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if density[i, j] >= min_shared_ogs and dist[i, j] > best_dist:
                best_dist = dist[i, j]
                best_i, best_j = i, j

    if best_dist < 0:
        raise ValueError(
            f"No species pair shares >= {min_shared_ogs} orthogroups. "
            "Try lowering min_shared_ogs."
        )

    seed_indices = [best_i, best_j]
    logger.info(
        f"Initial seeds: {species[best_i]}, {species[best_j]} "
        f"(dist={best_dist:.4f}, shared_ogs={int(density[best_i, best_j])})"
    )

    # Steps 2-3: Greedily add remaining seeds
    while len(seed_indices) < k:
        best_score = -1.0
        best_candidate = -1

        for c in range(n):
            if c in seed_indices:
                continue
            # Check density constraint: must share enough OGs with at least one seed
            if not any(density[c, s] >= min_shared_ogs for s in seed_indices):
                continue
            # p-dispersion: maximize the minimum distance to any existing seed
            min_dist = min(dist[c, s] for s in seed_indices)
            if min_dist > best_score:
                best_score = min_dist
                best_candidate = c

        if best_candidate < 0:
            logger.warning(
                f"Could only select {len(seed_indices)} seeds (no more candidates "
                f"meet min_shared_ogs={min_shared_ogs})"
            )
            break

        seed_indices.append(best_candidate)
        logger.info(
            f"  Seed {len(seed_indices)}: {species[best_candidate]} "
            f"(min_dist_to_seeds={best_score:.4f})"
        )

    return [species[i] for i in seed_indices]


def assign_nonseed_groups(
    seeds: list[str], dist_matrix: pd.DataFrame
) -> dict[str, str]:
    """Assign each non-seed species to its nearest seed (for SPACE Stage 2).

    Returns
    -------
    Dict mapping non-seed species code -> nearest seed species code.
    """
    all_species = list(dist_matrix.index)
    groups: dict[str, str] = {}
    for sp in all_species:
        if sp in seeds:
            continue
        dists_to_seeds = {s: dist_matrix.loc[sp, s] for s in seeds}
        groups[sp] = min(dists_to_seeds, key=dists_to_seeds.get)
    return groups


def evaluate_seed_set(
    seeds: list[str],
    dist_matrix: pd.DataFrame,
    og_matrix: pd.DataFrame,
    density_matrix: pd.DataFrame,
) -> dict:
    """Evaluate a seed set on phylogenetic spread, anchor quality, and coverage.

    Returns
    -------
    Dict with metrics:
    - seed_min_dist, seed_mean_dist: pairwise distance among seeds
    - seed_min_shared_ogs, seed_mean_shared_ogs: shared OGs between seed pairs
    - nonseed_max_dist, nonseed_mean_dist: max/mean distance from non-seed to nearest seed
    - og_coverage: fraction of all OGs present in >= 1 seed species
    - per_nonseed: list of dicts with nearest seed info for each non-seed
    """
    all_species = list(dist_matrix.index)
    n_seeds = len(seeds)

    # Seed pairwise distances
    seed_dists = []
    seed_shared = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            seed_dists.append(dist_matrix.loc[seeds[i], seeds[j]])
            seed_shared.append(density_matrix.loc[seeds[i], seeds[j]])

    # Non-seed to nearest seed
    groups = assign_nonseed_groups(seeds, dist_matrix)
    nonseed_dists = []
    per_nonseed = []
    for sp, nearest in sorted(groups.items()):
        d = dist_matrix.loc[sp, nearest]
        nonseed_dists.append(d)
        per_nonseed.append({
            "species": sp,
            "nearest_seed": nearest,
            "distance": float(d),
            "shared_ogs": int(density_matrix.loc[sp, nearest]),
        })

    # OG coverage: fraction of all OGs present in >= 1 seed
    seed_og_presence = og_matrix.loc[seeds].sum(axis=0)
    og_coverage = float((seed_og_presence > 0).mean())

    return {
        "seeds": seeds,
        "k": n_seeds,
        "seed_min_dist": float(min(seed_dists)) if seed_dists else 0.0,
        "seed_mean_dist": float(np.mean(seed_dists)) if seed_dists else 0.0,
        "seed_min_shared_ogs": int(min(seed_shared)) if seed_shared else 0,
        "seed_mean_shared_ogs": int(np.mean(seed_shared)) if seed_shared else 0,
        "nonseed_max_dist": float(max(nonseed_dists)) if nonseed_dists else 0.0,
        "nonseed_mean_dist": float(np.mean(nonseed_dists)) if nonseed_dists else 0.0,
        "og_coverage": og_coverage,
        "per_nonseed": per_nonseed,
        "groups": groups,
    }
