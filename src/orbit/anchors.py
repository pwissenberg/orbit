"""Ortholog anchors for the ORBIT rotation (paper Sections 2.3 and 2.5).

Two anchor constructions are used in the paper:

* **Pair anchors** (plant coexpression networks): every cross-species gene pair that shares an
  OrthoFinder orthogroup contributes one row to the anchor matrices, with uniform weight.
* **Centroid anchors** (STRING): eggNOG orthogroups at the eukaryotic root are large, so every
  shared orthogroup contributes one row per species, the centroid of its member proteins.

Both return raw-frame anchor matrices ``(A, B)`` with paired rows, ready for
:func:`orbit.align.procrustes_rotation`.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np


# --- orthogroup tables ---------------------------------------------------------------------

def read_orthofinder_orthogroups(path: str | Path, ids: Iterable[str]) -> dict[str, str]:
    """``{gene_id: orthogroup}`` for the genes in ``ids`` from an OrthoFinder
    ``*_transcripts_to_OG.tsv`` table (columns ``Transcript_ID``, ``Protein_ID``,
    ``Orthogroup``).

    Protein identifiers are matched exactly. If fewer than half of ``ids`` match, both sides
    are stripped of their version suffix (everything after the first ``.``) and matched again,
    which covers the species whose embedding identifiers lack the transcript version.
    """
    wanted = list(dict.fromkeys(ids))
    table: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            og = (row.get("Orthogroup") or "").strip()
            if og:
                table.setdefault(row["Protein_ID"], og)
    exact = {g: table[g] for g in wanted if g in table}
    if wanted and len(exact) / len(wanted) >= 0.5:
        return exact
    stripped = {}
    for pid, og in table.items():
        stripped.setdefault(pid.split(".", 1)[0], og)
    return {g: stripped[g.split(".", 1)[0]] for g in wanted if g.split(".", 1)[0] in stripped}


def _open_text(path: str | Path):
    """Open a gzipped or plain text file for reading, decided by the gzip magic bytes."""
    with open(path, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gzipped else open(path, "rt")


def read_eggnog_orthogroups(path: str | Path, taxa: Iterable[str],
                            only: Mapping[str, Collection[str]] | None = None,
                            ) -> dict[str, dict[str, str]]:
    """``{taxid: {protein_id: orthogroup}}`` for ``taxa`` from an eggNOG members table
    (``<level>.tsv.gz`` or plain; the orthogroup is the second column, the member list the
    last, with members written ``<taxid>.<protein>``). The first orthogroup listed for a
    protein wins. With ``only`` (``{taxid: proteins}``) only those proteins are kept, which
    bounds the memory to the proteins that have an embedding.
    """
    result: dict[str, dict[str, str]] = {str(t): {} for t in taxa}
    with _open_text(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            og = parts[1]
            for prot in parts[-1].split(","):
                taxid = prot.split(".", 1)[0]
                d = result.get(taxid)
                if d is not None and (only is None or prot in only[taxid]):
                    d.setdefault(prot, og)
    return result


# --- anchor matrices -----------------------------------------------------------------------

def _members_by_orthogroup(ids: list[str], gene2og: Mapping[str, str]) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(ids):
        og = gene2og.get(g)
        if og is not None:
            rows[og].append(i)
    return rows


def orthogroup_pair_anchors(X_a: np.ndarray, ids_a: list[str], og_a: Mapping[str, str],
                            X_b: np.ndarray, ids_b: list[str], og_b: Mapping[str, str],
                            ) -> tuple[np.ndarray, np.ndarray]:
    """One anchor row per cross-species gene pair sharing an orthogroup (uniform weights).

    Orthogroup members missing from an embedding are ignored. Rows are ordered by orthogroup
    name, then by the genes' order in ``ids_a`` and ``ids_b``.
    """
    X_a, X_b = np.asarray(X_a), np.asarray(X_b)
    ma, mb = _members_by_orthogroup(ids_a, og_a), _members_by_orthogroup(ids_b, og_b)
    rows_a, rows_b = [], []
    for og in sorted(set(ma) & set(mb)):
        for i in ma[og]:
            for j in mb[og]:
                rows_a.append(i)
                rows_b.append(j)
    return X_a[rows_a].reshape(len(rows_a), X_a.shape[1]), X_b[rows_b].reshape(len(rows_b), X_b.shape[1])


def orthogroup_centroids(X: np.ndarray, ids: list[str], gene2og: Mapping[str, str],
                         ) -> dict[str, np.ndarray]:
    """``{orthogroup: mean embedding of its members present in X}``."""
    X = np.asarray(X, dtype=np.float64)
    return {og: X[rows].mean(axis=0) for og, rows in _members_by_orthogroup(ids, gene2og).items()}


def orthogroup_centroid_anchors(X_a: np.ndarray, ids_a: list[str], og_a: Mapping[str, str],
                                X_b: np.ndarray, ids_b: list[str], og_b: Mapping[str, str],
                                ) -> tuple[np.ndarray, np.ndarray]:
    """One anchor row per shared orthogroup: the centroid of its members in each species."""
    ca, cb = orthogroup_centroids(X_a, ids_a, og_a), orthogroup_centroids(X_b, ids_b, og_b)
    return centroid_anchors(ca, cb, np.asarray(X_a).shape[1])


def centroid_anchors(centroids_a: Mapping[str, np.ndarray], centroids_b: Mapping[str, np.ndarray],
                     dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair two centroid tables on their shared orthogroups (sorted by name)."""
    shared = sorted(set(centroids_a) & set(centroids_b))
    if not shared:
        return np.empty((0, dim)), np.empty((0, dim))
    return (np.array([centroids_a[og] for og in shared]),
            np.array([centroids_b[og] for og in shared]))


# --- seed assignment -----------------------------------------------------------------------

def nearest_seed(orthogroups: Iterable[str], seed_orthogroups: Mapping[str, Iterable[str]]) -> str:
    """The seed sharing the most orthogroups with ``orthogroups``; ties go to the earlier seed
    in ``seed_orthogroups``. Raises ``ValueError`` if no seed shares any orthogroup."""
    ogs = set(orthogroups)
    best, best_n = None, 0
    for seed, seed_ogs in seed_orthogroups.items():
        n = len(ogs.intersection(seed_ogs))
        if n > best_n:
            best, best_n = seed, n
    if best is None:
        raise ValueError("no seed shares an orthogroup with this species")
    return best


# --- the orthogroup source of `orbit align` -------------------------------------------------

def load_orthogroups(source: str | Path, ids_by_species: Mapping[str, Sequence[str]],
                     ) -> tuple[dict[str, dict[str, str]], str]:
    """Read the orthogroup of every embedded gene of every species from ``source``.

    ``source`` is either a directory of OrthoFinder tables, one per species named
    ``<species>_transcripts_to_OG.tsv`` (plant track; returns kind ``"orthofinder"``), or a
    single eggNOG members table ``<level>.tsv.gz`` covering all species (STRING track; kind
    ``"eggnog"``). Returns ``({species: {gene: orthogroup}}, kind)`` restricted to the genes
    in ``ids_by_species``, i.e. those that have an embedding.
    """
    source = Path(source)
    if source.is_dir():
        table_of: dict[str, dict[str, str]] = {}
        for species, ids in ids_by_species.items():
            table = source / f"{species}_transcripts_to_OG.tsv"
            if not table.exists():
                raise FileNotFoundError(f"no OrthoFinder table for {species}: {table}")
            table_of[species] = read_orthofinder_orthogroups(table, ids)
        return table_of, "orthofinder"
    if source.is_file():
        wanted = {s: set(ids) for s, ids in ids_by_species.items()}
        return read_eggnog_orthogroups(source, wanted, only=wanted), "eggnog"
    raise FileNotFoundError(f"orthogroup source not found: {source}")
