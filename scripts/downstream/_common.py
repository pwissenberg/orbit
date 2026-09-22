"""Shared helpers of the downstream benchmark scripts: embedding arms and id mappings.

An *arm* is one ``{protein_id: vector}`` mapping that a benchmark trains classifiers on.
The arms of the paper are the aligned embeddings written by ``orbit align`` (plant:
``results/plant/<SPECIES>.h5``; STRING: ``results/string/<taxid>.h5``, identical to the Zenodo
release), the ProtT5 sequence embeddings, the FedCoder/SPACE baselines, and their
concatenations and PCA controls. Every script declares them on the command line:

    --arm NAME=PATH        a directory of per-species ``<species>.h5`` files, one ``.h5`` in
                           that layout, or an HDF5 with one dataset per protein (UniProt's
                           ``per-protein.h5``, SPACE's ``netgo_t5.h5``)
    --pca NAME=ARM:K       ARM projected onto its first K principal components
    --concat NAME=A+B      A and B side by side over the proteins they share
    --normalize-concat     L2-normalise each block before concatenating (the plant protocol,
                           Section 2.7); without it the raw vectors are joined (STRING, 2.8)
    --rekey NAME           the arm is keyed by the label identifiers (UniProt accessions, as
                           UniProt's ProtT5 file is); map its keys onto the embedding ids
                           through the script's id mapping

Every arm is scored on the proteins present in all arms, so the comparison is like for like.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

import h5py
import numpy as np

from orbit.io import read_embeddings

Arm = dict[str, np.ndarray]


# --- loading ------------------------------------------------------------------------------

def load_embedding_dir(directory: str | Path, species: Collection[str] | None = None) -> Arm:
    """Every vector of every ``<species>.h5`` under ``directory``, keyed by protein
    identifier, float32. ``species`` restricts the load to the files named after them
    (STRING taxids); if none of them names a file (plant gene ids carry no species prefix)
    everything is loaded."""
    directory = Path(directory)
    stems = sorted(p.stem for p in directory.glob("*.h5"))
    if species is not None:
        wanted = {str(s) for s in species}
        matching = [s for s in stems if s in wanted]
        if matching:
            stems = matching
    out: Arm = {}
    for stem in stems:
        X, ids = read_embeddings(directory / f"{stem}.h5")
        out.update(zip(ids, np.asarray(X, dtype=np.float32)))
    return out


def load_flat_h5(path: str | Path) -> Arm:
    """An HDF5 file with one dataset per protein (the dataset name is the identifier)."""
    out: Arm = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            out[key] = np.asarray(f[key][:], dtype=np.float32)
    return out


def load_arm(path: str | Path, species: Collection[str] | None = None) -> Arm:
    """Load an arm from a directory of per-species files, one such file, or a flat HDF5."""
    path = Path(path)
    if path.is_dir():
        return load_embedding_dir(path, species)
    with h5py.File(path, "r") as f:
        orbit_layout = "proteins" in f and "embeddings" in f
    if orbit_layout:
        X, ids = read_embeddings(path)
        return dict(zip(ids, np.asarray(X, dtype=np.float32)))
    return load_flat_h5(path)


# --- combining ----------------------------------------------------------------------------

def l2_normalize(embs: Mapping[str, np.ndarray], eps: float = 1e-12) -> Arm:
    """Scale every vector to unit length; zero vectors stay zero."""
    return {k: (np.asarray(v, np.float32) / (np.linalg.norm(v) + eps)).astype(np.float32)
            for k, v in embs.items()}


def concat(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], *, normalize: bool = False) -> Arm:
    """Two arms side by side over the proteins they share; ``normalize`` L2-scales each
    block first."""
    if normalize:
        a, b = l2_normalize(a), l2_normalize(b)
    return {k: np.concatenate([np.asarray(a[k], np.float32), np.asarray(b[k], np.float32)])
            for k in sorted(set(a) & set(b))}


def pca_reduce(embs: Mapping[str, np.ndarray], n_components: int, seed: int = 42) -> Arm:
    """Project an arm onto its first ``n_components`` principal components (the
    dimensionality-matching control of Section 2.8)."""
    from sklearn.decomposition import PCA

    keys = list(embs)
    X = np.vstack([embs[k] for k in keys])
    Z = PCA(n_components=n_components, random_state=seed).fit_transform(X).astype(np.float32)
    return dict(zip(keys, Z))


def feature_matrix(embs: Mapping[str, np.ndarray], ids: Iterable[str]) -> np.ndarray:
    """Rows of ``embs`` in the order of ``ids`` (``KeyError`` for a missing id)."""
    ids = list(ids)
    if not ids:
        raise ValueError("no proteins to build a feature matrix from")
    return np.vstack([embs[i] for i in ids]).astype(np.float32)


def shared_proteins(arms: Mapping[str, Mapping[str, np.ndarray]],
                    candidates: Iterable[str] | None = None) -> list[str]:
    """Sorted proteins present in every arm (and in ``candidates`` if given)."""
    keep = None
    for arm in arms.values():
        keep = set(arm) if keep is None else keep & set(arm)
    keep = keep or set()
    if candidates is not None:
        keep &= set(candidates)
    return sorted(keep)


# --- command line -------------------------------------------------------------------------

def add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("embedding arms")
    g.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                   help="aligned embeddings from `orbit align` (a directory of <species>.h5), "
                        "one such file, or a flat HDF5 with one dataset per protein (ProtT5)")
    g.add_argument("--pca", action="append", default=[], metavar="NAME=ARM:K",
                   help="add ARM reduced to K principal components")
    g.add_argument("--concat", action="append", default=[], metavar="NAME=A+B",
                   help="add the concatenation of arms A and B")
    g.add_argument("--normalize-concat", action="store_true",
                   help="L2-normalise each block before concatenating (plant protocol)")
    g.add_argument("--rekey", action="append", default=[], metavar="NAME",
                   help="arm keyed by the label ids (e.g. UniProt accessions): map its keys "
                        "onto the embedding ids through the script's id mapping")


def _split(spec: str, sep: str, what: str) -> tuple[str, str]:
    if spec.count(sep) != 1:
        raise SystemExit(f"error: {what} must be written NAME{sep}..., got {spec!r}")
    name, rest = spec.split(sep)
    if not name or not rest:
        raise SystemExit(f"error: {what} must be written NAME{sep}..., got {spec!r}")
    return name, rest


def rekey(embs: Mapping[str, np.ndarray], mapping: Mapping[str, str]) -> Arm:
    """Rename the keys of an arm through ``mapping``; keys without a mapping are dropped and
    the first key mapped onto a target wins."""
    out: Arm = {}
    for k, v in embs.items():
        t = mapping.get(k)
        if t is not None:
            out.setdefault(t, v)
    return out


def load_arms(args: argparse.Namespace, species: Collection[str] | None = None,
              rekey_map: Mapping[str, str] | None = None) -> dict[str, Arm]:
    """Build every declared arm, in declaration order: --arm (rekeyed through ``rekey_map``
    when listed in --rekey), then --pca, then --concat."""
    arms: dict[str, Arm] = {}
    declared = {_split(spec, "=", "--arm")[0] for spec in args.arm}
    for name in args.rekey:
        if name not in declared:
            raise SystemExit(f"error: --rekey {name}: unknown arm")
    if args.rekey and rekey_map is None:
        raise SystemExit("error: --rekey needs an id mapping from the label ids to the embedding ids; "
                         "this script was given none")
    for spec in args.arm:
        name, path = _split(spec, "=", "--arm")
        if not Path(path).exists():
            raise SystemExit(f"error: --arm {name}: {path} does not exist")
        arms[name] = load_arm(path, None if name in args.rekey else species)
        if name in args.rekey:
            arms[name] = rekey(arms[name], rekey_map)
    for spec in args.pca:
        name, rest = _split(spec, "=", "--pca")
        src, k = _split(rest, ":", "--pca")
        if src not in arms:
            raise SystemExit(f"error: --pca {name}: unknown arm {src!r}")
        arms[name] = pca_reduce(arms[src], int(k))
    for spec in args.concat:
        name, rest = _split(spec, "=", "--concat")
        a, b = _split(rest, "+", "--concat")
        for x in (a, b):
            if x not in arms:
                raise SystemExit(f"error: --concat {name}: unknown arm {x!r}")
        arms[name] = concat(arms[a], arms[b], normalize=args.normalize_concat)
    if not arms:
        raise SystemExit("error: declare at least one --arm NAME=PATH")
    return arms


def read_idmap(path: str | Path, columns: tuple[str, str] | None = None,
               keep: Collection[str] | None = None) -> dict[str, str]:
    """``{source: target}`` from a TSV with a header row. Default columns are ``From`` and
    ``To`` (SPACE's id mapping tables); pass ``columns`` for other layouts. With ``keep``
    only targets in it count (the genes that have an embedding), so a source id listed
    first with an absent isoform still maps onto the present one. The first mapping of a
    source id wins."""
    out: dict[str, str] = {}
    keep = None if keep is None else set(keep)
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        src, dst = columns or ("From", "To")
        if src not in (reader.fieldnames or []) or dst not in (reader.fieldnames or []):
            raise SystemExit(f"error: {path} has no columns {src!r} and {dst!r}; "
                             f"found {reader.fieldnames}")
        for row in reader:
            if row[src] and row[dst] and (keep is None or row[dst] in keep):
                out.setdefault(row[src], row[dst])
    return out


def require_paper_extra() -> None:
    """Exit with the install hint when the optional dependencies are missing."""
    import importlib.util

    missing = [m for m in ("sklearn", "pandas", "joblib") if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit("error: the downstream scripts need the optional dependencies: "
                         "uv sync --extra paper")


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def log(msg: str) -> None:
    print(msg, flush=True)
