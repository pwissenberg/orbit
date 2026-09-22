"""HDF5 layout of the ORBIT embedding files (the layout of the Zenodo archive).

Every file holds two parallel datasets: ``proteins`` (variable-length strings, one gene or
protein identifier per row) and ``embeddings`` (float32, ``n x d``).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def read_embeddings(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Return ``(X, ids)`` with ``X`` float32 ``(n, d)`` and ``ids`` a list of ``str``."""
    with h5py.File(path, "r") as f:
        X = f["embeddings"][:].astype(np.float32, copy=False)
        ids = [p.decode() if isinstance(p, bytes) else str(p) for p in f["proteins"][:]]
    return X, ids


def read_ids(path: str | Path) -> list[str]:
    """The identifiers of an embedding file without loading the matrix."""
    with h5py.File(path, "r") as f:
        return [p.decode() if isinstance(p, bytes) else str(p) for p in f["proteins"][:]]


def write_embeddings(path: str | Path, X: np.ndarray, ids: list[str]) -> None:
    """Write ``X`` (cast to float32) and ``ids`` in the Zenodo layout, overwriting ``path``."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != len(ids):
        raise ValueError(f"embeddings {X.shape} do not match {len(ids)} identifiers")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("proteins", data=np.array(list(ids), dtype=object),
                         dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("embeddings", data=X.astype(np.float32, copy=False))
