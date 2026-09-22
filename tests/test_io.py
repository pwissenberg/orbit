"""Tests for orbit.io: the HDF5 embedding layout of the Zenodo archive."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from orbit.io import read_embeddings, write_embeddings


def test_round_trip_preserves_ids_and_float32_values(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4)).astype(np.float32)
    ids = ["AT1G01010.1", "AT1G01020.1", "AT1G01030.1", "AT1G01040.1", "AT1G01050.1"]
    p = tmp_path / "ARATH.h5"
    write_embeddings(p, X, ids)
    Y, out_ids = read_embeddings(p)
    assert out_ids == ids
    assert Y.dtype == np.float32
    assert np.array_equal(X, Y)


def test_written_file_has_the_zenodo_layout(tmp_path):
    p = tmp_path / "x.h5"
    write_embeddings(p, np.zeros((2, 3), dtype=np.float64), ["a", "b"])
    with h5py.File(p) as f:
        assert set(f.keys()) == {"proteins", "embeddings"}
        assert f["embeddings"].dtype == np.float32
        assert f["embeddings"].shape == (2, 3)
        assert h5py.check_string_dtype(f["proteins"].dtype) is not None
        assert [x.decode() for x in f["proteins"][:]] == ["a", "b"]


def test_read_accepts_byte_and_str_identifiers(tmp_path):
    p = tmp_path / "bytes.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("proteins", data=np.array([b"g1", b"g2"], dtype=object),
                         dtype=h5py.special_dtype(vlen=bytes))
        f.create_dataset("embeddings", data=np.ones((2, 2), dtype=np.float32))
    X, ids = read_embeddings(p)
    assert ids == ["g1", "g2"]
    assert X.shape == (2, 2)


def test_write_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        write_embeddings(tmp_path / "bad.h5", np.zeros((3, 2)), ["a", "b"])
