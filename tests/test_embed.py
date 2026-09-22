"""Tests for orbit.embed (skipped unless the `embed` extra is installed)."""

from __future__ import annotations

import gzip
import importlib.util
import itertools

import numpy as np
import pytest

from orbit.embed import read_network
from orbit.io import read_embeddings

needs_embed_extra = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("pecanpy", "gensim", "numba")),
    reason="install with: uv sync --extra embed")


def two_cliques(path, gz=False, header=False):
    """Two 6-cliques joined by one weak edge."""
    a = [f"A{i}" for i in range(6)]
    b = [f"B{i}" for i in range(6)]
    lines = []
    if header:
        lines.append("Source\tTarget\tzScore")
    for u, v in itertools.combinations(a, 2):
        lines.append(f"{u}\t{v}\t2.0")
    for u, v in itertools.combinations(b, 2):
        lines.append(f"{u}\t{v}\t2.0")
    lines.append("A0\tB0\t0.1")
    text = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)
    return a, b


def test_read_network_numbers_genes_in_order_of_first_appearance_and_skips_a_header(tmp_path):
    p = tmp_path / "net.tsv"
    two_cliques(p, header=True)
    ids, src, dst, w = read_network(p)
    assert ids[:3] == ["A0", "A1", "A2"] and len(ids) == 12
    assert len(src) == len(dst) == len(w) == 31
    assert w[-1] == 0.1 and (ids[src[-1]], ids[dst[-1]]) == ("A0", "B0")


def test_read_network_reads_gzipped_files(tmp_path):
    p = tmp_path / "net.tsv.gz"
    two_cliques(p, gz=True)
    ids, *_ = read_network(p)
    assert len(ids) == 12


@needs_embed_extra
def test_embed_network_separates_the_two_cliques(tmp_path):
    from orbit.embed import embed_network

    p = tmp_path / "net.tsv.gz"
    a, b = two_cliques(p, gz=True)
    out = tmp_path / "net.h5"
    n = embed_network(p, out, dimensions=8, num_walks=20, walk_length=10, window_size=3, epochs=5, workers=1)
    assert n == 12
    X, ids = read_embeddings(out)
    assert X.shape == (12, 8) and X.dtype == np.float32 and set(ids) == set(a + b)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    pos = {g: i for i, g in enumerate(ids)}
    within = np.mean([S[pos[u], pos[v]] for grp in (a, b) for u, v in itertools.combinations(grp, 2)])
    across = np.mean([S[pos[u], pos[v]] for u in a for v in b])
    assert within > across


def test_cli_embed_explains_the_missing_extra_and_creates_nothing(tmp_path, monkeypatch):
    import orbit.embed
    from orbit.cli import main

    def unavailable():
        raise ImportError("No module named 'pecanpy'")

    monkeypatch.setattr(orbit.embed, "import_backends", unavailable)
    p = tmp_path / "net.tsv"
    two_cliques(p)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        main(["embed", str(p), "--out", str(out)])
    assert "uv sync --extra embed" in str(e.value)
    assert not out.exists()


@needs_embed_extra
def test_embed_network_rejects_weights_that_are_not_positive(tmp_path):
    """All-zero weights would normalise to NaN and negative weights would flip sign; both
    must be an error naming the cause instead of a silently corrupt embedding."""
    from orbit.embed import embed_network

    for weights in ("0", "-1"):
        net = tmp_path / f"net{weights}.tsv"
        net.write_text(f"a\tb\t{weights}\nb\tc\t{weights}\n")
        with pytest.raises(ValueError, match="positive"):
            embed_network(net, tmp_path / "out.h5", num_walks=1, walk_length=3, epochs=1, workers=1)
