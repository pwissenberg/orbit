"""Node2Vec embedding of a coexpression network with the SPACE defaults (paper Section 2.2).

Optional: needs the ``embed`` extra (pecanpy, gensim, numba). The procedure replays the one
SPACE uses for its per-species embeddings, so that embeddings made here are the same kind of
input ORBIT was evaluated on: edge weights are quantised to thousandths of the maximum weight,
walks come from pecanpy's second-order walker (``SparseOTF``) and the vectors from gensim's
skip-gram model with negative sampling. Random walks and multi-threaded training make the
result reproducible in distribution, not bit for bit.

While training, gensim 4.4 may print ``Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'``
a few times. That is a gensim/Cython 3 quirk (a dot product that is exactly -1.0 is mistaken
for an error code by the compiled inner loop) and affects one update out of about a billion;
the embeddings are fine.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import numpy as np

from orbit.io import write_embeddings

SPACE_DEFAULTS = dict(dimensions=128, p=0.3, q=0.7, num_walks=10, walk_length=50, window_size=5, epochs=5)

# nptyping, pulled in by pecanpy, still refers to aliases that NumPy 2 removed.
_NUMPY_ALIASES = {
    "bool8": "bool_", "object0": "object_", "int0": "intp", "uint0": "uintp", "void0": "void",
    "bytes0": "bytes_", "str0": "str_", "float_": "float64", "complex_": "complex128",
    "longfloat": "longdouble", "singlecomplex": "complex64", "cfloat": "complex128",
    "longcomplex": "clongdouble", "clongfloat": "clongdouble", "string_": "bytes_", "unicode_": "str_",
}


def import_backends():
    for name, target in _NUMPY_ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, getattr(np, target))
    import numba
    from gensim.models import Word2Vec
    from pecanpy import pecanpy as pp
    return pp, Word2Vec, numba


def read_network(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read an edge list ``geneA<TAB>geneB<TAB>weight`` (gzipped or plain; a header line is
    skipped). Returns ``(ids, src, dst, weight)`` with genes numbered in order of first
    appearance, as SPACE does."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    index: dict[str, int] = {}
    src, dst, w = [], [], []
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                weight = float(parts[2])
            except ValueError:
                continue  # header
            for g in (parts[0], parts[1]):
                if g not in index:
                    index[g] = len(index)
            src.append(index[parts[0]])
            dst.append(index[parts[1]])
            w.append(weight)
    if not w:
        raise ValueError(f"no edges read from {path}")
    return list(index), np.array(src), np.array(dst), np.array(w, dtype=np.float64)


def embed_network(network: str | Path, out_h5: str | Path, *, dimensions: int = 128, p: float = 0.3,
                  q: float = 0.7, num_walks: int = 10, walk_length: int = 50, window_size: int = 5,
                  epochs: int = 5, workers: int = -1, random_state: int = 1234) -> int:
    """Embed one network and write ``out_h5`` in the ORBIT layout. Returns the gene count."""
    pp, Word2Vec, numba = import_backends()
    ids, src, dst, w = read_network(network)
    w = np.trunc(w / w.max() * 1000) / 1000  # SPACE stores integer scores 0..1000, then divides
    n_workers = numba.config.NUMBA_DEFAULT_NUM_THREADS if workers == -1 else workers
    with tempfile.TemporaryDirectory() as tmp:
        edges = Path(tmp) / "edges.tsv"
        with open(edges, "w") as f:
            for s, d, x in zip(src, dst, w):
                f.write(f"{s}\t{d}\t{x}\n")
        graph = pp.SparseOTF(p, q, n_workers, False, False, 0, random_state)  # verbose, extend, gamma
        graph.read_edg(str(edges), True, False, "\t")  # weighted, undirected
    graph.preprocess_transition_probs()
    walks = graph.simulate_walks(num_walks, walk_length)
    model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, hs=0,
                     negative=5, workers=n_workers, epochs=epochs, seed=random_state)
    order = [ids[int(k)] for k in model.wv.index_to_key]
    write_embeddings(out_h5, model.wv.vectors, order)
    return len(order)
