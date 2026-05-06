"""End-to-end example: embed the five seed-species coexpression networks
shipped in `data/networks/` with Node2Vec.

Reproduces the embedding stage of the ORBIT pipeline using the same
hyperparameters reported in the paper (Table S1b winner):

    p = 1.0, q = 0.7, num_walks = 20, walk_length = 50, epochs = 10

Outputs one HDF5 file per species at `data/node2vec/{species}.h5`,
containing a 128-dim float32 embedding per gene plus the gene-id list.

Usage:
    uv run python scripts/example_embed_seeds.py            # all five seeds
    uv run python scripts/example_embed_seeds.py --species ARATH ORYSA
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import tempfile
import time
from pathlib import Path

import orbit._compat  # noqa: F401  patch deprecated numpy aliases used by pecanpy
from orbit.data_prep import convert_to_space_format, generate_embeddings

ROOT = Path(__file__).resolve().parent.parent

NETWORKS_DIR = ROOT / "data" / "networks"
NODE2VEC_DIR = ROOT / "data" / "node2vec"

SEED_SPECIES = ["ARATH", "ORYSA", "PICAB", "SELMO", "MARPO"]

N2V_PARAMS = dict(
    dimensions=128,
    p=1.0,
    q=0.7,
    num_walks=20,
    walk_length=50,
    epochs=10,
)


def embed_one(species: str) -> Path:
    """Run SPACE-format conversion + Node2Vec on one bundled seed network."""
    src_gz = NETWORKS_DIR / f"{species}.tsv.gz"
    if not src_gz.exists():
        raise SystemExit(f"missing network: {src_gz}")

    h5_out = NODE2VEC_DIR / f"{species}.h5"

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        clean_tsv = tmp_dir / f"{species}.tsv"
        print(f"[{species}] decompressing {src_gz.name}...")
        with gzip.open(src_gz, "rb") as fin, open(clean_tsv, "wb") as fout:
            shutil.copyfileobj(fin, fout)

        space_gz = tmp_dir / f"{species}.space.tsv.gz"
        print(f"[{species}] writing SPACE/pecanpy edge list...")
        convert_to_space_format(clean_tsv, space_gz)

        print(f"[{species}] training Node2Vec {N2V_PARAMS}")
        generate_embeddings(space_gz, h5_out, **N2V_PARAMS)

    print(f"[{species}] done in {time.time()-t0:.1f}s -> {h5_out}")
    return h5_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        nargs="*",
        default=SEED_SPECIES,
        choices=SEED_SPECIES,
        help="Subset of seed species to embed (default: all five).",
    )
    args = parser.parse_args()

    NODE2VEC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Embedding {len(args.species)} species: {args.species}")
    for sp in args.species:
        embed_one(sp)
    print("All seed embeddings written to", NODE2VEC_DIR)


if __name__ == "__main__":
    main()
