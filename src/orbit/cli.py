"""The ``orbit`` command line: one pipeline for both data tracks.

    orbit align   two-stage rotation of many species onto a reference (plant coexpression
                  networks with OrthoFinder anchors, or STRING networks with eggNOG anchors)
    orbit fetch   download the STRING inputs from the SPACE Zenodo record
    orbit embed   Node2Vec embedding of your own coexpression networks (optional extra)

Both tracks feed ``orbit align`` the same three inputs: a directory of per-species HDF5
embeddings named ``<species>.h5``, a seeds file with one species identifier per line, and an
orthogroup source. A directory of OrthoFinder tables gives one anchor row per ortholog pair;
an eggNOG members file gives one centroid row per shared orthogroup (paper Section 2.5).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from orbit import __version__
from orbit.align import iter_two_stage_rotations
from orbit.anchors import (
    centroid_anchors,
    load_orthogroups,
    nearest_seed,
    orthogroup_centroids,
    orthogroup_pair_anchors,
)
from orbit.io import read_embeddings, read_ids, write_embeddings

REPORT_COLUMNS = ["species", "role", "aligned_to", "n_anchors", "n_genes", "status"]


def log(msg: str) -> None:
    print(msg, flush=True)


def read_list(path: Path) -> list[str]:
    """One identifier per line, in order and without duplicates; blank lines and ``#``
    comments are ignored."""
    lines = (l.strip() for l in Path(path).read_text().splitlines())
    return list(dict.fromkeys(l for l in lines if l and not l.startswith("#")))


def read_assignment(path: Path) -> dict[str, str]:
    """``{non_seed: seed}`` from a JSON file; the ``{"groups": {...}}`` layout is accepted too."""
    d = json.loads(Path(path).read_text())
    if isinstance(d.get("groups"), dict):
        d = d["groups"]
    return {str(k): str(v) for k, v in d.items()}


# --- align --------------------------------------------------------------------------------

ALIGN_HELP = """\
Rotate every species onto the reference in two stages (paper Sections 2.3 and 2.5).

Stage 1 rotates each seed onto the reference, which is left unchanged; stage 2 rotates every
other species onto the seed with which it shares the most orthogroups, in that seed's aligned
coordinates, so all species end up in one frame. Each rotation is one closed-form orthogonal
Procrustes solve. Only the seeds and the species being placed are held in memory.

Plant coexpression track (OrthoFinder tables, one anchor row per ortholog pair):
  orbit align --embeddings data/plant/node2vec --orthogroups data/plant/orthogroups \\
              --seeds data/plant/seeds.txt --reference ARATH --out results/plant

STRING track (eggNOG members table, one centroid row per shared orthogroup; the released
files were solved without the proper-rotation constraint, hence --allow-reflection):
  orbit align --embeddings data/string/node2vec --orthogroups data/string/eggnog/2759.tsv.gz \\
              --seeds data/string/seeds.txt --reference 9606 --allow-reflection --out results/string

Add your own species to an already aligned frame (for example the released seed files):
  orbit align --embeddings my/embeddings --orthogroups my/orthogroups --seeds data/plant/seeds.txt \\
              --reference ARATH --seeds-aligned --species MYSPEC --out results/mine
"""


def cmd_align(a: argparse.Namespace) -> int:
    t0 = time.time()
    seeds = read_list(a.seeds)
    if not seeds:
        raise SystemExit(f"error: no seed species listed in {a.seeds}")
    if a.reference not in seeds:
        raise SystemExit(f"error: reference {a.reference} is not listed in {a.seeds}")
    available = sorted(p.stem for p in a.embeddings.glob("*.h5"))
    missing = [s for s in seeds if s not in available]
    if missing:
        raise SystemExit(f"error: no embedding under {a.embeddings} for seed(s) {missing}")
    wanted = list(dict.fromkeys(a.species)) if a.species is not None else available
    nonseeds = [s for s in wanted if s not in seeds]
    unknown = [s for s in nonseeds if s not in available]
    if unknown:
        raise SystemExit(f"error: no embedding under {a.embeddings} for {unknown}")

    # Pass 1: identifiers only, to restrict the orthogroup tables to genes that have a vector.
    species = seeds + nonseeds
    ids = {s: read_ids(a.embeddings / f"{s}.h5") for s in species}
    try:
        og, kind = load_orthogroups(a.orthogroups, ids)
    except OSError as e:  # missing table, unreadable file
        raise SystemExit(f"error: {e}")
    anchor_kind = a.anchors if a.anchors != "auto" else ("pairs" if kind == "orthofinder" else "centroids")
    log(f"{len(seeds)} seeds, {len(nonseeds)} non-seeds, {kind} orthogroups, "
        f"{anchor_kind} anchors ({time.time() - t0:.1f}s)")

    if a.assignment is not None:
        groups = read_assignment(a.assignment)
        missing = [s for s in nonseeds if s not in groups]
        if missing:
            raise SystemExit(f"error: {a.assignment} assigns no seed to {missing[:5]}")
        seed_of = {s: groups[s] for s in nonseeds}
        bad = sorted({v for v in seed_of.values() if v not in seeds})
        if bad:
            raise SystemExit(f"error: {a.assignment} assigns species to unknown seeds {bad}")
    else:
        seed_ogs = {s: set(og[s].values()) for s in seeds}
        seed_of = {}
        for s in nonseeds:
            try:
                seed_of[s] = nearest_seed(og[s].values(), seed_ogs)
            except ValueError:
                pass  # shares no orthogroup with any seed: reported as skipped below

    # Embeddings: the seeds stay loaded, a non-seed only while it is being placed.
    cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    def embedding(s):
        if s not in cache:
            cache[s] = read_embeddings(a.embeddings / f"{s}.h5")
        return cache[s]

    for s in seeds:
        embedding(s)
    dim = cache[a.reference][0].shape[1]
    n_anchors: dict[str, int] = {}
    if anchor_kind == "pairs":
        def anchors(source, target):
            (Xs, ids_s), (Xt, ids_t) = embedding(source), embedding(target)
            A, B = orthogroup_pair_anchors(Xs, ids_s, og[source], Xt, ids_t, og[target])
            n_anchors[source] = len(A)
            return A, B
    else:
        cent = {s: orthogroup_centroids(*embedding(s), og[s]) for s in seeds}

        def anchors(source, target):
            cs = cent[source] if source in cent else orthogroup_centroids(*embedding(source), og[source])
            A, B = centroid_anchors(cs, cent[target], dim)
            n_anchors[source] = len(A)
            return A, B

    a.out.mkdir(parents=True, exist_ok=True)
    if a.rotations is not None:
        a.rotations.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    t1 = time.time()
    try:
        for s, R in iter_two_stage_rotations(anchors, seeds, a.reference, seed_of,
                                             proper=not a.allow_reflection, min_anchors=a.min_anchors,
                                             seeds_aligned=a.seeds_aligned):
            if R is None:
                status[s] = "skipped"
                log(f"  {s}: {n_anchors.get(s, 0)} anchors to {seed_of.get(s) or a.reference}, skipped")
                continue
            X, gene_ids = embedding(s)
            if a.seeds_aligned and s in seeds:
                status[s] = "unchanged"  # already in the reference frame, not rewritten
            else:
                write_embeddings(a.out / f"{s}.h5", (X.astype(np.float64) @ R).astype(np.float32), gene_ids)
                status[s] = "ok"
            if a.rotations is not None:
                np.save(a.rotations / f"{s}.npy", R.astype(np.float32))
            if s not in seeds:
                cache.pop(s, None)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    log(f"solved {sum(v != 'skipped' for v in status.values())} rotations ({time.time() - t1:.1f}s)")

    rows = []
    for s in species:
        role = "reference" if s == a.reference else "seed" if s in seeds else "non-seed"
        target = "" if s == a.reference else a.reference if s in seeds else seed_of.get(s, "")
        st = status.get(s, "skipped")  # a non-seed without any shared orthogroup is never solved
        if st == "skipped" and s not in status:
            log(f"  {s}: no orthogroup shared with any seed, skipped")
        rows.append((s, role, target, n_anchors.get(s, 0), len(ids[s]), st))
    with open(a.out / "alignment_report.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(REPORT_COLUMNS)
        w.writerows(rows)
    placed = sum(r[-1] != "skipped" for r in rows)
    log(f"aligned {placed}/{len(species)} species -> {a.out} ({time.time() - t0:.1f}s total)")
    return 0 if placed == len(species) else 1


# --- fetch --------------------------------------------------------------------------------

def cmd_fetch_string(a: argparse.Namespace) -> int:
    from orbit.fetch import fetch_string

    try:
        return fetch_string(a.dest, taxa=a.taxa, all_=a.all, keep_archives=a.keep_archives)
    except (ValueError, RuntimeError, OSError) as e:  # bad taxa, failed download, no network
        raise SystemExit(f"error: {e}")


# --- embed --------------------------------------------------------------------------------

def cmd_embed(a: argparse.Namespace) -> int:
    from orbit.embed import embed_network, import_backends

    try:
        import_backends()
    except ImportError as e:
        raise SystemExit(f"error: `orbit embed` needs the optional dependencies (uv sync --extra embed): {e}")
    a.out.mkdir(parents=True, exist_ok=True)
    for net in a.networks:
        name = net.name
        for suffix in (".gz", ".tsv", ".txt"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        t0 = time.time()
        n = embed_network(net, a.out / f"{name}.h5", dimensions=a.dimensions, p=a.p, q=a.q,
                          num_walks=a.num_walks, walk_length=a.walk_length, window_size=a.window,
                          epochs=a.epochs, workers=a.workers, random_state=a.seed)
        log(f"{name}: {n} genes x {a.dimensions} -> {a.out / f'{name}.h5'} ({time.time() - t0:.0f}s)")
    return 0


# --- parser -------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orbit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"orbit {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{align,fetch,embed}")

    al = sub.add_parser("align", help="rotate species onto a reference (either track)",
                        description=ALIGN_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    al.add_argument("--embeddings", type=Path, required=True, metavar="DIR",
                    help="directory of <species>.h5 files (datasets 'proteins' and 'embeddings')")
    al.add_argument("--orthogroups", type=Path, required=True, metavar="PATH",
                    help="directory of OrthoFinder <species>_transcripts_to_OG.tsv tables, "
                         "or one eggNOG members table <level>.tsv.gz")
    al.add_argument("--seeds", type=Path, required=True, metavar="FILE",
                    help="one seed species per line; the order breaks ties in the nearest-seed rule")
    al.add_argument("--reference", required=True, metavar="ID",
                    help="the seed that defines the shared frame; it is left unchanged")
    al.add_argument("--out", type=Path, required=True, metavar="DIR",
                    help="output directory: <species>.h5 per species and alignment_report.tsv")
    al.add_argument("--species", nargs="*", default=None, metavar="ID",
                    help="non-seeds to align (default: every embedding that is not a seed)")
    al.add_argument("--assignment", type=Path, metavar="JSON",
                    help="{non_seed: seed} mapping that fixes stage 2 "
                         "(default: the seed with the most shared orthogroups)")
    al.add_argument("--anchors", choices=["auto", "pairs", "centroids"], default="auto",
                    help="one anchor row per ortholog pair or per orthogroup centroid "
                         "(default: pairs for OrthoFinder tables, centroids for eggNOG)")
    al.add_argument("--allow-reflection", action="store_true",
                    help="accept det R = -1, the unconstrained orthogonal Procrustes solution; "
                         "reproduces the released STRING files")
    al.add_argument("--seeds-aligned", action="store_true",
                    help="the seed files already share the reference frame (e.g. the released "
                         "ones): skip stage 1 and leave them unchanged")
    al.add_argument("--min-anchors", type=int, default=2, metavar="N",
                    help="skip a species with fewer anchor rows (default and minimum 2)")
    al.add_argument("--rotations", type=Path, metavar="DIR",
                    help="also save every rotation as <species>.npy (float32) under this directory")
    al.set_defaults(func=cmd_align)

    fe = sub.add_parser("fetch", help="download inputs", description="Download the inputs of a track.")
    fsub = fe.add_subparsers(dest="track", required=True, metavar="{string}")
    fs = fsub.add_parser("string", help="STRING inputs from the SPACE Zenodo record 15600639",
                         description="Fetch the SPACE Node2Vec embeddings and eggNOG orthogroups that "
                                     "`orbit align` needs for the STRING track (seeds are always included).")
    fs.add_argument("--taxa", nargs="*", default=[], metavar="TAXID",
                    help="non-seed NCBI taxids to fetch in addition to the 48 seeds")
    fs.add_argument("--all", action="store_true", help="download both archives completely (all 1,322 species)")
    fs.add_argument("--dest", type=Path, default=Path("data/string"), metavar="DIR",
                    help="destination directory (default data/string)")
    fs.add_argument("--keep-archives", action="store_true", help="with --all, keep the downloaded zip files")
    fs.set_defaults(func=cmd_fetch_string)

    em = sub.add_parser("embed", help="Node2Vec embedding of coexpression networks (needs --extra embed)",
                        description="Embed weighted networks (geneA<TAB>geneB<TAB>weight, gzipped or "
                                    "plain) with Node2Vec using the SPACE defaults of the paper "
                                    "(Section 2.2). Writes <out>/<network name>.h5. A few 'Exception "
                                    "ignored in ... our_dot_float' lines from gensim are harmless.")
    em.add_argument("networks", nargs="+", type=Path, metavar="NETWORK",
                    help="edge list files, e.g. data/plant/networks/MARPO.tsv.gz")
    em.add_argument("--out", type=Path, required=True, metavar="DIR", help="output directory")
    em.add_argument("--dimensions", type=int, default=128)
    em.add_argument("--p", type=float, default=0.3, help="return parameter (default 0.3)")
    em.add_argument("--q", type=float, default=0.7, help="in-out parameter (default 0.7)")
    em.add_argument("--num-walks", type=int, default=10, help="walks per node (default 10)")
    em.add_argument("--walk-length", type=int, default=50, help="default 50")
    em.add_argument("--window", type=int, default=5, help="skip-gram window (default 5)")
    em.add_argument("--epochs", type=int, default=5, help="default 5")
    em.add_argument("--workers", type=int, default=-1, help="threads (default: all)")
    em.add_argument("--seed", type=int, default=1234, help="random seed (default 1234)")
    em.set_defaults(func=cmd_embed)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
