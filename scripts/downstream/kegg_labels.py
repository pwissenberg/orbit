#!/usr/bin/env python3
"""KEGG pathway membership per gene, as the label table of ``go_transfer.py`` (Table S6).

Pathway-gene links come from the KEGG REST API (``https://rest.kegg.jp/link/<code>/pathway``)
and are cached under ``--cache``; the overview maps 01100 and 01110 are excluded. KEGG gene
ids are mapped onto the gene ids of the aligned embeddings produced before with
``orbit align`` (``results/plant/<SPECIES>.h5``): A. thaliana by locus (``AT1G01090`` to the
``.1`` isoform present in the embedding), every other species through
``https://rest.kegg.jp/conv/uniprot/<code>`` and the ``<SPECIES>_to_uniprot.tsv`` mapping.

Output: a TSV with ``protein``, ``term`` (pathway number) and ``aspect`` (``KEGG``).

    python scripts/downstream/kegg_labels.py --species ARATH=ath ORYSA=osa ZEAMA=zma GLYMA=gmx MEDTR=mtr POPTR=pop \\
        --embeddings results/plant --idmap-dir data/id_mapping --cache data/kegg --out data/kegg_labels.tsv
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import log, read_idmap  # noqa: E402
from orbit.io import read_ids  # noqa: E402

EXCLUDE = {"01100", "01110"}


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        log(f"downloading {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def read_links(path: Path, code: str, exclude: set[str] = EXCLUDE) -> dict[str, set[str]]:
    """``{KEGG gene: set(pathway numbers)}`` from a ``link/<code>/pathway`` response."""
    out: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            pathway = parts[0].replace(f"path:{code}", "")
            gene = parts[1].replace(f"{code}:", "")
            if pathway not in exclude:
                out.setdefault(gene, set()).add(pathway)
    return out


def read_conv(path: Path, code: str) -> dict[str, str]:
    """``{KEGG gene: UniProt accession}`` from a ``conv/uniprot/<code>`` response."""
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                out.setdefault(parts[0].replace(f"{code}:", ""), parts[1].replace("up:", ""))
    return out


def arath_locus_map(gene_ids: list[str]) -> dict[str, str]:
    """``{locus: gene id}`` preferring the ``.1`` isoform."""
    out: dict[str, str] = {}
    for gid in gene_ids:
        locus = gid.rsplit(".", 1)[0]
        if locus not in out or gid.endswith(".1"):
            out[locus] = gid
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", nargs="+", required=True, metavar="SPECIES=CODE",
                   help="species code of the embedding files and its KEGG organism code")
    p.add_argument("--embeddings", type=Path, required=True, metavar="DIR", help="<SPECIES>.h5 files")
    p.add_argument("--idmap-dir", type=Path, metavar="DIR", help="<SPECIES>_to_uniprot.tsv tables")
    p.add_argument("--idmap-columns", default="uniprot_accession,teagcn_id", metavar="SRC,DST")
    p.add_argument("--by-locus", nargs="*", default=["ARATH"], metavar="SPECIES",
                   help="species whose KEGG gene ids are loci of the embedding ids (default ARATH)")
    p.add_argument("--cache", type=Path, required=True, metavar="DIR", help="KEGG REST responses")
    p.add_argument("--out", type=Path, required=True, metavar="TSV")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    rows = []
    for spec in a.species:
        if spec.count("=") != 1:
            raise SystemExit(f"error: --species expects SPECIES=CODE, got {spec!r}")
        sp, code = spec.split("=")
        gene_ids = read_ids(a.embeddings / f"{sp}.h5")
        links = read_links(fetch(f"https://rest.kegg.jp/link/{code}/pathway", a.cache / f"kegg_{code}_links.tsv"), code)
        if sp in a.by_locus:
            to_gene = arath_locus_map(gene_ids)
        else:
            if a.idmap_dir is None:
                raise SystemExit(f"error: {sp} needs --idmap-dir to map KEGG genes via UniProt")
            conv = read_conv(fetch(f"https://rest.kegg.jp/conv/uniprot/{code}", a.cache / f"kegg_{code}_to_uniprot.tsv"), code)
            present = set(gene_ids)
            idmap = {u: g for u, g in read_idmap(a.idmap_dir / f"{sp}_to_uniprot.tsv",
                                                 tuple(a.idmap_columns.split(","))).items() if g in present}
            to_gene = {k: idmap[u] for k, u in conv.items() if u in idmap}
        n = 0
        for kegg_gene, pathways in links.items():
            gene = to_gene.get(kegg_gene)
            if gene is None:
                continue
            for pw in sorted(pathways):
                rows.append((gene, pw, "KEGG"))
            n += 1
        log(f"{sp} ({code}): {n} of {len(links)} KEGG genes mapped onto embedding ids")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("protein\tterm\taspect\n" + "".join(f"{g}\t{t}\t{asp}\n" for g, t, asp in rows))
    log(f"wrote {a.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
