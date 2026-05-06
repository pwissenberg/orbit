#!/usr/bin/env python3
"""Find orthogroups with highest mean cosine similarity in the aligned embedding space.

For each orthogroup with members from ≥2 species, computes mean pairwise cosine
similarity between all cross-species gene pairs. Outputs a ranked list for
identifying compelling zoom-in examples for UMAP visualization.

Usage:
    python scripts/find_best_orthogroups.py
    python scripts/find_best_orthogroups.py --aligned-dir results/aligned_embeddings_procrustes_n2v
    python scripts/find_best_orthogroups.py --min-species 5 --top 50
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parent.parent
OG_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
DEFAULT_ALIGNED_DIR = ROOT / "results" / "aligned_embeddings_procrustes_n2v"
OUTPUT_PATH = ROOT / "results" / "best_orthogroups.json"


def load_orthogroup_mapping(species: str) -> dict[str, str]:
    """Load protein -> orthogroup mapping for a species."""
    path = OG_DIR / f"{species}_transcripts_to_OG.tsv"
    if not path.exists():
        return {}
    mapping = {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                transcript_id, protein_id, og = parts[0], parts[1], parts[2]
                # Map both transcript and protein IDs
                mapping[transcript_id] = og
                mapping[protein_id] = og
    return mapping


def load_aligned_embeddings(species: str, aligned_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load aligned embeddings for a species."""
    path = aligned_dir / f"{species}.h5"
    with h5py.File(path, "r") as fh:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in fh["proteins"][:]]
        embeddings = fh["embeddings"][:].astype(np.float32)
    return embeddings, proteins


def main():
    parser = argparse.ArgumentParser(description="Find orthogroups with highest cosine similarity")
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR,
                        help="Directory with aligned H5 embeddings")
    parser.add_argument("--min-species", type=int, default=3,
                        help="Minimum number of species in orthogroup (default: 3)")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of top orthogroups to output (default: 100)")
    parser.add_argument("--max-og-size", type=int, default=500,
                        help="Skip orthogroups with more members (too generic, default: 500)")
    args = parser.parse_args()

    aligned_dir = args.aligned_dir
    h5_files = sorted(aligned_dir.glob("*.h5"))
    species_list = [f.stem for f in h5_files]
    print(f"Found {len(species_list)} species with aligned embeddings")

    # Step 1: Build orthogroup -> [(species, gene_idx, embedding)] mapping
    og_members: dict[str, list[tuple[str, str, np.ndarray]]] = defaultdict(list)
    n_mapped = 0

    for i, sp in enumerate(species_list):
        og_map = load_orthogroup_mapping(sp)
        if not og_map:
            print(f"  {sp}: no orthogroup mapping, skipping")
            continue

        embeddings, proteins = load_aligned_embeddings(sp, aligned_dir)

        # Normalize embeddings for cosine similarity = dot product
        norms = norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        matched = 0
        for idx, prot in enumerate(proteins):
            og = og_map.get(prot)
            if og is None:
                # Try stripping version suffix (.1, .2, etc.)
                base = prot.rsplit(".", 1)[0] if "." in prot else prot
                og = og_map.get(base)
            if og is not None:
                og_members[og].append((sp, prot, embeddings[idx]))
                matched += 1

        n_mapped += matched
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(species_list)}] {sp}: {matched}/{len(proteins)} genes mapped to OGs")

    print(f"\nTotal: {n_mapped:,} genes mapped across {len(og_members):,} orthogroups")

    # Step 2: Filter orthogroups by min species count and max size
    filtered = {}
    for og, members in og_members.items():
        species_in_og = set(sp for sp, _, _ in members)
        if len(species_in_og) >= args.min_species and len(members) <= args.max_og_size:
            filtered[og] = members

    print(f"Orthogroups with ≥{args.min_species} species and ≤{args.max_og_size} members: {len(filtered):,}")

    # Step 3: Compute mean cross-species cosine similarity per orthogroup
    results = []
    for j, (og, members) in enumerate(filtered.items()):
        species_in_og = set(sp for sp, _, _ in members)

        # Only compute CROSS-species similarities
        cross_sims = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if members[a][0] != members[b][0]:  # different species
                    sim = float(np.dot(members[a][2], members[b][2]))
                    cross_sims.append(sim)

        if not cross_sims:
            continue

        mean_sim = float(np.mean(cross_sims))
        results.append({
            "orthogroup": og,
            "mean_cosine_similarity": round(mean_sim, 6),
            "median_cosine_similarity": round(float(np.median(cross_sims)), 6),
            "min_cosine_similarity": round(float(np.min(cross_sims)), 6),
            "n_genes": len(members),
            "n_species": len(species_in_og),
            "n_cross_pairs": len(cross_sims),
            "species": sorted(species_in_og),
            "genes": [(sp, gene) for sp, gene, _ in members],
        })

        if (j + 1) % 5000 == 0:
            print(f"  [{j+1}/{len(filtered)}] orthogroups processed")

    # Sort by mean cosine similarity (descending)
    results.sort(key=lambda x: x["mean_cosine_similarity"], reverse=True)

    # Print top results
    print(f"\n{'='*70}")
    print(f"Top {min(args.top, len(results))} orthogroups by mean cross-species cosine similarity")
    print(f"{'='*70}")
    for i, r in enumerate(results[:args.top]):
        print(f"  {i+1:3d}. {r['orthogroup']:12s}  "
              f"cos={r['mean_cosine_similarity']:.4f}  "
              f"genes={r['n_genes']:3d}  "
              f"species={r['n_species']:3d}  "
              f"pairs={r['n_cross_pairs']:5d}")

    # Save full results
    output = {
        "aligned_dir": str(aligned_dir),
        "n_species": len(species_list),
        "n_orthogroups_total": len(og_members),
        "n_orthogroups_filtered": len(filtered),
        "min_species_filter": args.min_species,
        "max_og_size_filter": args.max_og_size,
        "top_orthogroups": results[:args.top],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved top {min(args.top, len(results))} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
