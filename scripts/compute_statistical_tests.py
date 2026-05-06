"""
Compute Mann-Whitney U tests for ortholog vs non-ortholog cosine similarity
across all species pairs. Tests whether orthologs are significantly closer
in the aligned embedding space than random gene pairs.
"""

import json
import numpy as np
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import mannwhitneyu
import time

ROOT = Path("/home/paulwissenberg/PLANT-SPACE")
ALIGNED_DIR = ROOT / "results/aligned_embeddings_proc_jaccard_iter_csls"
ORTHOLOGS_DIR = ROOT / "data/orthologs"
EVAL_FILE = ROOT / "results/evaluation_proc_jaccard_iter_csls.json"
OUT_FILE = ROOT / "results/statistical_tests_mannwhitney.json"

N_RANDOM = 10000  # number of random pairs to sample


def load_embeddings(species):
    path = ALIGNED_DIR / ("%s.h5" % species)
    with h5py.File(path) as f:
        emb = f["embeddings"][:]
        prots = [p.decode() for p in f["proteins"][:]]
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    return emb, prots


def load_ortholog_pairs(sp_a, sp_b):
    """Load ortholog pairs and return list of (idx_a, idx_b) tuples."""
    # Try seeds directory first, then nonseeds
    for subdir in ["seeds", "nonseeds"]:
        path = ORTHOLOGS_DIR / subdir / ("%s_%s.tsv" % (sp_a, sp_b))
        if path.exists():
            pairs = []
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        pairs.append((int(parts[0]), int(parts[1])))
            return pairs
        # Try reversed order
        path2 = ORTHOLOGS_DIR / subdir / ("%s_%s.tsv" % (sp_b, sp_a))
        if path2.exists():
            pairs = []
            with open(path2) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        pairs.append((int(parts[1]), int(parts[0])))
            return pairs
    return []


def test_one_pair(sp_a, sp_b):
    """Compute Mann-Whitney U for one species pair."""
    try:
        emb_a, prots_a = load_embeddings(sp_a)
        emb_b, prots_b = load_embeddings(sp_b)

        pairs = load_ortholog_pairs(sp_a, sp_b)
        if len(pairs) < 10:
            return sp_a, sp_b, None, "Too few pairs: %d" % len(pairs)

        # Filter valid indices
        valid = [(a, b) for a, b in pairs if a < len(emb_a) and b < len(emb_b)]
        if len(valid) < 10:
            return sp_a, sp_b, None, "Too few valid pairs: %d" % len(valid)

        # Cosine similarity for ortholog pairs
        idx_a = np.array([p[0] for p in valid])
        idx_b = np.array([p[1] for p in valid])
        ortho_sims = np.sum(emb_a[idx_a] * emb_b[idx_b], axis=1)

        # Cosine similarity for random pairs
        np.random.seed(42)
        rand_a = np.random.randint(0, len(emb_a), N_RANDOM)
        rand_b = np.random.randint(0, len(emb_b), N_RANDOM)
        random_sims = np.sum(emb_a[rand_a] * emb_b[rand_b], axis=1)

        # Mann-Whitney U test
        stat, pval = mannwhitneyu(ortho_sims, random_sims, alternative="greater")

        result = {
            "n_ortholog_pairs": len(valid),
            "n_random_pairs": N_RANDOM,
            "ortho_sim_mean": float(np.mean(ortho_sims)),
            "ortho_sim_median": float(np.median(ortho_sims)),
            "ortho_sim_std": float(np.std(ortho_sims)),
            "random_sim_mean": float(np.mean(random_sims)),
            "random_sim_median": float(np.median(random_sims)),
            "random_sim_std": float(np.std(random_sims)),
            "mann_whitney_U": float(stat),
            "p_value": float(pval),
        }
        return sp_a, sp_b, result, "OK (p=%.2e)" % pval

    except Exception as e:
        return sp_a, sp_b, None, "Error: %s" % str(e)


def main():
    t0 = time.time()

    # Get species pairs from evaluation
    with open(EVAL_FILE) as f:
        eval_data = json.load(f)

    pairs_to_test = [(d["species_a"], d["species_b"]) for d in eval_data]
    print("Testing %d species pairs" % len(pairs_to_test))

    results = {}
    with ProcessPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(test_one_pair, a, b): (a, b) for a, b in pairs_to_test}
        done = 0
        for fut in as_completed(futures):
            sp_a, sp_b, result, msg = fut.result()
            done += 1
            if result:
                results["%s_%s" % (sp_a, sp_b)] = result
            if done % 20 == 0:
                print("  %d/%d done (%.0fs)" % (done, len(pairs_to_test), time.time() - t0))

    # Summary
    p_values = [r["p_value"] for r in results.values()]
    sig = sum(1 for p in p_values if p < 0.05)
    print("\nResults: %d/%d pairs tested" % (len(results), len(pairs_to_test)))
    print("Significant (p < 0.05): %d/%d (%.1f%%)" % (sig, len(results), 100*sig/len(results)))
    print("Median p-value: %.2e" % np.median(p_values))
    print("Mean ortholog sim: %.4f vs random: %.4f" % (
        np.mean([r["ortho_sim_mean"] for r in results.values()]),
        np.mean([r["random_sim_mean"] for r in results.values()]),
    ))

    with open(str(OUT_FILE), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to %s" % OUT_FILE)
    print("Total time: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
