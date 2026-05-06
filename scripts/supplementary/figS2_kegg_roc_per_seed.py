#!/usr/bin/env python3
"""
Figure S2 — KEGG pathway ROC curves per species.

Per SPACE paper Fig. 2, plots cumulative TP vs FP curves on KEGG
pathway co-membership, comparing:
  - Procrustes-aligned (procn2v)
  - Raw N2V (node2vec)
  - ProtT5 (prott5)

Species included: 6 plants with KEGG pathway annotations available
on KEGG REST (ARATH, ORYSA, ZEAMA, GLYMA, MEDTR, POPTR). Of the
5 seeds, only ARATH and ORYSA have KEGG codes — PICAB, SELMO and
MARPO are not registered in KEGG.

Reuses helpers from scripts/benchmark_kegg.py for ID mapping and
embedding loading. Re-implements the TP/FP curve computation in a
vectorized form (using a sparse pathway-membership matrix and a
single argsort) to keep runtime manageable on species with ~10k
genes.

Spot check: per-species full_auc must match
~/PLANT-SPACE/results/downstream/kegg/kegg_comparison.json values.

Output:
  results/supplementary/figS2_kegg_roc_per_seed.{png,pdf}
  results/supplementary/figS2_kegg_roc_curves.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import faiss
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import auc

# Ensure logs flush
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=False, enqueue=False)

PROJECT_ROOT = Path.home() / "PLANT-SPACE"
OUT_DIR = PROJECT_ROOT / "results" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import benchmark_kegg as bk  # noqa: E402

SEED_SPECIES_WITH_KEGG = ["ARATH", "ORYSA"]
OTHER_KEGG_SPECIES = ["ZEAMA", "GLYMA", "MEDTR", "POPTR"]
ALL_SPECIES = SEED_SPECIES_WITH_KEGG + OTHER_KEGG_SPECIES

METHODS = {
    "procn2v": ("Procrustes (aligned N2V)", "#e07b54"),
    "node2vec": ("Node2Vec (raw)", "#7aaed4"),
    "prott5": ("ProtT5", "#5fb360"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def compute_kegg_roc_with_curves(
    embeddings: np.ndarray,
    protein_ids: list[str],
    gene_to_pathways: Dict[str, set],
    id_mapping: Dict[str, str],
    percent_threshold: float = 0.001,
    max_curve_points: int = 2500,
) -> dict:
    """Vectorized variant of bk.compute_kegg_roc that also returns curves.

    Steps:
      1. Map KEGG genes to embedding rows (drop unmapped).
      2. Build sparse gene x pathway membership matrix (dtype=bool).
      3. FAISS cosine search to get full distance matrix D (n x n).
      4. Build flat arrays of upper-triangle (i, j) pairs + sims.
      5. Sort pairs by sim descending (single np.argsort).
      6. Cumulative TP/FP via boolean array of co-membership for sorted pairs.
    """
    emb_id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    # KEGG gene index in embeddings + ordered pathway-set list
    kept_emb_indices: list[int] = []
    pathway_sets: list[set] = []
    for kegg_gene, pathways in gene_to_pathways.items():
        emb_id = id_mapping.get(kegg_gene)
        if emb_id and emb_id in emb_id_to_idx:
            kept_emb_indices.append(emb_id_to_idx[emb_id])
            pathway_sets.append(pathways)

    if len(kept_emb_indices) < 10:
        return {}

    # If duplicates (rare), keep first occurrence
    seen = {}
    uniq_emb: list[int] = []
    uniq_paths: list[set] = []
    for ei, ps in zip(kept_emb_indices, pathway_sets):
        if ei in seen:
            continue
        seen[ei] = True
        uniq_emb.append(ei)
        uniq_paths.append(ps)
    kegg_indices = np.asarray(uniq_emb, dtype=np.int64)
    pathway_sets = uniq_paths
    n_kegg = len(kegg_indices)

    # Build pathway membership matrix (n_kegg, n_pathways) as bool
    all_pathways = sorted(set().union(*pathway_sets))
    pathway_to_col = {p: i for i, p in enumerate(all_pathways)}
    n_pw = len(all_pathways)
    membership = np.zeros((n_kegg, n_pw), dtype=bool)
    for i, ps in enumerate(pathway_sets):
        for p in ps:
            membership[i, pathway_to_col[p]] = True

    # FAISS cosine: normalize then inner product
    kegg_emb = embeddings[kegg_indices].astype(np.float32)
    norms = np.linalg.norm(kegg_emb, axis=1, keepdims=True)
    kegg_emb = kegg_emb / np.maximum(norms, 1e-9)
    d = kegg_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(kegg_emb)
    D, I = index.search(kegg_emb, n_kegg)
    # D[i, k] = similarity of i to its k-th neighbor I[i, k] (incl self)

    # Build upper-triangle pair arrays (i < j) using FAISS output for ordering.
    # Each unordered pair appears twice in the FAISS matrix; we deduplicate.
    i_repeat = np.repeat(np.arange(n_kegg, dtype=np.int64), n_kegg)
    j_flat = I.ravel().astype(np.int64)
    sims_flat = D.ravel().astype(np.float32)
    mask = j_flat > i_repeat
    pi = i_repeat[mask]
    pj = j_flat[mask]
    sims = sims_flat[mask]

    # Sort by similarity descending
    order = np.argsort(-sims, kind="stable")
    pi = pi[order]
    pj = pj[order]

    # Co-membership computed via matmul on uint8 membership to avoid the
    # 30M x 656 bool tensor explosion. comem_matrix[i, j] = number of shared
    # pathways for genes (i, j); we only need >0.
    membership_u8 = membership.astype(np.uint8)
    comem_matrix = membership_u8 @ membership_u8.T  # (n_kegg, n_kegg) int
    shared_pair = comem_matrix[pi, pj] > 0
    shared = shared_pair

    # Counts
    is_tp = shared
    is_fp = ~shared
    tp_cum = np.cumsum(is_tp.astype(np.int64))
    fp_cum = np.cumsum(is_fp.astype(np.int64))

    max_tp = int(tp_cum[-1])
    max_fp = int(fp_cum[-1])
    use_max_fp = max(int(max_fp * percent_threshold), 1)

    # Partial AUC up to use_max_fp
    # Find first index where fp_cum exceeds cap
    cap_idx = int(np.searchsorted(fp_cum, use_max_fp, side="right"))
    cap_idx = min(cap_idx, len(fp_cum))
    tp_cap = tp_cum[:cap_idx]
    fp_cap = fp_cum[:cap_idx]
    tpr_cap = np.concatenate([[0.0], tp_cap / max_tp])
    fpr_cap = np.concatenate([[0.0], fp_cap / max_fp])
    pauc = float(auc(fpr_cap, tpr_cap)) if len(fpr_cap) > 1 else 0.0

    tpr_full = np.concatenate([[0.0], tp_cum / max_tp])
    fpr_full = np.concatenate([[0.0], fp_cum / max_fp])
    full_auc = float(auc(fpr_full, tpr_full))

    # Subsample for plot (log+lin mix)
    n = len(tp_cum)
    if n > max_curve_points:
        log_idx = np.unique(np.geomspace(1, n, num=max_curve_points // 2,
                                         dtype=np.int64) - 1)
        lin_idx = np.linspace(0, n - 1, num=max_curve_points // 2,
                              dtype=np.int64)
        keep = np.unique(np.concatenate([log_idx, lin_idx]))
        keep = keep[(keep >= 0) & (keep < n)]
    else:
        keep = np.arange(n)

    return {
        "partial_auc_0.1pct": round(pauc, 6),
        "full_auc": round(full_auc, 6),
        "n_kegg_genes": int(n_kegg),
        "n_pathways": int(n_pw),
        "max_tp": max_tp,
        "max_fp": max_fp,
        "fp_cap": use_max_fp,
        "tp_curve": tp_cum[keep].tolist(),
        "fp_curve": fp_cum[keep].tolist(),
    }


def main() -> None:
    cache_dir = PROJECT_ROOT / "data" / "annotations" / "kegg"
    json_cache = OUT_DIR / "figS2_kegg_roc_curves.json"
    # Fast path: if cached curves JSON exists with all expected species/methods,
    # skip embedding loads + FAISS and re-plot directly. Compute is unchanged.
    if json_cache.exists() and "--recompute" not in sys.argv:
        try:
            with open(json_cache) as f:
                cached = json.load(f)
            have_all = all(
                sp in cached and all(m in cached[sp] for m in METHODS)
                for sp in ALL_SPECIES
            )
        except Exception:
            have_all = False
        if have_all:
            logger.info(f"Using cached curves from {json_cache} (no recompute)")
            all_results = cached
            _plot_and_save(all_results)
            return

    all_results: dict = {}

    for species in ALL_SPECIES:
        kegg_code = bk.SPECIES_KEGG_CODE.get(species)
        if not kegg_code:
            logger.warning(f"No KEGG code for {species}, skipping")
            continue

        t0 = time.time()
        logger.info(f"=== {species} ({kegg_code}) ===")
        kegg_file = bk.download_kegg_pathways(kegg_code, cache_dir)
        gene_to_pathways = bk.load_kegg_annotations(kegg_file, kegg_code)

        species_results: dict = {}
        for method in METHODS:
            emb_path = PROJECT_ROOT / bk.METHODS[method] / f"{species}.h5"
            if not emb_path.exists():
                logger.warning(f"  {method}: missing {emb_path}")
                continue
            try:
                embeddings, protein_ids = bk.load_embeddings(method, species)
            except Exception as e:
                logger.error(f"  {method}: load failed: {e}")
                continue
            id_mapping = bk.build_id_mapping(species, set(protein_ids))
            tm = time.time()
            res = compute_kegg_roc_with_curves(
                embeddings, protein_ids, gene_to_pathways, id_mapping
            )
            dt = time.time() - tm
            if res:
                species_results[method] = res
                logger.info(
                    f"  {method}: pAUC={res['partial_auc_0.1pct']:.5f}  "
                    f"fullAUC={res['full_auc']:.5f}  n={res['n_kegg_genes']}  "
                    f"({dt:.1f}s)"
                )
        all_results[species] = species_results
        logger.info(f"  -> total {time.time() - t0:.1f}s")

    json_out = OUT_DIR / "figS2_kegg_roc_curves.json"
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved curves -> {json_out}")

    _plot_and_save(all_results)


def _plot_and_save(all_results: dict) -> None:
    # ----- Plot: 2x3 grid (6 species) -----
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4),
                             sharex=False, sharey=False)
    axes_list = axes.flatten()

    for ax, species in zip(axes_list, ALL_SPECIES):
        sp_res = all_results.get(species, {})
        if not sp_res:
            ax.set_visible(False)
            continue

        for method, (label, color) in METHODS.items():
            r = sp_res.get(method)
            if not r:
                continue
            fp = np.asarray(r["fp_curve"], dtype=float)
            tp = np.asarray(r["tp_curve"], dtype=float)
            # Normalize to FPR / TPR so all panels share a [0,1] y-axis
            # (cross-species visual comparability). Legend AUCs are the
            # un-normalized full-AUC values from the curves dict.
            max_tp = float(r.get("max_tp", tp[-1] if len(tp) else 1.0)) or 1.0
            max_fp = float(r.get("max_fp", fp[-1] if len(fp) else 1.0)) or 1.0
            tpr = tp / max_tp
            fpr = fp / max_fp
            ax.plot(
                fpr, tpr,
                color=color, linewidth=1.6,
                label=f"{label} (AUC={r['full_auc']:.3f})",
            )

        is_seed = species in SEED_SPECIES_WITH_KEGG
        title = species + (" *" if is_seed else "")
        ax.set_title(title)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlim(left=1e-5, right=1.0)
        ax.set_ylim(0.0, 1.0)

    fig.suptitle(
        "Figure S2 — KEGG pathway ROC curves (TPR vs FPR, log x). "
        "Y-axis normalized to TP / max_TP per species; legend AUCs are "
        "un-normalized full-AUC values. Asterisk (*) marks seed species.",
        fontsize=11.5, y=1.0,
    )
    fig.tight_layout()

    out_png = OUT_DIR / "figS2_kegg_roc_per_seed.png"
    out_pdf = OUT_DIR / "figS2_kegg_roc_per_seed.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
