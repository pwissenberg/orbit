#!/usr/bin/env python3
"""Fair ablation for Figure S5: basic Procrustes vs Procrustes+Jaccard+iter+CSLS,
both run on the SAME (current/optimized) Node2Vec embeddings.

The original figS5 was confounded: basic Procrustes had been run on the OLD N2V
embeddings (pre-grid-search defaults) while the headline Jaccard+iter+CSLS
variant was run on the NEW optimized N2V (p=1.0, q=0.7, walks=20, length=50,
epochs=10). This script re-runs basic Procrustes on the NEW N2V embeddings so
the ablation is a clean, fair comparison.

What "basic Procrustes" means here:
  - vanilla weighting (uniform 1.0 per ortholog pair; the third column of the
    pair files is ignored, same as run_improved_procrustes.py --weighting vanilla)
  - NO iterative refinement
  - NO CSLS during alignment
  - plain cosine similarity in the evaluation step (CSLS is never used inside
    evaluate_pair anyway, so this is automatic)
  - same OrthoFinder ortholog pair files in data/orthologs/{seeds,non_seeds}/
  - same 5 seed species and same nearest-seed groups

Outputs:
  - results/aligned_embeddings_basic_procrustes_new_n2v/{species}.h5
  - results/evaluation_basic_procrustes_new_n2v.json (cross-species)
  - results/evaluation_basic_procrustes_new_n2v_within.json (within-species)
  - results/supplementary/figS5_jaccard_csls_ablation.{png,pdf,csv}  (replaces old)
  - results/supplementary/figS5_jaccard_csls_ablation_OLD_confounded.{png,pdf,csv}
    are saved as backups before overwriting.

Hardware: parallelizes Stage 2 with ProcessPoolExecutor (96 cores available),
preloads all 153 N2V embeddings into RAM (~2.3 GB).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

ROOT = Path.home() / "PLANT-SPACE"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Import existing utilities — does NOT modify any source files
import run_improved_procrustes as rip  # noqa: E402

NODE2VEC_DIR = ROOT / "data" / "node2vec"
ORTHOLOGS_DIR = ROOT / "data" / "orthologs"
NETWORK_DIR = ROOT / "data" / "networks"
TRANSCRIPTS_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"

OUTPUT_ALIGNED = ROOT / "results" / "aligned_embeddings_basic_procrustes_new_n2v"
OUTPUT_CROSS = ROOT / "results" / "evaluation_basic_procrustes_new_n2v.json"
OUTPUT_WITHIN = ROOT / "results" / "evaluation_basic_procrustes_new_n2v_within.json"
SUPP_DIR = ROOT / "results" / "supplementary"


def run_basic_alignment(workers: int) -> float:
    """Stage 1 + Stage 2 basic Procrustes alignment using NEW N2V embeddings."""
    OUTPUT_ALIGNED.mkdir(parents=True, exist_ok=True)
    print(f"[align] preloading 153 N2V embeddings into RAM...", flush=True)
    t0 = time.perf_counter()
    preloaded = rip._preload_all_embeddings()
    print(f"[align] preload took {time.perf_counter()-t0:.1f}s", flush=True)

    with open(SEED_RESULTS) as fh:
        seed_info = json.load(fh)
    seeds = seed_info["seeds"]
    groups = seed_info["groups"]
    REFERENCE = "ARATH"
    non_ref_seeds = [s for s in seeds if s != REFERENCE]

    # ---------- Stage 1: seeds -> ARATH ----------
    print(f"[align] Stage 1: align {non_ref_seeds} -> {REFERENCE}", flush=True)
    t1 = time.perf_counter()

    ref_emb, ref_proteins = preloaded[REFERENCE]
    rip._save_aligned(REFERENCE, ref_emb, ref_proteins, OUTPUT_ALIGNED)

    aligned_seeds = {REFERENCE: (ref_emb, ref_proteins)}
    for seed in non_ref_seeds:
        emb_seed, prot_seed = preloaded[seed]
        aligned_emb, src_proteins, n_pairs = rip._align_one_species(
            seed, REFERENCE, ref_emb, ref_proteins, ORTHOLOGS_DIR,
            weighting="vanilla",
            iterative=False,
            use_csls=False,
            n_iters=0,
            csls_k=0,
            csls_threshold=0.0,
            use_gpu=False,
            gpu_id=None,
            emb_source=emb_seed,
            proteins_source=prot_seed,
            translate=False,
            scale=False,
        )
        rip._save_aligned(seed, aligned_emb, src_proteins, OUTPUT_ALIGNED)
        aligned_seeds[seed] = (aligned_emb, src_proteins)
        print(f"[align]   {seed} -> {REFERENCE}: {n_pairs} pairs", flush=True)

    print(f"[align] Stage 1 done in {time.perf_counter()-t1:.1f}s", flush=True)

    # ---------- Stage 2: non-seeds -> nearest seed (parallel) ----------
    # Update preloaded with ALIGNED seed embeddings (so workers see the right ref)
    for seed_name, (aligned_emb, aligned_prots) in aligned_seeds.items():
        preloaded[seed_name] = (aligned_emb, aligned_prots)

    non_seeds = sorted(groups.keys())
    work_items = []
    for ns in non_seeds:
        nearest_seed = groups[ns]
        # (sp_source, sp_ref, orthologs_dir, output_dir,
        #  weighting, iterative, use_csls, n_iters, csls_k, csls_threshold,
        #  use_gpu, gpu_id, translate, scale)
        work_items.append((
            ns, nearest_seed, str(ORTHOLOGS_DIR), str(OUTPUT_ALIGNED),
            "vanilla", False, False, 0, 0, 0.0,
            False, None, False, False,
        ))

    eff_workers = min(workers, len(work_items))
    print(f"[align] Stage 2: {len(work_items)} non-seeds with {eff_workers} workers", flush=True)
    t2 = time.perf_counter()

    if sys.platform == "linux":
        ctx = multiprocessing.get_context("fork")
    else:
        ctx = multiprocessing.get_context("spawn")

    # Set globals BEFORE fork so workers inherit via COW
    rip._PRELOADED_EMBEDDINGS = preloaded

    failed = []
    success = 0
    with ProcessPoolExecutor(
        max_workers=eff_workers,
        initializer=rip._init_worker,
        initargs=(preloaded, None),
        mp_context=ctx,
    ) as ex:
        futures = {ex.submit(rip._align_worker, item): item for item in work_items}
        for fut in as_completed(futures):
            sp_src, sp_ref, n_pairs, err = fut.result()
            if err:
                print(f"[align]   FAIL {sp_src} -> {sp_ref}: {err}", flush=True)
                failed.append(sp_src)
            else:
                success += 1
                if success % 20 == 0 or success == len(work_items):
                    print(f"[align]   [{success}/{len(work_items)}] aligned "
                          f"({time.perf_counter()-t2:.1f}s)", flush=True)

    rip._PRELOADED_EMBEDDINGS = None
    print(f"[align] Stage 2 done in {time.perf_counter()-t2:.1f}s — "
          f"{success} ok, {len(failed)} failed", flush=True)
    if failed:
        print(f"[align] FAILED: {failed}", flush=True)

    return time.perf_counter() - t0


def run_evaluation(workers: int) -> tuple[float, float]:
    """Cross-species + within-species evaluation."""
    from orbit.evaluate import evaluate_all_pairs, evaluate_all_within_species

    print(f"[eval] cross-species evaluation, {workers} workers", flush=True)
    t0 = time.perf_counter()
    results = evaluate_all_pairs(
        aligned_dir=OUTPUT_ALIGNED,
        network_dir=NETWORK_DIR,
        transcripts_dir=TRANSCRIPTS_DIR,
        seed_results_path=SEED_RESULTS,
        k=50,
        top_m=10,
        eval_mode="all",
        n_workers=workers,
    )
    with open(OUTPUT_CROSS, "w") as fh:
        json.dump(results, fh, indent=2)
    valid = [r for r in results if "error" not in r]
    avg_rho = float(np.mean([r["spearman_rho"] for r in valid])) if valid else 0.0
    avg_hits = float(np.mean([r["hits_at_k"] for r in valid])) if valid else 0.0
    print(f"[eval] cross done in {time.perf_counter()-t0:.1f}s — "
          f"{len(valid)} pairs, Spearman={avg_rho:.4f}, Hits@50={avg_hits:.4f}", flush=True)
    print(f"[eval] saved -> {OUTPUT_CROSS}", flush=True)
    cross_secs = time.perf_counter() - t0

    print(f"[eval] within-species evaluation", flush=True)
    t1 = time.perf_counter()
    within = evaluate_all_within_species(
        aligned_dir=OUTPUT_ALIGNED,
        network_dir=NETWORK_DIR,
        k=50,
        n_workers=workers,
    )
    with open(OUTPUT_WITHIN, "w") as fh:
        json.dump(within, fh, indent=2)
    valid_w = [r for r in within if "error" not in r]
    avg_p = float(np.mean([r["precision_at_k"] for r in valid_w])) if valid_w else 0.0
    avg_r = float(np.mean([r["recall_at_k"] for r in valid_w])) if valid_w else 0.0
    print(f"[eval] within done in {time.perf_counter()-t1:.1f}s — "
          f"{len(valid_w)} species, P@50={avg_p:.4f}, R@50={avg_r:.4f}", flush=True)
    print(f"[eval] saved -> {OUTPUT_WITHIN}", flush=True)
    within_secs = time.perf_counter() - t1
    return cross_secs, within_secs


def regenerate_figure() -> None:
    """Regenerate figS5 with the FAIR ablation data."""
    import matplotlib.pyplot as plt
    import pandas as pd

    cross_full = json.load(open(ROOT / "results/evaluation_proc_jaccard_iter_csls.json"))
    cross_base = json.load(open(OUTPUT_CROSS))
    within_full = json.load(open(ROOT / "results/evaluation_proc_jaccard_iter_csls_within.json"))
    within_base = json.load(open(OUTPUT_WITHIN))

    cross_metrics = ["hits_at_k", "mrr_at_k", "spearman_rho", "top_m_hits_at_k"]
    cross_labels = ["Hits@50", "MRR@50", "Spearman ρ", "Top-M Hits@50"]
    within_metrics = ["precision_at_k", "recall_at_k"]
    within_labels = ["Precision@50 (within)", "Recall@50 (within)"]

    def mean_metric(rows, key):
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    rows = []
    for key, lbl in zip(cross_metrics, cross_labels):
        m_b, s_b, n_b = mean_metric(cross_base, key)
        m_f, s_f, n_f = mean_metric(cross_full, key)
        rows.append({
            "category": "cross-species (n=158 pairs)",
            "metric": lbl,
            "basic_mean": m_b, "basic_std": s_b, "basic_n": n_b,
            "full_mean": m_f, "full_std": s_f, "full_n": n_f,
            "improvement_pct": 100.0 * (m_f - m_b) / m_b if m_b != 0 else float("nan"),
        })
    for key, lbl in zip(within_metrics, within_labels):
        m_b, s_b, n_b = mean_metric(within_base, key)
        m_f, s_f, n_f = mean_metric(within_full, key)
        rows.append({
            "category": "within-species (n=153)",
            "metric": lbl,
            "basic_mean": m_b, "basic_std": s_b, "basic_n": n_b,
            "full_mean": m_f, "full_std": s_f, "full_n": n_f,
            "improvement_pct": 100.0 * (m_f - m_b) / m_b if m_b != 0 else float("nan"),
        })

    df = pd.DataFrame(rows)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Backup the OLD (confounded) figure files ----------
    for ext in ("csv", "png", "pdf"):
        old = SUPP_DIR / f"figS5_jaccard_csls_ablation.{ext}"
        if old.exists():
            backup = SUPP_DIR / f"figS5_jaccard_csls_ablation_OLD_confounded.{ext}"
            shutil.copy2(old, backup)
            print(f"[fig] backed up old: {old.name} -> {backup.name}", flush=True)

    csv_path = SUPP_DIR / "figS5_jaccard_csls_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"[fig] CSV: {csv_path}", flush=True)
    print(df.to_string(index=False), flush=True)

    # ---------- Decide whether to drop the within-species panel ----------
    # If basic and full give within-species P@50 within 1e-3 of each other, the
    # rotation is empirically isometric → no story for within-species; show as a
    # parity annotation, not as bars.
    within_p_basic = mean_metric(within_base, "precision_at_k")[0]
    within_p_full  = mean_metric(within_full, "precision_at_k")[0]
    within_r_basic = mean_metric(within_base, "recall_at_k")[0]
    within_r_full  = mean_metric(within_full, "recall_at_k")[0]
    parity = (abs(within_p_basic - within_p_full) < 1e-3 and
              abs(within_r_basic - within_r_full) < 1e-3)
    print(f"[fig] within parity: P@50 basic={within_p_basic:.4f} full={within_p_full:.4f}  "
          f"R@50 basic={within_r_basic:.4f} full={within_r_full:.4f}  parity={parity}",
          flush=True)

    # ---------- Plot ----------
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]

    if parity:
        # Cross-species only: the four cross metrics. Within-species shown as a
        # textual annotation noting parity (the orthogonal-isometry test passed).
        labels = cross_labels
        sub = df[df["category"].str.startswith("cross")]
        basic_means = sub["basic_mean"].to_numpy()
        basic_stds  = sub["basic_std"].to_numpy()
        full_means  = sub["full_mean"].to_numpy()
        full_stds   = sub["full_std"].to_numpy()
        improvements = sub["improvement_pct"].to_numpy()

        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
        x = np.arange(len(labels))
        w = 0.38
        b1 = ax.bar(x - w/2, basic_means, w, yerr=basic_stds,
                    label="Basic Procrustes", color="#bcbcbc", edgecolor="black",
                    linewidth=0.4, capsize=3, error_kw={"linewidth": 0.7})
        b2 = ax.bar(x + w/2, full_means, w, yerr=full_stds,
                    label="Procrustes + Jaccard + iterative + CSLS",
                    color="#e07b54", edgecolor="black", linewidth=0.4, capsize=3,
                    error_kw={"linewidth": 0.7})

        for bars, means in [(b1, basic_means), (b2, full_means)]:
            for b, m in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)

        # Improvement % annotations between bar pairs
        for i, (mb, mf, imp) in enumerate(zip(basic_means, full_means, improvements)):
            top = max(mb, mf) + max(basic_stds[i], full_stds[i]) + 0.04
            ax.text(i, top, f"+{imp:.1f}%", ha="center", va="bottom",
                    fontsize=8, color="#666666", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ymax = max(full_means.max() + full_stds.max(), 0.5) * 1.18
        ax.set_ylim(top=ymax)
        ax.legend(loc="upper right", frameon=False, fontsize=9)

        ax.text(
            0.0, -0.30,
            f"Within-species P@50 = {within_p_basic:.4f} for both methods "
            f"(R@50 = {within_r_basic:.4f}); orthogonal Procrustes is isometric, "
            f"so within-species structure is identical to the input N2V embeddings.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#444444", wrap=True,
        )

        fig.suptitle(
            "Figure S5  Fair ablation: Jaccard + iterative + CSLS over basic Procrustes\n"
            "(both arms use the same optimized Node2Vec embeddings)",
            y=1.02, fontsize=10,
        )
    else:
        # Original 4+2 layout but with FAIR basic baseline.
        labels = df["metric"].tolist()
        basic_means = df["basic_mean"].to_numpy()
        basic_stds  = df["basic_std"].to_numpy()
        full_means  = df["full_mean"].to_numpy()
        full_stds   = df["full_std"].to_numpy()

        fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
        x = np.arange(len(labels))
        w = 0.38
        b1 = ax.bar(x - w/2, basic_means, w, yerr=basic_stds,
                    label="Basic Procrustes", color="#bcbcbc", edgecolor="black",
                    linewidth=0.4, capsize=3, error_kw={"linewidth": 0.7})
        b2 = ax.bar(x + w/2, full_means, w, yerr=full_stds,
                    label="Procrustes + Jaccard + iterative + CSLS",
                    color="#e07b54", edgecolor="black", linewidth=0.4, capsize=3,
                    error_kw={"linewidth": 0.7})

        for bars, means in [(b1, basic_means), (b2, full_means)]:
            for b, m in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(len(cross_metrics) - 0.5, color="black", linestyle=":",
                   linewidth=0.8, alpha=0.5)
        ymax = max(full_means.max() + full_stds.max(), 0.7) * 1.05
        ax.set_ylim(top=ymax)
        ax.text(len(cross_metrics) / 2 - 0.5, ymax * 0.97,
                "Cross-species (n=158 pairs)", ha="center", fontsize=9, alpha=0.7)
        ax.text(len(cross_metrics) + len(within_metrics) / 2 - 0.5, ymax * 0.97,
                "Within-species (n=153)", ha="center", fontsize=9, alpha=0.7)
        ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=9)
        fig.suptitle(
            "Figure S5  Fair ablation: Jaccard + iterative + CSLS over basic Procrustes "
            "(both arms use the same optimized Node2Vec embeddings)",
            y=1.0, fontsize=11,
        )

    fig.tight_layout()
    png_path = SUPP_DIR / "figS5_jaccard_csls_ablation.png"
    pdf_path = SUPP_DIR / "figS5_jaccard_csls_ablation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[fig] PNG: {png_path}", flush=True)
    print(f"[fig] PDF: {pdf_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--skip-align", action="store_true",
                    help="Skip alignment if aligned_dir already populated")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip evaluation (use existing JSONs)")
    ap.add_argument("--only-figure", action="store_true",
                    help="Only regenerate the figure (assumes JSONs exist)")
    args = ap.parse_args()

    print(f"[main] cwd={os.getcwd()}, workers={args.workers}", flush=True)
    print(f"[main] OUT aligned: {OUTPUT_ALIGNED}", flush=True)
    print(f"[main] OUT cross:   {OUTPUT_CROSS}", flush=True)
    print(f"[main] OUT within:  {OUTPUT_WITHIN}", flush=True)
    pipeline_start = time.perf_counter()

    if not args.only_figure and not args.skip_align:
        run_basic_alignment(args.workers)

    if not args.only_figure and not args.skip_eval:
        run_evaluation(args.workers)

    regenerate_figure()

    total = time.perf_counter() - pipeline_start
    print(f"[main] TOTAL: {total:.1f}s ({total/60:.2f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
