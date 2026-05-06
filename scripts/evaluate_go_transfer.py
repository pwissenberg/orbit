#!/usr/bin/env python3
"""Cross-species GO transfer experiment using CAFA benchmark data.

Train GO classifiers on ARATH, predict on ORYSA/ZEAMA/GLYMA/MEDTR.
Compare: vanilla, procrustes, prott5, vanilla+prott5, procrustes+prott5.

Usage:
    python3 scripts/evaluate_go_transfer.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed  # used for k-NN if needed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CAFA_PATH = Path.home() / "2026_02" / "go_labels_cafa_expanded.tsv"
IDMAP_DIR = ROOT / "data" / "annotations" / "id_mapping"
OBO_PATH = Path("/tmp/go-basic.obo")
OUTPUT_PATH = ROOT / "results" / "downstream" / "go_transfer_cafa.json"
LOG_PATH = ROOT / "results" / "downstream" / "go_transfer_progress.log"

TRAIN_SPECIES = "ARATH"
TEST_SPECIES = ["ORYSA", "ZEAMA", "GLYMA", "MEDTR"]
ASPECTS = ["MF", "BP", "CC"]
MIN_TRAIN_POSITIVES = 10
KNN_K = 50

EMB_DIRS = {
    "vanilla": ROOT / "results" / "aligned_embeddings",
    "procrustes": ROOT / "results" / "aligned_embeddings_proc_jaccard_iter_csls",
}
PROTT5_DIR = ROOT / "data" / "prott5"

# Methods: base embeddings + concatenations
BASE_METHODS = ["vanilla", "procrustes", "prott5"]
CONCAT_METHODS = ["vanilla_prott5", "procrustes_prott5"]
ALL_METHODS = BASE_METHODS + CONCAT_METHODS


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ── Data Loading ──────────────────────────────────────────────────────────


def download_obo() -> None:
    if OBO_PATH.exists() and OBO_PATH.stat().st_size > 100_000:
        log(f"OBO already cached at {OBO_PATH}")
        return
    log("Downloading go-basic.obo...")
    urllib.request.urlretrieve(
        "http://purl.obolibrary.org/obo/go/go-basic.obo", str(OBO_PATH)
    )
    log(f"Downloaded ({OBO_PATH.stat().st_size / 1e6:.1f} MB)")


def parse_go_aspects() -> dict[str, str]:
    """Parse GO ID -> aspect (MF/BP/CC) from OBO file."""
    ns_map = {
        "biological_process": "BP",
        "molecular_function": "MF",
        "cellular_component": "CC",
    }
    result = {}
    current_id = None
    with open(OBO_PATH) as f:
        for line in f:
            if line.startswith("id: GO:"):
                current_id = line.strip().split(": ", 1)[1]
            elif line.startswith("namespace:") and current_id:
                ns = line.strip().split(": ", 1)[1]
                if ns in ns_map:
                    result[current_id] = ns_map[ns]
                current_id = None
    return result


def load_cafa(go_aspects: dict[str, str]) -> pd.DataFrame:
    """Load CAFA expanded annotations with aspect mapping."""
    df = pd.read_csv(CAFA_PATH, sep="\t", usecols=["accession", "go_id"])
    df["aspect"] = df["go_id"].map(go_aspects)
    df = df.dropna(subset=["aspect"])
    log(f"CAFA: {len(df):,} annotations, {df.accession.nunique():,} proteins, "
        f"{df.go_id.nunique():,} GO terms")
    return df


def load_id_mapping(species: str) -> dict[str, str]:
    """Load uniprot_accession -> teagcn_id mapping."""
    path = IDMAP_DIR / f"{species}_to_uniprot.tsv"
    df = pd.read_csv(path, sep="\t", usecols=["teagcn_id", "uniprot_accession"])
    # Keep first mapping per accession
    return dict(zip(df["uniprot_accession"], df["teagcn_id"]))


def load_h5(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load embeddings from H5 file."""
    with h5py.File(path, "r") as f:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]
        embeddings = f["embeddings"][:]
    return embeddings, proteins


def build_species_data(
    species: str, cafa_df: pd.DataFrame, method: str
) -> dict | None:
    """Build (X, go_labels_per_aspect) for a species+method combination."""
    # ID mapping
    idmap = load_id_mapping(species)
    cafa_species = cafa_df[cafa_df["accession"].isin(idmap)]
    if len(cafa_species) == 0:
        return None

    # Map accessions to teagcn_ids
    cafa_species = cafa_species.copy()
    cafa_species["teagcn_id"] = cafa_species["accession"].map(idmap)

    # Load embeddings
    if method in ("vanilla", "procrustes"):
        emb_path = EMB_DIRS[method] / f"{species}.h5"
        X, prot_ids = load_h5(emb_path)
    elif method == "prott5":
        emb_path = PROTT5_DIR / f"{species}.h5"
        X, prot_ids = load_h5(emb_path)
    elif method == "vanilla_prott5":
        X_v, prot_v = load_h5(EMB_DIRS["vanilla"] / f"{species}.h5")
        X_t, prot_t = load_h5(PROTT5_DIR / f"{species}.h5")
        # Intersect proteins
        set_t = set(prot_t)
        idx_v = [i for i, p in enumerate(prot_v) if p in set_t]
        t_lookup = {p: i for i, p in enumerate(prot_t)}
        idx_t = [t_lookup[prot_v[i]] for i in idx_v]
        prot_ids = [prot_v[i] for i in idx_v]
        X = np.concatenate([X_v[idx_v], X_t[idx_t]], axis=1)
    elif method == "procrustes_prott5":
        X_p, prot_p = load_h5(EMB_DIRS["procrustes"] / f"{species}.h5")
        X_t, prot_t = load_h5(PROTT5_DIR / f"{species}.h5")
        set_t = set(prot_t)
        idx_p = [i for i, p in enumerate(prot_p) if p in set_t]
        t_lookup = {p: i for i, p in enumerate(prot_t)}
        idx_t = [t_lookup[prot_p[i]] for i in idx_p]
        prot_ids = [prot_p[i] for i in idx_p]
        X = np.concatenate([X_p[idx_p], X_t[idx_t]], axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build protein -> embedding index
    prot_to_idx = {p: i for i, p in enumerate(prot_ids)}

    # Filter to proteins with both CAFA labels and embeddings
    valid_tids = set(cafa_species["teagcn_id"].unique()) & set(prot_to_idx.keys())
    if not valid_tids:
        return None

    # Build label sets per aspect
    labels_by_aspect: dict[str, dict[str, set[str]]] = {a: {} for a in ASPECTS}
    for _, row in cafa_species[cafa_species["teagcn_id"].isin(valid_tids)].iterrows():
        tid = row["teagcn_id"]
        labels_by_aspect[row["aspect"]].setdefault(tid, set()).add(row["go_id"])

    # Build embedding matrix for valid proteins (ordered)
    valid_list = sorted(valid_tids)
    indices = [prot_to_idx[tid] for tid in valid_list]
    X_valid = X[indices]

    return {
        "X": X_valid,
        "proteins": valid_list,
        "labels_by_aspect": labels_by_aspect,
        "n_proteins": len(valid_list),
        "dim": X_valid.shape[1],
    }


# ── Evaluation Functions ─────────────────────────────────────────────────


def compute_fmax(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Compute Fmax and AUPRC for a single GO term."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0, 0.0
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    denom = precisions + recalls
    f_scores = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)
    fmax = float(f_scores.max())
    auprc = float(average_precision_score(y_true, y_scores))
    return fmax, auprc


def evaluate_logreg_aspect(
    method: str,
    test_species: str,
    aspect: str,
    train_data: dict | None,
    test_data: dict | None,
) -> dict:
    """Evaluate LogReg transfer for one (method, species, aspect)."""
    t0 = time.time()

    if train_data is None or test_data is None:
        return {
            "method": method, "classifier": "logreg", "test_species": test_species,
            "aspect": aspect, "n_terms": 0, "n_train": 0, "n_test": 0,
            "fmax": 0.0, "auprc": 0.0, "error": "no data",
        }

    # Get all GO terms for this aspect
    train_labels = train_data["labels_by_aspect"][aspect]
    test_labels = test_data["labels_by_aspect"][aspect]

    # Collect all GO terms seen in training data for this aspect
    all_go_terms = set()
    for p in train_data["proteins"]:
        all_go_terms |= train_labels.get(p, set())

    # Filter: ≥MIN_TRAIN_POSITIVES in train, ≥1 in test
    evaluable_terms = []
    for term in all_go_terms:
        n_train_pos = sum(1 for p in train_data["proteins"] if term in train_labels.get(p, set()))
        n_test_pos = sum(1 for p in test_data["proteins"] if term in test_labels.get(p, set()))
        if n_train_pos >= MIN_TRAIN_POSITIVES and n_test_pos >= 1:
            evaluable_terms.append(term)

    if not evaluable_terms:
        return {
            "method": method, "classifier": "logreg", "test_species": test_species,
            "aspect": aspect, "n_terms": 0, "n_train": train_data["n_proteins"],
            "n_test": test_data["n_proteins"], "fmax": 0.0, "auprc": 0.0,
        }

    # Build label matrices
    X_train = train_data["X"]
    X_test = test_data["X"]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    fmax_list, auprc_list = [], []
    for term in evaluable_terms:
        y_train = np.array([1 if term in train_labels.get(p, set()) else 0
                           for p in train_data["proteins"]])
        y_test = np.array([1 if term in test_labels.get(p, set()) else 0
                          for p in test_data["proteins"]])

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1)
        clf.fit(X_train_s, y_train)
        y_scores = clf.predict_proba(X_test_s)[:, 1]

        fmax, auprc = compute_fmax(y_test, y_scores)
        fmax_list.append(fmax)
        auprc_list.append(auprc)

    elapsed = time.time() - t0
    return {
        "method": method,
        "classifier": "logreg",
        "test_species": test_species,
        "aspect": aspect,
        "n_terms": len(evaluable_terms),
        "n_train": train_data["n_proteins"],
        "n_test": test_data["n_proteins"],
        "fmax": float(np.mean(fmax_list)),
        "fmax_std": float(np.std(fmax_list)),
        "auprc": float(np.mean(auprc_list)),
        "auprc_std": float(np.std(auprc_list)),
        "elapsed_s": round(elapsed, 1),
    }


def load_cafa_silent(go_aspects: dict[str, str]) -> pd.DataFrame:
    """Load CAFA without logging (for worker processes)."""
    df = pd.read_csv(CAFA_PATH, sep="\t", usecols=["accession", "go_id"])
    df["aspect"] = df["go_id"].map(go_aspects)
    return df.dropna(subset=["aspect"])


# ── k-NN Transfer ────────────────────────────────────────────────────────


def run_knn_transfer(
    method: str,
    test_species: str,
    cafa_df: pd.DataFrame,
) -> list[dict]:
    """k-NN transfer for all aspects using FAISS."""
    train_data = build_species_data(TRAIN_SPECIES, cafa_df, method)
    test_data = build_species_data(test_species, cafa_df, method)

    if train_data is None or test_data is None:
        return []

    X_train = train_data["X"].astype(np.float32)
    X_test = test_data["X"].astype(np.float32)

    # L2 normalize
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-12)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-12)

    # Build FAISS index
    try:
        import faiss
        index = faiss.IndexFlatIP(X_train.shape[1])
        try:
            index = faiss.index_cpu_to_all_gpus(index)
            log(f"  FAISS-GPU: {method}/{test_species}")
        except Exception:
            log(f"  FAISS-CPU fallback: {method}/{test_species}")
        index.add(X_train)
        _, I = index.search(X_test, KNN_K)
    except ImportError:
        # Pure numpy fallback
        log(f"  numpy kNN fallback: {method}/{test_species}")
        sims = X_test @ X_train.T
        I = np.argsort(-sims, axis=1)[:, :KNN_K]

    results = []
    for aspect in ASPECTS:
        train_labels = train_data["labels_by_aspect"][aspect]
        test_labels = test_data["labels_by_aspect"][aspect]

        all_go_terms = set()
        for p in train_data["proteins"]:
            all_go_terms |= train_labels.get(p, set())

        evaluable_terms = []
        for term in all_go_terms:
            n_train = sum(1 for p in train_data["proteins"] if term in train_labels.get(p, set()))
            n_test = sum(1 for p in test_data["proteins"] if term in test_labels.get(p, set()))
            if n_train >= MIN_TRAIN_POSITIVES and n_test >= 1:
                evaluable_terms.append(term)

        if not evaluable_terms:
            continue

        # Build train label lookup (index-based)
        train_proteins = train_data["proteins"]

        fmax_list, auprc_list = [], []
        for term in evaluable_terms:
            # Train labels per index
            train_pos = np.array([1 if term in train_labels.get(p, set()) else 0
                                  for p in train_proteins])
            y_test = np.array([1 if term in test_labels.get(p, set()) else 0
                              for p in test_data["proteins"]])

            # Score = fraction of k neighbors that are positive
            y_scores = np.mean(train_pos[I], axis=1).astype(np.float64)

            fmax, auprc = compute_fmax(y_test, y_scores)
            fmax_list.append(fmax)
            auprc_list.append(auprc)

        results.append({
            "method": method,
            "classifier": f"knn-{KNN_K}",
            "test_species": test_species,
            "aspect": aspect,
            "n_terms": len(evaluable_terms),
            "n_train": train_data["n_proteins"],
            "n_test": test_data["n_proteins"],
            "fmax": float(np.mean(fmax_list)),
            "fmax_std": float(np.std(fmax_list)),
            "auprc": float(np.mean(auprc_list)),
            "auprc_std": float(np.std(auprc_list)),
        })

    return results


# ── Output ───────────────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    log("\n" + "=" * 90)
    log("CROSS-SPECIES GO TRANSFER: ARATH -> {ORYSA, ZEAMA, GLYMA, MEDTR}")
    log("=" * 90)
    log(f"{'Method':<25s} {'Clf':8s} {'Species':8s} {'Aspect':7s} "
        f"{'Terms':>6s} {'N_test':>7s} {'Fmax':>7s} {'AUPRC':>7s}")
    log("-" * 90)

    for r in sorted(results, key=lambda x: (x["test_species"], x["aspect"], x["method"], x["classifier"])):
        if r.get("error"):
            continue
        if r["n_terms"] == 0:
            continue
        log(f"{r['method']:<25s} {r['classifier']:8s} {r['test_species']:8s} {r['aspect']:7s} "
            f"{r['n_terms']:>6d} {r['n_test']:>7d} {r['fmax']:>7.3f} {r['auprc']:>7.3f}")

    log("=" * 90)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    t_start = time.time()

    # Step 1: Load data
    log("Step 1: Loading data...")
    download_obo()
    go_aspects = parse_go_aspects()
    log(f"  GO aspect mapping: {len(go_aspects):,} terms")

    cafa_df = load_cafa(go_aspects)

    # Report per-species coverage
    for sp in [TRAIN_SPECIES] + TEST_SPECIES:
        idmap = load_id_mapping(sp)
        matched = cafa_df[cafa_df["accession"].isin(idmap)]["accession"].nunique()
        log(f"  {sp}: {matched:,} CAFA proteins with ID mapping")

    # Step 2: Preload all data (once, in main process)
    log("\nStep 2: Preloading embeddings and building species data...")
    all_data = {}  # (species, method) -> data dict
    for method in ALL_METHODS:
        for sp in [TRAIN_SPECIES] + TEST_SPECIES:
            key = (sp, method)
            if key not in all_data:
                data = build_species_data(sp, cafa_df, method)
                all_data[key] = data
                if data:
                    log(f"  {sp}/{method}: {data['n_proteins']} proteins, dim={data['dim']}")
                else:
                    log(f"  {sp}/{method}: no data")

    # Step 3: LogReg transfer (sequential — data already in memory, classifiers are fast)
    log(f"\nStep 3: LogReg transfer ({len(ALL_METHODS)} methods x "
        f"{len(TEST_SPECIES)} species x {len(ASPECTS)} aspects)...")

    results = []
    total_jobs = len(ALL_METHODS) * len(TEST_SPECIES) * len(ASPECTS)
    done = 0
    for method in ALL_METHODS:
        for sp in TEST_SPECIES:
            train_data = all_data.get((TRAIN_SPECIES, method))
            test_data = all_data.get((sp, method))
            for aspect in ASPECTS:
                result = evaluate_logreg_aspect(method, sp, aspect, train_data, test_data)
                results.append(result)
                done += 1
                if result["n_terms"] > 0:
                    log(f"  [{done}/{total_jobs}] {method}/{sp}/{aspect}: "
                        f"Fmax={result['fmax']:.3f} ({result['n_terms']} terms, "
                        f"{result.get('elapsed_s', '?')}s)")
                else:
                    log(f"  [{done}/{total_jobs}] {method}/{sp}/{aspect}: no evaluable terms")

    # Step 3: k-NN transfer
    log("\nStep 3: k-NN transfer (FAISS)...")
    for method in ALL_METHODS:
        for sp in TEST_SPECIES:
            knn_results = run_knn_transfer(method, sp, cafa_df)
            results.extend(knn_results)

    # Step 4: Output
    elapsed = time.time() - t_start
    log(f"\nTotal time: {elapsed:.0f}s")

    output = {
        "experiment": "cross_species_go_transfer_cafa",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_species": TRAIN_SPECIES,
        "test_species": TEST_SPECIES,
        "methods": ALL_METHODS,
        "min_train_positives": MIN_TRAIN_POSITIVES,
        "knn_k": KNN_K,
        "total_elapsed_s": round(elapsed, 1),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {OUTPUT_PATH}")

    print_summary(results)


if __name__ == "__main__":
    main()
