"""Downstream evaluation matching SPACE paper protocol.

Four evaluation tasks:
  1. Orthogroup distance analysis (all 153 species)
  2. Subcellular localization prediction
  3. GO function prediction
  4. UMAP visualization

Annotation sources (matching SPACE paper standards):
  - Subloc: DeepLoc 2.0 benchmark (Swiss-Prot experimental, ECO:0000269)
           with homology-partitioned 5-fold CV (max 30% seq identity)
           Source: Thumuluri et al., NAR 2022
  - Subloc LOSO: UniProt CC annotations for 6 plant species (all evidence)
  - GO:    QuickGO experimental evidence only (EXP/IDA/IPI/IMP/IGI/IEP/TAS/IC)
           Source: EBI QuickGO REST API
  - GO LOSO fallback: UniProt GO annotations (all evidence) for broader coverage

Methods compared:
  - procn2v:     Procrustes N2V alignment (128-d)
  - vanilla:     Vanilla SPACE / FedCoder (512-d)
  - jaccard:     Jaccard-weighted SPACE (512-d)
  - prott5:      ProtT5 sequence embeddings (1024-d)
  - procn2v_t5:  Procrustes N2V + ProtT5 concatenation (1152-d)
  - vanilla_t5:  Vanilla SPACE + ProtT5 concatenation (1536-d)
  - jaccard_t5:  Jaccard SPACE + ProtT5 concatenation (1536-d)
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

METHODS = {
    "procn2v": ROOT / "results" / "aligned_embeddings_procrustes_n2v",
    "vanilla": ROOT / "results" / "aligned_embeddings",
    "jaccard": ROOT / "results" / "aligned_embeddings_jaccard",
    "prott5": ROOT / "data" / "prott5",
    "space_pub": ROOT / "results" / "space_published",
    "procrustes_string": ROOT / "results" / "procrustes_string",
    "proc_improved": ROOT / "results" / "aligned_embeddings_proc_jaccard_iter_csls",
    "proc_full": ROOT / "results" / "aligned_embeddings_proc_jaccard_iter_csls_trans_scale",
    "space_v2": ROOT / "results" / "aligned_embeddings_v2",
}

SEEDS = ["ARATH", "ORYSA", "PICAB", "SELMO", "MARPO"]

ANNOTATED_SPECIES = ["ARATH", "GLYMA", "MEDTR", "ORYSA", "POPTR", "ZEAMA"]

SUBLOC_COMPARTMENTS = [
    "Cytoplasm",
    "Nucleus",
    "Cell membrane",
    "Plastid",
    "Extracellular",
    "Mitochondrion",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Lysosome/Vacuole",
    "Peroxisome",
]

OG_DIR = ROOT / "data" / "orthofinder" / "transcripts_to_og"
SUBLOC_DIR = ROOT / "data" / "annotations" / "subloc"
DEEPLOC_DIR = ROOT / "data" / "annotations" / "deeploc"
GO_DIR = ROOT / "data" / "annotations" / "go"
GO_EXP_DIR = ROOT / "data" / "annotations" / "go_exp"
SEED_RESULTS = ROOT / "results" / "seed_selection.json"

OUTPUT_DIR = ROOT / "results" / "downstream"


# ---------------------------------------------------------------------------
# 1. Embedding loading
# ---------------------------------------------------------------------------


def load_embeddings(species: str, method: str) -> tuple[np.ndarray, list[str]]:
    """Load protein IDs and embeddings for a species+method.

    Returns (embeddings, protein_ids).
    """
    if method.endswith("_t5") and method != "prott5":
        aligned_method = method.removesuffix("_t5")
        return load_concat_embeddings(species, aligned_method)

    h5_dir = METHODS[method]
    h5_path = h5_dir / f"{species}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"No embeddings for {species} at {h5_path}")

    with h5py.File(h5_path, "r") as f:
        proteins = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]
        embeddings = f["embeddings"][:]

    return embeddings, proteins


def load_concat_embeddings(
    species: str, aligned_method: str = "procn2v"
) -> tuple[np.ndarray, list[str]]:
    """Load aligned + ProtT5, L2-normalize each, concatenate.

    Only returns proteins present in both embedding sets.
    """
    aligned_dir = METHODS[aligned_method]
    with h5py.File(aligned_dir / f"{species}.h5", "r") as f:
        aligned_prots = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]
        aligned_emb = f["embeddings"][:]

    with h5py.File(METHODS["prott5"] / f"{species}.h5", "r") as f:
        t5_prots = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]
        t5_emb = f["embeddings"][:]

    t5_idx = {p: i for i, p in enumerate(t5_prots)}
    shared_prots = []
    aligned_rows = []
    t5_rows = []
    for i, p in enumerate(aligned_prots):
        if p in t5_idx:
            shared_prots.append(p)
            aligned_rows.append(i)
            t5_rows.append(t5_idx[p])

    if not shared_prots:
        raise ValueError(f"No shared proteins for {species} between {aligned_method} and prott5")

    A = aligned_emb[aligned_rows]
    T = t5_emb[t5_rows]

    eps = 1e-12
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + eps)

    concat = np.hstack([A, T])
    return concat, shared_prots


# ---------------------------------------------------------------------------
# 2. Orthogroup distance analysis
# ---------------------------------------------------------------------------


def _load_og_mapping(species: str) -> dict[str, str]:
    """Load Protein_ID -> Orthogroup mapping."""
    tsv_path = OG_DIR / f"{species}_transcripts_to_OG.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["Orthogroup"])
    return dict(zip(df["Protein_ID"], df["Orthogroup"]))


def _sample_og_pairs(
    prots_a: list[str],
    prots_b: list[str],
    og_a: dict[str, str],
    og_b: dict[str, str],
    n_sample: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Sample orthologous (same OG) and non-orthologous (random) protein pairs.

    Returns (positive_pairs, negative_pairs) as lists of (idx_a, idx_b).
    """
    idx_a = {p: i for i, p in enumerate(prots_a)}
    idx_b = {p: i for i, p in enumerate(prots_b)}

    og_to_a: dict[str, list[int]] = defaultdict(list)
    for p, i in idx_a.items():
        if p in og_a:
            og_to_a[og_a[p]].append(i)

    og_to_b: dict[str, list[int]] = defaultdict(list)
    for p, i in idx_b.items():
        if p in og_b:
            og_to_b[og_b[p]].append(i)

    shared = set(og_to_a.keys()) & set(og_to_b.keys())
    if not shared:
        return [], []

    pos_pairs = []
    for og in shared:
        for ia in og_to_a[og]:
            for ib in og_to_b[og]:
                pos_pairs.append((ia, ib))

    if len(pos_pairs) > n_sample:
        indices = rng.choice(len(pos_pairs), size=n_sample, replace=False)
        pos_pairs = [pos_pairs[i] for i in indices]

    n_neg = len(pos_pairs)
    all_a = list(range(len(prots_a)))
    all_b = list(range(len(prots_b)))
    neg_pairs = []
    for _ in range(n_neg):
        ia = rng.choice(all_a)
        ib = rng.choice(all_b)
        neg_pairs.append((ia, ib))

    return pos_pairs, neg_pairs


def _evaluate_og_pair_cached(
    sp_a: str,
    sp_b: str,
    method: str,
    emb_cache: dict[str, tuple[np.ndarray, list[str]]],
    og_cache: dict[str, dict[str, str]],
    n_sample: int,
    seed: int,
) -> dict | None:
    """Evaluate a single species pair using pre-loaded embeddings and OG mappings."""
    rng = np.random.default_rng(seed)

    if sp_a not in emb_cache or sp_b not in emb_cache:
        return None

    emb_a, prots_a = emb_cache[sp_a]
    emb_b, prots_b = emb_cache[sp_b]
    og_a = og_cache[sp_a]
    og_b = og_cache[sp_b]

    pos_pairs, neg_pairs = _sample_og_pairs(
        prots_a, prots_b, og_a, og_b, n_sample, rng
    )

    if not pos_pairs:
        logger.warning(f"  {sp_a}-{sp_b}: no shared orthogroups")
        return None

    eps = 1e-12
    norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + eps)
    norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + eps)

    pos_ia = np.array([p[0] for p in pos_pairs])
    pos_ib = np.array([p[1] for p in pos_pairs])
    pos_sims = (norm_a[pos_ia] * norm_b[pos_ib]).sum(axis=1)

    neg_ia = np.array([p[0] for p in neg_pairs])
    neg_ib = np.array([p[1] for p in neg_pairs])
    neg_sims = (norm_a[neg_ia] * norm_b[neg_ib]).sum(axis=1)

    n_test = min(len(pos_sims), len(neg_sims))
    try:
        stat, p_val = wilcoxon(pos_sims[:n_test], neg_sims[:n_test], alternative="greater")
    except ValueError:
        stat, p_val = np.nan, np.nan

    effect_size = float(np.mean(pos_sims) - np.mean(neg_sims))

    if sp_a in SEEDS and sp_b in SEEDS:
        pair_type = "seed-seed"
    elif sp_a in SEEDS or sp_b in SEEDS:
        pair_type = "seed-nonseed"
    else:
        pair_type = "nonseed-nonseed"

    record = {
        "species_a": sp_a,
        "species_b": sp_b,
        "pair_type": pair_type,
        "method": method,
        "n_pos_pairs": len(pos_pairs),
        "n_neg_pairs": len(neg_pairs),
        "pos_sim_mean": float(np.mean(pos_sims)),
        "pos_sim_std": float(np.std(pos_sims)),
        "neg_sim_mean": float(np.mean(neg_sims)),
        "neg_sim_std": float(np.std(neg_sims)),
        "effect_size": effect_size,
        "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
        "wilcoxon_p": float(p_val) if not np.isnan(p_val) else None,
    }

    logger.info(
        f"  {sp_a}-{sp_b} ({method}): "
        f"pos={np.mean(pos_sims):.4f}±{np.std(pos_sims):.4f}  "
        f"neg={np.mean(neg_sims):.4f}±{np.std(neg_sims):.4f}  "
        f"effect={effect_size:.4f}  p={p_val:.2e}"
    )
    return record


def evaluate_og_distances(
    method: str,
    species_pairs: list[tuple[str, str]],
    n_sample: int = 5000,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute cosine similarity for ortholog vs non-ortholog pairs.

    Preloads all embeddings and OG mappings sequentially (I/O bound),
    then parallelizes the computation across species pairs (CPU bound).
    """
    # Collect unique species needed
    all_species = set()
    for sp_a, sp_b in species_pairs:
        all_species.add(sp_a)
        all_species.add(sp_b)

    # Preload embeddings (sequential — h5py I/O)
    emb_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    for sp in sorted(all_species):
        try:
            emb_cache[sp] = load_embeddings(sp, method)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"  Skipping {sp}: {e}")

    logger.info(f"  Preloaded {len(emb_cache)}/{len(all_species)} species for {method}")

    # Preload OG mappings (sequential — TSV I/O)
    og_cache: dict[str, dict[str, str]] = {}
    for sp in emb_cache:
        og_cache[sp] = _load_og_mapping(sp)

    # Give each pair a deterministic seed
    base_rng = np.random.default_rng(42)
    pair_seeds = base_rng.integers(0, 2**31, size=len(species_pairs))

    # Parallelize computation (numpy/scipy — releases GIL)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_evaluate_og_pair_cached)(
            sp_a, sp_b, method, emb_cache, og_cache, n_sample, int(seed)
        )
        for (sp_a, sp_b), seed in zip(species_pairs, pair_seeds)
    )

    records = [r for r in results if r is not None]
    return pd.DataFrame(records)


def get_og_species_pairs() -> list[tuple[str, str]]:
    """Build species pairs for OG distance evaluation."""
    with open(SEED_RESULTS) as f:
        seed_info = json.load(f)
    seeds = seed_info["seeds"]
    groups = seed_info.get("groups", {})

    pairs = []
    for a, b in combinations(seeds, 2):
        pairs.append((a, b))

    rng = np.random.default_rng(42)
    nonseeds = list(groups.keys())
    for seed in seeds:
        assigned = [ns for ns, s in groups.items() if s == seed]
        if len(assigned) > 5:
            assigned = list(rng.choice(assigned, size=5, replace=False))
        for ns in assigned:
            pairs.append(tuple(sorted([seed, ns])))

    if len(nonseeds) >= 2:
        ns_pairs = list(combinations(nonseeds, 2))
        if len(ns_pairs) > 10:
            idx = rng.choice(len(ns_pairs), size=10, replace=False)
            ns_pairs = [ns_pairs[i] for i in idx]
        pairs.extend(ns_pairs)

    return list(set(pairs))


# ---------------------------------------------------------------------------
# 3. Subcellular localization prediction
# ---------------------------------------------------------------------------


def _make_subloc_clf(seed: int = 42, n_jobs: int = -1) -> MultiOutputClassifier:
    """Create subloc classifier matching SPACE paper exactly.

    SPACE uses MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=seed)).
    n_jobs parallelizes across output labels.
    """
    return MultiOutputClassifier(
        LogisticRegression(max_iter=1000, random_state=seed), n_jobs=n_jobs
    )


def _subloc_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Compute subloc evaluation metrics."""
    results = {
        "f1_micro": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(Y_true, Y_pred)),
        "jaccard_micro": float(jaccard_score(Y_true, Y_pred, average="micro", zero_division=0)),
    }

    mcc_per_comp = {}
    for i, comp in enumerate(SUBLOC_COMPARTMENTS):
        if Y_true[:, i].sum() > 0:
            mcc_per_comp[comp] = float(matthews_corrcoef(Y_true[:, i], Y_pred[:, i]))
    results["mcc_per_compartment"] = mcc_per_comp

    return results


def _load_subloc_annotations(species: str) -> pd.DataFrame:
    """Load subloc annotations (UniProt CC, all evidence). Long format."""
    path = SUBLOC_DIR / f"{species}_subloc.tsv"
    return pd.read_csv(path, sep="\t")


def _prepare_subloc_data(
    species_list: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Prepare embeddings + multilabel matrix for subloc (UniProt annotations).

    Returns (X, Y, species_indices, protein_ids).
    """
    mlb = MultiLabelBinarizer(classes=SUBLOC_COMPARTMENTS)

    all_X = []
    all_labels = []
    all_species_idx = []
    all_pids = []

    for sp_idx, species in enumerate(species_list):
        try:
            emb, prots = load_embeddings(species, method)
        except FileNotFoundError:
            logger.warning(f"  Skipping {species}: no embeddings for {method}")
            continue

        prot_to_idx = {p: i for i, p in enumerate(prots)}
        annot = _load_subloc_annotations(species)

        grouped = annot.groupby("teagcn_id")["compartment"].apply(list).to_dict()

        for pid, compartments in grouped.items():
            if pid not in prot_to_idx:
                continue
            valid = [c for c in compartments if c in SUBLOC_COMPARTMENTS]
            if not valid:
                continue
            all_X.append(emb[prot_to_idx[pid]])
            all_labels.append(valid)
            all_species_idx.append(sp_idx)
            all_pids.append(pid)

    if not all_X:
        raise ValueError(f"No annotated proteins found for {method}")

    X = np.array(all_X, dtype=np.float32)
    Y = mlb.fit_transform(all_labels)
    species_indices = np.array(all_species_idx, dtype=np.int32)

    logger.info(f"  Subloc data ({method}): {X.shape[0]} proteins, {X.shape[1]}-d, {Y.shape[1]} compartments")
    return X, Y, species_indices, all_pids


def evaluate_subloc_deeploc_cv(
    method: str,
    species_list: list[str] | None = None,
) -> dict:
    """DeepLoc 2.0 homology-partitioned 5-fold CV (SPACE paper protocol).

    Uses experimentally validated Swiss-Prot annotations (ECO:0000269)
    with pre-computed partitions ensuring max 30% sequence identity between folds.

    This is the direct head-to-head comparison with SPACE's published numbers.
    """
    if species_list is None:
        # Only species with sufficient DeepLoc coverage
        species_list = [sp for sp in ANNOTATED_SPECIES
                        if (DEEPLOC_DIR / f"{sp}_deeploc.tsv").exists()]

    all_X = []
    all_Y = []
    all_partitions = []
    all_pids = []

    for species in species_list:
        deeploc_path = DEEPLOC_DIR / f"{species}_deeploc.tsv"
        if not deeploc_path.exists():
            continue

        try:
            emb, prots = load_embeddings(species, method)
        except FileNotFoundError:
            logger.warning(f"  Skipping {species}: no embeddings for {method}")
            continue

        prot_to_idx = {p: i for i, p in enumerate(prots)}
        df = pd.read_csv(deeploc_path, sep="\t")

        for _, row in df.iterrows():
            pid = row["teagcn_id"]
            if pid not in prot_to_idx:
                continue

            labels = [int(row.get(c, 0)) for c in SUBLOC_COMPARTMENTS]
            if sum(labels) == 0:
                continue

            all_X.append(emb[prot_to_idx[pid]])
            all_Y.append(labels)
            all_partitions.append(int(row.get("Partition", 0)))
            all_pids.append(pid)

    if not all_X:
        raise ValueError(f"No DeepLoc proteins found for {method}")

    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.int32)
    partitions = np.array(all_partitions, dtype=np.int32)

    logger.info(
        f"  DeepLoc CV data ({method}): {X.shape[0]} proteins, {X.shape[1]}-d, "
        f"{len(np.unique(partitions))} partitions"
    )

    # Homology-partitioned CV using DeepLoc's Partition column
    unique_parts = sorted(np.unique(partitions))
    fold_results = []

    for fold_idx, held_out_part in enumerate(unique_parts):
        train_mask = partitions != held_out_part
        test_mask = partitions == held_out_part

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        clf = _make_subloc_clf(seed=42)
        clf.fit(X[train_mask], Y[train_mask])
        Y_pred = clf.predict(X[test_mask])

        metrics = _subloc_metrics(Y[test_mask], Y_pred)
        metrics["fold"] = fold_idx
        metrics["partition"] = int(held_out_part)
        metrics["n_test"] = int(test_mask.sum())
        metrics["n_train"] = int(train_mask.sum())
        fold_results.append(metrics)

        logger.info(
            f"  DeepLoc fold {fold_idx} (partition={held_out_part}, {method}): "
            f"F1-micro={metrics['f1_micro']:.4f}  "
            f"F1-macro={metrics['f1_macro']:.4f}  "
            f"n_test={metrics['n_test']}"
        )

    avg = {
        "method": method,
        "eval_type": "deeploc_cv",
        "annotation_source": "DeepLoc 2.0 (Swiss-Prot experimental, ECO:0000269)",
        "n_proteins": len(X),
        "n_folds": len(fold_results),
        "f1_micro": float(np.mean([r["f1_micro"] for r in fold_results])),
        "f1_macro": float(np.mean([r["f1_macro"] for r in fold_results])),
        "accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
        "jaccard_micro": float(np.mean([r["jaccard_micro"] for r in fold_results])),
        "per_fold": fold_results,
    }
    logger.info(
        f"  DeepLoc CV average ({method}): F1-micro={avg['f1_micro']:.4f}  F1-macro={avg['f1_macro']:.4f}"
    )
    return avg


def evaluate_subloc_loso(
    method: str,
    species_list: list[str] | None = None,
) -> dict:
    """Leave-one-species-out CV for subcellular localization.

    Uses UniProt CC annotations (all evidence codes) across 6 plant species.
    Tests whether alignment enables cross-species functional transfer.
    """
    if species_list is None:
        species_list = ANNOTATED_SPECIES

    X, Y, sp_idx, pids = _prepare_subloc_data(species_list, method)
    clf = _make_subloc_clf(seed=42)

    per_species = []
    for hold_idx, hold_species in enumerate(species_list):
        train_mask = sp_idx != hold_idx
        test_mask = sp_idx == hold_idx

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        clf.fit(X[train_mask], Y[train_mask])
        Y_pred = clf.predict(X[test_mask])

        metrics = _subloc_metrics(Y[test_mask], Y_pred)
        metrics["species"] = hold_species
        metrics["n_test"] = int(test_mask.sum())
        metrics["n_train"] = int(train_mask.sum())
        per_species.append(metrics)

        logger.info(
            f"  LOSO {hold_species} ({method}): "
            f"F1-micro={metrics['f1_micro']:.4f}  "
            f"F1-macro={metrics['f1_macro']:.4f}  "
            f"n_test={metrics['n_test']}"
        )

    avg = {
        "method": method,
        "eval_type": "loso",
        "annotation_source": "UniProt CC (all evidence codes)",
        "f1_micro": float(np.mean([r["f1_micro"] for r in per_species])),
        "f1_macro": float(np.mean([r["f1_macro"] for r in per_species])),
        "accuracy": float(np.mean([r["accuracy"] for r in per_species])),
        "jaccard_micro": float(np.mean([r["jaccard_micro"] for r in per_species])),
        "per_species": per_species,
    }
    logger.info(
        f"  LOSO average ({method}): F1-micro={avg['f1_micro']:.4f}  F1-macro={avg['f1_macro']:.4f}"
    )
    return avg


def evaluate_subloc_cv(
    method: str,
    species_list: list[str] | None = None,
    n_folds: int = 5,
) -> dict:
    """Random k-fold CV across pooled species (UniProt annotations)."""
    if species_list is None:
        species_list = ANNOTATED_SPECIES

    X, Y, sp_idx, pids = _prepare_subloc_data(species_list, method)

    single_labels = Y.argmax(axis=1)

    clf = _make_subloc_clf(seed=42)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, single_labels)):
        clf.fit(X[train_idx], Y[train_idx])
        Y_pred = clf.predict(X[test_idx])

        metrics = _subloc_metrics(Y[test_idx], Y_pred)
        metrics["fold"] = fold
        fold_results.append(metrics)

        logger.info(
            f"  CV fold {fold} ({method}): "
            f"F1-micro={metrics['f1_micro']:.4f}  "
            f"F1-macro={metrics['f1_macro']:.4f}"
        )

    avg = {
        "method": method,
        "eval_type": "cv",
        "annotation_source": "UniProt CC (all evidence codes)",
        "n_folds": n_folds,
        "f1_micro": float(np.mean([r["f1_micro"] for r in fold_results])),
        "f1_macro": float(np.mean([r["f1_macro"] for r in fold_results])),
        "accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
        "jaccard_micro": float(np.mean([r["jaccard_micro"] for r in fold_results])),
        "per_fold": fold_results,
    }
    logger.info(
        f"  CV average ({method}): F1-micro={avg['f1_micro']:.4f}  F1-macro={avg['f1_macro']:.4f}"
    )
    return avg


# ---------------------------------------------------------------------------
# 4. GO function prediction
# ---------------------------------------------------------------------------


def _load_go_annotations(
    species: str, aspect: str, min_annotations: int = 10,
    experimental_only: bool = True,
) -> pd.DataFrame:
    """Load GO annotations for a species and aspect.

    Args:
        experimental_only: If True, use QuickGO experimental evidence annotations
                          (data/annotations/go_exp/). If False, use all UniProt
                          annotations (data/annotations/go/).
    """
    if experimental_only:
        path = GO_EXP_DIR / f"{species}_goa.tsv"
        if not path.exists():
            logger.warning(
                f"  {species}: no experimental GO file, falling back to all evidence"
            )
            path = GO_DIR / f"{species}_goa.tsv"
    else:
        path = GO_DIR / f"{species}_goa.tsv"

    df = pd.read_csv(path, sep="\t")
    df = df[df["aspect"] == aspect]

    term_counts = df["go_term"].value_counts()
    valid_terms = term_counts[term_counts >= min_annotations].index
    df = df[df["go_term"].isin(valid_terms)]

    return df


def _compute_fmax(Y_true: np.ndarray, Y_prob: np.ndarray) -> float:
    """Compute protein-centric Fmax over thresholds.

    Uses 0.001 step size matching SPACE's cafaeval th_step parameter.
    """
    best_f = 0.0
    for t in np.arange(0.01, 1.0, 0.001):
        Y_pred = (Y_prob >= t).astype(int)
        if Y_pred.sum() == 0:
            continue
        f = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
        if f > best_f:
            best_f = f
    return best_f


def _compute_auprc_macro(Y_true: np.ndarray, Y_prob: np.ndarray) -> float:
    """Compute macro-averaged AUPRC across GO terms."""
    auprcs = []
    for i in range(Y_true.shape[1]):
        if Y_true[:, i].sum() == 0:
            continue
        auprc = float(average_precision_score(Y_true[:, i], Y_prob[:, i]))
        auprcs.append(auprc)
    return float(np.mean(auprcs)) if auprcs else 0.0


def evaluate_go_prediction(
    method: str,
    species_list: list[str] | None = None,
    min_annotations: int = 10,
    experimental_only: bool = True,
) -> dict:
    """LOSO GO term prediction with Fmax/AUPRC for each aspect.

    Args:
        experimental_only: If True, use experimental-evidence GO annotations
                          (EXP/IDA/IPI/IMP/IGI/IEP/TAS/IC from QuickGO).
    """
    if species_list is None:
        species_list = ANNOTATED_SPECIES

    annotation_source = (
        "QuickGO experimental (EXP/IDA/IPI/IMP/IGI/IEP/TAS/IC)"
        if experimental_only
        else "UniProt GO (all evidence codes)"
    )
    results = {"method": method, "annotation_source": annotation_source, "per_aspect": {}}

    for aspect in ["CC", "BP", "MF"]:
        logger.info(f"  GO {aspect} ({method})")

        all_X = []
        all_labels = []
        all_sp_idx = []

        all_terms = set()
        per_species_annot = {}
        for species in species_list:
            df = _load_go_annotations(species, aspect, min_annotations, experimental_only)
            per_species_annot[species] = df
            all_terms.update(df["go_term"].unique())

        if not all_terms:
            logger.warning(f"  No GO terms for {aspect} with >= {min_annotations} annotations")
            results["per_aspect"][aspect] = {"fmax": 0.0, "auprc": 0.0, "n_terms": 0}
            continue

        term_list = sorted(all_terms)
        term_to_idx = {t: i for i, t in enumerate(term_list)}

        for sp_idx, species in enumerate(species_list):
            try:
                emb, prots = load_embeddings(species, method)
            except FileNotFoundError:
                continue

            prot_to_emb_idx = {p: i for i, p in enumerate(prots)}
            df = per_species_annot[species]

            grouped = df.groupby("teagcn_id")["go_term"].apply(set).to_dict()

            for pid, terms in grouped.items():
                if pid not in prot_to_emb_idx:
                    continue
                valid = terms & all_terms
                if not valid:
                    continue
                all_X.append(emb[prot_to_emb_idx[pid]])
                all_labels.append(valid)
                all_sp_idx.append(sp_idx)

        if not all_X:
            results["per_aspect"][aspect] = {"fmax": 0.0, "auprc": 0.0, "n_terms": 0}
            continue

        X = np.array(all_X, dtype=np.float32)
        sp_indices = np.array(all_sp_idx, dtype=np.int32)

        Y = np.zeros((len(all_labels), len(term_list)), dtype=np.int32)
        for i, terms in enumerate(all_labels):
            for t in terms:
                Y[i, term_to_idx[t]] = 1

        # Keep terms present in at least 2 species (needed for LOSO)
        term_species = np.zeros((len(term_list), len(species_list)), dtype=bool)
        for i in range(len(all_labels)):
            for t in all_labels[i]:
                term_species[term_to_idx[t], sp_indices[i]] = True
        valid_terms_mask = term_species.sum(axis=1) >= 2
        if valid_terms_mask.sum() == 0:
            results["per_aspect"][aspect] = {"fmax": 0.0, "auprc": 0.0, "n_terms": 0}
            continue

        Y = Y[:, valid_terms_mask]
        filtered_terms = [t for t, v in zip(term_list, valid_terms_mask) if v]

        # LOSO evaluation
        fmax_list = []
        auprc_list = []
        per_species_results = []

        for hold_idx, hold_species in enumerate(species_list):
            train_mask = sp_indices != hold_idx
            test_mask = sp_indices == hold_idx

            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue

            X_train, Y_train = X[train_mask], Y[train_mask]
            X_test, Y_test = X[test_mask], Y[test_mask]

            # Per-term binary LogisticRegression (matching SPACE func_pred.py)
            clf = MultiOutputClassifier(
                LogisticRegression(max_iter=1000, random_state=42), n_jobs=-1
            )
            clf.fit(X_train, Y_train)

            # Get probability predictions
            try:
                Y_prob = np.column_stack([
                    est.predict_proba(X_test)[:, 1] if hasattr(est, "predict_proba")
                    else 1 / (1 + np.exp(-est.decision_function(X_test)))
                    for est in clf.estimators_
                ])
            except Exception:
                Y_prob = clf.predict_proba(X_test)
                if isinstance(Y_prob, list):
                    Y_prob = np.column_stack([p[:, 1] for p in Y_prob])

            fmax = _compute_fmax(Y_test, Y_prob)
            auprc = _compute_auprc_macro(Y_test, Y_prob)

            fmax_list.append(fmax)
            auprc_list.append(auprc)
            per_species_results.append({
                "species": hold_species,
                "fmax": fmax,
                "auprc": auprc,
                "n_test": int(test_mask.sum()),
            })

            logger.info(
                f"    LOSO {hold_species}: Fmax={fmax:.4f}  AUPRC={auprc:.4f}  n={test_mask.sum()}"
            )

        aspect_result = {
            "fmax": float(np.mean(fmax_list)) if fmax_list else 0.0,
            "auprc": float(np.mean(auprc_list)) if auprc_list else 0.0,
            "n_terms": len(filtered_terms),
            "per_species": per_species_results,
        }
        results["per_aspect"][aspect] = aspect_result

        logger.info(
            f"  GO {aspect} average ({method}): "
            f"Fmax={aspect_result['fmax']:.4f}  AUPRC={aspect_result['auprc']:.4f}  "
            f"n_terms={aspect_result['n_terms']}"
        )

    return results


# ---------------------------------------------------------------------------
# 5. UMAP visualization
# ---------------------------------------------------------------------------


def plot_umap_species(
    method: str,
    species_list: list[str] | None = None,
    n_per_species: int = 2000,
    output_path: Path | None = None,
) -> None:
    """UMAP plot colored by species for cross-species structure."""
    import umap

    if species_list is None:
        species_list = SEEDS

    if output_path is None:
        output_path = OUTPUT_DIR / "umap" / f"{method}_species_umap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    all_X = []
    all_labels = []

    for species in species_list:
        try:
            emb, prots = load_embeddings(species, method)
        except FileNotFoundError:
            logger.warning(f"  Skipping {species}: no embeddings")
            continue

        n = min(n_per_species, len(prots))
        idx = rng.choice(len(prots), size=n, replace=False)
        all_X.append(emb[idx])
        all_labels.extend([species] * n)

    if not all_X:
        logger.warning("No data for UMAP")
        return

    X = np.vstack(all_X).astype(np.float32)

    eps = 1e-12
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    logger.info(f"  Computing UMAP for {len(X)} points ({method})...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    coords = reducer.fit_transform(X)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_species = list(dict.fromkeys(all_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_species)))

    for sp, color in zip(unique_species, colors):
        mask = np.array([l == sp for l in all_labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=1, alpha=0.3, label=sp, color=color)

    ax.legend(markerscale=5, fontsize=9)
    ax.set_title(f"UMAP — {method} (species)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"  Saved {output_path}")


def plot_umap_subloc(
    method: str,
    species_list: list[str] | None = None,
    n_per_species: int = 3000,
    output_path: Path | None = None,
) -> None:
    """UMAP plot colored by subcellular compartment."""
    import umap

    if species_list is None:
        species_list = ANNOTATED_SPECIES

    if output_path is None:
        output_path = OUTPUT_DIR / "umap" / f"{method}_subloc_umap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    all_X = []
    all_labels = []

    for species in species_list:
        try:
            emb, prots = load_embeddings(species, method)
        except FileNotFoundError:
            continue

        prot_to_idx = {p: i for i, p in enumerate(prots)}
        annot = _load_subloc_annotations(species)

        first_comp = annot.drop_duplicates(subset=["teagcn_id"])
        matched = first_comp[first_comp["teagcn_id"].isin(prot_to_idx)]

        if len(matched) == 0:
            continue

        n = min(n_per_species, len(matched))
        sampled = matched.sample(n=n, random_state=42)

        for _, row in sampled.iterrows():
            idx = prot_to_idx[row["teagcn_id"]]
            all_X.append(emb[idx])
            all_labels.append(row["compartment"])

    if not all_X:
        logger.warning("No data for subloc UMAP")
        return

    X = np.vstack(all_X).astype(np.float32)

    eps = 1e-12
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    logger.info(f"  Computing subloc UMAP for {len(X)} points ({method})...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    coords = reducer.fit_transform(X)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(all_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_labels), 10)))

    for i, comp in enumerate(unique_labels):
        mask = np.array([l == comp for l in all_labels])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=1, alpha=0.3, label=comp, color=colors[i % len(colors)],
        )

    ax.legend(markerscale=5, fontsize=8, loc="best")
    ax.set_title(f"UMAP — {method} (subcellular localization)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# 6. Comparison summary
# ---------------------------------------------------------------------------


def build_comparison_summary(
    methods: list[str],
    og_results: dict[str, pd.DataFrame] | None = None,
    subloc_deeploc: dict[str, dict] | None = None,
    subloc_loso: dict[str, dict] | None = None,
    subloc_cv: dict[str, dict] | None = None,
    go_results: dict[str, dict] | None = None,
) -> dict:
    """Build side-by-side comparison across methods."""
    summary = {}

    for method in methods:
        entry = {"method": method}

        if og_results and method in og_results:
            df = og_results[method]
            entry["og_distances"] = {
                "mean_effect_size": float(df["effect_size"].mean()),
                "mean_pos_sim": float(df["pos_sim_mean"].mean()),
                "mean_neg_sim": float(df["neg_sim_mean"].mean()),
                "n_pairs": len(df),
            }

        if subloc_deeploc and method in subloc_deeploc:
            r = subloc_deeploc[method]
            entry["subloc_deeploc_cv"] = {
                "f1_micro": r["f1_micro"],
                "f1_macro": r["f1_macro"],
                "accuracy": r["accuracy"],
                "n_proteins": r.get("n_proteins", 0),
                "annotation_source": r.get("annotation_source", ""),
            }

        if subloc_loso and method in subloc_loso:
            r = subloc_loso[method]
            entry["subloc_loso"] = {
                "f1_micro": r["f1_micro"],
                "f1_macro": r["f1_macro"],
                "accuracy": r["accuracy"],
            }

        if subloc_cv and method in subloc_cv:
            r = subloc_cv[method]
            entry["subloc_cv"] = {
                "f1_micro": r["f1_micro"],
                "f1_macro": r["f1_macro"],
                "accuracy": r["accuracy"],
            }

        if go_results and method in go_results:
            r = go_results[method]
            entry["go_pred"] = {"annotation_source": r.get("annotation_source", "")}
            for aspect in ["CC", "BP", "MF"]:
                if aspect in r.get("per_aspect", {}):
                    a = r["per_aspect"][aspect]
                    entry["go_pred"][aspect] = {
                        "fmax": a["fmax"],
                        "auprc": a["auprc"],
                        "n_terms": a["n_terms"],
                    }

        summary[method] = entry

    return summary
