"""Seed species validation for SPACE alignment pipeline.

Validates seed species quality through:
  - Node2Vec stability (embedding variance across multiple runs)
  - Network quality metrics (density, degree distribution)
  - Ortholog coverage (fraction of genes with OrthoFinder assignments)

Purpose: Prevent cascade failures in downstream alignments by ensuring
         seed species meet quality criteria before aligning non-seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
from loguru import logger

from orbit.data_prep import generate_embeddings, _load_protein_to_og


# ---------------------------------------------------------------------------
# 1. Node2Vec stability measurement
# ---------------------------------------------------------------------------


def measure_node2vec_stability(
    species: str,
    network_gz: Path,
    network_dir: Path,
    output_dir: Path,
    n_runs: int = 5,
    dimensions: int = 128,
) -> dict:
    """Run Node2Vec n times with different seeds, measure embedding variance.

    Returns: {
        "species": str,
        "n_runs": int,
        "mean_variance": float,  # Average variance across genes
        "max_variance": float,   # Worst-case gene variance
        "median_variance": float,
        "std_variance": float,
        "pass": bool,  # mean_variance < 0.3 (threshold from requirements)
        "threshold": 0.3
    }
    """
    threshold = 0.3

    try:
        # Create output directory for temporary embeddings
        temp_dir = output_dir / "temp_stability"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate embeddings n_runs times with different random states
        run_files = []
        logger.info(f"  {species}: Running Node2Vec stability test ({n_runs} runs)")

        for i in range(n_runs):
            run_seed = 1234 + i  # Vary random state
            output_h5 = temp_dir / f"{species}_run{i}.h5"
            run_files.append(output_h5)

            logger.debug(f"    Run {i+1}/{n_runs} (seed={run_seed})")
            generate_embeddings(
                network_gz_path=network_gz,
                output_h5_path=output_h5,
                dimensions=dimensions,
                p=0.3,
                q=0.7,
                random_state=run_seed,
            )

        # Load all embeddings into numpy array
        # First pass: collect all gene IDs and build canonical order
        all_genes = set()
        run_data = []

        for h5_file in run_files:
            with h5py.File(h5_file, "r") as f:
                proteins = [p.decode() if isinstance(p, bytes) else p for p in f["proteins"][:]]
                emb = f["embeddings"][:]
                run_data.append((proteins, emb))
                all_genes.update(proteins)

        # Create canonical gene order (sorted for consistency)
        gene_ids = sorted(all_genes)
        n_genes = len(gene_ids)
        gene_to_idx = {g: i for i, g in enumerate(gene_ids)}

        # Second pass: align all runs to canonical gene order
        embeddings_list = []
        for proteins, emb in run_data:
            # Create mapping from run's gene order to canonical order
            aligned_emb = np.zeros((n_genes, emb.shape[1]), dtype=emb.dtype)
            for i, gene in enumerate(proteins):
                canonical_idx = gene_to_idx[gene]
                aligned_emb[canonical_idx] = emb[i]

            # L2-normalize embeddings
            norms = np.linalg.norm(aligned_emb, axis=1, keepdims=True)
            emb_norm = aligned_emb / (norms + 1e-12)
            embeddings_list.append(emb_norm)

        # Stack: (n_runs, N_genes, dimensions)
        embeddings = np.stack(embeddings_list, axis=0)
        n_runs_actual, n_genes, dim = embeddings.shape

        logger.debug(f"    Loaded {n_runs_actual} runs: {n_genes} genes × {dim} dimensions")

        # Compute pairwise cosine similarity variance per gene
        variances = []

        for gene_idx in range(n_genes):
            # Get (n_runs, dim) matrix for this gene
            gene_embs = embeddings[:, gene_idx, :]

            # Compute all pairwise cosine similarities
            # Since embeddings are L2-normalized, cosine = dot product
            similarity_matrix = gene_embs @ gene_embs.T  # (n_runs, n_runs)

            # Extract upper triangle (exclude diagonal)
            triu_indices = np.triu_indices(n_runs_actual, k=1)
            similarities = similarity_matrix[triu_indices]

            # Compute variance of these similarities
            var = np.var(similarities)
            variances.append(var)

        variances = np.array(variances)

        # Aggregate statistics
        mean_var = float(np.mean(variances))
        max_var = float(np.max(variances))
        median_var = float(np.median(variances))
        std_var = float(np.std(variances))

        # Pass/fail
        passed = mean_var < threshold

        # Cleanup: remove temporary run files
        for h5_file in run_files:
            h5_file.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass  # Directory not empty or already removed

        result = {
            "species": species,
            "n_runs": n_runs_actual,
            "n_genes": n_genes,
            "mean_variance": mean_var,
            "max_variance": max_var,
            "median_variance": median_var,
            "std_variance": std_var,
            "pass": passed,
            "threshold": threshold,
        }

        logger.info(
            f"  {species}: mean_variance={mean_var:.4f} "
            f"(threshold={threshold}) → {'PASS' if passed else 'FAIL'}"
        )

        return result

    except Exception as e:
        logger.error(f"  {species}: Node2Vec stability test failed: {e}")
        return {
            "species": species,
            "n_runs": 0,
            "n_genes": 0,
            "mean_variance": float('inf'),
            "max_variance": float('inf'),
            "median_variance": float('inf'),
            "std_variance": float('inf'),
            "pass": False,
            "threshold": threshold,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 2. Network quality metrics
# ---------------------------------------------------------------------------


def compute_network_quality(
    species: str,
    network_path: Path,
) -> dict:
    """Compute network density and other quality metrics.

    Returns: {
        "species": str,
        "n_genes": int,
        "n_edges": int,
        "density": float,  # n_edges / (n_genes * (n_genes - 1) / 2)
        "avg_degree": float,  # mean edges per gene
        "pass": bool,  # density in reasonable range (0.001 to 0.5)
        "density_min": 0.001,
        "density_max": 0.5
    }
    """
    density_min = 0.001
    density_max = 0.5

    try:
        # Load network neighborhoods
        neighbors = defaultdict(set)
        with open(network_path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
                neighbors[a].add(b)
                neighbors[b].add(a)

        n_genes = len(neighbors)
        n_edges = sum(len(nbrs) for nbrs in neighbors.values()) // 2  # Each edge counted twice

        # Compute density
        max_edges = n_genes * (n_genes - 1) // 2
        density = n_edges / max_edges if max_edges > 0 else 0.0

        # Compute average degree
        avg_degree = sum(len(nbrs) for nbrs in neighbors.values()) / n_genes if n_genes > 0 else 0.0

        # Pass/fail
        passed = density_min < density < density_max

        result = {
            "species": species,
            "n_genes": n_genes,
            "n_edges": n_edges,
            "density": density,
            "avg_degree": avg_degree,
            "pass": passed,
            "density_min": density_min,
            "density_max": density_max,
        }

        logger.info(
            f"  {species}: density={density:.4f} "
            f"({density_min}–{density_max}) → {'PASS' if passed else 'FAIL'}"
        )

        return result

    except Exception as e:
        logger.error(f"  {species}: Network quality check failed: {e}")
        return {
            "species": species,
            "n_genes": 0,
            "n_edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "pass": False,
            "density_min": density_min,
            "density_max": density_max,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 3. Ortholog coverage check
# ---------------------------------------------------------------------------


def check_ortholog_coverage(
    species: str,
    transcripts_dir: Path,
    network_path: Path,
) -> dict:
    """Check what fraction of genes have ortholog assignments.

    Returns: {
        "species": str,
        "n_genes_in_network": int,
        "n_genes_with_og": int,
        "coverage": float,  # n_with_og / n_in_network
        "pass": bool,  # coverage > 0.8 (80% coverage threshold)
        "threshold": 0.8
    }
    """
    threshold = 0.8

    try:
        # Load network genes
        network_genes = set()
        with open(network_path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                network_genes.add(parts[0])
                network_genes.add(parts[1])

        n_genes_in_network = len(network_genes)

        # Load OrthoFinder mapping
        og_map = _load_protein_to_og(transcripts_dir, species, network_genes)
        n_genes_with_og = len(og_map)

        # Compute coverage
        coverage = n_genes_with_og / n_genes_in_network if n_genes_in_network > 0 else 0.0

        # Pass/fail
        passed = coverage > threshold

        result = {
            "species": species,
            "n_genes_in_network": n_genes_in_network,
            "n_genes_with_og": n_genes_with_og,
            "coverage": coverage,
            "pass": passed,
            "threshold": threshold,
        }

        logger.info(
            f"  {species}: coverage={coverage:.2%} "
            f"(threshold={threshold}) → {'PASS' if passed else 'FAIL'}"
        )

        return result

    except Exception as e:
        logger.error(f"  {species}: Ortholog coverage check failed: {e}")
        return {
            "species": species,
            "n_genes_in_network": 0,
            "n_genes_with_og": 0,
            "coverage": 0.0,
            "pass": False,
            "threshold": threshold,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 4. Helper for writing JSON reports
# ---------------------------------------------------------------------------


def write_validation_report(
    output_path: Path,
    phase: str,
    validation_type: str,
    results: dict[str, dict],
    timestamp: str,
) -> None:
    """Write validation results to JSON file.

    Args:
        output_path: Path to output JSON file
        phase: Phase identifier (e.g., "01-data-preparation-seed-validation")
        validation_type: Type of validation (e.g., "stability", "quality")
        results: Dict mapping species codes to validation result dicts
        timestamp: ISO 8601 timestamp
    """
    # Compute summary
    passed = sum(1 for r in results.values() if r.get("pass", False))
    failed = len(results) - passed
    overall_status = "PASS" if failed == 0 else "FAIL"

    # Extract thresholds (from first result)
    first_result = next(iter(results.values()))
    thresholds = {k: v for k, v in first_result.items() if k.endswith(("threshold", "_min", "_max"))}

    report = {
        "phase": phase,
        "validation_type": validation_type,
        "timestamp": timestamp,
        "status": overall_status,
        "summary": {
            "total_seeds": len(results),
            "passed": passed,
            "failed": failed,
        },
        "thresholds": thresholds,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info(f"  Wrote validation report: {output_path}")
