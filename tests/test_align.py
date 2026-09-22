"""Tests for the ORBIT rotation core (orbit.align)."""

from __future__ import annotations

import numpy as np
import pytest

from orbit.align import (
    is_rotation,
    iter_two_stage_rotations,
    procrustes_rotation,
    rotate,
    rotation_from_alignment,
    two_stage_rotations,
)

D = 128


def random_rotation(rng: np.random.Generator, d: int = D) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# --- procrustes_rotation --------------------------------------------------------------

def test_rotation_is_orthogonal_with_positive_determinant():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, D))
    B = rng.standard_normal((500, D))
    R = procrustes_rotation(A, B)
    assert R.shape == (D, D)
    assert np.allclose(R.T @ R, np.eye(D), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
    assert is_rotation(R)


def test_known_rotation_is_recovered_exactly():
    rng = np.random.default_rng(1)
    R_true = random_rotation(rng)
    A = rng.standard_normal((2000, D))
    B = A @ R_true
    R = procrustes_rotation(A, B)
    assert np.allclose(R, R_true, atol=1e-10)


def test_reflection_is_corrected_to_a_proper_rotation():
    rng = np.random.default_rng(2)
    F = np.eye(D)
    F[0, 0] = -1  # a reflection, det -1
    A = rng.standard_normal((300, D))
    R = procrustes_rotation(A, A @ F)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
    assert not is_rotation(F)


def test_reflection_is_kept_when_reflections_are_allowed():
    rng = np.random.default_rng(5)
    F = np.eye(D)
    F[0, 0] = -1
    A = rng.standard_normal((300, D))
    R = procrustes_rotation(A, A @ F, proper=False)
    assert np.isclose(np.linalg.det(R), -1.0, atol=1e-8)
    assert np.allclose(A @ R, A @ F, atol=1e-8)
    assert is_rotation(R, proper=False) and not is_rotation(R)


def test_unconstrained_solution_equals_scipy_orthogonal_procrustes():
    """proper=False is the call that produced the released STRING files."""
    from scipy.linalg import orthogonal_procrustes

    rng = np.random.default_rng(6)
    A = rng.standard_normal((800, D))
    B = rng.standard_normal((800, D)) @ random_rotation(rng)
    R_scipy, _ = orthogonal_procrustes(A, B)
    assert np.allclose(procrustes_rotation(A, B, proper=False), R_scipy, atol=1e-10)


def test_within_species_geometry_is_preserved():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, D)).astype(np.float32)
    R = random_rotation(rng)
    Y = rotate(X, R)
    assert Y.dtype == np.float32
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    assert np.allclose(Xn @ Xn.T, Yn @ Yn.T, atol=1e-5)
    assert np.allclose(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1), rtol=1e-6)


def test_input_validation():
    with pytest.raises(ValueError):
        procrustes_rotation(np.zeros((3, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        procrustes_rotation(np.zeros((1, 4)), np.zeros((1, 4)))


def test_rotation_from_alignment_round_trips_a_float32_embedding():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((3000, D)).astype(np.float32)
    R_true = random_rotation(rng).astype(np.float32)
    Y = (X @ R_true).astype(np.float32)
    R = rotation_from_alignment(X, Y)
    assert np.abs(X.astype(np.float64) @ R - Y).max() < 1e-5


# --- two_stage_rotations ----------------------------------------------------------------

def _latent_species(rng, n=600, d=D, names=("ARATH", "ORYSA", "MARPO", "ABROB", "TRIAE")):
    """Every species is the same latent geometry Z in a private random orientation Q_s."""
    Z = rng.standard_normal((n, d))
    Q = {s: random_rotation(rng) for s in names}
    X = {s: Z @ Q[s] for s in names}
    return Z, Q, X


def test_two_stage_reference_is_identity_and_all_species_land_in_its_frame():
    rng = np.random.default_rng(10)
    Z, Q, X = _latent_species(rng)
    anchors = lambda s, r: (X[s][:200], X[r][:200])  # raw-frame anchors, 200 shared genes
    R = two_stage_rotations(anchors, seeds=["ARATH", "ORYSA", "MARPO"], reference="ARATH",
                            seed_of={"ABROB": "MARPO", "TRIAE": "ORYSA"})
    assert set(R) == {"ARATH", "ORYSA", "MARPO", "ABROB", "TRIAE"}
    assert np.array_equal(R["ARATH"], np.eye(D))
    for s in R:
        assert is_rotation(R[s])
        assert np.allclose(rotate(X[s], R[s]), rotate(X["ARATH"], R["ARATH"]), atol=1e-8)


def test_two_stage_nonseed_is_fitted_to_the_aligned_seed_not_the_raw_seed():
    rng = np.random.default_rng(11)
    Z, Q, X = _latent_species(rng)
    calls = []

    def anchors(s, r):
        calls.append((s, r))
        return X[s][:150], X[r][:150]

    R = two_stage_rotations(anchors, seeds=["ARATH", "MARPO"], reference="ARATH",
                            seed_of={"ABROB": "MARPO"})
    assert ("ABROB", "MARPO") in calls and ("ABROB", "ARATH") not in calls
    # ABROB reaches the reference frame by transitivity through MARPO
    assert np.allclose(rotate(X["ABROB"], R["ABROB"]), X["ARATH"], atol=1e-8)


def test_two_stage_omits_species_with_too_few_anchors():
    rng = np.random.default_rng(12)
    Z, Q, X = _latent_species(rng)

    def anchors(s, r):
        k = 1 if s == "TRIAE" else 100
        return X[s][:k], X[r][:k]

    R = two_stage_rotations(anchors, seeds=["ARATH", "ORYSA"], reference="ARATH",
                            seed_of={"TRIAE": "ORYSA"}, min_anchors=2)
    assert "TRIAE" not in R and "ORYSA" in R


def test_two_stage_validates_reference_and_seed_assignment():
    rng = np.random.default_rng(13)
    Z, Q, X = _latent_species(rng)
    anchors = lambda s, r: (X[s][:50], X[r][:50])
    with pytest.raises(ValueError):
        two_stage_rotations(anchors, seeds=["ORYSA"], reference="ARATH", seed_of={})
    with pytest.raises(ValueError):
        two_stage_rotations(anchors, seeds=["ARATH"], reference="ARATH", seed_of={"ABROB": "PICAB"})


def test_two_stage_can_return_improper_rotations_when_asked():
    rng = np.random.default_rng(14)
    Z, Q, X = _latent_species(rng)
    F = np.eye(D)
    F[0, 0] = -1
    X["ORYSA"] = X["ARATH"] @ F  # ORYSA is a mirror image of the reference
    anchors = lambda s, r: (X[s][:300], X[r][:300])
    R = two_stage_rotations(anchors, seeds=["ARATH", "ORYSA"], reference="ARATH", seed_of={},
                            proper=False)
    assert np.isclose(np.linalg.det(R["ORYSA"]), -1.0, atol=1e-8)
    assert np.allclose(rotate(X["ORYSA"], R["ORYSA"]), X["ARATH"], atol=1e-8)


def test_two_stage_with_aligned_seeds_skips_stage_one():
    """Seeds that already share the reference frame (the released files) get the identity
    and are never fitted; a non-seed is rotated straight onto its seed's coordinates."""
    rng = np.random.default_rng(15)
    Z, Q, X = _latent_species(rng)
    X["ORYSA"] = X["ARATH"].copy()  # already aligned
    calls = []

    def anchors(s, r):
        calls.append((s, r))
        return X[s][:200], X[r][:200]

    R = two_stage_rotations(anchors, seeds=["ARATH", "ORYSA"], reference="ARATH",
                            seed_of={"TRIAE": "ORYSA"}, seeds_aligned=True)
    assert ("ORYSA", "ARATH") not in calls and ("TRIAE", "ORYSA") in calls
    assert np.array_equal(R["ORYSA"], np.eye(D)) and np.array_equal(R["ARATH"], np.eye(D))
    assert np.allclose(rotate(X["TRIAE"], R["TRIAE"]), X["ARATH"], atol=1e-8)


def test_min_anchors_is_never_below_the_two_rows_a_solve_needs():
    rng = np.random.default_rng(16)
    Z, Q, X = _latent_species(rng)

    def anchors(s, r):
        k = 1 if s == "TRIAE" else 100
        return X[s][:k], X[r][:k]

    R = two_stage_rotations(anchors, seeds=["ARATH", "ORYSA"], reference="ARATH",
                            seed_of={"TRIAE": "ORYSA"}, min_anchors=0)
    assert "TRIAE" not in R and "ORYSA" in R


def test_iter_two_stage_yields_in_processing_order_with_none_for_unplaced_species():
    rng = np.random.default_rng(17)
    Z, Q, X = _latent_species(rng)

    def anchors(s, r):
        k = 1 if s == "MARPO" else 100
        return X[s][:k], X[r][:k]

    out = list(iter_two_stage_rotations(anchors, seeds=["ARATH", "ORYSA", "MARPO"], reference="ARATH",
                                        seed_of={"ABROB": "MARPO", "TRIAE": "ORYSA"}))
    assert [s for s, _ in out] == ["ORYSA", "MARPO", "ABROB", "TRIAE", "ARATH"]
    got = dict(out)
    assert got["MARPO"] is None and got["ABROB"] is None  # too few anchors; depends on MARPO
    assert got["TRIAE"] is not None and np.array_equal(got["ARATH"], np.eye(D))
