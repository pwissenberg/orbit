"""ORBIT core: the closed-form orthogonal Procrustes rotation (paper Section 2.3, Eq. 1-3).

For anchor matrices A (source) and B (reference) with one ortholog correspondence per row,
the orthogonal R minimising ||A R - B||_F is R = U V^T with U S V^T = A^T B (Schoenemann
1966). R is an isometry: within-species distances stay unchanged, only the orientation moves.
Anchor weighting, iterative refinement and CSLS (the ablations of Fig. S2) are not part of
ORBIT and are not implemented here.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.linalg import svd


def procrustes_rotation(source: np.ndarray, reference: np.ndarray, *,
                        proper: bool = True) -> np.ndarray:
    """Return the orthogonal ``R`` (``d x d``, ``R^T R = I``) that minimises
    ``||source @ R - reference||_F``.

    ``source`` and ``reference`` are ``(k, d)`` anchor matrices with paired rows. The
    computation is carried out in float64 regardless of the input dtype.

    With ``proper=True`` (the default, used for the plant coexpression networks) a
    reflection is corrected to a proper rotation (``det R = +1``) by flipping the sign of
    the last singular direction. With ``proper=False`` the unconstrained Schoenemann
    solution ``R = U V^T`` is returned, which may be a reflection (``det R = -1``); this is
    exactly ``scipy.linalg.orthogonal_procrustes(source, reference)[0]``, the call that
    produced the released STRING alignment. Either way ``R`` is an isometry, so
    within-species distances and cosine similarities are preserved exactly.
    """
    A = np.asarray(source, dtype=np.float64)
    B = np.asarray(reference, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("anchor matrices must be two-dimensional")
    if A.shape != B.shape:
        raise ValueError(f"anchor matrices must have the same shape, got {A.shape} and {B.shape}")
    if A.shape[0] < 2:
        raise ValueError(f"at least two anchor pairs are needed, got {A.shape[0]}")

    U, _, Vt = svd(A.T @ B)  # SciPy's SVD; NumPy for the products
    if proper and np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def rotate(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply rotation ``R`` to every row of ``X``; the result has the dtype of ``X``."""
    X = np.asarray(X)
    return (X.astype(np.float64) @ np.asarray(R, dtype=np.float64)).astype(X.dtype, copy=False)


def is_rotation(R: np.ndarray, atol: float = 1e-6, *, proper: bool = True) -> bool:
    """True if ``R`` is square and orthogonal to ``atol``; with ``proper`` (default) the
    determinant must also be +1, otherwise +1 or -1 are both accepted."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    d = R.shape[0]
    det = np.linalg.det(R)
    det_ok = abs(det - 1.0) < 1e-3 if proper else abs(abs(det) - 1.0) < 1e-3
    return bool(np.allclose(R.T @ R, np.eye(d), atol=atol) and det_ok)


def rotation_from_alignment(raw: np.ndarray, aligned: np.ndarray, *,
                            proper: bool = True) -> np.ndarray:
    """Recover the orthogonal map that takes ``raw`` onto ``aligned`` when
    ``aligned = raw @ R``.

    Every gene is an exact correspondence, so this is a Procrustes fit with the full
    embedding as anchors. Useful to check that a released aligned file is a pure isometry of
    its Node2Vec input. Pass ``proper=False`` when the released alignment may contain
    reflections (STRING).
    """
    return procrustes_rotation(raw, aligned, proper=proper)


def iter_two_stage_rotations(anchors, seeds, reference, seed_of, *, proper: bool = True,
                             min_anchors: int = 2, seeds_aligned: bool = False,
                             ) -> Iterator[tuple[str, np.ndarray | None]]:
    """Solve the two-stage ORBIT construction (paper Sections 2.3 and 2.5, Fig. 1) species by
    species and yield ``(species, R)`` as each rotation is ready.

    ``anchors(source, target)`` must return raw-frame anchor matrices ``(A, B)`` with paired
    rows for the two species. Stage 1 rotates every seed onto ``reference``; stage 2 rotates
    every non-seed in ``seed_of`` onto its assigned seed *after* that seed has been rotated, so
    every species ends up in the reference frame by transitivity. The order of the yields is
    the seeds other than the reference, then the non-seeds in the order of ``seed_of``, then
    the reference with the identity. A consumer can therefore write and discard one species at
    a time instead of holding every embedding in memory.

    ``R`` is ``None`` for a species that could not be placed: fewer than ``min_anchors`` anchor
    rows (never fewer than 2, the minimum of the solve) or a non-seed whose seed was not placed.

    ``seeds_aligned=True`` declares that the seed embeddings already share the reference frame
    (for example the released files): stage 1 is skipped, a non-seed is rotated straight onto
    its seed's coordinates, and every seed is finally yielded with the identity.
    ``proper=False`` allows reflections (see :func:`procrustes_rotation`).
    """
    seeds = list(seeds)
    if reference not in seeds:
        raise ValueError(f"reference {reference!r} is not one of the seeds {seeds}")
    unknown = {s for s in seed_of.values() if s not in seeds}
    if unknown:
        raise ValueError(f"non-seeds are assigned to unknown seeds {sorted(unknown)}")
    min_anchors = max(2, min_anchors)

    placed: dict[str, np.ndarray] = {}
    dim: int | None = None

    def solve(source, target, R_target):
        nonlocal dim
        A, B = anchors(source, target)
        A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
        dim = A.shape[1]
        if len(A) < min_anchors:
            return None
        if R_target is not None:
            B = B @ R_target
        return procrustes_rotation(A, B, proper=proper)

    if not seeds_aligned:
        for seed in seeds:
            if seed == reference:
                continue
            R = solve(seed, reference, None)
            if R is not None:
                placed[seed] = R
            yield seed, R
    for species, seed in seed_of.items():
        if seed == reference or seeds_aligned:
            R_target = None
        elif seed in placed:
            R_target = placed[seed]
        else:
            yield species, None  # its seed could not be placed
            continue
        yield species, solve(species, seed, R_target)
    if dim is None:
        raise ValueError("nothing to align: " + ("no non-seed to place onto the aligned seeds" if seeds_aligned
                                                 else "no seed other than the reference and no non-seed"))
    yield reference, np.eye(dim)
    if seeds_aligned:
        for seed in seeds:
            if seed != reference:
                yield seed, np.eye(dim)


def two_stage_rotations(anchors, seeds, reference, seed_of, *, proper: bool = True,
                        min_anchors: int = 2, seeds_aligned: bool = False) -> dict[str, np.ndarray]:
    """The rotations of the two-stage construction as ``{species: R}``; species that could
    not be placed are omitted. See :func:`iter_two_stage_rotations` for the arguments."""
    return {s: R for s, R in iter_two_stage_rotations(anchors, seeds, reference, seed_of, proper=proper,
                                                      min_anchors=min_anchors, seeds_aligned=seeds_aligned)
            if R is not None}
