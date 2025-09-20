"""Generalized Holographic Reduced Representations (GHRR).

Implements a small GHRR surface using per-dimension unitary slices. Binding is
matrix multiplication per dimension (non-commutative); bundling averages and
re-orthonormalizes; similarity uses the average real trace inner product.

References: Plate (HRR), unitary operators; vector-valued operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

axtype = np.complex64


@dataclass
class GHVec:
    """Hypervector of per-dimension (m×m) complex matrices.

    Attributes
    ----------
    data : np.ndarray
        Array of shape ``(D, m, m)``, dtype complex64. Each slice ``data[j]`` is
        a unitary operator.

    Example
    -------
    >>> g1 = sample_ghrr(4, 2)
    >>> g2 = sample_ghrr(4, 2)
    >>> g3 = gh_bind(g1, g2)  # per-dimension matmul
    >>> s = gh_similarity(g1, g1)
    >>> s > gh_similarity(g1, g2)
    True
    """

    data: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.data.shape[0])

    @property
    def m(self) -> int:
        return int(self.data.shape[1])


def _unitary_qr(mat: np.ndarray) -> np.ndarray:
    """Return a unitary factor via QR (phase-normalized)."""
    q, _ = np.linalg.qr(mat)
    # Phase normalization: ensure determinant magnitude ~1 per slice
    return q.astype(axtype)


def sample_ghrr(
    d: int, m: int, policy: str = "haar", rng: np.random.Generator | None = None
) -> GHVec:
    """Sample a :class:`GHVec` with approximate Haar unitary slices per dimension.

    Parameters
    ----------
    d : int
        Number of dimensions (slices).
    m : int
        Per-slice matrix size (m×m).
    policy : str
        Currently only ``"haar"``; future policies may constrain structure.
    rng : np.random.Generator | None
        Optional RNG.
    """
    if rng is None:
        rng = np.random.default_rng()
    mats = np.empty((d, m, m), dtype=axtype)
    for j in range(d):
        mat = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        mats[j] = _unitary_qr(mat)
    return GHVec(mats)


def gh_bind(a: GHVec, b: GHVec) -> GHVec:
    """GHRR binding: per-dimension matrix multiplication (non-commutative).

    Uses batched ``@`` (matmul) for vectorization.
    """
    if a.data.shape != b.data.shape:
        raise ValueError("Shape mismatch")
    out = a.data @ b.data
    return GHVec(out.astype(axtype))


def gh_bundle(a: GHVec, b: GHVec) -> GHVec:
    """Bundle by simple average; re-orthonormalize via QR per slice."""
    if a.data.shape != b.data.shape:
        raise ValueError("Shape mismatch")
    mean = (a.data + b.data) / 2
    out = np.stack([_unitary_qr(mean[j]) for j in range(mean.shape[0])], axis=0)
    return GHVec(out.astype(axtype))


def gh_similarity(a: GHVec, b: GHVec) -> float:
    """Similarity: ``(1/(mD)) * Re tr( sum_j a_j b_j^H )``.

    Vectorized via einsum/trace for clarity and speed.
    """
    d, m, _ = a.data.shape
    prod = a.data @ b.data.conj().transpose(0, 2, 1)
    traces = np.trace(prod, axis1=1, axis2=2).real
    return float(traces.mean() / m)
