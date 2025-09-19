"""Generalized Holographic Reduced Representations (GHRR) stubs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


axtype = np.complex64


@dataclass
class GHVec:
    """Hypervector containing per-dimension (m x m) complex matrices.

    Attributes:
        data: Array of shape (D, m, m), complex64.
    """
    data: np.ndarray

    @property
    def D(self) -> int:
        return int(self.data.shape[0])

    @property
    def m(self) -> int:
        return int(self.data.shape[1])


def _unitary_qr(M: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(M)
    return Q.astype(axtype)


def sample_ghrr(D: int, m: int, policy: str = "haar", rng: np.random.Generator | None = None) -> GHVec:
    """Sample a GHVec with approximate Haar unitary slices per dimension."""
    if rng is None:
        rng = np.random.default_rng()
    mats = np.empty((D, m, m), dtype=axtype)
    for j in range(D):
        M = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        mats[j] = _unitary_qr(M)
    return GHVec(mats)


def gh_bind(a: GHVec, b: GHVec) -> GHVec:
    """GHRR binding: per-dimension matrix multiplication (non-commutative)."""
    if a.data.shape != b.data.shape:
        raise ValueError("Shape mismatch")
    D, m, _ = a.data.shape
    out = np.empty_like(a.data)
    for j in range(D):
        out[j] = a.data[j] @ b.data[j]
    return GHVec(out)


def gh_bundle(a: GHVec, b: GHVec) -> GHVec:
    """Bundle by simple average; re-orthonormalize via QR."""
    if a.data.shape != b.data.shape:
        raise ValueError("Shape mismatch")
    D, m, _ = a.data.shape
    out = np.empty_like(a.data)
    for j in range(D):
        out[j] = _unitary_qr((a.data[j] + b.data[j]) / 2)
    return GHVec(out)


def gh_similarity(a: GHVec, b: GHVec) -> float:
    """Similarity: (1/(mD)) * Re tr( sum_j a_j b_j^H )."""
    D, m, _ = a.data.shape
    s = 0.0
    for j in range(D):
        s += np.trace(a.data[j] @ b.data[j].conj().T).real
    return float(s / (m * D))
