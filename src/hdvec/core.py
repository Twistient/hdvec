"""Core VSA primitives (superposition, binding, similarity, permutation).

Minimal NumPy implementations with optional Numba JIT. This module provides the
canonical operations of the complex phasor VSA/HRR/FHRR algebra used throughout
the library. All functions accept either ``np.ndarray`` or a ``BaseVector`` and
return a ``Vec`` (wrapper around a NumPy array) where appropriate.
"""

from __future__ import annotations

import numpy as np

from .base import BaseVector, Vec
from .utils import ensure_array, optional_njit, phase_normalize


@optional_njit(cache=True)
def _similarity_numba(a: np.ndarray, b: np.ndarray) -> float:
    # Real part of normalized inner product
    denom = max(1, a.size)
    # Note: Numba supports complex operations; cast to float for return
    val = (a.conj() * b).sum().real / denom
    return float(val)


def bind(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector, op: str = "hadamard") -> Vec:
    """Vector binding operation, configurable per VSA type.

    Supports:
      - hadamard: componentwise multiplication (phasor add)
      - cc: circular convolution via FFT
      - lcc: placeholder (raises NotImplementedError)

    Note: This transitional signature accepts BaseVector but returns a NumPy array.
    """
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    if op == "hadamard":
        out = a_arr * b_arr
        out = phase_normalize(out) if np.iscomplexobj(out) else out
        return Vec(out)
    if op == "cc":
        fa = np.fft.fft(a_arr)
        fb = np.fft.fft(b_arr)
        out = np.fft.ifft(fa * fb)
        out = (
            out.astype(np.complex64)
            if np.iscomplexobj(a_arr) or np.iscomplexobj(b_arr)
            else out.real
        )
        return Vec(out)
    if op == "lcc":
        raise NotImplementedError(
            "Localized circular convolution (lcc) is not implemented in stubs."
        )
    raise ValueError(f"Unknown binding op: {op}")


def inv(a: np.ndarray | BaseVector) -> Vec:
    """Elementwise inverse for phasor vectors.

    For complex phasors this is the complex conjugate; for real arrays this is
    an identity (no-op). Returns a Vec wrapper.
    """
    arr = ensure_array(a)
    if np.iscomplexobj(arr):
        return Vec(np.conj(arr))
    return Vec(arr)


def unbind(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec:
    """Unbind ``a`` with ``b`` via elementwise multiply with ``b``'s inverse.

    For phasors: ``unbind(a, b) = a ⊛ conj(b)``.
    """
    return bind(a, inv(b))


def bundle(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec:
    """Superposition (bundling) of two vectors with renormalization for phasors.

    Note: Accepts BaseVector inputs, returns a NumPy array (transitional API).
    """
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    out = a_arr + b_arr
    out = phase_normalize(out) if np.iscomplexobj(out) else out / 2.0
    return Vec(out)


def similarity(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> float:
    """Real part of normalized inner product ⟨a,b⟩/D.

    Returns a scalar in approximately [-1, 1] for unit-normalized phasors.

    Note: Accepts BaseVector inputs (transitional API).
    """
    a_arr = ensure_array(a).astype(np.complex64)
    b_arr = ensure_array(b).astype(np.complex64)
    # Use JIT when available
    return _similarity_numba(a_arr, b_arr)


def cosine(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> float:
    """Alias for :func:`similarity`.

    Returns the real part of the normalized inner product.
    """
    return similarity(a, b)


def permute(v: np.ndarray | BaseVector, shift: int) -> Vec:
    """Permutation as circular shift (roll) along the last axis.

    Note: Accepts BaseVector input, returns NumPy array.
    """
    v_arr = ensure_array(v)
    return Vec(np.roll(v_arr, shift, axis=-1))


def project_unitary(v: np.ndarray | BaseVector) -> Vec:
    """Project a vector to unit modulus (phase-only) if complex; else L2-normalize.

    This helps arrest drift across long chains of bindings.
    """
    arr = ensure_array(v)
    if np.iscomplexobj(arr):
        angles = np.angle(arr)
        return Vec(np.exp(1j * angles).astype(np.complex64))
    norm = float(np.linalg.norm(arr))
    return Vec(arr if norm == 0.0 else (arr / norm).astype(arr.dtype))


def topk(query: np.ndarray, codebook: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and scores by cosine similarity against ``codebook``.

    Args:
        query: (D,) array.
        codebook: (K, D) array of atoms.
        k: number of neighbors to return.
    Returns:
        (indices, scores) arrays of shape (k,).
    """
    q = ensure_array(query).astype(np.complex64 if np.iscomplexobj(codebook) else np.float32)
    cb = ensure_array(codebook)
    # Normalize
    denom = float(np.linalg.norm(q))
    qn = q / max(1e-12, denom)
    cn = cb / np.maximum(1e-12, np.linalg.norm(cb, axis=1, keepdims=True))
    sims = (np.conj(cn) * qn).sum(axis=1).real
    idx = np.argpartition(-sims, kth=min(k, sims.size - 1))[:k]
    order = np.argsort(-sims[idx])
    idx = idx[order]
    return idx, sims[idx]


def nearest(query: np.ndarray, codebook: np.ndarray) -> tuple[int, float]:
    """Return index and cosine score of the nearest atom in the codebook."""
    idxs, scores = topk(query, codebook, k=1)
    return int(idxs[0]), float(scores[0])


def topk_batch(
    queries: np.ndarray, codebook: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Batch top‑k cosine nearest neighbors.

    Args:
        queries: (N, D) array of query vectors.
        codebook: (K, D) array of atoms.
        k: number of neighbors.
    Returns:
        (indices, scores) each of shape (N, k).

    Example:
        >>> import numpy as np
        >>> Q = np.eye(3, dtype=np.complex64)
        >>> C = np.eye(3, dtype=np.complex64)
        >>> idx, sc = topk_batch(Q, C, k=1)
        >>> idx.ravel().tolist()
        [0, 1, 2]
    """
    q = ensure_array(queries)
    c = ensure_array(codebook)
    # Normalize rows
    qn = q / np.maximum(1e-12, np.linalg.norm(q, axis=1, keepdims=True))
    cn = c / np.maximum(1e-12, np.linalg.norm(c, axis=1, keepdims=True))
    # Cosine similarities: real part of inner product
    sims = (qn.conj() @ cn.T).real  # (N, K)
    # Argpartition per row
    part = np.argpartition(-sims, kth=np.minimum(k, sims.shape[1] - 1), axis=1)[:, :k]
    # Sort top-k per row
    row_indices = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row_indices, part], axis=1)
    top_idx = part[row_indices, order]
    top_scores = sims[row_indices, top_idx]
    return top_idx, top_scores


def nearest_batch(queries: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batch nearest neighbor by cosine.

    Returns arrays of shape (N,) for indices and scores.
    """
    idx, sc = topk_batch(queries, codebook, k=1)
    return idx[:, 0], sc[:, 0]


def circ_conv(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec:
    """Circular convolution via FFT (HRR form)."""
    aa = ensure_array(a)
    bb = ensure_array(b)
    fa = np.fft.fft(aa)
    fb = np.fft.fft(bb)
    out = np.fft.ifft(fa * fb)
    return Vec(
        out.astype(np.complex64) if (np.iscomplexobj(aa) or np.iscomplexobj(bb)) else out.real
    )


def circ_corr(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> Vec:
    """Circular correlation via FFT (inverse of circular convolution by ``b``).

    If ``c = circ_conv(a, b)``, then approximately ``a ≈ circ_corr(c, b)`` within
    numerical tolerance.
    """
    aa = ensure_array(a)
    bb = ensure_array(b)
    fa = np.fft.fft(aa)
    fb = np.fft.fft(bb)
    out = np.fft.ifft(fa * np.conj(fb))
    return Vec(
        out.astype(np.complex64) if (np.iscomplexobj(aa) or np.iscomplexobj(bb)) else out.real
    )


class Codebook:
    """Lightweight codebook of atoms with nearest/top-k cleanup utilities.

    Example
    -------
    >>> import numpy as np
    >>> from hdvec.core import Codebook
    >>> C = Codebook(np.eye(3, dtype=np.complex64))
    >>> C.nearest(np.array([1,0,0], dtype=np.complex64))
    (0, 1.0)
    >>> C.topk_batch(np.eye(3, dtype=np.complex64), k=1)[0].ravel().tolist()
    [0, 1, 2]
    """

    def __init__(self, atoms: np.ndarray, names: list[str] | None = None):
        if atoms.ndim != 2:
            raise ValueError("atoms must have shape (K, D)")
        self.atoms = atoms
        self.names = names

    def add(self, h: np.ndarray, name: str | None = None) -> None:
        h_arr = ensure_array(h)[None, :]
        self.atoms = np.concatenate([self.atoms, h_arr], axis=0)
        if self.names is not None:
            self.names.append(name or f"atom_{len(self.names)}")

    def nearest(self, q: np.ndarray) -> tuple[int, float]:
        return nearest(q, self.atoms)

    def topk(self, q: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        return topk(q, self.atoms, k=k)

    def nearest_batch(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch nearest neighbor by cosine.

        Returns arrays of shape (N,) for indices and scores.
        """
        return nearest_batch(q, self.atoms)

    def topk_batch(self, q: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Batch top‑k cosine nearest neighbors against this codebook."""
        return topk_batch(q, self.atoms, k=k)
