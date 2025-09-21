"""Similarity and retrieval metrics for hypervectors."""

from __future__ import annotations

import numpy as np

from ..base import BaseVector
from ..utils import ensure_array, optional_njit

__all__ = [
    "similarity",
    "cosine",
    "topk",
    "nearest",
    "topk_batch",
    "nearest_batch",
]


@optional_njit(cache=True)
def _similarity_numba(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(1, a.size)
    val = (a.conj() * b).sum().real / denom
    return float(val)


def similarity(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> float:
    """Return the real part of the normalized inner product."""
    a_arr = ensure_array(a).astype(np.complex64)
    b_arr = ensure_array(b).astype(np.complex64)
    return _similarity_numba(a_arr, b_arr)


def cosine(a: np.ndarray | BaseVector, b: np.ndarray | BaseVector) -> float:
    """Alias for :func:`similarity`."""
    return similarity(a, b)


def topk(query: np.ndarray, codebook: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and scores using cosine similarity."""
    q = ensure_array(query).astype(np.complex64 if np.iscomplexobj(codebook) else np.float32)
    cb = ensure_array(codebook)
    denom = float(np.linalg.norm(q))
    qn = q / max(1e-12, denom)
    cn = cb / np.maximum(1e-12, np.linalg.norm(cb, axis=1, keepdims=True))
    sims = (np.conj(cn) * qn).sum(axis=1).real
    idx = np.argpartition(-sims, kth=min(k, sims.size - 1))[:k]
    order = np.argsort(-sims[idx])
    idx = idx[order]
    return idx, sims[idx]


def nearest(query: np.ndarray, codebook: np.ndarray) -> tuple[int, float]:
    idxs, scores = topk(query, codebook, k=1)
    return int(idxs[0]), float(scores[0])


def topk_batch(
    queries: np.ndarray,
    codebook: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and scores for each row in ``queries``."""
    q = ensure_array(queries)
    c = ensure_array(codebook)
    qn = q / np.maximum(1e-12, np.linalg.norm(q, axis=1, keepdims=True))
    cn = c / np.maximum(1e-12, np.linalg.norm(c, axis=1, keepdims=True))
    sims = (qn.conj() @ cn.T).real
    part = np.argpartition(-sims, kth=np.minimum(k, sims.shape[1] - 1), axis=1)[:, :k]
    row_indices = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row_indices, part], axis=1)
    top_idx = part[row_indices, order]
    top_scores = sims[row_indices, top_idx]
    return top_idx, top_scores


def nearest_batch(queries: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx, sc = topk_batch(queries, codebook, k=1)
    return idx[:, 0], sc[:, 0]
