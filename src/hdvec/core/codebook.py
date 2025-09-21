"""Codebook utilities for associative cleanup operations."""

from __future__ import annotations

import numpy as np

from ..utils import ensure_array
from .metrics import nearest, nearest_batch, topk, topk_batch

__all__ = ["Codebook"]


class Codebook:
    """Lightweight codebook of atoms with nearest/top-k cleanup utilities."""

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
        return nearest_batch(q, self.atoms)

    def topk_batch(self, q: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        return topk_batch(q, self.atoms, k=k)
