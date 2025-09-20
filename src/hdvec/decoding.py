"""Decoding utilities: anchors, simple matching, and resonator stub."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import similarity


@dataclass
class AnchorMemory:
    anchors: np.ndarray  # (K, D)

    def query(self, y: np.ndarray) -> int:
        sims = [similarity(y, a) for a in self.anchors]
        return int(np.argmax(sims))


def decode_point(y: np.ndarray, anchors: np.ndarray, beta: float = 1.5) -> int:
    """Pick the index of the most similar anchor. `beta` unused in stub."""
    _ = beta
    mem = AnchorMemory(anchors)
    return mem.query(y)


def decode_function(y_f: np.ndarray, query: np.ndarray, iters: int = 10) -> np.ndarray:
    """Placeholder: return the query unchanged (no matching pursuit in stub)."""
    _ = iters
    return query


def resonator_decode(enc: np.ndarray, moduli: list[int]) -> dict:
    """Stub resonator decode: returns empty result."""
    _ = (enc, moduli)
    return {}
