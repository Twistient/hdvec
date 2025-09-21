"""Decoding utilities (stateless).

This module intentionally contains only stateless helpers that do not implement
iterative dynamics. Runtime decoders (e.g., resonator networks) belong in a
downstream package (e.g., hologram-resonator).
"""

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


__all__ = ["AnchorMemory", "decode_point"]
