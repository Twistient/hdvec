"""Scene encoding using positional and value encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from ..core import bind, inv
from ..core.codebook import Codebook
from ..utils import ensure_array
from .positional import Positional2DTorus

__all__ = ["FieldEncoder"]


@dataclass
class FieldEncoder:
    positional: Positional2DTorus
    value_codebook: np.ndarray | None = None
    value_encoder: Callable[[float], np.ndarray] | None = None

    def encode_grid(self, grid: np.ndarray) -> np.ndarray:
        arr = ensure_array(grid)
        H, W = arr.shape[:2]
        pos_grid = self.positional.sample_grid(H, W)
        scene = np.zeros(self.positional.D, dtype=np.complex64)
        for i in range(H):
            for j in range(W):
                pos = pos_grid[i, j]
                val = self._encode_value(arr[i, j])
                scene += pos * val
        return scene.astype(np.complex64)

    def read_cell(self, scene: np.ndarray, i: int, j: int, size: tuple[int, int]) -> np.ndarray:
        H, W = size
        pos = self.positional.sample_grid(H, W)[i, j]
        probe = bind(scene, inv(pos))
        if self.value_codebook is not None:
            idx, score = Codebook(self.value_codebook).nearest(ensure_array(probe))
            return np.array([idx, score], dtype=float)
        return ensure_array(probe)

    def translate(self, scene: np.ndarray, dx: float, dy: float) -> np.ndarray:
        shift = self.positional.trans(dx, dy)
        return ensure_array(bind(scene, shift))

    def _encode_value(self, value: np.ndarray) -> np.ndarray:
        if self.value_codebook is not None:
            idx = int(value)
            return self.value_codebook[idx]
        if self.value_encoder is not None:
            return ensure_array(self.value_encoder(float(value)))
        return ensure_array(value)
