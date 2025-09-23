"""Positional encoders for toroidal grids."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..core import bind
from ..utils import ensure_array
from .fpe import encode_fpe_vec, generate_base
from .residue import ResidueBases, encode_residue

__all__ = ["Positional2DTorus", "ResidueTorus"]


@dataclass
class Positional2DTorus:
    D: int
    beta: float = 1.0
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        rng = self.rng or np.random.default_rng()
        self.base_x = generate_base(self.D, beta=self.beta, rng=rng)
        self.base_y = generate_base(self.D, beta=self.beta, rng=rng)

    def pos(self, x: float, y: float) -> np.ndarray:
        return encode_fpe_vec(np.array([x, y], dtype=float), [self.base_x, self.base_y])

    def trans(self, dx: float, dy: float) -> np.ndarray:
        return self.pos(dx, dy)

    def sample_grid(self, height: int, width: int) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, width, endpoint=False)
        ys = np.linspace(0.0, 1.0, height, endpoint=False)
        grid = np.empty((height, width, self.D), dtype=np.complex64)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                grid[i, j] = self.pos(x, y)
        return grid


@dataclass
class ResidueTorus(Positional2DTorus):
    moduli: Iterable[int] | None = None

    def __post_init__(self) -> None:
        if self.moduli is None:
            raise ValueError("ResidueTorus requires moduli")
        moduli_list = list(self.moduli)
        if not moduli_list:
            raise ValueError("ResidueTorus requires at least one modulus")
        self.moduli = moduli_list
        self.bases = ResidueBases.from_moduli(moduli_list, self.D)

    def pos(self, x: float, y: float) -> np.ndarray:
        mods = self.bases.moduli
        if len(mods) < 2:
            raise ValueError("ResidueTorus requires at least two moduli")
        idx_x = int(np.floor(x * mods[0])) % mods[0]
        idx_y = int(np.floor(y * mods[1])) % mods[1]
        code_x = encode_residue(idx_x, self.bases.stack[0][None, :], [mods[0]])
        code_y = encode_residue(idx_y, self.bases.stack[1][None, :], [mods[1]])
        return ensure_array(bind(code_x, code_y))
