"""Permutation utilities and registries.

Provides cyclic rolls and dihedral D4 permutations for square grids. These are
meant to be used as role/position permutations in HD encodings; applying them to
hypervectors requires agreeing on a grid shape.
"""

from __future__ import annotations

import numpy as np


def roll(v: np.ndarray, shift: int) -> np.ndarray:
    """Cyclic roll along the last axis."""
    return np.asarray(np.roll(v, shift, axis=-1))


def dihedral_permutations(n: int) -> dict[str, np.ndarray]:
    """Return index permutations for the dihedral D4 group on an n×n grid.

    The returned dict maps names to index arrays of length n*n that can be used
    to permute flattened n×n arrays. Names include: rot90, rot180, rot270,
    flipx, flipy, flipdiag, flipanti.
    """
    idx = np.arange(n * n)
    grid = idx.reshape(n, n)
    perms: dict[str, np.ndarray] = {}

    # Rotations
    perms["rot90"] = np.rot90(grid, k=1).ravel()
    perms["rot180"] = np.rot90(grid, k=2).ravel()
    perms["rot270"] = np.rot90(grid, k=3).ravel()

    # Flips
    perms["flipx"] = np.flipud(grid).ravel()
    perms["flipy"] = np.fliplr(grid).ravel()
    perms["flipdiag"] = np.transpose(grid).ravel()
    perms["flipanti"] = np.fliplr(np.transpose(grid)).ravel()
    return perms


def apply_perm(v: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply a permutation (index map) to a flattened array and reshape back.

    For convenience, if ``v`` is 1-D of length n*n, returns 1-D. If ``v`` is
    2-D shape (n,n), applies permutation to its flattened view and reshapes.
    """
    arr = np.asarray(v)
    if arr.ndim == 1:
        return np.asarray(arr[perm])
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size == perm.size:
        n = arr.shape[0]
        flat = arr.reshape(n * n)[perm]
        return np.asarray(flat.reshape(n, n))
    raise ValueError("v must be length n*n or shape (n,n) matching perm")
