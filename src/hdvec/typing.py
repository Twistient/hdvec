"""Typing helpers for hdvec."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["VectorLike", "ArrayLike", "RNGLike"]


@runtime_checkable
class VectorLike(Protocol):
    data: np.ndarray


ArrayLike = np.ndarray

RNGLike = np.random.Generator
