"""Base vector abstraction for hdvec.

This provides a minimal ABC to enable future backend-agnostic vector objects
(e.g., NumPy arrays today, Torch tensors later) while allowing a transitional
phase where functions continue accepting/returning NumPy arrays.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from .utils import phase_normalize


class BaseVector(ABC):
    """Abstract base class for HD vectors.

    Implementations should wrap backend-native storage (e.g., np.ndarray) and
    maintain complex64 unit-modulus invariants for phasor-based encodings.
    """

    backend: Literal["numpy", "torch"] = "numpy"

    @property
    @abstractmethod
    def data(self) -> np.ndarray:  # pragma: no cover - interface only
        """Return the underlying array-like storage as a NumPy ndarray."""
        raise NotImplementedError

    @property
    def D(self) -> int:
        """Dimensionality inferred from the last axis of ``data``."""
        return int(self.data.shape[-1])

    @abstractmethod
    def normalize(self) -> "BaseVector":  # pragma: no cover - interface only
        """Return a normalized view/object respecting unit-modulus constraints."""
        raise NotImplementedError


class Vec(BaseVector):
    """Concrete 1-D vector wrapper used for tests/examples.

    Stores a NumPy ndarray and normalizes to unit-modulus if complex.
    """

    __slots__ = ("_data",)

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy ndarray")
        # Keep dtype; normalize below will cast complex to complex64
        self._data = data

    @property
    def data(self) -> np.ndarray:
        return self._data

    def normalize(self) -> "Vec":
        if np.iscomplexobj(self._data):
            self._data = phase_normalize(self._data)
        return self

    def __array__(self) -> np.ndarray:
        return self._data
