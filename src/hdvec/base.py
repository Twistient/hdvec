"""Base vector abstraction for hdvec.

This provides a minimal ABC to enable future backend-agnostic vector objects
(e.g., NumPy arrays today, Torch tensors later) while allowing a transitional
phase where functions continue accepting/returning NumPy arrays.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from .utils import phase_normalize

ArrayLike = np.ndarray


class BaseVector(ABC):
    """Abstract base for HD vectors.

    Wraps numpy arrays (default) or torch.Tensors (future optional backend).
    Enforces complex64 dtype and provides normalization hook.
    """

    __slots__ = ("data",)

    data: ArrayLike

    @abstractmethod
    def __init__(self, data: ArrayLike):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy ndarray in stubs")
        self.data = data

    def normalize(self) -> None:
        """Project to unit-modulus phasors when complex; no-op for real."""
        if np.iscomplexobj(self.data):
            self.data = phase_normalize(self.data)

    def __array__(self) -> np.ndarray:  # numpy interop
        return self.data


class Vec(BaseVector):
    """Concrete 1-D vector wrapper used for tests/examples."""

    def __init__(self, data: ArrayLike):
        super().__init__(data.astype(np.complex64) if np.iscomplexobj(data) else data)
        self.normalize()
