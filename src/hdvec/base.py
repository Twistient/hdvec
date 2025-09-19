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
