"""Custom exceptions for hdvec."""

from __future__ import annotations

__all__ = [
    "HDVecError",
    "ConfigurationError",
    "InvalidBindingError",
    "BundlingModeError",
    "ShapeMismatchError",
]


class HDVecError(Exception):
    """Base class for hdvec-specific exceptions."""


class ConfigurationError(HDVecError):
    """Raised when global configuration is invalid for an operation."""


class InvalidBindingError(HDVecError):
    """Raised when a binding operator is not supported."""


class BundlingModeError(HDVecError):
    """Raised when an unknown bundling normalization mode is requested."""


class ShapeMismatchError(HDVecError):
    """Raised when operands with incompatible shapes are provided."""
