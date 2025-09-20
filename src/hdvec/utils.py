"""Utilities for phase normalization, Hermitian enforcement, and noise injection.

This module also exposes an `optional_njit` decorator that applies Numba's njit when
Numba is available, or acts as a no-op otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypeVar, cast

import numpy as np

P = ParamSpec("P")
R = TypeVar("R")


class _NumbaFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


try:  # Optional JIT
    from numba import njit as _imported_njit
except Exception:  # pragma: no cover - fallback when numba is missing
    _numba_njit: _NumbaFactory | None = None
else:
    _numba_njit = cast(_NumbaFactory, _imported_njit)


def optional_njit(
    *njit_args: Any, **njit_kwargs: Any
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that applies numba.njit if available, else a no-op.

    Examples:
        @optional_njit(cache=True)
        def f(x):
            return x + 1
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if _numba_njit is None:
            return func
        decorated = _numba_njit(*njit_args, **njit_kwargs)(func)
        return decorated

    return decorator


def ensure_array(x: np.ndarray | Any) -> np.ndarray:
    """Return a NumPy ndarray from an input that may be a BaseVector-like object.

    If ``x`` is already a NumPy array, return it. Otherwise, if ``x`` has a
    ``.data`` attribute (e.g., a BaseVector), return that. This avoids using
    ``ndarray.data`` which is a buffer interface, not the array itself.
    """
    if isinstance(x, np.ndarray):
        return x
    data = getattr(x, "data", x)
    return np.asarray(data)


def phase_normalize(v: np.ndarray) -> np.ndarray:
    """Enforce unit-modulus: v <- exp(i * angle(v)).

    Works for real inputs by returning the same array.
    """
    if np.iscomplexobj(v):
        angles = np.angle(v)
        return np.exp(1j * angles).astype(np.complex64)
    return v


def hermitian_enforce(a: np.ndarray) -> np.ndarray:
    """Best-effort Hermitian/conjugate symmetry enforcement.

    If `a` is a 1D complex spectrum meant for a real-valued time domain, return `a` unchanged
    (a proper implementation would mirror bins). Placeholder for now.
    """
    return a


def inject_noise(
    v: np.ndarray, sigma: float, dist: str = "vonmises", rng: np.random.Generator | None = None
) -> np.ndarray:
    """Inject angular noise into a complex phasor vector.

    Args:
        v: Complex array of unit-modulus phasors.
        sigma: Concentration/scale of noise.
        dist: Distribution ("vonmises" or "normal").
        rng: Optional RNG.
    Returns:
        Noisy phasor vector with magnitudes ~1.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not np.iscomplexobj(v):
        return v
    if dist == "vonmises":
        noise = rng.vonmises(mu=0.0, kappa=1.0 / (sigma + 1e-12), size=v.shape)
    else:
        noise = rng.normal(loc=0.0, scale=sigma, size=v.shape)
    result = np.exp(1j * (np.angle(v) + noise)).astype(np.complex64)
    return cast(np.ndarray, result)
