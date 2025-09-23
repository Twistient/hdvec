"""Learned Fractional Power Encoding (LEP)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..utils import phase_normalize

__all__ = ["LEPConfig", "LEPModel", "lep_init", "lep_encode", "lep_step"]


@dataclass
class LEPConfig:
    D: int
    family: Literal["gaussian", "laplace", "cauchy"] = "gaussian"
    beta: float = 1.0
    learn_beta: bool = False
    lr: float = 1e-2


@dataclass
class LEPModel:
    base: np.ndarray
    beta: float


def lep_init(cfg: LEPConfig, rng: np.random.Generator | None = None) -> LEPModel:
    if rng is None:
        rng = np.random.default_rng()
    phases = _sample_family(cfg.family, cfg.D, cfg.beta, rng)
    base = np.exp(1j * phases).astype(np.complex64)
    base = phase_normalize(base)
    return LEPModel(base=base, beta=cfg.beta)


def lep_encode(x: np.ndarray, model: LEPModel) -> np.ndarray:
    base = model.base
    z = np.power(base, x)
    return phase_normalize(z)


def lep_step(
    model: LEPModel,
    grads_fn: Callable[[LEPModel], tuple[np.ndarray, float]],
    cfg: LEPConfig,
) -> LEPModel:
    base = model.base
    base_grad, beta_grad = grads_fn(model)
    updated = base * np.exp(1j * cfg.lr * base_grad)
    updated = phase_normalize(updated)
    beta = model.beta
    if cfg.learn_beta:
        beta = max(1e-4, beta - cfg.lr * beta_grad)
    return LEPModel(base=updated, beta=beta)


def _sample_family(family: str, dim: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    if family == "gaussian":
        return rng.normal(0.0, beta, size=dim)
    if family == "laplace":
        return rng.laplace(0.0, beta, size=dim)
    if family == "cauchy":
        return np.arctan(rng.standard_cauchy(size=dim) * beta)
    raise ValueError(f"Unknown family: {family}")
