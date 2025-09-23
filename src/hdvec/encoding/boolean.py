"""Boolean encoders and logical operators built on FPE bases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# core codebook
from ..core.codebook import Codebook
from ..utils import ensure_array
from .fpe import encode_boolean, make_circular_base

__all__ = [
    "BooleanEncoder",
    "logic_not",
    "logic_and",
    "logic_or",
    "logic_xor",
    "logic_not_vector",
    "logic_and_vector",
    "logic_or_vector",
    "logic_xor_vector",
    "apply_truth_table",
]


@dataclass
class BooleanEncoder:
    """Encodes Booleans using L=2 circular bases."""

    D: int
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.base = make_circular_base(self.D, 2, rng=self.rng)
        self.zero = np.ones(self.D, dtype=np.complex64)
        self.one = self.base

    def encode(self, bit: int) -> np.ndarray:
        if bit not in (0, 1):
            raise ValueError("BooleanEncoder expects bit in {0,1}")
        return encode_boolean(bit, self.base)

    def decode(self, vec: np.ndarray) -> int:
        arr = ensure_array(vec)
        sims = [self._similarity(arr, self.zero), self._similarity(arr, self.one)]
        return int(np.argmax(sims))

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float((np.conj(a) * b).mean().real)


def logic_not(vec: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    bit = encoder.decode(vec)
    return encoder.encode(1 - bit)


def logic_not_vector(vec: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    arr = ensure_array(vec)
    codebook = np.stack([encoder.zero, encoder.one], axis=0)
    idx, _ = Codebook(codebook).nearest(arr)
    return encoder.encode(1 - idx)


def _binary_logic(vec_a: np.ndarray, vec_b: np.ndarray, op: Callable[[int, int], int], encoder: BooleanEncoder) -> np.ndarray:
    a_bit = encoder.decode(vec_a)
    b_bit = encoder.decode(vec_b)
    return encoder.encode(op(a_bit, b_bit))


def logic_and(vec_a: np.ndarray, vec_b: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    return _binary_logic(vec_a, vec_b, lambda x, y: x & y, encoder)


def logic_and_vector(vec_a: np.ndarray, vec_b: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    return _binary_logic_vector(vec_a, vec_b, {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}, encoder)


def logic_or(vec_a: np.ndarray, vec_b: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    return _binary_logic(vec_a, vec_b, lambda x, y: x | y, encoder)


def logic_or_vector(vec_a: np.ndarray, vec_b: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    return _binary_logic_vector(vec_a, vec_b, {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}, encoder)


def logic_xor(vec_a: np.ndarray, vec_b: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    return _binary_logic(vec_a, vec_b, lambda x, y: x ^ y, encoder)


def logic_xor_vector(vec_a: np.ndarray, vec_b: np.ndarray, encoder: BooleanEncoder) -> np.ndarray:
    return _binary_logic_vector(vec_a, vec_b, {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}, encoder)


def apply_truth_table(inputs: tuple[int, int], table: dict[tuple[int, int], int]) -> int:
    return table[inputs]


def _binary_logic_vector(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    truth_table: dict[tuple[int, int], int],
    encoder: BooleanEncoder,
) -> np.ndarray:
    arr_a = ensure_array(vec_a)
    arr_b = ensure_array(vec_b)
    codebook = np.stack([encoder.zero, encoder.one], axis=0)
    idx_a, _ = Codebook(codebook).nearest(arr_a)
    idx_b, _ = Codebook(codebook).nearest(arr_b)
    return encoder.encode(truth_table[(int(idx_a), int(idx_b))])
