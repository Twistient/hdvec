"""Core algebra package with binding, metrics, and codebook utilities."""

from .algebra import (
    bind,
    bundle,
    circ_conv,
    circ_corr,
    inv,
    permute,
    project_unitary,
    unbind,
)
from .codebook import Codebook
from .metrics import cosine, nearest, nearest_batch, similarity, topk, topk_batch

__all__ = [
    "bind",
    "bundle",
    "circ_conv",
    "circ_corr",
    "inv",
    "permute",
    "project_unitary",
    "unbind",
    "Codebook",
    "cosine",
    "nearest",
    "nearest_batch",
    "similarity",
    "topk",
    "topk_batch",
]
