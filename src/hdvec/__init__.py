"""hdvec public API.

Minimal, NumPy-first stubs for hyperdimensional vector ops and encoders.
"""
from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover - very old Pythons
    PackageNotFoundError = Exception  # type: ignore
    def version(_: str) -> str:  # type: ignore
        return "0.0.0"

try:
    __version__ = version("hdvec")
except PackageNotFoundError:  # pragma: no cover - local editable import
    __version__ = "0.0.0"

from .base import BaseVector
from .config import Config, get_config
from .core import bind, bundle, permute, similarity
from .decoding import AnchorMemory, decode_function, decode_point, resonator_decode
from .fpe import FPEEncoder, encode_fpe, generate_base
from .ghrr import GHVec, gh_bind, gh_bundle, gh_similarity, sample_ghrr
from .residue import (
    ResidueEncoder,
    crt_reconstruct,
    encode_residue,
    res_add,
    res_mul,
)
from .vfa import VFAEncoder

__all__ = [
    "__version__",
    "bind",
    "bundle",
    "similarity",
    "permute",
    "Config",
    "get_config",
    "BaseVector",
    "FPEEncoder",
    "generate_base",
    "encode_fpe",
    "VFAEncoder",
    "GHVec",
    "sample_ghrr",
    "gh_bind",
    "gh_bundle",
    "gh_similarity",
    "ResidueEncoder",
    "encode_residue",
    "res_add",
    "res_mul",
    "crt_reconstruct",
    "AnchorMemory",
    "decode_point",
    "decode_function",
    "resonator_decode",
]
