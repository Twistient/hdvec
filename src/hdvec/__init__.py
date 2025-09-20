"""hdvec public API.

Minimal, NumPy-first stubs for hyperdimensional vector ops and encoders.
"""

from __future__ import annotations

# ruff: noqa: I001  # Import block ordering is maintained for readability of re-exports

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

# Re-exports for convenience (organized and deduplicated)
from .base import BaseVector
from .config import Config, get_config
from .core import (
    Codebook,
    Vec,
    bind,
    bundle,
    circ_corr,
    circ_conv,
    cosine,
    inv,
    nearest,
    permute,
    project_unitary,
    similarity,
    topk,
    unbind,
)
from .decoding import AnchorMemory, decode_function, decode_point, resonator_decode
from .fpe import (
    FPEEncoder,
    encode_boolean,
    encode_circular,
    encode_fpe,
    encode_fpe_vec,
    generate_base,
    make_circular_base,
)
from .ghrr import GHVec, gh_bind, gh_bundle, gh_similarity, sample_ghrr
from .permute import apply_perm, dihedral_permutations, roll
from .residue import (
    ResidueEncoder,
    crt_reconstruct,
    encode_residue,
    res_add,
    res_pow_scalar,
    res_mul_int,
)
from .vfa import (
    VFAEncoder,
    convolve,
    encode_function,
    encode_grid,
    read_cell,
    readout,
    shift,
    translate_grid,
)

__all__ = [
    "__version__",
    "Vec",
    "bind",
    "bundle",
    "similarity",
    "cosine",
    "permute",
    "inv",
    "unbind",
    "project_unitary",
    "topk",
    "nearest",
    "circ_conv",
    "circ_corr",
    "Codebook",
    "Config",
    "get_config",
    "BaseVector",
    "FPEEncoder",
    "generate_base",
    "encode_fpe",
    "encode_fpe_vec",
    "make_circular_base",
    "encode_circular",
    "encode_boolean",
    "VFAEncoder",
    "encode_function",
    "readout",
    "shift",
    "convolve",
    "encode_grid",
    "translate_grid",
    "read_cell",
    "GHVec",
    "sample_ghrr",
    "gh_bind",
    "gh_bundle",
    "gh_similarity",
    "ResidueEncoder",
    "encode_residue",
    "res_add",
    "res_pow_scalar",
    "res_mul_int",
    "crt_reconstruct",
    "AnchorMemory",
    "decode_point",
    "decode_function",
    "resonator_decode",
    "roll",
    "dihedral_permutations",
    "apply_perm",
]
