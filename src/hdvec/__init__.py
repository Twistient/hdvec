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
from .base import BaseVector, Vec
from .config import Config, get_config
from .core import (
    Codebook,
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
from .decoding import AnchorMemory, decode_point
from .encoding.boolean import (
    BooleanEncoder,
    apply_truth_table,
    logic_and,
    logic_not,
    logic_or,
    logic_xor,
)
from .encoding.fpe import (
    FPEEncoder,
    encode_boolean,
    encode_circular,
    encode_fpe,
    encode_fpe_vec,
    generate_base,
    make_circular_base,
)
from .encoding.fpe_learned import LEPConfig, LEPModel, lep_encode, lep_init, lep_step
from .encoding.positional import Positional2DTorus, ResidueTorus
from .encoding.residue import (
    ResidueBases,
    ResidueEncoder,
    crt_reconstruct,
    encode_residue,
    residue_correlations,
    residue_initial_guess,
    res_add,
    res_mul_int,
    res_pow_scalar,
)
from .encoding.vfa import (
    VFAEncoder,
    convolve,
    encode_function,
    encode_grid,
    read_cell,
    readout,
    shift,
    translate_grid,
)
from .encoding.scene import FieldEncoder
from .ghrr import GHVec, gh_adj, gh_bind, gh_bundle, gh_commutativity, gh_similarity, gh_unbind, gh_project_unitary, sample_ghrr
from .permute import apply_perm, dihedral_permutations, roll

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
    "LEPConfig",
    "LEPModel",
    "lep_init",
    "lep_encode",
    "lep_step",
    "BooleanEncoder",
    "logic_not",
    "logic_and",
    "logic_or",
    "logic_xor",
    "apply_truth_table",
    "Positional2DTorus",
    "ResidueTorus",
    "VFAEncoder",
    "encode_function",
    "readout",
    "shift",
    "convolve",
    "encode_grid",
    "translate_grid",
    "read_cell",
    "FieldEncoder",
    "GHVec",
    "sample_ghrr",
    "gh_adj",
    "gh_bind",
    "gh_bundle",
    "gh_similarity",
    "gh_commutativity",
    "gh_unbind",
    "gh_project_unitary",
    "ResidueBases",
    "ResidueEncoder",
    "encode_residue",
    "residue_correlations",
    "residue_initial_guess",
    "res_add",
    "res_pow_scalar",
    "res_mul_int",
    "crt_reconstruct",
    "AnchorMemory",
    "decode_point",
    "roll",
    "dihedral_permutations",
    "apply_perm",
]
