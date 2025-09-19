"""hdvec public API.

Minimal, NumPy-first stubs for hyperdimensional vector ops and encoders.
"""
from .core import bind, bundle, similarity, permute, Config
from .fpe import FPEEncoder, generate_base, encode_fpe
from .vfa import VFAEncoder
from .ghrr import GHVec, sample_ghrr, gh_bind, gh_bundle, gh_similarity
from .residue import ResidueEncoder, encode_residue, res_add, res_mul, crt_reconstruct
from .decoding import AnchorMemory, decode_point, decode_function, resonator_decode

__all__ = [
    "bind",
    "bundle",
    "similarity",
    "permute",
    "Config",
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
