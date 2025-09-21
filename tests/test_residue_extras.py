import numpy as np

from hdvec.base import Vec
from hdvec.encoding.residue import (
    ResidueEncoder,
    ResidueBases,
    encode_residue,
    res_decode_int,
    res_pow_scalar,
)


def test_res_pow_scalar_matches_integer_multiplication():
    moduli = [3, 5]
    D = 256
    enc = ResidueEncoder(moduli=moduli, D=D)
    x = 4
    k = 2
    zx = enc(x)
    zx2 = res_pow_scalar(zx, k, enc.bases)
    zprod = enc(x * k)
    # Similarity should be high
    sim = (np.conj(np.asarray(zprod)) * np.asarray(zx2)).sum().real / D
    assert sim > 0.8


def test_residue_bases_reuse_codebooks():
    bases = ResidueBases.from_moduli([3, 5], D=64)
    first = bases.codebooks
    second = bases.codebooks
    assert first is second


def test_encode_residue_with_raw_bases():
    moduli = [3, 7]
    bases = ResidueBases.from_moduli(moduli, D=32)
    stack = bases.stack
    vec = encode_residue(5, stack, moduli)
    assert isinstance(vec, Vec)


def test_residue_decode_roundtrip():
    moduli = [3, 5, 7]
    bases = ResidueBases.from_moduli(moduli, D=96)
    value = 37
    encoded = encode_residue(value, bases)
    decoded = res_decode_int(encoded, bases)
    M = 1
    for m in moduli:
        M *= m
    assert decoded % M == value % M
