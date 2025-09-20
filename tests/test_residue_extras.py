import numpy as np

from hdvec.residue import ResidueEncoder, encode_residue, res_decode_int, res_pow_scalar


def test_res_pow_scalar_matches_integer_multiplication():
    moduli = [3, 5]
    D = 256
    enc = ResidueEncoder(moduli=moduli, D=D)
    x = 4
    k = 2
    zx = enc(x)
    bases = np.stack(enc.bases, axis=0)
    zx2 = res_pow_scalar(zx, k, moduli, bases)
    zprod = enc(x * k)
    # Similarity should be high
    sim = (np.conj(np.asarray(zprod)) * np.asarray(zx2)).sum().real / D
    assert sim > 0.8
