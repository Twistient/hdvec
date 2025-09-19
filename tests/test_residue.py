import numpy as np

from hdvec.residue import ResidueEncoder, crt_reconstruct, res_add


def test_crt_small():
    moduli = [3, 5]
    # residues for x=11
    parts = np.array([11 % 3, 11 % 5])
    x = crt_reconstruct(parts, moduli)
    assert x == 11


def test_res_add_unit_modulus():
    rng = np.random.default_rng(0)
    D = 64
    a = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    c = res_add(a, b)
    assert np.allclose(np.abs(c), 1.0, atol=1e-6)
