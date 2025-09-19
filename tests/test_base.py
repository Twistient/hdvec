import numpy as np

from hdvec.base import Vec


def test_vec_normalize_unit_modulus():
    rng = np.random.default_rng(0)
    D = 64
    data = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    v = Vec(data * 1.5)
    v.normalize()
    assert np.allclose(np.abs(v.data), 1.0, atol=1e-6)
