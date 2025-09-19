import numpy as np

from hdvec.core import bind, bundle, similarity, permute


def test_bind_hadamard_unit_modulus():
    rng = np.random.default_rng(0)
    D = 256
    a = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    c = bind(a, b, op="hadamard")
    assert np.allclose(np.abs(c), 1.0, atol=1e-6)


def test_similarity_quasi_orthogonal():
    rng = np.random.default_rng(0)
    D = 1024
    a = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    s = similarity(a, b)
    assert abs(s) < 0.2


def test_permute_rolls():
    v = np.array([1, 2, 3, 4])
    assert np.all(permute(v, 1) == np.array([4, 1, 2, 3]))


def test_bundle_shapes():
    a = np.ones(8)
    b = np.ones(8)
out = bundle(a, b)
    assert np.asarray(out).shape == a.shape
