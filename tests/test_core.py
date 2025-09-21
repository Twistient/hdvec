import numpy as np
import pytest

from hdvec.core import bind, bundle, similarity, permute
from hdvec.errors import BundlingModeError


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


def test_bundle_mean_normalization():
    a = np.ones(4)
    b = np.full(4, 3.0)
    out_mean = bundle(a, b, normalize="mean")
    assert np.allclose(np.asarray(out_mean), np.full(4, 2.0))


def test_bundle_none_normalization():
    a = np.arange(4, dtype=float)
    b = a + 1.0
    out_none = bundle(a, b, normalize="none")
    assert np.allclose(np.asarray(out_none), a + b)


def test_bundle_l2_normalization():
    rng = np.random.default_rng(0)
    vecs = [rng.normal(size=8) for _ in range(3)]
    out = bundle(*vecs, normalize="l2")
    assert np.allclose(np.linalg.norm(np.asarray(out)), 1.0, atol=1e-6)


def test_bundle_phasor_complex():
    rng = np.random.default_rng(1)
    vecs = [np.exp(1j * rng.uniform(-np.pi, np.pi, size=16)).astype(np.complex64) for _ in range(3)]
    out = bundle(*vecs, normalize="phasor")
    mags = np.abs(np.asarray(out))
    assert np.allclose(mags, 1.0, atol=1e-6)


def test_bundle_with_weights():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    out = bundle(a, b, weights=[0.25, 0.75], normalize="mean")
    expected = (0.25 * a + 0.75 * b) / (0.25 + 0.75)
    assert np.allclose(np.asarray(out), expected)


def test_bundle_invalid_mode_raises():
    with pytest.raises(BundlingModeError):
        bundle(np.ones(2), np.ones(2), normalize="invalid")
