import numpy as np

from hdvec.utils import ensure_array, l2_normalize, phase_normalize


def test_phase_normalize_complex():
    rng = np.random.default_rng(0)
    vec = rng.normal(size=8) + 1j * rng.normal(size=8)
    normed = phase_normalize(vec)
    assert np.allclose(np.abs(normed), 1.0, atol=1e-6)


def test_l2_normalize():
    vec = np.array([3.0, 4.0])
    normed = l2_normalize(vec)
    assert np.allclose(np.linalg.norm(normed), 1.0)


def test_ensure_array_passthrough():
    arr = np.ones(4)
    assert ensure_array(arr) is arr
