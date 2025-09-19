import numpy as np
import pytest


@pytest.fixture
def rand_vec() -> np.ndarray:
    rng = np.random.default_rng(0)
    D = 128
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
