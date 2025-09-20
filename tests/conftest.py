from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def rand_vec() -> NDArray[np.complex64]:
    rng = np.random.default_rng(0)
    D = 128
    vec = np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)).astype(np.complex64)
    return cast(NDArray[np.complex64], vec)
