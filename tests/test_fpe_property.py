from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hdvec.encoding.fpe import encode_fpe, generate_base
from hdvec.core import bind


@settings(max_examples=50)
@given(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_fpe_homomorphism_property(x: float, y: float) -> None:
    D = 128
    base = generate_base(D, rng=np.random.default_rng(0))
    zx = encode_fpe(x, base)
    zy = encode_fpe(y, base)
    zxy = encode_fpe(x + y, base)
    bound = bind(zx, zy, op="hadamard")
    assert np.allclose(np.asarray(bound), zxy, atol=1e-6)
