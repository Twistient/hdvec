import numpy as np

from hdvec.fpe import generate_base, encode_fpe
from hdvec.decoding import decode_point


def test_decode_point_picks_best_anchor():
    D = 128
    base = generate_base(D)
    anchors = np.stack(
        [encode_fpe(0.0, base), encode_fpe(1.0, base), encode_fpe(2.0, base)], axis=0
    )
    y = encode_fpe(1.0, base)
    idx = decode_point(y, anchors)
    assert idx == 1
