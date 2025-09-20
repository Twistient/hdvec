import numpy as np

from hdvec.core import nearest_batch, topk_batch


def test_topk_batch_identity():
    dim, k = 16, 8
    C = np.eye(k, dim, dtype=np.complex64)
    Q = C.copy()
    idx, sc = topk_batch(Q, C, k=1)
    assert np.all(idx.ravel() == np.arange(k))
    i2, s2 = nearest_batch(Q, C)
    assert np.all(i2 == np.arange(k))
