import numpy as np

from hdvec.ghrr import sample_ghrr, gh_similarity, gh_bind


def test_ghrr_shapes_and_similarity():
    D, m = 8, 2
    a = sample_ghrr(D, m)
    b = sample_ghrr(D, m)
    s_self = gh_similarity(a, a)
    s_cross = gh_similarity(a, b)
    assert s_self > s_cross


def test_ghrr_bind_runs():
    a = sample_ghrr(4, 2)
    b = sample_ghrr(4, 2)
    c = gh_bind(a, b)
    assert c.data.shape == a.data.shape
