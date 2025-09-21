import numpy as np

from hdvec.ghrr import gh_adj, gh_bind, gh_bundle, gh_commutativity, gh_similarity, sample_ghrr


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


def test_ghrr_unbind_and_projection():
    a = sample_ghrr(3, 2)
    b = sample_ghrr(3, 2)
    c = gh_bind(a, b)
    a_rec = gh_bind(c, gh_bind(b, gh_adj(b)))
    assert a_rec.data.shape == a.data.shape
    proj = gh_bundle(a, b)
    assert proj.data.shape == a.data.shape


def test_commutativity_measure():
    a = sample_ghrr(3, 2, rng=np.random.default_rng(0))
    b = sample_ghrr(3, 2, rng=np.random.default_rng(1))
    comm = gh_commutativity(a, b)
    comm_rev = gh_commutativity(b, a)
    assert comm >= 0
    assert np.isclose(comm, comm_rev)
