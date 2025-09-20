import numpy as np

from hdvec.ghrr import sample_ghrr, gh_similarity, gh_bind, gh_unbind, gh_project_unitary


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
    a_rec = gh_unbind(c, b)
    # Similarity after unbinding should exceed random similarity
    assert gh_similarity(a, a_rec) > gh_similarity(a, b)
    # Projection keeps slices unitary-like (orthonormal columns)
    proj = gh_project_unitary(c)
    for j in range(proj.dim):
        I = proj.data[j].conj().T @ proj.data[j]
        assert np.allclose(I, np.eye(proj.m), atol=1e-5)
