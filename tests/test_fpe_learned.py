import numpy as np

from hdvec.encoding import fpe_learned as lep


def test_lep_init_and_encode():
    cfg = lep.LEPConfig(D=64)
    model = lep.lep_init(cfg, rng=np.random.default_rng(0))
    x = np.array(0.5)
    z = lep.lep_encode(x, model)
    assert z.shape == model.base.shape
    assert np.allclose(np.abs(z), 1.0, atol=1e-6)


def test_lep_step_gradient():
    cfg = lep.LEPConfig(D=32, learn_beta=True)
    model = lep.lep_init(cfg, rng=np.random.default_rng(1))

    def grads_fn(m: lep.LEPModel) -> tuple[np.ndarray, float]:
        grad = np.imag(m.base)
        beta_grad = 0.1
        return grad, beta_grad

    updated = lep.lep_step(model, grads_fn, cfg)
    assert updated.base.shape == model.base.shape
    assert updated.beta != model.beta
