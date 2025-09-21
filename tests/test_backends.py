import numpy as np
import pytest

from hdvec.config import override, get_config, set_backend_name
from hdvec.core.backends import Backend, available_backends, get_backend, register_backend, set_backend
from hdvec.errors import ConfigurationError


def test_numpy_backend_registered():
    assert "numpy" in available_backends()
    backend = get_backend("numpy")
    assert backend.supports_complex64
    assert backend.array_module is np


def test_register_and_switch_backend():
    custom = Backend(name="custom", supports_complex64=False, supports_fft=False)
    register_backend(custom, overwrite=True)
    set_backend("custom")
    cfg = get_config()
    set_backend_name("custom")
    assert cfg.backend == "custom"
    set_backend("numpy")
    set_backend_name("numpy")


def test_override_context_manager_restores_state():
    cfg = get_config()
    original_backend = cfg.backend
    original_dim = cfg.D
    with override(backend="numpy", D=2048) as new_cfg:
        assert new_cfg.D == 2048
        assert new_cfg.backend == "numpy"
    assert get_config().D == original_dim
    assert get_config().backend == original_backend


def test_override_invalid_field():
    with pytest.raises(ConfigurationError):
        with override(nonexistent=123):
            pass


def test_set_backend_name_invalid():
    with pytest.raises(ConfigurationError):
        set_backend_name("unknown")
