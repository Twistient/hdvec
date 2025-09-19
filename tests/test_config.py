from hdvec.config import Config, get_config


def test_config_singleton_and_fields():
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2
    assert isinstance(cfg1.D, int) and cfg1.D > 0
    # Toggle binding and ensure attribute present
    cfg1.binding = 'cc'
    assert get_config().binding == 'cc'
