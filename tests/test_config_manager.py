import pytest

from ifera import ConfigManager


def test_get_base_instrument_config_allowed_interval(config_manager: ConfigManager):
    cfg = config_manager.get_base_instrument_config(symbol="CL", interval="30m")
    assert cfg.interval == "30m"
    assert cfg.parent_config is None


def test_get_base_instrument_config_derived_interval(config_manager: ConfigManager):
    cfg = config_manager.get_base_instrument_config(symbol="CL", interval="1h")
    assert cfg.interval == "1h"
    assert cfg.parent_config is not None
    assert cfg.parent_config.interval == "30m"


def test_get_base_instrument_config_invalid_interval(config_manager: ConfigManager):
    with pytest.raises(ValueError):
        config_manager.get_base_instrument_config(symbol="CL", interval="45m")
