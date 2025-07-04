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


def test_get_base_instrument_config_contract_code_allowed(
    config_manager: ConfigManager,
):
    cfg = config_manager.get_base_instrument_config(
        symbol="CL", interval="30m", contract_code="M24"
    )
    assert cfg.contract_code == "M24"
    assert cfg.parent_config is not None
    assert cfg.parent_config.contract_code is None
    assert cfg.parent_config.interval == "30m"


def test_get_base_instrument_config_contract_code_and_interval(
    config_manager: ConfigManager,
):
    cfg = config_manager.get_base_instrument_config(
        symbol="CL", interval="1h", contract_code="M24"
    )
    assert cfg.interval == "1h"
    assert cfg.contract_code == "M24"
    assert cfg.parent_config is not None
    assert cfg.parent_config.interval == "30m"
    assert cfg.parent_config.contract_code is None


def test_get_config_with_contract_code(config_manager: ConfigManager):
    cfg = config_manager.get_config(
        broker_name="IBKR", symbol="CL", interval="30m", contract_code="M24"
    )
    assert cfg.contract_code == "M24"
    assert cfg.broker_symbol == "CLM24"
