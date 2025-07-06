import torch
import pytest

from ifera.policies import ScaledArtrMaintenancePolicy
from ifera.data_models import DataManager


class DummyData:
    def __init__(self, instrument):
        self.instrument = instrument
        self.data = torch.zeros((1, 1, 4), dtype=torch.float32)
        self.artr = torch.zeros((1, 1), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def convert_indices(self, *_args):
        return _args


@pytest.fixture
def dummy_instrument_data(base_instrument_config):
    return DummyData(base_instrument_config)


def test_scaled_artr_initialization(monkeypatch, dummy_instrument_data):
    def dummy_get(self, instrument_config, **_):
        return DummyData(instrument_config)

    monkeypatch.setattr(DataManager, "get_instrument_data", dummy_get)

    policy = ScaledArtrMaintenancePolicy(
        dummy_instrument_data,
        [dummy_instrument_data.instrument.interval, "1h"],
        atr_multiple=1.0,
        wait_for_breakeven=False,
        minimum_improvement=0.1,
    )

    assert policy.stage_count == 2
    assert [cfg.interval for cfg in policy.derived_configs] == [
        dummy_instrument_data.instrument.interval,
        "1h",
    ]


def test_scaled_artr_invalid_base(monkeypatch, dummy_instrument_data):
    monkeypatch.setattr(
        DataManager, "get_instrument_data", lambda self, config, **_: DummyData(config)
    )

    with pytest.raises(ValueError):
        ScaledArtrMaintenancePolicy(
            dummy_instrument_data,
            ["1h", "2h"],
            atr_multiple=1.0,
            wait_for_breakeven=False,
            minimum_improvement=0.1,
        )
