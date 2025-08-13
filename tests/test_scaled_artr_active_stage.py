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
        self.backadjust = False

    def convert_indices(self, _base, date_idx, time_idx):
        return date_idx, time_idx


@pytest.fixture
def dummy_instrument_data(base_instrument_config):
    return DummyData(base_instrument_config)


@pytest.mark.xfail(reason="torch.compile not available for while_loop execution")
def test_scaled_artr_active_position(monkeypatch, dummy_instrument_data):
    def dummy_get(self, instrument_config, **_):
        return DummyData(instrument_config)

    monkeypatch.setattr(DataManager, "get_instrument_data", dummy_get)

    policy = ScaledArtrMaintenancePolicy(
        dummy_instrument_data,
        [dummy_instrument_data.instrument.interval, "1h"],
        atr_multiple=1.0,
        wait_for_breakeven=False,
        minimum_improvement=0.1,
        batch_size=2,
    )

    state = {
        "date_idx": torch.tensor([0, 0]),
        "time_idx": torch.tensor([0, 0]),
        "entry_price": torch.tensor([1.0, 1.0]),
        "prev_stop_loss": torch.tensor([0.5, 0.5]),
        "position": torch.tensor([1, 0]),
        "base_price": torch.tensor([1.0, 1.0]),
        "maint_stage": torch.tensor([1, 0]),
    }

    _, stop_loss = policy(state)
    assert stop_loss.shape == (2,)
