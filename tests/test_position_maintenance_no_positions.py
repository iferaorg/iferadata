import torch
import pytest

from ifera.policies import ScaledArtrMaintenancePolicy, PercentGainMaintenancePolicy
from ifera.data_models import DataManager

torch._dynamo.config.capture_scalar_outputs = True

higher_ops_available = hasattr(torch, "cond") and hasattr(torch, "while_loop")
pytestmark = pytest.mark.skipif(
    not higher_ops_available, reason="torch._higher_order_ops not available"
)


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


@pytest.mark.xfail(reason="torch.while_loop cannot be captured in eager mode")
def test_scaled_artr_no_position(monkeypatch, dummy_instrument_data):
    def dummy_get(self, instrument_config, **_):
        return DummyData(instrument_config)

    monkeypatch.setattr(DataManager, "get_instrument_data", dummy_get)

    policy = ScaledArtrMaintenancePolicy(
        dummy_instrument_data,
        [dummy_instrument_data.instrument.interval, "1h"],
        atr_multiple=1.0,
        wait_for_breakeven=False,
        minimum_improvement=0.1,
        batch_size=1,
    )

    state = {
        "date_idx": torch.tensor([0]),
        "time_idx": torch.tensor([0]),
        "entry_price": torch.tensor([1.0]),
        "prev_stop_loss": torch.tensor([float("nan")]),
        "position": torch.tensor([0]),
        "base_price": torch.tensor([1.0]),
        "maint_stage": torch.tensor([1]),
    }

    _, stop_loss = policy(state)
    assert torch.isnan(stop_loss).all()


def test_percent_gain_no_position(dummy_instrument_data, monkeypatch):
    policy = PercentGainMaintenancePolicy(
        dummy_instrument_data,
        stage1_atr_multiple=1.0,
        trailing_stop=True,
        skip_stage1=False,
        keep_percent=0.5,
        anchor_type="entry",
        batch_size=1,
    )

    class DummyArtr(torch.nn.Module):
        def forward(self, date_idx, time_idx, position, action, prev_stop):
            _ = date_idx, time_idx, position, action
            return prev_stop

    monkeypatch.setattr(policy, "artr_policy", DummyArtr())

    state = {
        "prev_stop": torch.tensor([float("nan")]),
        "position": torch.tensor([0]),
        "entry_price": torch.tensor([1.0]),
        "date_idx": torch.tensor([0]),
        "time_idx": torch.tensor([0]),
        "maint_anchor": torch.tensor([1.5]),
        "maint_stage": torch.tensor([1]),
    }

    _, stop_loss = policy(state)
    assert torch.isnan(stop_loss).all()
