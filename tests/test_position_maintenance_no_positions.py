import torch
import pytest

from ifera.policies import ScaledArtrMaintenancePolicy, PercentGainMaintenancePolicy
from ifera.data_models import DataManager
from ifera.state import State

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
        self.multiplier = torch.ones((1, 1), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backadjust = False

    def convert_indices(self, _base, date_idx, time_idx):
        return date_idx, time_idx


@pytest.fixture
def dummy_instrument_data(base_instrument_config):
    return DummyData(base_instrument_config)


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
    )

    state = State.create(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        start_date_idx=torch.tensor([0]),
        start_time_idx=torch.tensor([0]),
    )
    state.entry_price = torch.tensor([1.0])
    state.prev_stop_loss = torch.tensor([float("nan")])
    state.position = torch.tensor([0])
    state.base_price = torch.tensor([1.0])
    state.maint_stage = torch.tensor([1])
    state.entry_date_idx = torch.tensor([0])
    state.entry_time_idx = torch.tensor([0])

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
    )

    class DummyArtr(torch.nn.Module):
        def forward(self, state, action):
            return state.prev_stop_loss

    monkeypatch.setattr(policy, "artr_policy", DummyArtr())

    state = State.create(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        start_date_idx=torch.tensor([0]),
        start_time_idx=torch.tensor([0]),
    )
    state.prev_stop = torch.tensor([float("nan")])
    state.position = torch.tensor([0])
    state.entry_price = torch.tensor([1.0])
    state.maint_anchor = torch.tensor([1.5])
    state.maint_stage = torch.tensor([1])

    _, stop_loss = policy(state)
    assert torch.isnan(stop_loss).all()
