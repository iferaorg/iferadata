import torch
import pytest

from ifera.policies import ScaledArtrMaintenancePolicy
from ifera.data_models import DataManager
import tensordict as td

higher_ops_available = hasattr(torch, "cond") and hasattr(torch, "while_loop")
pytestmark = pytest.mark.skipif(
    not higher_ops_available, reason="torch._higher_order_ops not available"
)


class DummyData:
    def __init__(self, instrument):
        self.instrument = instrument
        self.data = torch.zeros((2, 2, 4), dtype=torch.float32)  # More data
        self.artr = torch.ones((2, 2), dtype=torch.float32)  # More artr data
        self.multiplier = torch.ones((2, 2), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backadjust = False
        self.artr_alpha = 0.3  # Missing attribute
        self.artr_acrossday = False  # Missing attribute

    def convert_indices(self, _base, date_idx, time_idx):
        return date_idx, time_idx

    def calculate_artr(self, alpha: float, acrossday: bool) -> torch.Tensor:
        """Mock calculate_artr method."""
        self.artr_alpha = alpha
        self.artr_acrossday = acrossday
        return self.artr

    @property
    def valid_mask(self):
        return torch.ones((2, 2), dtype=torch.bool)  # Larger valid mask


@pytest.fixture
def dummy_instrument_data(base_instrument_config):
    return DummyData(base_instrument_config)


def test_scaled_artr_cond_predicate_is_tensor(monkeypatch, dummy_instrument_data):
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

    def fake_cond(pred, true_fn, false_fn, operands):
        assert isinstance(pred, torch.Tensor)
        assert pred.dtype == torch.bool
        return false_fn(*operands)

    monkeypatch.setattr(torch, "cond", fake_cond)

    state = td.TensorDict({
        "date_idx": torch.tensor([0]),
        "time_idx": torch.tensor([0]),
        "entry_price": torch.tensor([1.0]),
        "prev_stop_loss": torch.tensor([0.5]),
        "position": torch.tensor([1]),
        "base_price": torch.tensor([1.0]),
        "maint_stage": torch.tensor([0]),
        "entry_date_idx": torch.tensor([0]),
        "entry_time_idx": torch.tensor([0]),
        "has_position_mask": torch.tensor([True]),  # Has position
        "action": torch.tensor([0]),  # Missing action field
    }, batch_size=1, device=torch.device("cpu"))

    policy(state)
