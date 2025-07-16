import torch
import pytest

from ifera.policies import (
    TradingPolicy,
    AlwaysOpenPolicy,
    AlwaysFalseDonePolicy,
    SingleTradeDonePolicy,
    PositionMaintenancePolicy,
)
from ifera.data_models import DataManager
from ifera.config import BaseInstrumentConfig
from ifera.environments import SingleMarketEnv


class DummyData:
    def __init__(self, instrument, steps=1):
        self.instrument = instrument
        self.data = torch.zeros((1, steps, 4), dtype=torch.float32)
        self.artr = torch.zeros((1, steps), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backadjust = False

    @property
    def valid_mask(self):
        return torch.ones((1, self.data.shape[1]), dtype=torch.bool)

    def get_next_indices(self, date_idx, time_idx):
        next_time = time_idx + 1
        next_date = date_idx + (next_time >= self.data.shape[1]).long()
        next_time = torch.where(
            next_time >= self.data.shape[1], torch.zeros_like(next_time), next_time
        )
        return next_date, next_time


class DummyInitialStopLoss(torch.nn.Module):
    def forward(self, date_idx, time_idx, position, action, prev_stop):
        _ = position, prev_stop
        return torch.zeros_like(action, dtype=torch.float32)


class DummyMaintenance(PositionMaintenancePolicy):
    def reset(self, mask: torch.Tensor) -> None:
        pass

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ):
        return torch.zeros_like(position), prev_stop


class CloseAfterOneStep(PositionMaintenancePolicy):
    def reset(self, mask: torch.Tensor) -> None:
        pass

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ):
        action = torch.where(time_idx >= 1, -position, torch.zeros_like(position))
        return action, prev_stop


@pytest.fixture
def dummy_data_last_bar(base_instrument_config: BaseInstrumentConfig):
    return DummyData(base_instrument_config, steps=1)


@pytest.fixture
def dummy_data_three_steps(base_instrument_config: BaseInstrumentConfig):
    return DummyData(base_instrument_config, steps=3)


def test_trading_policy_done_override(monkeypatch, dummy_data_last_bar):
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_last_bar,
    )

    policy = TradingPolicy(
        instrument_data=dummy_data_last_bar,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=1, device=dummy_data_last_bar.device
        ),
        initial_stop_loss_policy=DummyInitialStopLoss(),
        position_maintenance_policy=DummyMaintenance(),
        trading_done_policy=AlwaysFalseDonePolicy(
            batch_size=1, device=dummy_data_last_bar.device
        ),
        batch_size=1,
    )

    d_idx = torch.tensor([0], dtype=torch.int32)
    t_idx = torch.tensor([0], dtype=torch.int32)
    position = torch.tensor([0], dtype=torch.int32)
    prev_stop = torch.tensor([float("nan")])
    entry_price = torch.tensor([float("nan")])

    _, _, done = policy(d_idx, t_idx, position, prev_stop, entry_price)
    assert done.item() is True


def test_single_market_env_rollout(monkeypatch, dummy_data_three_steps):
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps,
    )

    env = SingleMarketEnv(dummy_data_three_steps.instrument, "IBKR")
    trading_policy = TradingPolicy(
        instrument_data=env.instrument_data,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=1, device=env.instrument_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLoss(),
        position_maintenance_policy=CloseAfterOneStep(),
        trading_done_policy=SingleTradeDonePolicy(
            batch_size=1, device=env.instrument_data.device
        ),
        batch_size=1,
    )

    start_d = torch.tensor([0], dtype=torch.int32)
    start_t = torch.tensor([0], dtype=torch.int32)

    result = env.rollout(trading_policy, start_d, start_t, max_steps=5)
    assert env.done.item() is True
    assert env.position.item() == 0
    assert result.shape == (1,)


def test_single_market_env_reset_calls_done_policy(monkeypatch, dummy_data_three_steps):
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps,
    )

    env = SingleMarketEnv(dummy_data_three_steps.instrument, "IBKR")

    class TrackingDonePolicy(SingleTradeDonePolicy):
        def __init__(self) -> None:
            super().__init__(batch_size=2, device=env.instrument_data.device)
            self.reset_called = False
            self.last_mask = torch.empty(0, dtype=torch.bool)

        def reset(self, mask: torch.Tensor) -> None:  # pragma: no cover - simple flag
            self.reset_called = True
            self.last_mask = mask.clone()
            super().reset(mask)

    done_policy = TrackingDonePolicy()
    trading_policy = TradingPolicy(
        instrument_data=env.instrument_data,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=2, device=env.instrument_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLoss(),
        position_maintenance_policy=DummyMaintenance(),
        trading_done_policy=done_policy,
        batch_size=2,
    )

    start_d = torch.tensor([0, 0], dtype=torch.int32)
    start_t = torch.tensor([0, 0], dtype=torch.int32)

    env.reset(start_d, start_t, trading_policy)

    assert done_policy.reset_called is True
    assert done_policy.last_mask.shape[0] == start_d.shape[0]
