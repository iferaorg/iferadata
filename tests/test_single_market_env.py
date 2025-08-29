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
    def reset(self, state: dict[str, torch.Tensor]) -> None:
        _ = state
        return None

    def forward(self, state: dict[str, torch.Tensor], action: torch.Tensor):
        _ = state, action
        return torch.zeros_like(action, dtype=torch.float32)


class DummyMaintenance(PositionMaintenancePolicy):
    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        pass

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        _ = state
        return None

    def forward(self, state: dict[str, torch.Tensor]):
        position = state["position"]
        return torch.zeros_like(position), state["prev_stop_loss"]


class CloseAfterOneStep(PositionMaintenancePolicy):
    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        pass

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        _ = state
        return None

    def forward(self, state: dict[str, torch.Tensor]):
        time_idx = state["time_idx"]
        position = state["position"]
        action = torch.where(time_idx >= 1, -position, torch.zeros_like(position))
        return action, state["prev_stop_loss"]


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

    state = {
        "date_idx": d_idx,
        "time_idx": t_idx,
        "position": position,
        "prev_stop_loss": prev_stop,
        "entry_price": entry_price,
        "total_profit": torch.zeros(1),
        "done": torch.zeros(1, dtype=torch.bool),
    }

    _, _, done = policy(state)
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

    total_profit, d_idx, t_idx = env.rollout(
        trading_policy, start_d, start_t, max_steps=5
    )
    assert env.state["done"].item() is True
    assert env.state["position"].item() == 0
    assert total_profit.shape == (1,)
    assert d_idx.item() == 0
    assert t_idx.item() == 2


def test_single_market_env_rollout_returns_nan_if_never_done(
    monkeypatch, dummy_data_three_steps
):
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
        position_maintenance_policy=DummyMaintenance(),
        trading_done_policy=AlwaysFalseDonePolicy(
            batch_size=1, device=env.instrument_data.device
        ),
        batch_size=1,
    )

    start_d = torch.tensor([0], dtype=torch.int32)
    start_t = torch.tensor([0], dtype=torch.int32)

    total_profit, d_idx, t_idx = env.rollout(
        trading_policy, start_d, start_t, max_steps=2
    )

    assert env.state["done"].item() is False
    assert torch.isnan(d_idx).all()
    assert torch.isnan(t_idx).all()


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

        def reset(
            self, state: dict[str, torch.Tensor]
        ) -> None:  # pragma: no cover - simple flag
            self.reset_called = True
            self.last_mask = torch.ones(self.had_position.shape[0], dtype=torch.bool)
            super().reset(state)

        def masked_reset(
            self, mask: torch.Tensor
        ) -> None:  # pragma: no cover - simple flag
            self.reset_called = True
            self.last_mask = mask.clone()
            super().masked_reset(mask)

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


class OpenOnceWithStopLoss(PositionMaintenancePolicy):
    """Opens once, then does nothing. Provides a stop loss that gets hit immediately."""

    def __init__(self, stop_loss_price, batch_size=1, device=torch.device("cpu")):
        super().__init__()
        self.stop_loss_price = stop_loss_price
        self.register_buffer(
            "step_count", torch.zeros(batch_size, dtype=torch.int32, device=device)
        )

    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        self.step_count.fill_(0)

    def forward(self, state: dict[str, torch.Tensor]):
        position = state["position"]
        # Use step_count to determine if we should open (first step and no position)
        should_open = (self.step_count == 0) & (position == 0)
        self.step_count += 1

        action = torch.where(
            should_open, torch.ones_like(position), torch.zeros_like(position)
        )
        stop_loss = torch.where(
            should_open,
            torch.full_like(position, self.stop_loss_price, dtype=torch.float32),
            state["prev_stop_loss"],
        )
        return action, stop_loss


class DummyDataWithPriceGaps:
    """Dummy data that can trigger immediate stop losses."""

    def __init__(
        self,
        instrument,
        open_price=100.0,
        low_price=95.0,
        high_price=105.0,
        close_price=98.0,
    ):
        self.instrument = instrument
        # Create data with multiple time steps [open, high, low, close] that can trigger stop loss
        # First step: normal prices where stop loss will trigger
        # Second step: prices for next step
        self.data = torch.tensor(
            [
                [
                    [open_price, high_price, low_price, close_price],  # First time step
                    [
                        close_price,
                        close_price + 2,
                        close_price - 2,
                        close_price + 1,
                    ],  # Second time step
                ]
            ],
            dtype=torch.float32,
        )
        self.artr = torch.zeros((1, 2), dtype=torch.float32)
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


class DummyInitialStopLossWithValue(torch.nn.Module):
    def __init__(self, stop_loss_value):
        super().__init__()
        self.stop_loss_value = stop_loss_value

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        _ = state
        return None

    def forward(self, state: dict[str, torch.Tensor], action: torch.Tensor):
        _ = state
        return torch.full_like(action, self.stop_loss_value, dtype=torch.float32)


def test_single_market_env_entry_price_immediate_stop_loss(
    monkeypatch, base_instrument_config
):
    """Test that entry_price is set correctly even when position is immediately stopped out."""
    # Create data where low price (95) will trigger stop loss at 97
    dummy_data = DummyDataWithPriceGaps(
        base_instrument_config, open_price=100.0, low_price=95.0
    )

    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data,
    )

    env = SingleMarketEnv(dummy_data.instrument, "IBKR")

    # Policy that opens with stop loss at 97 (above low price of 95, so will trigger)
    trading_policy = TradingPolicy(
        instrument_data=env.instrument_data,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=1, device=env.instrument_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLossWithValue(stop_loss_value=97.0),
        position_maintenance_policy=OpenOnceWithStopLoss(
            stop_loss_price=97.0, batch_size=1, device=env.instrument_data.device
        ),
        trading_done_policy=SingleTradeDonePolicy(
            batch_size=1, device=env.instrument_data.device
        ),
        batch_size=1,
    )

    start_d = torch.tensor([0], dtype=torch.int32)
    start_t = torch.tensor([0], dtype=torch.int32)

    env.reset(start_d, start_t, trading_policy)

    # Take first step: should open position and immediately stop out
    step_result = env.step(trading_policy, env.state)

    # Bug: entry_price should be set to execution_price even though final position is 0
    assert not torch.isnan(
        step_result["entry_price"]
    ).item(), "entry_price should not be NaN even when immediately stopped out"
    assert (
        step_result["position"].item() == 0
    ), "position should be 0 after immediate stop out"

    # entry_price should be set to a reasonable value (not NaN)
    assert step_result["entry_price"].item() > 0, "entry_price should be positive"


def test_single_trade_done_policy_immediate_stop_loss(
    monkeypatch, base_instrument_config
):
    """Test that SingleTradeDonePolicy recognizes trades that are immediately stopped out."""
    # Create data where low price (95) will trigger stop loss at 97
    dummy_data = DummyDataWithPriceGaps(
        base_instrument_config, open_price=100.0, low_price=95.0
    )

    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data,
    )

    env = SingleMarketEnv(dummy_data.instrument, "IBKR")

    done_policy = SingleTradeDonePolicy(batch_size=1, device=env.instrument_data.device)

    # Policy that opens with stop loss at 97 (above low price of 95, so will trigger)
    trading_policy = TradingPolicy(
        instrument_data=env.instrument_data,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=1, device=env.instrument_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLossWithValue(stop_loss_value=97.0),
        position_maintenance_policy=OpenOnceWithStopLoss(
            stop_loss_price=97.0, batch_size=1, device=env.instrument_data.device
        ),
        trading_done_policy=done_policy,
        batch_size=1,
    )

    start_d = torch.tensor([0], dtype=torch.int32)
    start_t = torch.tensor([0], dtype=torch.int32)

    env.reset(start_d, start_t, trading_policy)

    # Initially, had_position should be False
    assert env.state["had_position"].item() is False

    # Take first step: should open position and immediately stop out
    step_result = env.step(trading_policy, env.state)

    # Bug: had_position should be True even though final position is 0
    assert (
        step_result["position"].item() == 0
    ), "position should be 0 after immediate stop out"
    assert (
        env.state["had_position"].item() is True
    ), "had_position should be True after opening a position, even if immediately stopped out"

    # Take another step with no action - done should be True since we had a position that returned to 0
    env.state.update(step_result)
    step_result2 = env.step(trading_policy, env.state)

    # Should signal done since position went from non-zero back to zero
    assert (
        step_result2["done"].item() is True
    ), "should signal done after position returned to zero"
