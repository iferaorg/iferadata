import torch
import pytest

from ifera.data_models import DataManager
from ifera.config import BaseInstrumentConfig
from ifera.environments import MultiGPUSingleMarketEnv
from ifera.policies import (
    TradingPolicy,
    AlwaysOpenPolicy,
    SingleTradeDonePolicy,
    clone_trading_policy_for_devices,
)

from tests.test_single_market_env import (
    DummyData,
    DummyInitialStopLoss,
    CloseAfterOneStep,
)


@pytest.fixture
def dummy_data_three_steps_multi(base_instrument_config: BaseInstrumentConfig):
    return DummyData(base_instrument_config, steps=3)


def test_multi_gpu_env_rollout(monkeypatch, dummy_data_three_steps_multi):
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_multi,
    )

    devices = [torch.device("cpu"), torch.device("cpu")]
    env = MultiGPUSingleMarketEnv(
        dummy_data_three_steps_multi.instrument, "IBKR", devices=devices
    )

    base_env = env.envs[0]
    base_policy = TradingPolicy(
        instrument_data=base_env.instrument_data,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=1, device=base_env.instrument_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLoss(),
        position_maintenance_policy=CloseAfterOneStep(),
        trading_done_policy=SingleTradeDonePolicy(
            batch_size=1, device=base_env.instrument_data.device
        ),
        batch_size=1,
    )

    start_d = torch.tensor([0, 0], dtype=torch.int32)
    start_t = torch.tensor([0, 0], dtype=torch.int32)

    total_profit, total_profit_percent, d_idx, t_idx, steps = env.rollout(
        base_policy, start_d, start_t, max_steps=5
    )
    assert total_profit.shape == (2,)
    assert torch.all(d_idx == 0)
    assert torch.all(t_idx == 2)
    assert isinstance(steps, int)
    assert steps > 0
    # Note: Since we're using parallel execution, we can't directly check env.state
    # as the state updates happen in separate processes


def test_multi_gpu_env_parallel_chunking(monkeypatch, dummy_data_three_steps_multi):
    """Test that parallel implementation correctly chunks inputs across devices."""
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_multi,
    )

    devices = [torch.device("cpu"), torch.device("cpu")]
    env = MultiGPUSingleMarketEnv(
        dummy_data_three_steps_multi.instrument, "IBKR", devices=devices
    )

    base_env = env.envs[0]
    base_policy = TradingPolicy(
        instrument_data=base_env.instrument_data,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=2, device=base_env.instrument_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLoss(),
        position_maintenance_policy=CloseAfterOneStep(),
        trading_done_policy=SingleTradeDonePolicy(
            batch_size=2, device=base_env.instrument_data.device
        ),
        batch_size=2,
    )

    # Test with larger batch size to verify chunking
    start_d = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    start_t = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

    total_profit, total_profit_percent, d_idx, t_idx, steps = env.rollout(
        base_policy, start_d, start_t, max_steps=5
    )

    # Verify results have correct shape and structure
    assert total_profit.shape == (4,), f"Expected shape (4,), got {total_profit.shape}"
    assert d_idx.shape == (4,), f"Expected shape (4,), got {d_idx.shape}"
    assert t_idx.shape == (4,), f"Expected shape (4,), got {t_idx.shape}"
    assert isinstance(steps, int), f"Expected steps to be int, got {type(steps)}"
    assert steps > 0, f"Expected steps > 0, got {steps}"

    # Test chunking function directly
    chunks = env._chunk_tensor(start_d)
    assert len(chunks) == 2, f"Expected 2 chunks for 2 devices, got {len(chunks)}"
    # Each chunk should have 2 elements (4 total / 2 devices = 2 per device)
    assert (
        chunks[0].shape[0] == 2
    ), f"First chunk should have 2 elements, got {chunks[0].shape[0]}"
    assert (
        chunks[1].shape[0] == 2
    ), f"Second chunk should have 2 elements, got {chunks[1].shape[0]}"
