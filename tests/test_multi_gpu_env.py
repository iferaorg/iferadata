import torch
import pytest
from unittest.mock import patch, MagicMock

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

    # Mock the environment creation to avoid GitHub API calls in workers
    with patch("ifera.environments.SingleMarketEnv") as mock_env_class:
        mock_env_instance = MagicMock()
        mock_env_class.return_value = mock_env_instance
        mock_env_instance.instrument_data = dummy_data_three_steps_multi

        # Mock the rollout method to return expected results
        def mock_rollout(policy, start_date_idx, start_time_idx, max_steps):
            batch_size = start_date_idx.shape[0]
            return (
                torch.zeros(batch_size),  # total_profit
                torch.zeros(batch_size),  # total_profit_percent
                torch.zeros(batch_size, dtype=torch.int32),  # final_date_idx
                torch.full((batch_size,), 2, dtype=torch.int32),  # final_time_idx
                3,  # steps
            )

        mock_env_instance.rollout = mock_rollout

        env = MultiGPUSingleMarketEnv(
            dummy_data_three_steps_multi.instrument, "IBKR", devices=devices
        )

        base_policy = TradingPolicy(
            instrument_data=dummy_data_three_steps_multi,
            open_position_policy=AlwaysOpenPolicy(
                1, batch_size=1, device=dummy_data_three_steps_multi.device
            ),
            initial_stop_loss_policy=DummyInitialStopLoss(),
            position_maintenance_policy=CloseAfterOneStep(),
            trading_done_policy=SingleTradeDonePolicy(
                batch_size=1, device=dummy_data_three_steps_multi.device
            ),
            batch_size=1,
        )

        start_d = torch.tensor([0, 0], dtype=torch.int32)
        start_t = torch.tensor([0, 0], dtype=torch.int32)

        # Mock the ProcessPoolExecutor to avoid actual multiprocessing and GitHub API calls
        def mock_submit(*args, **kwargs):
            mock_future = MagicMock()
            # Return result that matches expected shape
            mock_future.result.return_value = (
                torch.tensor([0.0]),  # profit chunk
                torch.tensor([0.0]),  # profit_percent chunk
                torch.tensor([0], dtype=torch.int32),  # date_idx chunk
                torch.tensor([2], dtype=torch.int32),  # time_idx chunk
                3,  # steps
            )
            return mock_future

        with patch("ifera.environments.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit = mock_submit

            total_profit, total_profit_percent, d_idx, t_idx, steps = env.rollout(
                base_policy, start_d, start_t, max_steps=5
            )

            assert total_profit.shape == (2,)
            assert total_profit_percent.shape == (2,)
            assert d_idx.shape == (2,)
            assert t_idx.shape == (2,)
            assert isinstance(steps, int)
            assert steps > 0


def test_multi_gpu_env_parallel_chunking(monkeypatch, dummy_data_three_steps_multi):
    """Test that parallel implementation correctly chunks inputs across devices."""
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_multi,
    )

    devices = [torch.device("cpu"), torch.device("cpu")]

    # Mock the environment creation to avoid GitHub API calls in workers
    with patch("ifera.environments.SingleMarketEnv") as mock_env_class:
        mock_env_instance = MagicMock()
        mock_env_class.return_value = mock_env_instance
        mock_env_instance.instrument_data = dummy_data_three_steps_multi

        env = MultiGPUSingleMarketEnv(
            dummy_data_three_steps_multi.instrument, "IBKR", devices=devices
        )

        base_policy = TradingPolicy(
            instrument_data=dummy_data_three_steps_multi,
            open_position_policy=AlwaysOpenPolicy(
                1, batch_size=2, device=dummy_data_three_steps_multi.device
            ),
            initial_stop_loss_policy=DummyInitialStopLoss(),
            position_maintenance_policy=CloseAfterOneStep(),
            trading_done_policy=SingleTradeDonePolicy(
                batch_size=2, device=dummy_data_three_steps_multi.device
            ),
            batch_size=2,
        )

        # Test with larger batch size to verify chunking
        start_d = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        start_t = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

        # Mock the ProcessPoolExecutor to avoid actual multiprocessing and GitHub API calls
        def mock_submit(*args, **kwargs):
            mock_future = MagicMock()
            # Return result that matches chunk size (2 items per device)
            mock_future.result.return_value = (
                torch.tensor([0.0, 0.0]),  # profit chunk (2 items)
                torch.tensor([0.0, 0.0]),  # profit_percent chunk (2 items)
                torch.tensor([0, 0], dtype=torch.int32),  # date_idx chunk (2 items)
                torch.tensor([2, 2], dtype=torch.int32),  # time_idx chunk (2 items)
                3,  # steps
            )
            return mock_future

        with patch("ifera.environments.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit = mock_submit

            total_profit, total_profit_percent, d_idx, t_idx, steps = env.rollout(
                base_policy, start_d, start_t, max_steps=5
            )

            # Verify results have correct shape and structure
            assert total_profit.shape == (
                4,
            ), f"Expected shape (4,), got {total_profit.shape}"
            assert total_profit_percent.shape == (
                4,
            ), f"Expected shape (4,), got {total_profit_percent.shape}"
            assert d_idx.shape == (4,), f"Expected shape (4,), got {d_idx.shape}"
            assert t_idx.shape == (4,), f"Expected shape (4,), got {t_idx.shape}"
            assert isinstance(
                steps, int
            ), f"Expected steps to be int, got {type(steps)}"
            assert steps > 0, f"Expected steps > 0, got {steps}"

            # Test chunking function directly
            chunks = env._chunk_tensor(start_d)
            assert (
                len(chunks) == 2
            ), f"Expected 2 chunks for 2 devices, got {len(chunks)}"
            # Each chunk should have 2 elements (4 total / 2 devices = 2 per device)
            assert (
                chunks[0].shape[0] == 2
            ), f"First chunk should have 2 elements, got {chunks[0].shape[0]}"
            assert (
                chunks[1].shape[0] == 2
            ), f"Second chunk should have 2 elements, got {chunks[1].shape[0]}"
