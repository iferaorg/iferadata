"""Test for CUDA multiprocessing fix in MultiGPUSingleMarketEnv."""

import multiprocessing
import pytest
import torch
from unittest.mock import patch, MagicMock

from ifera.environments import MultiGPUSingleMarketEnv
from ifera.config import BaseInstrumentConfig
from ifera.data_models import DataManager
from ifera.policies import TradingPolicy, AlwaysOpenPolicy, SingleTradeDonePolicy
from tests.test_single_market_env import (
    DummyData,
    DummyInitialStopLoss,
    CloseAfterOneStep,
)


@pytest.fixture
def dummy_data_three_steps_cuda(base_instrument_config: BaseInstrumentConfig):
    return DummyData(base_instrument_config, steps=3)


def test_cuda_device_detection():
    """Test that CUDA device detection works correctly."""
    # Test CPU-only devices
    devices = [torch.device("cpu"), torch.device("cpu")]
    has_cuda = any(device.type == "cuda" for device in devices)
    assert not has_cuda, "Should not detect CUDA for CPU-only devices"

    # Test mixed devices
    devices = [torch.device("cuda:0"), torch.device("cpu")]
    has_cuda = any(device.type == "cuda" for device in devices)
    assert has_cuda, "Should detect CUDA for mixed devices"

    # Test all CUDA devices
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    has_cuda = any(device.type == "cuda" for device in devices)
    assert has_cuda, "Should detect CUDA for all CUDA devices"


def test_multiprocessing_start_method_setting_cpu_only(
    monkeypatch, dummy_data_three_steps_cuda
):
    """Test that CPU-only devices don't change multiprocessing start method."""
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_cuda,
    )

    # Use CPU devices only
    devices = [torch.device("cpu"), torch.device("cpu")]
    env = MultiGPUSingleMarketEnv(
        dummy_data_three_steps_cuda.instrument, "IBKR", devices=devices
    )

    # Mock the ProcessPoolExecutor to avoid actual multiprocessing
    with patch("ifera.environments.ProcessPoolExecutor") as mock_executor:
        mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = (
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([0]),
            torch.tensor([2]),
            3,
        )

        base_env = env.envs[0]
        trading_policy = TradingPolicy(
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

        start_d = torch.tensor([0], dtype=torch.int32)
        start_t = torch.tensor([0], dtype=torch.int32)

        # The rollout method should not change multiprocessing start method for CPU devices
        with patch(
            "ifera.environments.multiprocessing.set_start_method"
        ) as mock_set_method:
            env.rollout(trading_policy, start_d, start_t, max_steps=5)

            # set_start_method should not be called for CPU-only devices
            mock_set_method.assert_not_called()


def test_multiprocessing_start_method_setting_cuda(
    monkeypatch, dummy_data_three_steps_cuda
):
    """Test that CUDA devices trigger multiprocessing start method change."""
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_cuda,
    )

    # Use CUDA devices (mocked)
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    # Mock the environment creation to avoid actual CUDA initialization
    with patch("ifera.environments.SingleMarketEnv") as mock_env_class:
        mock_env_instance = MagicMock()
        mock_env_class.return_value = mock_env_instance
        mock_env_instance.instrument_data = dummy_data_three_steps_cuda

        env = MultiGPUSingleMarketEnv(
            dummy_data_three_steps_cuda.instrument, "IBKR", devices=devices
        )

        # Mock the ProcessPoolExecutor to avoid actual multiprocessing
        with patch("ifera.environments.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = (
                torch.tensor([1.0]),
                torch.tensor([0.0]),
                torch.tensor([0]),
                torch.tensor([2]),
                3,
            )

            trading_policy = TradingPolicy(
                instrument_data=dummy_data_three_steps_cuda,
                open_position_policy=AlwaysOpenPolicy(
                    1, batch_size=1, device=dummy_data_three_steps_cuda.device
                ),
                initial_stop_loss_policy=DummyInitialStopLoss(),
                position_maintenance_policy=CloseAfterOneStep(),
                trading_done_policy=SingleTradeDonePolicy(
                    batch_size=1, device=dummy_data_three_steps_cuda.device
                ),
                batch_size=1,
            )

            start_d = torch.tensor([0], dtype=torch.int32)
            start_t = torch.tensor([0], dtype=torch.int32)

            # Mock the multiprocessing get_start_method to return 'fork' initially
            with patch(
                "ifera.environments.multiprocessing.get_start_method"
            ) as mock_get_method:
                with patch(
                    "ifera.environments.multiprocessing.set_start_method"
                ) as mock_set_method:
                    mock_get_method.return_value = "fork"  # Simulate fork method

                    env.rollout(trading_policy, start_d, start_t, max_steps=5)

                    # set_start_method should be called with 'spawn' and force=True for CUDA devices
                    mock_set_method.assert_called_once_with("spawn", force=True)


def test_multiprocessing_start_method_already_spawn(
    monkeypatch, dummy_data_three_steps_cuda
):
    """Test that if multiprocessing is already set to spawn, it's not changed again."""
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_cuda,
    )

    # Use CUDA devices (mocked)
    devices = [torch.device("cuda:0")]

    # Mock the environment creation to avoid actual CUDA initialization
    with patch("ifera.environments.SingleMarketEnv") as mock_env_class:
        mock_env_instance = MagicMock()
        mock_env_class.return_value = mock_env_instance
        mock_env_instance.instrument_data = dummy_data_three_steps_cuda

        env = MultiGPUSingleMarketEnv(
            dummy_data_three_steps_cuda.instrument, "IBKR", devices=devices
        )

        # Mock the ProcessPoolExecutor to avoid actual multiprocessing
        with patch("ifera.environments.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = (
                torch.tensor([1.0]),
                torch.tensor([0.0]),
                torch.tensor([0]),
                torch.tensor([2]),
                3,
            )

            trading_policy = TradingPolicy(
                instrument_data=dummy_data_three_steps_cuda,
                open_position_policy=AlwaysOpenPolicy(
                    1, batch_size=1, device=dummy_data_three_steps_cuda.device
                ),
                initial_stop_loss_policy=DummyInitialStopLoss(),
                position_maintenance_policy=CloseAfterOneStep(),
                trading_done_policy=SingleTradeDonePolicy(
                    batch_size=1, device=dummy_data_three_steps_cuda.device
                ),
                batch_size=1,
            )

            start_d = torch.tensor([0], dtype=torch.int32)
            start_t = torch.tensor([0], dtype=torch.int32)

            # Mock the multiprocessing get_start_method to return 'spawn' (already set)
            with patch(
                "ifera.environments.multiprocessing.get_start_method"
            ) as mock_get_method:
                with patch(
                    "ifera.environments.multiprocessing.set_start_method"
                ) as mock_set_method:
                    mock_get_method.return_value = "spawn"  # Already spawn

                    env.rollout(trading_policy, start_d, start_t, max_steps=5)

                    # set_start_method should NOT be called since it's already 'spawn'
                    mock_set_method.assert_not_called()


def test_cuda_tensor_chunks_moved_to_cpu_before_multiprocessing(
    monkeypatch, dummy_data_three_steps_cuda
):
    """Test that CUDA tensor chunks are moved to CPU before being passed to workers."""
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_cuda,
    )

    # Use CUDA devices (mocked)
    devices = [torch.device("cuda:0")]

    # Mock the environment creation to avoid actual CUDA initialization
    with patch("ifera.environments.SingleMarketEnv") as mock_env_class:
        mock_env_instance = MagicMock()
        mock_env_class.return_value = mock_env_instance
        mock_env_instance.instrument_data = dummy_data_three_steps_cuda

        env = MultiGPUSingleMarketEnv(
            dummy_data_three_steps_cuda.instrument, "IBKR", devices=devices
        )

        # Create test tensors
        start_date_idx = torch.tensor([0, 1], dtype=torch.int32)
        start_time_idx = torch.tensor([0, 1], dtype=torch.int32)

        # Create trading policy
        trading_policy = TradingPolicy(
            instrument_data=dummy_data_three_steps_cuda,
            open_position_policy=AlwaysOpenPolicy(
                1, batch_size=2, device=dummy_data_three_steps_cuda.device
            ),
            initial_stop_loss_policy=DummyInitialStopLoss(),
            position_maintenance_policy=CloseAfterOneStep(),
            trading_done_policy=SingleTradeDonePolicy(
                batch_size=2, device=dummy_data_three_steps_cuda.device
            ),
            batch_size=2,
        )

        # Mock the ProcessPoolExecutor to capture what gets passed to workers
        submitted_args = []

        def mock_submit(*args, **kwargs):
            submitted_args.append(args)
            # Mock the result
            mock_future = MagicMock()
            mock_future.result.return_value = (
                torch.tensor([1.0]),
                torch.tensor([0.0]),
                torch.tensor([0]),
                torch.tensor([2]),
                3,
            )
            return mock_future

        with patch("ifera.environments.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit = mock_submit

            # Call rollout
            env.rollout(trading_policy, start_date_idx, start_time_idx, max_steps=5)

            # Verify that workers were called
            assert len(submitted_args) == len(devices)

            # Check that the tensor chunks passed to workers are CPU tensors
            for args in submitted_args:
                # Args are: (worker_func, instrument_config, broker_name, backadjust,
                #           device, dtype, d_chunk, t_chunk, policy, max_steps)
                d_chunk = args[6]  # d_chunk argument
                t_chunk = args[7]  # t_chunk argument

                # Verify chunks are CPU tensors (not CUDA)
                assert d_chunk.device == torch.device(
                    "cpu"
                ), f"d_chunk should be on CPU, got {d_chunk.device}"
                assert t_chunk.device == torch.device(
                    "cpu"
                ), f"t_chunk should be on CPU, got {t_chunk.device}"
