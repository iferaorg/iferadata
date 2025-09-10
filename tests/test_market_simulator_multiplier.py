import torch
import pytest
from unittest.mock import Mock, patch
from ifera.market_simulator import MarketSimulatorIntraday
from ifera.data_models import InstrumentData, DataManager
from ifera.config import BaseInstrumentConfig, ConfigManager


def test_calculate_step_multiplier_applied(monkeypatch):
    """Test that commission and slippage are multiplied by the multiplier tensor."""

    # Create a mock instrument config
    mock_instrument = Mock()
    mock_instrument.symbol = "TEST"
    mock_instrument.type = "futures"
    mock_instrument.contract_multiplier = 1.0
    mock_instrument.commission = 1.0
    mock_instrument.min_commission = 0.0
    mock_instrument.max_commission_pct = 0.0
    mock_instrument.slippage = 0.1
    mock_instrument.reference_price = 100.0
    mock_instrument.min_slippage = 0.0

    # Create mock data with shape [2, 2, 5] (2 days, 2 time steps, 5 channels)
    # Channels: [open, high, low, close, volume]
    mock_data = torch.tensor(
        [
            [
                [100.0, 101.0, 99.0, 100.5, 1000.0],  # Day 0, Time 0
                [100.5, 102.0, 99.5, 101.0, 1100.0],
            ],  # Day 0, Time 1
            [
                [101.0, 102.0, 100.0, 101.5, 1200.0],  # Day 1, Time 0
                [101.5, 103.0, 100.5, 102.0, 1300.0],
            ],  # Day 1, Time 1
        ],
        dtype=torch.float32,
    )

    # Create mock multiplier tensor with shape [2, 2] (same as data shape excluding channels)
    # Different multipliers for different date/time combinations
    mock_multiplier = torch.tensor(
        [
            [1.0, 1.5],  # Day 0: Time 0 = 1.0x, Time 1 = 1.5x
            [2.0, 2.5],  # Day 1: Time 0 = 2.0x, Time 1 = 2.5x
        ],
        dtype=torch.float32,
    )

    # Create mock valid mask
    mock_mask = torch.ones((2, 2), dtype=torch.bool)

    # Create mock instrument data
    mock_instrument_data = Mock()
    mock_instrument_data.instrument = mock_instrument
    mock_instrument_data.data = mock_data
    mock_instrument_data.valid_mask = mock_mask
    mock_instrument_data.multiplier = mock_multiplier
    mock_instrument_data.device = torch.device("cpu")

    # Mock the ConfigManager to return the same instrument config
    mock_config_manager = Mock()
    mock_config_manager.get_config_from_base.return_value = mock_instrument

    with patch(
        "ifera.market_simulator.ConfigManager", return_value=mock_config_manager
    ):
        # Create market simulator
        simulator = MarketSimulatorIntraday(mock_instrument_data, "test_broker")

        # Test parameters for a specific date/time
        date_idx = torch.tensor([1])  # Day 1
        time_idx = torch.tensor([1])  # Time 1
        position = torch.tensor([0])  # No existing position
        action = torch.tensor([1])  # Buy 1 contract
        stop_loss = torch.tensor([float("nan")])  # No stop loss

        # Calculate step
        profit, new_position, execution_price, cashflow = simulator.calculate_step(
            date_idx, time_idx, position, action, stop_loss
        )

        # Verify that the multiplier was correctly applied
        expected_multiplier = mock_multiplier[1, 1]  # Should be 2.5

        # The slippage should be affected by multiplier
        current_price = mock_data[1, 1, 0]  # 101.5
        base_slippage = current_price * (
            mock_instrument.slippage / mock_instrument.reference_price
        )  # 101.5 * 0.001 = 0.1015
        expected_slippage = (
            base_slippage * expected_multiplier
        )  # 0.1015 * 2.5 = 0.25375
        expected_execution_price = (
            current_price + expected_slippage
        )  # 101.5 + 0.25375 = 101.75375

        # The commission should also be affected by multiplier
        base_commission = 1.0 * mock_instrument.commission  # 1.0 * 1.0 = 1.0
        expected_commission = base_commission * expected_multiplier  # 1.0 * 2.5 = 2.5

        # Check execution price (should include multiplied slippage)
        torch.testing.assert_close(
            execution_price, torch.tensor([expected_execution_price])
        )

        # Check that commission was multiplied in the cashflow calculation
        # cashflow = execution_price * (close_position - open_position) * contract_multiplier - commission - stop_commission
        # For our case: execution_price * (-1) * 1.0 - 2.5 - 0 = -101.75375 - 2.5 = -104.25375
        expected_cashflow = -expected_execution_price - expected_commission
        torch.testing.assert_close(
            cashflow, torch.tensor([expected_cashflow]), rtol=1e-5, atol=1e-5
        )


def test_calculate_step_multiplier_different_indices(monkeypatch):
    """Test that different date/time indices use different multiplier values."""

    # Create a mock instrument config
    mock_instrument = Mock()
    mock_instrument.symbol = "TEST"
    mock_instrument.type = "futures"
    mock_instrument.contract_multiplier = 1.0
    mock_instrument.commission = 1.0
    mock_instrument.min_commission = 0.0
    mock_instrument.max_commission_pct = 0.0
    mock_instrument.slippage = 0.1
    mock_instrument.reference_price = 100.0
    mock_instrument.min_slippage = 0.0

    # Create mock data with shape [2, 2, 5]
    mock_data = torch.tensor(
        [
            [[100.0, 101.0, 99.0, 100.5, 1000.0], [100.5, 102.0, 99.5, 101.0, 1100.0]],
            [
                [101.0, 102.0, 100.0, 101.5, 1200.0],
                [101.5, 103.0, 100.5, 102.0, 1300.0],
            ],
        ],
        dtype=torch.float32,
    )

    # Create mock multiplier tensor with different values
    mock_multiplier = torch.tensor([[1.0, 1.5], [2.0, 2.5]], dtype=torch.float32)

    mock_mask = torch.ones((2, 2), dtype=torch.bool)

    mock_instrument_data = Mock()
    mock_instrument_data.instrument = mock_instrument
    mock_instrument_data.data = mock_data
    mock_instrument_data.valid_mask = mock_mask
    mock_instrument_data.multiplier = mock_multiplier
    mock_instrument_data.device = torch.device("cpu")

    mock_config_manager = Mock()
    mock_config_manager.get_config_from_base.return_value = mock_instrument

    with patch(
        "ifera.market_simulator.ConfigManager", return_value=mock_config_manager
    ):
        simulator = MarketSimulatorIntraday(mock_instrument_data, "test_broker")

        # Test case 1: date_idx=0, time_idx=0 (multiplier=1.0)
        date_idx1 = torch.tensor([0])
        time_idx1 = torch.tensor([0])
        position = torch.tensor([0])
        action = torch.tensor([1])
        stop_loss = torch.tensor([float("nan")])

        profit1, new_position1, execution_price1, cashflow1 = simulator.calculate_step(
            date_idx1, time_idx1, position, action, stop_loss
        )

        # Test case 2: date_idx=0, time_idx=1 (multiplier=1.5)
        date_idx2 = torch.tensor([0])
        time_idx2 = torch.tensor([1])

        profit2, new_position2, execution_price2, cashflow2 = simulator.calculate_step(
            date_idx2, time_idx2, position, action, stop_loss
        )

        # The execution prices should be different due to different multipliers affecting slippage
        # Also cashflows should be different due to different multipliers affecting commission
        assert not torch.allclose(execution_price1, execution_price2)
        assert not torch.allclose(cashflow1, cashflow2)

        # Verify that the ratio of differences matches the multiplier ratio
        # Since slippage and commission are both multiplied, the effect should be proportional
        multiplier1 = mock_multiplier[0, 0]  # 1.0
        multiplier2 = mock_multiplier[0, 1]  # 1.5

        # The execution price difference from base price should scale with multiplier
        base_price1 = mock_data[0, 0, 0]  # 100.0
        base_price2 = mock_data[0, 1, 0]  # 100.5

        slippage_diff1 = execution_price1 - base_price1
        slippage_diff2 = execution_price2 - base_price2

        # Slippage should be proportional to multiplier (adjusted for different base prices)
        expected_ratio = (multiplier2 / multiplier1) * (base_price2 / base_price1)
        actual_ratio = slippage_diff2 / slippage_diff1
        torch.testing.assert_close(
            actual_ratio, torch.tensor([expected_ratio]), rtol=1e-5, atol=1e-5
        )
