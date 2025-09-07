"""
Test script for profit percent functionality.
"""
import torch
import pytest
from ifera.environments import SingleMarketEnv
from ifera.data_models import DataManager
from ifera.config import BaseInstrumentConfig
from ifera.policies import (
    TradingPolicy,
    AlwaysOpenPolicy,
    AlwaysFalseDonePolicy,
    StopLossPolicy,
    PositionMaintenancePolicy,
)


class DummyData:
    def __init__(self, instrument, steps=3):
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


class DummyInitialStopLoss(StopLossPolicy):
    def reset(self, state: dict[str, torch.Tensor]) -> None:
        _ = state
        return None

    def forward(self, state: dict[str, torch.Tensor], action: torch.Tensor):
        _ = state, action
        return torch.zeros_like(action, dtype=torch.float32)


class DummyMaintenance(PositionMaintenancePolicy):
    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        _ = state, mask
        return None

    def forward(self, state: dict[str, torch.Tensor]):
        _ = state
        action = torch.zeros_like(state["position"], dtype=torch.int32)
        stop_loss = torch.full_like(
            state["position"], float("nan"), dtype=torch.float32
        )
        return action, stop_loss


class TestTradingPolicy:
    def __init__(self, actions, stop_losses=None):
        self.actions = actions
        self.stop_losses = stop_losses or [float("nan")] * len(actions)
        self.step_count = 0

    def __call__(self, state):
        if self.step_count < len(self.actions):
            action = torch.tensor([self.actions[self.step_count]], dtype=torch.int32)
            stop_loss = torch.tensor([self.stop_losses[self.step_count]], dtype=torch.float32)
        else:
            action = torch.tensor([0], dtype=torch.int32)
            stop_loss = torch.tensor([float("nan")], dtype=torch.float32)
        
        done = torch.tensor([False], dtype=torch.bool)
        self.step_count += 1
        return action, stop_loss, done

    def reset(self, state):
        pass


def test_profit_percent_basic():
    """Test basic profit percent calculation."""
    
    # Create a proper BaseInstrumentConfig  
    config = BaseInstrumentConfig(
        symbol="CL",
        description="Light Sweet Crude Oil",
        currency="USD",
        type="futures",
        interval="30m",
        tradingStart="-06:00:00",
        tradingEnd="17:00:00",
        liquidStart="03:00:00", 
        liquidEnd="16:00:00",
        regularStart="09:30:00",
        regularEnd="16:00:00",
        contractMultiplier=1000,
        tickSize=0.01,
        startDate="2020-01-01",
        daysOfWeek=[0, 1, 2, 3, 4],
    )
    
    # Create mock data
    dummy_data = DummyData(config)
    
    # Mock DataManager
    original_method = DataManager.get_instrument_data
    def mock_get_instrument_data(self, config, **kwargs):
        return dummy_data
    
    DataManager.get_instrument_data = mock_get_instrument_data
    
    try:
        env = SingleMarketEnv(config, broker_name="IBKR")
        
        # Mock market simulator to return predictable results
        def mock_calculate_step(date_idx, time_idx, position, action, stop_loss):
            if action.item() == 1:  # Buy 1 contract at price 100
                profit = torch.tensor([0.0], dtype=torch.float32)  # No profit on entry
                new_position = torch.tensor([1], dtype=torch.int32)
                execution_price = torch.tensor([100.0], dtype=torch.float32)
            elif action.item() == -1:  # Sell 1 contract (close position) with profit
                profit = torch.tensor([200.0], dtype=torch.float32)  # $200 profit
                new_position = torch.tensor([0], dtype=torch.int32)
                execution_price = torch.tensor([102.0], dtype=torch.float32)
            else:
                profit = torch.tensor([0.0], dtype=torch.float32)
                new_position = position
                execution_price = torch.tensor([0.0], dtype=torch.float32)
            
            return profit, new_position, execution_price, None
        
        env.market_simulator.calculate_step = mock_calculate_step
        
        # Create trading policy: buy 1, then sell 1
        policy = TestTradingPolicy([1, -1, 0])
        
        # Reset environment
        state = env.reset(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            policy
        )
        
        # Check initial state
        assert "total_profit_percent" in state
        assert "entry_position" in state
        assert "entry_cost" in state
        
        # Step 1: Buy 1 contract at $100
        step_state = env.step(policy, state)
        
        # Check step result has profit_percent
        assert "profit_percent" in step_state
        assert step_state["profit_percent"].item() == 0.0  # No profit on entry
        assert step_state["entry_position"].item() == 1
        assert step_state["entry_cost"].item() == 100.0 * 1 * 1000  # price * position * multiplier
        
        # Update state
        for key in step_state:
            if key in state:
                state[key] = step_state[key]
        
        # Step 2: Sell 1 contract (close position) with $200 profit
        step_state = env.step(policy, state)
        
        # Expected profit percent: 200 / (100 * 1 * 1000) = 200 / 100000 = 0.002 = 0.2%
        expected_profit_percent = 200.0 / (100.0 * 1 * 1000)
        assert step_state["profit_percent"].item() == pytest.approx(expected_profit_percent)
        
        # After closing position, entry_position should be reset to 0
        assert step_state["entry_position"].item() == 0
        assert torch.isnan(step_state["entry_cost"]).item()
        
        print("✓ Basic profit percent test passed")
        
    finally:
        # Restore original method
        DataManager.get_instrument_data = original_method


def test_profit_percent_edge_case_enter_and_stop():
    """Test edge case where we enter and stop out in same step."""
    
    # Create a proper BaseInstrumentConfig
    config = BaseInstrumentConfig(
        symbol="CL",
        description="Light Sweet Crude Oil",
        currency="USD", 
        type="futures",
        interval="30m",
        tradingStart="-06:00:00",
        tradingEnd="17:00:00",
        liquidStart="03:00:00",
        liquidEnd="16:00:00", 
        regularStart="09:30:00",
        regularEnd="16:00:00",
        contractMultiplier=1000,
        tickSize=0.01,
        startDate="2020-01-01",
        daysOfWeek=[0, 1, 2, 3, 4],
    )
    
    # Create mock data
    dummy_data = DummyData(config)
    
    # Mock DataManager
    original_method = DataManager.get_instrument_data
    def mock_get_instrument_data(self, config, **kwargs):
        return dummy_data
    
    DataManager.get_instrument_data = mock_get_instrument_data
    
    try:
        env = SingleMarketEnv(config, broker_name="IBKR")
        
        # Mock market simulator to simulate entering and stopping out in same step
        def mock_calculate_step(date_idx, time_idx, position, action, stop_loss):
            if action.item() == 2:  # Buy 2 contracts, but get stopped out immediately
                profit = torch.tensor([-300.0], dtype=torch.float32)  # Loss from stop out
                new_position = torch.tensor([0], dtype=torch.int32)  # Stopped out, position = 0
                execution_price = torch.tensor([120.0], dtype=torch.float32)  # Entry price
            else:
                profit = torch.tensor([0.0], dtype=torch.float32)
                new_position = position
                execution_price = torch.tensor([0.0], dtype=torch.float32)
            
            return profit, new_position, execution_price, None
        
        env.market_simulator.calculate_step = mock_calculate_step
        
        # Create trading policy: buy 2 contracts (will be stopped out immediately)
        policy = TestTradingPolicy([2, 0])
        
        # Reset environment
        state = env.reset(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            policy
        )
        
        # Step 1: Buy 2 contracts but get stopped out immediately
        step_state = env.step(policy, state)
        
        # Even though final position is 0 due to stop out, 
        # entry_position should still reflect the original entry (2 contracts)
        # and profit_percent should be calculated correctly
        assert step_state["entry_position"].item() == 2  # Original entry position
        expected_entry_cost = 120.0 * 2 * 1000  # entry_price * abs(entry_position) * multiplier
        assert step_state["entry_cost"].item() == expected_entry_cost
        
        # Profit percent should be calculated as: profit / entry_cost
        expected_profit_percent = -300.0 / expected_entry_cost
        assert step_state["profit_percent"].item() == pytest.approx(expected_profit_percent)
        
        # Final position should be 0 due to stop out
        assert step_state["position"].item() == 0
        
        print("✓ Edge case enter and stop out test passed")
        
    finally:
        # Restore original method
        DataManager.get_instrument_data = original_method


if __name__ == "__main__":
    test_profit_percent_basic()
    test_profit_percent_edge_case_enter_and_stop()
    print("All tests passed!")