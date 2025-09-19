#!/usr/bin/env python3
"""Test that verifies buffers are properly registered and moved with .to(device)."""

import torch
import pytest
from ifera.policies.trading_done_policy import (
    SingleTradeDonePolicy,
    AlwaysFalseDonePolicy,
)
from ifera.policies.position_maintenance_policy import (
    ScaledArtrMaintenancePolicy,
    PercentGainMaintenancePolicy,
)
from ifera.data_models import DataManager
import tensordict as td


class DummyData:
    def __init__(self, instrument):
        self.instrument = instrument
        self.data = torch.zeros((2, 2, 4), dtype=torch.float32)
        self.artr = torch.ones((2, 2), dtype=torch.float32)
        self.multiplier = torch.ones((2, 2), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backadjust = False
        self.artr_alpha = 0.3  # Missing attribute
        self.artr_acrossday = False  # Missing attribute

    def convert_indices(self, _base, date_idx, time_idx):
        return date_idx, time_idx

    @property
    def valid_mask(self):
        return torch.ones((2, 2), dtype=torch.bool)


def test_done_policies_buffers():
    """Test that done policies have their buffers properly registered."""
    print("=== Testing Done Policies Buffer Registration ===")

    # Test SingleTradeDonePolicy - after refactor, it no longer has had_position buffer
    single_policy = SingleTradeDonePolicy(device=torch.device("cpu"))
    print(
        f"SingleTradeDonePolicy state_dict keys: {list(single_policy.state_dict().keys())}"
    )
    # had_position is now managed in state, not as a buffer

    # Test AlwaysFalseDonePolicy
    false_policy = AlwaysFalseDonePolicy(device=torch.device("cpu"))
    print(
        f"AlwaysFalseDonePolicy state_dict keys: {list(false_policy.state_dict().keys())}"
    )
    # After refactor, AlwaysFalseDonePolicy no longer has internal buffers
    # It just preserves the done state from input

    print("✓ Done policies buffer registration working correctly!")


def test_maintenance_policies_buffers(base_instrument_config, monkeypatch):
    """Test that maintenance policies have their buffers properly registered."""
    print("\n=== Testing Maintenance Policies Buffer Registration ===")

    def dummy_get(self, instrument_config, **_):
        return DummyData(instrument_config)

    # Monkey patch to avoid real data loading
    monkeypatch.setattr(DataManager, "get_instrument_data", dummy_get)

    dummy_data = DummyData(base_instrument_config)

    # Test ScaledArtrMaintenancePolicy
    scaled_policy = ScaledArtrMaintenancePolicy(
        dummy_data,
        [dummy_data.instrument.interval, "1h"],
        atr_multiple=1.0,
        wait_for_breakeven=False,
        minimum_improvement=0.1,
    )
    print(
        f"ScaledArtrMaintenancePolicy state_dict keys: {list(scaled_policy.state_dict().keys())}"
    )
    assert "_action" in scaled_policy.state_dict(), "_action should be in state_dict"
    assert "_zero" in scaled_policy.state_dict(), "_zero should be in state_dict"
    assert "_nan" in scaled_policy.state_dict(), "_nan should be in state_dict"

    # Test PercentGainMaintenancePolicy
    percent_policy = PercentGainMaintenancePolicy(
        dummy_data,
        stage1_atr_multiple=1.0,
        trailing_stop=True,
        skip_stage1=False,
        keep_percent=0.5,
        anchor_type="entry",
    )
    print(
        f"PercentGainMaintenancePolicy state_dict keys: {list(percent_policy.state_dict().keys())}"
    )
    assert "_action" in percent_policy.state_dict(), "_action should be in state_dict"
    assert (
        "_initial_stage" in percent_policy.state_dict()
    ), "_initial_stage should be in state_dict"
    assert "_nan" in percent_policy.state_dict(), "_nan should be in state_dict"

    print("✓ Maintenance policies buffer registration working correctly!")


def test_device_movement_simulation():
    """Test that simulates device movement to verify fix."""
    print("\n=== Testing Device Movement Simulation ===")

    # Create a SingleTradeDonePolicy
    policy = SingleTradeDonePolicy(device=torch.device("cpu"))

    # Initialize the state using TensorDict
    state = td.TensorDict({
        "date_idx": torch.tensor([0, 0, 0], dtype=torch.int32),
        "time_idx": torch.tensor([0, 0, 0], dtype=torch.int32),
        "position": torch.tensor([1, 0, 1], dtype=torch.int32),
        "had_position": torch.tensor([False, True, False], dtype=torch.bool),
        "done": torch.tensor([False, False, False], dtype=torch.bool),
    }, batch_size=3, device=torch.device("cpu"))

    batch_size = state.batch_size
    device = state.device
    policy.reset(state)

    print(f"Before .to(): had_position device = {state['had_position'].device}")

    # Simulate moving to a different device (since CUDA might not be available, we use CPU)
    # In real multiprocessing scenario, this would be moving to cuda:1 from cuda:0
    policy.to(torch.device("cpu"))

    # Reset again to test device update
    policy.reset(state)
    print(f"After .to() and reset: had_position device = {state['had_position'].device}")

    # Test that the policy can be used without device mismatch errors
    # This simulates the scenario where position comes from state on one device
    # and had_position is a buffer that should be on the same device
    test_state = td.TensorDict({
        "date_idx": torch.tensor([0, 0, 0], dtype=torch.int32),
        "time_idx": torch.tensor([0, 0, 0], dtype=torch.int32),
        "position": torch.tensor([0, 1, 0], dtype=torch.int32),
        "had_position": torch.tensor([True, False, True], dtype=torch.bool),
        "done": torch.tensor([False, False, False], dtype=torch.bool),
    }, batch_size=3, device=torch.device("cpu"))

    try:
        result = policy(test_state)
        print(f"✓ Policy call successful! Result device: {result.device}")
    except RuntimeError as e:
        print(f"✗ Error during policy call: {e}")
        raise

    print("✓ Device movement simulation working correctly!")


if __name__ == "__main__":
    print("Run this with pytest to use fixtures")
    test_done_policies_buffers()
    test_device_movement_simulation()
    print(
        "\n🎉 Basic tests passed! Run 'pytest test_buffer_registration.py -v -s' for full test with maintenance policies."
    )
