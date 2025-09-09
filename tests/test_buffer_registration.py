#!/usr/bin/env python3
"""Test that verifies buffers are properly registered and moved with .to(device)."""

import torch
import pytest
from ifera.policies.trading_done_policy import SingleTradeDonePolicy, AlwaysFalseDonePolicy
from ifera.policies.position_maintenance_policy import ScaledArtrMaintenancePolicy, PercentGainMaintenancePolicy
from ifera.data_models import DataManager

class DummyData:
    def __init__(self, instrument):
        self.instrument = instrument
        self.data = torch.zeros((2, 2, 4), dtype=torch.float32)
        self.artr = torch.ones((2, 2), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backadjust = False

    def convert_indices(self, _base, date_idx, time_idx):
        return date_idx, time_idx

def test_done_policies_buffers():
    """Test that done policies have their buffers properly registered."""
    print("=== Testing Done Policies Buffer Registration ===")
    
    # Test SingleTradeDonePolicy
    single_policy = SingleTradeDonePolicy(device=torch.device("cpu"))
    print(f"SingleTradeDonePolicy state_dict keys: {list(single_policy.state_dict().keys())}")
    assert "had_position" in single_policy.state_dict(), "had_position should be in state_dict"
    
    # Test AlwaysFalseDonePolicy  
    false_policy = AlwaysFalseDonePolicy(device=torch.device("cpu"))
    print(f"AlwaysFalseDonePolicy state_dict keys: {list(false_policy.state_dict().keys())}")
    assert "_false" in false_policy.state_dict(), "_false should be in state_dict"
    
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
    print(f"ScaledArtrMaintenancePolicy state_dict keys: {list(scaled_policy.state_dict().keys())}")
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
    print(f"PercentGainMaintenancePolicy state_dict keys: {list(percent_policy.state_dict().keys())}")
    assert "_action" in percent_policy.state_dict(), "_action should be in state_dict"
    assert "_initial_stage" in percent_policy.state_dict(), "_initial_stage should be in state_dict"
    assert "_nan" in percent_policy.state_dict(), "_nan should be in state_dict"
    
    print("✓ Maintenance policies buffer registration working correctly!")

def test_device_movement_simulation():
    """Test that simulates device movement to verify fix."""
    print("\n=== Testing Device Movement Simulation ===")
    
    # Create a SingleTradeDonePolicy
    policy = SingleTradeDonePolicy(device=torch.device("cpu"))
    
    # Initialize the buffer
    state = {
        "position": torch.tensor([1, 0, 1], dtype=torch.float32),
        "had_position": torch.tensor([False, True, False], dtype=torch.bool)
    }
    policy.reset(state)
    
    print(f"Before .to(): had_position device = {policy.had_position.device}")
    
    # Simulate moving to a different device (since CUDA might not be available, we use CPU)
    # In real multiprocessing scenario, this would be moving to cuda:1 from cuda:0
    policy.to(torch.device("cpu"))
    
    print(f"After .to(): had_position device = {policy.had_position.device}")
    
    # Test that the policy can be used without device mismatch errors
    # This simulates the scenario where position comes from state on one device
    # and had_position is a buffer that should be on the same device
    state_tensors = {
        "position": torch.tensor([0, 1, 0], dtype=torch.float32),
        "had_position": torch.tensor([True, False, True], dtype=torch.bool)
    }
    
    try:
        result = policy(state_tensors)
        print(f"✓ Policy call successful! Result device: {result.device}")
    except RuntimeError as e:
        print(f"✗ Error during policy call: {e}")
        raise
    
    print("✓ Device movement simulation working correctly!")

if __name__ == "__main__":
    print("Run this with pytest to use fixtures")
    test_done_policies_buffers()
    test_device_movement_simulation()
    print("\n🎉 Basic tests passed! Run 'pytest test_buffer_registration.py -v -s' for full test with maintenance policies.")