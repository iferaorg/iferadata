#!/usr/bin/env python3
"""Test to validate the reset method refactoring for multi-GPU support."""

import torch
from ifera.policies.trading_done_policy import SingleTradeDonePolicy, AlwaysFalseDonePolicy
from ifera.policies.open_position_policy import AlwaysOpenPolicy, OpenOncePolicy
from ifera.policies.position_maintenance_policy import ScaledArtrMaintenancePolicy, PercentGainMaintenancePolicy
from ifera.policies.stop_loss_policy import ArtrStopLossPolicy, InitialArtrStopLossPolicy
from ifera.policies.trading_policy import TradingPolicy
from ifera.data_models import DataManager
from ifera.config import BaseInstrumentConfig


class DummyInstrumentData:
    """Dummy data for testing."""
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.backadjust = False
        self.data = torch.randn(10, 100, 10, device=device, dtype=dtype)
        self.artr = torch.ones(10, 100, device=device, dtype=dtype)
        
        # Mock instrument config
        class MockInstrument:
            interval = "1m"
            symbol = "TEST"
            contract_code = "TEST2024"
        
        self.instrument = MockInstrument()


def test_reset_method_signature_compliance():
    """Test that all policies comply with the new reset method signature."""
    print("=== Testing Reset Method Signature Compliance ===")
    
    dummy_data = DummyInstrumentData()
    
    # Test TradingDonePolicy implementations
    policies_to_test = [
        AlwaysFalseDonePolicy(device=torch.device("cpu")),
        SingleTradeDonePolicy(device=torch.device("cpu")),
        AlwaysOpenPolicy(direction=1, device=torch.device("cpu")),
        OpenOncePolicy(direction=1, device=torch.device("cpu")),
        ArtrStopLossPolicy(dummy_data, atr_multiple=1.0),
        InitialArtrStopLossPolicy(dummy_data, atr_multiple=1.0),
    ]
    
    # Test that all policies accept the new signature
    state = {"position": torch.tensor([0, 1], dtype=torch.int32)}
    batch_size = 2
    device = torch.device("cpu")
    
    for policy in policies_to_test:
        try:
            policy.reset(state, batch_size, device)
            print(f"✓ {policy.__class__.__name__} reset method signature correct")
        except Exception as e:
            print(f"✗ {policy.__class__.__name__} failed: {e}")
            raise


def test_device_parameter_usage():
    """Test that policies use the device parameter correctly."""
    print("\n=== Testing Device Parameter Usage ===")
    
    # Test with different devices
    cpu_device = torch.device("cpu")
    
    # Test SingleTradeDonePolicy device handling
    policy = SingleTradeDonePolicy(device=cpu_device)
    
    # Initially on CPU
    initial_device = policy._device
    print(f"Initial device: {initial_device}")
    
    state = {}
    batch_size = 3
    new_device = torch.device("cpu")  # In real scenario this could be cuda:1
    
    # Reset with new device
    policy.reset(state, batch_size, new_device)
    
    # Check that device was updated
    assert policy._device == new_device, f"Device should be updated to {new_device}"
    assert state["had_position"].device == new_device, f"State tensor should be on {new_device}"
    assert state["had_position"].shape[0] == batch_size, f"Batch size should be {batch_size}"
    
    print(f"✓ Device correctly updated to {new_device}")
    print(f"✓ State tensor on correct device: {state['had_position'].device}")
    print(f"✓ Correct batch size: {state['had_position'].shape[0]}")


def test_batch_size_parameter_usage():
    """Test that policies use the batch_size parameter correctly."""
    print("\n=== Testing Batch Size Parameter Usage ===")
    
    policy = AlwaysFalseDonePolicy(device=torch.device("cpu"))
    
    # Test different batch sizes
    for batch_size in [1, 5, 10, 100]:
        state = {}
        device = torch.device("cpu")
        
        policy.reset(state, batch_size, device)
        
        # Check internal buffer was created with correct size
        assert policy._false.shape[0] == batch_size, f"Buffer should have batch_size {batch_size}"
        print(f"✓ Batch size {batch_size} handled correctly")


def test_no_guessing_from_state():
    """Test that policies don't guess batch_size from existing state tensors."""
    print("\n=== Testing No Guessing From State ===")
    
    policy = OpenOncePolicy(direction=1, device=torch.device("cpu"))
    
    # Create state with incorrect batch size to ensure policy doesn't use it
    state = {
        "position": torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),  # batch_size 5
        "some_other_tensor": torch.tensor([1, 2, 3], dtype=torch.float32)  # batch_size 3
    }
    
    # Reset with explicit batch_size different from state tensors
    correct_batch_size = 2
    device = torch.device("cpu")
    
    policy.reset(state, correct_batch_size, device)
    
    # Check that the policy used the explicit batch_size, not the state tensor sizes
    assert state["opened"].shape[0] == correct_batch_size, \
        f"Policy should create tensors with explicit batch_size {correct_batch_size}"
    
    print(f"✓ Policy used explicit batch_size {correct_batch_size}, not state tensor sizes")


def test_had_position_buffer_removed():
    """Test that SingleTradeDonePolicy.had_position member was removed."""
    print("\n=== Testing had_position Buffer Removal ===")
    
    policy = SingleTradeDonePolicy(device=torch.device("cpu"))
    
    # Check that had_position is not in the state_dict (no longer a buffer)
    state_dict_keys = list(policy.state_dict().keys())
    assert "had_position" not in state_dict_keys, \
        "had_position should no longer be a registered buffer"
    
    # Check that had_position is not an attribute
    assert not hasattr(policy, "had_position"), \
        "had_position should not be an instance attribute"
    
    print("✓ had_position buffer successfully removed")
    
    # Test that state["had_position"] is set directly
    state = {}
    batch_size = 3
    device = torch.device("cpu")
    
    policy.reset(state, batch_size, device)
    
    assert "had_position" in state, "had_position should be set in state dict"
    assert state["had_position"].shape[0] == batch_size, "had_position should have correct batch size"
    assert state["had_position"].device == device, "had_position should be on correct device"
    
    print("✓ had_position correctly managed in state dict")


def test_multi_gpu_simulation():
    """Simulate multi-GPU usage pattern."""
    print("\n=== Testing Multi-GPU Simulation ===")
    
    # Simulate different devices (in real scenario these would be cuda:0, cuda:1, etc.)
    devices = [torch.device("cpu"), torch.device("cpu")]  # Using CPU for both since CUDA might not be available
    batch_sizes = [2, 3]  # Different batch sizes per device
    
    policies = []
    for i, (device, batch_size) in enumerate(zip(devices, batch_sizes)):
        # Create policies for each "device"
        policy = SingleTradeDonePolicy(device=device)
        
        state = {}
        policy.reset(state, batch_size, device)
        
        # Verify correct setup
        assert policy._device == device, f"Policy {i} should be on device {device}"
        assert state["had_position"].shape[0] == batch_size, f"Policy {i} should have batch_size {batch_size}"
        assert state["had_position"].device == device, f"Policy {i} state should be on device {device}"
        
        policies.append((policy, state))
        print(f"✓ Policy {i}: device={device}, batch_size={batch_size}")
    
    print("✓ Multi-GPU simulation successful")


if __name__ == "__main__":
    test_reset_method_signature_compliance()
    test_device_parameter_usage()
    test_batch_size_parameter_usage()
    test_no_guessing_from_state()
    test_had_position_buffer_removed()
    test_multi_gpu_simulation()
    
    print("\n🎉 All refactoring requirements validated successfully!")