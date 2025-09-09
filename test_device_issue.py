#!/usr/bin/env python3
"""Test to reproduce the device issue with both done policies."""

import torch
from ifera.policies.trading_done_policy import SingleTradeDonePolicy, AlwaysFalseDonePolicy

def test_single_trade_done_policy():
    """Test that demonstrates the device issue with SingleTradeDonePolicy."""
    print("=== Testing SingleTradeDonePolicy ===")
    # Create policy on CPU
    policy = SingleTradeDonePolicy(device=torch.device("cpu"))
    
    # Create a dummy state to initialize buffers
    state = {
        "position": torch.tensor([1, 0, 1], dtype=torch.float32),
        "had_position": torch.tensor([False, True, False], dtype=torch.bool)
    }
    
    # Reset the policy to initialize internal buffers
    policy.reset(state)
    
    print("Before moving to device:")
    print(f"Policy device: {policy._device}")
    print(f"had_position device: {policy.had_position.device}")
    print(f"state_dict keys: {list(policy.state_dict().keys())}")
    
    # Simulate what happens during multi-device processing
    simulated_device = torch.device("cpu")  # In real scenario, this would be cuda:1
    policy.to(simulated_device)
    
    print("\nAfter moving to simulated device:")
    print(f"Policy device: {policy._device}")
    print(f"had_position device: {policy.had_position.device}")
    print(f"state_dict keys: {list(policy.state_dict().keys())}")
    
    print("✓ SingleTradeDonePolicy fixed!")

def test_always_false_done_policy():
    """Test that demonstrates the device issue with AlwaysFalseDonePolicy."""
    print("\n=== Testing AlwaysFalseDonePolicy ===")
    # Create policy on CPU
    policy = AlwaysFalseDonePolicy(device=torch.device("cpu"))
    
    # Create a dummy state to initialize buffers
    state = {
        "position": torch.tensor([1, 0, 1], dtype=torch.float32),
        "had_position": torch.tensor([False, True, False], dtype=torch.bool)
    }
    
    # Reset the policy to initialize internal buffers
    policy.reset(state)
    
    print("Before moving to device:")
    print(f"Policy device: {policy._device}")
    print(f"_false device: {policy._false.device}")
    print(f"state_dict keys: {list(policy.state_dict().keys())}")
    
    # Simulate what happens during multi-device processing
    simulated_device = torch.device("cpu")  # In real scenario, this would be cuda:1
    policy.to(simulated_device)
    
    print("\nAfter moving to simulated device:")
    print(f"Policy device: {policy._device}")
    print(f"_false device: {policy._false.device}")
    print(f"state_dict keys: {list(policy.state_dict().keys())}")
    
    print("✓ AlwaysFalseDonePolicy fixed!")

if __name__ == "__main__":
    test_single_trade_done_policy()
    test_always_false_done_policy()