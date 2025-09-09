#!/usr/bin/env python3
"""Test to reproduce the device issue with SingleTradeDonePolicy."""

import torch
from ifera.policies.trading_done_policy import SingleTradeDonePolicy

def test_device_movement():
    """Test that demonstrates the device issue."""
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
    # Create a new policy on a "different device" (we'll simulate this with CPU)
    # This simulates what happens when a policy is moved to a worker device
    simulated_device = torch.device("cpu")  # In real scenario, this would be cuda:1
    policy.to(simulated_device)
    
    print("\nAfter moving to simulated device:")
    print(f"Policy device: {policy._device}")
    print(f"had_position device: {policy.had_position.device}")
    print(f"state_dict keys: {list(policy.state_dict().keys())}")
    
    # Simulate the real issue: state comes from a different source with different device
    # In the real error, 'position' was on cuda:1 but 'had_position' was on cuda:0
    # We'll simulate by manually creating had_position on a different device reference
    
    # First, let's see what happens when we call reset and then try to use incompatible tensors
    policy.reset(state)
    
    # Manually create the device mismatch that causes the error
    # We'll create a state where position uses one tensor and had_position uses another
    # that simulate different devices
    print("\nSimulating device mismatch...")
    
    # Create position tensor
    position = torch.tensor([0, 1, 0], dtype=torch.float32)
    
    # Get the had_position from the policy (this should be on the same device as the policy)
    had_position = policy.had_position
    
    print(f"position device: {position.device}")
    print(f"had_position device: {had_position.device}")
    
    # Now test the operation that fails in the real scenario
    try:
        done = (position == 0) & had_position
        print(f"Success! done device: {done.device}")
    except RuntimeError as e:
        print(f"Error in bitwise operation: {e}")
    
    # The real issue is that had_position doesn't get moved when .to() is called
    # because it's not registered as a buffer

if __name__ == "__main__":
    test_device_movement()