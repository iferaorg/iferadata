"""Test to reproduce the device mismatch issue."""

import torch
import ifera

def test_device_mismatch():
    """Test that reproduces the device mismatch issue."""
    
    # Skip if no CUDA devices available
    if torch.cuda.device_count() < 2:
        print("Skipping test - need at least 2 CUDA devices")
        return
    
    # Create policy on cuda:0
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    done_policy = ifera.SingleTradeDonePolicy(device=device0)
    
    # Create a mock state on device1
    batch_size = 10
    state = {
        "position": torch.zeros(batch_size, dtype=torch.int32, device=device1),
        "done": torch.zeros(batch_size, dtype=torch.bool, device=device1),
    }
    
    # Reset the policy - this adds had_position to state on device0
    done_policy.reset(state)
    
    # Move policy to device1
    done_policy.to(device1)
    
    # Now the position is on device1, but had_position is still on device0
    print(f"position device: {state['position'].device}")
    print(f"had_position device: {state['had_position'].device}")
    
    # This should trigger the device mismatch error
    try:
        result = done_policy(state)
        print("No error occurred")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_device_mismatch()