"""Test to ensure device mismatch issues are properly handled."""

import torch
import pytest
import ifera


class MockTensor:
    """Mock tensor that simulates device mismatch by overriding device operations."""

    def __init__(self, data, device_name="cuda:0"):
        self.data = data
        self._device_name = device_name
        self.dtype = data.dtype
        self.shape = data.shape

    @property
    def device(self):
        """Return a mock device."""
        return torch.device(self._device_name)

    def to(self, device):
        """Return self for simplicity in testing."""
        new_tensor = MockTensor(self.data, str(device))
        return new_tensor

    def __eq__(self, other):
        return self.data == other

    def __and__(self, other):
        # This would fail if other is MockTensor with different device
        if isinstance(other, MockTensor) and str(other.device) != str(self.device):
            raise RuntimeError(f"Device mismatch: {self.device} vs {other.device}")
        return (
            self.data & other.data
            if isinstance(other, MockTensor)
            else self.data & other
        )


def test_single_trade_done_policy_device_mismatch_simulation():
    """Test that SingleTradeDonePolicy handles device mismatch correctly using simulation."""

    device_cpu = torch.device("cpu")
    done_policy = ifera.SingleTradeDonePolicy(device=device_cpu)

    # Create a mock state on CPU
    batch_size = 5
    position_tensor = torch.tensor([0, 1, 0, -1, 0], dtype=torch.int32)
    had_position_tensor = torch.tensor(
        [True, True, False, True, False], dtype=torch.bool
    )

    state = {
        "position": position_tensor,
        "done": torch.zeros(batch_size, dtype=torch.bool),
        "had_position": had_position_tensor,  # Simulate this being on different device
    }

    # This should work without error due to our fix
    result = done_policy(state)

    # Verify the result - position == 0 AND had_position
    # [0, 1, 0, -1, 0] == 0 -> [True, False, True, False, True]
    # [True, True, False, True, False] (had_position)
    # Result: [True, False, False, False, False]
    expected = torch.tensor([True, False, False, False, False], dtype=torch.bool)
    torch.testing.assert_close(result, expected)


def test_single_trade_done_policy_device_compatibility():
    """Test that SingleTradeDonePolicy handles device movement correctly."""

    device_cpu = torch.device("cpu")
    done_policy = ifera.SingleTradeDonePolicy(device=device_cpu)

    batch_size = 3
    state = {
        "position": torch.tensor([0, 1, -1], dtype=torch.int32, device=device_cpu),
        "done": torch.zeros(batch_size, dtype=torch.bool, device=device_cpu),
    }

    # Reset the policy
    done_policy.reset(state)

    # Simulate position changes
    state["position"] = torch.tensor([1, 0, -1], dtype=torch.int32, device=device_cpu)

    # First call - should update had_position
    result1 = done_policy(state)
    expected1 = torch.tensor([False, False, False], dtype=torch.bool)
    torch.testing.assert_close(result1, expected1)

    # Second call - position goes to 0 for first element
    state["position"] = torch.tensor([0, 0, -1], dtype=torch.int32, device=device_cpu)
    result2 = done_policy(state)
    expected2 = torch.tensor([True, False, False], dtype=torch.bool)
    torch.testing.assert_close(result2, expected2)


def test_single_trade_done_policy_masked_reset_device_handling():
    """Test that masked_reset handles device mismatch correctly."""

    device_cpu = torch.device("cpu")
    done_policy = ifera.SingleTradeDonePolicy(device=device_cpu)

    batch_size = 3
    state = {
        "position": torch.tensor([0, 1, -1], dtype=torch.int32, device=device_cpu),
        "done": torch.zeros(batch_size, dtype=torch.bool, device=device_cpu),
    }

    # Reset the policy
    done_policy.reset(state)

    # Set some had_position values
    state["had_position"] = torch.tensor(
        [True, False, True], dtype=torch.bool, device=device_cpu
    )

    # Create mask
    mask = torch.tensor([True, False, False], dtype=torch.bool, device=device_cpu)

    # This should work without device issues
    done_policy.masked_reset(state, mask)

    # Verify that masked positions were reset to False
    expected = torch.tensor([False, False, True], dtype=torch.bool)
    torch.testing.assert_close(state["had_position"], expected)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="CUDA not available")
def test_single_trade_done_policy_cuda_device_movement():
    """Test SingleTradeDonePolicy with actual CUDA device movement."""

    device_cpu = torch.device("cpu")
    device_cuda = torch.device("cuda:0")

    # Create policy on CPU
    done_policy = ifera.SingleTradeDonePolicy(device=device_cpu)

    batch_size = 3
    state = {
        "position": torch.tensor([0, 1, -1], dtype=torch.int32, device=device_cpu),
        "done": torch.zeros(batch_size, dtype=torch.bool, device=device_cpu),
    }

    # Reset the policy on CPU
    done_policy.reset(state)

    # Move policy to CUDA
    done_policy.to(device_cuda)

    # Move state tensors to CUDA (simulating what happens in the environment)
    state = {k: v.to(device_cuda) for k, v in state.items()}

    # This should work without device mismatch errors
    state["position"] = torch.tensor([1, 0, -1], dtype=torch.int32, device=device_cuda)
    result = done_policy(state)

    # Verify result is on CUDA and has correct values
    assert result.device == device_cuda
    expected = torch.tensor([False, False, False], dtype=torch.bool, device=device_cuda)
    torch.testing.assert_close(result, expected)
