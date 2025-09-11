"""Test to reproduce and verify fix for Torch Dynamo graph break issue."""

import torch
import pytest
from ifera.state import State
from ifera.policies.open_position_policy import OpenOncePolicy
from ifera.policies.trading_done_policy import SingleTradeDonePolicy


def test_torch_dynamo_compilation_with_state():
    """Test that policies work with torch.compile using State instead of dict."""

    # Create a simple compiled function that uses policies with state mutation
    @torch.compile
    def test_policy_mutation(state, policy):
        no_position_mask = torch.tensor([True, True])
        action = policy.forward(state, no_position_mask)
        return action

    # Test OpenOncePolicy which mutates state.opened
    state = State.create(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        start_date_idx=torch.tensor([0, 1]),
        start_time_idx=torch.tensor([0, 0]),
    )

    policy = OpenOncePolicy(direction=1, device=torch.device("cpu"))
    policy.reset(state, 2, torch.device("cpu"))

    # This should work without Dynamo graph break
    try:
        action = test_policy_mutation(state, policy)
        assert action.shape == (2,)
        assert state.opened.all()  # Should be set to True after mutation
        print("✓ OpenOncePolicy compilation test passed!")
    except Exception as e:
        pytest.fail(f"OpenOncePolicy compilation failed: {e}")

    # Test SingleTradeDonePolicy which mutates state.had_position
    @torch.compile
    def test_done_policy_mutation(state, policy):
        return policy.forward(state)

    state2 = State.create(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        start_date_idx=torch.tensor([0, 1]),
        start_time_idx=torch.tensor([0, 0]),
    )
    state2.position = torch.tensor(
        [0, 1], dtype=torch.int32
    )  # One position, one no position

    done_policy = SingleTradeDonePolicy(device=torch.device("cpu"))
    done_policy.reset(state2, 2, torch.device("cpu"))

    # Set had_position after reset (reset sets it to False)
    state2.had_position = torch.tensor([True, False])  # One had position

    try:
        done = test_done_policy_mutation(state2, done_policy)
        assert done.shape == (2,)
        assert done[0].item() is True  # position=0 and had_position=True -> done
        assert done[1].item() is False  # position=1 and had_position=False -> not done
        print("✓ SingleTradeDonePolicy compilation test passed!")
    except Exception as e:
        pytest.fail(f"SingleTradeDonePolicy compilation failed: {e}")


def test_dict_state_would_fail():
    """Demonstrate that the old dict-based approach would fail with torch.compile."""

    # This would fail with the old dict-based state
    def dict_based_mutation(state_dict):
        # This type of mutation would cause:
        # UncapturedHigherOrderOpError: Mutating a variable not in the current scope
        state_dict["opened"] = state_dict.get(
            "opened", torch.zeros(2, dtype=torch.bool)
        ) | torch.tensor([True, True])
        return state_dict["opened"]

    # Note: We can't actually test the failure because our code no longer uses dicts,
    # but this demonstrates the problematic pattern that we fixed
    print("✓ Dict-based mutation pattern identified (would fail with torch.compile)")


if __name__ == "__main__":
    test_torch_dynamo_compilation_with_state()
    test_dict_state_would_fail()
    print("All Dynamo compilation tests passed!")
