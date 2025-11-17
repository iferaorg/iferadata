"""Tests for the SplitTensorState class and tensor-based operations."""

import pandas as pd
import torch

from ifera.optionalpha import (
    FilterInfo,
    Split,
    SplitTensorState,
    _build_tensor_state_from_splits,
    _combine_dnf_with_and,
    _generate_child_splits_tensor_based,
    _merge_dnf_with_or,
    _splits_from_tensor_state,
)


def test_split_tensor_state_initialization():
    """Test that SplitTensorState can be initialized correctly."""
    device = torch.device("cpu")
    masks = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool, device=device
    )
    scores = torch.tensor([0.5, 0.8], dtype=torch.float32, device=device)
    dnf = [[[0]], [[1]]]  # Two splits, each with one literal
    all_literals = [
        FilterInfo(0, "filter_a", 1.5, "left"),
        FilterInfo(1, "filter_b", 2.0, "right"),
    ]

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals, split_objects=None
    )

    assert state.masks.shape == (2, 3)
    assert state.scores.shape == (2,)
    assert len(state.dnf) == 2
    assert len(state.all_literals) == 2
    assert state.split_objects is None


def test_split_tensor_state_get_split_depth1():
    """Test lazy Split construction from SplitTensorState for depth-1 splits."""
    device = torch.device("cpu")
    masks = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool, device=device
    )
    scores = torch.tensor([0.5, 0.8], dtype=torch.float32, device=device)
    dnf = [[[0]], [[1]]]  # Two splits, each with one literal
    all_literals = [
        FilterInfo(0, "filter_a", 1.5, "left"),
        FilterInfo(1, "filter_b", 2.0, "right"),
    ]

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals, split_objects=None
    )

    # Get split at index 0
    split = state.get_split(0)

    assert isinstance(split, Split)
    assert split.score == 0.5
    assert len(split.filters) == 1
    assert split.filters[0] == all_literals[0]
    assert len(split.parents) == 0


def test_split_tensor_state_get_split_child():
    """Test lazy Split construction from SplitTensorState for child splits."""
    device = torch.device("cpu")
    # Child split with AND of two literals
    masks = torch.tensor([[True, False, False]], dtype=torch.bool, device=device)
    scores = torch.tensor([0.9], dtype=torch.float32, device=device)
    dnf = [[[0, 1]]]  # One child split: literal 0 AND literal 1
    all_literals = [
        FilterInfo(0, "filter_a", 1.5, "left"),
        FilterInfo(1, "filter_b", 2.0, "right"),
    ]

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals, split_objects=None
    )

    # Get child split
    split = state.get_split(0)

    assert isinstance(split, Split)
    assert abs(split.score - 0.9) < 1e-6  # Use approximate comparison for floats
    assert len(split.filters) == 0  # Child splits have empty filters
    assert len(split.parents) == 1  # One parent set (conjunction)
    assert len(split.parents[0]) == 2  # Two parent splits in the set


def test_combine_dnf_with_and():
    """Test combining two DNF formulas with AND operation."""
    # dnf_a: (literal 0) OR (literal 1)
    dnf_a = [[0], [1]]
    # dnf_b: (literal 2)
    dnf_b = [[2]]

    result = _combine_dnf_with_and(dnf_a, dnf_b)

    # Expected: (literal 0 AND literal 2) OR (literal 1 AND literal 2)
    assert len(result) == 2
    assert sorted(result[0]) == [0, 2] or sorted(result[0]) == [1, 2]
    assert sorted(result[1]) == [0, 2] or sorted(result[1]) == [1, 2]


def test_combine_dnf_with_and_deduplication():
    """Test that combining DNF formulas removes duplicates."""
    # dnf_a: (literal 0)
    dnf_a = [[0]]
    # dnf_b: (literal 0)
    dnf_b = [[0]]

    result = _combine_dnf_with_and(dnf_a, dnf_b)

    # Expected: (literal 0) - no duplicates
    assert len(result) == 1
    assert result[0] == [0]


def test_merge_dnf_with_or():
    """Test merging multiple DNF formulas with OR operation."""
    dnf1 = [[0], [1]]  # (literal 0) OR (literal 1)
    dnf2 = [[2]]  # (literal 2)
    dnf3 = [[0]]  # (literal 0) - duplicate

    result = _merge_dnf_with_or([dnf1, dnf2, dnf3])

    # Expected: (literal 0) OR (literal 1) OR (literal 2) - no duplicates
    assert len(result) == 3
    literals = [conj[0] for conj in result if len(conj) == 1]
    assert sorted(literals) == [0, 1, 2]


def test_build_tensor_state_from_splits():
    """Test building SplitTensorState from a list of Split objects."""
    device = torch.device("cpu")
    mask1 = torch.tensor([True, False, True], dtype=torch.bool, device=device)
    mask2 = torch.tensor([False, True, False], dtype=torch.bool, device=device)

    filter1 = FilterInfo(0, "filter_a", 1.5, "left")
    filter2 = FilterInfo(1, "filter_b", 2.0, "right")

    split1 = Split(mask=mask1, filters=[filter1], parents=[])
    split1.score = 0.5
    split2 = Split(mask=mask2, filters=[filter2], parents=[])
    split2.score = 0.8

    splits = [split1, split2]

    state = _build_tensor_state_from_splits(splits, device)

    assert state.masks.shape == (2, 3)
    assert state.scores.shape == (2,)
    assert len(state.dnf) == 2
    assert len(state.all_literals) == 2
    assert state.split_objects == splits


def test_splits_from_tensor_state():
    """Test converting SplitTensorState back to list of Split objects."""
    device = torch.device("cpu")
    masks = torch.tensor([[True, False, True]], dtype=torch.bool, device=device)
    scores = torch.tensor([0.5], dtype=torch.float32, device=device)
    dnf = [[[0]]]
    all_literals = [FilterInfo(0, "filter_a", 1.5, "left")]

    # Create state with pre-built split objects
    mask = torch.tensor([True, False, True], dtype=torch.bool, device=device)
    split = Split(mask=mask, filters=[all_literals[0]], parents=[])
    split.score = 0.5
    splits = [split]

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals, split_objects=splits
    )

    result_splits = _splits_from_tensor_state(state)

    assert len(result_splits) == 1
    assert result_splits[0] == split


def test_splits_from_tensor_state_lazy():
    """Test lazy construction of Split objects from SplitTensorState."""
    device = torch.device("cpu")
    masks = torch.tensor([[True, False, True]], dtype=torch.bool, device=device)
    scores = torch.tensor([0.5], dtype=torch.float32, device=device)
    dnf = [[[0]]]
    all_literals = [FilterInfo(0, "filter_a", 1.5, "left")]

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals, split_objects=None
    )

    result_splits = _splits_from_tensor_state(state)

    assert len(result_splits) == 1
    assert isinstance(result_splits[0], Split)
    assert result_splits[0].score == 0.5


def test_generate_child_splits_tensor_based():
    """Test tensor-based child split generation."""
    device = torch.device("cpu")

    # Create two depth-1 splits
    mask1 = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    mask2 = torch.tensor([True, False, True], dtype=torch.bool, device=device)

    filter1 = FilterInfo(0, "filter_a", 1.5, "left")
    filter2 = FilterInfo(1, "filter_b", 2.0, "right")

    split1 = Split(mask=mask1, filters=[filter1], parents=[])
    split2 = Split(mask=mask2, filters=[filter2], parents=[])

    splits = [split1, split2]
    state = _build_tensor_state_from_splits(splits, device)

    # Generate child splits (combining split1 and split2)
    child_masks, valid_pairs, child_dnf_list = _generate_child_splits_tensor_based(
        state_previous=state,
        state_depth1=state,
        previous_indices=None,
        depth1_indices=None,
        device=device,
        min_samples=1,
    )

    # Should generate one child split: split1 AND split2
    # Expected mask: [True, False, False] (intersection)
    assert child_masks.shape[0] > 0  # At least one child
    assert child_masks.shape[1] == 3  # Same number of samples

    # Check that child DNF is correct
    assert len(child_dnf_list) > 0


def test_generate_child_splits_tensor_based_exclusion():
    """Test that mutually exclusive splits don't generate children."""
    device = torch.device("cpu")

    # Create two mutually exclusive splits (no overlap)
    mask1 = torch.tensor([True, False, False], dtype=torch.bool, device=device)
    mask2 = torch.tensor([False, True, True], dtype=torch.bool, device=device)

    filter1 = FilterInfo(0, "filter_a", 1.5, "left")
    filter2 = FilterInfo(1, "filter_b", 2.0, "right")

    split1 = Split(mask=mask1, filters=[filter1], parents=[])
    split2 = Split(mask=mask2, filters=[filter2], parents=[])

    splits = [split1, split2]
    state = _build_tensor_state_from_splits(splits, device)

    # Generate child splits with min_samples=2
    child_masks, valid_pairs, child_dnf_list = _generate_child_splits_tensor_based(
        state_previous=state,
        state_depth1=state,
        previous_indices=None,
        depth1_indices=None,
        device=device,
        min_samples=2,
    )

    # Should generate no children (mutually exclusive with min_samples=2)
    assert child_masks.shape[0] == 0
    assert len(child_dnf_list) == 0
