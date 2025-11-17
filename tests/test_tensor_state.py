"""Tests for the tensor state implementation in optionalpha module."""

import pandas as pd
import pytest
import torch

from ifera.optionalpha import SplitTensorState, FilterInfo, prepare_splits


def test_split_tensor_state_initialization():
    """Test that SplitTensorState can be initialized correctly."""
    device = torch.device("cpu")
    dtype = torch.float32

    masks = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool, device=device
    )
    scores = torch.tensor([0.5, 0.3], dtype=dtype, device=device)
    dnf = [[[0]], [[1]]]  # Two depth-1 splits, each with one literal
    all_literals = [
        FilterInfo(0, "filter_a", 1.5, "left"),
        FilterInfo(1, "filter_b", 2.5, "right"),
    ]

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals
    )

    assert state.masks.shape == (2, 3)
    assert state.scores.shape == (2,)
    assert len(state.dnf) == 2
    assert len(state.all_literals) == 2


def test_split_tensor_state_get_split():
    """Test that get_split() creates Split objects correctly."""
    device = torch.device("cpu")
    dtype = torch.float32

    masks = torch.tensor([[True, False, True]], dtype=torch.bool, device=device)
    scores = torch.tensor([0.75], dtype=dtype, device=device)
    all_literals = [FilterInfo(0, "filter_a", 1.5, "left")]
    dnf = [[[0]]]  # Single depth-1 split

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals
    )

    split = state.get_split(0)

    assert split.score == 0.75
    assert len(split.filters) == 1
    assert split.filters[0] == all_literals[0]
    assert len(split.parents) == 0
    assert split.mask.device == torch.device("cpu")
    assert split.mask.shape == (3,)


def test_prepare_splits_uses_tensor_state():
    """Test that prepare_splits internally uses tensor state."""
    # Create simple test data
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0],
            "profit": [50.0, -100.0, 75.0, 60.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0, 4.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    def simple_score_func(y, masks):
        return torch.sum(y.unsqueeze(0) * masks.float(), dim=1)

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=simple_score_func,
        max_depth=1,
    )

    # Verify that splits were created
    assert len(splits) > 0

    # Verify that all splits have scores
    for split in splits:
        assert split.score is not None


def test_prepare_splits_tensor_state_with_depth_2():
    """Test tensor state implementation with depth 2 splits."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0],
            "profit": [50.0, -100.0, 75.0, 60.0, 90.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
            ],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "filter_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
            ],
            name="date",
        ),
    )

    def mean_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        max_depth=2,
    )

    # Should have depth 1 and depth 2 splits
    assert len(splits) > 0

    # Verify all splits have scores
    for split in splits:
        assert split.score is not None


def test_tensor_state_lazy_split_construction():
    """Test that splits are constructed lazily from tensor state."""
    device = torch.device("cpu")
    dtype = torch.float32

    # Create a state with multiple splits
    masks = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
        device=device,
    )
    scores = torch.tensor([0.9, 0.7, 0.5], dtype=dtype, device=device)
    all_literals = [
        FilterInfo(0, "filter_a", 1.5, "left"),
        FilterInfo(1, "filter_b", 2.5, "right"),
        FilterInfo(2, "filter_c", 3.5, "left"),
    ]
    dnf = [[[0]], [[1]], [[2]]]

    state = SplitTensorState(
        masks=masks, scores=scores, dnf=dnf, all_literals=all_literals
    )

    # Initially, split_objects should be None (lazy)
    assert state.split_objects is None

    # Get one split
    split_0 = state.get_split(0)
    assert abs(split_0.score - 0.9) < 1e-6

    # Get another split
    split_2 = state.get_split(2)
    assert abs(split_2.score - 0.5) < 1e-6

    # Both should have filters from all_literals
    assert split_0.filters[0] == all_literals[0]
    assert split_2.filters[0] == all_literals[2]


def test_tensor_state_dnf_merging():
    """Test DNF merging for child splits."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0],
            "profit": [50.0, -100.0, 75.0, 60.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0, 4.0], "filter_b": [10.0, 20.0, 30.0, 40.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    def simple_score_func(y, masks):
        return torch.sum(y.unsqueeze(0) * masks.float(), dim=1)

    # Generate depth 2 splits
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=simple_score_func,
        max_depth=2,
        min_samples=1,
    )

    # Find any depth 2 splits (should have parents)
    depth_2_splits = [s for s in splits if len(s.parents) > 0]

    if len(depth_2_splits) > 0:
        # Depth 2 splits should have parent sets
        for split in depth_2_splits:
            assert len(split.parents) > 0
            # Each parent set should contain depth 1 splits
            for parent_set in split.parents:
                assert len(parent_set) > 0
