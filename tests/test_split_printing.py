"""Tests for Split class printing functionality."""

import pandas as pd
import torch

from ifera.optionalpha import Split, FilterInfo, prepare_splits


def test_split_str_simple_left_filter():
    """Test string representation of a simple split with left filter."""
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    filters = [FilterInfo(0, "filter_a", 1.5, "left")]
    parents = []

    split = Split(mask=mask, filters=filters, parents=parents)
    result = str(split)

    assert "Split filters:" in result
    assert "(filter_a <= 1.5)" in result


def test_split_str_simple_right_filter():
    """Test string representation of a simple split with right filter."""
    mask = torch.tensor([False, True, True], dtype=torch.bool)
    filters = [FilterInfo(0, "filter_b", 2.5, "right")]
    parents = []

    split = Split(mask=mask, filters=filters, parents=parents)
    result = str(split)

    assert "Split filters:" in result
    assert "(filter_b >= 2.5)" in result


def test_split_str_multiple_filters_or():
    """Test string representation with multiple filters (OR relationship)."""
    mask = torch.tensor([True, True, False], dtype=torch.bool)
    filters = [
        FilterInfo(0, "filter_a", 1.5, "left"),
        FilterInfo(1, "filter_b", 2.5, "left"),
    ]
    parents = []

    split = Split(mask=mask, filters=filters, parents=parents)
    result = str(split)

    # Multiple filters in the same split represent OR
    # Each should be on its own line
    lines = result.split("\n")
    assert len(lines) == 3  # "Split filters:" + 2 filter lines
    assert "(filter_a <= 1.5)" in result
    assert "(filter_b <= 2.5)" in result


def test_split_str_child_split_simple():
    """Test string representation of a child split (AND of two parents)."""
    # Create two parent splits
    parent_a_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    parent_a = Split(
        mask=parent_a_mask, filters=[FilterInfo(0, "filter_a", 1.5, "left")], parents=[]
    )

    parent_b_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    parent_b = Split(
        mask=parent_b_mask,
        filters=[FilterInfo(1, "filter_b", 2.5, "right")],
        parents=[],
    )

    # Create child split (AND of parent_a and parent_b)
    child_mask = parent_a_mask & parent_b_mask
    child = Split(mask=child_mask, filters=[], parents=[(parent_a, parent_b)])

    result = str(child)

    # Child should show AND relationship
    assert "Split filters:" in result
    assert "(filter_a <= 1.5) & (filter_b >= 2.5)" in result


def test_split_str_child_split_with_or_parents():
    """Test child split where one parent has multiple filters (OR)."""
    # Parent A has two filters (OR relationship)
    parent_a_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    parent_a = Split(
        mask=parent_a_mask,
        filters=[
            FilterInfo(0, "filter_a1", 1.5, "left"),
            FilterInfo(1, "filter_a2", 3.5, "left"),
        ],
        parents=[],
    )

    # Parent B has one filter
    parent_b_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    parent_b = Split(
        mask=parent_b_mask,
        filters=[FilterInfo(2, "filter_b", 2.5, "right")],
        parents=[],
    )

    # Create child split
    child_mask = parent_a_mask & parent_b_mask
    child = Split(mask=child_mask, filters=[], parents=[(parent_a, parent_b)])

    result = str(child)

    # Should expand to DNF: (A1 OR A2) AND B = (A1 AND B) OR (A2 AND B)
    assert "Split filters:" in result
    lines = result.strip().split("\n")
    # Should have "Split filters:" + 2 conjunctions
    assert len(lines) == 3
    assert "(filter_a1 <= 1.5) & (filter_b >= 2.5)" in result
    assert "(filter_a2 <= 3.5) & (filter_b >= 2.5)" in result


def test_split_str_multiple_parent_pairs():
    """Test child split with multiple parent pairs (OR relationship)."""
    # Create parent splits
    parent_a = Split(
        mask=torch.tensor([True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_a", 1.0, "left")],
        parents=[],
    )

    parent_b = Split(
        mask=torch.tensor([False, True], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_b", 2.0, "right")],
        parents=[],
    )

    parent_c = Split(
        mask=torch.tensor([True, False], dtype=torch.bool),
        filters=[FilterInfo(2, "filter_c", 3.0, "left")],
        parents=[],
    )

    # Child with multiple parent pairs
    child = Split(
        mask=torch.tensor([False, False], dtype=torch.bool),
        filters=[],
        parents=[(parent_a, parent_b), (parent_c, parent_b)],
    )

    result = str(child)

    # Should show both combinations (OR relationship between parent pairs)
    lines = result.strip().split("\n")
    assert len(lines) == 3  # "Split filters:" + 2 conjunctions
    assert "(filter_a <= 1) & (filter_b >= 2)" in result
    # Note: terms are sorted by filter_idx, so filter_b (idx=1) comes before filter_c (idx=2)
    assert "(filter_b >= 2) & (filter_c <= 3)" in result


def test_split_str_nested_parents():
    """Test child split with nested parent structure (depth 3)."""
    # Create depth 1 splits
    split_a = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_a", 1.0, "left")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_b", 2.0, "right")],
        parents=[],
    )

    split_c = Split(
        mask=torch.tensor([True, True, True], dtype=torch.bool),
        filters=[FilterInfo(2, "filter_c", 3.0, "left")],
        parents=[],
    )

    # Create depth 2 split: A AND B
    split_d = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b)],
    )

    # Create depth 3 split: (A AND B) AND C
    split_e = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split_d, split_c)],
    )

    result = str(split_e)

    # Should expand to: A AND B AND C
    assert "Split filters:" in result
    assert "(filter_a <= 1) & (filter_b >= 2) & (filter_c <= 3)" in result


def test_split_str_with_prepare_splits():
    """Test printing splits created by prepare_splits function."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0, 180.0], "profit": [50.0, 100.0, 75.0, 90.0]},
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

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=2,
    )

    # Find a depth 1 split
    depth_1_splits = [s for s in splits if len(s.parents) == 0]
    assert len(depth_1_splits) > 0

    # Test printing a depth 1 split
    result = str(depth_1_splits[0])
    assert "Split filters:" in result
    # Should have filter name, operator, and threshold
    assert (
        "filter_" in result
        or "is_" in result
        or "reward_per_risk" in result
        or "open_minutes" in result
    )

    # Find a child split if any
    child_splits = [s for s in splits if len(s.parents) > 0]
    if len(child_splits) > 0:
        result = str(child_splits[0])
        assert "Split filters:" in result
        # Child split should have "&" in it (changed from "and")
        assert " & " in result


def test_split_str_empty_split():
    """Test string representation of an empty split (edge case)."""
    mask = torch.tensor([False, False], dtype=torch.bool)
    split = Split(mask=mask, filters=[], parents=[])

    result = str(split)

    assert "Split filters:" in result
    assert "(empty)" in result


def test_print_split():
    """Test that print() works on Split objects."""
    mask = torch.tensor([True, False], dtype=torch.bool)
    filters = [FilterInfo(0, "test_filter", 5.0, "left")]
    split = Split(mask=mask, filters=filters, parents=[])

    # Should not raise any exception
    result = str(split)
    assert result is not None
    assert len(result) > 0
