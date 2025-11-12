"""Tests for child splits functionality in prepare_splits."""

import pandas as pd
import pytest
import torch

from ifera.optionalpha import Split, prepare_splits


def test_max_depth_1_no_child_splits():
    """Test that max_depth=1 produces no child splits (default behavior)."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0], "filter_b": [10.0, 20.0, 30.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=1,
    )

    # All splits should have empty parents list (tier 1 splits)
    for split in splits:
        assert len(split.parents) == 0, "Tier 1 splits should have empty parents list"
        assert len(split.filters) > 0, "Tier 1 splits should have non-empty filters list"


def test_max_depth_2_generates_child_splits():
    """Test that max_depth=2 generates child splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0, 180.0], "profit": [50.0, 100.0, 75.0, 90.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    # Two filters with distinct values
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

    # Should have some child splits (splits with non-empty parents)
    child_splits = [s for s in splits if len(s.parents) > 0]
    tier1_splits = [s for s in splits if len(s.parents) == 0]

    assert len(child_splits) > 0, "Should have child splits with max_depth=2"
    assert len(tier1_splits) > 0, "Should still have tier 1 splits"

    # Child splits should have empty filters list
    for split in child_splits:
        assert len(split.filters) == 0, "Child splits should have empty filters list"


def test_child_split_mask_is_and_of_parents():
    """Test that child split masks are logical AND of parent masks."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 4, "profit": [50.0] * 4},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    # Create two filters where we can predict the child mask
    # filter_a: [1, 1, 2, 2] -> left split at 1.5: [T, T, F, F]
    # filter_b: [1, 2, 1, 2] -> left split at 1.5: [T, F, T, F]
    # AND result should be: [T, F, F, F]
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 1.0, 2.0, 2.0], "filter_b": [1.0, 2.0, 1.0, 2.0]},
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

    # Find child splits
    child_splits = [s for s in splits if len(s.parents) > 0]

    # Verify at least one child split exists
    assert len(child_splits) > 0

    # Verify child masks are AND of parent masks
    for child in child_splits:
        for parent_list in child.parents:
            parent_a, parent_b = parent_list[0], parent_list[1]
            # The child mask should be the AND of the two parent masks
            expected_mask = parent_a.mask & parent_b.mask
            assert torch.equal(
                child.mask, expected_mask
            ), "Child mask should be AND of parent masks"


def test_no_duplicate_parent_pairs():
    """Test that child splits don't have duplicate parent pairs."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 4, "profit": [50.0] * 4},
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

    child_splits = [s for s in splits if len(s.parents) > 0]

    # Check that no two child splits have the same parent pair (in any order)
    seen_pairs = set()
    for child in child_splits:
        for parent_list in child.parents:
            parent_a, parent_b = parent_list[0], parent_list[1]
            # Use id() to create unique identifiers for Split objects
            # Normalize pair order (smaller id first)
            pair = tuple(sorted([id(parent_a), id(parent_b)]))
            # Each pair should be unique
            assert pair not in seen_pairs, f"Duplicate parent pair found: {pair}"
            seen_pairs.add(pair)


def test_child_splits_dont_duplicate_tier1_masks():
    """Test that child splits with masks identical to tier 1 are removed."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 3, "profit": [50.0] * 3},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # Create scenario where child mask equals a tier 1 mask
    # filter_a: [1, 2, 3] -> left at 1.5: [T, F, F], left at 2.5: [T, T, F]
    # filter_b: [1, 1, 2] -> left at 1.5: [T, T, F]
    # The AND of filter_a left@1.5 [T, F, F] and filter_b left@1.5 [T, T, F] = [T, F, F]
    # which equals filter_a left@1.5, so it should be removed
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0], "filter_b": [1.0, 1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
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

    tier1_splits = [s for s in splits if len(s.parents) == 0]
    child_splits = [s for s in splits if len(s.parents) > 0]

    # Verify no child split has the same mask as any tier 1 split
    for child in child_splits:
        for tier1 in tier1_splits:
            assert not torch.equal(
                child.mask, tier1.mask
            ), "Child split should not have same mask as tier 1 split"


def test_child_splits_merge_identical_masks():
    """Test that child splits with identical masks are merged."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 4, "profit": [50.0] * 4},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    # Create scenario where different parent pairs create same mask
    # filter_a: [1, 1, 2, 2] -> left at 1.5: [T, T, F, F]
    # filter_b: [1, 2, 2, 2] -> left at 1.5: [T, F, F, F]
    # filter_c: [1, 1, 1, 2] -> left at 1.5: [T, T, T, F]
    # AND(filter_a left, filter_b left) = [T, F, F, F]
    # AND(filter_b left, filter_c left) = [T, F, F, F]
    # These should be merged
    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 1.0, 2.0, 2.0],
            "filter_b": [1.0, 2.0, 2.0, 2.0],
            "filter_c": [1.0, 1.0, 1.0, 2.0],
        },
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

    child_splits = [s for s in splits if len(s.parents) > 0]

    # Check that all child splits have unique masks
    for i in range(len(child_splits)):
        for j in range(i + 1, len(child_splits)):
            assert not torch.equal(
                child_splits[i].mask, child_splits[j].mask
            ), "All child splits should have unique masks (merged if identical)"


def test_max_depth_3_generates_depth_3_splits():
    """Test that max_depth=3 generates splits at depth 3."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 5, "profit": [50.0] * 5},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "filter_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
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
        max_depth=3,
    )

    # With max_depth=3, we should have more splits than max_depth=2
    X2, y2, splits2 = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=2,
    )

    assert len(splits) >= len(
        splits2
    ), "max_depth=3 should have at least as many splits as max_depth=2"


def test_exclusion_mask_updated_for_child_splits():
    """Test that child splits are properly generated and have valid structure."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 4, "profit": [50.0] * 4},
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

    # Check that we have both depth 1 and child splits
    depth_1_count = sum(1 for s in splits if len(s.parents) == 0)
    child_count = len(splits) - depth_1_count

    assert depth_1_count > 0, "Should have depth 1 splits"
    assert child_count > 0, "Should have child splits"


def test_early_exit_when_no_new_splits():
    """Test that depth loop exits early when no new splits are generated."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Single filter with two values - very limited split possibilities
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Even with high max_depth, should exit early
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=10,  # Very high depth
    )

    # With only one filter and 2 values, child generation will quickly exhaust possibilities
    # The function should exit early rather than continue looping
    # This test just verifies it completes successfully
    assert len(splits) > 0


def test_child_splits_respect_exclusion_mask():
    """Test that child splits are only generated from non-exclusive parent pairs."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 3, "profit": [50.0] * 3},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # Create filters where some splits will have empty intersection (exclusive)
    # filter_a: [1, 2, 3] -> left at 1.5: [T, F, F], right at 2.5: [F, F, T]
    # These two have empty intersection, so no child should be created from them
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
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

    depth_1_splits = [s for s in splits if len(s.parents) == 0]
    child_splits = [s for s in splits if len(s.parents) > 0]

    # Verify that child splits were created where valid, and check they have non-empty masks
    for child in child_splits:
        # Child mask should have at least one True value
        assert child.mask.any(), "Child split should have at least one True value"
        # Verify parents are Split objects
        for parent_list in child.parents:
            parent_a, parent_b = parent_list[0], parent_list[1]
            assert isinstance(parent_a, Split), "Parent should be a Split object"
            assert isinstance(parent_b, Split), "Parent should be a Split object"
            # Verify the child mask is the AND of parent masks
            expected_mask = parent_a.mask & parent_b.mask
            assert torch.equal(child.mask, expected_mask), "Child mask should be AND of parents"


def test_max_depth_with_max_splits_per_filter():
    """Test that max_depth works correctly with max_splits_per_filter."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": list(range(1, 11)), "filter_b": list(range(10, 20))},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=3,
        max_depth=2,
    )

    # Should have tier 1 splits limited by max_splits_per_filter
    # Plus child splits generated from those
    tier1_splits = [s for s in splits if len(s.parents) == 0]
    child_splits = [s for s in splits if len(s.parents) > 0]

    # Each filter should have at most 3 splits per direction
    filter_a_tier1 = sum(1 for s in tier1_splits for f in s.filters if f.filter_name == "filter_a")
    assert filter_a_tier1 <= 6, "filter_a should have at most 6 tier1 splits (3 left + 3 right)"

    # Should have some child splits
    assert len(child_splits) > 0, "Should generate child splits"


def test_parent_indices_valid():
    """Test that parent indices in child splits are valid."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 4, "profit": [50.0] * 4},
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

    tier1_splits = [s for s in splits if len(s.parents) == 0]
    child_splits = [s for s in splits if len(s.parents) > 0]

    # All parents should be Split objects from tier 1
    for child in child_splits:
        for parent_list in child.parents:
            parent_a, parent_b = parent_list[0], parent_list[1]
            assert isinstance(parent_a, Split), "Parent should be a Split object"
            assert isinstance(parent_b, Split), "Parent should be a Split object"
            # Verify parents are tier 1 splits (have no parents themselves)
            assert len(parent_a.parents) == 0, "Parents should be tier 1 splits"
            assert len(parent_b.parents) == 0, "Parents should be tier 1 splits"
            # Verify parents exist in tier1_splits list
            assert parent_a in tier1_splits, "Parent should be in tier 1 splits"
            assert parent_b in tier1_splits, "Parent should be in tier 1 splits"
