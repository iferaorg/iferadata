"""Tests for max_splits_per_filter parameter in prepare_splits."""

import pandas as pd
import pytest
import torch

from ifera.optionalpha import prepare_splits


def test_max_splits_per_filter_none():
    """Test that max_splits_per_filter=None creates all possible splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    # Filter with 10 unique values -> 9 possible splits
    filters_df = pd.DataFrame(
        {"filter_a": list(range(1, 11))},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=None,
    )

    # Count splits for filter_a
    filter_a_left = 0
    filter_a_right = 0
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a":
                if direction == "left":
                    filter_a_left += 1
                elif direction == "right":
                    filter_a_right += 1

    # With 10 unique values, we should have 9 thresholds
    # Each threshold creates a left and right split
    assert filter_a_left == 9
    assert filter_a_right == 9


def test_max_splits_per_filter_basic():
    """Test that max_splits_per_filter limits the number of splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    # Filter with 10 unique values -> 9 possible splits
    filters_df = pd.DataFrame(
        {"filter_a": list(range(1, 11))},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    max_splits = 3
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=max_splits,
    )

    # Count splits for filter_a
    filter_a_left = 0
    filter_a_right = 0
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a":
                if direction == "left":
                    filter_a_left += 1
                elif direction == "right":
                    filter_a_right += 1

    # Should have at most max_splits left splits and max_splits right splits
    assert filter_a_left <= max_splits
    assert filter_a_right <= max_splits


def test_max_splits_per_filter_even_distribution():
    """Test that max_splits_per_filter distributes samples evenly."""
    # Create 20 samples with values 1-10 (2 samples per value)
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 20, "profit": [50.0] * 20},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 21)], name="date"),
    )

    # Each value appears twice
    filters_df = pd.DataFrame(
        {"filter_a": [i // 2 + 1 for i in range(20)]},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 21)], name="date"),
    )

    max_splits = 3
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=max_splits,
    )

    # Check that left splits distribute samples roughly evenly
    filter_a_left_splits = []
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a" and direction == "left":
                filter_a_left_splits.append((threshold, split.mask))

    # With 3 splits, we create 4 buckets
    # With 20 samples, ideal distribution is 5 samples per bucket
    if len(filter_a_left_splits) == 3:
        # Sort by threshold
        filter_a_left_splits.sort(key=lambda x: x[0])

        # Count samples in each bucket
        bucket_sizes = []
        prev_mask = torch.zeros(20, dtype=torch.bool)
        for threshold, mask in filter_a_left_splits:
            bucket_size = (mask & ~prev_mask).sum().item()
            bucket_sizes.append(bucket_size)
            prev_mask = mask

        # Last bucket (samples not in any left split)
        last_bucket_size = (~prev_mask).sum().item()
        bucket_sizes.append(last_bucket_size)

        # Check that buckets are reasonably balanced
        # Allow some variation due to the constraint that equal values stay together
        min_size = min(bucket_sizes)
        max_size = max(bucket_sizes)
        # With 20 samples and 4 buckets, ideal is 5 per bucket
        # Allow range of [2, 8] to account for grouping constraint
        assert min_size >= 2, f"Bucket too small: {bucket_sizes}"
        assert max_size <= 8, f"Bucket too large: {bucket_sizes}"


def test_max_splits_per_filter_respects_equal_values():
    """Test that samples with equal filter values stay together."""
    # Create samples where many have the same value
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 15, "profit": [50.0] * 15},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 16)], name="date"),
    )

    # Values: [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
    # 3 unique values, so 2 possible splits
    filters_df = pd.DataFrame(
        {"filter_a": [1] * 5 + [2] * 5 + [3] * 5},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 16)], name="date"),
    )

    max_splits = 1
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=max_splits,
    )

    # Find the filter_a left split
    filter_a_left_split = None
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a" and direction == "left":
                filter_a_left_split = split
                break

    assert filter_a_left_split is not None

    # Verify that all samples with value 1 are either all in or all out
    filter_values = torch.tensor(filters_df["filter_a"].values, dtype=torch.float32)
    mask = filter_a_left_split.mask

    # Check value 1
    val1_mask = filter_values == 1
    val1_in_split = mask[val1_mask]
    assert val1_in_split.all() or (~val1_in_split).all()

    # Check value 2
    val2_mask = filter_values == 2
    val2_in_split = mask[val2_mask]
    assert val2_in_split.all() or (~val2_in_split).all()

    # Check value 3
    val3_mask = filter_values == 3
    val3_in_split = mask[val3_mask]
    assert val3_in_split.all() or (~val3_in_split).all()


def test_max_splits_per_filter_multiple_filters():
    """Test that max_splits_per_filter applies independently to each filter."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    # Two filters with different numbers of unique values
    filters_df = pd.DataFrame(
        {
            "filter_a": list(range(1, 11)),  # 10 unique values -> 9 possible splits
            "filter_b": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
            ],  # 5 unique values -> 4 possible splits
        },
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    max_splits = 2
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=max_splits,
    )

    # Count splits for each filter
    filter_a_left = 0
    filter_b_left = 0
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if direction == "left":
                if filter_name == "filter_a":
                    filter_a_left += 1
                elif filter_name == "filter_b":
                    filter_b_left += 1

    # filter_a has 9 possible splits, should be limited to 2
    assert filter_a_left <= max_splits
    # filter_b has 4 possible splits, should be limited to 2
    assert filter_b_left <= max_splits


def test_max_splits_per_filter_less_than_possible():
    """Test behavior when max_splits_per_filter is less than possible splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 5, "profit": [50.0] * 5},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    # 5 unique values -> 4 possible splits
    filters_df = pd.DataFrame(
        {"filter_a": [1, 2, 3, 4, 5]},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    max_splits = 2
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=max_splits,
    )

    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )

    assert filter_a_left == max_splits


def test_max_splits_per_filter_more_than_possible():
    """Test behavior when max_splits_per_filter is more than possible splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 3, "profit": [50.0] * 3},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 4)], name="date"),
    )

    # 3 unique values -> 2 possible splits
    filters_df = pd.DataFrame(
        {"filter_a": [1, 2, 3]},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 4)], name="date"),
    )

    max_splits = 5  # More than the 2 possible splits
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=max_splits,
    )

    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )

    # Should use all 2 possible splits, not limited by max_splits
    assert filter_a_left == 2


def test_max_splits_per_filter_with_left_only():
    """Test that max_splits_per_filter works with left_only_filters."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": list(range(1, 11))},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    max_splits = 3
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        left_only_filters=["filter_a"],
        right_only_filters=[],
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_splits_per_filter=max_splits,
    )

    filter_a_left = 0
    filter_a_right = 0
    for split in splits:
        for _, fname, _, direction in split.filters:
            if fname == "filter_a":
                if direction == "left":
                    filter_a_left += 1
                elif direction == "right":
                    filter_a_right += 1

    # Should have at most max_splits left splits
    assert filter_a_left <= max_splits
    # Should have no right splits (left_only_filters)
    assert filter_a_right == 0


def test_max_splits_per_filter_with_right_only():
    """Test that max_splits_per_filter works with right_only_filters."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": list(range(1, 11))},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    max_splits = 3
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        left_only_filters=[],
        right_only_filters=["filter_a"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_splits_per_filter=max_splits,
    )

    filter_a_left = 0
    filter_a_right = 0
    for split in splits:
        for _, fname, _, direction in split.filters:
            if fname == "filter_a":
                if direction == "left":
                    filter_a_left += 1
                elif direction == "right":
                    filter_a_right += 1

    # Should have no left splits (right_only_filters)
    assert filter_a_left == 0
    # Should have at most max_splits right splits
    assert filter_a_right <= max_splits


def test_max_splits_per_filter_uneven_value_distribution():
    """Test with highly uneven distribution of values."""
    # Many samples with value 1, few with other values
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 25, "profit": [50.0] * 25},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 26)], name="date"),
    )

    # 20 samples with value 1, 1 sample each for values 2-6
    filters_df = pd.DataFrame(
        {"filter_a": [1] * 20 + [2, 3, 4, 5, 6]},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 26)], name="date"),
    )

    max_splits = 3
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=max_splits,
    )

    # Count splits
    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )

    # Should create at most max_splits splits
    assert filter_a_left <= max_splits

    # Verify that all samples with value 1 stay together
    filter_values = torch.tensor(filters_df["filter_a"].values, dtype=torch.float32)
    for split in splits:
        for _, fname, _, direction in split.filters:
            if fname == "filter_a" and direction == "left":
                mask = split.mask
                val1_mask = filter_values == 1
                val1_in_split = mask[val1_mask]
                # All value-1 samples should be consistently in or out
                assert val1_in_split.all() or (~val1_in_split).all()
