"""Tests for min_samples parameter in prepare_splits."""

import pandas as pd
import pytest
import torch

from ifera.optionalpha import prepare_splits


def test_min_samples_default():
    """Test that min_samples defaults to 1."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 5, "profit": [50.0] * 5},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    # 5 unique values -> 4 possible splits
    filters_df = pd.DataFrame(
        {"filter_a": [1, 2, 3, 4, 5]},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
    )

    # With default min_samples=1, all splits should be created
    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )
    filter_a_right = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "right"
    )

    # With 5 unique values, we should have 4 thresholds
    assert filter_a_left == 4
    assert filter_a_right == 4


def test_min_samples_2():
    """Test that min_samples=2 filters out splits with fewer samples."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 5, "profit": [50.0] * 5},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    # 5 unique values -> 4 possible splits
    filters_df = pd.DataFrame(
        {"filter_a": [1, 2, 3, 4, 5]},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        min_samples=2,
    )

    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )
    filter_a_right = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "right"
    )

    # With min_samples=2:
    # Left splits: need at least 2 samples on left side
    # - split at 1.5: left has 1 sample -> EXCLUDED
    # - split at 2.5: left has 2 samples -> OK
    # - split at 3.5: left has 3 samples -> OK
    # - split at 4.5: left has 4 samples -> OK
    assert filter_a_left == 3

    # Right splits: need at least 2 samples on right side
    # - split at 1.5: right has 4 samples -> OK
    # - split at 2.5: right has 3 samples -> OK
    # - split at 3.5: right has 2 samples -> OK
    # - split at 4.5: right has 1 sample -> EXCLUDED
    assert filter_a_right == 3


def test_min_samples_asymmetric_thresholds():
    """Test that left and right splits can have different thresholds."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    # 10 unique values
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
        min_samples=3,
    )

    # Collect left and right thresholds separately
    left_thresholds = []
    right_thresholds = []
    for split in splits:
        for _, fname, threshold, direction in split.filters:
            if fname == "filter_a":
                if direction == "left":
                    left_thresholds.append(threshold)
                elif direction == "right":
                    right_thresholds.append(threshold)

    # Left splits need at least 3 samples on left
    # Valid splits: 2.5 (3 left), 3.5 (4 left), ..., 9.5 (9 left)
    # Invalid: 1.5 (1 left), (actually we'd get splits starting from min_samples)
    assert len(left_thresholds) == 7  # splits from 3.5 to 9.5

    # Right splits need at least 3 samples on right
    # Valid splits: 1.5 (9 right), 2.5 (8 right), ..., 7.5 (3 right)
    # Invalid: 8.5 (2 right), 9.5 (1 right)
    assert len(right_thresholds) == 7  # splits from 1.5 to 7.5

    # Verify thresholds are different
    assert min(left_thresholds) != min(right_thresholds)
    assert max(left_thresholds) != max(right_thresholds)


def test_min_samples_with_repeated_values():
    """Test min_samples with repeated values."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 15, "profit": [50.0] * 15},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 16)], name="date"),
    )

    # 3 unique values, 5 samples each
    filters_df = pd.DataFrame(
        {"filter_a": [1] * 5 + [2] * 5 + [3] * 5},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 16)], name="date"),
    )

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        min_samples=6,
    )

    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )
    filter_a_right = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "right"
    )

    # With min_samples=6:
    # Left splits: need at least 6 samples on left side
    # - split at 1.5: left has 5 samples -> EXCLUDED
    # - split at 2.5: left has 10 samples -> OK
    assert filter_a_left == 1

    # Right splits: need at least 6 samples on right side
    # - split at 1.5: right has 10 samples -> OK
    # - split at 2.5: right has 5 samples -> EXCLUDED
    assert filter_a_right == 1


def test_min_samples_exclusion_mask():
    """Test that exclusion mask excludes splits with insufficient samples."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 5, "profit": [50.0] * 5},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1, 2, 3, 4, 5]},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    # With min_samples=1, all splits should be valid
    X1, y1, splits1, exclusion_mask1 = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        min_samples=1,
    )

    # With min_samples=3, some splits should be excluded
    X2, y2, splits2, exclusion_mask2 = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        min_samples=3,
    )

    # Check that splits with fewer than 3 samples are not created
    for split in splits2:
        sample_count = split.mask.sum().item()
        assert sample_count >= 3, f"Split has {sample_count} samples, less than min_samples=3"

    # Verify we have fewer splits with min_samples=3
    assert len(splits2) < len(splits1)


def test_min_samples_with_max_splits_per_filter():
    """Test that min_samples works with max_splits_per_filter."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 20, "profit": [50.0] * 20},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 21)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": list(range(1, 21))},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 21)], name="date"),
    )

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_splits_per_filter=3,
        min_samples=5,
    )

    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )
    filter_a_right = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "right"
    )

    # Should have at most 3 splits per direction
    assert filter_a_left <= 3
    assert filter_a_right <= 3

    # All splits should have at least 5 samples
    for split in splits:
        if any(f.filter_name == "filter_a" for f in split.filters):
            sample_count = split.mask.sum().item()
            assert sample_count >= 5


def test_min_samples_with_child_splits():
    """Test that min_samples works with child splits (max_depth > 1)."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": list(range(1, 11)), "filter_b": list(range(10, 20))},
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
        max_depth=2,
        min_samples=3,
    )

    # Check that all splits (including child splits) have at least 3 samples
    for split in splits:
        sample_count = split.mask.sum().item()
        assert sample_count >= 3, f"Split has {sample_count} samples, less than min_samples=3"


def test_min_samples_zero():
    """Test that min_samples=0 behaves like min_samples=1."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 5, "profit": [50.0] * 5},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1, 2, 3, 4, 5]},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    # min_samples=0 should still create all splits (like min_samples=1)
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        min_samples=0,
    )

    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )

    # Even with min_samples=0, we can't split with 0 samples
    # But we should still get splits with 1+ sample
    assert filter_a_left == 4


def test_min_samples_larger_than_half():
    """Test min_samples larger than half the dataset."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 10, "profit": [50.0] * 10},
        index=pd.DatetimeIndex([f"2022-01-{i:02d}" for i in range(1, 11)], name="date"),
    )

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
        min_samples=6,
    )

    filter_a_left = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "left"
    )
    filter_a_right = sum(
        1
        for split in splits
        for _, fname, _, direction in split.filters
        if fname == "filter_a" and direction == "right"
    )

    # With min_samples=6 and 10 samples total:
    # Left splits: need 6+ samples on left
    # - Valid: splits at 6.5 (6 left), 7.5 (7 left), 8.5 (8 left), 9.5 (9 left)
    assert filter_a_left == 4

    # Right splits: need 6+ samples on right
    # - Valid: splits at 1.5 (9 right), 2.5 (8 right), 3.5 (7 right), 4.5 (6 right)
    assert filter_a_right == 4


def test_min_samples_all_splits_excluded():
    """Test behavior when min_samples excludes all splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0] * 5, "profit": [50.0] * 5},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1, 2, 3, 4, 5]},
        index=pd.DatetimeIndex([f"2022-01-0{i}" for i in range(1, 6)], name="date"),
    )

    # min_samples=10 is larger than total samples, so no splits should be created
    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        min_samples=10,
    )

    filter_a_splits = sum(
        1 for split in splits for _, fname, _, _ in split.filters if fname == "filter_a"
    )

    # No splits should be created for filter_a
    assert filter_a_splits == 0
