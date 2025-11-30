"""Tests for the filter evaluation functionality in optionalpha module."""

import pandas as pd
import pytest
import torch

from ifera.optionalpha import prepare_splits


def test_evaluate_filters_basic():
    """Test basic filter evaluation with cross-validation."""
    # Create test data with multiple filters
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0, 110.0, 190.0, 130.0],
            "profit": [50.0, -100.0, 75.0, 60.0, 90.0, 40.0, -80.0, 65.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
                "2022-01-16",
                "2022-01-17",
            ],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "filter_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
                "2022-01-16",
                "2022-01-17",
            ],
            name="date",
        ),
    )

    # Define a simple score function
    def mean_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    # Run prepare_splits with filter evaluation
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        filter_eval_folds=3,
        filter_eval_repeats=2,
    )

    # Verify that splits were created
    assert len(splits) > 0, "Should have created some splits"


def test_evaluate_filters_with_min_score_improvement():
    """Test filter evaluation with min_score_improvement filtering."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0, 110.0],
            "profit": [50.0, -100.0, 75.0, 60.0, 90.0, 40.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
            ],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "filter_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
            ],
            name="date",
        ),
    )

    def mean_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    # Get splits without filtering
    _, _, splits_unfiltered = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        filter_eval_folds=2,
        filter_eval_repeats=1,
    )

    # Get splits with aggressive filtering (should remove some)
    _, _, splits_filtered = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        filter_eval_folds=2,
        filter_eval_repeats=1,
        min_score_improvement=1.0,  # Very high threshold to filter some out
    )

    # Should have fewer splits after filtering
    # (This may not always be true depending on data, but likely with high threshold)
    assert len(splits_filtered) <= len(
        splits_unfiltered
    ), "Filtered splits should be <= unfiltered"


def test_evaluate_filters_without_score_func():
    """Test that filter evaluation is skipped when score_func is None."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Run without score_func - should not crash
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=None,
        filter_eval_folds=3,
        filter_eval_repeats=2,
    )

    assert len(splits) > 0, "Should have created splits"


def test_evaluate_filters_with_custom_folds_and_repeats():
    """Test filter evaluation with custom fold and repeat settings."""
    trades_df = pd.DataFrame(
        {
            "risk": [
                100.0,
                200.0,
                150.0,
                120.0,
                180.0,
                110.0,
                190.0,
                130.0,
                140.0,
                160.0,
            ],
            "profit": [50.0, -100.0, 75.0, 60.0, 90.0, 40.0, -80.0, 65.0, 55.0, 70.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
                "2022-01-16",
                "2022-01-17",
                "2022-01-18",
                "2022-01-19",
            ],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
                "2022-01-16",
                "2022-01-17",
                "2022-01-18",
                "2022-01-19",
            ],
            name="date",
        ),
    )

    def simple_score_func(y, masks):
        return torch.sum(y.unsqueeze(0) * masks.float(), dim=1)

    # Test with 4 folds and 3 repeats
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=simple_score_func,
        filter_eval_folds=4,
        filter_eval_repeats=3,
    )

    assert len(splits) > 0, "Should have created splits"


def test_evaluate_filters_integration_with_depth_2():
    """Test that filter evaluation works correctly with max_depth > 1."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0, 110.0],
            "profit": [50.0, -100.0, 75.0, 60.0, 90.0, 40.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
            ],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "filter_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
            ],
            name="date",
        ),
    )

    def mean_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    # Test with depth 2 and filter evaluation
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=2,
        score_func=mean_score_func,
        filter_eval_folds=2,
        filter_eval_repeats=1,
    )

    # Should have created splits
    assert len(splits) > 0, "Should have created splits"


def test_evaluate_filters_with_small_dataset():
    """Test filter evaluation with a small dataset (edge case)."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, -100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    def simple_score_func(y, masks):
        return torch.sum(y.unsqueeze(0) * masks.float(), dim=1)

    # Test with small dataset - should handle padding correctly
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=simple_score_func,
        filter_eval_folds=2,
        filter_eval_repeats=1,
    )

    assert len(splits) > 0, "Should have created splits even with small dataset"


def test_cv_indices_padding():
    """Test that cross-validation indices are properly padded."""
    from ifera.optionalpha import _create_cv_indices

    device = torch.device("cpu")

    # Test with n_samples not divisible by n_folds
    n_samples = 7
    n_folds = 3
    n_repeats = 2

    randperm, n_samples_padded, n_samples_train, n_samples_valid = _create_cv_indices(
        n_samples, n_folds, n_repeats, device
    )

    # Check that n_samples_padded is divisible by n_folds
    assert (
        n_samples_padded % n_folds == 0
    ), "Padded samples should be divisible by folds"

    # Check that n_samples_padded is the smallest such value >= n_samples
    assert n_samples_padded >= n_samples, "Padded samples should be >= original samples"
    assert n_samples_padded - n_folds < n_samples, "Padded samples should be minimal"

    # Check train/validation sizes
    assert (
        n_samples_valid == n_samples_padded // n_folds
    ), "Validation size should be 1/n_folds of padded"
    assert (
        n_samples_train == n_samples_padded - n_samples_valid
    ), "Training size should be complementary"

    # Check randperm shape
    assert randperm.shape == (
        n_repeats,
        n_samples_padded,
    ), "Randperm should have correct shape"


def test_cv_splits_creation():
    """Test that train/validation splits are correctly created."""
    from ifera.optionalpha import _create_cv_indices, _create_cv_splits

    device = torch.device("cpu")
    n_samples = 8
    n_folds = 4
    n_repeats = 2

    # Create test y tensor
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=device)

    # Create CV indices
    randperm, n_samples_padded, n_samples_train, n_samples_valid = _create_cv_indices(
        n_samples, n_folds, n_repeats, device
    )

    # Create CV splits
    y_train, y_valid = _create_cv_splits(
        y,
        randperm,
        n_folds,
        n_samples_padded,
        n_samples_train,
        n_samples_valid,
        n_samples,
    )

    # Check shapes
    assert y_train.shape == (
        n_repeats,
        n_folds,
        n_samples_train,
    ), "y_train should have correct shape"
    assert y_valid.shape == (
        n_repeats,
        n_folds,
        n_samples_valid,
    ), "y_valid should have correct shape"

    # Check that each fold uses different validation data
    for k in range(n_repeats):
        for fold in range(n_folds):
            # Validation set should have n_samples_valid elements
            assert (
                len(y_valid[k, fold]) == n_samples_valid
            ), "Validation set should have correct size"

            # Training set should have n_samples_train elements
            assert (
                len(y_train[k, fold]) == n_samples_train
            ), "Training set should have correct size"


def test_evaluate_filters_no_filters():
    """Test that evaluation handles case with no valid filters gracefully."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Create filters with constant values (no splits will be generated)
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 1.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    def simple_score_func(y, masks):
        return torch.sum(y.unsqueeze(0) * masks.float(), dim=1)

    # Should handle gracefully even with no valid splits
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=simple_score_func,
        filter_eval_folds=2,
        filter_eval_repeats=1,
    )

    # May have computed columns that create splits
    assert isinstance(splits, list), "Should return a list of splits"


def test_evaluate_filters_vectorized_multiple_filter_groups():
    """Test vectorized evaluation with multiple filter groups.

    This test validates that the vectorized implementation correctly processes
    multiple filter+direction groups simultaneously, ensuring each group gets
    its own best split and score improvement.
    """
    # Create test data with multiple distinct filters
    trades_df = pd.DataFrame(
        {
            "risk": [
                100.0,
                150.0,
                120.0,
                180.0,
                90.0,
                200.0,
                130.0,
                170.0,
                110.0,
                160.0,
            ],
            "profit": [50.0, -30.0, 40.0, -50.0, 70.0, -80.0, 35.0, -20.0, 60.0, -10.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-17",
                "2022-01-18",
                "2022-01-19",
                "2022-01-20",
                "2022-01-21",
            ],
            name="date",
        ),
    )

    # Create multiple filters with different value distributions
    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "filter_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "filter_c": [5.0, 5.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-17",
                "2022-01-18",
                "2022-01-19",
                "2022-01-20",
                "2022-01-21",
            ],
            name="date",
        ),
    )

    def mean_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    # Run with multiple filters and folds
    _, _, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        filter_eval_folds=3,
        filter_eval_repeats=2,
        min_samples=2,
    )

    # Should have created splits from multiple filter groups
    assert len(splits) > 0, "Should have created splits"

    # Verify splits have scores assigned
    scored_splits = [s for s in splits if s.score is not None]
    assert len(scored_splits) > 0, "Should have scored splits"


def test_evaluate_filters_consistency():
    """Test that vectorized evaluation produces consistent results.

    This test verifies that the evaluation results are deterministic
    when run with the same random seed.
    """
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0, 110.0],
            "profit": [50.0, -100.0, 75.0, 60.0, 90.0, 40.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
            ],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=pd.DatetimeIndex(
            [
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
            ],
            name="date",
        ),
    )

    def mean_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    # Set seed for reproducibility
    torch.manual_seed(42)
    _, _, splits1 = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        filter_eval_folds=2,
        filter_eval_repeats=2,
        min_samples=2,
    )

    # Reset seed and run again
    torch.manual_seed(42)
    _, _, splits2 = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        filter_eval_folds=2,
        filter_eval_repeats=2,
        min_samples=2,
    )

    # Results should be identical
    assert len(splits1) == len(splits2), "Should have same number of splits"

    # Compare scores (they should be identical with same seed)
    scores1 = sorted([s.score for s in splits1 if s.score is not None])
    scores2 = sorted([s.score for s in splits2 if s.score is not None])
    assert len(scores1) == len(scores2), "Should have same number of scored splits"
    for s1, s2 in zip(scores1, scores2):
        assert abs(s1 - s2) < 1e-6, f"Scores should match: {s1} vs {s2}"
