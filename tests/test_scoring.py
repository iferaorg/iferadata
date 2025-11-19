"""Tests for the scoring functionality in optionalpha module."""

import pandas as pd
import pytest
import torch

from ifera.optionalpha import Split, prepare_splits


def test_split_score_initialization():
    """Test that Split objects are initialized with score = None."""
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    split = Split(mask=mask, filters=[], parents=[])

    assert split.score is None


def test_prepare_splits_with_score_func():
    """Test that prepare_splits scores splits when score_func is provided."""
    # Create simple test data
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, -100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # Define a simple score function that returns the sum of y values for masked samples
    def simple_score_func(y, masks):
        # y: (n_samples,)
        # masks: (batch_size, n_samples)
        # Return sum of y for each mask
        scores = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        return scores

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=simple_score_func,
    )

    # Verify that all splits have been scored (not None)
    for split in splits:
        assert split.score is not None, "Split should have been scored"
        assert isinstance(split.score, float), "Score should be a float"


def test_prepare_splits_keep_best_n_requires_score_func():
    """Test that keep_best_n requires score_func to be provided."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    with pytest.raises(
        ValueError, match="score_func must be provided when keep_best_n is not None"
    ):
        prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            keep_best_n=5,
        )


def test_prepare_splits_keep_best_n_filters_splits():
    """Test that keep_best_n keeps only top n scoring splits."""
    # Create test data that will generate multiple splits
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0],
            "profit": [50.0, -100.0, 75.0, 60.0, 90.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
        ),
    )

    # Define a score function that scores based on mean y value
    def mean_score_func(y, masks):
        # Return mean of y for each mask
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        # Avoid division by zero
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    # First get all splits without keep_best_n
    _, _, all_splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
    )

    total_splits = len(all_splits)

    # Now get splits with keep_best_n=3
    keep_n = 3
    _, _, top_splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=mean_score_func,
        keep_best_n=keep_n,
    )

    # Should have at most keep_n splits
    assert len(top_splits) <= keep_n, f"Should have at most {keep_n} splits"

    # If we had more splits initially, verify we kept the best ones
    if total_splits > keep_n:
        assert len(top_splits) == keep_n, f"Should have exactly {keep_n} splits"

        # Verify that top_splits are sorted by score (descending)
        scores = [split.score for split in top_splits]
        assert all(s is not None for s in scores), "All splits should have scores"
        scores_typed = [s for s in scores if s is not None]  # Type guard
        assert scores_typed == sorted(
            scores_typed, reverse=True
        ), "Splits should be sorted by score"

        # Verify that all top_splits have higher scores than any non-selected split
        top_min_score = min(scores_typed)
        all_scores = [split.score for split in all_splits]
        assert all(s is not None for s in all_scores), "All splits should have scores"
        all_scores_typed = [s for s in all_scores if s is not None]  # Type guard
        all_scores_sorted = sorted(all_scores_typed, reverse=True)
        # The top_n scores from all_splits should match top_splits scores
        expected_top_scores = all_scores_sorted[:keep_n]
        assert set(scores_typed) == set(expected_top_scores), "Should keep the top scoring splits"


def test_prepare_splits_keep_best_n_with_depth_2():
    """Test that keep_best_n works correctly with max_depth > 1."""
    # Create test data
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0],
            "profit": [50.0, -100.0, 75.0, 60.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0, 4.0], "filter_b": [10.0, 20.0, 30.0, 40.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"],
            name="date",
        ),
    )

    # Define a simple score function
    def simple_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        return scores

    # Test with depth 2 and keep_best_n
    _, _, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=2,
        score_func=simple_score_func,
        keep_best_n=5,
    )

    # Should have at most 5 splits
    assert len(splits) <= 5, "Should have at most 5 splits"

    # All splits should have scores
    for split in splits:
        assert split.score is not None, "Split should have been scored"


def test_prepare_splits_scoring_with_empty_masks():
    """Test that scoring handles edge cases with masks."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Define a score function that handles edge cases
    def safe_score_func(y, masks):
        sums = torch.sum(y.unsqueeze(0) * masks.float(), dim=1)
        counts = torch.sum(masks.float(), dim=1)
        # Return -inf for empty masks
        scores = torch.where(counts > 0, sums / counts, torch.full_like(sums, float("-inf")))
        return scores

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=safe_score_func,
    )

    # All splits should be scored
    for split in splits:
        assert isinstance(split.score, float), "Score should be a float"


def test_prepare_splits_keep_best_n_less_than_splits():
    """Test keep_best_n when there are fewer splits than n."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Single value filter will generate limited splits
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 1.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    def simple_score_func(y, masks):
        return torch.sum(y.unsqueeze(0) * masks.float(), dim=1)

    _, _, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=simple_score_func,
        keep_best_n=100,  # Request more than available
    )

    # Should return all available splits
    assert len(splits) > 0, "Should have some splits"


def test_prepare_splits_score_ordering():
    """Test that splits are correctly ordered by score when keep_best_n is used."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0],
            "profit": [100.0, 50.0, 75.0, 60.0, 90.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
        ),
    )

    # Score function that returns count of True values in mask
    def count_score_func(y, masks):
        return torch.sum(masks.float(), dim=1)

    _, _, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=count_score_func,
        keep_best_n=3,
    )

    # Verify splits are ordered by score descending
    scores = [split.score for split in splits]
    assert all(s is not None for s in scores), "All splits should have scores"
    scores_typed = [s for s in scores if s is not None]  # Type guard
    assert scores_typed == sorted(
        scores_typed, reverse=True
    ), "Splits should be ordered by score descending"


def test_prepare_splits_early_exit_with_keep_best_n():
    """Test that loop exits early when keep_best_n results in empty previous_depth_splits."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0],
            "profit": [50.0, -100.0, 75.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12"],
            name="date",
        ),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0], "filter_b": [10.0, 20.0, 30.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12"],
            name="date",
        ),
    )

    # Score function that heavily favors depth 1 splits
    def depth_1_favoring_score_func(y, masks):
        # Return high scores for masks with 1 or 2 True values
        counts = torch.sum(masks.float(), dim=1)
        # Give higher scores to depth 1 splits (typically have more samples)
        return counts * 10.0

    # Use keep_best_n that's smaller than depth 1 splits count
    # This should cause early exit in depth 2 iteration
    _, _, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=3,
        score_func=depth_1_favoring_score_func,
        keep_best_n=4,
    )

    # Should have at most 4 splits
    assert len(splits) <= 4, "Should have at most 4 splits"

    # Check that no errors occurred
    for split in splits:
        assert split.score is not None, "Split should have been scored"


def test_score_func_receives_correct_parameters():
    """Test that score_func receives y and masks with correct shapes."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, -100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    n_samples = len(trades_df)
    received_shapes = []

    def shape_checking_score_func(y, masks):
        # Record the shapes
        received_shapes.append((y.shape, masks.shape))
        # Return dummy scores
        return torch.zeros(masks.shape[0])

    _, _, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        score_func=shape_checking_score_func,
    )

    # Verify that score_func was called
    assert len(received_shapes) > 0, "score_func should have been called"

    # Verify shapes
    # Note: With filter evaluation, score_func is called with different y sizes
    # (train/validation splits during CV), so we check that:
    # 1. y is always 1D
    # 2. masks is always 2D
    # 3. masks second dimension matches y size
    for y_shape, masks_shape in received_shapes:
        assert len(y_shape) == 1, f"y should be 1D, got {y_shape}"
        assert len(masks_shape) == 2, f"masks should be 2D, got {masks_shape}"
        assert (
            masks_shape[1] == y_shape[0]
        ), f"masks second dimension should match y size, got {masks_shape[1]} vs {y_shape[0]}"
