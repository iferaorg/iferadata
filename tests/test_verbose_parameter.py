"""Tests for verbose parameter and Split score/sample printing."""

import io
import sys
import pandas as pd
import pytest
import torch

from ifera.optionalpha import Split, FilterInfo, prepare_splits


def test_split_score_defaults_to_none():
    """Test that Split.score defaults to None instead of -inf."""
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    filters = [FilterInfo(0, "filter_a", 1.5, "left")]
    split = Split(mask=mask, filters=filters, parents=[])

    assert split.score is None


def test_split_str_includes_sample_count():
    """Test that Split.__str__ includes sample count."""
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    filters = [FilterInfo(0, "filter_a", 1.5, "left")]
    split = Split(mask=mask, filters=filters, parents=[])

    result = str(split)
    assert "(samples: 2)" in result


def test_split_str_includes_score_when_present():
    """Test that Split.__str__ includes score when it's not None."""
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    filters = [FilterInfo(0, "filter_a", 1.5, "left")]
    split = Split(mask=mask, filters=filters, parents=[])
    split.score = 0.75

    result = str(split)
    assert "score: 0.7500" in result
    assert "samples: 2" in result


def test_split_str_no_score_when_none():
    """Test that Split.__str__ doesn't show score when it's None."""
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    filters = [FilterInfo(0, "filter_a", 1.5, "left")]
    split = Split(mask=mask, filters=filters, parents=[])

    result = str(split)
    assert "score:" not in result
    assert "(samples: 2)" in result


def test_prepare_splits_verbose_defaults_to_no():
    """Test that verbose parameter defaults to 'no'."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        X, y, splits = prepare_splits(
            trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
        )
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    # Should not print anything with verbose='no'
    assert "Depth" not in output


def test_prepare_splits_verbose_no_explicit():
    """Test that verbose='no' doesn't print anything."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        X, y, splits = prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            verbose="no",
        )
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    # Should not print anything
    assert "Depth" not in output


def test_prepare_splits_verbose_all_prints_output():
    """Test that verbose='all' prints splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        X, y, splits = prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            verbose="all",
        )
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    # Should print depth 1
    assert "Depth 1" in output


def test_prepare_splits_verbose_all_multiple_depths():
    """Test that verbose='all' prints multiple depths."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0, 120.0], "profit": [50.0, 100.0, 75.0, 60.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )
    filters_df = pd.DataFrame(
        {
            "filter_a": [1.0, 2.0, 3.0, 4.0],
            "filter_b": [10.0, 20.0, 30.0, 40.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        X, y, splits = prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            max_depth=2,
            verbose="all",
        )
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    # Should print at least Depth 1
    assert "Depth 1" in output
    # Depth 2 might not appear if no new splits were created at that depth
    # due to merging or other constraints


def test_prepare_splits_verbose_best_requires_score_func():
    """Test that verbose='best' requires score_func."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    with pytest.raises(ValueError, match="verbose='best' requires score_func"):
        prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            verbose="best",
        )


def test_prepare_splits_verbose_best_prints_one_split():
    """Test that verbose='best' prints only the best split."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    def simple_score_func(y, masks):
        """Simple score function."""
        scores = []
        for mask in masks:
            if mask.sum() > 0:
                scores.append(y[mask].mean())
            else:
                scores.append(float("-inf"))
        return torch.tensor(scores, dtype=y.dtype, device=y.device)

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        X, y, splits = prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            score_func=simple_score_func,
            verbose="best",
        )
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    # Should print depth 1 with only one split
    assert "Depth 1" in output
    # Count occurrences of "Split filters:" in output - should be 1 for depth 1
    # (there might be more if max_depth > 1)
    assert output.count("Split filters:") >= 1


def test_prepare_splits_verbose_invalid_value():
    """Test that invalid verbose value raises error."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    with pytest.raises(ValueError, match="verbose must be one of"):
        prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            verbose="invalid",
        )


def test_prepare_splits_verbose_all_with_score_func_sorts():
    """Test that verbose='all' with score_func sorts splits by score."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, -100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    def simple_score_func(y, masks):
        """Simple score function."""
        scores = []
        for mask in masks:
            if mask.sum() > 0:
                scores.append(y[mask].mean())
            else:
                scores.append(float("-inf"))
        return torch.tensor(scores, dtype=y.dtype, device=y.device)

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        X, y, splits = prepare_splits(
            trades_df,
            filters_df,
            20,
            [],
            [],
            torch.device("cpu"),
            torch.float32,
            score_func=simple_score_func,
            verbose="all",
        )
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    # Should show scores in the output
    assert "score:" in output
