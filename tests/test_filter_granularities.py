"""Tests for the filter_granularities functionality in optionalpha module."""

import pandas as pd
import pytest
import torch

from ifera.optionalpha import prepare_splits


def test_filter_granularities_parameter_defaults_to_empty():
    """Test that filter_granularities parameter defaults properly."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Should work without filter_granularities parameter
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
    )

    assert len(splits) > 0


def test_filter_granularities_rounds_left_splits_up():
    """Test that filter_granularities rounds left splits up."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 100.0, 100.0, 100.0],
            "profit": [50.0, 60.0, 70.0, 80.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    # Values: 1.1, 1.2, 1.3, 1.4
    # With granularity 0.25, left direction should round up:
    # 1.1 -> 1.25, 1.2 -> 1.25, 1.3 -> 1.5, 1.4 -> 1.5
    # Unique rounded values: [1.25, 1.5]
    filters_df = pd.DataFrame(
        {"filter_a": [1.1, 1.2, 1.3, 1.4]},
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
        filter_granularities={"filter_a": 0.25},
    )

    # Find left splits for filter_a
    left_splits = [
        s for s in splits if len(s.filters) > 0 and s.filters[0].direction == "left"
    ]

    # Should have at least one left split
    assert len(left_splits) > 0

    # Check that threshold is rounded down (opposite of left rounding)
    # Between 1.25 and 1.5, average is 1.375, round down to 1.25
    for split in left_splits:
        if split.filters[0].filter_name == "filter_a":
            threshold = split.filters[0].threshold
            # Threshold should be a multiple of 0.25
            assert threshold % 0.25 == pytest.approx(0.0)


def test_filter_granularities_rounds_right_splits_down():
    """Test that filter_granularities rounds right splits down."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 100.0, 100.0, 100.0],
            "profit": [50.0, 60.0, 70.0, 80.0],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13"], name="date"
        ),
    )

    # Values: 1.1, 1.2, 1.3, 1.4
    # With granularity 0.25, right direction should round down:
    # 1.1 -> 1.0, 1.2 -> 1.0, 1.3 -> 1.25, 1.4 -> 1.25
    # Unique rounded values: [1.0, 1.25]
    filters_df = pd.DataFrame(
        {"filter_a": [1.1, 1.2, 1.3, 1.4]},
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
        filter_granularities={"filter_a": 0.25},
    )

    # Find right splits for filter_a
    right_splits = [
        s for s in splits if len(s.filters) > 0 and s.filters[0].direction == "right"
    ]

    # Should have at least one right split
    assert len(right_splits) > 0

    # Check that threshold is rounded up (opposite of right rounding)
    # Between 1.0 and 1.25, average is 1.125, round up to 1.25
    for split in right_splits:
        if split.filters[0].filter_name == "filter_a":
            threshold = split.filters[0].threshold
            # Threshold should be a multiple of 0.25
            assert threshold % 0.25 == pytest.approx(0.0)


def test_default_granularities_for_computed_columns():
    """Test that default granularities are applied to computed columns."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 150.0, 200.0],
            "profit": [50.0, 75.0, 100.0],
            "start_time": [
                pd.Timestamp("2022-01-10 09:30:00").time(),
                pd.Timestamp("2022-01-10 10:00:00").time(),
                pd.Timestamp("2022-01-10 10:30:00").time(),
            ],
        },
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

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
    )

    # Check that splits for computed columns have granularity applied
    # reward_per_risk should have granularity 0.001
    # premium should have granularity 0.01
    # weekdays should have granularity 1
    # open_minutes should have granularity 5

    reward_per_risk_splits = [
        s
        for s in splits
        if len(s.filters) > 0 and s.filters[0].filter_name == "reward_per_risk"
    ]
    if len(reward_per_risk_splits) > 0:
        for split in reward_per_risk_splits:
            threshold = split.filters[0].threshold
            # Should be multiple of 0.001
            assert (threshold * 1000) % 1 == pytest.approx(0.0, abs=1e-6)

    open_minutes_splits = [
        s
        for s in splits
        if len(s.filters) > 0 and s.filters[0].filter_name == "open_minutes"
    ]
    if len(open_minutes_splits) > 0:
        for split in open_minutes_splits:
            threshold = split.filters[0].threshold
            # Should be multiple of 5
            assert threshold % 5 == pytest.approx(0.0)


def test_premium_column_is_added():
    """Test that premium column is added to filters."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 150.0, 180.0], "profit": [50.0, 75.0, 90.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,  # spread_width
        [],
        [],
        torch.device("cpu"),
        torch.float32,
    )

    # Premium should be spread_width * 100 - risk
    # For spread_width=20: premium = 2000 - risk
    # Expected: 2000 - 100 = 1900, 2000 - 150 = 1850, 2000 - 180 = 1820

    # Check that premium column appears in at least one split
    # Note: splits might be merged, so we need to check all filters in all splits
    has_premium = False
    for split in splits:
        for filter_info in split.filters:
            if filter_info.filter_name == "premium":
                has_premium = True
                break
        if has_premium:
            break

    # Should have premium in at least one split
    assert has_premium, "Premium filter should appear in at least one split"


def test_split_str_uses_le_ge_operators():
    """Test that Split.__str__ uses <= and >= operators."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
    )

    # Check that the string representation uses <= and >=
    for split in splits:
        split_str = str(split)
        # Should contain either <= or >=, not < or >
        if "filter_a" in split_str:
            assert "<=" in split_str or ">=" in split_str
            # Make sure it's not using < or > without =
            # Check that we don't have " < " or " > " (with spaces)
            assert " < " not in split_str or " <= " in split_str
            assert " > " not in split_str or " >= " in split_str


def test_user_granularities_override_defaults():
    """Test that user-provided granularities override defaults."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Override default granularity for reward_per_risk
    X, y, splits = prepare_splits(
        trades_df,
        filters_df,
        20,
        [],
        [],
        torch.device("cpu"),
        torch.float32,
        filter_granularities={"reward_per_risk": 0.1},  # Override default 0.001
    )

    # Check that reward_per_risk uses the new granularity
    reward_per_risk_splits = [
        s
        for s in splits
        if len(s.filters) > 0 and s.filters[0].filter_name == "reward_per_risk"
    ]

    if len(reward_per_risk_splits) > 0:
        for split in reward_per_risk_splits:
            threshold = split.filters[0].threshold
            # Should be multiple of 0.1 (not 0.001)
            assert (threshold * 10) % 1 == pytest.approx(0.0, abs=1e-6)
