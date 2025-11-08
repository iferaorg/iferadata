"""Tests for the optionalpha module."""

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from ifera.optionalpha import (
    FilterInfo,
    Split,
    _extract_dollar_amount,
    _parse_time,
    parse_filter_log,
    parse_trade_log,
    prepare_splits,
)
from bs4 import BeautifulSoup


@pytest.fixture
def grid_example_html():
    """Load the grid_example.txt file."""
    test_dir = Path(__file__).parent
    grid_file = test_dir / "grid_example.txt"
    with open(grid_file, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def simple_html():
    """Create a simple HTML grid for basic testing."""
    return """
    <grid>
        <row>
            <bd>
                <div class="cell symbol">
                    <div class="clip">
                        <span class="sym">SPX</span>
                        <span>Long Call</span>
                    </div>
                </div>
                <div class="cell closeTime">
                    <div class="clip">Jan 10, 2022</div>
                    <div class="clip">3:47pm → 4:00pm</div>
                </div>
                <div class="cell status">
                    <span class="lbl">Expired</span>
                </div>
                <div class="cell risk">
                    <span class="val pos">$380</span>
                </div>
                <div class="cell pnl">
                    <span class="val pos pnl">$1,149</span>
                </div>
            </bd>
        </row>
        <row>
            <bd>
                <div class="cell symbol">
                    <div class="clip">
                        <span class="sym">SPX</span>
                        <span>Long Call</span>
                    </div>
                </div>
                <div class="cell closeTime">
                    <div class="clip">Jan 14, 2022</div>
                    <div class="clip">3:46pm → 4:00pm</div>
                </div>
                <div class="cell status">
                    <span class="lbl">Expired</span>
                </div>
                <div class="cell risk">
                    <span class="val pos">$350</span>
                </div>
                <div class="cell pnl">
                    <span class="val neg pnl">-$350</span>
                </div>
            </bd>
        </row>
    </grid>
    """


def test_parse_trade_log_simple(simple_html):
    """Test parsing a simple HTML grid."""
    df = parse_trade_log(simple_html)

    # Check DataFrame shape
    assert len(df) == 2
    assert list(df.columns) == [
        "symbol",
        "trade_type",
        "start_time",
        "end_time",
        "status",
        "risk",
        "profit",
    ]

    # Check that index is DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"

    # Check first row
    from datetime import time as dt_time

    assert df.iloc[0]["symbol"] == "SPX"
    assert df.iloc[0]["trade_type"] == "Long Call"
    assert df.index[0] == pd.Timestamp("2022-01-10")
    assert df.iloc[0]["start_time"] == dt_time(15, 47)
    assert df.iloc[0]["end_time"] == dt_time(16, 0)
    assert df.iloc[0]["status"] == "Expired"
    assert df.iloc[0]["risk"] == 380.0
    assert df.iloc[0]["profit"] == 1149.0

    # Check second row (negative P/L)
    assert df.iloc[1]["symbol"] == "SPX"
    assert df.iloc[1]["trade_type"] == "Long Call"
    assert df.index[1] == pd.Timestamp("2022-01-14")
    assert df.iloc[1]["profit"] == -350.0


def test_parse_trade_log_grid_example(grid_example_html):
    """Test parsing the full grid_example.txt file."""
    df = parse_trade_log(grid_example_html)

    # Check that we parsed a substantial number of rows
    assert len(df) > 100

    # Check column names
    assert list(df.columns) == [
        "symbol",
        "trade_type",
        "start_time",
        "end_time",
        "status",
        "risk",
        "profit",
    ]

    # Check that index is DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"

    # Check data types
    assert df["symbol"].dtype == object
    assert df["trade_type"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df["start_time"].dtype == object  # datetime.time objects
    assert df["end_time"].dtype == object  # datetime.time objects
    assert df["status"].dtype == object
    assert pd.api.types.is_numeric_dtype(df["risk"])
    assert pd.api.types.is_numeric_dtype(df["profit"])

    # Check for NaN values (should not exist in numeric columns)
    assert df["risk"].isna().sum() == 0
    assert df["profit"].isna().sum() == 0

    # Check some sample values
    first_row = df.iloc[0]
    assert first_row["symbol"] == "SPX"
    assert first_row["trade_type"] == "Long Call"
    assert first_row["status"] == "Expired"

    # Check that both positive and negative P/L values exist
    has_positive_pl: bool = bool((df["profit"] > 0).any())
    has_negative_pl: bool = bool((df["profit"] < 0).any())
    all_risk_nonnegative: bool = bool((df["risk"] >= 0).all())

    assert has_positive_pl
    assert has_negative_pl
    assert all_risk_nonnegative


def test_parse_trade_log_missing_dollar_values():
    """Test parsing when dollar values are missing."""
    html = """
    <grid>
        <row>
            <bd>
                <div class="cell symbol">
                    <div class="clip">
                        <span class="sym">SPX</span>
                        <span>Long Call</span>
                    </div>
                </div>
                <div class="cell closeTime">
                    <div class="clip">Jan 10, 2022</div>
                    <div class="clip">3:47pm → 4:00pm</div>
                </div>
                <div class="cell status">
                    <span class="lbl">Expired</span>
                </div>
                <div class="cell risk">
                </div>
                <div class="cell pnl">
                </div>
            </bd>
        </row>
    </grid>
    """
    df = parse_trade_log(html)

    # Check that missing values are replaced with 0
    assert df.iloc[0]["risk"] == 0.0
    assert df.iloc[0]["profit"] == 0.0
    assert df["risk"].isna().sum() == 0
    assert df["profit"].isna().sum() == 0


def test_parse_trade_log_empty_string():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="HTML string cannot be empty"):
        parse_trade_log("")


def test_parse_trade_log_whitespace_only():
    """Test that whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="HTML string cannot be empty"):
        parse_trade_log("   \n\t  ")


def test_parse_trade_log_no_rows():
    """Test that HTML with no rows raises ValueError."""
    html = "<grid><hd></hd></grid>"
    with pytest.raises(ValueError, match="No trade rows found"):
        parse_trade_log(html)


def test_parse_trade_log_invalid_html():
    """Test that invalid HTML still raises appropriate error."""
    html = "<invalid>not a grid</invalid>"
    with pytest.raises(ValueError, match="No trade rows found"):
        parse_trade_log(html)


def test_extract_dollar_amount_positive():
    """Test extracting positive dollar amounts."""
    html = '<div><span class="val">$1,234</span></div>'
    cell = BeautifulSoup(html, "html.parser").find("div")
    assert _extract_dollar_amount(cell) == 1234.0


def test_extract_dollar_amount_negative():
    """Test extracting negative dollar amounts."""
    html = '<div><span class="val">-$1,234</span></div>'
    cell = BeautifulSoup(html, "html.parser").find("div")
    assert _extract_dollar_amount(cell) == -1234.0


def test_extract_dollar_amount_with_decimals():
    """Test extracting dollar amounts with decimals."""
    html = '<div><span class="val">$1,234.56</span></div>'
    cell = BeautifulSoup(html, "html.parser").find("div")
    assert _extract_dollar_amount(cell) == 1234.56


def test_extract_dollar_amount_no_comma():
    """Test extracting dollar amounts without commas."""
    html = '<div><span class="val">$123</span></div>'
    cell = BeautifulSoup(html, "html.parser").find("div")
    assert _extract_dollar_amount(cell) == 123.0


def test_extract_dollar_amount_none_cell():
    """Test extracting from None cell."""
    assert _extract_dollar_amount(None) == 0.0


def test_extract_dollar_amount_no_val_span():
    """Test extracting when no val span exists."""
    html = "<div>$1,234</div>"
    cell = BeautifulSoup(html, "html.parser").find("div")
    assert _extract_dollar_amount(cell) == 0.0


def test_extract_dollar_amount_empty_span():
    """Test extracting from empty span."""
    html = '<div><span class="val"></span></div>'
    cell = BeautifulSoup(html, "html.parser").find("div")
    assert _extract_dollar_amount(cell) == 0.0


def test_parse_trade_log_result_no_nan_values(grid_example_html):
    """Explicitly verify no NaN values in output."""
    df = parse_trade_log(grid_example_html)

    # Check numeric columns for NaN values
    has_nan_risk: bool = bool(df["risk"].isna().any())
    has_nan_profit: bool = bool(df["profit"].isna().any())

    assert not has_nan_risk
    assert not has_nan_profit

    # Check that numeric columns don't have NaN
    assert df["risk"].isna().sum() == 0
    assert df["profit"].isna().sum() == 0


def test_parse_time_valid():
    """Test parsing valid time strings."""
    from datetime import time as dt_time

    assert _parse_time("3:47pm") == dt_time(15, 47)
    assert _parse_time("4:00pm") == dt_time(16, 0)
    assert _parse_time("10:30am") == dt_time(10, 30)
    assert _parse_time("12:00pm") == dt_time(12, 0)


def test_parse_time_invalid():
    """Test parsing invalid time strings."""
    assert _parse_time("") is None
    assert _parse_time("   ") is None
    assert _parse_time("invalid") is None


@pytest.fixture
def filter_log_range_width():
    """Load the SPX-ORB-L-RANGE_WIDTH.txt filter log file."""
    test_dir = Path(__file__).parent
    filter_file = test_dir / "filter_examples" / "SPX-ORB-L-RANGE_WIDTH.txt"
    with open(filter_file, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def filter_log_skip_fomc():
    """Load the SPX-ORB-L-SKIP_FOMC.txt filter log file."""
    test_dir = Path(__file__).parent
    filter_file = test_dir / "filter_examples" / "SPX-ORB-L-SKIP_FOMC.txt"
    with open(filter_file, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def simple_filter_html():
    """Create a simple filter log HTML for basic testing."""
    return """
    <div class="rows">
        <div class="flex" style="padding:1.2rem 2rem;font-size:1.5rem;border-top:var(--border);align-items:flex-start;">
            <div style="width:12rem;">Jan 3, 2022</div>
            <div style="flex:1;">Min opening range<desc style="display:block;font-size:1.4rem;margin-top:.3rem;">Opening Range: 4,758.17 - 4,795.86, Width: 0.79%</desc></div>
        </div>
        <div class="flex" style="padding:1.2rem 2rem;font-size:1.5rem;border-top:var(--border);align-items:flex-start;">
            <div style="width:12rem;">Jan 26, 2022</div>
            <div style="flex:1;">FOMC Meeting</div>
        </div>
    </div>
    """


def test_parse_filter_log_simple(simple_filter_html):
    """Test parsing a simple filter log HTML."""
    df = parse_filter_log(simple_filter_html)

    # Check DataFrame shape
    assert len(df) == 2
    assert list(df.columns) == ["date", "filter_type", "description"]

    # Check first row (with description)
    assert df.iloc[0]["date"] == pd.Timestamp("2022-01-03")
    assert df.iloc[0]["filter_type"] == "Min opening range"
    assert (
        df.iloc[0]["description"] == "Opening Range: 4,758.17 - 4,795.86, Width: 0.79%"
    )

    # Check second row (without description)
    assert df.iloc[1]["date"] == pd.Timestamp("2022-01-26")
    assert df.iloc[1]["filter_type"] == "FOMC Meeting"
    assert df.iloc[1]["description"] == ""


def test_parse_filter_log_range_width(filter_log_range_width):
    """Test parsing the SPX-ORB-L-RANGE_WIDTH.txt filter log file."""
    df = parse_filter_log(filter_log_range_width)

    # Check that we parsed a substantial number of rows
    assert len(df) > 900

    # Check column names
    assert list(df.columns) == ["date", "filter_type", "description"]

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["filter_type"].dtype == object
    assert df["description"].dtype == object

    # Check for NaN values (should not exist)
    assert df["filter_type"].isna().sum() == 0
    assert df["description"].isna().sum() == 0

    # Check some sample values
    first_row = df.iloc[0]
    assert first_row["date"] == pd.Timestamp("2022-01-03")
    assert first_row["filter_type"] == "Min opening range"
    assert "Opening Range:" in first_row["description"]
    assert "Width:" in first_row["description"]

    # All rows should have descriptions in this file
    assert (df["description"] != "").all()


def test_parse_filter_log_skip_fomc(filter_log_skip_fomc):
    """Test parsing the SPX-ORB-L-SKIP_FOMC.txt filter log file."""
    df = parse_filter_log(filter_log_skip_fomc)

    # Check that we parsed the expected number of rows
    assert len(df) == 30

    # Check column names
    assert list(df.columns) == ["date", "filter_type", "description"]

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["filter_type"].dtype == object
    assert df["description"].dtype == object

    # Check for NaN values (should not exist)
    assert df["filter_type"].isna().sum() == 0
    assert df["description"].isna().sum() == 0

    # Check some sample values
    first_row = df.iloc[0]
    assert first_row["date"] == pd.Timestamp("2022-01-26")
    assert first_row["filter_type"] == "FOMC Meeting"
    assert first_row["description"] == ""

    # All rows should be FOMC Meeting
    assert (df["filter_type"] == "FOMC Meeting").all()
    # All rows should have no description in this file
    assert (df["description"] == "").all()


def test_parse_filter_log_empty_string():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="HTML string cannot be empty"):
        parse_filter_log("")


def test_parse_filter_log_whitespace_only():
    """Test that whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="HTML string cannot be empty"):
        parse_filter_log("   \n\t  ")


def test_parse_filter_log_no_entries():
    """Test that HTML with no filter entries raises ValueError."""
    html = "<div class='rows'></div>"
    with pytest.raises(ValueError, match="No filter entries found"):
        parse_filter_log(html)


def test_parse_filter_log_invalid_html():
    """Test that invalid HTML still raises appropriate error."""
    html = "<invalid>not a filter log</invalid>"
    with pytest.raises(ValueError, match="No filter entries found"):
        parse_filter_log(html)


def test_parse_filter_log_result_no_nan_values(filter_log_range_width):
    """Explicitly verify no NaN values in output."""
    df = parse_filter_log(filter_log_range_width)

    # Check all columns for NaN values
    has_nan_date: bool = bool(df["date"].isna().any())
    has_nan_filter_type: bool = bool(df["filter_type"].isna().any())
    has_nan_description: bool = bool(df["description"].isna().any())

    assert not has_nan_date
    assert not has_nan_filter_type
    assert not has_nan_description


def test_parse_trade_log_duplicate_dates_raises_error():
    """Test that duplicate dates in trade log raise an error."""
    html = """
    <grid>
        <row>
            <bd>
                <div class="cell symbol">
                    <div class="clip">
                        <span class="sym">SPX</span>
                        <span>Long Call</span>
                    </div>
                </div>
                <div class="cell closeTime">
                    <div class="clip">Jan 10, 2022</div>
                    <div class="clip">3:47pm → 4:00pm</div>
                </div>
                <div class="cell status">
                    <span class="lbl">Expired</span>
                </div>
                <div class="cell risk">
                    <span class="val pos">$380</span>
                </div>
                <div class="cell pnl">
                    <span class="val pos pnl">$1,149</span>
                </div>
            </bd>
        </row>
        <row>
            <bd>
                <div class="cell symbol">
                    <div class="clip">
                        <span class="sym">SPX</span>
                        <span>Long Put</span>
                    </div>
                </div>
                <div class="cell closeTime">
                    <div class="clip">Jan 10, 2022</div>
                    <div class="clip">3:50pm → 4:00pm</div>
                </div>
                <div class="cell status">
                    <span class="lbl">Closed</span>
                </div>
                <div class="cell risk">
                    <span class="val pos">$400</span>
                </div>
                <div class="cell pnl">
                    <span class="val pos pnl">$200</span>
                </div>
            </bd>
        </row>
    </grid>
    """
    with pytest.raises(ValueError, match="Duplicate dates found in trade log"):
        parse_trade_log(html)


def test_check_and_eliminate_duplicates_same_values():
    """Test that duplicates with same values are eliminated."""
    from ifera.optionalpha import _check_and_eliminate_duplicates

    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2022-01-10"),
                pd.Timestamp("2022-01-10"),
                pd.Timestamp("2022-01-11"),
            ],
            "filter": [1, 1, 1],
        }
    )

    result = _check_and_eliminate_duplicates(df, "filter")

    # Should have only 2 rows now
    assert len(result) == 2
    assert result["date"].tolist() == [
        pd.Timestamp("2022-01-10"),
        pd.Timestamp("2022-01-11"),
    ]


def test_check_and_eliminate_duplicates_different_values():
    """Test that duplicates with different values raise an error."""
    from ifera.optionalpha import _check_and_eliminate_duplicates

    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2022-01-10"),
                pd.Timestamp("2022-01-10"),
                pd.Timestamp("2022-01-11"),
            ],
            "filter": [1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError, match="Duplicate date .* found with different values"
    ):
        _check_and_eliminate_duplicates(df, "filter")


def test_prepare_splits_basic():
    """Test basic functionality of prepare_splits."""
    # Create simple trades dataframe
    trades_df = pd.DataFrame(
        {
            "symbol": ["SPX", "SPX", "SPX"],
            "risk": [100.0, 200.0, 150.0],
            "profit": [50.0, -100.0, 75.0],
        },
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # Create simple filters dataframe
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0], "filter_b": [10.0, 20.0, 30.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    spread_width = 20
    device = torch.device("cpu")
    dtype = torch.float32

    X, y, splits = prepare_splits(
        trades_df, filters_df, spread_width, [], [], device, dtype
    )

    # Check X shape - should have 3 rows and 9 columns
    # (2 filters + reward_per_risk + 5 weekday filters + open_minutes)
    assert X.shape == (3, 9)
    assert X.device == device
    assert X.dtype == dtype

    # Check y shape and values
    assert y.shape == (3,)
    assert y.device == device
    assert y.dtype == dtype
    # RoR should be profit / risk
    expected_y = torch.tensor([0.5, -0.5, 0.5], dtype=dtype, device=device)
    assert torch.allclose(y, expected_y)

    # Check reward_per_risk column (column index 2 in X, after filter_a and filter_b)
    # reward_per_risk = (spread_width * 100 - risk) / risk
    expected_rpr = [
        (20 * 100 - 100) / 100,
        (20 * 100 - 200) / 200,
        (20 * 100 - 150) / 150,
    ]
    assert torch.allclose(
        X[:, 2], torch.tensor(expected_rpr, dtype=dtype, device=device)
    )

    # Check splits - with the new merging logic, splits with identical masks are merged
    # We should have at least a few unique splits
    assert len(splits) > 0

    # Check split properties
    for split in splits:
        assert isinstance(split, Split)
        assert hasattr(split, "mask")
        assert hasattr(split, "filters")
        assert isinstance(split.filters, list)
        assert len(split.filters) > 0
        # Each filter tuple should have 4 elements: (filter_idx, filter_name, threshold, direction)
        for filter_tuple in split.filters:
            assert len(filter_tuple) == 4
            filter_idx, filter_name, threshold, direction = filter_tuple
            assert filter_idx in range(9)  # 9 columns now
            assert isinstance(filter_name, str)
            assert isinstance(threshold, float)
            assert direction in ["left", "right"]
        assert split.mask.dtype == torch.bool
        assert split.mask.shape == (3,)  # 3 samples


def test_prepare_splits_left_only_filters():
    """Test that left_only_filters prevents right splits."""
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
        left_only_filters=["filter_a"],
        right_only_filters=[],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Check that filter_a has only left direction in the filters
    has_filter_a_left = False
    has_filter_a_right = False

    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a":
                if direction == "left":
                    has_filter_a_left = True
                elif direction == "right":
                    has_filter_a_right = True

    assert has_filter_a_left, "Expected at least one left split for filter_a"
    assert not has_filter_a_right, "Expected no right splits for filter_a"


def test_prepare_splits_right_only_filters():
    """Test that right_only_filters prevents left splits."""
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
        left_only_filters=[],
        right_only_filters=["filter_a"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Check that filter_a has only right direction in the filters
    has_filter_a_left = False
    has_filter_a_right = False

    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a":
                if direction == "left":
                    has_filter_a_left = True
                elif direction == "right":
                    has_filter_a_right = True

    assert has_filter_a_right, "Expected at least one right split for filter_a"
    assert not has_filter_a_left, "Expected no left splits for filter_a"


def test_prepare_splits_exclusion_mask_same_filter_direction():
    """Test that splits from same filter and direction have subset relationships."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # With merging, we should have fewer splits than before
    # The exact number depends on mask uniqueness
    assert len(splits) > 0

    # Find splits containing filter_a with left direction
    left_splits = []
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a" and direction == "left":
                left_splits.append(split)
                break

    # If there are multiple left splits, verify they have subset relationships
    # (one includes the other, meaning they would be exclusive)
    if len(left_splits) >= 2:
        for i in range(len(left_splits)):
            for j in range(i + 1, len(left_splits)):
                # Check if one is a subset of the other
                mask_i = left_splits[i].mask
                mask_j = left_splits[j].mask
                intersection = (mask_i & mask_j).sum().item()
                sum_i = mask_i.sum().item()
                sum_j = mask_j.sum().item()
                # At least one should be a subset of the other
                is_subset = (intersection == sum_i) or (intersection == sum_j)
                assert is_subset, "Left splits from same filter should have subset relationship"


def test_prepare_splits_missing_dates_in_filters():
    """Test that missing dates in trades raise an error."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Filters missing the 2022-01-11 date
    filters_df = pd.DataFrame(
        {"filter_a": [1.0]},
        index=pd.DatetimeIndex(["2022-01-10"], name="date"),
    )

    with pytest.raises(ValueError, match="Trades dataframe contains dates not found"):
        prepare_splits(
            trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
        )


def test_prepare_splits_extra_dates_in_filters():
    """Test that extra dates in filters are removed."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    # Filters have an extra date
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # X should only have 2 rows (matching trades_df)
    assert X.shape[0] == 2
    assert y.shape[0] == 2


def test_prepare_splits_single_value_filter():
    """Test that filters with single unique value generate no splits."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # filter_a has only one unique value
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 1.0, 1.0], "filter_b": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Should have splits generated
    assert len(splits) > 0

    # Verify filter_a generates no splits
    has_filter_a = False
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_a":
                has_filter_a = True
                break

    assert not has_filter_a, "filter_a should not generate any splits"

    # Verify filter_b generates splits
    has_filter_b = False
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "filter_b":
                has_filter_b = True
                break

    assert has_filter_b, "filter_b should generate splits"


def test_prepare_splits_mask_values():
    """Test that split masks correctly identify rows."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Find a split containing filter_a left split at threshold 1.5
    found_left_1_5 = None
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if (
                filter_name == "filter_a"
                and direction == "left"
                and abs(threshold - 1.5) < 0.01
            ):
                found_left_1_5 = split
                break
        if found_left_1_5:
            break

    assert found_left_1_5 is not None, "Should find a left split at threshold 1.5"

    # Should include only the first row (value 1.0 <= 1.5)
    expected_mask = torch.tensor([True, False, False], dtype=torch.bool)
    assert torch.equal(found_left_1_5.mask, expected_mask)

    # Find a split containing filter_a right split at threshold 2.5
    found_right_2_5 = None
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if (
                filter_name == "filter_a"
                and direction == "right"
                and abs(threshold - 2.5) < 0.01
            ):
                found_right_2_5 = split
                break
        if found_right_2_5:
            break

    assert found_right_2_5 is not None, "Should find a right split at threshold 2.5"

    # Should include only the last row (value 3.0 >= 2.5)
    expected_mask = torch.tensor([False, False, True], dtype=torch.bool)
    assert torch.equal(found_right_2_5.mask, expected_mask)


def test_prepare_splits_exclusion_empty_intersection():
    """Test that the function works correctly with filters that could create empty intersections."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # Two filters with non-overlapping splits
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 5.0, 10.0], "filter_b": [10.0, 5.0, 1.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # All splits should be valid (have at least 1 sample)
    for split in splits:
        assert split.mask.sum().item() >= 1, "All splits should have at least 1 sample"


def test_prepare_splits_device_dtype():
    """Test that output tensors use the correct device and dtype."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0], "profit": [50.0, 100.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    device = torch.device("cpu")
    dtype = torch.float64

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], device, dtype
    )

    # Check device and dtype
    assert X.device == device
    assert X.dtype == dtype
    assert y.device == device
    assert y.dtype == dtype

    # Masks should be bool
    for split in splits:
        assert split.mask.dtype == torch.bool
        assert split.mask.device == device


def test_prepare_splits_weekday_filters():
    """Test that weekday filters are added and configured as left-only."""
    from datetime import time as dt_time

    # Create trades dataframe with start_time column covering different weekdays
    # 2022-01-10 = Monday, 2022-01-11 = Tuesday, 2022-01-12 = Wednesday
    # 2022-01-13 = Thursday, 2022-01-14 = Friday
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0, 120.0, 180.0],
            "profit": [50.0, 100.0, 75.0, 60.0, 90.0],
            "start_time": [
                dt_time(15, 30),
                dt_time(14, 45),
                dt_time(16, 0),
                dt_time(15, 15),
                dt_time(13, 30),
            ],
        },
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
        ),
    )

    # Create simple filters dataframe
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 1.0, 1.0, 1.0, 1.0]},
        index=pd.DatetimeIndex(
            ["2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
            name="date",
        ),
    )

    device = torch.device("cpu")
    dtype = torch.float32

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], device, dtype
    )

    # Check that weekday columns were added
    # X should have: filter_a, reward_per_risk, is_monday, is_tuesday, is_wednesday,
    # is_thursday, is_friday, open_minutes
    assert X.shape == (5, 8)

    # Check weekday values (column indices 2-6)
    # 2022-01-10 = Monday
    assert X[0, 2].item() == 1  # is_monday
    assert X[0, 3].item() == 0  # is_tuesday
    assert X[0, 4].item() == 0  # is_wednesday
    assert X[0, 5].item() == 0  # is_thursday
    assert X[0, 6].item() == 0  # is_friday

    # 2022-01-11 = Tuesday
    assert X[1, 2].item() == 0  # is_monday
    assert X[1, 3].item() == 1  # is_tuesday

    # 2022-01-12 = Wednesday
    assert X[2, 4].item() == 1  # is_wednesday

    # 2022-01-13 = Thursday
    assert X[3, 5].item() == 1  # is_thursday

    # 2022-01-14 = Friday
    assert X[4, 6].item() == 1  # is_friday

    # Check that weekday filters generate only left splits (not right)
    weekday_names = [
        "is_monday",
        "is_tuesday",
        "is_wednesday",
        "is_thursday",
        "is_friday",
    ]

    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name in weekday_names:
                assert (
                    direction == "left"
                ), f"Weekday filter {filter_name} should only have left direction"


def test_prepare_splits_open_minutes():
    """Test that open_minutes filter is added based on start_time."""
    from datetime import time as dt_time

    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0],
            "profit": [50.0, 100.0, 75.0],
            "start_time": [
                dt_time(15, 30),  # 15*60 + 30 = 930
                dt_time(14, 45),  # 14*60 + 45 = 885
                dt_time(16, 0),  # 16*60 + 0 = 960
            ],
        },
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Check that open_minutes column was added (last column)
    # X should have 8 columns: filter_a, reward_per_risk, 5 weekday filters, open_minutes
    assert X.shape == (3, 8)

    # Check open_minutes values (last column, index 7)
    assert X[0, 7].item() == 930  # 15:30
    assert X[1, 7].item() == 885  # 14:45
    assert X[2, 7].item() == 960  # 16:00

    # Check that open_minutes generates splits
    has_open_minutes = False
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "open_minutes":
                has_open_minutes = True
                break

    assert has_open_minutes, "open_minutes should generate splits"


def test_prepare_splits_open_minutes_missing_start_time():
    """Test that open_minutes defaults to 0 when start_time is missing."""
    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0],
            "profit": [50.0, 100.0],
            # No start_time column
        },
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Check that open_minutes column was added (last column, index 7) with default value 0
    assert X.shape == (2, 8)
    assert X[0, 7].item() == 0
    assert X[1, 7].item() == 0

    # With only one unique value (0), open_minutes should generate no splits
    has_open_minutes = False
    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "open_minutes":
                has_open_minutes = True
                break

    assert (
        not has_open_minutes
    ), "open_minutes should not generate splits when all values are 0"


def test_prepare_splits_open_minutes_with_none():
    """Test that open_minutes handles None values in start_time."""
    from datetime import time as dt_time

    trades_df = pd.DataFrame(
        {
            "risk": [100.0, 200.0, 150.0],
            "profit": [50.0, 100.0, 75.0],
            "start_time": [dt_time(15, 30), None, dt_time(16, 0)],
        },
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Check that open_minutes handles None (should default to 0)
    assert X[0, 7].item() == 930  # 15:30
    assert X[1, 7].item() == 0  # None -> 0
    assert X[2, 7].item() == 960  # 16:00


def test_prepare_splits_merge_identical_masks():
    """Test that splits with identical masks are merged correctly."""
    # Create a scenario where different filters produce the same mask
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # filter_a and filter_b have the same values, so they will produce identical masks
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0], "filter_b": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Check that at least one split has multiple filters
    found_merged = False
    for split in splits:
        if len(split.filters) > 1:
            found_merged = True
            # Check that the filters in this split have different names
            filter_names = [f[1] for f in split.filters]
            # If we have both filter_a and filter_b in the same split, they were merged
            if "filter_a" in filter_names and "filter_b" in filter_names:
                # Verify they have the same threshold and direction
                filter_a_entries = [f for f in split.filters if f[1] == "filter_a"]
                filter_b_entries = [f for f in split.filters if f[1] == "filter_b"]
                if filter_a_entries and filter_b_entries:
                    # They should have matching thresholds and directions
                    assert abs(filter_a_entries[0][2] - filter_b_entries[0][2]) < 0.01
                    assert filter_a_entries[0][3] == filter_b_entries[0][3]
                break

    assert found_merged, "Expected to find at least one merged split"

    # Verify all splits have unique masks
    masks = [split.mask for split in splits]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not torch.equal(
                masks[i], masks[j]
            ), "All splits should have unique masks"


def test_prepare_splits_reward_per_risk_right_only():
    """Test that reward_per_risk filter only generates right splits (not left)."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Check that reward_per_risk exists and has only right direction splits
    has_reward_per_risk_left = False
    has_reward_per_risk_right = False

    for split in splits:
        for filter_idx, filter_name, threshold, direction in split.filters:
            if filter_name == "reward_per_risk":
                if direction == "left":
                    has_reward_per_risk_left = True
                elif direction == "right":
                    has_reward_per_risk_right = True

    assert not has_reward_per_risk_left, "reward_per_risk should not have left splits"
    assert (
        has_reward_per_risk_right
    ), "reward_per_risk should have at least one right split"
