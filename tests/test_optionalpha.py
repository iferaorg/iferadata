"""Tests for the optionalpha module."""

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from ifera.optionalpha import (
    parse_trade_log,
    parse_filter_log,
    _extract_dollar_amount,
    _parse_time,
    prepare_splits,
    Split,
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

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df, filters_df, spread_width, [], [], device, dtype
    )

    # Check X shape - should have 3 rows and 3 columns (2 filters + reward_per_risk)
    assert X.shape == (3, 3)
    assert X.device == device
    assert X.dtype == dtype

    # Check y shape and values
    assert y.shape == (3,)
    assert y.device == device
    assert y.dtype == dtype
    # RoR should be profit / risk
    expected_y = torch.tensor([0.5, -0.5, 0.5], dtype=dtype, device=device)
    assert torch.allclose(y, expected_y)

    # Check reward_per_risk column (last column in X)
    # reward_per_risk = (spread_width * 100 - risk) / risk
    expected_rpr = [
        (20 * 100 - 100) / 100,
        (20 * 100 - 200) / 200,
        (20 * 100 - 150) / 150,
    ]
    assert torch.allclose(
        X[:, 2], torch.tensor(expected_rpr, dtype=dtype, device=device)
    )

    # Check splits - should have splits for all 3 columns (filter_a, filter_b, reward_per_risk)
    # filter_a has values [1, 2, 3] -> thresholds at 1.5, 2.5 -> 4 splits (2 thresholds * 2 directions)
    # filter_b has values [10, 20, 30] -> thresholds at 15, 25 -> 4 splits
    # reward_per_risk has 3 unique values -> 2 thresholds -> 4 splits
    # Total: 12 splits
    assert len(splits) == 12

    # Check split properties
    for split in splits:
        assert isinstance(split, Split)
        assert split.filter_idx in [
            0,
            1,
            2,
        ]  # Three columns (2 filters + reward_per_risk)
        assert split.direction in ["left", "right"]
        assert split.mask.dtype == torch.bool
        assert split.mask.shape == (3,)  # 3 samples

    # Check exclusion mask shape
    assert exclusion_mask.shape == (len(splits), len(splits))
    assert exclusion_mask.dtype == torch.bool


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

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        left_only_filters=["filter_a"],
        right_only_filters=[],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Should have left splits for filter_a only (1 threshold * 1 direction = 1 split)
    # Plus splits for reward_per_risk
    filter_a_splits = [s for s in splits if s.filter_name == "filter_a"]
    assert len(filter_a_splits) == 1
    assert all(s.direction == "left" for s in filter_a_splits)


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

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df,
        filters_df,
        20,
        left_only_filters=[],
        right_only_filters=["filter_a"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Should have right splits for filter_a only (1 threshold * 1 direction = 1 split)
    # Plus splits for reward_per_risk
    filter_a_splits = [s for s in splits if s.filter_name == "filter_a"]
    assert len(filter_a_splits) == 1
    assert all(s.direction == "right" for s in filter_a_splits)


def test_prepare_splits_exclusion_mask_same_filter_direction():
    """Test that splits from same filter and direction are mutually exclusive."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Should have splits for filter_a and reward_per_risk
    # filter_a: 2 thresholds * 2 directions = 4 splits
    # reward_per_risk: 2 thresholds * 2 directions = 4 splits
    # Total: 8 splits
    assert len(splits) == 8

    # Find left splits for filter_a
    left_splits = [
        i
        for i, s in enumerate(splits)
        if s.direction == "left" and s.filter_name == "filter_a"
    ]
    assert len(left_splits) == 2

    # Left splits for filter_a should exclude each other
    assert exclusion_mask[left_splits[0], left_splits[1]].item()
    assert exclusion_mask[left_splits[1], left_splits[0]].item()

    # Find right splits for filter_a
    right_splits = [
        i
        for i, s in enumerate(splits)
        if s.direction == "right" and s.filter_name == "filter_a"
    ]
    assert len(right_splits) == 2

    # Right splits for filter_a should exclude each other
    assert exclusion_mask[right_splits[0], right_splits[1]].item()
    assert exclusion_mask[right_splits[1], right_splits[0]].item()


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

    X, y, splits, exclusion_mask = prepare_splits(
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

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # filter_b should generate 4 splits (2 thresholds * 2 directions)
    # reward_per_risk should also generate splits (it has 3 unique values)
    # So total should be 8 splits
    assert len(splits) >= 4
    # Verify filter_a generates no splits
    assert not any(s.filter_name == "filter_a" for s in splits)
    # Verify filter_b generates splits
    assert any(s.filter_name == "filter_b" for s in splits)


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

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Find the left split at threshold 1.5
    left_split_1_5 = next(
        s for s in splits if s.direction == "left" and abs(s.threshold - 1.5) < 0.01
    )

    # Should include only the first row (value 1.0 <= 1.5)
    expected_mask = torch.tensor([True, False, False], dtype=torch.bool)
    assert torch.equal(left_split_1_5.mask, expected_mask)

    # Find the right split at threshold 2.5
    right_split_2_5 = next(
        s for s in splits if s.direction == "right" and abs(s.threshold - 2.5) < 0.01
    )

    # Should include only the last row (value 3.0 >= 2.5)
    expected_mask = torch.tensor([False, False, True], dtype=torch.bool)
    assert torch.equal(right_split_2_5.mask, expected_mask)


def test_prepare_splits_exclusion_empty_intersection():
    """Test that splits with empty intersection are mutually exclusive."""
    trades_df = pd.DataFrame(
        {"risk": [100.0, 200.0, 150.0], "profit": [50.0, 100.0, 75.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    # Two filters with non-overlapping splits
    filters_df = pd.DataFrame(
        {"filter_a": [1.0, 5.0, 10.0], "filter_b": [10.0, 5.0, 1.0]},
        index=pd.DatetimeIndex(["2022-01-10", "2022-01-11", "2022-01-12"], name="date"),
    )

    X, y, splits, exclusion_mask = prepare_splits(
        trades_df, filters_df, 20, [], [], torch.device("cpu"), torch.float32
    )

    # Check for splits that have empty intersection
    for i, split_i in enumerate(splits):
        for j, split_j in enumerate(splits):
            if i != j:
                combined = split_i.mask & split_j.mask
                if not combined.any():
                    assert exclusion_mask[i, j].item(), (
                        f"Splits {i} and {j} have empty intersection "
                        f"but are not marked as exclusive"
                    )


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

    X, y, splits, exclusion_mask = prepare_splits(
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

    assert exclusion_mask.dtype == torch.bool
    assert exclusion_mask.device == device
