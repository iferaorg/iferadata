"""Tests for the optionalpha module."""

import os
from pathlib import Path

import pandas as pd
import pytest

from ifera.optionalpha import parse_trade_log, _extract_dollar_amount, _parse_time
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
        "date",
        "start_time",
        "end_time",
        "status",
        "risk",
        "profit",
    ]

    # Check first row
    from datetime import time as dt_time

    assert df.iloc[0]["symbol"] == "SPX"
    assert df.iloc[0]["trade_type"] == "Long Call"
    assert df.iloc[0]["date"] == pd.Timestamp("2022-01-10")
    assert df.iloc[0]["start_time"] == dt_time(15, 47)
    assert df.iloc[0]["end_time"] == dt_time(16, 0)
    assert df.iloc[0]["status"] == "Expired"
    assert df.iloc[0]["risk"] == 380.0
    assert df.iloc[0]["profit"] == 1149.0

    # Check second row (negative P/L)
    assert df.iloc[1]["symbol"] == "SPX"
    assert df.iloc[1]["trade_type"] == "Long Call"
    assert df.iloc[1]["date"] == pd.Timestamp("2022-01-14")
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
        "date",
        "start_time",
        "end_time",
        "status",
        "risk",
        "profit",
    ]

    # Check data types
    assert df["symbol"].dtype == object
    assert df["trade_type"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
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
