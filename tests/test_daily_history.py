"""Daily fallback sessions preserve Oracle intraday and calendar boundaries."""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from ifera.allocation.portfolio_allocation import _nyse_market_dates
from ifera.strategy_optimization._data import StrategyData
from ifera.strategy_optimization import _daily_history


def _strategy(trades=(("SPX", date(2024, 1, 10)),)):
    days = [session for _, session in trades]
    return StrategyData(
        pl.DataFrame(),
        pl.DataFrame(
            {
                "symbol": [symbol for symbol, _ in trades],
                "opened": [datetime.combine(session, time(13)) for session in days],
            }
        ),
        pl.DataFrame(),
        tuple(_nyse_market_dates(min(days), max(days))["date"].to_list()),
        Path("unused.csv"),
    )


def _minutes(session, symbol="SPX", count=390):
    start = datetime.combine(session, time(9, 31))
    return pl.DataFrame(
        {
            "symbol": [symbol] * count,
            "datetime": [start + timedelta(minutes=index) for index in range(count)],
            "price": [5000.0 + index / 100 for index in range(count)],
        },
        schema={
            "symbol": pl.String,
            "datetime": pl.Datetime("us"),
            "price": pl.Float64,
        },
    )


def _daily(dates):
    return pl.DataFrame(
        {
            "date": list(dates),
            "open": [100.0] * len(dates),
            "high": [102.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [101.0] * len(dates),
        },
        schema={
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


@pytest.fixture
def download(monkeypatch):
    mock = MagicMock(side_effect=lambda _symbol, dates: _daily(dates))
    monkeypatch.setattr(_daily_history, "fetch_daily_history", mock)
    return mock


def test_complete_oracle_history_needs_no_download(download):
    underlying = pl.concat([_minutes(date(2024, 1, 8)), _minutes(date(2024, 1, 9))])
    result = _daily_history.complete_daily_history(_strategy(), underlying, 3)
    download.assert_not_called()
    assert result.is_empty()
    assert result.schema == pl.Schema(
        {
            "symbol": pl.String,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        }
    )


def test_missing_session_fetches_only_its_daily_bar(download):
    original = _minutes(date(2024, 1, 8))
    result = _daily_history.complete_daily_history(_strategy(), original, 3)
    download.assert_called_once_with("SPX", (date(2024, 1, 9),))
    assert result["date"].to_list() == [date(2024, 1, 9)]
    assert result["symbol"].to_list() == ["SPX"]
    assert original.equals(_minutes(date(2024, 1, 8)))


def test_missing_and_partial_sessions_share_one_download(download):
    partial = _minutes(date(2024, 1, 8)).slice(1)
    result = _daily_history.complete_daily_history(_strategy(), partial, 3)
    expected = (date(2024, 1, 8), date(2024, 1, 9))
    download.assert_called_once_with("SPX", expected)
    assert result["date"].to_list() == list(expected)


@pytest.mark.parametrize("bad_price", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_required_snapshot_makes_daily_history_incomplete(download, bad_price):
    underlying = pl.concat(
        [
            _minutes(date(2024, 1, 8)),
            _minutes(date(2024, 1, 9)).with_columns(
                pl.when(pl.int_range(pl.len()) == 150)
                .then(pl.lit(bad_price, dtype=pl.Float64))
                .otherwise(pl.col("price"))
                .alias("price")
            ),
        ]
    )
    _daily_history.complete_daily_history(_strategy(), underlying, 3)
    download.assert_called_once_with("SPX", (date(2024, 1, 9),))


def test_duplicate_timestamp_does_not_hide_a_missing_minute(download):
    day = _minutes(date(2024, 1, 9))
    underlying = pl.concat([_minutes(date(2024, 1, 8)), day.slice(1), day.slice(1, 1)])
    _daily_history.complete_daily_history(_strategy(), underlying, 3)
    download.assert_called_once_with("SPX", (date(2024, 1, 9),))


def test_whole_minute_alignment_is_required(download):
    day = _minutes(date(2024, 1, 9)).with_columns(
        pl.when(pl.int_range(pl.len()) == 150)
        .then(pl.col("datetime") + pl.duration(seconds=1))
        .otherwise(pl.col("datetime"))
        .alias("datetime")
    )
    _daily_history.complete_daily_history(
        _strategy(), pl.concat([_minutes(date(2024, 1, 8)), day]), 3
    )
    download.assert_called_once_with("SPX", (date(2024, 1, 9),))


def test_optional_open_snapshot_does_not_trigger_download(download):
    underlying = pl.concat([_minutes(date(2024, 1, 8)), _minutes(date(2024, 1, 9))])
    opening = pl.DataFrame(
        {
            "symbol": ["SPX"],
            "datetime": [datetime(2024, 1, 8, 9, 30)],
            "price": [4990.0],
        }
    )
    result = _daily_history.complete_daily_history(
        _strategy(), pl.concat([opening, underlying]), 3
    )
    assert result.is_empty()
    download.assert_not_called()


def test_early_close_session_is_complete_at_1300(download):
    data = _strategy((("SPX", date(2024, 12, 2)),))
    result = _daily_history.complete_daily_history(
        data, _minutes(date(2024, 11, 29), count=210), 3
    )
    assert result.is_empty()
    download.assert_not_called()


def test_early_close_session_requires_its_closing_snapshot(download):
    data = _strategy((("SPX", date(2024, 12, 2)),))
    _daily_history.complete_daily_history(
        data, _minutes(date(2024, 11, 29), count=209), 3
    )
    download.assert_called_once_with("SPX", (date(2024, 11, 29),))


def test_latest_trade_day_and_future_dates_are_never_downloaded(download):
    current_and_future = pl.concat(
        [_minutes(date(2024, 1, 10), count=1), _minutes(date(2024, 1, 11))]
    )
    result = _daily_history.complete_daily_history(_strategy(), current_and_future, 3)
    expected = (date(2024, 1, 8), date(2024, 1, 9))
    download.assert_called_once_with("SPX", expected)
    assert result["date"].to_list() == list(expected)
    assert current_and_future.height == 391


def test_previous_trade_day_can_be_history_for_a_later_trade(download):
    data = _strategy((("SPX", date(2024, 1, 8)), ("SPX", date(2024, 1, 10))))
    underlying = pl.concat(
        [_minutes(date(2024, 1, 8), count=1), _minutes(date(2024, 1, 9))]
    )
    result = _daily_history.complete_daily_history(data, underlying, 1)
    download.assert_called_once_with("SPX", (date(2024, 1, 8),))
    assert result["date"].to_list() == [date(2024, 1, 8)]


def test_each_symbol_uses_its_own_last_trade_boundary(download):
    data = _strategy((("SPX", date(2024, 1, 10)), ("AAPL", date(2024, 1, 9))))
    _daily_history.complete_daily_history(data, pl.DataFrame(), 1)
    assert download.call_args_list[0].args == (
        "SPX",
        (date(2024, 1, 8), date(2024, 1, 9)),
    )
    assert download.call_args_list[1].args == ("AAPL", (date(2024, 1, 8),))


def test_weekends_and_holidays_are_not_requested(download):
    data = _strategy((("SPX", date(2024, 1, 16)),))
    _daily_history.complete_daily_history(data, pl.DataFrame(), 4)
    download.assert_called_once_with("SPX", (date(2024, 1, 12),))


def test_missing_downloaded_bar_is_explicit(download):
    download.side_effect = lambda _symbol, _dates: _daily([date(2024, 1, 8)])
    with pytest.raises(ValueError, match="Missing daily OHLC for SPX: 2024-01-09"):
        _daily_history.complete_daily_history(_strategy(), pl.DataFrame(), 3)


def test_duplicate_downloaded_bars_are_explicit(download):
    download.side_effect = lambda _symbol, _dates: _daily(
        [date(2024, 1, 8), date(2024, 1, 8), date(2024, 1, 9)]
    )
    with pytest.raises(ValueError, match="Duplicate daily OHLC dates"):
        _daily_history.complete_daily_history(_strategy(), pl.DataFrame(), 3)


def test_provider_extra_current_or_future_bars_are_discarded(download):
    download.side_effect = lambda _symbol, _dates: _daily(
        [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]
    )
    result = _daily_history.complete_daily_history(_strategy(), pl.DataFrame(), 3)
    assert result["date"].to_list() == [date(2024, 1, 8), date(2024, 1, 9)]
