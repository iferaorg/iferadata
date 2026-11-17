"""Explicit date bounds preserve ledger identity and full market-day calendars."""

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ifera.strategy_optimization._data import StrategyData
from ifera.strategy_optimization._date_window import select_strategy_dates


def _data(reverse=False):
    days = [date(2024, 1, 8), date(2024, 1, 10), date(2024, 1, 12)]
    if reverse:
        days.reverse()
    raw = pl.DataFrame(
        {"Source": [str(day) for day in days], "Custom": ["a", "b", "c"]}
    )
    trades = pl.DataFrame(
        {
            "trade_index": [0, 1, 2],
            "opened": [
                datetime.combine(day, datetime.min.time()) + timedelta(hours=13)
                for day in days
            ],
            "ror": [1.0, 2.0, 3.0],
        }
    )
    legs = pl.DataFrame(
        {
            "trade_index": [0, 0, 1, 1, 2, 2],
            "direction": ["short", "long"] * 3,
            "strike": [100.0, 90.0, 105.0, 95.0, 110.0, 100.0],
        }
    )
    calendar = tuple(date(2024, 1, 8) + timedelta(days=index) for index in range(5))
    return StrategyData(raw, trades, legs, calendar, Path("original.csv"))


def test_no_bounds_returns_identical_data():
    data = _data()
    assert select_strategy_dates(data) is data


@pytest.mark.parametrize("reverse", [False, True])
def test_inclusive_filter_preserves_source_order_and_reindexes_legs(reverse):
    data = _data(reverse)
    original = (data.raw.clone(), data.trades.clone(), data.legs.clone())
    result = select_strategy_dates(data, date(2024, 1, 9), date(2024, 1, 12))
    indices = [0, 1] if reverse else [1, 2]
    assert result.raw.equals(data.raw[indices])
    assert result.raw.columns == data.raw.columns
    assert result.trades["trade_index"].to_list() == [0, 1]
    assert result.trades["ror"].to_list() == ([1.0, 2.0] if reverse else [2.0, 3.0])
    assert result.legs["trade_index"].to_list() == [0, 0, 1, 1]
    expected_strikes = (
        [100.0, 90.0, 105.0, 95.0] if reverse else [105.0, 95.0, 110.0, 100.0]
    )
    assert result.legs["strike"].to_list() == expected_strikes
    assert result.path == data.path
    for before, after in zip(original, (data.raw, data.trades, data.legs)):
        assert before.equals(after)


def test_idle_boundary_sessions_remain_in_full_calendar():
    result = select_strategy_dates(_data(), date(2024, 1, 9), date(2024, 1, 11))
    assert result.trades["opened"].dt.date().to_list() == [date(2024, 1, 10)]
    assert result.dates == (date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11))


def test_unbounded_and_wider_endpoints_keep_original_calendar_extent():
    data = _data()
    assert (
        select_strategy_dates(data, end_date=date(2024, 1, 10)).dates == data.dates[:3]
    )
    assert (
        select_strategy_dates(data, start_date=date(2024, 1, 10)).dates
        == data.dates[2:]
    )
    wider = select_strategy_dates(data, date(2020, 1, 1), date(2025, 1, 1))
    assert wider.dates == data.dates
    assert wider.raw.equals(data.raw)
    assert wider.trades.equals(data.trades)
    assert wider.legs.equals(data.legs)


@pytest.mark.parametrize(
    "invalid", ["2024-01-09", datetime(2024, 1, 9), 20240109, False]
)
@pytest.mark.parametrize("argument", ["start_date", "end_date"])
def test_invalid_date_types_are_rejected(invalid, argument):
    with pytest.raises(TypeError, match=f"{argument} must be a date"):
        select_strategy_dates(_data(), **{argument: invalid})


def test_reversed_range_is_rejected():
    with pytest.raises(ValueError, match="start_date must not be after end_date"):
        select_strategy_dates(_data(), date(2024, 1, 12), date(2024, 1, 8))


@pytest.mark.parametrize(
    "start,end",
    [
        (date(2024, 1, 9), date(2024, 1, 9)),
        (date(2025, 1, 1), None),
        (None, date(2023, 1, 1)),
    ],
)
def test_empty_trade_window_is_rejected(start, end):
    with pytest.raises(ValueError, match="contains no strategy trades"):
        select_strategy_dates(_data(), start, end)
