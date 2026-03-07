"""Tests for Polars-based data processing helpers."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast

import polars as pl

from ifera.config import BaseInstrumentConfig
from ifera.data_processing import (
    add_missing_rows,
    aggregate_by_second,
    calculate_time_columns,
)


@dataclass
class _InstrumentStub:
    trading_start: timedelta


def test_add_missing_rows_returns_polars_and_fills_time_gaps():
    """Missing offset rows should be added and forward-filled in Polars."""
    group = pl.DataFrame(
        {
            "trade_date": [datetime(2022, 1, 10).date(), datetime(2022, 1, 10).date()],
            "offset_time_seconds": [0, 120],
            "open": [10.0, 12.0],
            "high": [11.0, 13.0],
            "low": [9.0, 11.0],
            "close": [10.5, 12.5],
            "volume": [100, 200],
        }
    )

    filled = add_missing_rows(group=group, start_time=0, end_time=120, time_step=60)

    assert isinstance(filled, pl.DataFrame)
    assert filled.select("offset_time_seconds").to_series().to_list() == [0, 60, 120]
    assert filled.select("volume").to_series().to_list() == [100, 0, 200]


def test_aggregate_by_second_returns_polars_ohlcv_with_vwap():
    """Second-level aggregation should compute OHLCV and VWAP in Polars."""
    raw = pl.DataFrame(
        {
            "Date": ["2022-01-10", "2022-01-10", "2022-01-10"],
            "Time": ["09:30:00", "09:30:00", "09:30:01"],
            "Bid": [10.0, 12.0, 20.0],
            "Ask": [11.0, 13.0, 21.0],
            "Price": [10.5, 12.5, 20.5],
            "Size": [1, 3, 2],
        }
    )

    aggregated = aggregate_by_second(raw, max_decimals=2)

    assert isinstance(aggregated, pl.DataFrame)
    assert aggregated.height == 2
    assert "VWAP" in aggregated.columns
    first_row = aggregated.filter(pl.col("Time") == "09:30:00")
    assert first_row.select("Volume").to_series().item() == 4
    assert first_row.select("VWAP").to_series().item() == 12.0


def test_calculate_time_columns_returns_polars_with_offsets():
    """Datetime decomposition should produce Polars date and offset columns."""
    instrument = _InstrumentStub(trading_start=timedelta(hours=9, minutes=30))
    raw = pl.DataFrame(
        {
            "date_time": [
                datetime(2022, 1, 10, 9, 30, 0),
                datetime(2022, 1, 10, 10, 0, 0),
            ],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [100, 120],
        }
    )

    with_time = calculate_time_columns(raw, cast(BaseInstrumentConfig, instrument))

    assert isinstance(with_time, pl.DataFrame)
    assert with_time.select("offset_time_seconds").to_series().to_list() == [0, 1800]
    assert with_time.select("trade_date").to_series().to_list() == [
        datetime(2022, 1, 10).date(),
        datetime(2022, 1, 10).date(),
    ]
