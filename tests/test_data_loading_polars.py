"""Tests for Polars-based data loading."""

from datetime import date, datetime
from unittest.mock import Mock

import polars as pl

from ifera.data_loading import load_data, read_csv_with_progress
from ifera.parquet_datasets import write_parquet_dataset


def test_read_csv_with_progress_returns_polars_dataframe(tmp_path):
    """CSV chunk loading should return a Polars DataFrame."""
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "2022-01-01,09:30:00,1.0,2.0,0.5,1.5,100\n"
        "2022-01-01,10:00:00,1.5,2.5,1.0,2.0,200\n",
        encoding="utf-8",
    )

    kwargs = {
        "header": None,
        "names": ["date", "time", "open", "high", "low", "close", "volume"],
        "dtype": {
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "int32",
        },
    }

    loaded = read_csv_with_progress(str(csv_path), kwargs, zipfile=False)

    assert isinstance(loaded, pl.DataFrame)
    assert loaded.height == 2
    assert loaded.columns == ["date", "time", "open", "high", "low", "close", "volume"]


def test_load_data_raw_returns_polars_with_datetime_column(tmp_path, monkeypatch):
    """Raw load should parse date/time into a single datetime column."""
    dataset_root = tmp_path / "raw.parquet"
    write_parquet_dataset(
        dataset_root,
        pl.DataFrame(
            {
                "date": [date(2022, 1, 1), date(2022, 1, 1)],
                "time": ["09:30:00", "10:00:00"],
                "open": [1.0, 1.5],
                "high": [2.0, 2.5],
                "low": [0.5, 1.0],
                "close": [1.5, 2.0],
                "volume": [100, 200],
            }
        ),
        partition_columns=["date"],
    )

    monkeypatch.setattr(
        "ifera.data_loading.make_instrument_path",
        lambda **_: dataset_root,
    )

    loaded = load_data(raw=True, instrument=Mock(), zipfile=False)

    assert isinstance(loaded, pl.DataFrame)
    assert "date_time" in loaded.columns
    assert "date" not in loaded.columns
    assert "time" not in loaded.columns
    assert loaded.select("date_time").to_series().to_list()[0] == datetime(
        2022, 1, 1, 9, 30
    )


def test_load_data_processed_returns_polars_dataframe(tmp_path, monkeypatch):
    """Processed load should retain the expected table schema in Polars."""
    trade_date_ord = date(2022, 1, 1).toordinal()
    dataset_root = tmp_path / "processed.parquet"
    write_parquet_dataset(
        dataset_root,
        pl.DataFrame(
            {
                "trade_date": [date(2022, 1, 1), date(2022, 1, 1)],
                "ord_date": [trade_date_ord, trade_date_ord],
                "time_seconds": [34200, 36000],
                "ord_trade_date": [trade_date_ord, trade_date_ord],
                "offset_time_seconds": [0, 1800],
                "open": [1.0, 1.5],
                "high": [2.0, 2.5],
                "low": [0.5, 1.0],
                "close": [1.5, 2.0],
                "volume": [100, 200],
            }
        ),
        partition_columns=["trade_date"],
    )

    monkeypatch.setattr(
        "ifera.data_loading.make_instrument_path",
        lambda **_: dataset_root,
    )

    loaded = load_data(raw=False, instrument=Mock(), zipfile=False)

    assert isinstance(loaded, pl.DataFrame)
    assert loaded.height == 2
    assert loaded.columns == [
        "date",
        "time",
        "trade_date",
        "offset_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
