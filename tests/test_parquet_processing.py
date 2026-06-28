"""Tests for parquet-based processing and loading."""

from __future__ import annotations

import datetime as dt

import polars as pl

from ifera.data_loading import load_data
from ifera.data_processing import process_data
from ifera.enums import Source
from ifera.file_utils import make_instrument_path
from ifera.settings import settings


def test_process_data_writes_trade_date_partitioned_dataset(
    tmp_path, monkeypatch, base_instrument_config
):
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    start = dt.datetime(2022, 1, 9, 18, 0)
    timestamps = [
        start + index * base_instrument_config.time_step
        for index in range(base_instrument_config.total_steps)
    ]
    raw_df = pl.DataFrame(
        {
            "date_time": timestamps,
            "open": [float(index + 1) for index in range(len(timestamps))],
            "high": [float(index + 2) for index in range(len(timestamps))],
            "low": [float(index) for index in range(len(timestamps))],
            "close": [float(index + 1.5) for index in range(len(timestamps))],
            "volume": [100 + index for index in range(len(timestamps))],
        }
    )

    process_data(raw_df, instrument=base_instrument_config, zipfile=False)

    dataset_root = make_instrument_path(Source.PROCESSED, base_instrument_config)
    partition_names = sorted(
        path.name for path in dataset_root.iterdir() if path.is_dir()
    )
    loaded = load_data(raw=False, instrument=base_instrument_config, zipfile=False)

    assert partition_names == ["trade_date=2022-01-10"]
    assert loaded.height == base_instrument_config.total_steps
    assert loaded["trade_date"].n_unique() == 1
    assert loaded["trade_date"][0] == dt.date(2022, 1, 10).toordinal()
