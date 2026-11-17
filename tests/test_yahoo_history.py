"""Daily Yahoo fallback preserves nominal prices and a reusable CSV cache."""

from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ifera.settings import settings
from ifera.strategy_optimization import _yahoo_history as yahoo


def _history(days, close=None, splits=None, timezone="America/New_York"):
    close = np.asarray(close if close is not None else [25.0] * len(days), dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.1,
            "Low": close * 0.9,
            "Close": close,
            "Adj Close": close * 0.5,
            "Stock Splits": splits if splits is not None else [0.0] * len(days),
            "Dividends": [2.0] * len(days),
        },
        index=pd.DatetimeIndex(days, tz=timezone),
    )


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(yahoo, "_market_today", lambda: date(2024, 1, 20))
    ticker = MagicMock()
    ticker.history.return_value = _history(["2024-01-02"])
    factory = MagicMock(return_value=ticker)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=factory))
    return factory, ticker.history


def test_split_after_requested_range_is_reversed_and_download_flags_are_explicit(
    provider, tmp_path
):
    factory, history = provider
    history.return_value = _history(
        ["2024-01-02", "2024-01-15"], close=[25.0, 100.0], splits=[0.0, 4.0]
    )
    result = yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    assert result["date"].to_list() == [date(2024, 1, 2)]
    assert result.select("open", "high", "low", "close").row(0) == pytest.approx(
        (100.0, 110.0, 90.0, 100.0)
    )
    factory.assert_called_once_with("^GSPC")
    assert history.call_args.kwargs == {
        "start": "2024-01-02",
        "end": "2024-01-21",
        "interval": "1d",
        "auto_adjust": False,
        "back_adjust": False,
        "actions": True,
        "repair": False,
        "rounding": False,
        "keepna": True,
        "prepost": False,
        "timeout": 30,
    }
    cached = pl.read_csv(tmp_path / "SPX.csv", try_parse_dates=True)
    assert cached["symbol"].to_list() == ["SPX"]
    assert cached["yahoo_symbol"].to_list() == ["^GSPC"]
    assert cached["price_basis"].to_list() == ["nominal_v1"]
    assert cached["date"].to_list() == [date(2024, 1, 2)]


def test_multiple_and_reverse_splits_exclude_each_events_own_bar(provider, tmp_path):
    _, history = provider
    history.return_value = _history(
        ["2024-01-02", "2024-01-15", "2024-01-17"],
        close=[50, 100, 150],
        splits=[3, 4, 0.5],
    )
    result = yahoo.fetch_daily_history(
        "AAPL", {date(2024, 1, day) for day in [2, 15, 17]}, tmp_path
    )
    assert result["close"].to_list() == [100.0, 50.0, 150.0]


def test_dividends_and_adj_close_do_not_change_nominal_prices(provider, tmp_path):
    result = yahoo.fetch_daily_history("SPY", {date(2024, 1, 2)}, tmp_path)
    assert result["close"].to_list() == [25.0]


def test_split_action_row_without_prices_still_adjusts_earlier_bar(provider, tmp_path):
    _, history = provider
    history.return_value = _history(
        ["2024-01-02", "2024-01-15"], close=[25.0, np.nan], splits=[0, 4]
    )
    result = yahoo.fetch_daily_history("AAPL", {date(2024, 1, 2)}, tmp_path)
    assert result["close"].to_list() == [100.0]


def test_announced_future_split_is_not_applied(provider, tmp_path):
    _, history = provider
    history.return_value = _history(["2024-01-02", "2024-02-01"], splits=[0, 10])
    result = yahoo.fetch_daily_history("AAPL", {date(2024, 1, 2)}, tmp_path)
    assert result["close"].to_list() == [25.0]


def test_cache_hit_requires_no_yfinance_import_or_download(
    provider, tmp_path, monkeypatch
):
    _, history = provider
    required = {date(2024, 1, 2)}
    first = yahoo.fetch_daily_history("SPX", required, tmp_path)
    monkeypatch.setattr(yahoo, "import_module", MagicMock(side_effect=AssertionError))
    second = yahoo.fetch_daily_history("SPX", required, tmp_path)
    assert second.equals(first)
    assert history.call_count == 1


def test_incremental_download_preserves_already_nominal_cached_prices(
    provider, tmp_path
):
    _, history = provider
    history.return_value = _history(
        ["2024-01-02", "2024-01-15"], close=[25.0, 100.0], splits=[0, 4]
    )
    yahoo.fetch_daily_history("AAPL", {date(2024, 1, 2)}, tmp_path)
    history.return_value = _history(
        ["2024-01-03", "2024-01-15", "2024-01-18"],
        close=[20.0, 80.0, 160.0],
        splits=[0, 4, 2],
    )
    required = {date(2024, 1, 2), date(2024, 1, 3)}
    result = yahoo.fetch_daily_history("AAPL", required, tmp_path)
    assert result["close"].to_list() == [100.0, 160.0]
    assert history.call_count == 2
    assert history.call_args.kwargs["start"] == "2024-01-03"
    assert yahoo.fetch_daily_history("AAPL", required, tmp_path).equals(result)
    assert history.call_count == 2


@pytest.mark.parametrize(
    "symbol, ticker",
    [
        ("NDX", "^NDX"),
        ("DJI", "^DJI"),
        ("RUT", "^RUT"),
        ("VIX", "^VIX"),
        ("AAPL", "AAPL"),
    ],
)
def test_index_aliases_and_regular_symbols(provider, tmp_path, symbol, ticker):
    factory, _ = provider
    yahoo.fetch_daily_history(symbol, {date(2024, 1, 2)}, tmp_path)
    factory.assert_called_once_with(ticker)


def test_default_cache_location_and_safe_symbol_filename(
    provider, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    yahoo.fetch_daily_history("a/b", {date(2024, 1, 2)})
    assert (tmp_path / "cache/strategy_optimization/daily/A%2FB.csv").exists()


def test_no_requested_dates_has_no_filesystem_or_network_effect(provider, tmp_path):
    factory, _ = provider
    result = yahoo.fetch_daily_history("SPX", set(), tmp_path / "missing")
    assert result.schema == yahoo._BAR_SCHEMA
    assert result.is_empty()
    factory.assert_not_called()
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("required", [{date(2024, 1, 20)}, {date(2024, 1, 21)}])
def test_current_or_future_days_cannot_be_cached(provider, tmp_path, required):
    factory, _ = provider
    with pytest.raises(ValueError, match="completed sessions"):
        yahoo.fetch_daily_history("SPX", required, tmp_path)
    factory.assert_not_called()


def test_rejects_timestamp_dates_and_empty_symbols(provider, tmp_path):
    with pytest.raises(TypeError, match="dates"):
        yahoo.fetch_daily_history("SPX", {datetime(2024, 1, 2)}, tmp_path)
    with pytest.raises(ValueError, match="symbol"):
        yahoo.fetch_daily_history(" ", {date(2024, 1, 2)}, tmp_path)


@pytest.mark.parametrize(
    "problem", ["basis", "symbol", "ticker", "columns", "date", "duplicate", "price"]
)
def test_corrupt_or_incompatible_cache_is_not_silently_replaced(
    provider, tmp_path, problem
):
    _, history = provider
    yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    path = tmp_path / "SPX.csv"
    frame = pl.read_csv(path)
    if problem == "basis":
        frame = frame.with_columns(pl.lit("adjusted").alias("price_basis"))
    elif problem == "symbol":
        frame = frame.with_columns(pl.lit("SPY").alias("symbol"))
    elif problem == "ticker":
        frame = frame.with_columns(pl.lit("SPY").alias("yahoo_symbol"))
    elif problem == "columns":
        frame = frame.drop("price_basis")
    elif problem == "date":
        frame = frame.with_columns(pl.lit("not-a-date").alias("date"))
    elif problem == "duplicate":
        frame = pl.concat([frame, frame])
    else:
        frame = frame.with_columns(pl.lit(-1.0).alias("close"))
    frame.write_csv(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid Yahoo daily cache"):
        yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    assert path.read_bytes() == original
    assert history.call_count == 1


@pytest.mark.parametrize(
    "problem", ["empty", "missing_date", "missing_splits", "duplicate", "not_frame"]
)
def test_bad_provider_history_is_not_cached(provider, tmp_path, problem):
    _, history = provider
    if problem == "empty":
        history.return_value = pd.DataFrame()
    elif problem == "missing_date":
        history.return_value = _history(["2024-01-03"])
    elif problem == "missing_splits":
        history.return_value = _history(["2024-01-02"]).drop(columns="Stock Splits")
    elif problem == "duplicate":
        history.return_value = _history(["2024-01-02", "2024-01-02"])
    else:
        history.return_value = None
    with pytest.raises(ValueError, match="Yahoo"):
        yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    assert not (tmp_path / "SPX.csv").exists()


@pytest.mark.parametrize("split", [np.nan, np.inf, -2])
def test_invalid_split_ratios_are_not_cached(provider, tmp_path, split):
    _, history = provider
    history.return_value = _history(["2024-01-02"], splits=[split])
    with pytest.raises(ValueError, match="split ratios"):
        yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    assert not (tmp_path / "SPX.csv").exists()


@pytest.mark.parametrize(
    "column, value", [("Open", 0), ("High", 1), ("Low", 100), ("Close", np.inf)]
)
def test_invalid_price_bars_are_not_cached(provider, tmp_path, column, value):
    _, history = provider
    history.return_value.loc[:, column] = value
    with pytest.raises(ValueError, match="positive OHLC"):
        yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    assert not (tmp_path / "SPX.csv").exists()


def test_network_failure_is_clear_and_leaves_no_cache(provider, tmp_path):
    _, history = provider
    history.side_effect = OSError("network unavailable")
    with pytest.raises(
        ValueError, match="Could not download Yahoo daily history.*network"
    ):
        yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    assert not (tmp_path / "SPX.csv").exists()


def test_session_dates_preserve_provider_timezone(provider, tmp_path):
    _, history = provider
    history.return_value = _history(["2024-01-02"], timezone="Asia/Tokyo")
    result = yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    assert result["date"].to_list() == [date(2024, 1, 2)]


def test_failed_atomic_replace_preserves_existing_csv(provider, tmp_path, monkeypatch):
    _, history = provider
    yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)
    path = tmp_path / "SPX.csv"
    original = path.read_bytes()
    history.return_value = _history(["2024-01-03"])
    monkeypatch.setattr(Path, "replace", MagicMock(side_effect=OSError("write failed")))
    with pytest.raises(ValueError, match="Could not write Yahoo daily cache"):
        yahoo.fetch_daily_history("SPX", {date(2024, 1, 3)}, tmp_path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_merge_preserves_a_concurrent_callers_additions(provider, tmp_path):
    _, history = provider
    yahoo.fetch_daily_history("SPX", {date(2024, 1, 2)}, tmp_path)

    def concurrent_download(**_kwargs):
        latest = pl.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 4)],
                "open": [25.0, 30.0],
                "high": [27.5, 33.0],
                "low": [22.5, 27.0],
                "close": [25.0, 30.0],
            },
            schema=yahoo._BAR_SCHEMA,
        )
        yahoo._write_cache(tmp_path / "SPX.csv", latest, "SPX", "^GSPC")
        return _history(["2024-01-03"])

    history.side_effect = concurrent_download
    result = yahoo.fetch_daily_history(
        "SPX", {date(2024, 1, 2), date(2024, 1, 3)}, tmp_path
    )
    assert result["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert pl.read_csv(tmp_path / "SPX.csv")["date"].to_list() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
