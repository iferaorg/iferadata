"""Cache Yahoo daily bars in the nominal price units used by ThetaData."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, datetime, timedelta
from importlib import import_module
from pathlib import Path
import tempfile
from typing import cast
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl

from ifera.settings import settings

_PRICE_BASIS = "nominal_v1"
_YAHOO_SYMBOLS = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "DJI": "^DJI",
    "RUT": "^RUT",
    "VIX": "^VIX",
}
_BAR_SCHEMA = pl.Schema(
    {
        "date": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
    }
)
_CACHE_COLUMNS = [*_BAR_SCHEMA, "symbol", "yahoo_symbol", "price_basis"]


def _market_today() -> date:
    """Use the provider's New York session date when excluding unfinished bars."""
    return datetime.now(ZoneInfo("America/New_York")).date()


def _validate_bars(frame: pl.DataFrame, label: str) -> pl.DataFrame:
    """Require unique, finite, positive and internally consistent daily OHLC."""
    try:
        bars = frame.select(*_BAR_SCHEMA).cast(_BAR_SCHEMA, strict=False)
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"Invalid {label}: missing daily OHLC columns") from exc
    invalid_price = pl.any_horizontal(
        *[
            pl.col(column).is_null()
            | ~pl.col(column).is_finite()
            | (pl.col(column) <= 0)
            for column in ("open", "high", "low", "close")
        ]
    )
    invalid_range = (pl.col("high") < pl.max_horizontal("open", "low", "close")) | (
        pl.col("low") > pl.min_horizontal("open", "high", "close")
    )
    if (
        bars["date"].null_count()
        or bars["date"].n_unique() != bars.height
        or bars.select((invalid_price | invalid_range).any()).item()
    ):
        raise ValueError(
            f"Invalid {label}: expected unique dates and valid positive OHLC"
        )
    return bars.sort("date")


def _read_cache(
    path: Path, symbol: str, yahoo_symbol: str, today: date
) -> pl.DataFrame:
    """Read only versioned nominal bars for the expected underlying and provider."""
    if not path.exists():
        return pl.DataFrame(schema=_BAR_SCHEMA)
    try:
        cached = pl.read_csv(path, schema_overrides={"date": pl.Date})
        if set(cached.columns) != set(_CACHE_COLUMNS):
            raise ValueError("unexpected cache columns or price-basis version")
        incorrect = cached.select(
            pl.any_horizontal(
                pl.col("symbol").is_null() | (pl.col("symbol") != symbol),
                pl.col("yahoo_symbol").is_null()
                | (pl.col("yahoo_symbol") != yahoo_symbol),
                pl.col("price_basis").is_null()
                | (pl.col("price_basis") != _PRICE_BASIS),
                pl.col("date").is_null() | (pl.col("date") >= today),
            ).any()
        ).item()
        if incorrect:
            raise ValueError("incorrect provenance, price basis, or unfinished session")
        return _validate_bars(cached, "Yahoo daily cache")
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Invalid Yahoo daily cache {path}: {exc}") from exc


def _nominal_bars(
    history: pd.DataFrame, requested: set[date], today: date
) -> pl.DataFrame:
    """Undo later stock splits while leaving cash dividends unadjusted.

    Yahoo OHLC is split-adjusted even with auto_adjust=False. Split events after
    the requested range therefore also participate. A split does not multiply
    the bar on its own effective date. Adj Close is deliberately never used.
    """
    needed = {"Open", "High", "Low", "Close", "Stock Splits"}
    if history.empty or not needed <= set(history.columns):
        raise ValueError(
            "Yahoo history requires OHLC and Stock Splits to verify price basis"
        )
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Yahoo history requires dated daily sessions")
    history = history.sort_index()
    index = cast(pd.DatetimeIndex, history.index)
    history = history.loc[np.asarray(index.date) <= today]
    dates = list(cast(pd.DatetimeIndex, history.index).date)
    if len(set(dates)) != len(dates):
        raise ValueError("Yahoo history contains duplicate daily sessions")
    splits = np.asarray(
        pd.to_numeric(cast(pd.Series, history["Stock Splits"]), errors="coerce"),
        dtype=np.float64,
    )
    if not np.isfinite(splits).all() or (splits < 0).any():
        raise ValueError("Yahoo history contains invalid stock split ratios")
    ratios = np.where(splits == 0, 1.0, splits)
    cumulative = np.cumprod(ratios[::-1])[::-1]
    factors = np.concatenate((cumulative[1:], [1.0])) if dates else cumulative
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("Yahoo history cannot be converted to nominal price units")
    positions = [index for index, day in enumerate(dates) if day in requested]
    values: dict[str, list[date] | list[float]] = {
        "date": [dates[index] for index in positions]
    }
    for source, target in zip(("Open", "High", "Low", "Close"), tuple(_BAR_SCHEMA)[1:]):
        prices = np.asarray(
            pd.to_numeric(cast(pd.Series, history[source]), errors="coerce"),
            dtype=np.float64,
        )
        values[target] = (prices[positions] * factors[positions]).tolist()
    return _validate_bars(
        pl.DataFrame(values, schema=_BAR_SCHEMA, strict=False), "Yahoo daily history"
    )


def _download_bars(
    symbol: str, yahoo_symbol: str, requested: set[date], today: date
) -> pl.DataFrame:
    """Fetch through the current date so subsequent split events are included."""
    try:
        ticker_factory = getattr(import_module("yfinance"), "Ticker")
        history = ticker_factory(yahoo_symbol).history(
            start=min(requested).isoformat(),
            end=(today + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            back_adjust=False,
            actions=True,
            repair=False,
            rounding=False,
            keepna=True,
            prepost=False,
            timeout=30,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not download Yahoo daily history for {symbol}: {exc}"
        ) from exc
    if not isinstance(history, pd.DataFrame):
        raise ValueError(f"Yahoo daily history for {symbol} is not a price table")
    bars = _nominal_bars(history, requested, today)
    missing = requested - set(bars["date"].to_list())
    if missing:
        raise ValueError(
            f"Yahoo daily history for {symbol} is missing sessions: "
            + ", ".join(day.isoformat() for day in sorted(missing))
        )
    return bars


def _write_cache(
    path: Path, bars: pl.DataFrame, symbol: str, yahoo_symbol: str
) -> None:
    """Replace a complete cache atomically without exposing partial CSV writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = bars.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(yahoo_symbol).alias("yahoo_symbol"),
        pl.lit(_PRICE_BASIS).alias("price_basis"),
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stamped.write_csv(stream)
        temporary.replace(path)
    except OSError as exc:
        raise ValueError(f"Could not write Yahoo daily cache {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_daily_history(
    symbol: str,
    required_dates: Collection[date],
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Return requested completed daily bars, downloading and caching gaps only.

    CSV bars use nominal/as-traded units, compatible with ThetaData quote prices:
    later splits are undone and dividend adjustment is disabled. A warm cache
    makes no Yahoo requests. Cached nominal bars are never re-adjusted.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a nonempty underlying symbol")
    symbol = symbol.strip().upper()
    requested = set(required_dates)
    if any(not isinstance(day, date) or isinstance(day, datetime) for day in requested):
        raise TypeError("required_dates must contain dates")
    if not requested:
        return pl.DataFrame(schema=_BAR_SCHEMA)
    today = _market_today()
    if any(day >= today for day in requested):
        raise ValueError(
            "Yahoo daily fallback requires completed sessions before today"
        )
    yahoo_symbol = _YAHOO_SYMBOLS.get(symbol, symbol)
    folder = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(settings.DATA_FOLDER) / "cache" / "strategy_optimization" / "daily"
    )
    path = folder / f"{quote(symbol, safe='')}.csv"
    cached = _read_cache(path, symbol, yahoo_symbol, today)
    missing = requested - set(cached["date"].to_list())
    if missing:
        downloaded = _download_bars(symbol, yahoo_symbol, missing, today)
        # Re-read after downloading to preserve additions made by another caller.
        latest = _read_cache(path, symbol, yahoo_symbol, today)
        cached = (
            pl.concat([latest, downloaded])
            .unique(subset=["date"], keep="first", maintain_order=True)
            .sort("date")
        )
        _write_cache(path, cached, symbol, yahoo_symbol)
    return cached.filter(pl.col("date").is_in(sorted(requested))).sort("date")
