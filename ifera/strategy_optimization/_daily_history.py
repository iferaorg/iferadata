"""Complete historical daily OHLC without substituting intraday observations."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from holidays.constants import HALF_DAY
from holidays.financial.ny_stock_exchange import NYSE
import polars as pl

from ifera.allocation.portfolio_allocation import _nyse_market_dates

from ._yahoo_history import _BAR_SCHEMA, fetch_daily_history

if TYPE_CHECKING:
    from ._data import StrategyData

_DAILY_SCHEMA = pl.Schema({"symbol": pl.String, **_BAR_SCHEMA})


def _required_sessions(start: date, end: date) -> pl.DataFrame:
    """Describe full and early-close sessions in an inclusive historical range."""
    sessions = _nyse_market_dates(start, end)
    half_days = NYSE(years=range(start.year, end.year + 1), categories=(HALF_DAY,))
    return sessions.with_columns(
        pl.Series(
            "expected_minutes",
            [210 if session in half_days else 390 for session in sessions["date"]],
            dtype=pl.Int32,
        )
    )


def _complete_dates(
    underlying: pl.DataFrame, symbol: str, sessions: pl.DataFrame
) -> set[date]:
    """Require every valid whole-minute snapshot from 09:31 through the close."""
    if underlying.is_empty() or sessions.is_empty():
        return set()
    price = pl.col("price").cast(pl.Float64, strict=False)
    frame = (
        underlying.filter(pl.col("symbol") == symbol)
        .with_columns(
            pl.col("datetime").dt.date().alias("date"),
            (
                pl.col("datetime").dt.hour().cast(pl.Int32) * 60
                + pl.col("datetime").dt.minute().cast(pl.Int32)
            ).alias("minute"),
            (
                (pl.col("datetime").dt.second() == 0)
                & (pl.col("datetime").dt.microsecond() == 0)
            ).alias("aligned"),
            (price.is_not_null() & price.is_finite() & (price > 0)).alias("valid"),
        )
        .join(sessions, on="date", how="inner", validate="m:1")
        .filter(
            pl.col("aligned")
            & (pl.col("minute") >= 571)
            & (pl.col("minute") <= 570 + pl.col("expected_minutes"))
        )
        .group_by("date")
        .agg(
            pl.col("minute").n_unique().alias("observed"),
            pl.col("expected_minutes").first(),
            pl.col("valid").all().alias("valid"),
        )
        .filter((pl.col("observed") == pl.col("expected_minutes")) & pl.col("valid"))
    )
    return set(frame["date"].to_list())


def complete_daily_history(
    data: StrategyData, underlying: pl.DataFrame, lookback_days: int
) -> pl.DataFrame:
    """Download only missing/incomplete prior-session OHLC, grouped by symbol.

    The latest trade date for each symbol is excluded: its finalized daily bar
    cannot be used at that day's entry or exit. Earlier trade dates may need a
    daily replacement for later indicators, but their minute prices remain the
    caller's responsibility. Complete Oracle sessions always take precedence.
    """
    start = data.dates[0] - timedelta(days=lookback_days)
    replacements = []
    for symbol in data.trades["symbol"].unique(maintain_order=True).to_list():
        last_trade = (
            data.trades.filter(pl.col("symbol") == symbol)["opened"].dt.date().max()
        )
        if not isinstance(last_trade, date):
            raise ValueError(f"No trade date for {symbol}")
        sessions = _required_sessions(start, last_trade - timedelta(days=1))
        required = set(sessions["date"].to_list())
        missing = tuple(
            sorted(required - _complete_dates(underlying, symbol, sessions))
        )
        if not missing:
            continue
        history = fetch_daily_history(symbol, missing).filter(
            pl.col("date").is_in(missing)
        )
        available = set(history["date"].to_list())
        absent = set(missing) - available
        if absent:
            raise ValueError(
                f"Missing daily OHLC for {symbol}: "
                + ", ".join(session.isoformat() for session in sorted(absent))
            )
        if history["date"].n_unique() != history.height:
            raise ValueError(f"Duplicate daily OHLC dates for {symbol}")
        replacements.append(
            history.with_columns(pl.lit(symbol).alias("symbol")).select(
                _DAILY_SCHEMA.names()
            )
        )
    if not replacements:
        return pl.DataFrame(schema=_DAILY_SCHEMA)
    return (
        pl.concat(replacements, how="vertical_relaxed")
        .cast(_DAILY_SCHEMA)
        .sort("symbol", "date")
    )
