"""Read single-strategy CSV ledgers and minute data through Oracle synonyms."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from typing import cast

import oracledb
import polars as pl
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from ifera.allocation._portfolio_blocking import (
    _datetime_expression,
    _matching_column,
    parse_strategy_trade_legs,
)
from ifera.allocation._portfolio_returns import strategy_return_expression
from ifera.allocation.portfolio_allocation import _nyse_market_dates, _portfolio_path
from ifera.settings import settings

from ._daily_history import complete_daily_history
from ._oracle_queries import OPTION_SQL, UNDERLYING_SQL


@dataclass(frozen=True)
class StrategyData:
    """Original ledger, normalized accounting and option legs, and market calendar.

    ``opening_price`` is a signed debit in option points; credits are negative.
    ``premium``, ``risk`` and ``profit`` are account totals. ``ror`` and
    ``reward_risk`` are percentages. CSV Max Profit/Max Loss are per-combo values,
    while Premium/Risk/P&L include Quantity. Timestamps use naive New York market time.
    """

    raw: pl.DataFrame
    trades: pl.DataFrame
    legs: pl.DataFrame
    dates: tuple[date, ...]
    path: Path


@dataclass(frozen=True)
class MarketData:
    """Minute prices and optional daily bars for historical coverage gaps.

    Supplemental daily history has symbol/date/open/high/low/close columns. It
    replaces absent or incomplete historical sessions, never intraday marks.
    """

    underlying: pl.DataFrame
    options: pl.DataFrame
    daily_history: pl.DataFrame | None = None


def _required(raw: pl.DataFrame, requested: str, path: Path) -> str:
    """Resolve a required CSV field with the allocation package's matching rule."""
    name = _matching_column(raw.columns, requested)
    if name is None:
        raise ValueError(f"{path.name} is missing the {requested!r} column")
    return name


def _numeric(raw: pl.DataFrame, requested: str, path: Path) -> pl.Expr:
    """Parse account values, rejecting null and nonfinite numbers."""
    value = pl.col(_required(raw, requested, path)).cast(pl.Float64, strict=False)
    if raw.select((value.is_null() | ~value.is_finite()).any()).item():
        raise ValueError(f"{path.name} requires finite {requested} values")
    return value


def _signed_opening_price(trades: pl.DataFrame, legs: pl.DataFrame) -> pl.Series:
    """Infer debit/credit from expiration payoffs and the ledger's original risk."""
    prices: list[float] = []
    groups = legs.partition_by("trade_index", as_dict=True)
    for row in trades.iter_rows(named=True):
        frame = groups[(row["trade_index"],)]
        contracts = list(frame.select("direction", "right", "strike").iter_rows())
        magnitude = abs(row["opening_price"])
        directions = set(frame["direction"].to_list())
        if len(directions) == 1:
            prices.append(magnitude if "long" in directions else -magnitude)
            continue
        knots = {0.0, *(float(strike) for _, _, strike in contracts)}
        minimum_payoff = min(
            sum(
                (1.0 if direction == "long" else -1.0)
                * max(0.0, spot - strike if right == "call" else strike - spot)
                for direction, right, strike in contracts
            )
            for spot in knots
        )
        scale = 100.0 * row["quantity"]
        debit_error = abs((magnitude - minimum_payoff) * scale - row["risk"])
        credit_error = abs((-magnitude - minimum_payoff) * scale - row["risk"])
        prices.append(magnitude if debit_error <= credit_error else -magnitude)
    return pl.Series("opening_price", prices)


def load_strategy(portfolio_name: str, strategy_name: str) -> StrategyData:
    """Load one 0DTE strategy, preserving the original CSV field order and values."""
    if (
        not isinstance(strategy_name, str)
        or not strategy_name
        or strategy_name in {".", ".."}
        or "/" in strategy_name
        or "\\" in strategy_name
    ):
        raise ValueError("strategy_name must be a single nonempty filename stem")
    filename = (
        strategy_name if strategy_name.endswith(".csv") else f"{strategy_name}.csv"
    )
    return load_strategy_csv(_portfolio_path(portfolio_name) / filename)


def load_strategy_csv(path: Path | str) -> StrategyData:
    """Read an existing 0DTE strategy CSV without changing its path or contents.

    Arbitrary paths use the same ledger validation, accounting, leg parsing, and
    market calendar as a strategy loaded from a named portfolio.
    """
    if isinstance(path, str) and not path:
        raise ValueError("path must be a nonempty CSV file path")
    path = Path(path)
    raw = pl.read_csv(path, infer_schema_length=None)
    if raw.is_empty():
        raise ValueError(f"Strategy CSV is empty: {path.name}")
    opened = _required(raw, "Opened", path)
    legs = parse_strategy_trade_legs(raw, path, 0, opened)
    if legs is None:
        raise ValueError(f"{path.name} requires complete Symbol/Exp/Legs metadata")
    identities = legs.unique("trade_index", maintain_order=True).select(
        "trade_index", "symbol", "expiration"
    )
    trades = (
        raw.with_row_index("trade_index")
        .select(
            pl.col("trade_index").cast(pl.Int64),
            _datetime_expression(opened).alias("opened"),
            _datetime_expression(_required(raw, "Closed", path)).alias("closed"),
            _numeric(raw, "Quantity", path).alias("quantity"),
            _numeric(raw, "Risk", path).alias("risk"),
            _numeric(raw, "Premium", path).abs().alias("premium"),
            _numeric(raw, "P/L", path).alias("profit"),
            (strategy_return_expression(raw, path) * 100.0).alias("ror"),
            _numeric(raw, "Open Price", path).alias("opening_price"),
        )
        .join(identities, on="trade_index", how="left", validate="1:1")
    )
    reward_column = _matching_column(raw.columns, "Reward/Risk")
    reward = (
        _numeric(raw, "Reward/Risk", path)
        if reward_column is not None
        else _numeric(raw, "Max Profit", path)
        * _numeric(raw, "Quantity", path)
        / _numeric(raw, "Risk", path)
        * 100.0
    )
    trades = trades.with_columns(raw.select(reward.alias("reward_risk"))["reward_risk"])
    if trades.filter((pl.col("quantity") <= 0) | (pl.col("risk") <= 0)).height:
        raise ValueError(f"{path.name} requires positive Quantity and Risk")
    if trades.filter(pl.col("quantity") != pl.col("quantity").floor()).height:
        raise ValueError(f"{path.name} requires whole option contract quantities")
    if trades.filter(
        (pl.col("opened").dt.date() != pl.col("closed").dt.date())
        | (pl.col("opened").dt.date() != pl.col("expiration"))
    ).height:
        raise ValueError("Only same-day 0DTE trades are supported")
    trade_dates = trades["opened"].dt.date()
    if trade_dates.n_unique() != trades.height:
        raise ValueError("Only one trade per market day is supported")
    dates = tuple(
        _nyse_market_dates(min(trade_dates), max(trade_dates))["date"].to_list()
    )
    if set(trade_dates.to_list()) - set(dates):
        raise ValueError("Strategy contains a non-market trade date")
    trades = trades.with_columns(_signed_opening_price(trades, legs)).sort(
        "trade_index"
    )
    return StrategyData(raw, trades, legs, dates, path)


_FETCH_ROWS = 8192
_UNDERLYING_SCHEMA = pl.Schema(
    {"symbol": pl.String, "datetime": pl.Datetime("us"), "price": pl.Float64}
)
_OPTION_SCHEMA = pl.Schema(
    {
        "trade_index": pl.Int64,
        "datetime": pl.Datetime("us"),
        "direction": pl.String,
        "right": pl.String,
        "strike": pl.Float64,
        "bid": pl.Float64,
        "ask": pl.Float64,
        "mid": pl.Float64,
        "delta": pl.Float64,
    }
)


def validate_lookback_days(value: int) -> None:
    """Validate the calendar lookback shared by the loader and CSV facade."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("lookback_days must be a positive integer")


def _requests(data: StrategyData, lookback_days: int) -> tuple[str, str]:
    """Encode all history windows and exact dated contracts as bound JSON."""
    start = datetime.combine(data.dates[0] - timedelta(days=lookback_days), time())
    end = datetime.combine(data.dates[-1] + timedelta(days=1), time())
    history = [
        {
            "symbol": symbol,
            "start": start.isoformat(timespec="microseconds"),
            "end": end.isoformat(timespec="microseconds"),
        }
        for symbol in data.trades["symbol"].unique(maintain_order=True).to_list()
    ]
    contracts = data.legs.select("trade_index", "direction", "right", "strike").join(
        data.trades.select("trade_index", "symbol", "expiration", "opened", "closed"),
        on="trade_index",
        how="inner",
        validate="m:1",
    )
    options = []
    for leg in contracts.iter_rows(named=True):
        opened = leg["opened"].replace(second=0, microsecond=0)
        # Even a zero-duration trade requires its entry snapshot.
        closed = max(leg["closed"], opened + timedelta(microseconds=1))
        options.append(
            {
                "trade_index": leg["trade_index"],
                "symbol": leg["symbol"],
                "expiration": leg["expiration"].isoformat(),
                "start": opened.isoformat(timespec="microseconds"),
                "end": closed.isoformat(timespec="microseconds"),
                "direction": leg["direction"],
                "right": leg["right"][0].upper(),
                "strike": leg["strike"],
            }
        )
    return (
        json.dumps(history, allow_nan=False, separators=(",", ":")),
        json.dumps(options, allow_nan=False, separators=(",", ":")),
    )


def _fetch_frame(
    connection: oracledb.Connection,
    cursor: oracledb.Cursor,
    query: str,
    requests: str,
    schema: pl.Schema,
) -> pl.DataFrame:
    """Execute one fixed query and fetch numeric rows in large network batches.

    An explicit temporary CLOB supports request documents larger than 32 KB.
    Merely binding a long Python string as CLOB can raise ORA-01460. The driver
    releases the temporary LOB when its Python/bind references are released;
    LOB.close() only applies after LOB.open() and must not be used here.
    """
    request_lob = connection.createlob(oracledb.DB_TYPE_CLOB)
    request_lob.write(requests)
    cursor.execute(query, {"requests": request_lob})
    chunks = []
    while rows := cursor.fetchmany(_FETCH_ROWS):
        chunks.append(pl.DataFrame(rows, schema=schema, orient="row", strict=False))
    return pl.concat(chunks) if chunks else pl.DataFrame(schema=schema)


class OracleMarketDataSource:
    """Read the known Oracle schema with two static, batched SQL statements.

    History uses one indexed underlying-price observation per minute; option
    quotes use exact dated contracts and their entry-to-close minute windows.
    OPT_SYMBOL joins translate underlying symbols inside each query. No catalog
    queries, reflection, source selection, or per-trade database calls occur.

    Supply an existing engine to retain its lifetime, or configure IFERA_DB_USER,
    IFERA_DB_PASSWORD and IFERA_DSN. Only an internally created engine is disposed
    by close() or when this source leaves its context manager.
    """

    def __init__(
        self, engine: Engine | None = None, *, fill_daily_history: bool = True
    ) -> None:
        self._owns_engine = engine is None
        self._fill_daily_history = fill_daily_history
        if engine is None:
            if not all(
                (settings.IFERA_DB_USER, settings.IFERA_DB_PASSWORD, settings.IFERA_DSN)
            ):
                raise ValueError(
                    "Configure IFERA_DB_USER, IFERA_DB_PASSWORD and IFERA_DSN"
                )
            engine = create_engine(
                URL.create(
                    "oracle+oracledb",
                    username=settings.IFERA_DB_USER,
                    password=settings.IFERA_DB_PASSWORD,
                    query={"dsn": settings.IFERA_DSN},
                ),
                pool_pre_ping=True,
            )
        self.engine = engine

    def __enter__(self) -> OracleMarketDataSource:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    def close(self) -> None:
        """Dispose only an engine created and owned by this source."""
        if self._owns_engine:
            self.engine.dispose()

    def load(self, data: StrategyData, lookback_days: int = 400) -> MarketData:
        """Load all requested symbols and trades using two fixed Oracle queries."""
        validate_lookback_days(lookback_days)
        history_requests, option_requests = _requests(data, lookback_days)
        with self.engine.connect() as connection:
            native = cast(oracledb.Connection, connection.connection.dbapi_connection)
            cursor = native.cursor()
            try:
                cursor.arraysize = _FETCH_ROWS
                cursor.prefetchrows = _FETCH_ROWS + 1
                cursor.setinputsizes(requests=oracledb.DB_TYPE_CLOB)
                underlying = _fetch_frame(
                    native, cursor, UNDERLYING_SQL, history_requests, _UNDERLYING_SCHEMA
                ).filter(pl.col("price").is_not_null())
                underlying = underlying.unique(
                    subset=["symbol", "datetime"], keep="first", maintain_order=True
                ).sort("symbol", "datetime")
                expected = set(data.trades["symbol"].to_list())
                missing = expected - set(underlying["symbol"].unique().to_list())
                if missing:
                    raise ValueError(
                        "No underlying minute history or option root for "
                        + ", ".join(sorted(missing))
                    )
                options = (
                    _fetch_frame(
                        native, cursor, OPTION_SQL, option_requests, _OPTION_SCHEMA
                    )
                    .with_columns(
                        pl.when(pl.col("right").str.to_lowercase().str.starts_with("c"))
                        .then(pl.lit("call"))
                        .otherwise(pl.lit("put"))
                        .alias("right")
                    )
                    .sort("trade_index", "datetime", "right", "strike", "direction")
                )
                absent = set(data.trades["trade_index"].to_list()) - set(
                    options["trade_index"].unique().to_list()
                )
                if absent:
                    descriptions = "; ".join(
                        f"trade {trade['trade_index']} "
                        f"(date={trade['opened'].date().isoformat()}, "
                        f"symbol={trade['symbol']}, "
                        f"opened={trade['opened'].isoformat(sep=' ')}, "
                        f"closed={trade['closed'].isoformat(sep=' ')}, "
                        f"expiration={trade['expiration'].isoformat()})"
                        for trade in data.trades.filter(
                            pl.col("trade_index").is_in(sorted(absent))
                        )
                        .sort("trade_index")
                        .iter_rows(named=True)
                    )
                    raise ValueError(
                        f"No option quotes for {descriptions}; source={data.path.name}"
                    )
            finally:
                cursor.close()
        daily_history = (
            complete_daily_history(data, underlying, lookback_days)
            if self._fill_daily_history
            else None
        )
        return MarketData(underlying, options, daily_history)
