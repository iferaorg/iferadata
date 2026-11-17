"""Prepare minute observations and account for complete option combinations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from numbers import Real

import polars as pl
import torch

from ._data import MarketData, StrategyData
from ._evaluation import OptimizationMode, StrategyTensors
from ._events import event_flags, load_event_dates
from ._features import FeatureSpec, build_features

_LEG_NAMES = ("short_call", "short_put", "long_call", "long_put")


@dataclass(frozen=True)
class PreparedStrategy:
    """Tensor observations and corresponding CPU records for CSV output."""

    data: StrategyData
    features: tuple[FeatureSpec, ...]
    tensors: StrategyTensors
    exits: pl.DataFrame


def validate_slippage(value: float) -> float:
    """Require adverse slippage expressed in whole-combination price points."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("exit_slippage must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("exit_slippage must be finite and nonnegative")
    return float(value)


def _underlying_history(
    market: MarketData, symbol: str, trade_dates: frozenset[date]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Derive daily bars and causal intraday OHLC from regular-session prices."""
    minute = (
        market.underlying.filter(
            (pl.col("symbol") == symbol)
            & pl.col("datetime").dt.time().is_between(time(9, 30), time(16))
        )
        .sort("datetime")
        .with_columns(pl.col("datetime").dt.date().alias("date"))
    )
    supplemental = market.daily_history
    if supplemental is not None and not supplemental.is_empty():
        columns = ["date", "open", "high", "low", "close"]
        if "volume" in supplemental.columns:
            columns.append("volume")
        supplemental = supplemental.filter(pl.col("symbol") == symbol).select(columns)
        # Non-trading sessions only need the replacement daily bar. Trade dates
        # retain their actual minute observations and strict validation below.
        replaced_dates = set(supplemental["date"].to_list()) - trade_dates
        minute = minute.filter(~pl.col("date").is_in(sorted(replaced_dates)))
    if minute.is_empty() or minute["datetime"].n_unique() != minute.height:
        raise ValueError(f"Missing or duplicate minute underlying data for {symbol}")
    if minute.select(
        (
            pl.col("price").is_null()
            | ~pl.col("price").is_finite()
            | (pl.col("price") <= 0)
        ).any()
    ).item():
        raise ValueError(f"Invalid underlying prices for {symbol}")
    has_minute_volume = "volume" in minute.columns
    has_volume = has_minute_volume or (
        supplemental is not None and "volume" in supplemental.columns
    )
    aggregations = [
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
    ]
    if has_minute_volume:
        volume = pl.col("volume").cast(pl.Float64, strict=False)
        minute = minute.with_columns(
            pl.when(volume.is_finite() & (volume >= 0.0))
            .then(volume)
            .otherwise(float("nan"))
            .alias("volume")
        )
        # A missing minute invalidates subsequent totals; summing only known
        # values would silently invent a complete session's trading volume.
        aggregations.append(pl.col("volume").sum().alias("volume"))
    elif has_volume:
        aggregations.append(pl.lit(float("nan")).alias("volume"))
    daily = minute.group_by("date", maintain_order=True).agg(aggregations)
    if supplemental is not None and not supplemental.is_empty():
        if supplemental["date"].n_unique() != supplemental.height:
            raise ValueError(f"Duplicate supplemental daily history for {symbol}")
        if has_volume:
            if "volume" not in supplemental.columns:
                supplemental = supplemental.with_columns(
                    pl.lit(float("nan")).alias("volume")
                )
            else:
                volume = pl.col("volume").cast(pl.Float64, strict=False)
                supplemental = supplemental.with_columns(
                    pl.when(volume.is_finite() & (volume >= 0.0))
                    .then(volume)
                    .otherwise(float("nan"))
                    .alias("volume")
                )
        daily = (
            pl.concat((supplemental, daily), how="vertical_relaxed")
            .unique("date", keep="first", maintain_order=True)
            .sort("date")
        )
    previous = daily.select("date", pl.col("close").shift(1).alias("prev_close"))
    minute = minute.with_columns(
        pl.col("price").first().over("date").alias("day_open"),
        pl.col("price").cum_max().over("date").alias("day_high"),
        pl.col("price").cum_min().over("date").alias("day_low"),
    ).join(previous, on="date", how="left", validate="m:1")
    if has_volume:
        minute = minute.with_columns(
            (
                pl.col("volume").cum_sum().over("date")
                if has_minute_volume
                else pl.lit(float("nan"))
            ).alias("day_volume")
        )
    return minute, daily


def _combination_quotes(quotes: pl.DataFrame, leg_count: int) -> pl.DataFrame:
    """Aggregate a simultaneous complete set of legs into signed position marks."""
    invalid = quotes.select(
        pl.any_horizontal(
            *[
                pl.col(name).is_null() | ~pl.col(name).is_finite() | (pl.col(name) < 0)
                for name in ("bid", "ask", "mid")
            ],
            pl.col("bid") > pl.col("ask"),
        ).any()
    ).item()
    if invalid:
        raise ValueError("Option quotes require finite nonnegative bid/ask/mid prices")
    keys = ["datetime", "direction", "right", "strike"]
    if quotes.unique(subset=keys).height != quotes.height:
        raise ValueError("Duplicate option leg quotes at a minute")
    long = pl.col("direction") == "long"
    sign = pl.when(long).then(1.0).otherwise(-1.0)
    frame = quotes.group_by("datetime").agg(
        pl.len().alias("leg_count"),
        (sign * pl.col("mid")).sum().alias("combo_mid"),
        pl.when(long)
        .then(pl.col("bid"))
        .otherwise(-pl.col("ask"))
        .sum()
        .alias("combo_bid"),
        pl.when(long)
        .then(pl.col("ask"))
        .otherwise(-pl.col("bid"))
        .sum()
        .alias("combo_ask"),
        *[
            pl.when(
                (pl.col("direction") == leg.split("_", maxsplit=1)[0])
                & (pl.col("right") == leg.split("_")[1])
            )
            .then(pl.col("delta").fill_null(float("nan")))
            .otherwise(0.0)
            .sum()
            .alias(f"{leg}_delta")
            for leg in _LEG_NAMES
        ],
    )
    return (
        frame.filter(pl.col("leg_count") == leg_count)
        .drop("leg_count")
        .with_columns(
            (pl.col("combo_ask") - pl.col("combo_bid")).alias("position_bid_ask_spread")
        )
    )


def _timestamps(opened: datetime, closed: datetime, exits: bool) -> list[datetime]:
    """Include entry's minute and each subsequent whole minute before close."""
    first = opened.replace(second=0, microsecond=0)
    if not exits:
        return [first]
    count = max(1, math.ceil((closed - first).total_seconds() / 60))
    return [first + timedelta(minutes=index) for index in range(count)]


def _position_moneyness(legs: pl.DataFrame) -> pl.Expr:
    """Return the greatest leg moneyness in percent of its option strike.

    Calls use their lowest strike and puts their highest: each is the most
    in-the-money option of that right at any positive underlying price. Long
    and short directions do not change an option's intrinsic moneyness.
    """
    call_strike, put_strike = legs.select(
        pl.col("strike").filter(pl.col("right") == "call").min().alias("call"),
        pl.col("strike").filter(pl.col("right") == "put").max().alias("put"),
    ).row(0)
    expressions = []
    if call_strike is not None:
        expressions.append(100.0 * (pl.col("price") - call_strike) / call_strike)
    if put_strike is not None:
        expressions.append(100.0 * (put_strike - pl.col("price")) / put_strike)
    return pl.max_horizontal(expressions) if expressions else pl.lit(float("nan"))


def _trade_observations(
    trade: dict,
    legs: pl.DataFrame,
    market: MarketData,
    minute: pl.DataFrame,
    mode: OptimizationMode,
    slippage: float,
    needs_quotes: bool,
) -> pl.DataFrame:
    """Join exact-minute quotes; never fill a quote from a later observation."""
    stamps = _timestamps(
        trade["opened"], trade["closed"], mode != OptimizationMode.ENTRY_FILTER
    )
    frame = pl.DataFrame({"datetime": stamps}).join(
        minute, on="datetime", how="left", validate="1:1"
    )
    if frame["price"].null_count():
        raise ValueError(f"Missing underlying minute for trade {trade['trade_index']}")
    quotes = market.options.filter(pl.col("trade_index") == trade["trade_index"])
    if mode != OptimizationMode.ENTRY_FILTER or needs_quotes:
        frame = frame.join(
            _combination_quotes(quotes, legs.height),
            on="datetime",
            how="left",
            validate="1:1",
        )
        if mode != OptimizationMode.ENTRY_FILTER and frame["combo_mid"].null_count():
            raise ValueError(
                f"Missing complete option minute for trade {trade['trade_index']}"
            )
    else:
        frame = frame.with_columns(
            *[
                pl.lit(float("nan")).alias(name)
                for name in (
                    "combo_mid",
                    "combo_bid",
                    "combo_ask",
                    "position_bid_ask_spread",
                )
            ],
            *[pl.lit(float("nan")).alias(f"{leg}_delta") for leg in _LEG_NAMES],
        )
    value = pl.col("combo_mid") - slippage
    profit = (value - trade["opening_price"]) * 100.0 * trade["quantity"]
    premium = abs(trade["premium"])
    frame = frame.with_columns(
        pl.lit(trade["trade_index"]).alias("trade_index"),
        pl.col("price").alias("underlying_price"),
        pl.lit(premium).alias("premium"),
        pl.lit(trade["risk"]).alias("risk"),
        pl.lit(trade["reward_risk"]).alias("reward_risk"),
        _position_moneyness(legs).alias("position_itm_pct"),
        ((pl.col("datetime") - trade["opened"]).dt.total_seconds() / 60).alias(
            "minutes_open"
        ),
        value.alias("liquidation_value"),
        profit.alias("profit"),
        (profit / premium * 100.0 if premium else pl.lit(float("nan"))).alias(
            "return_pct"
        ),
        (profit / trade["risk"] * 100.0).alias("ror_pct"),
    )
    return frame.with_columns(
        *[
            expression.alias(f"{column}_{component}")
            for column in ("return_pct", "ror_pct")
            for component, expression in (
                ("high", pl.col(column).cum_max()),
                ("low", pl.col(column).cum_min()),
            )
        ]
    )


def _ignored_features(
    legs: pl.DataFrame, features: tuple[FeatureSpec, ...]
) -> torch.Tensor:
    """Ignore only leg-delta predicates whose specified leg is absent."""
    present = {
        f"{row['direction']}_{row['right']}" for row in legs.iter_rows(named=True)
    }
    return torch.tensor(
        [
            feature.template == "leg_delta" and feature.parameters["leg"] not in present
            for feature in features
        ],
        dtype=torch.bool,
    )


def prepare_strategy(
    data: StrategyData,
    market: MarketData,
    features: tuple[FeatureSpec, ...],
    mode: OptimizationMode | str,
    exit_slippage: float = 0.0,
) -> PreparedStrategy:
    """Precompute every feature and attainable exit before running evolution.

    Slippage subtracts option price points from the entire signed combination's
    liquidation value once, then multiplies by 100 and position quantity. Original
    closes retain the original accounting and never incur new exit slippage.
    """
    selected_mode = OptimizationMode(mode)
    slippage = validate_slippage(exit_slippage)
    entries, exits, entry_ignored, exit_ignored, records = [], [], [], [], []
    event_features = [
        feature for feature in features if feature.template == "event_day"
    ]
    events = (
        load_event_dates(
            data.dates,
            [feature.parameters["event_type"] for feature in event_features],
            offsets=[feature.parameters["days"] for feature in event_features],
        )
        if event_features
        else {}
    )
    event_columns = pl.DataFrame(
        {
            "date": list(data.dates),
            **{
                f"__event_{feature.name}": event_flags(
                    data.dates,
                    feature.parameters["event_type"],
                    feature.parameters["days"],
                    events=events,
                )
                for feature in event_features
            },
        }
    )
    history = {
        symbol: _underlying_history(
            market,
            symbol,
            frozenset(
                data.trades.filter(pl.col("symbol") == symbol)["opened"]
                .dt.date()
                .to_list()
            ),
        )
        for symbol in data.trades["symbol"].unique().to_list()
    }
    needs_quotes = any(
        feature.template in ("leg_delta", "position_bid_ask_spread")
        for feature in features
    )
    for trade in data.trades.iter_rows(named=True):
        minute, daily = history[trade["symbol"]]
        legs = data.legs.filter(pl.col("trade_index") == trade["trade_index"])
        observations = _trade_observations(
            trade, legs, market, minute, selected_mode, slippage, needs_quotes
        )
        if event_features:
            observations = observations.join(
                event_columns, on="date", how="left", validate="m:1"
            )
        values = build_features(observations, daily, features, events=events)
        ignored = _ignored_features(legs, features)
        entries.append(values[:1])
        entry_ignored.append(ignored[None])
        exits.append(values[1:])
        exit_ignored.append(ignored.expand(values.shape[0] - 1, -1))
        records.append(observations.slice(1))
    exit_frame = pl.concat(records, how="vertical_relaxed")
    dates = {day: index for index, day in enumerate(data.dates)}
    tensors = StrategyTensors(
        entry_features=torch.cat(entries),
        exit_features=torch.cat(exits),
        entry_ignored=torch.cat(entry_ignored),
        exit_ignored=torch.cat(exit_ignored),
        exit_trades=torch.tensor(exit_frame["trade_index"].to_list(), dtype=torch.long),
        exit_ror=torch.tensor(exit_frame["ror_pct"].to_list(), dtype=torch.float32),
        original_ror=torch.tensor(data.trades["ror"].to_list(), dtype=torch.float32),
        trade_days=torch.tensor(
            [dates[opened.date()] for opened in data.trades["opened"].to_list()],
            dtype=torch.long,
        ),
        market_days=len(data.dates),
    )
    return PreparedStrategy(data, features, tensors, exit_frame)
