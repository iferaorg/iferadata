"""Load individual dated trades when daily strategy aggregation loses positions.

Legacy CSVs without any Closed columns continue through the daily-return loader.
In a mixed portfolio, a strategy without Closed is treated as intraday: its close
is inferred as 16:00 on the opening date, or its opening time if that is later.
This inference supplies no intraday mark-to-market information.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl

from ._portfolio_blocking import (
    _datetime_expression,
    _matching_column,
    parse_strategy_trade_legs,
)
from ._position_model import PositionModel, PositionTrade

_OPTION_COLUMNS = ("Symbol", "Exp", "Legs")
_LEG_KEYS = ["symbol", "expiration", "right", "strike"]


def _read_csv(path: Path) -> pl.DataFrame:
    """Read a nonempty strategy CSV with a useful filename in error messages."""
    try:
        frame = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Could not read strategy CSV {path.name}: {exc}") from exc
    if frame.is_empty():
        raise ValueError(f"Strategy CSV is empty: {path.name}")
    return frame


def _required_column(frame: pl.DataFrame, name: str, path: Path) -> str:
    """Find one required column using the shared case-insensitive matching rule."""
    column = _matching_column(frame.columns, name)
    if column is None:
        raise ValueError(f"{path.name} is missing the {name!r} column")
    return column


def _strategy_trades(
    raw: pl.DataFrame, path: Path, strategy_index: int
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Normalize individual trades and, when supplied, their option-leg data."""
    opened_column = _required_column(raw, "Opened", path)
    return_column = _required_column(raw, "ROR", path)
    closed_column = _matching_column(raw.columns, "Closed")
    opened = _datetime_expression(opened_column)
    if closed_column is None:
        closed = pl.max_horizontal(
            opened, opened.dt.truncate("1d") + pl.duration(hours=16)
        )
    else:
        closed = _datetime_expression(closed_column)
    try:
        frame = raw.with_row_index("trade_index").select(
            pl.lit(strategy_index, dtype=pl.Int64).alias("strategy_index"),
            pl.col("trade_index").cast(pl.Int64),
            opened.alias("opened"),
            closed.alias("closed"),
            (pl.col(return_column).cast(pl.Float64, strict=True) / 100.0).alias(
                "return_on_risk"
            ),
        )
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"Could not parse {path.name}: {exc}") from exc

    for column in ("opened", "closed"):
        if frame[column].null_count():
            raise ValueError(
                f"{path.name} contains an invalid {column.title()} timestamp"
            )
    if frame.filter(pl.col("closed") < pl.col("opened")).height:
        raise ValueError(f"{path.name} contains a Closed time before Opened")
    if frame.filter(
        pl.col("return_on_risk").is_null() | ~pl.col("return_on_risk").is_finite()
    ).height:
        raise ValueError(f"{path.name} contains a non-finite ROR value")

    # Closed alone is a complete position schema; option metadata is all-or-none.
    has_option_metadata = any(
        _matching_column(raw.columns, column) is not None for column in _OPTION_COLUMNS
    )
    legs = (
        parse_strategy_trade_legs(raw, path, strategy_index, opened_column)
        if has_option_metadata
        else None
    )
    return frame, legs


def _position_blockers(
    trades: pl.DataFrame, strategy_legs: tuple[pl.DataFrame | None, ...]
) -> dict[int, tuple[int, ...]]:
    """Resolve leg conflicts to exact trade identities, including across dates."""
    frames = [frame for frame in strategy_legs if frame is not None]
    if len(frames) < 2:
        return {}
    identities = trades.select("strategy_index", "trade_index", "trade_id")
    legs = pl.concat(frames, how="vertical").join(
        identities, on=["strategy_index", "trade_index"], how="left", validate="m:1"
    )
    pairs = legs.join(legs, on=_LEG_KEYS, how="inner", suffix="_other").filter(
        (pl.col("strategy_index") < pl.col("strategy_index_other"))
        & (pl.col("direction") != pl.col("direction_other"))
        & (pl.col("opened") < pl.col("closed_other"))
        & (pl.col("opened_other") < pl.col("closed"))
    )
    left_is_earlier = (pl.col("opened") < pl.col("opened_other")) | (
        (pl.col("opened") == pl.col("opened_other"))
        & (pl.col("strategy_index") < pl.col("strategy_index_other"))
    )
    conflicts = pairs.select(
        pl.when(left_is_earlier)
        .then(pl.col("trade_id_other"))
        .otherwise(pl.col("trade_id"))
        .alias("blocked"),
        pl.when(left_is_earlier)
        .then(pl.col("trade_id"))
        .otherwise(pl.col("trade_id_other"))
        .alias("blocked_by"),
    ).unique()
    blockers: dict[int, set[int]] = {}
    for blocked, blocked_by in conflicts.iter_rows():
        blockers.setdefault(int(blocked), set()).add(int(blocked_by))
    return {index: tuple(sorted(earlier)) for index, earlier in blockers.items()}


def load_position_model(
    csv_paths: tuple[Path, ...],
    strategy_names: tuple[str, ...],
    valuation_method: str,
    market_dates: Callable[[date, date], pl.DataFrame],
) -> PositionModel | None:
    """Load a trade ledger only for multi-day or multiple-entry strategies.

    Closed is independent of optional Symbol/Exp/Legs blocking metadata. If no
    strategy supplies Closed, return None to preserve the legacy loader and its
    duplicate-date warning. Otherwise repeated entries on a strategy-day trigger
    position accounting even if their holding intervals do not overlap.
    """
    if len(csv_paths) != len(strategy_names) or not csv_paths:
        raise ValueError(
            "csv_paths and strategy_names must have the same nonzero length"
        )
    raw_frames = tuple(_read_csv(path) for path in csv_paths)
    if not any(
        _matching_column(frame.columns, "Closed") is not None for frame in raw_frames
    ):
        return None
    loaded = tuple(
        _strategy_trades(raw, path, strategy_index)
        for strategy_index, (raw, path) in enumerate(zip(raw_frames, csv_paths))
    )
    trades = pl.concat([frame for frame, _ in loaded], how="vertical").sort(
        "opened", "strategy_index", "trade_index"
    )
    multiday = trades.filter(pl.col("opened").dt.date() != pl.col("closed").dt.date())
    repeated_day = (
        trades.with_columns(pl.col("opened").dt.date().alias("date"))
        .group_by("strategy_index", "date")
        .len()
        .filter(pl.col("len") > 1)
    )
    if multiday.is_empty() and repeated_day.is_empty():
        return None

    trades = trades.with_row_index("trade_id")
    opened_dates = trades["opened"].dt.date().to_list()
    closed_dates = trades["closed"].dt.date().to_list()
    dates = tuple(market_dates(min(opened_dates), max(closed_dates))["date"].to_list())
    nonmarket = sorted((set(opened_dates) | set(closed_dates)) - set(dates))
    if nonmarket:
        invalid_dates = ", ".join(str(value) for value in nonmarket)
        raise ValueError(
            f"Position trades contain non-NYSE market dates: {invalid_dates}"
        )
    blockers = _position_blockers(trades, tuple(legs for _, legs in loaded))
    positions = tuple(
        PositionTrade(
            strategy_index=int(row["strategy_index"]),
            opened=row["opened"],
            closed=row["closed"],
            return_on_risk=float(row["return_on_risk"]),
            blockers=blockers.get(int(row["trade_id"]), ()),
        )
        for row in trades.iter_rows(named=True)
    )
    return PositionModel(dates, strategy_names, positions, valuation_method)
