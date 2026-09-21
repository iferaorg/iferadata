"""Parse option legs and derive cross-strategy trade-block relationships."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl

from ._capital_allocation_blocks import Block

_METADATA_COLUMNS = ("Closed", "Symbol", "Exp", "Legs")
_DATETIME_FORMATS = (
    "%b %-d, %Y %-I:%M%p",
    "%b %-d, %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)
_JOIN_KEYS = ["symbol", "expiration", "right", "strike"]


def _matching_column(columns: list[str], requested: str) -> str | None:
    """Return one case-insensitive column match, rejecting ambiguous schemas."""
    matches = [
        column
        for column in columns
        if column.strip().casefold() == requested.casefold()
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple columns match {requested!r}")
    return None if not matches else matches[0]


def _datetime_expression(column_name: str) -> pl.Expr:
    """Parse one backtest timestamp using the supported CSV formats."""
    value = pl.col(column_name).cast(pl.String).str.strip_chars()
    parsed: list[pl.Expr] = []
    for date_format in _DATETIME_FORMATS:
        if "%H" in date_format or "%I" in date_format or "%-I" in date_format:
            parsed.append(value.str.to_datetime(date_format, strict=False))
        else:
            parsed.append(
                value.str.to_date(date_format, strict=False).cast(pl.Datetime("us"))
            )
    return pl.coalesce(parsed)


def _date_expression(column_name: str) -> pl.Expr:
    """Parse one backtest date using the supported CSV formats."""
    return _datetime_expression(column_name).dt.date()


def parse_strategy_trade_legs(
    raw_frame: pl.DataFrame,
    path: Path,
    strategy_index: int,
    opened_column: str,
) -> pl.DataFrame | None:
    """Return normalized, exploded option legs or None for a legacy minimal CSV."""
    metadata_columns = {
        requested: _matching_column(raw_frame.columns, requested)
        for requested in _METADATA_COLUMNS
    }
    present = [name for name, column in metadata_columns.items() if column is not None]
    if not present:
        return None
    if len(present) != len(_METADATA_COLUMNS):
        missing = ", ".join(
            name for name, column in metadata_columns.items() if column is None
        )
        raise ValueError(
            f"{path.name} has incomplete trade-block metadata; missing: {missing}"
        )

    closed_column = metadata_columns["Closed"]
    symbol_column = metadata_columns["Symbol"]
    expiration_column = metadata_columns["Exp"]
    legs_column = metadata_columns["Legs"]
    if any(
        column is None
        for column in (closed_column, symbol_column, expiration_column, legs_column)
    ):
        raise RuntimeError("trade-block metadata validation was inconsistent")
    closed_column = cast(str, closed_column)
    symbol_column = cast(str, symbol_column)
    expiration_column = cast(str, expiration_column)
    legs_column = cast(str, legs_column)

    try:
        trades = raw_frame.with_row_index("trade_index").select(
            pl.lit(strategy_index, dtype=pl.Int64).alias("strategy_index"),
            pl.col("trade_index").cast(pl.Int64),
            _datetime_expression(opened_column).alias("opened"),
            _datetime_expression(closed_column).alias("closed"),
            _date_expression(opened_column).alias("date"),
            pl.col(symbol_column)
            .cast(pl.String)
            .str.strip_chars()
            .str.to_uppercase()
            .alias("symbol"),
            _date_expression(expiration_column).alias("expiration"),
            pl.col(legs_column).cast(pl.String).str.split("/").alias("leg"),
        )
    except pl.exceptions.PolarsError as exc:
        raise ValueError(
            f"Could not parse trade metadata in {path.name}: {exc}"
        ) from exc

    required = ("opened", "closed", "date", "symbol", "expiration", "leg")
    if any(trades[column].null_count() for column in required):
        raise ValueError(
            f"{path.name} contains invalid or missing trade-block metadata"
        )
    if trades.filter(pl.col("closed") < pl.col("opened")).height:
        raise ValueError(f"{path.name} contains a Closed time before Opened")
    if trades.filter(pl.col("symbol") == "").height:
        raise ValueError(f"{path.name} contains an empty Symbol")

    leg = pl.col("leg").cast(pl.String).str.strip_chars().str.replace_all(r"\s+", " ")
    exploded = (
        trades.explode("leg", empty_as_null=True)
        .with_columns(leg.str.split(" ").alias("leg_parts"))
        .with_columns(
            pl.col("leg_parts")
            .list.get(0, null_on_oob=True)
            .str.to_lowercase()
            .alias("direction"),
            pl.col("leg_parts")
            .list.get(1, null_on_oob=True)
            .str.to_lowercase()
            .alias("right"),
            pl.col("leg_parts")
            .list.get(2, null_on_oob=True)
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
            .alias("strike"),
        )
    )
    invalid_leg = (
        (pl.col("leg_parts").list.len() != 3)
        | ~pl.col("direction").is_in(("long", "short"))
        | ~pl.col("right").is_in(("put", "call"))
        | pl.col("strike").is_null()
        | ~pl.col("strike").is_finite()
        | (pl.col("strike") < 0.0)
    )
    if exploded.filter(invalid_leg).height:
        invalid_legs = exploded.filter(invalid_leg)["leg"].unique()
        examples = ", ".join(repr(value) for value in invalid_legs.head(3).to_list())
        raise ValueError(f"{path.name} contains invalid Legs syntax: {examples}")
    return exploded.drop("leg", "leg_parts")


def build_block_list(
    strategy_trade_legs: tuple[pl.DataFrame | None, ...],
    synchronized_dates: pl.Series,
) -> tuple[Block, ...]:
    """Find overlapping, opposite option legs and return indexed block triples."""
    leg_frames = [frame for frame in strategy_trade_legs if frame is not None]
    if len(leg_frames) < 2:
        return ()
    legs = pl.concat(leg_frames, how="vertical", rechunk=True)
    pairs = legs.join(legs, on=_JOIN_KEYS, how="inner", suffix="_other").filter(
        (pl.col("strategy_index") < pl.col("strategy_index_other"))
        & (pl.col("direction") != pl.col("direction_other"))
        & (pl.col("opened") < pl.col("closed_other"))
        & (pl.col("opened_other") < pl.col("closed"))
    )
    if pairs.is_empty():
        return ()

    left_is_earlier = (pl.col("opened") < pl.col("opened_other")) | (
        (pl.col("opened") == pl.col("opened_other"))
        & (pl.col("strategy_index") < pl.col("strategy_index_other"))
    )
    dated_blocks = pairs.select(
        pl.when(left_is_earlier)
        .then(pl.col("date_other"))
        .otherwise(pl.col("date"))
        .alias("date"),
        pl.when(left_is_earlier)
        .then(pl.col("strategy_index_other"))
        .otherwise(pl.col("strategy_index"))
        .alias("blocked"),
        pl.when(left_is_earlier)
        .then(pl.col("strategy_index"))
        .otherwise(pl.col("strategy_index_other"))
        .alias("blocked_by"),
    ).unique()

    indexed_dates = pl.DataFrame(
        {
            "date": synchronized_dates,
            "day_idx": pl.arange(0, synchronized_dates.len(), eager=True),
        }
    )
    indexed_blocks = dated_blocks.join(
        indexed_dates, on="date", how="inner", validate="m:1"
    ).select("day_idx", "blocked", "blocked_by")
    return tuple(
        sorted(
            (int(day_idx), int(blocked), int(blocked_by))
            for day_idx, blocked, blocked_by in indexed_blocks.iter_rows()
        )
    )
