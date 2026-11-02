"""Build fractional trade returns from account P/L and per-contract leg fees."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

import polars as pl

from ._portfolio_blocking import _matching_column


def validate_fees_per_leg(value: float) -> float:
    """Require a finite, nonnegative fee for one leg on one transaction side."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("fees_per_leg must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("fees_per_leg must be finite and nonnegative")
    return result


def strategy_return_expression(
    raw: pl.DataFrame, path: Path, fees_per_leg: float = 0.0
) -> pl.Expr:
    """Return fractional net RoR, preferring accounting values over rounded ROR.

    P/L and Risk are position totals, so only fees are multiplied by Quantity.
    Expired trades incur entry fees; all other statuses incur entry and exit
    fees. Legacy ROR-only inputs remain supported only with zero fees and no
    accounting columns. Any supplied accounting columns must be complete.
    """
    fee = validate_fees_per_leg(fees_per_leg)
    profit_column = _optional_column(raw, "P/L", path)
    risk_column = _optional_column(raw, "Risk", path)
    if profit_column is None and risk_column is None and fee == 0.0:
        return _number(raw, "ROR", path) / 100.0
    profit = _number(raw, "P/L", path)
    risk = _number(raw, "Risk", path, positive=True)
    if fee == 0.0:
        return profit / risk
    quantity = _number(raw, "Quantity", path, positive=True)
    legs_column = _column(raw, "Legs", path)
    status_column = _column(raw, "Status", path)
    legs = pl.col(legs_column).cast(pl.String).str.split("/")
    invalid_legs = (
        legs.is_null()
        | legs.list.eval(pl.element().str.strip_chars().eq("")).list.any()
    )
    if raw.select(invalid_legs.any()).item():
        raise ValueError(f"{path.name} contains missing or empty Legs entries")
    expired = (
        pl.col(status_column)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .eq("expired")
        .fill_null(False)
    )
    sides = pl.when(expired).then(1.0).otherwise(2.0)
    # Cast counts before arithmetic so integer overflow cannot hide total fees.
    fees = fee * legs.list.len().cast(pl.Float64) * sides * quantity
    return (profit - fees) / risk


def _optional_column(raw: pl.DataFrame, name: str, path: Path) -> str | None:
    """Resolve an optional field with its source filename in ambiguity errors."""
    try:
        return _matching_column(raw.columns, name)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _column(raw: pl.DataFrame, name: str, path: Path) -> str:
    """Require one case-insensitive field, reporting its source filename."""
    column = _optional_column(raw, name, path)
    if column is None:
        raise ValueError(f"{path.name} is missing the {name!r} column")
    return column


def _number(
    raw: pl.DataFrame, name: str, path: Path, positive: bool = False
) -> pl.Expr:
    """Validate a numeric account field without narrowing its input precision."""
    column = _column(raw, name, path)
    value = pl.col(column).cast(pl.Float64, strict=False)
    invalid = value.is_null() | ~value.is_finite()
    if positive:
        invalid |= value <= 0.0
    if raw.select(invalid.any()).item():
        if name == "ROR":
            raise ValueError(f"{path.name} contains a non-finite ROR value")
        requirement = "finite positive" if positive else "finite"
        raise ValueError(f"{path.name} requires {requirement} {name} values")
    return value
