"""Render effective predicates and export the winning trade ledger."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import polars as pl
from torch import Tensor

from ._evaluation import OptimizationMode, StrategyEvaluation
from ._features import FeatureSpec
from ._preparation import PreparedStrategy
from ._thresholds import round_bounds


def _threshold(value: float, feature: FeatureSpec) -> str:
    """Undo normalization for readable thresholds, including clock times."""
    raw = feature.min_value + value * (feature.max_value - feature.min_value)
    if value == 0:
        raw = feature.min_value
    elif value == 1:
        raw = feature.max_value
    elif feature.granularity is not None:
        raw = round(raw / feature.granularity) * feature.granularity
        raw = min(max(raw, feature.min_value), feature.max_value)
    precision = 0
    if feature.granularity is not None:
        precision = max(0, -int(Decimal(str(feature.granularity)).as_tuple().exponent))
        if value in (0, 1):
            precision = max(precision, -int(Decimal(str(raw)).as_tuple().exponent))
    if feature.template == "time":
        seconds = round(raw, precision)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        width = 2 + (precision + 1 if precision else 0)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:0{width}.{precision}f}"
    if feature.granularity is not None:
        rendered = f"{raw:.{precision}f}"
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
    return f"{raw:.6g}"


def _format_phase(
    bounds: Tensor,
    active: Tensor,
    features: tuple[FeatureSpec, ...],
    phase: str,
) -> str:
    """Omit inactive clauses, irrelevant features, and unrestricted atoms."""
    clauses = []
    for intervals, enabled in zip(bounds.tolist(), active.tolist()):
        if not enabled:
            continue
        atoms = []
        for (lower, upper), feature in zip(intervals, features):
            if feature.category not in (phase, "any") or (lower == 0 and upper == 1):
                continue
            if feature.kind == "bool":
                atoms.append(feature.name if lower == 1 else f"NOT {feature.name}")
            elif lower > 0 and upper < 1:
                atoms.append(
                    f"{_threshold(lower, feature)} <= {feature.name}"
                    f" <= {_threshold(upper, feature)}"
                )
            elif lower > 0:
                atoms.append(f"{feature.name} >= {_threshold(lower, feature)}")
            else:
                atoms.append(f"{feature.name} <= {_threshold(upper, feature)}")
        clauses.append("(" + " AND ".join(atoms) + ")" if atoms else "TRUE")
    return " OR ".join(clauses) if clauses else "FALSE"


def format_conditions(
    bounds: Tensor,
    active: Tensor,
    features: tuple[FeatureSpec, ...],
    mode: OptimizationMode | str,
) -> str:
    """Describe an unbatched winning DNF with thresholds in original units."""
    selected_mode = OptimizationMode(mode)
    bounds, active = (
        round_bounds(bounds.detach().cpu(), features),
        active.detach().cpu(),
    )
    if selected_mode == OptimizationMode.COMBINED:
        split = (bounds.shape[0] + 1) // 2
        entry = _format_phase(bounds[:split], active[:split], features, "entry")
        exit_rule = _format_phase(bounds[split:], active[split:], features, "exit")
        return f"Entry filter: {entry}\nEarly exit: {exit_rule}"
    if selected_mode == OptimizationMode.ENTRY_FILTER:
        return "Entry filter: " + _format_phase(bounds, active, features, "entry")
    return "Early exit: " + _format_phase(bounds, active, features, "exit")


def winning_trades(
    prepared: PreparedStrategy, evaluation: StrategyEvaluation
) -> pl.DataFrame:
    """Keep the source fields and order, changing accounting only for early exits."""
    entered = evaluation.entered[0].cpu().tolist()
    indices = evaluation.exit_indices[0].cpu().tolist()
    rows = []
    for index, original in enumerate(prepared.data.raw.iter_rows(named=True)):
        if not entered[index]:
            continue
        if indices[index] >= 0:
            mark = prepared.exits.row(indices[index], named=True)
            trade = prepared.data.trades.row(index, named=True)
            sign = 1.0 if trade["opening_price"] >= 0 else -1.0
            bid, ask = sorted((sign * mark["combo_bid"], sign * mark["combo_ask"]))
            updates = {
                "closed": mark["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "p/l": mark["profit"],
                "ror": mark["ror_pct"],
                "return %": mark["return_pct"],
                "close price": sign * mark["liquidation_value"],
                "close bid price": bid,
                "close ask price": ask,
                "price at close": mark["underlying_price"],
                "status": "closed",
            }
            original.update(
                {
                    column: updates[column.strip().casefold()]
                    for column in original
                    if column.strip().casefold() in updates
                }
            )
        rows.append(original)
    if not rows:
        return prepared.data.raw.clear()
    return pl.from_dicts(rows, infer_schema_length=None).select(
        prepared.data.raw.columns
    )


def save_strategy(frame: pl.DataFrame, output: Path, source: Path) -> Path:
    """Write the requested ledger without allowing the original to be replaced."""
    if output.resolve() == source.resolve():
        raise ValueError("output_csv must differ from the source backtest CSV")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output)
    return output
