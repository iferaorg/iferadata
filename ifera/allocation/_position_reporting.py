"""Polars frames and reporting for estimated daily position-account paths."""

from __future__ import annotations

import polars as pl
import torch

from ._position_model import PositionModel


def position_return_frame(
    model: PositionModel, returns_dtype: type[pl.DataType]
) -> pl.DataFrame:
    """Build the dated unit-risk proxy used for strategy correlations/selection."""
    tensor_dtype = torch.float64 if returns_dtype == pl.Float64 else torch.float32
    returns = model.returns_tensor(dtype=tensor_dtype)
    columns = [pl.Series("date", model.dates, dtype=pl.Date)]
    columns.extend(
        pl.Series(name, returns[:, index].cpu().numpy(), nan_to_null=True)
        for index, name in enumerate(model.strategy_names)
    )
    return pl.DataFrame(columns)


def simulate_position_frame(
    model: PositionModel,
    allocations: torch.Tensor,
    max_total_allocation: float,
    returns_dtype: type[pl.DataType],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replay a full entry-allocation schedule without resetting carried positions."""
    calculation_dtype = torch.promote_types(
        torch.float64 if returns_dtype == pl.Float64 else torch.float32,
        allocations.dtype,
    )
    schedule = allocations.detach().to(device="cpu", dtype=calculation_dtype)
    if schedule.ndim == 1:
        schedule = schedule.expand(len(model.dates), -1)
    simulation = model.simulate(schedule, max_total_allocation)
    frame = position_return_frame(model, returns_dtype).with_columns(
        pl.Series("active", simulation.active.cpu().numpy()),
        pl.Series(
            "combined_return",
            simulation.daily_returns.to(device="cpu", dtype=torch.float64).numpy(),
        ),
    )
    return frame, simulation.trade_results


def print_position_report(method: str | None, trades: pl.DataFrame | None) -> None:
    """Describe estimated marks, per-entry weights, and actually admitted trades."""
    if method is None or trades is None:
        return
    print("Position accounting:")
    print("  Allocations are per NEW trade, sized from start-of-day net liquidation.")
    print(
        "  Entry risk stays fixed until close; open positions carry across WFA folds."
    )
    print(f"  Estimated unrealized P/L method: {method}")
    if method == "linear":
        print(
            "  Marks interpolate final P/L; unobserved intra-trade drawdowns are unknown."
        )
    else:
        print("  Open positions are marked at cost; all P/L is recognized at close.")
    print(
        "  Capital is reserved per open trade; unaffordable entries are skipped in full."
    )
    print(
        "  Correlations use estimated daily unit-risk returns, including inactive zeros."
    )
    if not trades.is_empty():
        for status, count in trades.group_by("status").len().sort("status").iter_rows():
            print(f"  Trades {status}: {count}")
