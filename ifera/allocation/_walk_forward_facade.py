"""Public walk-forward facade kept separate from portfolio reporting helpers."""

# The CSV facades intentionally share their strategy-selection tail arguments.
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import torch

if TYPE_CHECKING:
    from .portfolio_allocation import WalkForwardPortfolioAllocationResult


def walk_forward_capital_allocation_from_csv(
    portfolio_name: str,
    alpha: float,
    grid_increment: float | None = None,
    training_weeks: int | None = None,
    first_training_weeks: int | None = None,
    embargo_weeks: int | None = None,
    simulation_weeks: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    bootstrap_on: bool = False,
    bootstrap_runs: int = 1024,
    bootstrap_length: int = 256,
    percentile: float = 10.0,
    max_total_allocation: float = 1.0,
    annualization_periods: int = 252,
    risk_free_rate: float = 0.0,
    show_plot: bool = True,
    refinement_runs: int = 0,
    refinement_divisor: float = 2.0,
    ADD_limit: float | None = None,  # pylint: disable=invalid-name
    allocation_alpha: float = 1.0,
    returns_dtype: type[pl.DataType] | None = None,
    max_strategies: int | None = None,
    min_correlation: float | None = None,
    max_correlation: float | None = None,
    approximate_add_search: bool = False,
    allocation_tolerance: float | None = None,
    objective: str = "kelly",
    CMGR_limit: float | None = None,  # pylint: disable=invalid-name
    unrealized_pnl_method: str = "linear",
) -> WalkForwardPortfolioAllocationResult:
    """Run rolling portfolio optimization and report out-of-sample results."""
    from .walk_forward_allocation import (  # pylint: disable=import-outside-toplevel
        walk_forward_capital_allocation_from_csv as run_walk_forward,
    )

    if returns_dtype is None:
        returns_dtype = pl.Float32

    return run_walk_forward(
        portfolio_name=portfolio_name,
        alpha=alpha,
        grid_increment=grid_increment,
        training_weeks=training_weeks,
        first_training_weeks=first_training_weeks,
        embargo_weeks=embargo_weeks,
        simulation_weeks=simulation_weeks,
        device=device,
        dtype=dtype,
        bootstrap_on=bootstrap_on,
        bootstrap_runs=bootstrap_runs,
        bootstrap_length=bootstrap_length,
        percentile=percentile,
        max_total_allocation=max_total_allocation,
        annualization_periods=annualization_periods,
        risk_free_rate=risk_free_rate,
        show_plot=show_plot,
        refinement_runs=refinement_runs,
        refinement_divisor=refinement_divisor,
        ADD_limit=ADD_limit,
        allocation_alpha=allocation_alpha,
        returns_dtype=returns_dtype,
        max_strategies=max_strategies,
        min_correlation=min_correlation,
        max_correlation=max_correlation,
        approximate_add_search=approximate_add_search,
        allocation_tolerance=allocation_tolerance,
        objective=objective,
        CMGR_limit=CMGR_limit,
        unrealized_pnl_method=unrealized_pnl_method,
    )
