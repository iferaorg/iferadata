"""Lightweight CSV facades for the tensor-only allocation module."""

# The walk-forward import is an intentional public API re-export. Both thin
# facades also forward their shared trailing strategy-selection parameters.
# pylint: disable=duplicate-code,unused-import

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._walk_forward_facade import walk_forward_capital_allocation_from_csv

if TYPE_CHECKING:
    from .portfolio_allocation import PortfolioAllocationResult


def find_optimal_capital_allocation_from_csv(
    portfolio_name: str,
    alpha: float,
    grid_increment: float | None = None,
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
    returns_dtype: type | None = None,
    max_strategies: int | None = None,
    min_correlation: float | None = None,
    max_correlation: float | None = None,
    approximate_add_search: bool = False,
    allocation_tolerance: float | None = None,
    objective: str = "kelly",
    CMGR_limit: float | None = None,  # pylint: disable=invalid-name
    unrealized_pnl_method: str = "linear",
) -> PortfolioAllocationResult:
    """Load a named CSV portfolio, optimize it, and report the result."""
    from .portfolio_allocation import (  # pylint: disable=import-outside-toplevel
        find_optimal_capital_allocation_from_csv as find_from_csv,
    )

    if returns_dtype is None:
        from polars import Float32  # pylint: disable=import-outside-toplevel

        returns_dtype = Float32

    return find_from_csv(
        portfolio_name,
        alpha,
        grid_increment,
        device,
        dtype,
        bootstrap_on,
        bootstrap_runs,
        bootstrap_length,
        percentile,
        max_total_allocation,
        annualization_periods,
        risk_free_rate,
        show_plot,
        refinement_runs,
        refinement_divisor,
        ADD_limit=ADD_limit,
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
