"""Reuse allocation statistics and display strategy optimization progress."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import matplotlib.pyplot as plt
import polars as pl
import torch
from matplotlib.figure import Figure
from torch import Tensor

from ifera.allocation.capital_allocation import find_optimal_capital_allocation
from ifera.allocation.portfolio_allocation import (
    PortfolioStatistics,
    _add_equity_curve,
    _combined_daily_results,
    _portfolio_statistics,
    _print_combined_statistics,
    _safe_ratio,
)

from ifera.strategy_optimization.metrics import _validate


def _validate_statistics_inputs(
    dates: tuple[date, ...],
    daily_ror: Tensor,
    active: Tensor,
    annualization_periods: int,
) -> None:
    """Validate the calendar, percentage returns, and explicit trade activity."""
    if daily_ror.ndim != 1 or active.ndim != 1 or active.shape != daily_ror.shape:
        raise ValueError(
            "daily_ror and active must be equally sized one-dimensional tensors"
        )
    if active.dtype != torch.bool:
        raise TypeError("active must have a boolean dtype")
    if (
        len(dates) != daily_ror.numel()
        or not dates
        or tuple(sorted(set(dates))) != dates
    ):
        raise ValueError(
            "dates must match the returns and be nonempty, unique, and sorted"
        )
    _validate(daily_ror.unsqueeze(0), annualization_periods)


def strategy_statistics(
    dates: tuple[date, ...],
    daily_ror: Tensor,
    active: Tensor,
    *,
    annualization_periods: int = 252,
) -> tuple[pl.DataFrame, PortfolioStatistics]:
    """Return allocation-style statistics and daily returns for one strategy.

    Input RoR uses percent; ``combined_return`` in the returned frame uses
    fractions to match the allocation package. Compounded equity treats returns
    at or below -100% as absorbing bankruptcy, while arithmetic statistics and
    the returned daily returns retain the original realized losses.
    """
    _validate_statistics_inputs(dates, daily_ror, active, annualization_periods)
    raw = pl.DataFrame(
        {
            "date": dates,
            "active": active.detach().cpu().tolist(),
            "combined_return": (daily_ror.detach().cpu().double() / 100.0).tolist(),
        },
        schema={"date": pl.Date, "active": pl.Boolean, "combined_return": pl.Float64},
    )
    daily = _add_equity_curve(
        raw.with_columns(pl.col("combined_return").clip(lower_bound=-1.0))
    )
    statistics = _portfolio_statistics(daily, annualization_periods, 0.0, None)
    statistics = _arithmetic_statistics(raw, statistics, annualization_periods)
    return daily.with_columns(raw["combined_return"]), statistics


def allocated_strategy_statistics(
    dates: tuple[date, ...],
    daily_ror: Tensor,
    active: Tensor,
    *,
    annualization_periods: int = 252,
) -> tuple[pl.DataFrame, PortfolioStatistics, float]:
    """Allocate the winning strategy once and report its funded equity path.

    The full original market calendar is retained. Inactive sessions are NaN in
    the one-strategy allocation input, while active breakeven trades remain zero.
    Trade returns and exported accounting retain their original risk basis.
    """
    _validate_statistics_inputs(dates, daily_ror, active, annualization_periods)
    returns = torch.where(
        active.to(device=daily_ror.device), daily_ror.double() / 100.0, torch.nan
    )
    allocation = find_optimal_capital_allocation(
        returns.unsqueeze(1),
        alpha=0.0,
        grid_increment=0.001,
        device=returns.device,
        dtype=torch.float64,
        bootstrap_on=False,
        refinement_runs=0,
        ADD_limit=-0.1,
    )
    weight = float(allocation[0].item())
    synchronized = pl.DataFrame(
        {"date": dates, "strategy_return": returns.detach().cpu().tolist()},
        schema={"date": pl.Date, "strategy_return": pl.Float64},
    ).with_columns(pl.col("strategy_return").fill_nan(None))
    daily = _combined_daily_results(synchronized, ("strategy_return",), [weight])
    statistics = _portfolio_statistics(daily, annualization_periods, 0.0, None)
    return daily, statistics, weight


def _arithmetic_statistics(
    raw: pl.DataFrame, statistics: PortfolioStatistics, annualization_periods: int
) -> PortfolioStatistics:
    """Keep the actual P/L in statistics unaffected by the bankruptcy convention."""
    returns = pl.col("combined_return")
    summary = raw.select(
        returns.mean().alias("mean"),
        returns.std(ddof=1).alias("volatility"),
        returns.clip(upper_bound=0.0).pow(2).mean().sqrt().alias("downside"),
        returns.filter(returns > 0.0).sum().alias("gross_profit"),
        returns.filter(returns < 0.0).sum().abs().alias("gross_loss"),
        returns.filter(pl.col("active")).max().alias("best_day"),
        returns.filter(pl.col("active")).min().alias("worst_day"),
    ).row(0, named=True)
    mean = float(summary["mean"])
    volatility = float(summary["volatility"] or 0.0)
    scale = math.sqrt(annualization_periods)
    return replace(
        statistics,
        average_daily_return=mean,
        annualized_volatility=volatility * scale,
        sharpe_ratio=_safe_ratio(mean, volatility) * scale,
        sortino_ratio=_safe_ratio(mean, float(summary["downside"])) * scale,
        profit_factor=_safe_ratio(
            float(summary["gross_profit"]), float(summary["gross_loss"])
        ),
        best_day=float(summary["best_day"] or 0.0),
        worst_day=float(summary["worst_day"] or 0.0),
    )


def print_strategy_statistics(statistics: PortfolioStatistics) -> None:
    """Print the same financial statistics as the allocation package."""
    _print_combined_statistics(statistics)


def plot_objective_history(history: tuple[float, ...]) -> Figure:
    """Build a progress figure; the caller controls showing or saving it."""
    if not history or any(
        not math.isfinite(value) or value <= 0.0 for value in history
    ):
        raise ValueError("history must contain finite positive objective values")
    figure, axes = plt.subplots()
    axes.plot(range(len(history)), history)
    axes.set_xlabel("Generation")
    axes.set_ylabel("Best objective value")
    axes.set_title("Strategy optimization progress")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure
