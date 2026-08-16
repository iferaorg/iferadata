"""Walk-forward simulation for portfolio capital allocation."""

# This internal orchestration module deliberately reuses sibling reporting
# helpers and mirrors the public optimizer's keyword forwarding.
# pylint: disable=cyclic-import,duplicate-code,protected-access

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from datetime import date, timedelta
from numbers import Real

import polars as pl
import torch

from . import capital_allocation
from . import portfolio_allocation as portfolio


@dataclass(frozen=True)
class _WalkForwardWindow:
    """Half-open calendar boundaries for one walk-forward fold."""

    training_start: date
    training_stop: date
    simulation_start: date
    simulation_stop: date


def walk_forward_capital_allocation_from_csv(
    portfolio_name: str,
    alpha: float,
    grid_increment: float,
    training_weeks: int,
    embargo_weeks: int,
    simulation_weeks: int,
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
    returns_dtype: type[pl.DataType] = pl.Float32,
) -> portfolio.WalkForwardPortfolioAllocationResult:
    """Optimize rolling training windows and simulate allocations out of sample.

    Weeks are Monday-anchored calendar weeks. A dataset's partial opening week
    is excluded, but a holiday-shortened week is complete when the first data
    date is its first NYSE session. Each fixed-length training window is
    followed by the embargo and simulation windows. The complete structure
    advances by ``simulation_weeks`` until the final, optionally partial,
    simulation reaches the end of the dataset.

    Every fold independently calls
    :func:`ifera.allocation.find_optimal_capital_allocation`, so
    bootstrap samples and all other optimizer state are fresh. Aggregate
    statistics use only the concatenated out-of-sample returns. Historical ADD
    is reported; bootstrap ADD is intentionally omitted. Strategy returns and
    optimizer inputs use ``returns_dtype``, which defaults to ``polars.Float32``.

    After the first fold, ``allocation_alpha`` applies a recursive EMA between
    the newly calculated optimum and the preceding applied allocation. This is
    a post-processing rule: changing overlap constraints or a nonlinear
    ``ADD_limit`` can mean the smoothed allocation does not satisfy the latest
    training fold's constraints, and it is not silently projected or clamped.
    """
    training_count = _validate_week_count(training_weeks, "training_weeks", True)
    embargo_count = _validate_week_count(embargo_weeks, "embargo_weeks", False)
    simulation_count = _validate_week_count(simulation_weeks, "simulation_weeks", True)
    allocation_alpha_value = _validate_allocation_alpha(allocation_alpha)
    annualization_value = portfolio._validate_annualization_periods(
        annualization_periods
    )
    risk_free_value = portfolio._validate_risk_free_rate(risk_free_rate)
    if not isinstance(show_plot, bool):
        raise TypeError("show_plot must be a bool")

    strategy_names, synchronized_returns, _ = portfolio._load_portfolio_returns(
        portfolio_name, returns_dtype
    )
    first_date = synchronized_returns["date"].item(0)
    last_date = synchronized_returns["date"].item(-1)
    aligned_start = _first_complete_week_start(first_date)
    windows = _walk_forward_windows(
        aligned_start,
        last_date,
        training_count,
        embargo_count,
        simulation_count,
    )
    if not windows:
        raise ValueError("Dataset has no walk-forward simulation period")

    folds: list[portfolio.WalkForwardFold] = []
    simulation_frames: list[pl.DataFrame] = []
    weighted_allocations: list[tuple[torch.Tensor, int]] = []
    previous_applied_allocation: torch.Tensor | None = None
    for window in portfolio.tqdm(
        windows, total=len(windows), desc="Walk-forward portfolio", unit="fold"
    ):
        training = _date_slice(
            synchronized_returns, window.training_start, window.training_stop
        )
        simulation = _date_slice(
            synchronized_returns, window.simulation_start, window.simulation_stop
        )
        allocation = capital_allocation.find_optimal_capital_allocation(
            training.select(strategy_names).to_torch(),
            alpha=alpha,
            grid_increment=grid_increment,
            device=device,
            dtype=dtype,
            bootstrap_on=bootstrap_on,
            bootstrap_runs=bootstrap_runs,
            bootstrap_length=bootstrap_length,
            percentile=percentile,
            max_total_allocation=max_total_allocation,
            refinement_runs=refinement_runs,
            refinement_divisor=refinement_divisor,
            ADD_limit=ADD_limit,
        )
        calculated_allocation = allocation.detach().clone()
        applied_allocation = _applied_allocation(
            calculated_allocation,
            previous_applied_allocation,
            allocation_alpha_value,
        )
        previous_applied_allocation = applied_allocation
        weights = applied_allocation.to(device="cpu", dtype=torch.float64).tolist()
        simulation_frames.append(
            portfolio._apply_allocation(simulation, strategy_names, weights)
        )
        weighted_allocations.append((applied_allocation, simulation.height))
        folds.append(
            portfolio.WalkForwardFold(
                training_start_date=training["date"].item(0),
                training_end_date=training["date"].item(-1),
                simulation_start_date=simulation["date"].item(0),
                simulation_end_date=simulation["date"].item(-1),
                calculated_allocation=calculated_allocation,
                applied_allocation=applied_allocation,
            )
        )

    simulated_returns = pl.concat(simulation_frames, how="vertical")
    daily_results = portfolio._add_equity_curve(simulated_returns)
    statistics = portfolio._portfolio_statistics(
        daily_results, annualization_value, risk_free_value, None
    )
    average_allocation = _weighted_average_allocation(weighted_allocations)
    diversification = portfolio._portfolio_diversification(
        simulated_returns.select(strategy_names).to_torch(),
        strategy_names,
        average_allocation,
    )
    result = portfolio.WalkForwardPortfolioAllocationResult(
        strategy_names=strategy_names,
        folds=tuple(folds),
        average_allocation=average_allocation,
        daily_results=daily_results,
        statistics=statistics,
        diversification=diversification,
    )
    _print_report(portfolio_name, result)
    if show_plot:
        portfolio._plot_equity_curve(
            portfolio_name,
            daily_results,
            title_prefix="Walk-forward simulated equity curve",
        )
        _plot_allocations(portfolio_name, result)
    return result


def _validate_week_count(value: int, name: str, strictly_positive: bool) -> int:
    """Validate a whole-calendar-week duration."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    minimum = 1 if strictly_positive else 0
    if count < minimum:
        requirement = "positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} must be {requirement}")
    return count


def _validate_allocation_alpha(value: float) -> float:
    """Return a finite EMA coefficient from zero through one."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("allocation_alpha must be a real number")
    coefficient = float(value)
    if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("allocation_alpha must be finite and in [0, 1]")
    return coefficient


def _applied_allocation(
    calculated: torch.Tensor,
    previous_applied: torch.Tensor | None,
    allocation_alpha: float,
) -> torch.Tensor:
    """Apply one recursive EMA step while preserving tensor dtype and device."""
    if previous_applied is None or allocation_alpha == 1.0:
        return calculated.clone()
    if allocation_alpha == 0.0:
        return previous_applied.clone()
    return torch.lerp(previous_applied, calculated, allocation_alpha)


def _monday(calendar_date: date) -> date:
    """Return the Monday anchoring a calendar date's week."""
    return calendar_date - timedelta(days=calendar_date.weekday())


def _first_complete_week_start(first_date: date) -> date:
    """Return the first Monday whose available NYSE week is complete."""
    week_start = _monday(first_date)
    expected_sessions = portfolio._nyse_market_dates(
        week_start, week_start + timedelta(days=6)
    )
    first_expected = expected_sessions["date"].item(0)
    if first_date == first_expected:
        return week_start
    return week_start + timedelta(weeks=1)


def _walk_forward_windows(
    aligned_start: date,
    last_date: date,
    training_weeks: int,
    embargo_weeks: int,
    simulation_weeks: int,
) -> tuple[_WalkForwardWindow, ...]:
    """Build every half-open rolling calendar window with a simulation row."""
    windows: list[_WalkForwardWindow] = []
    try:
        shift = timedelta(weeks=simulation_weeks)
        training_duration = timedelta(weeks=training_weeks)
        embargo_duration = timedelta(weeks=embargo_weeks)
        simulation_duration = timedelta(weeks=simulation_weeks)
    except OverflowError as exc:
        raise ValueError("Walk-forward week parameters are too large") from exc
    training_start = aligned_start
    while True:
        training_stop = training_start + training_duration
        simulation_start = training_stop + embargo_duration
        if simulation_start > last_date:
            break
        windows.append(
            _WalkForwardWindow(
                training_start=training_start,
                training_stop=training_stop,
                simulation_start=simulation_start,
                simulation_stop=simulation_start + simulation_duration,
            )
        )
        training_start += shift
    return tuple(windows)


def _date_slice(frame: pl.DataFrame, start: date, stop: date) -> pl.DataFrame:
    """Return rows in a half-open calendar-date interval."""
    sliced = frame.filter((pl.col("date") >= start) & (pl.col("date") < stop))
    if sliced.is_empty():
        raise ValueError("Walk-forward period contains no NYSE market sessions")
    return sliced


def _weighted_average_allocation(
    weighted_allocations: list[tuple[torch.Tensor, int]],
) -> torch.Tensor:
    """Return a simulation-market-day-weighted average allocation on CPU."""
    strategy_count = weighted_allocations[0][0].numel()
    average = torch.zeros(strategy_count, dtype=torch.float64)
    total_days = 0
    for allocation, market_days in weighted_allocations:
        average += allocation.to(device="cpu", dtype=torch.float64) * market_days
        total_days += market_days
    average /= total_days
    return average


def _print_report(
    portfolio_name: str, result: portfolio.WalkForwardPortfolioAllocationResult
) -> None:
    """Print fold context, aggregate allocations, and out-of-sample statistics."""
    print(f"\nWalk-forward portfolio simulation: {portfolio_name}")
    print(f"Folds: {len(result.folds)}")
    print(f"Period: {result.statistics.start_date} to {result.statistics.end_date}")
    print("Simulation-day-weighted average applied allocations:")
    weights = result.average_allocation.tolist()
    for strategy_name, weight in zip(result.strategy_names, weights):
        print(f"  {strategy_name}: {portfolio._format_percentage(weight)}")
    print(f"  Total: {portfolio._format_percentage(sum(weights))}")
    portfolio._print_combined_statistics(result.statistics)
    portfolio._print_diversification(result.diversification)


def _plot_allocations(
    portfolio_name: str, result: portfolio.WalkForwardPortfolioAllocationResult
) -> None:
    """Display every strategy's piecewise-constant simulated allocation."""
    # Imported lazily so headless callers do not pay Matplotlib's import cost.
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.ticker import (  # pylint: disable=import-outside-toplevel
        PercentFormatter,
    )

    fold_dates = [fold.simulation_start_date for fold in result.folds]
    fold_dates.append(result.folds[-1].simulation_end_date + timedelta(days=1))
    applied_allocations = torch.stack(
        [
            fold.applied_allocation.detach().to(device="cpu", dtype=torch.float64)
            for fold in result.folds
        ]
    )
    calculated_allocations = torch.stack(
        [
            fold.calculated_allocation.detach().to(device="cpu", dtype=torch.float64)
            for fold in result.folds
        ]
    )
    applied_allocations = torch.cat(
        (applied_allocations, applied_allocations[-1:].clone()), dim=0
    )
    calculated_allocations = torch.cat(
        (calculated_allocations, calculated_allocations[-1:].clone()), dim=0
    )

    _, axis = plt.subplots(figsize=(11, 6))
    color_map = plt.get_cmap("turbo", len(result.strategy_names))
    for strategy_index, strategy_name in enumerate(result.strategy_names):
        color = tuple(color_map(strategy_index))
        axis.step(
            fold_dates,  # pyright: ignore[reportArgumentType]
            calculated_allocations[:, strategy_index].tolist(),
            where="post",
            label="_nolegend_",
            color=color,
            linewidth=0.75,
            linestyle="--",
            alpha=0.25,
        )
        axis.step(
            fold_dates,  # pyright: ignore[reportArgumentType]
            applied_allocations[:, strategy_index].tolist(),
            where="post",
            label=strategy_name,
            color=color,
            linewidth=1.75,
        )
    axis.set_title(f"Walk-forward allocations: {portfolio_name}")
    axis.set_xlabel("Simulation period")
    axis.set_ylabel("Allocation")
    axis.set_ylim(bottom=0.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(visible=True, alpha=0.3)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="Applied allocations\n(calculated: dashed)",
    )
    plt.tight_layout()
    plt.show()
