"""Walk-forward simulation for portfolio capital allocation."""

# This internal orchestration module deliberately reuses sibling reporting
# helpers and mirrors the public optimizer's keyword forwarding.
# pylint: disable=cyclic-import,duplicate-code,protected-access

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from numbers import Real

import polars as pl
import torch

from . import capital_allocation
from . import portfolio_allocation as portfolio
from ._capital_allocation_blocks import (
    Block,
    remap_block_list,
    slice_block_list,
)
from ._strategy_selection import validate_strategy_selection_parameters


@dataclass(frozen=True)
class _WalkForwardWindow:
    """Half-open calendar boundaries for one walk-forward fold."""

    training_start: date
    training_stop: date
    simulation_start: date
    simulation_stop: date


_Support = tuple[int, ...]
_AllocationOptimizer = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class _DiscoveredFold:
    """A first-pass fold with its selected strategy regime."""

    window: _WalkForwardWindow
    training_start_date: date
    training_end_date: date
    calculated_allocation: torch.Tensor
    support: _Support
    bootstrap_seed: int | None
    block_list: tuple[Block, ...]


@dataclass(frozen=True)
class _FoldAllocation:
    """Calculated and applied allocations ready for simulation."""

    window: _WalkForwardWindow
    training_start_date: date
    training_end_date: date
    calculated_allocation: torch.Tensor
    applied_allocation: torch.Tensor


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
    returns_dtype: type[pl.DataType] = pl.Float32,
    max_strategies: int | None = None,
    min_correlation: float | None = None,
    max_correlation: float | None = None,
    approximate_add_search: bool = False,
    allocation_tolerance: float | None = None,
    objective: str = "kelly",
    CMGR_limit: float | None = None,  # pylint: disable=invalid-name
) -> portfolio.WalkForwardPortfolioAllocationResult:
    """Optimize rolling training windows and simulate allocations out of sample.

    Weeks are Monday-anchored calendar weeks. A dataset's partial opening week
    is excluded, but a holiday-shortened week is complete when the first data
    date is its first NYSE session. ``first_training_weeks`` can extend the
    initial training window; later folds use ``training_weeks`` and are
    positioned so out-of-sample simulation periods remain continuous. Each
    training window is followed by the embargo and simulation windows. The
    simulation advances by ``simulation_weeks`` until the final, optionally
    partial, period reaches the end of the dataset.

    Every fold gets fresh optimizer state and bootstrap row samples. Discovery
    and counterfactual regime searches within the same fold deliberately replay
    those row samples for a fair comparison. Aggregate statistics use only the
    concatenated out-of-sample returns. Historical ADD is reported; bootstrap
    ADD is intentionally omitted. Strategy returns and optimizer inputs use
    ``returns_dtype``, which defaults to ``polars.Float32``.

    Without strategy-selection limits, ``allocation_alpha`` applies one
    recursive EMA to consecutive fold optima. With any selection limit, a
    first pass discovers every support regime that is ever selected. A second
    pass independently optimizes and advances every discovered regime on every
    fold, including before its first selection and while it is inactive. The
    simulated allocation switches to the already-smoothed state of the support
    selected in the first pass. Only supports that win are replayed, and every
    counterfactual target uses that fold's own training data; discovering a
    future-winning support does not introduce future returns into earlier targets.

    ``objective="minimize_add"`` minimizes drawdown severity, ignoring ``alpha``;
    without a positive ``CMGR_limit`` its optimum is all-zero allocations. A
    positive ``CMGR_limit`` constrains either objective using the inclusive
    duration between that fold's first and last observed training dates, matching
    the report's compound monthly growth formula. Background regime searches
    use the same fold-local duration. The floor constrains training results only;
    it is not a guarantee for the out-of-sample simulation. An infeasible growth
    floor raises ValueError.

    EMA is post-processing: changing overlap constraints, an ``ADD_limit``, or
    a ``CMGR_limit`` can mean a smoothed allocation does not satisfy the latest
    training fold's constraints, and it is not silently projected or clamped.
    """
    training_count = _validate_week_count(training_weeks, "training_weeks", True)
    first_training_count = _validate_first_training_weeks(
        first_training_weeks, training_count
    )
    embargo_count = _validate_week_count(embargo_weeks, "embargo_weeks", False)
    simulation_count = _validate_week_count(simulation_weeks, "simulation_weeks", True)
    allocation_alpha_value = _validate_allocation_alpha(allocation_alpha)
    maximum, minimum_correlation, maximum_correlation = (
        validate_strategy_selection_parameters(
            max_strategies, min_correlation, max_correlation
        )
    )
    selection_requested = any(
        value is not None
        for value in (maximum, minimum_correlation, maximum_correlation)
    )
    annualization_value = portfolio._validate_annualization_periods(
        annualization_periods
    )
    risk_free_value = portfolio._validate_risk_free_rate(risk_free_rate)
    if not isinstance(show_plot, bool):
        raise TypeError("show_plot must be a bool")

    (
        strategy_names,
        synchronized_returns,
        _,
        portfolio_block_list,
    ) = portfolio._load_portfolio_returns(portfolio_name, returns_dtype)
    first_date = synchronized_returns["date"].item(0)
    last_date = synchronized_returns["date"].item(-1)
    aligned_start = _first_complete_week_start(first_date)
    windows = _walk_forward_windows(
        aligned_start,
        last_date,
        training_count,
        first_training_count,
        embargo_count,
        simulation_count,
    )
    if not windows:
        raise ValueError("Dataset has no walk-forward simulation period")

    base_optimizer = partial(
        capital_allocation.find_optimal_capital_allocation,
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
        approximate_add_search=approximate_add_search,
        allocation_tolerance=allocation_tolerance,
        objective=objective,
        CMGR_limit=CMGR_limit,
    )
    algorithm_counts: Counter[str] = Counter()
    optimizer = _algorithm_tracking_optimizer(base_optimizer, algorithm_counts)
    if selection_requested:
        fold_allocations = _selection_regime_allocations(
            synchronized_returns,
            strategy_names,
            windows,
            optimizer,
            allocation_alpha_value,
            bootstrap_on,
            maximum,
            minimum_correlation,
            maximum_correlation,
            portfolio_block_list,
        )
    else:
        fold_allocations = _ordinary_fold_allocations(
            synchronized_returns,
            strategy_names,
            windows,
            optimizer,
            allocation_alpha_value,
            portfolio_block_list,
        )

    folds: list[portfolio.WalkForwardFold] = []
    simulation_frames: list[pl.DataFrame] = []
    weighted_allocations: list[tuple[torch.Tensor, int]] = []
    for fold_allocation in fold_allocations:
        simulation, simulation_blocks = _date_slice_with_blocks(
            synchronized_returns,
            fold_allocation.window.simulation_start,
            fold_allocation.window.simulation_stop,
            portfolio_block_list,
        )
        weights = fold_allocation.applied_allocation.to(
            device="cpu", dtype=torch.float64
        ).tolist()
        simulation_frames.append(
            portfolio._apply_allocation(
                simulation, strategy_names, weights, simulation_blocks
            )
        )
        weighted_allocations.append(
            (fold_allocation.applied_allocation, simulation.height)
        )
        folds.append(
            portfolio.WalkForwardFold(
                training_start_date=fold_allocation.training_start_date,
                training_end_date=fold_allocation.training_end_date,
                simulation_start_date=simulation["date"].item(0),
                simulation_end_date=simulation["date"].item(-1),
                calculated_allocation=fold_allocation.calculated_allocation,
                applied_allocation=fold_allocation.applied_allocation,
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
        optimization_algorithm_counts=tuple(algorithm_counts.items()),
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


def _algorithm_tracking_optimizer(
    optimizer: _AllocationOptimizer,
    algorithm_counts: Counter[str],
) -> _AllocationOptimizer:
    """Wrap an optimizer and count its selected algorithms without ADD reporting."""

    def tracked_optimizer(returns: torch.Tensor, **kwargs: object) -> torch.Tensor:
        diagnostics = capital_allocation.CapitalAllocationDiagnostics(
            request_bootstrap_average_drawdown=False
        )
        allocation = optimizer(returns, diagnostics=diagnostics, **kwargs)
        algorithm_counts[diagnostics.algorithm or "Custom/unknown optimizer"] += 1
        return allocation

    return tracked_optimizer


def _ordinary_fold_allocations(
    synchronized_returns: pl.DataFrame,
    strategy_names: tuple[str, ...],
    windows: tuple[_WalkForwardWindow, ...],
    optimizer: _AllocationOptimizer,
    allocation_alpha: float,
    block_list: tuple[Block, ...],
) -> tuple[_FoldAllocation, ...]:
    """Calculate the original one-pass global EMA when selection is disabled."""
    fold_allocations: list[_FoldAllocation] = []
    previous_applied: torch.Tensor | None = None
    for window in portfolio.tqdm(
        windows, total=len(windows), desc="Walk-forward portfolio", unit="fold"
    ):
        training, training_blocks = _date_slice_with_blocks(
            synchronized_returns,
            window.training_start,
            window.training_stop,
            block_list,
        )
        calculated = optimizer(
            training.select(strategy_names).to_torch(),
            block_list=list(training_blocks),
            calendar_days=(training["date"].item(-1) - training["date"].item(0)).days
            + 1,
        )
        calculated = calculated.detach().clone()
        applied = _applied_allocation(calculated, previous_applied, allocation_alpha)
        previous_applied = applied
        fold_allocations.append(
            _FoldAllocation(
                window=window,
                training_start_date=training["date"].item(0),
                training_end_date=training["date"].item(-1),
                calculated_allocation=calculated,
                applied_allocation=applied,
            )
        )
    return tuple(fold_allocations)


def _selection_regime_allocations(
    synchronized_returns: pl.DataFrame,
    strategy_names: tuple[str, ...],
    windows: tuple[_WalkForwardWindow, ...],
    optimizer: _AllocationOptimizer,
    allocation_alpha: float,
    bootstrap_on: bool,
    max_strategies: int | None,
    min_correlation: float | None,
    max_correlation: float | None,
    block_list: tuple[Block, ...],
) -> tuple[_FoldAllocation, ...]:
    """Discover selected supports, then advance each support on every fold."""
    discovered_folds, supports = _discover_support_regimes(
        synchronized_returns,
        strategy_names,
        windows,
        optimizer,
        bootstrap_on,
        max_strategies,
        min_correlation,
        max_correlation,
        block_list,
    )
    if allocation_alpha == 1.0:
        return tuple(
            _FoldAllocation(
                window=discovered.window,
                training_start_date=discovered.training_start_date,
                training_end_date=discovered.training_end_date,
                calculated_allocation=discovered.calculated_allocation,
                applied_allocation=discovered.calculated_allocation.clone(),
            )
            for discovered in discovered_folds
        )
    applied_by_support: dict[_Support, torch.Tensor] = {}
    fold_allocations: list[_FoldAllocation] = []
    progress = portfolio.tqdm(
        enumerate(discovered_folds),
        total=len(discovered_folds),
        desc=f"Smoothing {len(supports)} allocation regimes",
        unit="fold",
    )
    for fold_index, discovered in progress:
        training = (
            _date_slice(
                synchronized_returns,
                discovered.window.training_start,
                discovered.window.training_stop,
            )
            .select(strategy_names)
            .to_torch()
        )
        supports_to_update = (
            supports if allocation_alpha > 0.0 or fold_index == 0 else ()
        )
        for support in supports_to_update:
            target = _regime_target(
                support,
                discovered,
                training,
                optimizer,
            )
            applied_by_support[support] = _applied_allocation(
                target,
                applied_by_support.get(support),
                allocation_alpha,
            )
        fold_allocations.append(
            _FoldAllocation(
                window=discovered.window,
                training_start_date=discovered.training_start_date,
                training_end_date=discovered.training_end_date,
                calculated_allocation=discovered.calculated_allocation,
                applied_allocation=applied_by_support[discovered.support].clone(),
            )
        )
    return tuple(fold_allocations)


def _discover_support_regimes(
    synchronized_returns: pl.DataFrame,
    strategy_names: tuple[str, ...],
    windows: tuple[_WalkForwardWindow, ...],
    optimizer: _AllocationOptimizer,
    bootstrap_on: bool,
    max_strategies: int | None,
    min_correlation: float | None,
    max_correlation: float | None,
    block_list: tuple[Block, ...],
) -> tuple[tuple[_DiscoveredFold, ...], tuple[_Support, ...]]:
    """Run constrained optimization and collect supports in first-seen order."""
    seeds = _fold_bootstrap_seeds(len(windows), bootstrap_on)
    discovered_folds: list[_DiscoveredFold] = []
    supports: list[_Support] = []
    seen_supports: set[_Support] = set()
    progress = portfolio.tqdm(
        windows,
        total=len(windows),
        desc="Detecting allocation regimes",
        unit="fold",
    )
    for window, bootstrap_seed in zip(progress, seeds, strict=True):
        training, training_blocks = _date_slice_with_blocks(
            synchronized_returns,
            window.training_start,
            window.training_stop,
            block_list,
        )
        calculated = optimizer(
            training.select(strategy_names).to_torch(),
            bootstrap_seed=bootstrap_seed,
            max_strategies=max_strategies,
            min_correlation=min_correlation,
            max_correlation=max_correlation,
            block_list=list(training_blocks),
            calendar_days=(training["date"].item(-1) - training["date"].item(0)).days
            + 1,
        )
        calculated = calculated.detach().clone()
        support = _allocation_support(calculated)
        if support not in seen_supports:
            seen_supports.add(support)
            supports.append(support)
        discovered_folds.append(
            _DiscoveredFold(
                window=window,
                training_start_date=training["date"].item(0),
                training_end_date=training["date"].item(-1),
                calculated_allocation=calculated,
                support=support,
                bootstrap_seed=bootstrap_seed,
                block_list=training_blocks,
            )
        )
    return tuple(discovered_folds), tuple(supports)


def _regime_target(
    support: _Support,
    discovered: _DiscoveredFold,
    training_returns: torch.Tensor,
    optimizer: _AllocationOptimizer,
) -> torch.Tensor:
    """Return one support's full-width target for a training fold."""
    if support == discovered.support:
        return discovered.calculated_allocation
    if not support:
        return torch.zeros_like(discovered.calculated_allocation)
    try:
        subset_allocation = optimizer(
            training_returns[:, list(support)],
            bootstrap_seed=discovered.bootstrap_seed,
            block_list=list(remap_block_list(discovered.block_list, support)),
            calendar_days=(
                discovered.training_end_date - discovered.training_start_date
            ).days
            + 1,
        )
    except ValueError as exc:
        raise ValueError(
            f"Walk-forward background support {support} failed for training "
            f"{discovered.training_start_date} through "
            f"{discovered.training_end_date}: {exc}"
        ) from exc
    full_allocation = torch.zeros_like(discovered.calculated_allocation)
    full_allocation[list(support)] = subset_allocation.to(
        device=full_allocation.device, dtype=full_allocation.dtype
    )
    return full_allocation


def _allocation_support(allocation: torch.Tensor) -> _Support:
    """Return the exact positive strategy indices defining an allocation regime."""
    indices = torch.nonzero(allocation > 0.0, as_tuple=False).flatten()
    return tuple(int(index) for index in indices.to(device="cpu").tolist())


def _fold_bootstrap_seeds(
    fold_count: int, bootstrap_on: bool
) -> tuple[int | None, ...]:
    """Draw compact per-fold seeds for reproducible counterfactual paths."""
    if not bootstrap_on:
        return (None,) * fold_count
    first_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    seed_modulus = torch.iinfo(torch.int64).max + 1
    return tuple(
        (first_seed + fold_index) % seed_modulus for fold_index in range(fold_count)
    )


def _validate_week_count(value: int | None, name: str, strictly_positive: bool) -> int:
    """Validate a whole-calendar-week duration."""
    if value is None or isinstance(value, bool):
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


def _validate_first_training_weeks(value: int | None, training_weeks: int) -> int:
    """Validate and resolve the optional longer initial training duration."""
    if value is None:
        return training_weeks
    first_training_weeks = _validate_week_count(
        value, "first_training_weeks", strictly_positive=True
    )
    if first_training_weeks < training_weeks:
        raise ValueError(
            "first_training_weeks must be greater than or equal to training_weeks"
        )
    return first_training_weeks


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
    first_training_weeks: int,
    embargo_weeks: int,
    simulation_weeks: int,
) -> tuple[_WalkForwardWindow, ...]:
    """Build every half-open rolling calendar window with a simulation row."""
    windows: list[_WalkForwardWindow] = []
    try:
        shift = timedelta(weeks=simulation_weeks)
        training_duration = timedelta(weeks=training_weeks)
        first_training_duration = timedelta(weeks=first_training_weeks)
        embargo_duration = timedelta(weeks=embargo_weeks)
        simulation_duration = timedelta(weeks=simulation_weeks)
        first_training_stop = aligned_start + first_training_duration
        simulation_start = first_training_stop + embargo_duration
    except OverflowError as exc:
        raise ValueError("Walk-forward week parameters are too large") from exc
    first_fold = True
    while True:
        if simulation_start > last_date:
            break
        if first_fold:
            training_start = aligned_start
            training_stop = first_training_stop
            first_fold = False
        else:
            training_stop = simulation_start - embargo_duration
            training_start = training_stop - training_duration
        windows.append(
            _WalkForwardWindow(
                training_start=training_start,
                training_stop=training_stop,
                simulation_start=simulation_start,
                simulation_stop=simulation_start + simulation_duration,
            )
        )
        simulation_start += shift
    return tuple(windows)


def _date_slice(frame: pl.DataFrame, start: date, stop: date) -> pl.DataFrame:
    """Return rows in a half-open calendar-date interval."""
    sliced = frame.filter((pl.col("date") >= start) & (pl.col("date") < stop))
    if sliced.is_empty():
        raise ValueError("Walk-forward period contains no NYSE market sessions")
    return sliced


def _date_slice_with_blocks(
    frame: pl.DataFrame,
    start: date,
    stop: date,
    block_list: tuple[Block, ...],
) -> tuple[pl.DataFrame, tuple[Block, ...]]:
    """Return one date slice and its zero-based block relationships."""
    index_name = "__global_allocation_day_idx"
    indexed = frame.with_row_index(index_name)
    sliced = indexed.filter((pl.col("date") >= start) & (pl.col("date") < stop))
    if sliced.is_empty():
        raise ValueError("Walk-forward period contains no NYSE market sessions")
    start_index = int(sliced[index_name].item(0))
    stop_index = int(sliced[index_name].item(-1)) + 1
    return sliced.drop(index_name), slice_block_list(
        block_list, start_index, stop_index
    )


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
    portfolio._print_optimization_algorithms(result.optimization_algorithm_counts)
    print(f"Folds: {len(result.folds)}")
    print(f"Period: {result.statistics.start_date} to {result.statistics.end_date}")
    print("Simulation-day-weighted average applied allocations:")
    weights = result.average_allocation.tolist()
    unused_strategy_names = []
    for strategy_name, weight in zip(result.strategy_names, weights):
        if weight <= 0.0:
            unused_strategy_names.append(strategy_name)
            continue
        print(f"  {strategy_name}: {portfolio._format_percentage(weight)}")
    print(f"  Total: {portfolio._format_percentage(sum(weights))}")
    if unused_strategy_names:
        print("Strategies not used:\n  " + "\n  ".join(unused_strategy_names))
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
    used_strategy_indices = (
        torch.nonzero(torch.any(applied_allocations > 0.0, dim=0), as_tuple=False)
        .flatten()
        .tolist()
    )
    applied_allocations = torch.cat(
        (applied_allocations, applied_allocations[-1:].clone()), dim=0
    )
    calculated_allocations = torch.cat(
        (calculated_allocations, calculated_allocations[-1:].clone()), dim=0
    )

    _, axis = plt.subplots(figsize=(11, 6))
    color_map = plt.get_cmap("turbo", max(1, len(used_strategy_indices)))
    for color_index, strategy_index in enumerate(used_strategy_indices):
        strategy_name = result.strategy_names[strategy_index]
        color = tuple(color_map(color_index))
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
    if used_strategy_indices:
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            title="Applied allocations\n(calculated: dashed)",
        )
    plt.tight_layout()
    plt.show()
