"""Optimize and report one CSV options strategy using evolutionary DNFs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import polars as pl
import torch
from matplotlib.figure import Figure
from torch import Tensor
from tqdm.auto import tqdm

from ifera.algorithms.evolutionary_dnf import (
    DNFOptimizationResult,
    EvolutionaryDNFOptimizer,
)
from ifera.allocation.portfolio_allocation import PortfolioStatistics
from ifera.settings import settings

from ._conditions import SavedConditions, save_conditions
from ._data import (
    MarketData,
    OracleMarketDataSource,
    StrategyData,
    load_strategy,
    validate_lookback_days,
)
from ._date_window import select_strategy_dates
from ._evaluation import OptimizationMode, StrategyEvaluator
from ._features import FeatureSpec, load_features
from ._output import format_conditions, save_strategy, winning_trades
from ._preparation import PreparedStrategy, prepare_strategy, validate_slippage
from ._reporting import (
    allocated_strategy_statistics,
    plot_objective_history,
    print_strategy_statistics,
)
from ._thresholds import ThresholdRounder
from .metrics import annualized_return

StrategyMetric = Callable[[Tensor], Tensor]


class _StrategyDNFOptimizer(EvolutionaryDNFOptimizer):
    """Penalize effective rounded predicates while evolving continuous genomes."""

    def __init__(self, features: tuple[FeatureSpec, ...], **parameters: Any) -> None:
        boolean_count = sum(feature.kind == "bool" for feature in features)
        super().__init__(
            num_boolean_features=boolean_count,
            num_continuous_features=len(features) - boolean_count,
            **parameters,
        )
        self._round_bounds = ThresholdRounder(features, self.device)

    def complexity(
        self, bounds: Tensor, active_conjunctions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Count restrictions after rounding, including newly unrestricted tails."""
        return super().complexity(self._round_bounds(bounds), active_conjunctions)

    def _simplification_bounds(self, bounds: Tensor) -> Tensor:
        """Simplify only restrictions that remain after threshold rounding."""
        return self._round_bounds(bounds)


class MarketDataSource(Protocol):
    """Allow offline data or a caller-owned SQLAlchemy source to be supplied."""

    def load(self, data: StrategyData, lookback_days: int = 400) -> MarketData:
        """Return underlying history and minute option quotes for these trades."""
        ...


@dataclass(frozen=True)
class StrategyOptimizationResult:
    """Winning trades, allocated daily performance, capital fraction, and progress."""

    mode: OptimizationMode
    optimization: DNFOptimizationResult
    features: tuple[FeatureSpec, ...]
    conditions: str
    trades: pl.DataFrame
    daily_results: pl.DataFrame
    statistics: PortfolioStatistics
    objective_history: tuple[float, ...]
    fitness_history: tuple[float, ...]
    output_csv: Path
    figure: Figure | None
    allocation: float = 1.0

    @property
    def conditions_yaml(self) -> Path:
        """Return the sidecar containing reproducible entry/exit predicates."""
        return self.output_csv.with_suffix(".conditions.yaml")


def _load_market(
    data: StrategyData,
    source: MarketDataSource | None,
    lookback_days: int,
    fill_daily_history: bool,
) -> MarketData:
    """Dispose only connections owned by this facade, including on failure."""
    if source is not None:
        return source.load(data, lookback_days=lookback_days)
    with OracleMarketDataSource(fill_daily_history=fill_daily_history) as oracle:
        return oracle.load(data, lookback_days=lookback_days)


def _evolve(
    prepared: PreparedStrategy,
    mode: OptimizationMode,
    metric: StrategyMetric,
    parameters: Mapping[str, Any] | None,
    observation_batch_size: int | None,
    show_progress: bool,
) -> tuple[
    DNFOptimizationResult, StrategyEvaluator, tuple[float, ...], tuple[float, ...]
]:
    """Collect generation statistics without retaining population storage."""
    options = dict(parameters or {})
    if {"num_boolean_features", "num_continuous_features"} & options.keys():
        raise ValueError("Feature counts are determined by the selected feature file")
    max_population = options.pop("max_population", 1024)
    generations = options.pop("generations", 100)
    options.setdefault("objective_batch_size", 32)
    options["simplify"] = True
    optimizer = _StrategyDNFOptimizer(prepared.features, **options)
    evaluator = StrategyEvaluator(
        prepared.tensors.to(optimizer.device),
        prepared.features,
        mode,
        observation_batch_size,
    )
    objective_history: list[float] = []
    fitness_history: list[float] = []

    def objective(bounds: Tensor, active: Tensor) -> Tensor:
        values = metric(evaluator.evaluate(bounds, active).daily_ror)
        if (
            not isinstance(values, Tensor)
            or values.shape != (bounds.shape[0],)
            or not values.is_floating_point()
            or values.device != bounds.device
        ):
            raise ValueError(
                "metric must return a floating (batch,) tensor on the input device"
            )
        if not torch.isfinite(values).all():
            raise ValueError("metric must return finite values")
        return values.clamp_min(1e-6)

    with tqdm(
        total=generations, desc="Strategy optimization", disable=not show_progress
    ) as progress:

        def observe(
            generation: int,
            _bounds: Tensor,
            _active: Tensor,
            objectives: Tensor,
            fitness: Tensor,
        ) -> None:
            objective_history.append(float(objectives.max().item()))
            fitness_history.append(float(fitness.max().item()))
            if generation:
                progress.update(1)
            progress.set_postfix(
                objective=objective_history[-1], fitness=fitness_history[-1]
            )

        optimizer.set_evaluation_callback(observe)
        result = optimizer.optimize(
            objective, max_population=max_population, generations=generations
        )
        objective_history[-1] = result.objective_value
        fitness_history[-1] = result.fitness
        progress.set_postfix(objective=result.objective_value, fitness=result.fitness)
    result = replace(result, bounds=evaluator.round_bounds(result.bounds))
    return result, evaluator, tuple(objective_history), tuple(fitness_history)


def optimize_strategy_from_csv(
    portfolio_name: str,
    strategy_name: str,
    mode: OptimizationMode | str = OptimizationMode.ENTRY_FILTER,
    *,
    optimizer_parameters: Mapping[str, Any] | None = None,
    features_file: Path | str | None = None,
    metric: StrategyMetric = annualized_return,
    exit_slippage: float = 0.0,
    output_csv: Path | str | None = None,
    market_data_source: MarketDataSource | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    lookback_days: int = 400,
    observation_batch_size: int | None = None,
    show_progress: bool = True,
    show_plot: bool = True,
    print_report: bool = True,
) -> StrategyOptimizationResult:
    """Evolve an entry filter, early exit rule, or both for one 0DTE strategy.

    Reads ``DATA_FOLDER/results/portfolios/{portfolio_name}/{strategy_name}.csv``.
    ``optimizer_parameters`` accepts EvolutionaryDNFOptimizer constructor options
    except feature counts, plus ``generations`` and ``max_population``. Its device
    defaults to CUDA when available. Final greedy simplification is always
    enabled, including when options request ``simplify=False``. Combined mode
    assigns the first ceil(C/2)
    conjunction slots to entry and the remainder to exits; irrelevant categories
    are ignored within each phase. Empty/inactive DNFs evaluate false.

    ``metric`` receives percentage RoR shaped (candidates, all market days), with
    zero for no-trade days. Slippage is an adverse number of option price points
    applied once to the complete combination, independent of its number of legs.
    Earlier exits are evaluated strictly after entry and before the original
    close. Original closes preserve their recorded P/L, prices, and status.
    Optional inclusive ``start_date``/``end_date`` restrict the input calendar
    explicitly, for example when Oracle covers only part of the backtest.

    The final report allocates capital to the winner using alpha=0, a 0.001 grid
    increment, historical ADD_limit=-0.1, no bootstrap, and no refinement. The
    returned daily results/statistics use this allocation; ``allocation`` stores
    the chosen fraction. Metric inputs and exported trades retain trade-level RoR.

    The default export is outside the source portfolio, under
    ``DATA_FOLDER/results/strategy_optimization/{portfolio_name}``. Callers may
    supply a data source for reproducible offline evaluation. The returned graph
    is available when ``show_plot=True``; histories are always retained. Their
    final entries describe the simplified winner, with ``generations + 1`` values.
    """
    selected_mode = OptimizationMode(mode)
    validate_slippage(exit_slippage)
    if not callable(metric):
        raise TypeError("metric must be callable")
    validate_lookback_days(lookback_days)
    features = tuple(
        feature
        for feature in load_features(features_file)
        if selected_mode == OptimizationMode.COMBINED
        or feature.category
        in (
            "any",
            "entry" if selected_mode == OptimizationMode.ENTRY_FILTER else "exit",
        )
    )
    data = select_strategy_dates(
        load_strategy(portfolio_name, strategy_name), start_date, end_date
    )
    destination = (
        Path(output_csv)
        if output_csv is not None
        else Path(settings.DATA_FOLDER)
        / "results"
        / "strategy_optimization"
        / portfolio_name
        / f"{data.path.stem}_{selected_mode.value}.csv"
    )
    if destination.resolve() == data.path.resolve():
        raise ValueError("output_csv must differ from the source backtest CSV")
    if (
        features_file is not None
        and destination.resolve() == Path(features_file).resolve()
    ):
        raise ValueError("output_csv must differ from the input feature YAML")
    conditions_path = destination.with_suffix(".conditions.yaml")
    protected = (data.path, destination) + (
        (Path(features_file),) if features_file is not None else ()
    )
    if conditions_path.resolve() in {path.resolve() for path in protected}:
        raise ValueError("condition YAML must differ from source and output files")
    market = _load_market(
        data,
        market_data_source,
        lookback_days,
        any(feature.needs_daily_history for feature in features),
    )
    prepared = prepare_strategy(data, market, features, selected_mode, exit_slippage)
    best, evaluator, objectives, fitness = _evolve(
        prepared,
        selected_mode,
        metric,
        optimizer_parameters,
        observation_batch_size,
        show_progress,
    )
    evaluation = evaluator.evaluate(best.bounds[None], best.active_conjunctions[None])
    entered = evaluation.entered[0]
    active = torch.zeros(len(data.dates), dtype=torch.bool, device=entered.device)
    active.scatter_(0, evaluator.tensors.trade_days, entered)
    daily, statistics, allocation = allocated_strategy_statistics(
        data.dates, evaluation.daily_ror[0], active
    )
    trades = winning_trades(prepared, evaluation)
    conditions = format_conditions(
        best.bounds, best.active_conjunctions, features, selected_mode
    )
    save_conditions(
        SavedConditions(
            mode=selected_mode,
            features=features,
            bounds=best.bounds,
            active=best.active_conjunctions,
            exit_slippage=float(exit_slippage),
            lookback_days=lookback_days,
            start_date=data.dates[0],
            end_date=data.dates[-1],
        ),
        conditions_path,
        protected_paths=protected,
    )
    output = save_strategy(trades, destination, data.path)
    figure = plot_objective_history(objectives) if show_plot else None
    if print_report:
        print(f"Strategy: {portfolio_name}/{data.path.stem}")
        print(conditions)
        print(f"Capital allocation: {allocation:.2%}")
        print_strategy_statistics(statistics)
        print(f"Optimized trades saved to: {output}")
        print(f"Winning conditions saved to: {conditions_path}")
    if show_plot:
        plt.show()
    return StrategyOptimizationResult(
        selected_mode,
        best,
        features,
        conditions,
        trades,
        daily,
        statistics,
        objectives,
        fitness,
        output,
        figure,
        allocation,
    )
