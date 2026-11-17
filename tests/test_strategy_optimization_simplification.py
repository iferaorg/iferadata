"""Strategy winners use greedy simplification, rounded complexity, and histories."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import torch

from ifera.strategy_optimization import optimizer as optimizer_module
from ifera.strategy_optimization._data import StrategyData
from ifera.strategy_optimization._evaluation import OptimizationMode, StrategyTensors
from ifera.strategy_optimization._features import FeatureSpec
from ifera.strategy_optimization._output import format_conditions
from ifera.strategy_optimization._preparation import PreparedStrategy
from tests.test_strategy_optimization import strategy_market


def _prepared(*, granularity=None):
    features = tuple(
        FeatureSpec(name, "continuous", "any", "gap", granularity=granularity)
        for name in ("signal", "redundant")
    )
    tensors = StrategyTensors(
        entry_features=torch.tensor([[0.75, 0.75], [0.75, 0.75], [0.25, 0.75]]),
        exit_features=torch.empty(0, 2),
        entry_ignored=torch.zeros(3, 2, dtype=torch.bool),
        exit_ignored=torch.empty(0, 2, dtype=torch.bool),
        exit_trades=torch.empty(0, dtype=torch.long),
        exit_ror=torch.empty(0),
        original_ror=torch.tensor([5.0, 5.0, -0.1]),
        trade_days=torch.tensor([0, 2, 3]),
        market_days=5,
    )
    data = StrategyData(
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        tuple(date(2024, 1, day) for day in range(8, 13)),
        Path("simplification-test.csv"),
    )
    return PreparedStrategy(data, features, tensors, pl.DataFrame())


def _seed_population(monkeypatch, bounds, active):
    def initialize(_optimizer, population, _generator):
        population.bounds.copy_(bounds.expand_as(population.bounds))
        population.active.copy_(active.expand_as(population.active))

    monkeypatch.setattr(
        optimizer_module._StrategyDNFOptimizer, "_initialize", initialize
    )


@pytest.mark.parametrize("requested", [None, False, True])
def test_strategy_always_simplifies_even_if_compact_options_disable_it(
    monkeypatch, requested
):
    prepared = _prepared()
    bounds = torch.tensor([[[0.5, 1.0], [0.5, 1.0]]])
    active = torch.tensor([True])
    _seed_population(monkeypatch, bounds, active)
    parameters = {
        "device": "cpu",
        "max_conjunctions": 1,
        "max_population": 2,
        "generations": 0,
        "continuous_complexity_penalty": 0.2,
    }
    if requested is not None:
        parameters["simplify"] = requested
    original_parameters = parameters.copy()
    observed_returns = []

    def metric(returns):
        observed_returns.append(returns.clone())
        return returns.sum(dim=1) + 1.0

    result, evaluator, objectives, fitness = optimizer_module._evolve(
        prepared, OptimizationMode.ENTRY_FILTER, metric, parameters, 2, False
    )

    assert parameters == original_parameters
    assert result.objective_value == pytest.approx(10.9)
    assert result.objective_value < 11.0
    assert result.fitness > 11.0 / 1.4
    assert result.continuous_conditions == result.boolean_conditions == 0
    assert result.simplification_steps == 2
    assert result.simplification_evaluations >= result.simplification_steps
    assert result.evaluations == 2 + result.simplification_evaluations
    torch.testing.assert_close(result.bounds, torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))
    torch.testing.assert_close(result.active_conjunctions, active)
    assert objectives == (result.objective_value,)
    assert fitness == (result.fitness,)
    assert observed_returns[0].tolist() == [[5.0, 0.0, 5.0, 0.0, 0.0]] * 2
    evaluation = evaluator.evaluate(
        result.bounds[None], result.active_conjunctions[None]
    )
    assert evaluation.entered.tolist() == [[True, True, True]]
    assert (
        format_conditions(result.bounds, active, prepared.features, "entry_filter")
        == "Entry filter: TRUE"
    )


@pytest.mark.parametrize("generations", [0, 2])
def test_history_keeps_generation_length_with_simplified_final_values(
    monkeypatch, generations
):
    prepared = _prepared()
    bounds = torch.tensor([[[0.5, 1.0], [0.5, 1.0]]])
    _seed_population(monkeypatch, bounds, torch.tensor([True]))

    result, _evaluator, objectives, fitness = optimizer_module._evolve(
        prepared,
        OptimizationMode.ENTRY_FILTER,
        lambda values: values.sum(dim=1) + 1.0,
        {
            "device": "cpu",
            "max_conjunctions": 1,
            "max_population": 2,
            "generations": generations,
            "mutation_rate": 0.0,
            "conjunction_mutation_rate": 0.0,
            "crossover_rate": 0.0,
            "continuous_complexity_penalty": 0.2,
            "seed": 1729,
        },
        2,
        False,
    )

    assert len(objectives) == len(fitness) == generations + 1
    assert objectives[-1] == result.objective_value
    assert fitness[-1] == result.fitness
    assert result.simplification_steps == 2
    if generations:
        assert objectives[:-1] == (11.0,) * generations
        assert result.objective_value < objectives[-2]
        assert fitness[-1] > fitness[-2]


def test_rounded_unrestricted_atoms_are_not_simplification_neighbors(monkeypatch):
    prepared = _prepared(granularity=0.25)
    bounds = torch.tensor([[[0.001, 1.0], [0.001, 1.0]]])
    active = torch.tensor([True])
    _seed_population(monkeypatch, bounds, active)

    result, _evaluator, objectives, fitness = optimizer_module._evolve(
        prepared,
        OptimizationMode.ENTRY_FILTER,
        lambda values: values.sum(dim=1) + 1.0,
        {"device": "cpu", "max_conjunctions": 1, "max_population": 2, "generations": 0},
        2,
        False,
    )

    assert result.simplification_steps == result.simplification_evaluations == 0
    assert result.evaluations == 2
    assert result.continuous_conditions == 0
    assert objectives == (result.objective_value,)
    assert fitness == (result.fitness,)
    torch.testing.assert_close(result.bounds, torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))
    torch.testing.assert_close(result.active_conjunctions, active)


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_facade_exports_and_reports_the_simplified_winner(
    strategy_market, tmp_path, monkeypatch, capsys, mode
):
    data, market = strategy_market
    features = tmp_path / "features.yaml"
    features.write_text("features:\n  clock:\n    template: time\n", encoding="utf-8")
    clauses = 2 if mode == OptimizationMode.COMBINED else 1
    bounds = torch.tensor([[[0.25, 1.0]]]).repeat(clauses, 1, 1)
    _seed_population(monkeypatch, bounds, torch.ones(clauses, dtype=torch.bool))

    class OfflineSource:
        def load(self, _data, lookback_days=400):
            return market

    result = optimizer_module.optimize_strategy_from_csv(
        "sample",
        "strategy",
        mode,
        features_file=features,
        market_data_source=OfflineSource(),
        metric=lambda values: values.abs().sum(dim=1) + 1.0,
        optimizer_parameters={
            "device": "cpu",
            "max_population": 2,
            "generations": 0,
            "max_conjunctions": clauses,
            "continuous_complexity_penalty": 0.2,
            "simplify": False,
        },
        observation_batch_size=2,
        show_progress=False,
        show_plot=False,
    )

    assert result.optimization.simplification_steps == clauses
    assert result.optimization.continuous_conditions == 0
    assert result.optimization.active_conjunctions.tolist() == [True] * clauses
    assert "clock" not in result.conditions
    assert result.conditions.count("TRUE") == clauses
    assert result.conditions in capsys.readouterr().out
    assert result.objective_history == (result.optimization.objective_value,)
    assert result.fitness_history == (result.optimization.fitness,)
    assert result.trades.height == data.raw.height
    assert pl.read_csv(result.output_csv).equals(result.trades)
    if mode == OptimizationMode.ENTRY_FILTER:
        assert result.trades.equals(data.raw)
        assert result.daily_results["strategy_return"].to_list() == [0.25, None, -0.5]
    else:
        assert result.trades["Closed"].to_list() == [
            "2024-01-08 13:01:00",
            "2024-01-10 13:01:00",
        ]
        assert result.trades["Status"].to_list() == ["closed", "closed"]
