"""Raw-unit threshold grids apply to DNF decisions, objectives, and reporting."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import torch

from ifera.algorithms.evolutionary_dnf import DNFOptimizationResult
from ifera.strategy_optimization import optimizer as optimizer_module
from ifera.strategy_optimization._data import StrategyData
from ifera.strategy_optimization._evaluation import (
    OptimizationMode,
    StrategyEvaluator,
    StrategyTensors,
)
from ifera.strategy_optimization._features import FeatureSpec, load_features
from ifera.strategy_optimization._output import format_conditions
from ifera.strategy_optimization._preparation import PreparedStrategy
from ifera.strategy_optimization._thresholds import round_bounds
from ifera.strategy_optimization.optimizer import _evolve


def _feature(*, minimum=0.0, maximum=1.0, granularity=0.25):
    return FeatureSpec(
        "change",
        "continuous",
        "any",
        "gap",
        minimum,
        maximum,
        granularity=granularity,
    )


def _observations(features):
    tensors = StrategyTensors(
        entry_features=torch.tensor([[0.5], [0.25]]),
        exit_features=torch.tensor([[0.5], [0.75], [0.5], [0.75]]),
        entry_ignored=torch.zeros(2, 1, dtype=torch.bool),
        exit_ignored=torch.zeros(4, 1, dtype=torch.bool),
        exit_trades=torch.tensor([0, 0, 1, 1]),
        exit_ror=torch.tensor([10.0, 20.0, 30.0, 40.0]),
        original_ror=torch.tensor([5.0, -5.0]),
        trade_days=torch.tensor([0, 2]),
        market_days=3,
    )
    data = StrategyData(
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        (date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)),
        Path("granularity-test.csv"),
    )
    return PreparedStrategy(data, features, tensors, pl.DataFrame())


def _candidate(mode, *, lower=0.51):
    clauses = 2 if mode == OptimizationMode.COMBINED else 1
    bounds = torch.tensor([lower, 1.0]).repeat(1, clauses, 1, 1)
    return bounds, torch.ones(1, clauses, dtype=torch.bool)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_round_bounds_preserves_boolean_and_unconfigured_features(batched, dtype):
    features = (
        FeatureSpec("monday", "bool", "any", "is_day", parameters={"day": "Monday"}),
        _feature(minimum=-1.0, maximum=1.0),
        _feature(minimum=-1.0, maximum=1.0, granularity=None),
    )
    bounds = torch.tensor(
        [
            [[1.0, 1.0], [0.18, 0.91], [0.18, 0.91]],
            [[0.0, 1.0], [0.6, 0.74], [0.0, 1.0]],
        ],
        dtype=dtype,
    )
    expected = bounds.clone()
    expected[:, 1] = torch.tensor([[0.125, 0.875], [0.625, 0.75]], dtype=dtype)
    if batched:
        bounds, expected = bounds.unsqueeze(0).repeat(2, 1, 1, 1), expected.unsqueeze(
            0
        ).repeat(2, 1, 1, 1)
    original = bounds.clone()

    actual = round_bounds(bounds, features)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(bounds, original, rtol=0, atol=0)
    assert actual.shape == bounds.shape
    assert actual.dtype == dtype
    assert actual.device == bounds.device
    assert torch.equal(actual[..., 0, :], bounds[..., 0, :])
    assert torch.equal(actual[..., 2, :], bounds[..., 2, :])


def test_grid_is_zero_anchored_and_rounds_negative_values():
    features = (_feature(minimum=-0.7, maximum=0.9),)
    bounds = torch.tensor([[[0.0625, 0.8125]]], dtype=torch.float64)
    # Raw values -0.6 and 0.6 project to -0.5 and 0.5, independently of min=-0.7.
    expected = torch.tensor([[[0.125, 0.75]]], dtype=torch.float64)

    torch.testing.assert_close(round_bounds(bounds, features), expected)


def test_grid_uses_torch_round_ties_to_even():
    features = (_feature(minimum=-2.0, maximum=2.0, granularity=1.0),)
    bounds = torch.tensor([[[0.125, 0.375]], [[0.625, 0.875]]], dtype=torch.float64)
    # Raw pairs (-1.5, -0.5), (0.5, 1.5) round to (-2, 0), (0, 2).
    expected = torch.tensor([[[0.0, 0.5]], [[0.5, 1.0]]], dtype=torch.float64)

    torch.testing.assert_close(round_bounds(bounds, features), expected, rtol=0, atol=0)


def test_rounded_values_are_clamped_to_off_grid_feature_range():
    features = (_feature(minimum=0.2, maximum=0.8, granularity=1.0),)
    bounds = torch.tensor([[[0.01, 0.99]]], dtype=torch.float64)
    # Raw 0.206 and 0.794 project to 0 and 1, then clamp to the feature's range.
    torch.testing.assert_close(
        round_bounds(bounds, features),
        torch.tensor([[[0.0, 1.0]]], dtype=torch.float64),
    )


def test_unrestricted_endpoint_sentinels_remain_exact_for_off_grid_range():
    features = (_feature(minimum=0.2, maximum=0.8),)
    bounds = torch.tensor(
        [[[0.0, 1.0]], [[0.0, 0.6]], [[0.4, 1.0]]], dtype=torch.float64
    )
    expected = torch.tensor(
        [[[0.0, 1.0]], [[0.0, 0.5]], [[0.5, 1.0]]], dtype=torch.float64
    )

    actual = round_bounds(bounds, features)

    torch.testing.assert_close(actual, expected)
    assert actual[0, 0, 0] == 0.0
    assert actual[0, 0, 1] == 1.0
    assert actual[1, 0, 0] == 0.0
    assert actual[2, 0, 1] == 1.0
    torch.testing.assert_close(round_bounds(actual, features), actual, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,expected,exit_indices",
    [
        (OptimizationMode.ENTRY_FILTER, [5.0, 0.0, 0.0], [-1, -1]),
        (OptimizationMode.EARLY_EXIT, [10.0, 0.0, 30.0], [0, 2]),
        (OptimizationMode.COMBINED, [10.0, 0.0, 0.0], [0, -1]),
    ],
)
def test_all_evaluation_modes_use_projected_thresholds(mode, expected, exit_indices):
    prepared = _observations((_feature(),))
    bounds, active = _candidate(mode)
    original = bounds.clone()

    actual = StrategyEvaluator(prepared.tensors, prepared.features, mode).evaluate(
        bounds, active
    )

    assert actual.daily_ror.tolist() == [expected]
    assert actual.exit_indices.tolist() == [exit_indices]
    torch.testing.assert_close(bounds, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,expected",
    [
        (OptimizationMode.ENTRY_FILTER, [5.0, 0.0, 0.0]),
        (OptimizationMode.EARLY_EXIT, [10.0, 0.0, 30.0]),
        (OptimizationMode.COMBINED, [10.0, 0.0, 0.0]),
    ],
)
def test_evolutionary_objective_scores_projected_conditions(
    monkeypatch, mode, expected
):
    prepared = _observations((_feature(),))
    bounds, active = _candidate(mode)
    seen = []

    def metric(returns):
        seen.append(returns.clone())
        return returns.sum(dim=1) + 1.0

    def scripted_optimize(optimizer, objective, max_population, generations):
        assert max_population == 2
        assert generations == 0
        values = objective(bounds, active)
        scores = optimizer.fitness(values, bounds, active)
        if optimizer._evaluation_callback is not None:
            optimizer._evaluation_callback(0, bounds, active, values.double(), scores)
        boolean, continuous = optimizer.complexity(bounds, active)
        return DNFOptimizationResult(
            bounds=bounds[0].clone(),
            active_conjunctions=active[0].clone(),
            objective_value=float(values[0]),
            fitness=float(scores[0]),
            boolean_conditions=int(boolean[0]),
            continuous_conditions=int(continuous[0]),
            population_size=1,
            generations=0,
            evaluations=1,
        )

    monkeypatch.setattr(
        optimizer_module._StrategyDNFOptimizer, "optimize", scripted_optimize
    )
    result, evaluator, objective_history, _fitness_history = _evolve(
        prepared,
        mode,
        metric,
        {"device": "cpu", "max_population": 2, "generations": 0},
        2,
        False,
    )

    assert len(seen) == 1
    assert seen[0].tolist() == [expected]
    assert result.objective_value == sum(expected) + 1.0
    assert objective_history == (sum(expected) + 1.0,)
    torch.testing.assert_close(
        result.bounds, round_bounds(bounds[0], prepared.features), rtol=0, atol=0
    )
    final = evaluator.evaluate(result.bounds[None], result.active_conjunctions[None])
    assert final.daily_ror.tolist() == [expected]
    printed = format_conditions(
        result.bounds, result.active_conjunctions, prepared.features, mode
    )
    assert "0.51" not in printed
    assert "change >= 0.5" in printed


def test_strategy_optimizer_penalizes_projected_complexity_without_mutation():
    features = (
        FeatureSpec("monday", "bool", "any", "is_day", parameters={"day": "Monday"}),
        _feature(),
    )
    optimizer = optimizer_module._StrategyDNFOptimizer(
        features, device="cpu", max_conjunctions=1
    )
    bounds = torch.tensor([[[[1.0, 1.0], [0.0001, 1.0]]], [[[1.0, 1.0], [0.0, 1.0]]]])
    original = bounds.clone()
    active = torch.ones(2, 1, dtype=torch.bool)

    boolean, continuous = optimizer.complexity(bounds, active)
    fitness = optimizer.fitness(torch.tensor([5.0, 5.0]), bounds, active)

    assert boolean.tolist() == [1, 1]
    assert continuous.tolist() == [0, 0]
    torch.testing.assert_close(
        fitness, torch.full((2,), 5.0 / 1.01, dtype=torch.float64)
    )
    torch.testing.assert_close(bounds, original, rtol=0, atol=0)


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_real_evolution_returns_consistent_projected_winner(mode):
    prepared = _observations((_feature(),))

    def metric(returns):
        return returns.sum(dim=1) + 20.0

    result, evaluator, objective_history, fitness_history = _evolve(
        prepared,
        mode,
        metric,
        {
            "device": "cpu",
            "seed": 1729,
            "max_population": 24,
            "generations": 3,
            "max_conjunctions": 3,
            "objective_batch_size": 8,
        },
        2,
        False,
    )

    projected = round_bounds(result.bounds, prepared.features)
    torch.testing.assert_close(projected, result.bounds, rtol=0, atol=0)
    bounds = result.bounds[None]
    active = result.active_conjunctions[None]
    reevaluated = metric(evaluator.evaluate(bounds, active).daily_ror)
    optimizer = optimizer_module._StrategyDNFOptimizer(
        prepared.features, device="cpu", max_conjunctions=3
    )
    boolean, continuous = optimizer.complexity(bounds, active)
    fitness = optimizer.fitness(reevaluated, bounds, active)

    assert float(reevaluated[0]) == result.objective_value
    assert int(boolean[0]) == result.boolean_conditions
    assert int(continuous[0]) == result.continuous_conditions
    assert float(fitness[0]) == result.fitness
    assert len(objective_history) == len(fitness_history) == 4
    assert fitness_history[-1] == result.fitness
    assert result.generations == 3
    assert result.evaluations > 24


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_printed_conditions_use_rounded_values(mode):
    bounds, active = _candidate(mode)
    phase = "Early exit" if mode == OptimizationMode.EARLY_EXIT else "Entry filter"
    expected = f"{phase}: (change >= 0.5)"
    if mode == OptimizationMode.COMBINED:
        expected += "\nEarly exit: (change >= 0.5)"

    assert format_conditions(bounds[0], active[0], (_feature(),), mode) == expected


def test_printout_omits_conditions_rounded_to_unrestricted():
    bounds = torch.tensor([[[0.01, 0.99]]])
    assert (
        format_conditions(bounds, torch.tensor([True]), (_feature(),), "entry_filter")
        == "Entry filter: TRUE"
    )


def test_default_time_grid_prints_and_evaluates_whole_minutes():
    feature = next(feature for feature in load_features() if feature.template == "time")
    assert feature.granularity == 60.0
    prepared = _observations((feature,))
    threshold_seconds = 13 * 3600 + 14 * 60
    prepared.tensors.entry_features[:, 0] = torch.tensor(
        [threshold_seconds / 86400, (threshold_seconds - 1) / 86400]
    )
    bounds = torch.tensor([[[[(threshold_seconds + 23) / 86400, 1.0]]]])
    active = torch.tensor([[True]])

    actual = StrategyEvaluator(prepared.tensors, (feature,), "entry_filter").evaluate(
        bounds, active
    )

    assert actual.entered.tolist() == [[True, False]]
    assert (
        format_conditions(bounds[0], active[0], (feature,), "entry_filter")
        == "Entry filter: (time >= 13:14:00)"
    )


def test_fractional_time_grid_keeps_fractional_seconds_in_printout():
    feature = FeatureSpec(
        "clock", "continuous", "any", "time", 0, 86400, granularity=0.25
    )
    raw = 13 * 3600 + 14 * 60 + 12.34
    bounds = torch.tensor([[[raw / 86400, 1.0]]])

    assert (
        format_conditions(bounds, torch.tensor([True]), (feature,), "entry_filter")
        == "Entry filter: (clock >= 13:14:12.25)"
    )


def test_fine_grid_prints_full_precision_for_large_raw_values():
    feature = _feature(minimum=1_000_000, maximum=1_000_100, granularity=0.01)
    normalized = (1_000_001.234 - feature.min_value) / (
        feature.max_value - feature.min_value
    )
    bounds = torch.tensor([[[normalized, 1.0]]])

    assert (
        format_conditions(bounds, torch.tensor([True]), (feature,), "entry_filter")
        == "Entry filter: (change >= 1000001.23)"
    )


def test_off_grid_clipped_endpoint_keeps_range_precision_in_printout():
    feature = _feature(minimum=1.3, maximum=9.7, granularity=10.0)
    bounds = torch.tensor([[[0.0, 0.01]]])

    assert (
        format_conditions(bounds, torch.tensor([True]), (feature,), "entry_filter")
        == "Entry filter: (change <= 1.3)"
    )


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_unrestricted_nan_feature_still_passes_after_rounding(mode):
    prepared = _observations((_feature(minimum=0.2, maximum=0.8),))
    prepared.tensors.entry_features.fill_(torch.nan)
    prepared.tensors.exit_features.fill_(torch.nan)
    bounds, active = _candidate(mode, lower=0.0)

    actual = StrategyEvaluator(prepared.tensors, prepared.features, mode).evaluate(
        bounds, active
    )

    assert actual.entered.tolist() == [[True, True]]
    expected = (
        [5.0, 0.0, -5.0] if mode == OptimizationMode.ENTRY_FILTER else [10.0, 0.0, 30.0]
    )
    assert actual.daily_ror.tolist() == [expected]


@pytest.mark.parametrize("endpoint,comparison", [(0, ">="), (1, "<=")])
def test_true_one_cent_boundary_passes_and_prints_exact_grid_value(
    endpoint, comparison
):
    feature = _feature(minimum=-5.0, maximum=5.0, granularity=0.01)
    prepared = _observations((feature,))
    normalized_one_cent = (0.01 - feature.min_value) / (
        feature.max_value - feature.min_value
    )
    prepared.tensors.entry_features.fill_(normalized_one_cent)
    bounds = torch.tensor([[[[0.0, 1.0]]]])
    bounds[0, 0, 0, endpoint] = (0.014 + 5.0) / 10.0
    active = torch.tensor([[True]])

    projected = round_bounds(bounds, (feature,))
    actual = StrategyEvaluator(prepared.tensors, (feature,), "entry_filter").evaluate(
        bounds, active
    )

    assert projected[0, 0, 0, endpoint] == prepared.tensors.entry_features[0, 0]
    assert actual.entered.tolist() == [[True, True]]
    assert (
        format_conditions(bounds[0], active[0], (feature,), "entry_filter")
        == f"Entry filter: (change {comparison} 0.01)"
    )
    torch.testing.assert_close(
        round_bounds(projected, (feature,)), projected, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_projection_matches_cpu_without_mutation(dtype):
    features = (_feature(minimum=-0.7, maximum=0.9),)
    bounds = torch.tensor(
        [[[[0.0, 1.0]], [[0.0625, 0.8125]]], [[[0.0, 0.8125]], [[0.0625, 1.0]]]],
        dtype=dtype,
    )
    resident = bounds.cuda()
    original = resident.clone()

    projected = round_bounds(resident, features)

    torch.testing.assert_close(projected.cpu(), round_bounds(bounds, features))
    torch.testing.assert_close(resident, original, rtol=0, atol=0)
    assert projected.device == resident.device
    assert projected.dtype == dtype
