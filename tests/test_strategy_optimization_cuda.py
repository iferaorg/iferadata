"""CUDA predicate evaluation agrees with the eager CPU strategy interpreter."""

from dataclasses import replace

import pytest
import torch

from ifera.strategy_optimization._evaluation import (
    OptimizationMode,
    StrategyEvaluator,
    StrategyTensors,
)
from ifera.strategy_optimization._features import FeatureSpec

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


@pytest.fixture
def ragged_observations():
    """Five trades have zero, one, three, two, and one possible exit minutes."""
    features = (
        FeatureSpec("weekday", "bool", "any", "is_day", parameters={"day": "Monday"}),
        FeatureSpec("premium", "continuous", "entry", "position_premium"),
        FeatureSpec("return", "continuous", "exit", "ror_pct"),
        FeatureSpec(
            "delta", "continuous", "any", "leg_delta", parameters={"leg": "short_put"}
        ),
    )
    missing = float("nan")
    tensors = StrategyTensors(
        entry_features=torch.tensor(
            [
                [1.0, 0.2, missing, missing],
                [0.0, 0.8, missing, 0.3],
                [1.0, 0.5, missing, missing],
                [0.0, 0.1, missing, 0.9],
                [1.0, 0.9, missing, missing],
            ]
        ),
        exit_features=torch.tensor(
            [
                [0.0, missing, 0.2, 0.3],
                [1.0, missing, 0.1, missing],
                [1.0, missing, 0.6, missing],
                [1.0, missing, 0.9, missing],
                [0.0, missing, missing, 0.9],
                [0.0, missing, 0.9, 0.9],
                [1.0, missing, 0.7, missing],
            ]
        ),
        entry_ignored=torch.tensor(
            [
                [False, False, False, True],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, True],
            ]
        ),
        exit_ignored=torch.tensor([[False] * 4] * 6 + [[False, False, False, True]]),
        exit_trades=torch.tensor([1, 2, 2, 2, 3, 3, 4]),
        exit_ror=torch.tensor([15.0, -25.0, -10.0, 5.0, 8.0, 20.0, -2.0]),
        original_ror=torch.tensor([9.0, -30.0, 0.0, 12.0, -5.0]),
        trade_days=torch.tensor([0, 2, 3, 5, 7]),
        market_days=8,
    )
    return features, tensors


def _population(device):
    """Keep conjunction slices strided after the CPU-to-device transfer."""
    storage = torch.zeros(5, 6, 4, 2)
    storage[..., 1] = 1.0
    bounds = storage[:, ::2]
    enabled = torch.zeros(5, 6, dtype=torch.bool)
    active = enabled[:, ::2]

    active[1, [0, 2]] = True
    bounds[1, 0, 0] = torch.tensor([1.0, 1.0])
    bounds[1, 0, 3] = torch.tensor([0.2, 0.6])
    bounds[1, 2, 2] = torch.tensor([0.55, 1.0])
    bounds[1, 2, 3] = torch.tensor([0.2, 0.6])

    active[2] = True

    active[3] = True
    bounds[3, 0, 0] = torch.tensor([1.0, 1.0])
    bounds[3, 0, 1] = torch.tensor([0.4, 1.0])
    bounds[3, :2, 2] = torch.tensor([1.0, 1.0])
    bounds[3, 1, 0] = torch.tensor([0.0, 0.0])
    bounds[3, 2, 1] = torch.tensor([0.7, 1.0])
    bounds[3, 2, 2] = torch.tensor([0.8, 1.0])

    active[4, 1] = True
    bounds[4, 1, 0] = torch.tensor([0.0, 0.0])
    return storage.to(device)[:, ::2], enabled.to(device)[:, ::2]


def _assert_equivalent(actual, expected):
    """Predicate decisions and gathered returns should agree exactly."""
    for name in ("entered", "exit_indices", "daily_ror"):
        observed = getattr(actual, name)
        assert observed.device.type == "cuda"
        torch.testing.assert_close(
            observed.cpu(), getattr(expected, name), rtol=0, atol=0
        )


@pytest.mark.parametrize("device, expected", [("cpu", 512), ("cuda", 32768)])
def test_observation_batch_defaults_follow_device(
    ragged_observations, device, expected
):
    features, tensors = ragged_observations
    evaluator = StrategyEvaluator(
        tensors.to(torch.device(device)), features, "combined"
    )
    assert evaluator.observation_batch_size == expected


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_cuda_matches_eager_for_strided_odd_clauses_and_chunk_tails(
    ragged_observations, mode
):
    features, tensors = ragged_observations
    cpu_bounds, cpu_active = _population("cpu")
    bounds, active = _population("cuda")
    assert not bounds.is_contiguous()
    assert not active.is_contiguous()
    expected = StrategyEvaluator(tensors, features, mode, 3).evaluate(
        cpu_bounds, cpu_active
    )
    evaluator = StrategyEvaluator(tensors.to(torch.device("cuda")), features, mode, 3)
    actual = evaluator.evaluate(bounds, active)
    _assert_equivalent(actual, expected)
    for start, stop in ((0, 3), (3, 5)):
        subset = evaluator.evaluate(bounds[start:stop], active[start:stop])
        for name in ("entered", "exit_indices", "daily_ror"):
            torch.testing.assert_close(
                getattr(subset, name), getattr(actual, name)[start:stop], rtol=0, atol=0
            )
    assert (actual.daily_ror[:, [1, 4, 6]] == 0).all()


def test_cuda_ignores_absent_legs_but_rejects_unknown_existing_leg_delta(
    ragged_observations,
):
    features, tensors = ragged_observations
    bounds = torch.zeros(1, 1, 4, 2)
    bounds[..., 1] = 1.0
    bounds[0, 0, 3] = torch.tensor([0.2, 0.6])
    active = torch.ones(1, 1, dtype=torch.bool)
    expected = StrategyEvaluator(tensors, features, "entry_filter", 3).evaluate(
        bounds, active
    )
    actual = StrategyEvaluator(
        tensors.to(torch.device("cuda")), features, "entry_filter", 3
    ).evaluate(bounds.cuda(), active.cuda())
    _assert_equivalent(actual, expected)
    assert actual.entered.cpu().tolist() == [[True, True, False, False, True]]


def test_cuda_missing_indicator_cannot_satisfy_a_restricted_exit(ragged_observations):
    features, tensors = ragged_observations
    bounds = torch.zeros(1, 1, 4, 2)
    bounds[..., 1] = 1.0
    bounds[0, 0, 2] = torch.tensor([0.5, 1.0])
    active = torch.ones(1, 1, dtype=torch.bool)
    expected = StrategyEvaluator(tensors, features, "early_exit", 3).evaluate(
        bounds, active
    )
    actual = StrategyEvaluator(
        tensors.to(torch.device("cuda")), features, "early_exit", 3
    ).evaluate(bounds.cuda(), active.cuda())
    _assert_equivalent(actual, expected)
    assert actual.exit_indices.cpu().tolist() == [[-1, -1, 2, 5, 6]]


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_cuda_empty_exit_observations_preserve_original_trade_closes(
    ragged_observations, mode
):
    features, tensors = ragged_observations
    tensors = replace(
        tensors,
        exit_features=tensors.exit_features[:0],
        exit_ignored=tensors.exit_ignored[:0],
        exit_trades=tensors.exit_trades[:0],
        exit_ror=tensors.exit_ror[:0],
    )
    cpu_bounds, cpu_active = _population("cpu")
    bounds, active = _population("cuda")
    expected = StrategyEvaluator(tensors, features, mode, 3).evaluate(
        cpu_bounds, cpu_active
    )
    actual = StrategyEvaluator(
        tensors.to(torch.device("cuda")), features, mode, 3
    ).evaluate(bounds, active)
    _assert_equivalent(actual, expected)
    assert (actual.exit_indices == -1).all()


def test_cuda_combined_single_clause_has_no_early_exit_slots(ragged_observations):
    features, tensors = ragged_observations
    bounds = torch.zeros(2, 1, 4, 2)
    bounds[..., 1] = 1.0
    active = torch.tensor([[True], [False]])
    expected = StrategyEvaluator(tensors, features, "combined", 3).evaluate(
        bounds, active
    )
    actual = StrategyEvaluator(
        tensors.to(torch.device("cuda")), features, "combined", 3
    ).evaluate(bounds.cuda(), active.cuda())
    _assert_equivalent(actual, expected)
    assert (actual.exit_indices == -1).all()


def test_cuda_exit_reduction_supports_interleaved_chronological_trade_groups(
    ragged_observations,
):
    features, tensors = ragged_observations
    order = torch.tensor([1, 0, 4, 2, 6, 5, 3])
    tensors = replace(
        tensors,
        exit_features=tensors.exit_features[order],
        exit_ignored=tensors.exit_ignored[order],
        exit_trades=tensors.exit_trades[order],
        exit_ror=tensors.exit_ror[order],
    )
    cpu_bounds, cpu_active = _population("cpu")
    bounds, active = _population("cuda")
    expected = StrategyEvaluator(tensors, features, "early_exit", 3).evaluate(
        cpu_bounds, cpu_active
    )
    actual = StrategyEvaluator(
        tensors.to(torch.device("cuda")), features, "early_exit", 3
    ).evaluate(bounds, active)
    _assert_equivalent(actual, expected)


def test_cuda_zero_features_preserve_active_clause_semantics(ragged_observations):
    _, tensors = ragged_observations
    tensors = replace(
        tensors,
        entry_features=tensors.entry_features[:, :0],
        exit_features=tensors.exit_features[:, :0],
        entry_ignored=tensors.entry_ignored[:, :0],
        exit_ignored=tensors.exit_ignored[:, :0],
    )
    bounds = torch.empty(2, 3, 0, 2)
    active = torch.tensor([[True, False, True], [False, False, False]])
    expected = StrategyEvaluator(tensors, (), "combined", 3).evaluate(bounds, active)
    actual = StrategyEvaluator(
        tensors.to(torch.device("cuda")), (), "combined", 3
    ).evaluate(bounds.cuda(), active.cuda())
    _assert_equivalent(actual, expected)
