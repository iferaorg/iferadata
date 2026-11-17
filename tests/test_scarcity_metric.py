"""Scarcity wrappers preserve child scores and apply the requested frequency curve."""

import math
from unittest.mock import Mock

import pytest
import torch

import ifera.strategy_optimization as optimization
from ifera.strategy_optimization.metrics import with_scarcity_penalty

MIN_OBJECTIVE = 1e-6


def _constant_metric(returns):
    return returns.new_full((returns.shape[0],), 10.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frequency_curve_counts_positive_and_negative_trades(dtype):
    returns = torch.zeros(7, 8, dtype=dtype)
    for row, count in enumerate((0, 1, 2, 4, 6, 7, 8)):
        returns[row, :count] = torch.tensor(
            [1.0 if index % 2 == 0 else -1.0 for index in range(count)], dtype=dtype
        )
    original = returns.clone()
    metric = with_scarcity_penalty(_constant_metric, 0.25, 0.75)

    result = metric(returns)

    expected = returns.new_tensor(
        [
            MIN_OBJECTIVE,
            MIN_OBJECTIVE,
            10.0 * math.sqrt(MIN_OBJECTIVE),
            10.0 * math.sqrt(0.5 + 0.5 * MIN_OBJECTIVE),
            10.0,
            10.0,
            10.0,
        ]
    )
    torch.testing.assert_close(result, expected)
    assert result.shape == (7,)
    assert result.dtype == dtype
    assert result.device == returns.device
    torch.testing.assert_close(returns, original)


def test_child_runs_once_even_when_every_candidate_is_below_lower_threshold():
    returns = torch.zeros(2, 5)
    child_scores = torch.tensor([1e20, -100.0])
    child = Mock(return_value=child_scores)
    metric = with_scarcity_penalty(child, 0.2, 0.8)

    torch.testing.assert_close(metric(returns), torch.full((2,), MIN_OBJECTIVE))

    child.assert_called_once()
    assert child.call_args.args[0] is returns
    torch.testing.assert_close(child_scores, torch.tensor([1e20, -100.0]))


def test_zero_lower_and_unit_upper_include_both_endpoints():
    metric = with_scarcity_penalty(_constant_metric, 0, 1)
    returns = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    torch.testing.assert_close(
        metric(returns), torch.tensor([10.0 * math.sqrt(MIN_OBJECTIVE), 10.0])
    )


def test_narrow_threshold_interval_preserves_the_lower_boundary():
    metric = with_scarcity_penalty(_constant_metric, 0.5, math.nextafter(0.5, 1.0))
    returns = torch.tensor([[1.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
    torch.testing.assert_close(
        metric(returns), torch.tensor([10.0 * math.sqrt(MIN_OBJECTIVE), 10.0])
    )


def test_above_upper_preserves_child_values_exactly_and_retains_child_dtype():
    scores = torch.tensor(
        [1e-12, 0.0, -3.0, torch.finfo(torch.float64).max], dtype=torch.float64
    )
    child = Mock(return_value=scores)
    metric = with_scarcity_penalty(child, 0.2, 0.8)

    result = metric(torch.ones(4, 5, dtype=torch.float32))

    assert torch.equal(result, scores)
    assert result.dtype == scores.dtype


def test_projection_does_not_add_an_extra_objective_floor():
    returns = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    child = Mock(return_value=torch.tensor([MIN_OBJECTIVE, 0.0]))
    metric = with_scarcity_penalty(child, 0.0, 0.5)

    result = metric(returns)

    assert result[0].item() == pytest.approx(
        MIN_OBJECTIVE * math.sqrt(0.5 + 0.5 * MIN_OBJECTIVE)
    )
    assert 0.0 < result[0].item() < MIN_OBJECTIVE
    assert result[1].item() == 0.0


def test_scores_are_independent_of_batch_size_and_order():
    returns = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [2.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-2.0, -3.0, -4.0, -5.0],
        ]
    )

    def child(values):
        return 2.0 + values.square().sum(dim=1)

    metric = with_scarcity_penalty(child, 0.2, 0.8)
    together = metric(returns)
    separately = torch.cat([metric(returns[index : index + 1]) for index in range(4)])
    order = torch.tensor([2, 0, 3, 1])

    torch.testing.assert_close(together, separately)
    torch.testing.assert_close(metric(returns[order]), together[order])


def test_generated_metrics_capture_their_own_thresholds():
    returns = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    permissive = with_scarcity_penalty(_constant_metric, 0.0, 0.25)
    restrictive = with_scarcity_penalty(_constant_metric, 0.5, 1.0)

    assert permissive(returns).item() == 10.0
    assert restrictive(returns).item() == pytest.approx(MIN_OBJECTIVE)
    assert permissive(returns).item() == 10.0


def test_wrapper_is_available_from_public_package():
    assert optimization.with_scarcity_penalty is with_scarcity_penalty
    assert "with_scarcity_penalty" in optimization.__all__


@pytest.mark.parametrize(
    "lower,upper",
    [(-0.1, 0.5), (0.2, 1.1), (0.5, 0.5), (0.8, 0.2), (math.nan, 1), (0, math.inf)],
)
def test_invalid_numeric_thresholds_are_rejected(lower, upper):
    with pytest.raises(ValueError, match="threshold"):
        with_scarcity_penalty(_constant_metric, lower, upper)


@pytest.mark.parametrize("lower,upper", [(True, 1.0), (0.0, "1.0")])
def test_thresholds_must_be_real_non_boolean_numbers(lower, upper):
    with pytest.raises(TypeError, match="threshold"):
        with_scarcity_penalty(_constant_metric, lower, upper)


def test_child_metric_must_be_callable():
    with pytest.raises(TypeError, match="metric"):
        with_scarcity_penalty(None, 0.1, 0.5)


@pytest.mark.parametrize(
    "returns,error",
    [
        (torch.ones(2), ValueError),
        (torch.empty(2, 0), ValueError),
        (torch.tensor([[1, 2]]), TypeError),
        (torch.tensor([[math.nan]]), ValueError),
        (torch.tensor([[math.inf]]), ValueError),
    ],
)
def test_invalid_daily_return_inputs_are_rejected(returns, error):
    with pytest.raises(error):
        with_scarcity_penalty(_constant_metric, 0.1, 0.5)(returns)


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 2.0],
        torch.tensor([1, 2]),
        torch.ones(2, 1),
        torch.ones(3),
        torch.tensor([math.nan, 1.0]),
        torch.tensor([1.0, math.inf]),
    ],
)
def test_invalid_child_metric_scores_are_rejected(scores):
    metric = with_scarcity_penalty(Mock(return_value=scores), 0.1, 0.5)
    with pytest.raises(ValueError):
        metric(torch.ones(2, 4))


def test_child_metric_must_return_scores_on_the_input_device():
    scores = torch.empty(2, device="meta")
    metric = with_scarcity_penalty(Mock(return_value=scores), 0.1, 0.5)
    with pytest.raises(ValueError):
        metric(torch.ones(2, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_wrapper_matches_cpu_and_keeps_scores_on_cuda():
    returns = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0]]
    )
    metric = with_scarcity_penalty(_constant_metric, 0.25, 0.75)
    result = metric(returns.cuda())
    assert result.device.type == "cuda"
    torch.testing.assert_close(result.cpu(), metric(returns))
