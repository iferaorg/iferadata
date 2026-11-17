"""Batched objective values use percent units and stable downside semantics."""

import math

import pytest
import torch

from ifera.strategy_optimization.metrics import (
    annualized_return,
    return_over_add,
    sortino,
)


@pytest.mark.parametrize("metric", [annualized_return, sortino, return_over_add])
def test_metric_zero_days_and_negative_objectives(metric):
    returns = torch.tensor([[0.0, 0.0], [-2.0, -1.0]], dtype=torch.float64)
    original = returns.clone()
    result = metric(returns)
    assert result.shape == (2,)
    assert result.dtype == returns.dtype
    assert result.device == returns.device
    assert result.tolist() == pytest.approx([1.0, 1e-6])
    torch.testing.assert_close(returns, original)


def test_annualized_return_includes_nontrading_days():
    returns = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.01, 0.01, 0.0, 0.0]])
    torch.testing.assert_close(
        annualized_return(returns, annualization_periods=4),
        torch.tensor([4.0, 1.02]),
    )


def test_sortino_uses_all_market_days_for_downside_rms():
    returns = torch.tensor([[2.0, -1.0, 0.0, 0.0]], dtype=torch.float64)
    # Annual numerator is 1%, annual downside RMS is sqrt(1/4)*sqrt(4)=1%.
    assert sortino(returns, annualization_periods=4).item() == pytest.approx(2.0)


@pytest.mark.parametrize("metric", [sortino, return_over_add])
def test_negative_numerator_is_not_divided(metric):
    returns = torch.tensor([[-0.1, 0.05]], dtype=torch.float64)
    assert metric(returns, annualization_periods=2).item() == pytest.approx(0.95)


def test_add_uses_fractional_compounding_and_percent_denominator():
    returns = torch.tensor([[20.0, -10.0, 10.0]], dtype=torch.float64)
    # Equity 1.2, 1.08, 1.188; drawdown 0%, -10%, -1%.
    add_percent = math.sqrt((10.0**2 + 1.0**2) / 3.0)
    result = return_over_add(returns, annualization_periods=3)
    assert result.item() == pytest.approx(20.0 / add_percent + 1.0)


def test_add_treats_total_loss_as_absorbing_bankruptcy():
    returns = torch.tensor([[-100.0, 200.0], [-125.0, 200.0]], dtype=torch.float64)
    result = return_over_add(returns, annualization_periods=2)
    torch.testing.assert_close(result, torch.tensor([2.0, 1.75], dtype=torch.float64))


@pytest.mark.parametrize("metric", [sortino, return_over_add])
def test_zero_denominator_remains_finite(metric):
    result = metric(torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert torch.isfinite(result).all()
    assert (result > 0.0).all()
    assert result[0] > 1.0
    assert result[1] == 1.0


@pytest.mark.parametrize("metric", [annualized_return, sortino, return_over_add])
@pytest.mark.parametrize(
    "returns,error",
    [
        (torch.ones(2), ValueError),
        (torch.empty(2, 0), ValueError),
        (torch.tensor([[1, 2]]), TypeError),
        (torch.tensor([[float("nan")]]), ValueError),
        (torch.tensor([[float("inf")]]), ValueError),
    ],
)
def test_invalid_metric_input(metric, returns, error):
    with pytest.raises(error):
        metric(returns)


@pytest.mark.parametrize("metric", [annualized_return, sortino, return_over_add])
@pytest.mark.parametrize("periods", [0, -1, True, 2.5])
def test_invalid_annualization(metric, periods):
    with pytest.raises(ValueError, match="positive integer"):
        metric(torch.ones(2, 3), annualization_periods=periods)


@pytest.mark.parametrize("metric", [annualized_return, sortino, return_over_add])
def test_finite_input_overflow_still_has_finite_objective(metric):
    returns = torch.tensor([[torch.finfo(torch.float32).max]])
    result = metric(returns)
    assert torch.isfinite(result).all()
    assert (result > 0.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("metric", [annualized_return, sortino, return_over_add])
def test_cuda_metric_matches_cpu(metric):
    returns = torch.tensor([[2.0, -1.0, 0.0], [-0.1, 0.05, 0.0]])
    torch.testing.assert_close(metric(returns.cuda()).cpu(), metric(returns))
