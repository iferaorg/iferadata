"""Focused edge-case tests for hard-ADD Kelly search shortcuts."""

from __future__ import annotations

import math

import pytest
import torch

from ifera.allocation._add_constrained_kelly import (
    SCALAR_ALGORITHM,
    try_add_constrained_kelly,
    validate_approximate_add_search,
)
from ifera.allocation._strategy_selection import StrategySelectionConstraints
from ifera.allocation.capital_allocation import find_optimal_capital_allocation


def _historical_average_drawdown(
    returns: torch.Tensor, allocation: torch.Tensor
) -> float:
    """Calculate historical ADD using the allocation's reported precision."""
    portfolio_returns = returns @ allocation.to(dtype=returns.dtype)
    log_equity = torch.log1p(portfolio_returns).cumsum(dim=0)
    running_peak = torch.cummax(log_equity, dim=0).values.clamp_min(0.0)
    drawdowns = torch.expm1(log_equity - running_peak)
    return -float(torch.sqrt(torch.mean(drawdowns.square())).item())


def _empty_conflicts() -> torch.Tensor:
    """Return an empty prior-conflict index tensor."""
    return torch.empty(0, dtype=torch.int64)


def test_complete_conflict_graph_uses_exact_singleton_search():
    """Pair conflicts alone can reduce every feasible support to one strategy."""
    transposed_returns = torch.tensor(
        [
            [0.30, -0.20, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.50, -0.15, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.40, -0.20],
        ],
        dtype=torch.float64,
    )
    constraints = StrategySelectionConstraints(
        maximum_active=3,
        prior_conflicts=(
            _empty_conflicts(),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
        ),
    )

    result = try_add_constrained_kelly(
        transposed_returns,
        raw_allocation=None,
        max_total_allocation=1.0,
        add_limit=-0.05,
        selection_constraints=constraints,
        bootstrap_indices=None,
        percentile=10.0,
        approximate_add_search=False,
    )

    assert result is not None
    assert result.exact
    assert result.algorithm == SCALAR_ALGORITHM
    assert torch.count_nonzero(result.allocation) == 1
    assert result.allocation[1] > 0.0


def test_zero_and_negative_growth_strategies_select_cash():
    """Cash beats zero, consistently losing, and volatility-drag strategies."""
    transposed_returns = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.10, -0.20, -0.05, -0.10],
            [0.10, -0.10, 0.10, -0.10],
        ],
        dtype=torch.float64,
    )
    constraints = StrategySelectionConstraints(
        maximum_active=1,
        prior_conflicts=(_empty_conflicts(),) * 3,
    )

    result = try_add_constrained_kelly(
        transposed_returns,
        raw_allocation=None,
        max_total_allocation=1.0,
        add_limit=-0.10,
        selection_constraints=constraints,
        bootstrap_indices=None,
        percentile=10.0,
        approximate_add_search=False,
    )

    assert result is not None
    assert result.exact
    assert torch.equal(result.allocation, torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize(
    ("add_limit", "expected_fraction"),
    [(-1.0, 5.0 / 6.0), (0.0, 0.0)],
    ids=["least_restrictive", "no_drawdown"],
)
def test_add_limit_endpoints(add_limit: float, expected_fraction: float):
    """Both valid ADD-limit endpoints retain their inclusive semantics."""
    transposed_returns = torch.tensor([[0.30, -0.20]], dtype=torch.float64)

    result = try_add_constrained_kelly(
        transposed_returns,
        raw_allocation=None,
        max_total_allocation=1.0,
        add_limit=add_limit,
        selection_constraints=None,
        bootstrap_indices=None,
        percentile=10.0,
        approximate_add_search=False,
    )

    assert result is not None
    expected = torch.tensor([expected_fraction], dtype=torch.float64)
    torch.testing.assert_close(result.allocation, expected, rtol=0.0, atol=2e-9)


@pytest.mark.parametrize("value", [0, 1, None, "true", torch.tensor(False)])
def test_approximate_add_search_requires_a_bool(value: object):
    """Truth-like values must not silently enable an approximate optimizer."""
    with pytest.raises(TypeError, match="approximate_add_search"):
        validate_approximate_add_search(value)  # type: ignore[arg-type]


def test_low_precision_output_preserves_the_hard_add_limit():
    """Casting the continuous boundary must not round into infeasibility."""
    returns = torch.tensor([[0.30], [-0.20]], dtype=torch.float64)
    exact_boundary = 0.3539
    add_limit = -(exact_boundary * 0.20 / math.sqrt(2.0))

    allocation = find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        dtype=torch.float16,
        ADD_limit=add_limit,
    )

    assert allocation.dtype == torch.float16
    assert _historical_average_drawdown(returns, allocation) >= add_limit
