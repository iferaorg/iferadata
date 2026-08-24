"""Tests for the continuous Kelly optimizer."""

import math

import pytest
import torch

from ifera.allocation._kelly_optimizer import maximize_kelly_growth


def _mean_log_growth(
    transposed_returns: torch.Tensor, allocation: torch.Tensor
) -> float:
    """Evaluate normalized log growth independently from production code."""
    portfolio_returns = allocation @ transposed_returns
    if bool(torch.any(portfolio_returns <= -1.0)):
        return -math.inf
    return float(torch.log1p(portfolio_returns).mean())


def _dense_simplex_oracle(
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    steps: int = 1_000,
) -> float:
    """Return a dense two-strategy simplex lower bound on the optimum."""
    units = torch.arange(steps + 1, dtype=torch.float64)
    first, second = torch.meshgrid(units, units, indexing="ij")
    allocations = torch.stack((first.flatten(), second.flatten()), dim=1) / steps
    allocations *= max_total_allocation
    allocations = allocations[allocations.sum(dim=1) <= max_total_allocation]
    portfolio_returns = allocations @ transposed_returns.to(dtype=torch.float64)
    valid = torch.amin(portfolio_returns, dim=1) > -1.0
    scores = torch.full((allocations.shape[0],), -torch.inf, dtype=torch.float64)
    scores[valid] = torch.log1p(portfolio_returns[valid]).mean(dim=1)
    return float(torch.amax(scores))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_simplex_solver_finds_analytic_fraction_and_preserves_dtype(dtype):
    """The two-outcome scalar Kelly fraction is one half."""
    transposed_returns = torch.tensor([[1.0, -0.5]], dtype=dtype)

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="simplex",
        tolerance=1e-8 if dtype == torch.float64 else 1e-6,
    )

    expected = torch.tensor([0.5], dtype=dtype)
    torch.testing.assert_close(result.allocation, expected, rtol=2e-5, atol=2e-6)
    assert result.allocation.dtype == dtype
    assert result.allocation.device == transposed_returns.device
    assert result.converged
    assert result.frank_wolfe_gap <= (1e-8 if dtype == torch.float64 else 1e-6)
    assert result.mean_log_growth == pytest.approx(
        _mean_log_growth(transposed_returns, result.allocation)
    )


def test_box_solver_optimizes_disjoint_strategies_independently():
    """A box allows each disjoint strategy to use its own full capital budget."""
    transposed_returns = torch.tensor(
        [
            [1.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, -0.5],
        ],
        dtype=torch.float64,
    )

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="box",
        tolerance=1e-10,
    )

    torch.testing.assert_close(
        result.allocation,
        torch.tensor([0.5, 0.75], dtype=torch.float64),
        rtol=1e-7,
        atol=1e-8,
    )
    assert result.converged
    assert result.frank_wolfe_gap <= 1e-10


@pytest.mark.parametrize("feasible_set", ["simplex", "box"])
def test_allowed_mask_excludes_strategies_exactly(feasible_set):
    """A branch-and-bound exclusion mask leaves exact zeros in the result."""
    transposed_returns = torch.tensor(
        [
            [1.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, -0.5],
        ],
        dtype=torch.float64,
    )

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set=feasible_set,
        allowed_mask=torch.tensor([True, False]),
        tolerance=1e-10,
    )

    torch.testing.assert_close(
        result.allocation,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
        rtol=1e-7,
        atol=1e-8,
    )
    assert result.allocation[1].item() == 0.0


def test_all_losing_strategies_return_exact_zero_without_an_update():
    """Cash is the exact optimum when every initial directional gain is negative."""
    transposed_returns = torch.tensor(
        [[-0.1, -0.2, -0.3], [-0.4, -0.1, -0.2]], dtype=torch.float64
    )

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="simplex",
        tolerance=1e-10,
    )

    assert torch.equal(result.allocation, torch.zeros(2, dtype=torch.float64))
    assert result.mean_log_growth == 0.0
    assert result.frank_wolfe_gap == 0.0
    assert result.upper_bound == 0.0
    assert result.iterations == 0
    assert result.converged


def test_away_step_removes_a_stale_strategy_to_an_exact_zero():
    """An atom favored at cash can be fully removed at the portfolio optimum."""
    transposed_returns = torch.tensor(
        [
            [-0.4256, 1.1833, 0.3096, 0.1228, 1.2113, 0.8532, 0.3578, -0.4596],
            [-0.6483, 0.1451, 1.4889, 0.2521, 1.2144, 0.9488, 0.5964, -0.7700],
            [-0.5427, -0.2836, 1.3180, 0.0044, 0.2256, 0.0743, -0.6622, -0.4223],
        ],
        dtype=torch.float64,
    )
    assert transposed_returns.mean(dim=1).argmax().item() == 1

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="simplex",
        tolerance=1e-10,
    )

    assert torch.equal(
        result.allocation, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert result.iterations == 2
    assert result.converged


def test_line_search_stays_solvent_when_linear_oracle_vertex_is_bankrupt():
    """The solver stops before a bankrupt atom and finds the interior optimum."""
    transposed_returns = torch.tensor([[4.0, -2.0]], dtype=torch.float64)

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="simplex",
        tolerance=1e-10,
    )

    torch.testing.assert_close(
        result.allocation,
        torch.tensor([0.125], dtype=torch.float64),
        rtol=1e-7,
        atol=1e-9,
    )
    assert bool(torch.all(1.0 + result.allocation @ transposed_returns > 0.0))
    assert math.isfinite(result.mean_log_growth)


def test_cleanup_preserves_a_small_but_resolvable_interior_optimum():
    """Exact-zero cleanup must not invalidate the FW convergence certificate."""
    gain = 0.20
    optimum = 5e-7
    loss = gain / (1.0 + 2.0 * gain * optimum)
    transposed_returns = torch.tensor([[gain, -loss]], dtype=torch.float32)

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="simplex",
    )

    assert result.converged
    assert 0.0 < result.allocation.item() < 2e-6
    assert result.frank_wolfe_gap >= 0.0


def test_float32_line_search_can_retreat_multiple_dtype_steps_from_insolvency():
    """A rounded zero at the first nextafter point must not stop optimization."""
    transposed_returns = (
        0.5 * torch.randn((5, 10), generator=torch.Generator().manual_seed(0))
    ).clamp(-0.999, 3.0)

    result = maximize_kelly_growth(
        transposed_returns,
        2.0,
        feasible_set="simplex",
    )

    assert result.converged
    assert result.frank_wolfe_gap <= 1e-6
    assert bool(torch.all(1.0 + result.allocation @ transposed_returns > 0.0))


def test_unfinished_solver_still_returns_a_valid_frank_wolfe_upper_bound():
    """The FW gap bounds the unknown optimum even before convergence."""
    transposed_returns = torch.tensor(
        [
            [0.8, -0.2, -0.1, 0.4, 0.1],
            [-0.4, 0.9, 0.1, -0.2, 0.3],
        ],
        dtype=torch.float64,
    )
    dense_lower_bound = _dense_simplex_oracle(transposed_returns, 0.8)

    result = maximize_kelly_growth(
        transposed_returns,
        0.8,
        feasible_set="simplex",
        tolerance=1e-14,
        max_iterations=1,
    )

    assert not result.converged
    assert result.frank_wolfe_gap >= 0.0
    assert result.mean_log_growth <= dense_lower_bound + 1e-6
    assert result.upper_bound >= dense_lower_bound - 1e-12
    assert result.upper_bound == pytest.approx(
        result.mean_log_growth + result.frank_wolfe_gap
    )


def test_zero_budget_and_empty_allowed_set_return_cash():
    """Degenerate feasible sets do not enter the iterative solver."""
    transposed_returns = torch.tensor([[0.5, -0.1], [0.4, -0.2]])

    zero_budget = maximize_kelly_growth(transposed_returns, 0.0, feasible_set="simplex")
    no_allowed = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="box",
        allowed_mask=torch.tensor([False, False]),
    )

    assert torch.equal(zero_budget.allocation, torch.zeros(2))
    assert torch.equal(no_allowed.allocation, torch.zeros(2))
    assert zero_budget.converged and no_allowed.converged


def test_allocation_tolerance_is_invariant_to_a_tiny_objective_scale():
    """Small returns must not turn an allocation threshold into a loose gap."""
    transposed_returns = torch.full((1, 252), 1e-6, dtype=torch.float64)

    result = maximize_kelly_growth(
        transposed_returns,
        1.0,
        feasible_set="simplex",
        allocation_tolerance=0.01,
    )

    torch.testing.assert_close(
        result.allocation,
        torch.ones(1, dtype=torch.float64),
        rtol=0.0,
        atol=0.01,
    )
    assert result.converged


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"feasible_set": "triangle"}, ValueError, "feasible_set"),
        ({"feasible_set": "simplex", "tolerance": 0.0}, ValueError, "tolerance"),
        (
            {"feasible_set": "simplex", "max_iterations": True},
            TypeError,
            "max_iterations",
        ),
    ],
)
def test_solver_parameter_validation(kwargs, exception, message):
    """Internal solver controls reject ambiguous or unusable values."""
    transposed_returns = torch.tensor([[0.1, -0.1]])

    with pytest.raises(exception, match=message):
        maximize_kelly_growth(transposed_returns, 1.0, **kwargs)
