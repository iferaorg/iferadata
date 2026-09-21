"""Tests for spatial ADD search inside one fixed positive support."""

from __future__ import annotations

import pytest
import torch

from ifera.allocation import _add_branch_and_bound


def _search(
    transposed_returns: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    add_limit: float = -1.0,
):
    """Run a small exact-support search without a grid-cost estimate."""
    return _add_branch_and_bound.try_exact_support_add_branch_and_bound(
        transposed_returns=transposed_returns,
        root_lower=lower,
        root_upper=upper,
        max_total_allocation=1.0,
        add_limit=add_limit,
        maximal_cliques=[(0,)] if lower.numel() == 1 else [],
        bootstrap_indices=None,
        percentile=10.0,
        allocation_tolerance=1e-4,
        grid_steps=None,
    )


def test_exact_support_retains_a_negative_growth_lower_corner():
    """Cash must not displace the best point in a required-positive support."""
    returns = torch.tensor([[-0.1, -0.1]], dtype=torch.float64)
    lower = torch.tensor([0.2], dtype=torch.float64)
    upper = torch.tensor([0.8], dtype=torch.float64)

    outcome = _search(returns, lower, upper)

    assert outcome.completed
    assert outcome.allocation is not None
    assert outcome.incumbent_allocation is not None
    torch.testing.assert_close(outcome.allocation, lower)
    assert outcome.incumbent_mean_log_growth < 0.0


def test_exact_support_can_complete_without_an_add_feasible_point():
    """An infeasible fixed support returns no fabricated cash incumbent."""
    returns = torch.tensor([[-0.5]], dtype=torch.float64)
    point = torch.tensor([1.0], dtype=torch.float64)

    outcome = _search(returns, point, point, add_limit=-0.1)

    assert outcome.completed
    assert outcome.allocation is None
    assert outcome.incumbent_allocation is None
    assert outcome.incumbent_mean_log_growth == -torch.inf
    assert outcome.bootstrap_average_drawdown is None


def test_exact_support_rejects_mismatched_root_bounds():
    """Root tensors must describe the supplied allocation space exactly."""
    returns = torch.tensor([[0.1, -0.1]], dtype=torch.float64)

    with pytest.raises(ValueError, match="root_lower.*shape"):
        _search(
            returns,
            torch.tensor([0.1, 0.2], dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )


def test_exact_support_rejects_a_reversed_root_interval():
    """Every exact-support lower coordinate must be at most its upper bound."""
    returns = torch.tensor([[0.1, -0.1]], dtype=torch.float64)

    with pytest.raises(ValueError, match="root_lower must not exceed root_upper"):
        _search(
            returns,
            torch.tensor([0.6], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
        )


def test_exact_support_rejects_a_coordinate_that_can_cross_zero():
    """Every variable coordinate needs a positive floor to keep blocking fixed."""
    returns = torch.tensor([[0.1, -0.1]], dtype=torch.float64)

    with pytest.raises(ValueError, match="fixed at zero or have a positive lower"):
        _search(
            returns,
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
        )


def test_exact_support_grid_estimate_ignores_fixed_zero_coordinates():
    """The pilot compares its cost with the support grid, not the full dimension."""
    upper = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    estimate = _add_branch_and_bound._estimate_exact_support_grid_candidates(
        upper,
        maximum=1.0,
        maximal_cliques=[(0, 1, 2, 3)],
        grid_steps=10,
    )

    assert estimate == 11
