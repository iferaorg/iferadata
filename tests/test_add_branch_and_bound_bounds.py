"""Focused correctness tests for spatial ADD bounds and search safeguards."""

from __future__ import annotations

import pytest
import torch

from ifera.allocation._add_branch_and_bound import (
    BRANCH_AND_BOUND_ALGORITHM,
    allocation_box_drawdown_bounds,
    tighten_allocation_box,
    try_add_branch_and_bound,
)
from ifera.allocation._capital_allocation_drawdown import (
    bootstrap_average_drawdown,
    negative_rms_drawdown,
)
from ifera.allocation._strategy_selection import StrategySelectionConstraints


def _box_returns() -> torch.Tensor:
    """Return two strategies with nontrivial peaks and troughs on every ray."""
    return torch.tensor(
        [
            [0.10, -0.10, 0.05, -0.08, 0.04, -0.06],
            [0.08, -0.12, 0.04, -0.05, 0.02, -0.07],
        ],
        dtype=torch.float64,
    )


def _box_allocations(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Sample corners and interior points throughout a two-dimensional box."""
    fractions = torch.linspace(0.0, 1.0, 7, dtype=lower.dtype)
    coordinates = torch.cartesian_prod(fractions, fractions)
    return lower + coordinates * (upper - lower)


def _historical_drawdowns(
    allocations: torch.Tensor, transposed_returns: torch.Tensor
) -> torch.Tensor:
    """Evaluate production historical ADD for allocation rows."""
    portfolio_returns = allocations @ transposed_returns
    return negative_rms_drawdown(portfolio_returns.log1p(), dim=1)


def _bootstrap_indices() -> torch.Tensor:
    """Return deterministic whole-day resamples with varied path orderings."""
    return torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [4, 5, 0, 1, 2, 3],
            [1, 3, 5, 0, 2, 4],
            [0, 2, 1, 5, 4, 3],
            [2, 3, 4, 5, 0, 1],
            [5, 4, 3, 2, 1, 0],
        ],
        dtype=torch.int64,
    )


def _empty_conflicts(strategy_count: int) -> tuple[torch.Tensor, ...]:
    """Return a compact conflict representation with no forbidden pairs."""
    return tuple(torch.empty(0, dtype=torch.int64) for _ in range(strategy_count))


def test_historical_box_bounds_contain_all_sampled_allocations():
    """Historical interval bounds contain corners and box-interior ADD values."""
    transposed_returns = _box_returns()
    lower = torch.tensor([0.20, 0.30], dtype=torch.float64)
    upper = torch.tensor([0.35, 0.50], dtype=torch.float64)

    lower_bound, upper_bound = allocation_box_drawdown_bounds(
        lower,
        upper,
        transposed_returns,
        bootstrap_indices=None,
        percentile=10.0,
    )
    actual = _historical_drawdowns(_box_allocations(lower, upper), transposed_returns)

    assert lower_bound <= actual.amin()
    assert actual.amax() <= upper_bound
    assert lower_bound < upper_bound < 0.0


def test_bootstrap_box_bounds_contain_fixed_path_percentiles():
    """Quantile bounds contain every allocation under the same bootstrap paths."""
    transposed_returns = _box_returns()
    lower = torch.tensor([0.20, 0.30], dtype=torch.float64)
    upper = torch.tensor([0.35, 0.50], dtype=torch.float64)
    indices = _bootstrap_indices()
    percentile = 25.0

    lower_bound, upper_bound = allocation_box_drawdown_bounds(
        lower,
        upper,
        transposed_returns,
        bootstrap_indices=indices,
        percentile=percentile,
    )
    allocations = _box_allocations(lower, upper)
    actual = bootstrap_average_drawdown(
        allocations @ transposed_returns,
        indices,
        percentile,
    )

    assert lower_bound <= actual.amin()
    assert actual.amax() <= upper_bound
    assert lower_bound < upper_bound < 0.0


def test_degenerate_box_bounds_converge_on_its_exact_drawdown():
    """A zero-width box differs from exact ADD only by rounding protection."""
    transposed_returns = _box_returns()
    allocation = torch.tensor([0.27, 0.41], dtype=torch.float64)

    lower_bound, upper_bound = allocation_box_drawdown_bounds(
        allocation,
        allocation,
        transposed_returns,
        bootstrap_indices=None,
        percentile=10.0,
    )
    exact = _historical_drawdowns(allocation.unsqueeze(0), transposed_returns)[0]

    assert lower_bound <= exact <= upper_bound
    assert float((upper_bound - lower_bound).item()) < 1e-12


def test_clique_tightening_projects_coordinate_ranges_and_rejects_empty_box():
    """Clique budgets tighten each coordinate and detect infeasible lower bounds."""
    lower = torch.tensor([0.30, 0.20, 0.10], dtype=torch.float64)
    upper = torch.tensor([0.90, 0.80, 0.90], dtype=torch.float64)

    tightened = tighten_allocation_box(
        lower,
        upper,
        maximal_cliques=[(0, 1), (1, 2)],
        max_total_allocation=0.75,
    )

    assert tightened is not None
    tightened_lower, tightened_upper = tightened
    torch.testing.assert_close(tightened_lower, lower)
    torch.testing.assert_close(
        tightened_upper,
        torch.tensor([0.55, 0.45, 0.55], dtype=torch.float64),
    )
    assert (
        tighten_allocation_box(
            torch.tensor([0.50, 0.30], dtype=torch.float64),
            torch.tensor([0.70, 0.60], dtype=torch.float64),
            maximal_cliques=[(0, 1)],
            max_total_allocation=0.75,
        )
        is None
    )


def test_support_enumeration_cap_requests_grid_fallback_without_searching_nodes():
    """Too many legal supports fail cheaply into the complete grid search."""
    strategy_count = 10
    transposed_returns = torch.linspace(
        -0.02,
        0.04,
        strategy_count * 4,
        dtype=torch.float64,
    ).reshape(strategy_count, 4)
    constraints = StrategySelectionConstraints(
        maximum_active=4,
        prior_conflicts=_empty_conflicts(strategy_count),
    )

    outcome = try_add_branch_and_bound(
        transposed_returns=transposed_returns,
        raw_allocation=None,
        max_total_allocation=1.0,
        add_limit=-0.20,
        maximal_cliques=[tuple(range(strategy_count))],
        selection_constraints=constraints,
        bootstrap_indices=None,
        percentile=10.0,
        allocation_tolerance=0.01,
        grid_steps=None,
    )

    assert outcome.allocation is None
    assert not outcome.completed
    assert outcome.pilot_nodes == 0
    assert torch.equal(
        outcome.incumbent_allocation,
        torch.zeros(strategy_count, dtype=torch.float64),
    )


def test_large_bootstrap_workload_skips_pilot_before_drawdown_evaluation():
    """A serial pilot cannot compete with the batched grid at WFA-scale work."""
    transposed_returns = _box_returns()
    bootstrap_indices = torch.empty((4096, 1024), device="meta", dtype=torch.int64)

    def unexpected_bootstrap(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("an oversized bootstrap pilot must not evaluate ADD")

    outcome = try_add_branch_and_bound(
        transposed_returns=transposed_returns,
        raw_allocation=torch.tensor([0.4, 0.4], dtype=torch.float64),
        max_total_allocation=1.0,
        add_limit=-0.05,
        maximal_cliques=[(0, 1)],
        selection_constraints=None,
        bootstrap_indices=bootstrap_indices,
        percentile=10.0,
        allocation_tolerance=0.01,
        grid_steps=100,
        bootstrap_drawdown_calculator=unexpected_bootstrap,
    )

    assert outcome.allocation is None
    assert outcome.pilot_nodes == 0
    assert torch.equal(
        outcome.incumbent_allocation, torch.zeros(2, dtype=torch.float64)
    )


def test_invalid_raw_allocation_cannot_become_or_prune_against_an_incumbent():
    """The internal seam does not trust a caller-supplied Kelly allocation."""
    transposed_returns = _box_returns()

    outcome = try_add_branch_and_bound(
        transposed_returns=transposed_returns,
        raw_allocation=torch.tensor([1.0, 1.0], dtype=torch.float64),
        max_total_allocation=1.0,
        add_limit=-0.20,
        maximal_cliques=[(0, 1)],
        selection_constraints=None,
        bootstrap_indices=None,
        percentile=10.0,
        allocation_tolerance=0.05,
        grid_steps=None,
    )

    assert outcome.incumbent_allocation.sum() <= 1.0


def test_small_historical_search_completes_near_dense_grid_optimum():
    """A promising two-strategy search completes with an accurate feasible result."""
    transposed_returns = torch.tensor(
        [
            [0.47, -0.25, 0.17, 0.08],
            [0.06, 0.34, 0.01, -0.10],
        ],
        dtype=torch.float64,
    )
    add_limit = -0.027

    outcome = try_add_branch_and_bound(
        transposed_returns=transposed_returns,
        raw_allocation=None,
        max_total_allocation=1.0,
        add_limit=add_limit,
        maximal_cliques=[(0, 1)],
        selection_constraints=None,
        bootstrap_indices=None,
        percentile=10.0,
        allocation_tolerance=0.01,
        grid_steps=None,
    )

    assert outcome.completed
    assert outcome.algorithm == BRANCH_AND_BOUND_ALGORITHM
    assert outcome.allocation is not None
    assert outcome.pilot_nodes > 0
    allocation = outcome.allocation
    actual_add = _historical_drawdowns(allocation.unsqueeze(0), transposed_returns)[0]
    assert actual_add >= add_limit
    assert allocation.sum() <= 1.0

    fractions = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
    dense_allocations = torch.cartesian_prod(fractions, fractions)
    dense_allocations = dense_allocations[dense_allocations.sum(dim=1) <= 1.0 + 1e-12]
    dense_add = _historical_drawdowns(dense_allocations, transposed_returns)
    dense_growth = torch.log1p(dense_allocations @ transposed_returns).mean(dim=1)
    dense_growth.masked_fill_(dense_add < add_limit, -torch.inf)
    dense_winner = dense_allocations[torch.argmax(dense_growth)]

    torch.testing.assert_close(allocation, dense_winner, rtol=0.0, atol=0.01)


def test_small_bootstrap_search_reuses_paths_and_returns_the_retained_add():
    """A manageable bootstrap workload can complete without recomputing paths."""
    transposed_returns = torch.tensor(
        [
            [0.47, -0.25, 0.17, 0.08],
            [0.06, 0.34, 0.01, -0.10],
        ],
        dtype=torch.float64,
    )
    indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [0, 1, 0, 1],
            [3, 2, 1, 0],
            [2, 3, 2, 3],
            [0, 2, 1, 3],
            [1, 3, 0, 2],
            [2, 0, 3, 1],
        ],
        dtype=torch.int64,
    )
    add_limit = -0.027

    outcome = try_add_branch_and_bound(
        transposed_returns=transposed_returns,
        raw_allocation=None,
        max_total_allocation=1.0,
        add_limit=add_limit,
        maximal_cliques=[(0, 1)],
        selection_constraints=None,
        bootstrap_indices=indices,
        percentile=10.0,
        allocation_tolerance=0.02,
        grid_steps=None,
    )

    assert outcome.completed
    assert outcome.allocation is not None
    assert outcome.bootstrap_average_drawdown is not None
    exact = bootstrap_average_drawdown(
        (outcome.allocation @ transposed_returns).unsqueeze(0),
        indices,
        10.0,
    )[0]
    torch.testing.assert_close(outcome.bootstrap_average_drawdown, exact)
    assert exact >= add_limit


def test_completed_search_obeys_support_conflicts_and_disjoint_capital_reuse():
    """Spatial roots cover legal supports while independent risk reuses capital."""
    transposed_returns = torch.tensor(
        [
            [0.30, -0.20, 0.00, 0.00],
            [0.00, 0.00, 0.32, -0.20],
            [0.00, 0.00, 0.25, -0.18],
        ],
        dtype=torch.float64,
    )
    constraints = StrategySelectionConstraints(
        maximum_active=2,
        prior_conflicts=(
            torch.empty(0, dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        ),
    )

    outcome = try_add_branch_and_bound(
        transposed_returns=transposed_returns,
        raw_allocation=None,
        max_total_allocation=1.0,
        add_limit=-0.50,
        maximal_cliques=[],
        selection_constraints=constraints,
        bootstrap_indices=None,
        percentile=10.0,
        allocation_tolerance=0.02,
        grid_steps=None,
    )

    assert outcome.completed
    assert outcome.allocation is not None
    allocation = outcome.allocation
    assert torch.count_nonzero(allocation) <= 2
    assert not (allocation[0] > 0.0 and allocation[1] > 0.0)
    assert torch.all(allocation <= 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_completed_historical_search_cuda_matches_cpu():
    """The serial spatial control flow preserves the CUDA tensor result."""
    transposed_returns = torch.tensor(
        [
            [0.47, -0.25, 0.17, 0.08],
            [0.06, 0.34, 0.01, -0.10],
        ],
        dtype=torch.float32,
    )
    common = {
        "raw_allocation": None,
        "max_total_allocation": 1.0,
        "add_limit": -0.027,
        "maximal_cliques": [(0, 1)],
        "selection_constraints": None,
        "bootstrap_indices": None,
        "percentile": 10.0,
        "allocation_tolerance": 0.01,
        "grid_steps": None,
    }

    cpu = try_add_branch_and_bound(
        transposed_returns=transposed_returns,
        **common,
    )
    cuda = try_add_branch_and_bound(
        transposed_returns=transposed_returns.cuda(),
        **common,
    )

    assert cpu.completed and cuda.completed
    assert cpu.allocation is not None and cuda.allocation is not None
    assert cuda.allocation.device.type == "cuda"
    torch.testing.assert_close(
        cuda.allocation.cpu(), cpu.allocation, rtol=0.0, atol=0.01
    )
