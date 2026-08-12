"""Tests for grid-search capital allocation."""

import itertools
import math

import pytest
import torch

import ifera._capital_allocation_memory as capital_allocation_memory
import ifera.capital_allocation as capital_allocation
from ifera.capital_allocation import find_optimal_capital_allocation


def _grid_steps(requested_increment: float) -> int:
    """Return the independently calculated number of equal grid intervals."""
    reciprocal = 1.0 / requested_increment
    nearest_integer = round(reciprocal)
    if math.isclose(reciprocal, nearest_integer, rel_tol=1e-12, abs_tol=1e-12):
        return nearest_integer
    return math.ceil(reciprocal)


def _objective(returns: torch.Tensor, allocation: torch.Tensor, alpha: float) -> float:
    """Evaluate the specified objective without using production helpers."""
    daily_returns = returns @ allocation
    if torch.any(daily_returns <= -1.0):
        return -math.inf

    wealth = torch.cumprod(1.0 + daily_returns, dim=0)
    wealth_with_initial = torch.cat((torch.ones(1, dtype=wealth.dtype), wealth), dim=0)
    running_peak = torch.cummax(wealth_with_initial, dim=0).values[1:]
    daily_drawdown = wealth / running_peak - 1.0
    average_daily_drawdown = -torch.sqrt(torch.mean(daily_drawdown.square()))
    if average_daily_drawdown <= -1.0:
        return -math.inf

    score = torch.log1p(daily_returns).sum()
    score += alpha * returns.shape[0] * torch.log1p(average_daily_drawdown)
    return float(score)


def _bootstrap_average_drawdown(
    returns: torch.Tensor,
    allocation: torch.Tensor,
    bootstrap_indices: torch.Tensor,
    percentile: float,
) -> float:
    """Evaluate bootstrapped ADD independently on a fixed shared-row sample."""
    daily_returns = torch.nan_to_num(returns, nan=0.0) @ allocation
    if torch.any(daily_returns <= -1.0):
        return -math.inf

    sampled_returns = daily_returns[bootstrap_indices]
    wealth = torch.cumprod(1.0 + sampled_returns, dim=1)
    initial_wealth = torch.ones((wealth.shape[0], 1), dtype=wealth.dtype)
    wealth_with_initial = torch.cat((initial_wealth, wealth), dim=1)
    running_peak = torch.cummax(wealth_with_initial, dim=1).values[:, 1:]
    daily_drawdown = wealth / running_peak - 1.0
    run_drawdowns = -torch.sqrt(torch.mean(daily_drawdown.square(), dim=1))
    pessimistic_drawdown = torch.quantile(run_drawdowns, percentile / 100.0)
    return float(pessimistic_drawdown)


def _bootstrap_objective(
    returns: torch.Tensor,
    allocation: torch.Tensor,
    alpha: float,
    bootstrap_indices: torch.Tensor,
    percentile: float,
) -> float:
    """Evaluate the bootstrapped objective independently on a fixed sample."""
    daily_returns = torch.nan_to_num(returns, nan=0.0) @ allocation
    if torch.any(daily_returns <= -1.0):
        return -math.inf
    pessimistic_drawdown = _bootstrap_average_drawdown(
        returns, allocation, bootstrap_indices, percentile
    )

    growth = torch.log1p(daily_returns).sum()
    score = growth + alpha * returns.shape[0] * math.log1p(pessimistic_drawdown)
    return float(score)


def _brute_force_optimum(
    returns: torch.Tensor,
    alpha: float,
    requested_increment: float,
    max_total_allocation: float = 1.0,
) -> torch.Tensor:
    """Find the optimum by enumerating a small integer grid in Python."""
    returns_cpu = returns.detach().to(device="cpu", dtype=torch.float64)
    steps = _grid_steps(requested_increment)
    unit_limit = math.floor(max_total_allocation * steps + 1e-12)
    strategy_count = returns.shape[1]
    best_score = -math.inf
    best_allocation = None

    for indices in itertools.product(range(unit_limit + 1), repeat=strategy_count):
        if sum(indices) > unit_limit:
            continue
        allocation = torch.tensor(indices, dtype=torch.float64) / steps
        score = _objective(returns_cpu, allocation, alpha)
        if score > best_score:
            best_score = score
            best_allocation = allocation

    assert best_allocation is not None
    return best_allocation


def _brute_force_bootstrap_optimum(
    returns: torch.Tensor,
    alpha: float,
    requested_increment: float,
    bootstrap_indices: torch.Tensor,
    percentile: float,
) -> torch.Tensor:
    """Find a small bootstrapped optimum with an independent Python grid."""
    returns_cpu = returns.detach().to(device="cpu", dtype=torch.float64)
    indices_cpu = bootstrap_indices.to(device="cpu")
    steps = _grid_steps(requested_increment)
    strategy_count = returns.shape[1]
    best_score = -math.inf
    best_allocation = None

    for indices in itertools.product(range(steps + 1), repeat=strategy_count):
        if sum(indices) > steps:
            continue
        allocation = torch.tensor(indices, dtype=torch.float64) / steps
        score = _bootstrap_objective(
            returns_cpu, allocation, alpha, indices_cpu, percentile
        )
        if score > best_score:
            best_score = score
            best_allocation = allocation

    assert best_allocation is not None
    return best_allocation


def _overlap_graph(returns: torch.Tensor) -> list[list[bool]]:
    """Return the pairwise non-disjointness graph from the NaN activity mask."""
    active = ~torch.isnan(returns)
    strategy_count = returns.shape[1]
    return [
        [
            bool(torch.any(active[:, left] & active[:, right]))
            for right in range(strategy_count)
        ]
        for left in range(strategy_count)
    ]


def _satisfies_clique_constraints(
    allocation: torch.Tensor,
    overlap: list[list[bool]],
    max_total_allocation: float = 1.0,
) -> bool:
    """Check all cliques, which is equivalent to checking maximal cliques."""
    strategy_count = allocation.numel()
    for clique_size in range(2, strategy_count + 1):
        for clique in itertools.combinations(range(strategy_count), clique_size):
            if all(
                overlap[left][right]
                for left, right in itertools.combinations(clique, 2)
            ):
                if float(allocation[list(clique)].sum()) > max_total_allocation + 1e-12:
                    return False
    return True


def _brute_force_disjoint_optimum(
    returns: torch.Tensor,
    alpha: float,
    requested_increment: float,
    max_total_allocation: float = 1.0,
) -> torch.Tensor:
    """Find the optimum using an independent exhaustive clique-constrained grid."""
    returns_cpu = torch.nan_to_num(
        returns.detach().to(device="cpu", dtype=torch.float64), nan=0.0
    )
    overlap = _overlap_graph(returns)
    steps = _grid_steps(requested_increment)
    unit_limit = math.floor(max_total_allocation * steps + 1e-12)
    strategy_count = returns.shape[1]
    best_score = -math.inf
    best_allocation = None

    for indices in itertools.product(range(unit_limit + 1), repeat=strategy_count):
        allocation = torch.tensor(indices, dtype=torch.float64) / steps
        if not _satisfies_clique_constraints(allocation, overlap, max_total_allocation):
            continue
        score = _objective(returns_cpu, allocation, alpha)
        if score > best_score:
            best_score = score
            best_allocation = allocation

    assert best_allocation is not None
    return best_allocation


def test_matches_independent_brute_force_oracle():
    returns = torch.tensor(
        [
            [0.18, -0.06, 0.04],
            [-0.12, 0.16, 0.03],
            [0.08, -0.04, -0.02],
            [-0.06, 0.10, 0.05],
        ],
        dtype=torch.float64,
    )
    alpha = 0.7

    expected = _brute_force_optimum(returns, alpha, 0.25)
    actual = find_optimal_capital_allocation(
        returns, alpha, 0.25, device=torch.device("cpu")
    )

    assert torch.allclose(actual.to(torch.float64), expected)


def test_non_divisor_increment_is_reduced_to_reciprocal_grid():
    returns = torch.tensor([[0.4], [-0.3], [0.4], [-0.3]])

    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.3, device=torch.device("cpu")
    )

    assert torch.equal(actual, torch.tensor([0.5]))


def test_refinement_finds_an_optimum_between_initial_grid_points():
    positive_return = 0.4
    continuous_optimum = 0.375
    loss_size = positive_return / (1.0 + 2.0 * positive_return * continuous_optimum)
    returns = torch.tensor([[positive_return], [-loss_size]], dtype=torch.float64)

    initial = find_optimal_capital_allocation(
        returns, 0.0, 0.25, device=torch.device("cpu")
    )
    refined = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        refinement_runs=1,
        refinement_divisor=2.0,
    )

    assert initial.item() in (0.25, 0.5)
    assert torch.equal(refined, torch.tensor([0.375], dtype=torch.float64))
    assert _objective(returns, refined, 0.0) > _objective(returns, initial, 0.0)


def test_refinement_recenters_after_an_interior_improvement_until_stable(monkeypatch):
    """An interior local optimum does not prove that a shifted window cannot improve."""
    returns = torch.tensor(
        [
            [-0.3013638862286593, 0.4384715768500549],
            [-0.08551437920067961, 0.5758210716927266],
            [0.6872710296193094, 0.3718194083554962],
            [0.517600540896737, 0.1329772343005512],
            [0.05431605072734791, -0.7188189391343465],
        ],
        dtype=torch.float64,
    )
    alpha = 1.398769029312754
    observed_ranges = []
    original_score = capital_allocation._allocation_scores

    def recording_score(allocations, *args, **kwargs):
        observed_ranges.append(
            torch.stack(
                (
                    torch.amin(allocations, dim=0),
                    torch.amax(allocations, dim=0),
                )
            ).cpu()
        )
        return original_score(allocations, *args, **kwargs)

    monkeypatch.setattr(capital_allocation, "_allocation_scores", recording_score)
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 100)

    actual = find_optimal_capital_allocation(
        returns,
        alpha,
        0.25,
        device=torch.device("cpu"),
        refinement_runs=1,
        refinement_divisor=2.0,
    )

    assert torch.equal(actual, torch.tensor([0.375, 0.125], dtype=torch.float64))
    assert len(observed_ranges) == 4
    assert torch.equal(
        observed_ranges[0],
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
    )
    assert torch.equal(
        observed_ranges[1],
        torch.tensor([[0.5, 0.0], [1.0, 0.5]], dtype=torch.float64),
    )
    assert torch.equal(
        observed_ranges[2],
        torch.tensor([[0.375, 0.0], [0.875, 0.5]], dtype=torch.float64),
    )
    assert torch.equal(
        observed_ranges[3],
        torch.tensor([[0.125, 0.0], [0.625, 0.5]], dtype=torch.float64),
    )


def test_multiple_refinements_divide_the_increment_successively():
    positive_return = 0.4
    continuous_optimum = 0.4375
    loss_size = positive_return / (1.0 + 2.0 * positive_return * continuous_optimum)
    returns = torch.tensor([[positive_return], [-loss_size]], dtype=torch.float64)

    one_refinement = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        refinement_runs=1,
        refinement_divisor=2.0,
    )
    two_refinements = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        refinement_runs=2,
        refinement_divisor=2.0,
    )

    assert torch.equal(one_refinement, torch.tensor([0.5], dtype=torch.float64))
    assert torch.equal(
        two_refinements, torch.tensor([continuous_optimum], dtype=torch.float64)
    )
    assert _objective(returns, two_refinements, 0.0) > _objective(
        returns, one_refinement, 0.0
    )


@pytest.mark.parametrize(
    ("returns", "expected_refined_grid"),
    [
        (
            torch.tensor([[-0.1], [-0.2]], dtype=torch.float64),
            torch.tensor([0.0, 0.125, 0.25, 0.375, 0.5], dtype=torch.float64),
        ),
        (
            torch.tensor([[0.1], [0.2]], dtype=torch.float64),
            torch.tensor([0.5, 0.625, 0.75, 0.875, 1.0], dtype=torch.float64),
        ),
    ],
    ids=["lower_boundary", "upper_boundary"],
)
def test_refinement_shifts_boundary_window_without_losing_grid_points(
    returns: torch.Tensor,
    expected_refined_grid: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
):
    observed_grids = []
    original_score = capital_allocation._allocation_scores

    def recording_score(*args, **kwargs):
        observed_grids.append(torch.unique(args[0][:, 0]).cpu())
        return original_score(*args, **kwargs)

    monkeypatch.setattr(capital_allocation, "_allocation_scores", recording_score)
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 100)

    find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        refinement_runs=1,
        refinement_divisor=2.0,
    )

    assert len(observed_grids) == 2
    assert torch.equal(
        observed_grids[0],
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64),
    )
    assert torch.equal(observed_grids[1], expected_refined_grid)


def test_refinement_uses_adjusted_non_divisor_increment():
    positive_return = 0.4
    continuous_optimum = 0.375
    loss_size = positive_return / (1.0 + 2.0 * positive_return * continuous_optimum)
    returns = torch.tensor([[positive_return], [-loss_size]], dtype=torch.float64)

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.3,
        device=torch.device("cpu"),
        refinement_runs=1,
        refinement_divisor=2.0,
    )

    assert torch.equal(actual, torch.tensor([0.375], dtype=torch.float64))


def test_zero_refinement_runs_preserves_the_original_search():
    returns = torch.tensor(
        [[0.2, -0.1], [-0.15, 0.25], [0.1, -0.05]], dtype=torch.float64
    )
    expected = _brute_force_optimum(returns, 0.4, 0.25)

    default = find_optimal_capital_allocation(
        returns, 0.4, 0.25, device=torch.device("cpu")
    )
    explicit = find_optimal_capital_allocation(
        returns,
        0.4,
        0.25,
        device=torch.device("cpu"),
        refinement_runs=0,
        refinement_divisor=2.0,
    )

    assert torch.equal(default, expected)
    assert torch.equal(explicit, expected)


def test_increment_is_not_rounded_up_beyond_float_tolerance():
    returns = torch.tensor([[0.5], [-0.4]], dtype=torch.float64)

    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.2499999999999, device=torch.device("cpu")
    )

    assert torch.equal(actual, torch.tensor([0.2], dtype=torch.float64))


def test_allocation_respects_simplex_constraint():
    returns = torch.tensor([[0.05, 0.10, 0.20]]).repeat(4, 1)

    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.25, device=torch.device("cpu")
    )

    assert torch.equal(actual, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.all(actual >= 0.0)
    assert actual.sum() <= 1.0 + torch.finfo(actual.dtype).eps


def test_max_total_allocation_allows_single_strategy_leverage():
    returns = torch.tensor([[0.1], [0.2]])

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        max_total_allocation=2.0,
    )

    assert torch.equal(actual, torch.tensor([2.0]))


def test_max_total_allocation_caps_overlapping_strategies():
    returns = torch.tensor([[0.1, 0.2]])

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        max_total_allocation=1.5,
    )

    assert torch.equal(actual, torch.tensor([0.0, 1.5]))
    assert actual.sum() == 1.5


def test_max_total_allocation_uses_highest_grid_point_below_cap():
    returns = torch.tensor([[0.1]])

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        max_total_allocation=1.3,
    )

    assert torch.equal(actual, torch.tensor([1.25]))


def test_max_total_allocation_combines_with_bootstrapped_drawdown():
    returns = torch.tensor([[0.1], [0.2]])

    actual = find_optimal_capital_allocation(
        returns,
        1.0,
        0.5,
        device=torch.device("cpu"),
        bootstrap_on=True,
        bootstrap_runs=3,
        bootstrap_length=4,
        max_total_allocation=2.0,
    )

    assert torch.equal(actual, torch.tensor([2.0]))


def test_only_strategies_two_and_three_are_disjoint(monkeypatch):
    returns = torch.tensor(
        [
            [0.0, 0.0, 0.2, math.nan],
            [0.0, 0.0, math.nan, 0.2],
        ]
    )
    scored_candidate_counts = []
    original_score = capital_allocation._allocation_scores

    def counting_score(*args, **kwargs):
        scored_candidate_counts.append(args[0].shape[0])
        return original_score(*args, **kwargs)

    monkeypatch.setattr(capital_allocation, "_allocation_scores", counting_score)

    actual = find_optimal_capital_allocation(
        returns, 0.0, 1.0, device=torch.device("cpu")
    )

    assert torch.equal(actual, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert sum(scored_candidate_counts) == 6


def test_none_disjoint_uses_maximal_clique_not_per_row_constraints():
    returns = torch.tensor(
        [
            [0.2, 0.2, math.nan],
            [math.nan, 0.2, 0.2],
            [0.2, math.nan, 0.2],
        ]
    )

    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.5, device=torch.device("cpu")
    )

    assert torch.equal(torch.sort(actual).values, torch.tensor([0.0, 0.5, 0.5]))
    assert actual.sum() == 1.0


def test_all_disjoint_allows_full_allocation_to_every_strategy():
    returns = torch.tensor(
        [
            [0.2, math.nan, math.nan],
            [math.nan, 0.3, math.nan],
            [math.nan, math.nan, 0.4],
        ]
    )

    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.5, device=torch.device("cpu")
    )

    assert torch.equal(actual, torch.ones(3))


def test_disjoint_strategies_each_receive_the_leveraged_cap():
    returns = torch.tensor([[0.2, math.nan], [math.nan, 0.3]])

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.5,
        device=torch.device("cpu"),
        max_total_allocation=1.5,
    )

    assert torch.equal(actual, torch.tensor([1.5, 1.5]))


def test_refinement_preserves_overlap_constraints_and_disjoint_capital_reuse():
    positive_return = 0.4
    continuous_optimum = 0.875
    loss_size = positive_return / (1.0 + 2.0 * positive_return * continuous_optimum)
    returns = torch.tensor(
        [
            [0.0, 0.0, positive_return, math.nan],
            [0.0, 0.0, -loss_size, math.nan],
            [0.0, 0.0, math.nan, positive_return],
            [0.0, 0.0, math.nan, -loss_size],
        ],
        dtype=torch.float64,
    )

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        refinement_runs=1,
        refinement_divisor=2.0,
    )

    expected = torch.tensor([0.0, 0.0, 0.875, 0.875], dtype=torch.float64)
    assert torch.equal(actual, expected)
    assert actual.sum() > 1.0
    assert _satisfies_clique_constraints(actual, _overlap_graph(returns))


def test_refinement_supports_leveraged_allocation_ranges():
    positive_return = 0.4
    continuous_optimum = 1.375
    loss_size = positive_return / (1.0 + 2.0 * positive_return * continuous_optimum)
    returns = torch.tensor([[positive_return], [-loss_size]], dtype=torch.float64)

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        max_total_allocation=1.5,
        refinement_runs=1,
        refinement_divisor=2.0,
    )

    assert torch.equal(actual, torch.tensor([continuous_optimum], dtype=torch.float64))
    assert actual.item() <= 1.5


def test_low_precision_refinement_does_not_round_up_clique_capacity():
    """A large dtype tolerance must not admit a whole extra refinement unit."""
    returns = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64)

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.3,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_total_allocation=0.6,
        refinement_runs=1,
        refinement_divisor=math.sqrt(2.0),
    )

    assert float(actual.sum()) <= 0.6 + torch.finfo(torch.bfloat16).eps


def test_refinement_rejects_grid_points_collapsed_by_output_dtype():
    """Every conceptual local point must remain distinct in the requested dtype."""
    positive_return = 0.1
    continuous_optimum = 3.8
    loss_size = positive_return / (1.0 + 2.0 * positive_return * continuous_optimum)
    returns = torch.tensor([[positive_return], [-loss_size]], dtype=torch.float64)

    with pytest.raises(ValueError, match="increment|dtype|small"):
        find_optimal_capital_allocation(
            returns,
            0.0,
            0.2,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_total_allocation=5.0,
            refinement_runs=1,
            refinement_divisor=7.3,
        )


def test_all_disjoint_grid_batches_match_cartesian_order():
    batches = list(
        capital_allocation._cartesian_allocation_unit_batches(
            allocation_unit_limit=2,
            strategy_count=3,
            device=torch.device("cpu"),
            batch_size=5,
        )
    )

    expected = torch.tensor(list(itertools.product(range(3), repeat=3)))
    assert torch.equal(torch.cat(batches), expected)
    assert max(batch.shape[0] for batch in batches) <= 5


def test_arbitrary_disjointness_graph_matches_clique_oracle(monkeypatch):
    returns = torch.tensor(
        [
            [0.30, 0.20, math.nan, math.nan],
            [math.nan, 0.20, 0.25, math.nan],
            [0.30, math.nan, 0.25, math.nan],
            [math.nan, math.nan, 0.25, 0.15],
        ],
        dtype=torch.float64,
    )
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 2)

    expected = _brute_force_disjoint_optimum(returns, 0.0, 0.5)
    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.5, device=torch.device("cpu")
    )

    assert torch.equal(expected, torch.tensor([0.5, 0.0, 0.5, 0.5]))
    assert torch.equal(actual, expected)


def test_leveraged_arbitrary_disjointness_graph_matches_clique_oracle():
    returns = torch.tensor(
        [
            [0.30, 0.20, math.nan, math.nan],
            [math.nan, 0.20, 0.25, math.nan],
            [0.30, math.nan, 0.25, math.nan],
            [math.nan, math.nan, 0.25, 0.15],
        ],
        dtype=torch.float64,
    )

    expected = _brute_force_disjoint_optimum(returns, 0.0, 0.5, 1.5)
    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.5,
        device=torch.device("cpu"),
        max_total_allocation=1.5,
    )

    assert torch.equal(actual, expected)


def test_nan_and_zero_have_same_returns_but_different_activity():
    inactive_returns = torch.tensor([[0.2, math.nan], [math.nan, 0.2]])
    active_zero_returns = torch.tensor([[0.2, 0.0], [0.0, 0.2]])
    assert torch.equal(torch.nan_to_num(inactive_returns), active_zero_returns)

    inactive_actual = find_optimal_capital_allocation(
        inactive_returns, 0.0, 0.5, device=torch.device("cpu")
    )
    active_zero_actual = find_optimal_capital_allocation(
        active_zero_returns, 0.0, 0.5, device=torch.device("cpu")
    )

    assert torch.equal(inactive_actual, torch.tensor([1.0, 1.0]))
    assert torch.equal(active_zero_actual, torch.tensor([0.5, 0.5]))


def test_large_strategy_count_uses_only_valid_simplex_points():
    returns = torch.arange(1, 21, dtype=torch.float32).unsqueeze(0) / 100.0

    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.5, device=torch.device("cpu")
    )

    expected = torch.zeros(20)
    expected[-1] = 1.0
    assert torch.equal(actual, expected)


def test_large_complete_overlap_graph_avoids_python_recursion_limit():
    strategy_count = 1_050
    active = torch.ones((1, strategy_count), dtype=torch.bool)

    cliques = capital_allocation._find_maximal_overlap_cliques(active)

    assert cliques == [tuple(range(strategy_count))]


def test_large_near_complete_graph_avoids_python_recursion_limit():
    strategy_count = 1_050
    active = torch.ones((2, strategy_count), dtype=torch.bool)
    active[0, -1] = False
    active[1, -2] = False

    cliques = capital_allocation._find_maximal_overlap_cliques(active)

    expected = {
        tuple(range(strategy_count - 1)),
        tuple((*range(strategy_count - 2), strategy_count - 1)),
    }
    assert set(cliques) == expected


def test_large_disjoint_generator_avoids_python_recursion_limit():
    strategy_count = 1_050
    cliques_by_strategy = [
        torch.empty(0, dtype=torch.int64) for _ in range(strategy_count)
    ]
    batches = capital_allocation._constrained_allocation_unit_batches(
        allocation_unit_limit=1,
        strategy_count=strategy_count,
        clique_count=0,
        cliques_by_strategy=cliques_by_strategy,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert torch.equal(
        next(batches), torch.zeros((1, strategy_count), dtype=torch.int64)
    )


def test_refined_generator_handles_unequal_clique_limits_across_tiny_batches():
    """Separate residual clique budgets remain exact when traversal is resumed."""
    device = torch.device("cpu")
    previous = torch.tensor([0.5, 0.5, 0.25], dtype=torch.float64)
    maximal_cliques = [(0, 1), (1, 2)]
    _, previous_indices = capital_allocation._refinement_grid_lower_bounds(
        previous, 4, 0.125, 1.0
    )
    clique_limits = capital_allocation._refinement_clique_unit_limits(
        previous,
        previous_indices,
        maximal_cliques,
        4,
        0.125,
        1.0,
    )
    cliques_by_strategy = capital_allocation._clique_indices_by_strategy(
        maximal_cliques, 3, device
    )

    batches = capital_allocation._constrained_allocation_unit_batches(
        allocation_unit_limit=4,
        strategy_count=3,
        clique_count=2,
        cliques_by_strategy=cliques_by_strategy,
        device=device,
        batch_size=2,
        clique_unit_limits=clique_limits,
    )
    actual = torch.cat(list(batches))
    expected = torch.tensor(
        [
            units
            for units in itertools.product(range(5), repeat=3)
            if units[0] + units[1] <= 4 and units[1] + units[2] <= 6
        ],
        dtype=torch.int64,
    )

    assert torch.equal(clique_limits, torch.tensor([4, 6]))
    assert torch.equal(actual, expected)


def test_drawdown_penalty_changes_optimum_and_is_scaled_by_duration():
    returns = torch.tensor([[0.2], [-0.1], [0.2], [-0.1]])

    no_penalty = find_optimal_capital_allocation(
        returns, 0.0, 0.25, device=torch.device("cpu")
    )
    with_penalty = find_optimal_capital_allocation(
        returns, 0.5, 0.25, device=torch.device("cpu")
    )

    assert torch.equal(no_penalty, torch.tensor([1.0]))
    assert torch.equal(with_penalty, torch.tensor([0.5]))


def test_drawdown_uses_initial_wealth_and_only_post_return_days():
    returns = torch.tensor([[-0.1], [0.2]])

    actual = find_optimal_capital_allocation(
        returns, 0.5, 0.25, device=torch.device("cpu")
    )

    assert torch.equal(actual, torch.tensor([0.5]))


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_historical_add_limit_hard_filters_initial_and_refinement_grids(alpha):
    """Every search pass excludes allocations below the historical ADD floor."""
    returns = torch.tensor([[0.50], [-0.20]], dtype=torch.float64)

    unrestricted = find_optimal_capital_allocation(
        returns,
        alpha,
        0.5,
        device="cpu",
        refinement_runs=1,
    )
    constrained = find_optimal_capital_allocation(
        returns,
        alpha,
        0.5,
        device="cpu",
        refinement_runs=1,
        ADD_limit=-0.10,
    )

    assert torch.equal(unrestricted, torch.tensor([1.0], dtype=torch.float64))
    # The refined 0.75 point has ADD ~= -0.106, so it must not replace 0.50.
    assert torch.equal(constrained, torch.tensor([0.5], dtype=torch.float64))


def test_bootstrap_matches_fixed_shared_row_sample_oracle(monkeypatch):
    returns = torch.tensor(
        [
            [0.30, -0.20],
            [-0.25, 0.35],
            [0.15, math.nan],
            [-0.10, 0.10],
        ],
        dtype=torch.float64,
    )
    bootstrap_indices = torch.tensor(
        [
            [0, 1, 0, 1, 2],
            [1, 0, 1, 0, 3],
            [0, 2, 3, 1, 0],
            [1, 3, 2, 0, 1],
        ]
    )
    randint_calls = []

    def fixed_randint(high, size, *, device, dtype):
        randint_calls.append((high, size, device, dtype))
        return bootstrap_indices.to(device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 1)
    expected = _brute_force_bootstrap_optimum(
        returns, 0.3, 0.5, bootstrap_indices, 25.0
    )
    assert torch.equal(expected, torch.tensor([0.5, 0.5], dtype=torch.float64))

    actual = find_optimal_capital_allocation(
        returns,
        0.3,
        0.5,
        device=torch.device("cpu"),
        bootstrap_on=True,
        bootstrap_runs=4,
        bootstrap_length=5,
        percentile=25.0,
    )

    assert torch.equal(actual, expected)
    assert randint_calls == [(4, (4, 5), torch.device("cpu"), torch.int64)]


@pytest.mark.parametrize("alpha", [0.0, 0.01])
def test_bootstrap_add_limit_uses_percentile_add_and_reuses_scoring_result(
    monkeypatch, alpha
):
    """The bootstrap floor filters candidates without recomputing the winner's ADD."""
    returns = torch.tensor([[0.50], [-0.20]], dtype=torch.float64)
    bootstrap_indices = torch.tensor([[1, 1], [1, 1]], dtype=torch.int64)
    randint_calls = 0
    bootstrap_calls = 0
    original_bootstrap = capital_allocation._bootstrap_average_drawdown

    def fixed_randint(high, size, *, device, dtype):
        nonlocal randint_calls
        randint_calls += 1
        assert high == 2
        assert size == (2, 2)
        return bootstrap_indices.to(device=device, dtype=dtype)

    def counting_bootstrap(*args, **kwargs):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_bootstrap(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    monkeypatch.setattr(
        capital_allocation, "_bootstrap_average_drawdown", counting_bootstrap
    )
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 100)

    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    constrained = find_optimal_capital_allocation(
        returns,
        alpha,
        0.5,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=2,
        bootstrap_length=2,
        percentile=10.0,
        ADD_limit=-0.20,
        refinement_runs=1,
        diagnostics=diagnostics,
    )

    # Historically, full allocation has ADD ~= -0.141 and satisfies this limit.
    historical = find_optimal_capital_allocation(
        returns,
        alpha,
        0.5,
        device="cpu",
        bootstrap_on=False,
        ADD_limit=-0.20,
    )
    expected_add = _bootstrap_average_drawdown(
        returns, constrained, bootstrap_indices, percentile=10.0
    )
    assert torch.equal(historical, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(constrained, torch.tensor([0.5], dtype=torch.float64))
    assert expected_add >= -0.20
    assert diagnostics.bootstrap_average_drawdown == pytest.approx(expected_add)
    assert randint_calls == 1
    # One vectorized evaluation per grid; no third evaluation just for diagnostics.
    assert bootstrap_calls == 2


def test_diagnostics_retain_the_optimal_candidates_bootstrap_add(monkeypatch):
    """Reporting reuses the ADD calculated while scoring the winning candidate."""
    returns = torch.tensor(
        [[0.30, -0.20], [-0.25, 0.35], [0.15, math.nan], [-0.10, 0.10]],
        dtype=torch.float64,
    )
    bootstrap_indices = torch.tensor(
        [[0, 1, 0, 1, 2], [1, 0, 1, 0, 3], [0, 2, 3, 1, 0], [1, 3, 2, 0, 1]]
    )
    bootstrap_calls = 0
    original_bootstrap = capital_allocation._bootstrap_average_drawdown

    def fixed_randint(high, size, *, device, dtype):
        del high, size
        return bootstrap_indices.to(device=device, dtype=dtype)

    def counting_bootstrap(*args, **kwargs):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_bootstrap(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    monkeypatch.setattr(
        capital_allocation, "_bootstrap_average_drawdown", counting_bootstrap
    )
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 2)

    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    result = find_optimal_capital_allocation(
        returns,
        0.3,
        0.5,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=4,
        bootstrap_length=5,
        percentile=25.0,
        diagnostics=diagnostics,
    )

    assert torch.equal(result, torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert diagnostics.bootstrap_average_drawdown == pytest.approx(
        _bootstrap_average_drawdown(returns, result, bootstrap_indices, percentile=25.0)
    )
    assert bootstrap_calls == 3


def test_zero_alpha_diagnostics_bootstrap_only_the_final_allocation(monkeypatch):
    """An unpenalized search evaluates bootstrap ADD once for its final result."""
    returns = torch.tensor([[0.20], [-0.10], [0.15]], dtype=torch.float64)
    bootstrap_indices = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64)
    bootstrap_calls = 0
    original_bootstrap = capital_allocation._bootstrap_average_drawdown

    def fixed_randint(high, size, *, device, dtype):
        del high, size
        return bootstrap_indices.to(device=device, dtype=dtype)

    def counting_bootstrap(*args, **kwargs):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_bootstrap(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    monkeypatch.setattr(
        capital_allocation, "_bootstrap_average_drawdown", counting_bootstrap
    )

    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    result = find_optimal_capital_allocation(
        returns,
        0.0,
        0.5,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=2,
        bootstrap_length=3,
        percentile=10.0,
        diagnostics=diagnostics,
    )

    assert diagnostics.bootstrap_average_drawdown == pytest.approx(
        _bootstrap_average_drawdown(returns, result, bootstrap_indices, percentile=10.0)
    )
    assert bootstrap_calls == 1


def test_bootstrap_defaults_use_fixed_length(monkeypatch):
    calls = []

    def zero_indices(high, size, *, device, dtype):
        calls.append((high, size, device, dtype))
        return torch.zeros(size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", zero_indices)
    returns = torch.tensor([[0.1], [-0.1], [0.2]])

    find_optimal_capital_allocation(
        returns,
        0.1,
        1.0,
        device=torch.device("cpu"),
        bootstrap_on=True,
    )

    assert calls == [(3, (1024, 256), torch.device("cpu"), torch.int64)]


def test_bootstrap_indices_are_shared_across_all_refinements(monkeypatch):
    generated_indices = torch.tensor(
        [[0, 1, 2, 1], [2, 0, 1, 0], [1, 2, 0, 2]], dtype=torch.int64
    )
    randint_calls = []
    observed_indices = []
    observed_grids = []
    score_by_allocation = {
        0.25: 1.0,
        0.375: 2.0,
        0.625: 3.0,
        0.6875: 4.0,
        0.8125: 5.0,
    }

    def fixed_randint(high, size, *, device, dtype):
        randint_calls.append((high, size, device, dtype))
        return generated_indices.to(device=device, dtype=dtype)

    def recording_score(allocations, *args, **kwargs):
        del args
        observed_indices.append(kwargs["bootstrap_indices"].clone())
        observed_grids.append(torch.unique(allocations[:, 0]).cpu())
        scores = torch.full(
            (allocations.shape[0],),
            -1.0,
            device=allocations.device,
            dtype=allocations.dtype,
        )
        for allocation, score in score_by_allocation.items():
            scores[allocations[:, 0] == allocation] = score
        return scores

    monkeypatch.setattr(torch, "randint", fixed_randint)
    monkeypatch.setattr(capital_allocation, "_allocation_scores", recording_score)
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 100)
    returns = torch.tensor([[0.2], [-0.1], [0.1]], dtype=torch.float64)

    actual = find_optimal_capital_allocation(
        returns,
        0.4,
        0.25,
        device=torch.device("cpu"),
        bootstrap_on=True,
        bootstrap_runs=3,
        bootstrap_length=4,
        refinement_runs=2,
        refinement_divisor=2.0,
    )

    assert randint_calls == [(3, (3, 4), torch.device("cpu"), torch.int64)]
    assert torch.equal(actual, torch.tensor([0.8125], dtype=torch.float64))
    assert len(observed_indices) == 7
    assert all(
        torch.equal(indices.cpu(), generated_indices) for indices in observed_indices
    )
    expected_grids = [
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.125, 0.25, 0.375, 0.5],
        [0.125, 0.25, 0.375, 0.5, 0.625],
        [0.375, 0.5, 0.625, 0.75, 0.875],
        [0.5, 0.5625, 0.625, 0.6875, 0.75],
        [0.5625, 0.625, 0.6875, 0.75, 0.8125],
        [0.6875, 0.75, 0.8125, 0.875, 0.9375],
    ]
    assert all(
        torch.equal(grid, torch.tensor(expected, dtype=torch.float64))
        for grid, expected in zip(observed_grids, expected_grids, strict=True)
    )


def test_bootstrap_is_skipped_when_disabled_or_alpha_is_zero(monkeypatch):
    def unexpected_randint(*args, **kwargs):
        raise AssertionError("bootstrap indices should not be generated")

    monkeypatch.setattr(torch, "randint", unexpected_randint)
    returns = torch.tensor([[0.2], [-0.1]])

    default_result = find_optimal_capital_allocation(
        returns, 0.5, 0.5, device=torch.device("cpu")
    )
    explicit_off_result = find_optimal_capital_allocation(
        returns,
        0.5,
        0.5,
        device=torch.device("cpu"),
        bootstrap_on=False,
    )
    zero_alpha_result = find_optimal_capital_allocation(
        returns,
        0.0,
        0.5,
        device=torch.device("cpu"),
        bootstrap_on=True,
    )

    assert torch.equal(default_result, explicit_off_result)
    assert torch.equal(zero_alpha_result, torch.tensor([1.0]))


def test_bootstrap_penalty_uses_actual_duration_and_original_growth():
    short_returns = torch.tensor([[0.2], [-0.1]], dtype=torch.float64)
    long_returns = short_returns.repeat((2, 1))
    allocations = torch.tensor([[0.5]], dtype=torch.float64)
    bootstrap_indices = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])

    short_score = capital_allocation._allocation_scores(
        allocations,
        short_returns.transpose(0, 1),
        0.4,
        bootstrap_indices=bootstrap_indices,
        percentile=10.0,
    )
    long_score = capital_allocation._allocation_scores(
        allocations,
        long_returns.transpose(0, 1),
        0.4,
        bootstrap_indices=bootstrap_indices,
        percentile=10.0,
    )

    assert torch.allclose(long_score, 2.0 * short_score)


def test_lower_bootstrap_percentile_is_more_pessimistic():
    returns = torch.tensor([[0.2], [-0.3], [0.1]], dtype=torch.float64)
    allocations = torch.tensor([[1.0]], dtype=torch.float64)
    bootstrap_indices = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 2, 1], [2, 0, 2, 0]]
    )

    lower_score = capital_allocation._allocation_scores(
        allocations,
        returns.transpose(0, 1),
        0.5,
        bootstrap_indices=bootstrap_indices,
        percentile=0.0,
    )
    upper_score = capital_allocation._allocation_scores(
        allocations,
        returns.transpose(0, 1),
        0.5,
        bootstrap_indices=bootstrap_indices,
        percentile=100.0,
    )

    assert torch.all(lower_score < upper_score)


@pytest.mark.parametrize("returns_dtype", [torch.float32, torch.float64])
def test_default_dtype_and_device(returns_dtype: torch.dtype):
    returns = torch.tensor([[0.2], [-0.1]], dtype=returns_dtype)

    actual = find_optimal_capital_allocation(returns, 0.0, 0.5)

    expected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert actual.dtype == returns_dtype
    assert actual.device == expected_device


def test_explicit_dtype_and_device_override_defaults():
    returns = torch.tensor([[0.2], [-0.1]], dtype=torch.float64)

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert actual.dtype == torch.float32
    assert actual.device == torch.device("cpu")


def test_dtype_only_controls_allocation_not_return_precision():
    returns = torch.tensor([[0.100000001, 0.1]], dtype=torch.float64)

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(actual, torch.tensor([1.0, 0.0]))


def test_log_growth_preserves_small_float32_returns():
    returns = torch.tensor([[1e-8]], dtype=torch.float32)

    actual = find_optimal_capital_allocation(
        returns, 0.0, 1.0, device=torch.device("cpu")
    )

    assert torch.equal(actual, torch.tensor([1.0]))


def test_search_retains_best_result_across_small_batches(monkeypatch):
    returns = torch.tensor(
        [
            [0.18, -0.06, 0.04],
            [-0.12, 0.16, 0.03],
            [0.08, -0.04, -0.02],
            [-0.06, 0.10, 0.05],
        ],
        dtype=torch.float64,
    )
    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 2)

    expected = _brute_force_optimum(returns, 0.7, 0.25)
    actual = find_optimal_capital_allocation(
        returns, 0.7, 0.25, device=torch.device("cpu")
    )

    assert torch.equal(actual, expected)


def test_nonfinite_intermediate_does_not_poison_batch():
    returns = torch.full((2, 6), torch.finfo(torch.float32).max)

    actual = find_optimal_capital_allocation(
        returns, 1.0, 1.0 / 6.0, device=torch.device("cpu")
    )

    expected = torch.zeros(6)
    expected[-1] = 1.0
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_oracle():
    returns = torch.tensor(
        [[0.15, -0.05], [-0.08, 0.12], [0.09, -0.03]], dtype=torch.float64
    )

    expected = _brute_force_optimum(returns, 0.4, 0.25)
    actual = find_optimal_capital_allocation(
        returns, 0.4, 0.25, device=torch.device("cuda:0")
    )

    assert actual.device == torch.device("cuda:0")
    assert torch.allclose(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_constrained_cuda_matches_cpu_oracle():
    returns = torch.tensor(
        [
            [0.30, 0.20, math.nan, math.nan],
            [math.nan, 0.20, 0.25, math.nan],
            [0.30, math.nan, 0.25, math.nan],
            [math.nan, math.nan, 0.25, 0.15],
        ],
        dtype=torch.float64,
    )

    expected = _brute_force_disjoint_optimum(returns, 0.4, 0.5)
    actual = find_optimal_capital_allocation(
        returns, 0.4, 0.5, device=torch.device("cuda:0")
    )

    assert actual.device == torch.device("cuda:0")
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bootstrap_cuda_scores_match_cpu():
    returns = torch.tensor(
        [[0.2, -0.1], [-0.15, 0.3], [0.1, 0.05]], dtype=torch.float64
    )
    allocations = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
        dtype=torch.float64,
    )
    bootstrap_indices = torch.arange(17 * 7).reshape(17, 7) % returns.shape[0]

    expected = capital_allocation._allocation_scores(
        allocations,
        returns.transpose(0, 1),
        0.4,
        bootstrap_indices=bootstrap_indices,
        percentile=10.0,
    )
    actual = capital_allocation._allocation_scores(
        allocations.cuda(),
        returns.transpose(0, 1).cuda(),
        0.4,
        bootstrap_indices=bootstrap_indices.cuda(),
        percentile=10.0,
    )

    assert torch.allclose(actual.cpu(), expected)


@pytest.mark.parametrize(
    "returns",
    [
        torch.tensor([0.1, -0.1]),
        torch.zeros((2, 2, 1)),
        torch.empty((0, 2)),
        torch.empty((2, 0)),
    ],
    ids=["one_dimensional", "three_dimensional", "no_times", "no_strategies"],
)
def test_rejects_invalid_return_shapes(returns: torch.Tensor):
    with pytest.raises(ValueError):
        find_optimal_capital_allocation(returns, 0.0, 0.25)


@pytest.mark.parametrize(
    "returns",
    [
        torch.tensor([[1, -1]]),
        torch.tensor([[True, False]]),
        torch.tensor([[1.0 + 0.0j]]),
    ],
    ids=["integer", "boolean", "complex"],
)
def test_rejects_non_floating_returns(returns: torch.Tensor):
    with pytest.raises(TypeError):
        find_optimal_capital_allocation(returns, 0.0, 0.25)


@pytest.mark.parametrize("bad_value", [math.inf, -math.inf])
def test_rejects_infinite_returns(bad_value: float):
    returns = torch.tensor([[0.1], [bad_value]])

    with pytest.raises(ValueError):
        find_optimal_capital_allocation(returns, 0.0, 0.25)


@pytest.mark.parametrize("alpha", [-0.1, math.nan, math.inf, -math.inf])
def test_rejects_invalid_alpha(alpha: float):
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(ValueError):
        find_optimal_capital_allocation(returns, alpha, 0.25)


def test_zero_max_total_allocation_returns_cash_only():
    returns = torch.tensor([[0.1], [0.2]])

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        max_total_allocation=0.0,
    )

    assert torch.equal(actual, torch.tensor([0.0]))


def test_single_point_grid_is_not_rescored_during_refinement(monkeypatch):
    """A one-point initial grid cannot gain resolution while retaining one point."""
    calls = 0
    original_score = capital_allocation._allocation_scores

    def recording_score(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_score(*args, **kwargs)

    monkeypatch.setattr(capital_allocation, "_allocation_scores", recording_score)
    returns = torch.tensor([[0.1], [0.2]])

    actual = find_optimal_capital_allocation(
        returns,
        0.0,
        0.25,
        device=torch.device("cpu"),
        max_total_allocation=0.0,
        refinement_runs=10,
    )

    assert torch.equal(actual, torch.tensor([0.0]))
    assert calls == 1


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, -math.inf])
def test_rejects_invalid_max_total_allocation(value: float):
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(ValueError):
        find_optimal_capital_allocation(returns, 0.0, 0.25, max_total_allocation=value)


def test_rejects_boolean_max_total_allocation():
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(TypeError):
        find_optimal_capital_allocation(returns, 0.0, 0.25, max_total_allocation=True)


@pytest.mark.parametrize(
    ("parameter", "value", "exception"),
    [
        ("refinement_runs", True, TypeError),
        ("refinement_runs", 1.5, TypeError),
        ("refinement_runs", -1, ValueError),
        ("refinement_divisor", True, TypeError),
        ("refinement_divisor", "invalid", TypeError),
        ("refinement_divisor", 1.0, ValueError),
        ("refinement_divisor", 0.0, ValueError),
        ("refinement_divisor", -2.0, ValueError),
        ("refinement_divisor", math.nan, ValueError),
        ("refinement_divisor", math.inf, ValueError),
    ],
)
def test_rejects_invalid_refinement_parameters(parameter, value, exception):
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(exception):
        find_optimal_capital_allocation(
            returns,
            0.0,
            0.25,
            device=torch.device("cpu"),
            **{parameter: value},
        )


@pytest.mark.parametrize(
    ("parameter", "value", "exception"),
    [
        ("bootstrap_on", 1, TypeError),
        ("bootstrap_runs", True, TypeError),
        ("bootstrap_runs", 1.5, TypeError),
        ("bootstrap_runs", 0, ValueError),
        ("bootstrap_runs", -1, ValueError),
        ("bootstrap_length", True, TypeError),
        ("bootstrap_length", 2.5, TypeError),
        ("bootstrap_length", 0, ValueError),
        ("bootstrap_length", -1, ValueError),
        ("percentile", True, TypeError),
        ("percentile", -0.1, ValueError),
        ("percentile", 100.1, ValueError),
        ("percentile", math.nan, ValueError),
        ("percentile", math.inf, ValueError),
    ],
)
def test_rejects_invalid_bootstrap_parameters(parameter, value, exception):
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(exception):
        find_optimal_capital_allocation(
            returns,
            0.0,
            0.25,
            device=torch.device("cpu"),
            **{parameter: value},
        )


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (True, TypeError),
        ("invalid", TypeError),
        (0.1, ValueError),
        (-1.1, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
    ],
)
def test_rejects_invalid_add_limit(value, exception):
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(exception, match="ADD_limit"):
        find_optimal_capital_allocation(
            returns,
            0.0,
            0.25,
            device="cpu",
            ADD_limit=value,
        )


def test_add_limit_accepts_closed_drawdown_range_endpoints():
    returns = torch.tensor([[0.50], [-0.20]], dtype=torch.float64)

    no_effect = find_optimal_capital_allocation(
        returns, 0.0, 0.5, device="cpu", ADD_limit=-1.0
    )
    no_drawdown = find_optimal_capital_allocation(
        returns, 0.0, 0.5, device="cpu", ADD_limit=0.0
    )

    assert torch.equal(no_effect, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(no_drawdown, torch.tensor([0.0], dtype=torch.float64))


def test_rejects_invalid_diagnostics_object():
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(TypeError, match="diagnostics"):
        find_optimal_capital_allocation(
            returns,
            0.0,
            0.25,
            device="cpu",
            diagnostics=object(),  # type: ignore[arg-type]
        )


def test_rejects_bootstrap_workspace_larger_than_memory_budget(monkeypatch):
    monkeypatch.setattr(capital_allocation_memory, "_BATCH_MEMORY_BUDGET_BYTES", 1024)
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(ValueError, match="one candidate"):
        find_optimal_capital_allocation(
            returns,
            0.5,
            0.25,
            device=torch.device("cpu"),
            bootstrap_on=True,
            bootstrap_runs=1,
            bootstrap_length=32,
        )


@pytest.mark.parametrize("increment", [0.0, -0.1, 1.1, math.nan, math.inf])
def test_rejects_invalid_increment(increment: float):
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(ValueError):
        find_optimal_capital_allocation(returns, 0.0, increment)


def test_rejects_non_floating_output_dtype():
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(TypeError):
        find_optimal_capital_allocation(returns, 0.0, 0.25, dtype=torch.int64)


def test_rejects_unsupported_floating_output_dtype():
    unsupported_dtype = getattr(torch, "float8_e4m3fn", None)
    if unsupported_dtype is None:
        pytest.skip("PyTorch has no float8 dtype")
    returns = torch.tensor([[0.1], [-0.1]])

    with pytest.raises(TypeError):
        find_optimal_capital_allocation(returns, 0.0, 0.25, dtype=unsupported_dtype)


def test_candidates_with_nonpositive_portfolio_multipliers_are_ignored():
    returns = torch.tensor([[-1.5], [4.0]])

    expected = _brute_force_optimum(returns, 0.0, 0.25)
    actual = find_optimal_capital_allocation(
        returns, 0.0, 0.25, device=torch.device("cpu")
    )

    assert torch.equal(expected, torch.tensor([0.25], dtype=torch.float64))
    assert torch.allclose(actual.to(torch.float64), expected)


def test_accepts_noncontiguous_returns():
    returns = torch.tensor([[0.4, -0.3, 0.4, -0.3], [-0.1, 0.2, -0.1, 0.2]]).transpose(
        0, 1
    )
    assert not returns.is_contiguous()

    expected = _brute_force_optimum(returns, 0.3, 0.25)
    actual = find_optimal_capital_allocation(
        returns, 0.3, 0.25, device=torch.device("cpu")
    )

    assert torch.allclose(actual.to(torch.float64), expected)
