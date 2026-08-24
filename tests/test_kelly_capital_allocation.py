"""Integration tests for the continuous Kelly allocation path."""

from __future__ import annotations

import math

import pytest
import torch

import ifera.allocation.capital_allocation as capital_allocation
from ifera.allocation import _capital_allocation_memory as allocation_memory
from ifera.allocation import _kelly_selection


def _two_outcome_kelly_fraction(gain: float, loss: float) -> float:
    """Return the equal-probability Kelly fraction for +gain and -loss."""
    return (gain - loss) / (2.0 * gain * loss)


def _historical_average_drawdown(
    returns: torch.Tensor, allocation: torch.Tensor
) -> float:
    """Calculate production-equivalent historical ADD for one allocation."""
    portfolio_returns = torch.nan_to_num(returns, nan=0.0) @ allocation
    log_equity = torch.log1p(portfolio_returns).cumsum(dim=0)
    running_peak = torch.cummax(log_equity, dim=0).values.clamp_min(0.0)
    drawdowns = torch.expm1(log_equity - running_peak)
    return -float(torch.sqrt(torch.mean(drawdowns.square())).item())


def _drawdown_counterexample_returns() -> torch.Tensor:
    """Return a two-strategy sample whose Kelly ray breaches a tight ADD floor."""
    return torch.tensor(
        [
            [0.47, 0.06],
            [-0.25, 0.34],
            [0.17, 0.01],
            [0.08, -0.10],
        ],
        dtype=torch.float64,
    )


def test_raw_kelly_uses_continuous_optimum_and_grid_is_optional():
    """Raw Kelly ignores grid resolution and recovers an interior optimum."""
    gain, loss = 0.30, 0.20
    returns = torch.tensor([[gain], [-loss]], dtype=torch.float64)
    expected = torch.tensor(
        [_two_outcome_kelly_fraction(gain, loss)], dtype=torch.float64
    )

    without_grid = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
    )
    with_coarse_grid = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=0.5,
        device="cpu",
    )

    torch.testing.assert_close(without_grid, expected, rtol=0.0, atol=2e-7)
    torch.testing.assert_close(with_coarse_grid, expected, rtol=0.0, atol=2e-7)


def test_raw_kelly_ignores_grid_bootstrap_and_refinement_parameters():
    """Parameters belonging only to grid/drawdown scoring are unused."""
    returns = torch.tensor([[0.30], [-0.20]], dtype=torch.float64)

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=0,
        bootstrap_length=0,
        percentile=math.nan,
        refinement_runs=-1,
        refinement_divisor=1.0,
    )

    expected = torch.tensor([5.0 / 6.0], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-7)


def test_raw_kelly_does_not_call_grid_scorer(monkeypatch: pytest.MonkeyPatch):
    """Eligible raw Kelly searches bypass candidate generation and scoring."""
    returns = torch.tensor(
        [[0.30, 0.0], [-0.20, 0.0], [0.0, 0.25], [0.0, -0.20]],
        dtype=torch.float64,
    )

    def unexpected_grid_score(*_args, **_kwargs):
        raise AssertionError("the grid scorer must not be called")

    monkeypatch.setattr(capital_allocation, "_allocation_scores", unexpected_grid_score)

    actual = capital_allocation.find_optimal_capital_allocation(
        returns, alpha=0.0, grid_increment=1.0, device="cpu"
    )

    assert torch.count_nonzero(actual) == 2


def test_raw_kelly_uses_supplied_grid_if_continuous_solver_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
):
    """A numerical continuous-solver failure degrades to the exact grid path."""

    def unfinished_solver(*_args, **_kwargs):
        return type(
            "UnfinishedResult",
            (),
            {
                "allocation": torch.zeros(1, dtype=torch.float64),
                "mean_log_growth": 0.0,
                "upper_bound": 1.0,
                "converged": False,
            },
        )()

    monkeypatch.setattr(_kelly_selection, "maximize_kelly_growth", unfinished_solver)
    returns = torch.tensor([[0.2], [0.2]], dtype=torch.float64)

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=0.5,
        device="cpu",
    )

    torch.testing.assert_close(actual, torch.tensor([1.0], dtype=torch.float64))


def test_raw_kelly_grid_fallback_ignores_bootstrap_controls_without_diagnostics():
    """Bootstrap controls remain irrelevant when overlap geometry needs a grid."""
    returns = torch.tensor(
        [
            [0.2, 0.1, math.nan],
            [math.nan, 0.1, 0.2],
        ],
        dtype=torch.float64,
    )

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=0.5,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=0,
        bootstrap_length=0,
        percentile=math.nan,
    )

    assert actual.shape == (3,)
    assert bool(torch.isfinite(actual).all())


def test_raw_kelly_grid_fallback_sizes_batches_without_diagnostic_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
):
    """Final-winner diagnostics must not shrink every growth-scoring batch."""
    calls: list[tuple[int, int, bool]] = []

    def recording_batch_size(*args):
        calls.append((args[-3], args[-2], args[-1]))
        return 64

    monkeypatch.setattr(
        capital_allocation, "_candidate_batch_size", recording_batch_size
    )
    returns = torch.tensor(
        [
            [0.2, 0.1, math.nan],
            [math.nan, 0.1, 0.2],
        ],
        dtype=torch.float64,
    )
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=0.5,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=7,
        bootstrap_length=9,
        diagnostics=diagnostics,
    )

    assert calls == [(7, 9, False)]
    assert diagnostics.bootstrap_average_drawdown is not None


def test_diagnostic_only_batch_planning_reserves_shared_bootstrap_indices(
    monkeypatch: pytest.MonkeyPatch,
):
    """Shared indices consume memory without charging a path to every candidate."""
    monkeypatch.setattr(allocation_memory, "_BATCH_MEMORY_BUDGET_BYTES", 40_000)
    common = (10, 1, 0, torch.float32, torch.device("cpu"), False)

    ordinary = allocation_memory.candidate_batch_size(*common, 0, 0)
    diagnostic_only = allocation_memory.candidate_batch_size(
        *common,
        10,
        10,
        bootstrap_paths=False,
    )
    bootstrap_scoring = allocation_memory.candidate_batch_size(*common, 10, 10)

    assert bootstrap_scoring < diagnostic_only < ordinary


def test_diagnostic_only_search_rejects_an_oversized_final_bootstrap_workspace(
    monkeypatch: pytest.MonkeyPatch,
):
    """Final-winner path memory must fit in addition to the shared indices."""
    monkeypatch.setattr(allocation_memory, "_BATCH_MEMORY_BUDGET_BYTES", 10_000)

    with pytest.raises(ValueError, match="one candidate"):
        allocation_memory.search_candidate_batch_size(
            time_count=10,
            strategy_count=1,
            clique_count=0,
            dtype=torch.float32,
            device=torch.device("cpu"),
            uses_prefix_expansion=False,
            bootstrap_needed=True,
            drawdown_required=False,
            bootstrap_runs=1,
            bootstrap_length=400,
            maximum_candidates=1_000,
        )


def test_compiled_bootstrap_batch_planning_reserves_padded_map_indices(
    monkeypatch: pytest.MonkeyPatch,
):
    """A short run count still maps one padded 16-run chunk under compilation."""
    monkeypatch.setattr(allocation_memory, "_BATCH_MEMORY_BUDGET_BYTES", 30_000)

    with pytest.raises(ValueError, match="bootstrap index"):
        allocation_memory.candidate_batch_size(
            time_count=10,
            strategy_count=1,
            clique_count=0,
            dtype=torch.float32,
            device=torch.device("cpu"),
            uses_prefix_expansion=False,
            bootstrap_runs=1,
            bootstrap_length=400,
        )


def test_raw_kelly_solves_disjoint_capital_budgets_independently():
    """Disjoint strategies may each use the complete allocation budget."""
    returns = torch.tensor(
        [[0.30, math.nan], [-0.20, math.nan], [math.nan, 0.40], [math.nan, -0.25]],
        dtype=torch.float64,
    )

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
        max_total_allocation=1.0,
    )

    expected = torch.tensor(
        [
            _two_outcome_kelly_fraction(0.30, 0.20),
            _two_outcome_kelly_fraction(0.40, 0.25),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=3e-7)
    assert float(actual.sum()) > 1.0


def test_raw_kelly_selection_constraint_chooses_global_best_support():
    """Continuous optimization preserves hard strategy-selection semantics."""
    returns = torch.tensor(
        [
            [0.30, 0.0, 0.0],
            [-0.20, 0.0, 0.0],
            [0.0, 0.50, 0.0],
            [0.0, -0.20, 0.0],
            [0.0, 0.0, 0.22],
            [0.0, 0.0, -0.20],
        ],
        dtype=torch.float64,
    )

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
        max_strategies=1,
    )

    assert torch.count_nonzero(actual) == 1
    assert actual[1] > 0.0


def test_add_feasible_raw_kelly_is_global_optimum_without_grid():
    """A feasible unconstrained winner also solves the ADD-constrained problem."""
    returns = _drawdown_counterexample_returns()
    raw_kelly = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
    )

    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    constrained = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        ADD_limit=-0.03,
        diagnostics=diagnostics,
    )

    assert torch.count_nonzero(raw_kelly) == 2
    assert _historical_average_drawdown(returns, raw_kelly) >= -0.03
    torch.testing.assert_close(constrained, raw_kelly, rtol=0.0, atol=2e-8)
    assert diagnostics.algorithm == (
        "Continuous raw Kelly + exact ADD feasibility precheck"
    )


def test_scalar_historical_add_limit_is_solved_exactly_without_grid():
    """Scaling the sole strategy finds the continuous historical ADD boundary."""
    returns = torch.tensor([[0.30], [-0.20]], dtype=torch.float64)
    expected_fraction = 0.05 * math.sqrt(2.0) / 0.20

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        ADD_limit=-0.05,
    )

    expected = torch.tensor([expected_fraction], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-6)
    assert _historical_average_drawdown(returns, actual) >= -0.05 - 1e-9


def test_scalar_bootstrap_add_limit_is_solved_exactly_without_grid(
    monkeypatch: pytest.MonkeyPatch,
):
    """The scalar bisection also uses the requested fixed bootstrap paths."""
    returns = torch.tensor([[0.30], [-0.20]], dtype=torch.float64)
    bootstrap_indices = torch.tensor([[0, 1]] * 8, dtype=torch.int64)

    def fixed_bootstrap_indices(*_args, **_kwargs):
        return bootstrap_indices

    monkeypatch.setattr(
        capital_allocation, "_sample_bootstrap_indices", fixed_bootstrap_indices
    )

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=8,
        bootstrap_length=2,
        percentile=10.0,
        ADD_limit=-0.05,
    )

    expected = torch.tensor([0.05 * math.sqrt(2.0) / 0.20], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-6)


def test_max_one_strategy_selects_best_add_constrained_disjoint_support():
    """Every scalar support is evaluated before choosing the global best one."""
    nan = math.nan
    returns = torch.tensor(
        [
            [0.30, nan, nan],
            [-0.20, nan, nan],
            [nan, 0.50, nan],
            [nan, -0.15, nan],
            [nan, nan, 0.40],
            [nan, nan, -0.20],
        ],
        dtype=torch.float64,
    )

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        ADD_limit=-0.05,
        max_strategies=1,
    )

    # Strategy 1 remains underwater for three of the six observations. Its
    # exact boundary is -0.15*f*sqrt(3/6) == -0.05, and its constrained
    # growth is greater than either of the other two scalar supports.
    expected_fraction = 0.05 / (0.15 * math.sqrt(3.0 / 6.0))
    expected = torch.tensor([0.0, expected_fraction, 0.0], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-6)


def test_drawdown_constrained_search_can_complete_without_a_grid():
    """A promising multidimensional ADD search uses spatial branch-and-bound."""
    returns = _drawdown_counterexample_returns()
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        ADD_limit=-0.027,
        allocation_tolerance=1e-4,
        diagnostics=diagnostics,
    )

    assert _historical_average_drawdown(returns, actual) >= -0.027
    assert diagnostics.algorithm == "Adaptive spatial ADD branch-and-bound"


def test_approximate_add_search_scales_the_fixed_kelly_ray_to_feasibility():
    """The explicitly approximate mode may continuously scale the Kelly ray."""
    returns = _drawdown_counterexample_returns()
    add_limit = -0.027
    raw_kelly = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
    )
    exact = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
        ADD_limit=add_limit,
        allocation_tolerance=1e-4,
    )

    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    approximate = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        ADD_limit=add_limit,
        approximate_add_search=True,
        diagnostics=diagnostics,
    )

    assert 0.0 < float(approximate.sum()) < float(raw_kelly.sum())
    assert _historical_average_drawdown(returns, approximate) >= add_limit - 1e-9
    assert _historical_average_drawdown(returns, approximate) == pytest.approx(
        add_limit, abs=2e-6
    )
    torch.testing.assert_close(
        approximate / approximate.sum(),
        raw_kelly / raw_kelly.sum(),
        rtol=0.0,
        atol=2e-6,
    )
    assert diagnostics.algorithm == ("Approximate full-Kelly fixed-ray ADD bisection")
    exact_growth = torch.log1p(returns @ exact).sum()
    approximate_growth = torch.log1p(returns @ approximate).sum()
    assert exact_growth > approximate_growth


def test_approximate_add_search_requires_a_boolean():
    """The opt-in approximation cannot be enabled by truthy non-bool values."""
    returns = _drawdown_counterexample_returns()

    with pytest.raises(TypeError, match="approximate_add_search must be a bool"):
        capital_allocation.find_optimal_capital_allocation(
            returns,
            alpha=0.0,
            device="cpu",
            ADD_limit=-0.027,
            approximate_add_search=1,  # type: ignore[arg-type]
        )


def test_bootstrap_scalar_supports_share_indices_and_reuse_winner_drawdown(
    monkeypatch: pytest.MonkeyPatch,
):
    """Support comparisons are fair and diagnostics do not bootstrap twice."""
    nan = math.nan
    returns = torch.tensor(
        [
            [0.30, nan, nan],
            [-0.20, nan, nan],
            [nan, 0.50, nan],
            [nan, -0.15, nan],
            [nan, nan, 0.40],
            [nan, nan, -0.20],
        ],
        dtype=torch.float64,
    )
    bootstrap_indices = torch.arange(6, dtype=torch.int64).repeat(4, 1)
    sampler_calls = 0
    evaluated_rows: list[tuple[torch.Tensor, float]] = []
    original_bootstrap = capital_allocation._bootstrap_average_drawdown

    def fixed_bootstrap_indices(*_args, **_kwargs):
        nonlocal sampler_calls
        sampler_calls += 1
        return bootstrap_indices

    def recording_bootstrap(
        portfolio_returns: torch.Tensor,
        indices: torch.Tensor,
        percentile: float,
    ) -> torch.Tensor:
        assert indices.data_ptr() == bootstrap_indices.data_ptr()
        drawdowns = original_bootstrap(portfolio_returns, indices, percentile)
        evaluated_rows.extend(
            (row.detach().clone(), float(drawdown.item()))
            for row, drawdown in zip(portfolio_returns, drawdowns, strict=True)
        )
        return drawdowns

    monkeypatch.setattr(
        capital_allocation, "_sample_bootstrap_indices", fixed_bootstrap_indices
    )
    monkeypatch.setattr(
        capital_allocation, "_bootstrap_average_drawdown", recording_bootstrap
    )
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=None,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=4,
        bootstrap_length=6,
        percentile=10.0,
        ADD_limit=-0.05,
        max_strategies=1,
        diagnostics=diagnostics,
    )

    filled_returns = torch.nan_to_num(returns, nan=0.0)
    winning_returns = filled_returns @ actual
    winning_evaluations = [
        drawdown
        for row, drawdown in evaluated_rows
        if torch.equal(row, winning_returns)
    ]
    assert sampler_calls == 1
    assert len(winning_evaluations) == 1
    assert diagnostics.bootstrap_average_drawdown == pytest.approx(
        winning_evaluations[0]
    )
    assert diagnostics.algorithm == "Exact scalar ADD-constrained Kelly (bisection)"


def test_drawdown_penalty_still_requires_a_grid():
    """The nonconcave modified objective retains the grid implementation."""
    returns = torch.tensor([[0.30], [-0.20]], dtype=torch.float64)

    with pytest.raises(ValueError, match="grid_increment"):
        capital_allocation.find_optimal_capital_allocation(
            returns,
            alpha=0.1,
            grid_increment=None,
            device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_continuous_kelly_cuda_matches_cpu():
    """The continuous raw-Kelly path preserves device and matches CPU."""
    returns = torch.tensor(
        [
            [0.30, -0.05, 0.02],
            [-0.20, 0.25, -0.04],
            [0.10, -0.10, 0.08],
            [-0.05, 0.12, -0.02],
        ],
        dtype=torch.float32,
    )

    expected = capital_allocation.find_optimal_capital_allocation(
        returns, alpha=0.0, device="cpu", max_total_allocation=0.8
    )
    actual = capital_allocation.find_optimal_capital_allocation(
        returns, alpha=0.0, device="cuda:0", max_total_allocation=0.8
    )

    assert actual.device == torch.device("cuda:0")
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bootstrap_fixed_ray_add_search_cuda_matches_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    """The new bootstrap-constrained fixed-ray route preserves CUDA results."""
    returns = _drawdown_counterexample_returns().to(dtype=torch.float32)
    fixed_indices = torch.arange(returns.shape[0]).repeat(16, 1)

    def fixed_bootstrap_indices(
        _time_count: int,
        _bootstrap_runs: int,
        _bootstrap_length: int,
        device: torch.device,
        _bootstrap_seed: int | None,
    ) -> torch.Tensor:
        return fixed_indices.to(device=device)

    monkeypatch.setattr(
        capital_allocation, "_sample_bootstrap_indices", fixed_bootstrap_indices
    )
    common = {
        "alpha": 0.0,
        "bootstrap_on": True,
        "bootstrap_runs": 16,
        "bootstrap_length": returns.shape[0],
        "ADD_limit": -0.027,
        "approximate_add_search": True,
    }
    cpu_diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    cuda_diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    expected = capital_allocation.find_optimal_capital_allocation(
        returns,
        device="cpu",
        diagnostics=cpu_diagnostics,
        **common,
    )
    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        device="cuda:0",
        diagnostics=cuda_diagnostics,
        **common,
    )

    assert actual.device == torch.device("cuda:0")
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
    assert cuda_diagnostics.bootstrap_average_drawdown == pytest.approx(
        cpu_diagnostics.bootstrap_average_drawdown, abs=2e-5
    )
