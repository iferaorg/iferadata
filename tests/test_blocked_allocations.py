"""Tests for allocation-dependent blocked strategy trades."""

from __future__ import annotations

import datetime
import itertools
import math
from pathlib import Path

import polars as pl
import pytest
import torch

import ifera.allocation._capital_allocation_memory as allocation_memory
import ifera.allocation.capital_allocation as capital_allocation
import ifera.allocation.portfolio_allocation as portfolio_allocation
from ifera.allocation._capital_allocation_blocks import (
    build_allocation_blocks,
    remap_block_list,
)
from ifera.settings import settings


def _blocked_grid_oracle(
    returns: torch.Tensor,
    grid_increment: float,
    max_total_allocation: float,
    block_list: list[tuple[int, int, int]],
) -> torch.Tensor:
    """Return the best small-grid allocation using independent blocking logic."""
    steps = round(1.0 / grid_increment)
    unit_limit = math.floor(max_total_allocation * steps + 1e-12)
    best_score = -math.inf
    best_allocation: torch.Tensor | None = None
    for units in itertools.product(range(unit_limit + 1), repeat=returns.shape[1]):
        if sum(units) > unit_limit:
            continue
        allocation = torch.tensor(units, dtype=returns.dtype) / steps
        effective_returns = returns.clone()
        for day_index, blocked, blocked_by in block_list:
            if allocation[blocked_by] != 0.0:
                effective_returns[day_index, blocked] = 0.0
        daily_returns = effective_returns @ allocation
        if bool(torch.any(daily_returns <= -1.0)):
            continue
        score = float(torch.log1p(daily_returns).sum())
        if score > best_score:
            best_score = score
            best_allocation = allocation

    assert best_allocation is not None
    return best_allocation


def test_blocker_at_exact_zero_does_not_suppress_the_blocked_strategy():
    """Zero is a distinct boundary: only a positive blocker allocation blocks."""
    returns = torch.tensor([[1.0, 0.2]], dtype=torch.float64)

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=1e-30,
        grid_increment=0.5,
        device="cpu",
        block_list=[(0, 0, 1)],
    )

    torch.testing.assert_close(actual, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_multiple_blockers_use_or_semantics_without_double_suppression():
    """Several active blockers remove one strategy-day return exactly once."""
    returns = torch.tensor([[0.1, -0.4, 0.8], [0.2, 0.8, -0.4]], dtype=torch.float64)
    block_list = [(0, 0, 1), (0, 0, 2)]
    expected = _blocked_grid_oracle(
        returns,
        grid_increment=1.0,
        max_total_allocation=3.0,
        block_list=block_list,
    )

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=1e-30,
        grid_increment=1.0,
        max_total_allocation=3.0,
        device="cpu",
        block_list=block_list,
    )

    torch.testing.assert_close(
        expected, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("block_list", "error_type", "message"),
    [
        ([(0, 1)], ValueError, "three"),
        ([(-1, 0, 1)], ValueError, "day"),
        ([(1, 0, 1)], ValueError, "day"),
        ([(0, -1, 1)], ValueError, "blocked index"),
        ([(0, 0, 2)], ValueError, "blocked_by index"),
        ([(0, 1, 1)], ValueError, "same|itself"),
        ([(0.0, 0, 1)], TypeError, "integer"),
    ],
)
def test_invalid_block_list_is_rejected(
    block_list: list[tuple[int, ...]],
    error_type: type[Exception],
    message: str,
):
    """Malformed and out-of-range trade relationships fail before searching."""
    returns = torch.tensor([[0.2, 0.1]], dtype=torch.float64)

    with pytest.raises(error_type, match=message):
        capital_allocation.find_optimal_capital_allocation(
            returns,
            alpha=0.0,
            grid_increment=1.0,
            device="cpu",
            block_list=block_list,
        )


def test_blocked_continuous_kelly_enforces_the_exact_support_floor():
    """Each active blocker stays at the requested positive allocation floor."""
    returns = torch.tensor([[1.0, 0.1], [-0.8, 0.1]], dtype=torch.float64)
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    allocation = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
        allocation_tolerance=0.01,
        block_list=[(1, 0, 1)],
        diagnostics=diagnostics,
    )

    torch.testing.assert_close(
        allocation, torch.tensor([0.99, 0.01], dtype=torch.float64), atol=2e-7, rtol=0
    )
    assert diagnostics.algorithm is not None
    assert "exact-support" in diagnostics.algorithm.lower()
    assert "kelly" in diagnostics.algorithm.lower()


def test_blocked_support_floor_survives_output_dtype_casting():
    """Casting rounds an active coordinate upward rather than back to zero."""
    allocation = capital_allocation.find_optimal_capital_allocation(
        torch.tensor([[1.0, 0.1], [-0.8, 0.1]], dtype=torch.float64),
        alpha=0.0,
        dtype=torch.float32,
        device="cpu",
        allocation_tolerance=0.1,
        block_list=[(1, 0, 1)],
    )

    assert allocation.dtype == torch.float32
    assert float(allocation[1].to(dtype=torch.float64).item()) >= 0.1


def test_blocked_continuous_kelly_compares_exact_zero_subsupports():
    """A singleton can beat every positive-blocker regime and remain exactly zero."""
    allocation = capital_allocation.find_optimal_capital_allocation(
        torch.tensor([[0.3, -0.99], [-0.01, -0.99]], dtype=torch.float64),
        alpha=0.0,
        device="cpu",
        allocation_tolerance=0.01,
        block_list=[(1, 0, 1)],
    )

    torch.testing.assert_close(
        allocation, torch.tensor([1.0, 0.0], dtype=torch.float64), atol=2e-7, rtol=0
    )


def test_blocked_support_guard_falls_back_to_the_original_grid():
    """An exponential support expansion is abandoned before continuous solves."""
    strategy_count = 9
    returns = torch.full((2, strategy_count), 0.01, dtype=torch.float64)
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    allocation = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=1.0,
        device="cpu",
        block_list=[(0, 0, 1)],
        diagnostics=diagnostics,
    )

    assert torch.count_nonzero(allocation) == 1
    assert diagnostics.algorithm == "Blocking-aware Raw-Kelly allocation grid fallback"

    with pytest.raises(ValueError, match="grid_increment.*blocking"):
        capital_allocation.find_optimal_capital_allocation(
            returns,
            alpha=0.0,
            device="cpu",
            block_list=[(0, 0, 1)],
        )


def test_blocked_support_guard_counts_only_selection_compatible_supports():
    """A cardinality cap avoids rejecting a large but cheaply enumerable input."""
    returns = torch.full((2, 8), 0.01, dtype=torch.float64)
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    allocation = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        device="cpu",
        max_strategies=1,
        block_list=[(0, 0, 1)],
        diagnostics=diagnostics,
    )

    assert torch.count_nonzero(allocation) == 1
    assert diagnostics.algorithm is not None
    assert "exact-support" in diagnostics.algorithm.lower()


def test_blocked_support_approximate_add_ray_never_crosses_the_floor():
    """The optional ray starts at the exact-support floor, not at cash."""
    returns = torch.tensor(
        [[0.5, 0.01], [-0.1, 0.01], [-0.1, 0.01]], dtype=torch.float64
    )
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    allocation = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        ADD_limit=-0.03,
        approximate_add_search=True,
        device="cpu",
        allocation_tolerance=0.01,
        block_list=[(1, 0, 1)],
        diagnostics=diagnostics,
    )

    assert allocation[0].item() > 0.4
    assert allocation[1].item() == pytest.approx(0.01)
    effective_returns = returns.clone()
    effective_returns[1, 0] = 0.0
    drawdown = capital_allocation._negative_rms_drawdown(
        torch.log1p(effective_returns @ allocation), dim=0
    )
    assert float(drawdown.item()) >= -0.03
    assert diagnostics.algorithm is not None
    assert "approximate" in diagnostics.algorithm.lower()


def test_blocked_raw_kelly_add_precheck_does_not_require_a_grid():
    """A support-wise raw winner that meets ADD remains exact globally."""
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    allocation = capital_allocation.find_optimal_capital_allocation(
        torch.tensor([[1.0, 0.1], [-0.8, 0.1]], dtype=torch.float64),
        alpha=0.0,
        ADD_limit=-0.01,
        device="cpu",
        allocation_tolerance=0.01,
        block_list=[(1, 0, 1)],
        diagnostics=diagnostics,
    )

    torch.testing.assert_close(
        allocation, torch.tensor([0.99, 0.01], dtype=torch.float64), atol=2e-7, rtol=0
    )
    assert diagnostics.algorithm is not None
    assert "exact-support" in diagnostics.algorithm.lower()
    assert "add" in diagnostics.algorithm.lower()


def test_blocked_support_spatial_add_search_handles_a_binding_limit():
    """Fixed-support boxes retain valid ADD bounds above their positive floors."""
    returns = torch.tensor(
        [[0.5, 0.01], [-0.1, 0.01], [-0.1, 0.01]], dtype=torch.float64
    )
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    allocation = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        ADD_limit=-0.03,
        device="cpu",
        allocation_tolerance=0.01,
        block_list=[(1, 0, 1)],
        diagnostics=diagnostics,
    )

    assert torch.all(allocation >= 0.01)
    assert allocation.sum() <= 1.0
    effective_returns = returns.clone()
    effective_returns[1, 0] = 0.0
    portfolio_returns = effective_returns @ allocation
    drawdown = capital_allocation._negative_rms_drawdown(
        torch.log1p(portfolio_returns), dim=0
    )
    assert float(drawdown.item()) >= -0.03
    assert float(torch.log1p(portfolio_returns).sum().item()) > 0.15
    assert diagnostics.algorithm is not None
    assert "exact-support" in diagnostics.algorithm.lower()
    assert "branch-and-bound" in diagnostics.algorithm.lower()


def test_blocked_support_bootstrap_indices_are_shared_across_subsupports(
    monkeypatch: pytest.MonkeyPatch,
):
    """One bootstrap sample is reused for every exact-support ADD comparison."""
    bootstrap_indices = torch.arange(3, dtype=torch.int64).repeat(8, 1)
    randint_calls = 0

    def fixed_randint(high, size, *, device, dtype, **_kwargs):
        nonlocal randint_calls
        randint_calls += 1
        assert high == 3
        assert size == (8, 3)
        return bootstrap_indices.to(device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    returns = torch.tensor(
        [[0.5, 0.01], [-0.1, 0.01], [-0.1, 0.01]], dtype=torch.float64
    )

    allocation = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        ADD_limit=-0.03,
        bootstrap_on=True,
        bootstrap_runs=8,
        bootstrap_length=3,
        percentile=10.0,
        device="cpu",
        allocation_tolerance=0.01,
        block_list=[(1, 0, 1)],
        diagnostics=diagnostics,
    )

    assert randint_calls == 1
    assert torch.all(allocation >= 0.01)
    effective_returns = returns.clone()
    effective_returns[1, 0] = 0.0
    expected_add = capital_allocation._bootstrap_average_drawdown(
        (effective_returns @ allocation).unsqueeze(0),
        bootstrap_indices,
        10.0,
    )[0]
    assert diagnostics.bootstrap_average_drawdown == pytest.approx(
        float(expected_add.item())
    )


def test_empty_block_list_preserves_the_continuous_kelly_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """An explicitly empty list has no effect on fast-path eligibility."""
    sentinel = torch.tensor([0.25], dtype=torch.float64)

    def continuous_solver(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(
        capital_allocation, "_maximize_kelly_with_selection", continuous_solver
    )

    actual = capital_allocation.find_optimal_capital_allocation(
        torch.tensor([[0.1]], dtype=torch.float64),
        alpha=0.0,
        device="cpu",
        block_list=[],
    )

    assert actual is sentinel


def test_blocked_returns_are_used_by_historical_add_limit():
    """The hard drawdown constraint sees the candidate-specific effective path."""
    returns = torch.tensor([[1.0, 0.2], [-0.5, 0.2]], dtype=torch.float64)

    blocked = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=1e-30,
        grid_increment=1.0,
        max_total_allocation=2.0,
        ADD_limit=-0.05,
        device="cpu",
        block_list=[(1, 0, 1)],
    )
    unblocked = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=1e-30,
        grid_increment=1.0,
        max_total_allocation=2.0,
        ADD_limit=-0.05,
        device="cpu",
    )

    torch.testing.assert_close(blocked, torch.tensor([1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(unblocked, torch.tensor([0.0, 2.0], dtype=torch.float64))


def test_final_winner_bootstrap_diagnostics_use_blocked_returns(
    monkeypatch: pytest.MonkeyPatch,
):
    """Diagnostic-only bootstrap ADD does not reconstruct an unblocked path."""
    bootstrap_indices = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)

    def fixed_randint(high, size, *, device, dtype, **_kwargs):
        assert high == 2
        assert size == (2, 2)
        return bootstrap_indices.to(device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    allocation = capital_allocation.find_optimal_capital_allocation(
        torch.tensor([[1.0, 0.2], [-0.5, 0.2]], dtype=torch.float64),
        alpha=1e-30,
        grid_increment=1.0,
        max_total_allocation=2.0,
        bootstrap_on=True,
        bootstrap_runs=2,
        bootstrap_length=2,
        diagnostics=diagnostics,
        device="cpu",
        block_list=[(1, 0, 1)],
    )

    torch.testing.assert_close(
        allocation, torch.tensor([1.0, 1.0], dtype=torch.float64)
    )
    assert diagnostics.bootstrap_average_drawdown == pytest.approx(0.0)


def test_bootstrap_add_limit_uses_blocked_returns(
    monkeypatch: pytest.MonkeyPatch,
):
    """The hard ADD floor uses the shared bootstrap paths after trade suppression."""
    bootstrap_indices = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)

    def fixed_randint(high, size, *, device, dtype, **_kwargs):
        assert high == 2
        assert size == (2, 2)
        return bootstrap_indices.to(device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    allocation = capital_allocation.find_optimal_capital_allocation(
        torch.tensor([[1.0, 0.2], [-0.5, 0.2]], dtype=torch.float64),
        alpha=1e-30,
        grid_increment=1.0,
        max_total_allocation=2.0,
        ADD_limit=-0.05,
        bootstrap_on=True,
        bootstrap_runs=2,
        bootstrap_length=2,
        device="cpu",
        block_list=[(1, 0, 1)],
    )

    torch.testing.assert_close(
        allocation, torch.tensor([1.0, 1.0], dtype=torch.float64)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_cuda_blocking_matches_eager_with_duplicate_day_indices():
    """Compiled scatter correction matches eager scoring for grouped CUDA blocks."""
    device = torch.device("cuda:0")
    allocations = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.5, 0.5], [0.5, 0.5, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    transposed_returns = torch.tensor(
        [[0.4, -0.3], [0.1, 0.2], [-0.2, 0.3]],
        device=device,
        dtype=torch.float32,
    )
    blocks = build_allocation_blocks(
        ((0, 0, 1), (0, 0, 2), (0, 2, 1)),
        strategy_count=3,
        device=device,
        dtype=torch.float32,
    )
    assert blocks is not None
    bootstrap_indices = (
        torch.arange(17 * 7, device=device, dtype=torch.int64).reshape(17, 7) * 3
    ) % transposed_returns.shape[1]
    score_kwargs = {
        "bootstrap_indices": bootstrap_indices,
        "percentile": 10.0,
        "add_limit": -0.9,
        "block_day_indices": blocks.day_indices,
        "blocked_strategy_indices": blocks.blocked_indices,
        "blocker_incidence": blocks.blocker_incidence,
    }

    with torch.compiler.set_stance("force_eager"):
        expected = capital_allocation._allocation_scores(
            allocations, transposed_returns, 0.3, **score_kwargs
        )
    actual = capital_allocation._allocation_scores(
        allocations, transposed_returns, 0.3, **score_kwargs
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)


def test_block_workspace_reduces_the_planned_candidate_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    """Candidate planning reserves vectorized blocker-incidence intermediates."""
    monkeypatch.setattr(allocation_memory, "_BATCH_MEMORY_BUDGET_BYTES", 100_000)
    common = {
        "time_count": 10,
        "strategy_count": 5,
        "clique_count": 1,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "uses_prefix_expansion": False,
        "bootstrap_runs": 0,
        "bootstrap_length": 0,
    }

    ordinary = allocation_memory.candidate_batch_size(**common)
    blocked = allocation_memory.candidate_batch_size(**common, block_group_count=100)

    assert blocked < ordinary


def test_strategy_subset_blocks_are_filtered_and_remapped():
    """Counterfactual WFA support regimes receive their own strategy indexing."""
    blocks = ((0, 3, 1), (1, 2, 3), (2, 0, 2))

    remapped = remap_block_list(blocks, (1, 3))

    assert remapped == ((0, 1, 0),)


def _portfolio_directory(tmp_path: Path, portfolio_name: str) -> Path:
    """Create the directory layout expected by the CSV allocation facade."""
    directory = tmp_path / "results" / "portfolios" / portfolio_name
    directory.mkdir(parents=True)
    return directory


def _write_block_strategy(
    path: Path,
    opened: list[str],
    closed: list[str],
    expiration: list[str],
    legs: list[str],
    returns_on_risk: list[float],
) -> None:
    """Write the CSV fields used by return loading and trade-block detection."""
    pl.DataFrame(
        {
            "Opened": opened,
            "Closed": closed,
            "Symbol": ["SPX"] * len(opened),
            "Exp": expiration,
            "Legs": legs,
            "ROR": returns_on_risk,
        }
    ).write_csv(path)


def _capturing_optimizer(captured: dict[str, object], allocation: list[float]):
    """Return a facade-compatible optimizer that records keyword arguments."""

    def optimizer(
        returns: torch.Tensor,
        alpha: float,
        grid_increment: float | None,
        **kwargs: object,
    ) -> torch.Tensor:
        captured.update(kwargs)
        captured["returns"] = returns.clone()
        diagnostics = kwargs.get("diagnostics")
        if isinstance(diagnostics, capital_allocation.CapitalAllocationDiagnostics):
            diagnostics.algorithm = "Blocked-trade test optimizer"
        return torch.tensor(allocation, dtype=returns.dtype)

    return optimizer


def test_csv_wrapper_detects_conflicting_legs_and_applies_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """CSV trade intervals produce day-local blocks used by optimization and output."""
    directory = _portfolio_directory(tmp_path, "blocked_trades")
    _write_block_strategy(
        directory / "alpha.csv",
        ["Jan 6, 2025 10:00am", "Jan 7, 2025 10:00am"],
        ["Jan 6, 2025 11:00am", "Jan 7, 2025 11:00am"],
        ["Jan 6, 2025", "Jan 7, 2025"],
        [
            (
                "long put 4,900 / short put 4,950 / "
                "short call 5,000 / long call 5,050"
            ),
            (
                "long put 4,900 / short put 4,950 / "
                "short call 5,000 / long call 5,050"
            ),
        ],
        [10.0, 10.0],
    )
    _write_block_strategy(
        directory / "beta.csv",
        # The first trade opens later; the second ties and loses by strategy index.
        ["Jan 6, 2025 10:30am", "Jan 7, 2025 10:00am"],
        ["Jan 6, 2025 11:30am", "Jan 7, 2025 11:30am"],
        ["Jan 6, 2025", "Jan 7, 2025"],
        [
            "long call 5,000 / short call 5,100",
            "long call 5,000 / short call 5,100",
        ],
        [20.0, 20.0],
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _capturing_optimizer(captured, [0.5, 0.5]),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "blocked_trades",
        alpha=0.0,
        grid_increment=0.5,
        device="cpu",
        show_plot=False,
    )

    assert captured["block_list"] == [(0, 1, 0), (1, 1, 0)]
    assert result.daily_results["combined_return"].to_list() == pytest.approx(
        [0.05, 0.05]
    )


def test_csv_wrapper_warns_and_aggregates_multiple_trades_per_strategy_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The documented simplified duplicate-day behavior warns instead of rejecting."""
    directory = _portfolio_directory(tmp_path, "duplicate_trade_day")
    _write_block_strategy(
        directory / "alpha.csv",
        ["Jan 6, 2025 10:00am", "Jan 6, 2025 12:00pm"],
        ["Jan 6, 2025 11:00am", "Jan 6, 2025 1:00pm"],
        ["Jan 6, 2025", "Jan 6, 2025"],
        ["long call 5,000", "long put 4,900"],
        [10.0, 20.0],
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _capturing_optimizer(captured, [1.0]),
    )

    with pytest.warns(UserWarning, match="multiple trades|more than once"):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            "duplicate_trade_day",
            alpha=0.0,
            grid_increment=1.0,
            device="cpu",
            show_plot=False,
        )

    torch.testing.assert_close(
        captured["returns"], torch.tensor([[0.3]], dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("opened", "closed", "symbol", "expiration", "leg", "expected"),
    [
        (
            "Jan 6, 2025 10:30am",
            "Jan 6, 2025 11:30am",
            "spx",
            "Jan 6, 2025",
            "long call 5,000",
            ((0, 1, 0),),
        ),
        (
            "Jan 6, 2025 11:00am",
            "Jan 6, 2025 12:00pm",
            "SPX",
            "Jan 6, 2025",
            "long call 5,000",
            (),
        ),
        (
            "Jan 6, 2025 10:30am",
            "Jan 6, 2025 11:30am",
            "SPX",
            "Jan 7, 2025",
            "long call 5,000",
            (),
        ),
        (
            "Jan 6, 2025 10:30am",
            "Jan 6, 2025 11:30am",
            "SPX",
            "Jan 6, 2025",
            "short call 5,000",
            (),
        ),
        (
            "Jan 6, 2025 10:30am",
            "Jan 6, 2025 11:30am",
            "NDX",
            "Jan 6, 2025",
            "long call 5,000",
            (),
        ),
        (
            "Jan 6, 2025 10:30am",
            "Jan 6, 2025 11:30am",
            "SPX",
            "Jan 6, 2025",
            "long put 5,000",
            (),
        ),
        (
            "Jan 6, 2025 10:30am",
            "Jan 6, 2025 11:30am",
            "SPX",
            "Jan 6, 2025",
            "long call 5,005",
            (),
        ),
    ],
    ids=(
        "conflict",
        "touching-intervals",
        "expiration",
        "direction",
        "symbol",
        "right",
        "strike",
    ),
)
def test_csv_block_detection_requires_every_conflict_criterion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened: str,
    closed: str,
    symbol: str,
    expiration: str,
    leg: str,
    expected: tuple[tuple[int, int, int], ...],
):
    """Symbol, expiry, overlap, right/strike, and opposite direction all matter."""
    directory = _portfolio_directory(tmp_path, "block_criteria")

    def write_one(
        name: str,
        opened: str,
        closed: str,
        symbol: str,
        expiration: str,
        leg: str,
    ) -> None:
        pl.DataFrame(
            {
                "Opened": [opened],
                "Closed": [closed],
                "Symbol": [symbol],
                "Exp": [expiration],
                "Legs": [leg],
                "ROR": [1.0],
            }
        ).write_csv(directory / f"{name}.csv")

    write_one(
        "s0",
        "Jan 6, 2025 10:00am",
        "Jan 6, 2025 11:00am",
        "SPX",
        "Jan 6, 2025",
        "short call 5,000",
    )
    write_one("s1", opened, closed, symbol, expiration, leg)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    _, _, _, blocks = portfolio_allocation._load_portfolio_returns(
        "block_criteria", pl.Float32
    )

    assert blocks == expected


@pytest.mark.parametrize(
    ("extra_columns", "message"),
    [
        ({"Closed": ["Jan 6, 2025 11:00am"]}, "incomplete"),
        (
            {
                "Closed": ["Jan 6, 2025 11:00am"],
                "Symbol": ["SPX"],
                "Exp": ["Jan 6, 2025"],
                "Legs": ["sell call 5,000"],
            },
            "Legs",
        ),
    ],
)
def test_csv_block_metadata_errors_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_columns: dict[str, list[str]],
    message: str,
):
    """Partial metadata and nonstandard legs fail instead of silently misblocking."""
    directory = _portfolio_directory(tmp_path, "invalid_block_metadata")
    columns = {
        "Opened": ["Jan 6, 2025 10:00am"],
        "ROR": [1.0],
        **extra_columns,
    }
    pl.DataFrame(columns).write_csv(directory / "strategy.csv")
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match=message):
        portfolio_allocation._load_portfolio_returns(
            "invalid_block_metadata", pl.Float32
        )


def test_walk_forward_rebases_blocks_for_each_training_and_simulation_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Every fold receives zero-based training blocks and applies simulation blocks."""
    directory = _portfolio_directory(tmp_path, "walk_forward_blocks")
    market_dates = portfolio_allocation._nyse_market_dates(
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 24),
    )["date"].to_list()
    opened_alpha = [f"{market_date} 10:00:00" for market_date in market_dates]
    opened_beta = [f"{market_date} 10:30:00" for market_date in market_dates]
    closed_alpha = [f"{market_date} 11:00:00" for market_date in market_dates]
    closed_beta = [f"{market_date} 11:30:00" for market_date in market_dates]
    expiration = [str(market_date) for market_date in market_dates]
    _write_block_strategy(
        directory / "alpha.csv",
        opened_alpha,
        closed_alpha,
        expiration,
        ["short call 5,000"] * len(market_dates),
        [1.0] * len(market_dates),
    )
    _write_block_strategy(
        directory / "beta.csv",
        opened_beta,
        closed_beta,
        expiration,
        ["long call 5,000"] * len(market_dates),
        [2.0] * len(market_dates),
    )
    calls: list[dict[str, object]] = []

    def optimizer(
        returns: torch.Tensor,
        alpha: float,
        grid_increment: float | None,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append({"returns": returns.clone(), **kwargs})
        diagnostics = kwargs.get("diagnostics")
        if isinstance(diagnostics, capital_allocation.CapitalAllocationDiagnostics):
            diagnostics.algorithm = "Walk-forward blocked-trade test optimizer"
        return torch.tensor([0.5, 0.5], dtype=returns.dtype)

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )

    result = capital_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_blocks",
        alpha=0.0,
        grid_increment=0.5,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        device="cpu",
        show_plot=False,
    )

    assert len(calls) == 2
    for call in calls:
        training_returns = call["returns"]
        assert isinstance(training_returns, torch.Tensor)
        expected_training_blocks = [
            (day_index, 1, 0) for day_index in range(training_returns.shape[0])
        ]
        assert call["block_list"] == expected_training_blocks
    assert result.daily_results["combined_return"].to_list() == pytest.approx(
        [0.005] * result.daily_results.height
    )


def test_pf_003_examples_include_the_known_four_leg_conflict():
    """The example portfolio exercises the standardized arbitrary-leg parser."""
    strategy_names, _, _, block_list = portfolio_allocation._load_portfolio_returns(
        "pf_003", pl.Float32
    )

    assert strategy_names[0] == "145 Sandwitch"
    assert strategy_names[4] == "ORB60_low"
    assert (7, 0, 4) in block_list
