"""Tests for adaptive ADD search and its shared allocation tolerance."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch

import ifera.allocation.capital_allocation as capital_allocation
import ifera.allocation.portfolio_allocation as portfolio_allocation
from ifera.allocation import _kelly_optimizer, _kelly_selection
from ifera.settings import settings


def _drawdown_counterexample_returns() -> torch.Tensor:
    """Return a multidimensional sample with a binding ADD constraint."""
    return torch.tensor(
        [
            [0.47, 0.06],
            [-0.25, 0.34],
            [0.17, 0.01],
            [0.08, -0.10],
        ],
        dtype=torch.float64,
    )


def _historical_average_drawdown(
    returns: torch.Tensor, allocation: torch.Tensor
) -> float:
    """Calculate historical ADD with the production definition."""
    portfolio_returns = torch.nan_to_num(returns, nan=0.0) @ allocation
    log_equity = torch.log1p(portfolio_returns).cumsum(dim=0)
    running_peak = torch.cummax(log_equity, dim=0).values.clamp_min(0.0)
    drawdowns = torch.expm1(log_equity - running_peak)
    return -float(torch.sqrt(torch.mean(drawdowns.square())).item())


def _frank_wolfe_returns() -> torch.Tensor:
    """Return a small problem whose optimum takes several allocation moves."""
    return torch.tensor(
        [
            [0.1460085286689272, 0.057971125505950495, -0.039352146840909756],
            [0.06544926960583038, -0.052188308232762604, -0.2648050820807074],
            [0.09852321079557312, 0.011071181599356786, 0.039979283676221146],
            [0.11217302930325501, -0.11861208761038104, -0.059049857316087855],
            [-0.18134337869374204, 0.043077978530683916, 0.05073303894675572],
            [-0.1503077118822718, 0.1248430781219828, -0.1301222500487608],
        ],
        dtype=torch.float64,
    ).T.contiguous()


def _write_portfolio(path: Path, opened: list[str]) -> None:
    """Write one minimal strategy CSV spanning the requested observations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    returns_on_risk = [1.0 if index % 2 == 0 else -0.5 for index in range(len(opened))]
    pl.DataFrame({"Opened": opened, "ROR": returns_on_risk}).write_csv(path)


def test_allocation_tolerance_controls_frank_wolfe_result_accuracy():
    """The shared threshold measures allocation movement, not objective scale."""
    returns = _frank_wolfe_returns()
    expected = torch.tensor(
        [0.46011368182125806, 0.539886318178742, 0.0], dtype=torch.float64
    )

    coarse = _kelly_optimizer.maximize_kelly_growth(
        returns,
        1.0,
        "simplex",
        allocation_tolerance=0.1,
    )
    fine = _kelly_optimizer.maximize_kelly_growth(
        returns,
        1.0,
        "simplex",
        allocation_tolerance=1e-5,
    )

    assert coarse.iterations < fine.iterations
    assert torch.amax(torch.abs(coarse.allocation - expected)).item() <= 0.1
    torch.testing.assert_close(fine.allocation, expected, rtol=0.0, atol=1e-5)


def test_core_forwards_allocation_tolerance_to_every_frank_wolfe_solve(
    monkeypatch: pytest.MonkeyPatch,
):
    """The tensor entry point and support-selection layer share one threshold."""
    observed: list[float | None] = []
    original_solver = _kelly_selection.maximize_kelly_growth

    def recording_solver(*args: object, **kwargs: object):
        observed.append(kwargs.get("allocation_tolerance"))
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(_kelly_selection, "maximize_kelly_growth", recording_solver)

    capital_allocation.find_optimal_capital_allocation(
        _frank_wolfe_returns().T,
        alpha=0.0,
        device="cpu",
        allocation_tolerance=2e-4,
    )

    assert observed
    assert observed == [2e-4] * len(observed)


@pytest.mark.parametrize("invalid", [0.0, -1e-4, math.nan, math.inf])
def test_allocation_tolerance_must_be_finite_and_positive(invalid: float):
    """A nonpositive or nonfinite stopping threshold is ambiguous."""
    with pytest.raises(ValueError, match="allocation_tolerance"):
        capital_allocation.find_optimal_capital_allocation(
            _drawdown_counterexample_returns(),
            alpha=0.0,
            device="cpu",
            allocation_tolerance=invalid,
        )


@pytest.mark.parametrize("invalid", [True, "0.001", object()])
def test_allocation_tolerance_rejects_non_real_values(invalid: object):
    """Booleans and coercible strings are not accepted as numeric thresholds."""
    with pytest.raises(TypeError, match="allocation_tolerance must be"):
        capital_allocation.find_optimal_capital_allocation(
            _drawdown_counterexample_returns(),
            alpha=0.0,
            device="cpu",
            allocation_tolerance=invalid,  # type: ignore[arg-type]
        )


def test_effective_add_pilot_bypasses_grid_and_reports_branch_algorithm(
    monkeypatch: pytest.MonkeyPatch,
):
    """A successful pilot outcome becomes the result without grid scoring."""
    expected = torch.tensor([0.4, 0.2], dtype=torch.float64)
    observed: dict[str, object] = {}

    def successful_pilot(*_args: object, **kwargs: object):
        observed.update(kwargs)
        return SimpleNamespace(
            allocation=expected,
            algorithm="Adaptive spatial ADD branch-and-bound",
            bootstrap_average_drawdown=None,
        )

    def unexpected_grid_score(*_args: object, **_kwargs: object):
        raise AssertionError("a successful branch-and-bound search bypasses the grid")

    monkeypatch.setattr(
        capital_allocation,
        "_try_add_branch_and_bound",
        successful_pilot,
        raising=False,
    )
    monkeypatch.setattr(capital_allocation, "_allocation_scores", unexpected_grid_score)
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    actual = capital_allocation.find_optimal_capital_allocation(
        _drawdown_counterexample_returns(),
        alpha=0.0,
        grid_increment=0.25,
        device="cpu",
        ADD_limit=-0.027,
        allocation_tolerance=5e-4,
        diagnostics=diagnostics,
    )

    torch.testing.assert_close(actual, expected)
    assert observed["allocation_tolerance"] == 5e-4
    assert diagnostics.algorithm is not None
    assert "branch-and-bound" in diagnostics.algorithm.casefold()
    assert "grid" not in diagnostics.algorithm.casefold()


def test_ineffective_add_pilot_falls_back_to_grid_and_reports_both_algorithms(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unpromising bounded pilot adds little work and preserves grid behavior."""
    pilot_calls: list[dict[str, object]] = []

    def ineffective_pilot(*_args: object, **kwargs: object):
        pilot_calls.append(kwargs)
        return None

    monkeypatch.setattr(
        capital_allocation,
        "_try_add_branch_and_bound",
        ineffective_pilot,
        raising=False,
    )
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()

    actual = capital_allocation.find_optimal_capital_allocation(
        _drawdown_counterexample_returns(),
        alpha=0.0,
        grid_increment=0.25,
        device="cpu",
        ADD_limit=-0.027,
        allocation_tolerance=5e-4,
        diagnostics=diagnostics,
    )

    assert pilot_calls
    assert pilot_calls[0]["allocation_tolerance"] == 5e-4
    torch.testing.assert_close(actual, torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert diagnostics.algorithm is not None
    assert "pilot" in diagnostics.algorithm.casefold()
    assert "grid" in diagnostics.algorithm.casefold()


def test_add_search_result_always_satisfies_the_hard_limit():
    """Adaptive search and its fallback never return an infeasible incumbent."""
    returns = _drawdown_counterexample_returns()
    add_limit = -0.027

    actual = capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        grid_increment=0.1,
        device="cpu",
        ADD_limit=add_limit,
        allocation_tolerance=1e-4,
    )

    assert torch.count_nonzero(actual) > 0
    assert _historical_average_drawdown(returns, actual) >= add_limit - 1e-10


def test_csv_wrapper_forwards_tolerance_and_reports_selected_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """The direct CSV result identifies the algorithm selected by its search."""
    _write_portfolio(
        tmp_path / "results" / "portfolios" / "direct" / "strategy.csv",
        ["Jan 6, 2025 10:00am", "Jan 7, 2025 10:00am"],
    )
    observed: list[float | None] = []

    def optimizer(
        returns: torch.Tensor,
        *_args: object,
        diagnostics: capital_allocation.CapitalAllocationDiagnostics | None = None,
        allocation_tolerance: float | None = None,
        **_kwargs: object,
    ) -> torch.Tensor:
        observed.append(allocation_tolerance)
        if diagnostics is not None:
            diagnostics.algorithm = "Adaptive spatial ADD branch-and-bound"
        return torch.tensor([0.5], dtype=returns.dtype)

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )

    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "direct",
        alpha=0.0,
        grid_increment=0.25,
        device="cpu",
        allocation_tolerance=3e-4,
        show_plot=False,
    )

    assert observed == [3e-4]
    assert result.optimization_algorithm_counts == (
        ("Adaptive spatial ADD branch-and-bound", 1),
    )
    assert "Optimization algorithm: Adaptive spatial ADD branch-and-bound" in (
        capsys.readouterr().out
    )


def test_wfa_forwards_tolerance_each_fold_and_reports_mixed_algorithms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Walk-forward reporting identifies branch continuation and grid fallback."""
    _write_portfolio(
        tmp_path / "results" / "portfolios" / "rolling" / "strategy.csv",
        ["Jan 6, 2025 10:00am", "Jan 31, 2025 10:00am"],
    )
    observed: list[float | None] = []

    def optimizer(
        returns: torch.Tensor,
        *_args: object,
        diagnostics: capital_allocation.CapitalAllocationDiagnostics | None = None,
        allocation_tolerance: float | None = None,
        **_kwargs: object,
    ) -> torch.Tensor:
        observed.append(allocation_tolerance)
        if diagnostics is not None:
            diagnostics.algorithm = (
                "Adaptive spatial ADD branch-and-bound"
                if len(observed) % 2
                else "ADD branch-and-bound pilot + allocation grid fallback"
            )
        return torch.tensor([0.5], dtype=returns.dtype)

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )

    result = portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "rolling",
        alpha=0.0,
        grid_increment=0.25,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        device="cpu",
        allocation_tolerance=7e-4,
        show_plot=False,
    )

    assert len(observed) == len(result.folds)
    assert observed == [7e-4] * len(observed)
    assert result.optimization_algorithm_counts == (
        ("Adaptive spatial ADD branch-and-bound", 2),
        ("ADD branch-and-bound pilot + allocation grid fallback", 1),
    )
    output = capsys.readouterr().out
    assert "Optimization algorithms used:" in output
    assert "Adaptive spatial ADD branch-and-bound: 2 optimizer calls" in output
    assert (
        "ADD branch-and-bound pilot + allocation grid fallback: 1 optimizer calls"
        in output
    )
