"""Tests for selecting constrained strategy subsets in CSV portfolios."""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import polars as pl
import pytest
import torch

import ifera.allocation as allocation
import ifera.allocation.capital_allocation as capital_allocation
import ifera.allocation.portfolio_allocation as portfolio_allocation
import ifera.allocation.walk_forward_allocation as walk_forward_allocation
from ifera.settings import settings


def _portfolio_directory(tmp_path: Path, portfolio_name: str) -> Path:
    """Create one temporary portfolio directory."""
    directory = tmp_path / "results" / "portfolios" / portfolio_name
    directory.mkdir(parents=True)
    return directory


def _write_fractional_returns(
    path: Path,
    dates: list[datetime.date],
    fractional_returns: list[float],
) -> None:
    """Write dated fractional returns in the CSV wrapper's percentage format."""
    pl.DataFrame(
        {
            "Opened": [f"{market_date.isoformat()} 10:00:00" for market_date in dates],
            "ROR": [100.0 * value for value in fractional_returns],
        }
    ).write_csv(path)


def _market_dates(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Return NYSE dates as a typed Python list."""
    return portfolio_allocation._nyse_market_dates(start, end)["date"].to_list()


def _write_correlation_case(directory: Path) -> None:
    """Write a pair whose correlation is just above 0.4 and whose mix is best."""
    dates = _market_dates(datetime.date(2025, 1, 6), datetime.date(2025, 1, 16))
    _write_fractional_returns(
        directory / "alpha.csv",
        dates,
        [0.17, 0.03, -0.31, 0.22, 0.54, 0.31, -0.42, 0.16],
    )
    _write_fractional_returns(
        directory / "beta.csv",
        dates,
        [0.0, -0.31, -0.22, -0.28, 0.47, 0.36, 0.20, 0.39],
    )


def _run_regime_walk_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    portfolio_name: str,
    winning_allocations: list[list[float]],
    regime_targets: dict[tuple[int, ...], list[list[float]]],
    allocation_alpha: float | None,
    **selection_constraints: object,
) -> portfolio_allocation.WalkForwardPortfolioAllocationResult:
    """Mock both WFA passes and verify fixed-support counterfactual calls."""
    directory = _portfolio_directory(tmp_path, portfolio_name)
    start_date = datetime.date(2025, 1, 6)
    end_date = start_date + datetime.timedelta(
        weeks=len(winning_allocations) + 1, days=-3
    )
    dates = _market_dates(start_date, end_date)
    strategy_count = len(winning_allocations[0])
    strategy_names = ("alpha", "beta", "gamma")[:strategy_count]
    for strategy_index, strategy_name in enumerate(strategy_names, start=1):
        strategy_returns = [
            strategy_index * (0.002 if index % 2 == 0 else -0.001)
            for index in range(len(dates))
        ]
        _write_fractional_returns(
            directory / f"{strategy_name}.csv", dates, strategy_returns
        )

    winning_supports = [
        tuple(index for index, value in enumerate(target) if value > 0.0)
        for target in winning_allocations
    ]
    discovered_supports = set(winning_supports)
    assert discovered_supports == set(regime_targets)
    assert all(
        len(targets) == len(winning_allocations) for targets in regime_targets.values()
    )
    optimizer_calls: list[dict[str, object]] = []
    training_returns: list[torch.Tensor] = []
    fold_seeds: list[object] = []

    def sequential_optimizer(
        returns: torch.Tensor,
        alpha: float,
        grid_increment: float | None,
        **kwargs: object,
    ) -> torch.Tensor:
        del alpha, grid_increment
        is_selection_call = any(
            kwargs.get(name) is not None
            for name in ("max_strategies", "min_correlation", "max_correlation")
        )
        target_device = kwargs.get("device") or returns.device
        target_dtype = kwargs.get("dtype") or returns.dtype
        seed = kwargs.get("bootstrap_seed")
        if is_selection_call:
            fold_index = len(training_returns)
            training_returns.append(returns.detach().clone())
            fold_seeds.append(seed)
            allocation = winning_allocations[fold_index]
            support = winning_supports[fold_index]
        else:
            assert seed in fold_seeds
            fold_index = fold_seeds.index(seed)
            support = next(
                candidate_support
                for candidate_support in discovered_supports
                if candidate_support
                and returns.shape[1] == len(candidate_support)
                and torch.equal(
                    returns,
                    training_returns[fold_index][:, list(candidate_support)],
                )
            )
            assert support != winning_supports[fold_index]
            assert all(
                name not in kwargs
                for name in ("max_strategies", "min_correlation", "max_correlation")
            )
            full_target = regime_targets[support][fold_index]
            allocation = [full_target[index] for index in support]
        optimizer_calls.append(
            {
                "fold_index": fold_index,
                "support": support,
                "selection": is_selection_call,
                "bootstrap_seed": seed,
            }
        )
        return torch.tensor(allocation, device=target_device, dtype=target_dtype)

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        sequential_optimizer,
    )
    monkeypatch.setattr(
        portfolio_allocation, "tqdm", lambda iterable, **_kwargs: iterable
    )
    kwargs: dict[str, object] = {
        "portfolio_name": portfolio_name,
        "alpha": 0.0,
        "grid_increment": 0.1,
        "training_weeks": 1,
        "embargo_weeks": 0,
        "simulation_weeks": 1,
        "device": "cpu",
        "bootstrap_on": True,
        "bootstrap_runs": 2,
        "bootstrap_length": 3,
        "show_plot": False,
        **selection_constraints,
    }
    if allocation_alpha is not None:
        kwargs["allocation_alpha"] = allocation_alpha

    result = allocation.walk_forward_capital_allocation_from_csv(**kwargs)

    fold_count = len(winning_allocations)
    assert len(training_returns) == fold_count
    assert len(fold_seeds) == fold_count
    assert all(seed is not None for seed in fold_seeds)
    assert len(set(fold_seeds)) == fold_count
    assert all(call["selection"] for call in optimizer_calls[:fold_count])
    coefficient = 1.0 if allocation_alpha is None else allocation_alpha
    if coefficient == 1.0:
        expected_counterfactuals: set[tuple[int, tuple[int, ...]]] = set()
    elif coefficient == 0.0:
        expected_counterfactuals = {
            (0, support)
            for support in discovered_supports
            if support and support != winning_supports[0]
        }
    else:
        expected_counterfactuals = {
            (fold_index, support)
            for fold_index, winning_support in enumerate(winning_supports)
            for support in discovered_supports
            if support and support != winning_support
        }
    actual_counterfactuals = {
        (call["fold_index"], call["support"]) for call in optimizer_calls[fold_count:]
    }
    assert actual_counterfactuals == expected_counterfactuals
    assert len(optimizer_calls) == fold_count + len(expected_counterfactuals)
    for call in optimizer_calls[fold_count:]:
        fold_index = call["fold_index"]
        assert call["bootstrap_seed"] == fold_seeds[fold_index]
    return result


def test_csv_subset_limits_are_independent_and_defaults_preserve_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Each optional limit works alone or together, while None changes nothing."""
    directory = _portfolio_directory(tmp_path, "independent_limits")
    _write_correlation_case(directory)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    common = {
        "portfolio_name": "independent_limits",
        "alpha": 1e-30,
        "grid_increment": 0.5,
        "device": "cpu",
        "show_plot": False,
    }

    unrestricted = allocation.find_optimal_capital_allocation_from_csv(**common)
    explicit_none = allocation.find_optimal_capital_allocation_from_csv(
        **common,
        max_strategies=None,
        min_correlation=None,
        max_correlation=None,
    )
    minimum_only = allocation.find_optimal_capital_allocation_from_csv(
        **common, min_correlation=0.4
    )
    maximum_only = allocation.find_optimal_capital_allocation_from_csv(
        **common, max_correlation=0.4
    )
    bounded = allocation.find_optimal_capital_allocation_from_csv(
        **common, min_correlation=0.4, max_correlation=0.41
    )
    count_only = allocation.find_optimal_capital_allocation_from_csv(
        **common, max_strategies=1
    )
    all_limits = allocation.find_optimal_capital_allocation_from_csv(
        **common,
        max_strategies=1,
        min_correlation=0.4,
        max_correlation=0.41,
    )

    pair = torch.tensor([0.5, 0.5], dtype=torch.float32)
    alpha_only = torch.tensor([1.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(unrestricted.allocation, pair)
    torch.testing.assert_close(explicit_none.allocation, unrestricted.allocation)
    torch.testing.assert_close(minimum_only.allocation, pair)
    torch.testing.assert_close(maximum_only.allocation, alpha_only)
    torch.testing.assert_close(bounded.allocation, pair)
    torch.testing.assert_close(count_only.allocation, alpha_only)
    torch.testing.assert_close(all_limits.allocation, alpha_only)
    assert all(
        result.strategy_names == ("alpha", "beta")
        for result in (
            unrestricted,
            explicit_none,
            minimum_only,
            maximum_only,
            bounded,
            count_only,
            all_limits,
        )
    )


def test_max_strategies_selects_best_subset_and_zero_fills_excluded_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The score comparison retains the best allowed strategies in full output."""
    directory = _portfolio_directory(tmp_path, "best_count_subset")
    dates = _market_dates(datetime.date(2025, 1, 13), datetime.date(2025, 1, 15))
    for name, strategy_date, strategy_return in (
        ("alpha", dates[0], 0.10),
        ("beta", dates[1], 0.20),
        ("gamma", dates[2], 0.30),
    ):
        _write_fractional_returns(
            directory / f"{name}.csv", [strategy_date], [strategy_return]
        )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    unrestricted = allocation.find_optimal_capital_allocation_from_csv(
        "best_count_subset", 0.0, 1.0, device="cpu", show_plot=False
    )
    limited = allocation.find_optimal_capital_allocation_from_csv(
        "best_count_subset",
        0.0,
        1.0,
        device="cpu",
        show_plot=False,
        max_strategies=2,
    )

    assert unrestricted.strategy_names == ("alpha", "beta", "gamma")
    torch.testing.assert_close(unrestricted.allocation, torch.ones(3))
    torch.testing.assert_close(limited.allocation, torch.tensor([0.0, 1.0, 1.0]))
    assert limited.daily_results["combined_return"].to_list() == pytest.approx(
        [0.0, 0.20, 0.30]
    )


def test_correlation_limits_replace_inactive_nans_with_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Disjoint positive return blocks have correlation -1 after zero filling."""
    directory = _portfolio_directory(tmp_path, "inactive_zero_correlation")
    dates = _market_dates(datetime.date(2025, 1, 13), datetime.date(2025, 1, 16))
    _write_fractional_returns(directory / "left.csv", dates[:2], [0.10, 0.10])
    _write_fractional_returns(directory / "right.csv", dates[2:], [0.20, 0.20])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    result = allocation.find_optimal_capital_allocation_from_csv(
        "inactive_zero_correlation",
        alpha=0.0,
        grid_increment=1.0,
        device="cpu",
        show_plot=False,
        max_strategies=2,
        min_correlation=-1.0,
        max_correlation=-0.9,
    )

    torch.testing.assert_close(result.allocation, torch.tensor([1.0, 1.0]))
    assert result.diversification.correlation_matrix[0, 1].item() == pytest.approx(-1.0)


def test_undefined_correlation_pair_is_ineligible_when_a_bound_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A constant strategy cannot satisfy even an otherwise permissive bound."""
    directory = _portfolio_directory(tmp_path, "undefined_correlation")
    dates = _market_dates(datetime.date(2025, 1, 6), datetime.date(2025, 1, 16))
    _write_fractional_returns(
        directory / "alpha.csv",
        dates,
        [0.17, 0.03, -0.31, 0.22, 0.54, 0.31, -0.42, 0.16],
    )
    _write_fractional_returns(directory / "constant.csv", dates, [0.04] * len(dates))
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    unrestricted = allocation.find_optimal_capital_allocation_from_csv(
        "undefined_correlation",
        1e-30,
        0.5,
        device="cpu",
        show_plot=False,
    )
    constrained = allocation.find_optimal_capital_allocation_from_csv(
        "undefined_correlation",
        1e-30,
        0.5,
        device="cpu",
        show_plot=False,
        max_correlation=1.0,
    )

    torch.testing.assert_close(unrestricted.allocation, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(constrained.allocation, torch.tensor([1.0, 0.0]))
    assert math.isnan(constrained.diversification.correlation_matrix[0, 1].item())


def test_subset_ties_are_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Equal-scoring valid supports produce the same full allocation every run."""
    directory = _portfolio_directory(tmp_path, "deterministic_tie")
    dates = _market_dates(datetime.date(2025, 1, 13), datetime.date(2025, 1, 15))
    values = [0.10, -0.02, 0.05]
    _write_fractional_returns(directory / "alpha.csv", dates, values)
    _write_fractional_returns(directory / "beta.csv", dates, values)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    kwargs = {
        "portfolio_name": "deterministic_tie",
        "alpha": 0.0,
        "grid_increment": 1.0,
        "device": "cpu",
        "show_plot": False,
        "max_strategies": 1,
    }

    first = allocation.find_optimal_capital_allocation_from_csv(**kwargs)
    second = allocation.find_optimal_capital_allocation_from_csv(**kwargs)

    torch.testing.assert_close(first.allocation, second.allocation)
    assert torch.count_nonzero(first.allocation).item() == 1


def test_direct_subset_bootstrap_draws_indices_once_for_all_valid_supports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """All support comparisons in one run use the same bootstrap paths."""
    directory = _portfolio_directory(tmp_path, "shared_subset_bootstrap")
    _write_correlation_case(directory)
    randint_calls: list[tuple[int, tuple[int, ...]]] = []
    original_randint = torch.randint

    def recording_randint(high, size, **kwargs):
        randint_calls.append((high, size))
        return original_randint(high, size, **kwargs)

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(torch, "randint", recording_randint)

    allocation.find_optimal_capital_allocation_from_csv(
        "shared_subset_bootstrap",
        alpha=0.1,
        grid_increment=1.0,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=2,
        bootstrap_length=3,
        show_plot=False,
        max_strategies=1,
    )

    assert randint_calls == [(8, (2, 3))]


def test_subset_prefix_search_coalesces_valid_candidates_into_scoring_batches(
    monkeypatch: pytest.MonkeyPatch,
):
    """Support pruning scores every valid point once in full bounded batches."""
    returns = torch.full((4, 4), math.nan, dtype=torch.float64)
    returns.fill_diagonal_(0.10)
    scored_allocations: list[torch.Tensor] = []
    original_score = capital_allocation._allocation_scores

    def recording_score(allocations, *args, **kwargs):
        scored_allocations.append(allocations.detach().to(device="cpu"))
        return original_score(allocations, *args, **kwargs)

    monkeypatch.setattr(capital_allocation, "_candidate_batch_size", lambda *args: 3)
    monkeypatch.setattr(capital_allocation, "_allocation_scores", recording_score)

    capital_allocation.find_optimal_capital_allocation(
        returns,
        alpha=1e-30,
        grid_increment=0.5,
        device="cpu",
        max_strategies=2,
    )

    candidates = torch.cat(scored_allocations)
    # One zero point, 4 * 2 one-strategy points, and C(4, 2) * 2**2 pairs.
    assert candidates.shape == (33, 4)
    assert all(batch.shape[0] == 3 for batch in scored_allocations)
    assert torch.all(torch.count_nonzero(candidates, dim=1) <= 2)
    assert torch.unique(candidates, dim=0).shape[0] == candidates.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_subset_refinement_matches_cpu():
    """GPU prefix pruning matches CPU with all selection limits enabled."""
    returns = torch.tensor(
        [
            [0.20, 0.10, math.nan],
            [-0.10, 0.30, math.nan],
            [math.nan, math.nan, 0.40],
            [0.25, -0.15, math.nan],
        ],
        dtype=torch.float32,
    )
    kwargs = {
        "alpha": 0.2,
        "grid_increment": 0.25,
        "max_strategies": 2,
        "min_correlation": -1.0,
        "max_correlation": 0.8,
        "refinement_runs": 1,
    }

    expected = capital_allocation.find_optimal_capital_allocation(
        returns, device="cpu", **kwargs
    )
    actual = capital_allocation.find_optimal_capital_allocation(
        returns, device="cuda:0", **kwargs
    )

    torch.testing.assert_close(actual.cpu(), expected)
    assert actual.device == torch.device("cuda:0")


def test_walk_forward_reselects_fold_local_subsets_and_prewarms_each_ema_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A support selected later already contains its counterfactual EMA history."""
    directory = _portfolio_directory(tmp_path, "fold_local_subsets")
    dates = _market_dates(datetime.date(2025, 1, 13), datetime.date(2025, 1, 31))
    first_pattern = [0.40, -0.20, 0.40, -0.20, 0.10]
    first_inverse = [-0.20, 0.40, -0.20, 0.40, 0.10]
    first_small = [0.1 * value for value in first_pattern]
    second_pattern = [0.40, -0.20, 0.40, -0.20]
    second_inverse = [-0.20, 0.40, -0.20, 0.40]
    second_small = [0.1 * value for value in second_pattern]
    final_week_length = len(dates) - 9
    returns_by_name = {
        "alpha": first_pattern + second_pattern + [0.01] * final_week_length,
        "beta": first_inverse + second_small + [0.01] * final_week_length,
        "gamma": first_small + second_inverse + [0.01] * final_week_length,
    }
    for name, strategy_returns in returns_by_name.items():
        _write_fractional_returns(directory / f"{name}.csv", dates, strategy_returns)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        portfolio_allocation, "tqdm", lambda iterable, **_kwargs: iterable
    )

    result = allocation.walk_forward_capital_allocation_from_csv(
        "fold_local_subsets",
        alpha=0.0,
        grid_increment=0.5,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        allocation_alpha=0.5,
        device="cpu",
        show_plot=False,
        max_strategies=2,
        max_correlation=-0.9,
        ADD_limit=-1.0,
    )

    assert result.strategy_names == ("alpha", "beta", "gamma")
    assert len(result.folds) == 2
    torch.testing.assert_close(
        result.folds[0].calculated_allocation, torch.tensor([0.5, 0.5, 0.0])
    )
    torch.testing.assert_close(
        result.folds[0].applied_allocation, torch.tensor([0.5, 0.5, 0.0])
    )
    torch.testing.assert_close(
        result.folds[1].calculated_allocation, torch.tensor([0.5, 0.0, 0.5])
    )
    torch.testing.assert_close(
        result.folds[1].applied_allocation, torch.tensor([0.75, 0.0, 0.25])
    )
    assert all(
        torch.count_nonzero(fold.applied_allocation).item() <= 2
        for fold in result.folds
    )


def test_walk_forward_selection_ema_uses_current_counterfactual_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both active and inactive supports consume a new optimum on every fold."""
    winning = [[1.0, 0.0], [0.0, 0.6], [0.4, 0.0]]
    regime_targets = {
        (0,): [[1.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
        (1,): [[0.0, 1.0], [0.0, 0.6], [0.0, 0.2]],
    }

    result = _run_regime_walk_forward(
        tmp_path,
        monkeypatch,
        "reappearing_supports",
        winning,
        regime_targets,
        allocation_alpha=0.5,
        max_strategies=1,
    )

    expected_applied = ([1.0, 0.0], [0.0, 0.8], [0.5, 0.0])
    assert len(result.folds) == len(winning)
    for fold, expected_calculated, expected in zip(
        result.folds, winning, expected_applied, strict=True
    ):
        torch.testing.assert_close(
            fold.calculated_allocation, torch.tensor(expected_calculated)
        )
        torch.testing.assert_close(fold.applied_allocation, torch.tensor(expected))


def test_walk_forward_selection_ema_treats_empty_support_as_a_separate_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An all-zero optimum neither erases nor contaminates another support's EMA."""
    winning = [[1.0, 0.0], [0.0, 0.0], [0.4, 0.0]]
    regime_targets = {
        (): [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        (0,): [[1.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
    }

    result = _run_regime_walk_forward(
        tmp_path,
        monkeypatch,
        "empty_support_regime",
        winning,
        regime_targets,
        allocation_alpha=0.5,
        max_strategies=1,
    )

    expected_applied = (
        [1.0, 0.0],
        [0.0, 0.0],
        [0.5, 0.0],
    )
    for fold, expected in zip(result.folds, expected_applied, strict=True):
        torch.testing.assert_close(fold.applied_allocation, torch.tensor(expected))


@pytest.mark.parametrize(
    "selection_constraints",
    [
        {"max_strategies": 1},
        {"min_correlation": -1.0},
        {"max_correlation": 1.0},
        {"min_correlation": -0.5, "max_correlation": 0.5},
    ],
)
def test_walk_forward_each_selection_constraint_enables_support_specific_ema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_constraints: dict[str, object],
):
    """Every independently optional selection limit activates regime tracking."""
    winning = [[1.0, 0.0], [0.0, 0.6], [0.4, 0.0]]
    regime_targets = {
        (0,): [[1.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
        (1,): [[0.0, 1.0], [0.0, 0.6], [0.0, 0.2]],
    }

    result = _run_regime_walk_forward(
        tmp_path,
        monkeypatch,
        "independent_regime_limit",
        winning,
        regime_targets,
        allocation_alpha=0.5,
        **selection_constraints,
    )

    expected_applied = ([1.0, 0.0], [0.0, 0.8], [0.5, 0.0])
    for fold, expected in zip(result.folds, expected_applied, strict=True):
        torch.testing.assert_close(fold.applied_allocation, torch.tensor(expected))


@pytest.mark.parametrize(
    ("allocation_alpha", "expected_applied"),
    [
        (0.0, ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0])),
        (1.0, ([1.0, 0.0], [0.0, 0.6], [0.4, 0.0])),
        (None, ([1.0, 0.0], [0.0, 0.6], [0.4, 0.0])),
    ],
)
def test_walk_forward_selection_regime_ema_alpha_endpoints_and_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation_alpha: float | None,
    expected_applied: tuple[list[float], ...],
):
    """Endpoints freeze after fold one or skip all counterfactual optimization."""
    winning = [[1.0, 0.0], [0.0, 0.6], [0.4, 0.0]]
    regime_targets = {
        (0,): [[1.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
        (1,): [[0.0, 1.0], [0.0, 0.6], [0.0, 0.2]],
    }

    result = _run_regime_walk_forward(
        tmp_path,
        monkeypatch,
        "regime_alpha_endpoint",
        winning,
        regime_targets,
        allocation_alpha=allocation_alpha,
        max_strategies=1,
    )

    for fold, expected in zip(result.folds, expected_applied, strict=True):
        torch.testing.assert_close(fold.applied_allocation, torch.tensor(expected))


@pytest.mark.parametrize(
    ("constraint_kwargs", "error_type", "message"),
    [
        ({"max_strategies": True}, TypeError, "max_strategies"),
        ({"max_strategies": 1.5}, TypeError, "max_strategies"),
        ({"max_strategies": 0}, ValueError, "max_strategies"),
        ({"min_correlation": True}, TypeError, "min_correlation"),
        ({"min_correlation": "low"}, TypeError, "min_correlation"),
        ({"min_correlation": math.nan}, ValueError, "min_correlation"),
        ({"min_correlation": -1.01}, ValueError, "min_correlation"),
        ({"max_correlation": math.inf}, ValueError, "max_correlation"),
        ({"max_correlation": 1.01}, ValueError, "max_correlation"),
        (
            {"min_correlation": 0.5, "max_correlation": 0.4},
            ValueError,
            "min_correlation|max_correlation",
        ),
    ],
)
def test_csv_wrapper_rejects_invalid_subset_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constraint_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
):
    """The direct CSV entry point validates every optional subset limit."""
    directory = _portfolio_directory(tmp_path, "invalid_subset_limit")
    _write_fractional_returns(
        directory / "strategy.csv", [datetime.date(2025, 1, 13)], [0.01]
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(error_type, match=message):
        allocation.find_optimal_capital_allocation_from_csv(
            "invalid_subset_limit",
            alpha=0.0,
            grid_increment=1.0,
            device="cpu",
            show_plot=False,
            **constraint_kwargs,
        )


@pytest.mark.parametrize(
    "constraint_kwargs",
    [
        {"max_strategies": 0},
        {"min_correlation": -1.01},
        {"max_correlation": 1.01},
        {"min_correlation": 0.5, "max_correlation": 0.4},
    ],
)
def test_walk_forward_wrapper_rejects_invalid_subset_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constraint_kwargs: dict[str, object],
):
    """Walk-forward exposes the same validated constraint contract."""
    directory = _portfolio_directory(tmp_path, "invalid_wfa_subset_limit")
    dates = _market_dates(datetime.date(2025, 1, 6), datetime.date(2025, 1, 17))
    _write_fractional_returns(directory / "strategy.csv", dates, [0.01] * len(dates))
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises((TypeError, ValueError), match="strategies|correlation"):
        allocation.walk_forward_capital_allocation_from_csv(
            "invalid_wfa_subset_limit",
            alpha=0.0,
            grid_increment=1.0,
            training_weeks=1,
            embargo_weeks=0,
            simulation_weeks=1,
            device="cpu",
            show_plot=False,
            **constraint_kwargs,
        )


def test_high_level_entry_points_forward_subset_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The canonical CSV and walk-forward facades pass all three parameters on."""
    directory = _portfolio_directory(tmp_path, "forward_subset_limits")
    dates = _market_dates(datetime.date(2025, 1, 6), datetime.date(2025, 1, 17))
    _write_fractional_returns(directory / "alpha.csv", dates, [0.01] * len(dates))
    _write_fractional_returns(directory / "beta.csv", dates, [0.02] * len(dates))
    optimizer_calls: list[dict[str, object]] = []

    def fake_optimizer(returns: torch.Tensor, **kwargs: object) -> torch.Tensor:
        optimizer_calls.append(kwargs)
        return torch.tensor([1.0, 0.0], dtype=returns.dtype)

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", fake_optimizer
    )

    direct_result = allocation.find_optimal_capital_allocation_from_csv(
        "forward_subset_limits",
        alpha=0.0,
        grid_increment=1.0,
        device="cpu",
        show_plot=False,
        max_strategies=1,
        min_correlation=-0.5,
        max_correlation=0.5,
    )

    assert direct_result.strategy_names == ("alpha", "beta")
    assert optimizer_calls[0]["max_strategies"] == 1
    assert optimizer_calls[0]["min_correlation"] == -0.5
    assert optimizer_calls[0]["max_correlation"] == 0.5

    sentinel = object()
    internal_calls: list[dict[str, object]] = []

    def fake_walk_forward(**kwargs: object):
        internal_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        walk_forward_allocation,
        "walk_forward_capital_allocation_from_csv",
        fake_walk_forward,
    )
    walk_forward_result = allocation.walk_forward_capital_allocation_from_csv(
        "forward_subset_limits",
        alpha=0.0,
        grid_increment=1.0,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        show_plot=False,
        max_strategies=1,
        min_correlation=-0.5,
        max_correlation=0.5,
    )

    assert walk_forward_result is sentinel
    assert internal_calls[0]["max_strategies"] == 1
    assert internal_calls[0]["min_correlation"] == -0.5
    assert internal_calls[0]["max_correlation"] == 0.5
