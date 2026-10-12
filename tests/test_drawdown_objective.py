"""Independent checks of drawdown minimization with a calendar CMGR floor."""

import itertools
import math

import pytest
import torch

import ifera.allocation.capital_allocation as allocation


def _drawdown(daily_returns: torch.Tensor) -> torch.Tensor:
    """Compute negative RMS drawdown using compounded wealth including cash."""
    wealth = (1.0 + daily_returns).cumprod(dim=-1)
    initial = torch.ones((*wealth.shape[:-1], 1), dtype=wealth.dtype)
    peaks = torch.cat((initial, wealth), dim=-1).cummax(dim=-1).values[..., 1:]
    return -(wealth / peaks - 1.0).square().mean(dim=-1).sqrt()


@pytest.mark.parametrize("bootstrap_on", [False, True])
@pytest.mark.parametrize("max_strategies", [None, 1])
def test_drawdown_objective_matches_exhaustive_grid(bootstrap_on, max_strategies):
    """Risk ranks feasible portfolios; actual growth determines the monthly floor."""
    returns = torch.tensor(
        [[0.15, 0.04], [-0.12, -0.015], [0.20, 0.06], [-0.08, -0.02]],
        dtype=torch.float64,
    )
    calendar_days, cmgr_limit = 30, 0.025
    indices = torch.randint(4, (17, 7), generator=torch.Generator().manual_seed(42))
    best_add = -math.inf
    expected = None
    for units in itertools.product(range(11), repeat=2):
        if sum(units) > 10 or (max_strategies == 1 and all(units)):
            continue
        weights = torch.tensor(units, dtype=torch.float64) / 10
        daily = returns @ weights
        cmgr = math.expm1(float(daily.log1p().sum()) / (calendar_days / 365.25 * 12))
        if cmgr < cmgr_limit:
            continue
        add = (
            float(torch.quantile(_drawdown(daily[indices]), 0.1))
            if bootstrap_on
            else float(_drawdown(daily))
        )
        if add > best_add:
            best_add, expected = add, weights
    assert expected is not None
    diagnostics = allocation.CapitalAllocationDiagnostics()
    result = allocation.find_optimal_capital_allocation(
        returns,
        alpha=9.0,
        objective="minimize_add",
        CMGR_limit=cmgr_limit,
        calendar_days=calendar_days,
        grid_increment=0.1,
        device="cpu",
        max_strategies=max_strategies,
        bootstrap_on=bootstrap_on,
        bootstrap_runs=17,
        bootstrap_length=7,
        bootstrap_seed=42,
        diagnostics=diagnostics,
    )
    torch.testing.assert_close(result, expected)
    assert "Drawdown-minimizing" in diagnostics.algorithm
    if bootstrap_on:
        assert diagnostics.bootstrap_average_drawdown == pytest.approx(best_add)


@pytest.mark.parametrize("cmgr_limit", [None, 0.0, -0.1, -2.0])
def test_drawdown_without_positive_growth_floor_is_cash(cmgr_limit, monkeypatch):
    """Zero is returned without grid enumeration or sampling even with good returns."""

    def unexpected(*_args, **_kwargs):
        pytest.fail("The trivial cash solution must not run a search or bootstrap")

    monkeypatch.setattr(allocation, "_allocation_scores", unexpected)
    monkeypatch.setattr(allocation, "_maximize_kelly_with_selection", unexpected)
    monkeypatch.setattr(allocation, "_sample_bootstrap_indices", unexpected)
    diagnostics = allocation.CapitalAllocationDiagnostics()
    result = allocation.find_optimal_capital_allocation(
        torch.tensor([[0.1, 0.2], [0.2, 0.1]]),
        alpha=0.0,
        objective="minimize_add",
        CMGR_limit=cmgr_limit,
        device="cpu",
        dtype=torch.float64,
        bootstrap_on=True,
        diagnostics=diagnostics,
    )
    torch.testing.assert_close(result, torch.zeros(2, dtype=torch.float64))
    assert diagnostics.bootstrap_average_drawdown == 0.0
    assert "cash" in diagnostics.algorithm.lower()


@pytest.mark.parametrize("cap", [0.0, 1.0])
def test_unachievable_cmgr_reports_no_feasible_grid_allocation(cap):
    """Cash must never be returned as though it satisfies a positive CMGR limit."""
    with pytest.raises(ValueError, match="[Nn]o feasible.*CMGR_limit"):
        allocation.find_optimal_capital_allocation(
            torch.tensor([[-0.1], [-0.2]], dtype=torch.float64),
            alpha=0.0,
            objective="minimize_add",
            CMGR_limit=0.01,
            calendar_days=30,
            grid_increment=0.1,
            max_total_allocation=cap,
            device="cpu",
        )


def test_cmgr_bound_uses_calendar_duration_and_refines_risk():
    """A longer calendar span requires more growth, and refinements reduce risk."""
    returns = torch.tensor([[0.30], [-0.10]], dtype=torch.float64)
    kwargs = dict(
        alpha=0.0,
        objective="minimize_add",
        CMGR_limit=0.025,
        grid_increment=0.25,
        device="cpu",
    )
    coarse = allocation.find_optimal_capital_allocation(
        returns, calendar_days=30, **kwargs
    )
    refined = allocation.find_optimal_capital_allocation(
        returns, calendar_days=30, refinement_runs=2, **kwargs
    )
    longer = allocation.find_optimal_capital_allocation(
        returns, calendar_days=90, **kwargs
    )
    assert 0 < float(refined[0]) < float(coarse[0]) < float(longer[0])
    refined_cmgr = math.expm1(
        float((returns @ refined).log1p().sum()) / (30 / 365.25 * 12)
    )
    assert refined_cmgr >= kwargs["CMGR_limit"]


def test_growth_floor_and_drawdown_use_blocked_returns():
    """Activating a blocker cannot claim the suppressed strategy's growth."""
    diagnostics = allocation.CapitalAllocationDiagnostics()
    weights = allocation.find_optimal_capital_allocation(
        torch.tensor([[0.2, 0.1], [-0.01, -0.001]], dtype=torch.float64),
        alpha=0.0,
        objective="minimize_add",
        CMGR_limit=0.05,
        calendar_days=30,
        block_list=[(0, 1, 0)],
        grid_increment=0.25,
        device="cpu",
        diagnostics=diagnostics,
    )
    torch.testing.assert_close(weights, torch.tensor([0.0, 0.5], dtype=torch.float64))
    assert diagnostics.algorithm.startswith("Blocking-aware")


def test_drawdown_mode_reuses_capital_for_disjoint_strategies():
    """A CMGR target can require a sum above one when strategies are disjoint."""
    nan = math.nan
    returns = torch.tensor(
        [[0.15, nan], [-0.02, nan], [nan, 0.15], [nan, -0.02]],
        dtype=torch.float64,
    )
    weights = allocation.find_optimal_capital_allocation(
        returns,
        alpha=0.0,
        objective="minimize_add",
        CMGR_limit=0.20,
        calendar_days=30,
        grid_increment=0.25,
        device="cpu",
    )
    assert float(weights.sum()) > 1.0
    assert bool((weights <= 1.0).all())
    daily = torch.nan_to_num(returns) @ weights
    assert math.expm1(float(daily.log1p().sum()) / (30 / 365.25 * 12)) >= 0.20


@pytest.mark.parametrize("objective", ["kelly", "minimize_add"])
def test_monthly_growth_and_add_floors_are_both_enforced(objective):
    """A binding growth floor cannot bypass a zero-drawdown requirement."""
    with pytest.raises(ValueError, match="No feasible.*CMGR_limit"):
        allocation.find_optimal_capital_allocation(
            torch.tensor([[0.2], [-0.1]], dtype=torch.float64),
            alpha=0.0,
            objective=objective,
            CMGR_limit=0.01,
            calendar_days=30,
            ADD_limit=0.0,
            approximate_add_search=True,
            grid_increment=0.25,
            device="cpu",
        )


def test_cmgr_limit_can_constrain_default_kelly_objective():
    """A positive monthly floor keeps the existing Kelly score on feasible points."""
    weights = allocation.find_optimal_capital_allocation(
        torch.tensor([[0.2], [-0.1]], dtype=torch.float64),
        alpha=0.0,
        CMGR_limit=0.025,
        calendar_days=30,
        grid_increment=0.25,
        device="cpu",
    )
    torch.testing.assert_close(weights, torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"objective": "unknown"}, ValueError, "objective"),
        ({"objective": True}, TypeError, "objective"),
        ({"CMGR_limit": math.nan}, ValueError, "CMGR_limit"),
        ({"CMGR_limit": math.inf}, ValueError, "CMGR_limit"),
        ({"CMGR_limit": True}, TypeError, "CMGR_limit"),
        ({"CMGR_limit": 0.01}, ValueError, "calendar_days"),
        ({"calendar_days": 0}, ValueError, "calendar_days"),
        ({"calendar_days": 1.5}, TypeError, "calendar_days"),
        ({"calendar_days": True}, TypeError, "calendar_days"),
    ],
)
def test_objective_parameters_are_validated(kwargs, error, message):
    """Objective spelling, monthly fractions, and elapsed calendar time are checked."""
    with pytest.raises(error, match=message):
        allocation.find_optimal_capital_allocation(
            torch.ones((2, 1)), alpha=0.0, device="cpu", **kwargs
        )
