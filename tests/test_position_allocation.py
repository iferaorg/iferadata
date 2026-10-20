"""Check that position-ledger searches score actual candidate account paths."""

from datetime import date, datetime, time, timedelta

import pytest
import torch

from ifera.allocation import capital_allocation
from ifera.allocation._position_model import PositionModel, PositionTrade


class _PositionModel:
    """Supply deterministic account paths independently of the proxy returns."""

    def __init__(self, path_function, days=3, strategies=1):
        self.dates = tuple(date(2024, 1, 2) + timedelta(days=i) for i in range(days))
        self.strategy_names = tuple(f"strategy_{i}" for i in range(strategies))
        self.trades = ()
        self.path_function = path_function
        self.calls = []

    def evaluate(self, allocations, max_total_allocation):
        self.calls.append((allocations.clone(), max_total_allocation))
        return self.path_function(allocations)


def _find(model, **kwargs):
    proxy = torch.full((len(model.dates), len(model.strategy_names)), -0.5)
    return capital_allocation.find_optimal_capital_allocation(
        proxy,
        alpha=kwargs.pop("alpha", 0.0),
        device="cpu",
        position_model=model,
        **kwargs,
    )


def test_position_search_uses_ledger_and_per_trade_bounds():
    model = _PositionModel(
        lambda f: (f.sum(dim=1) * 0.1).unsqueeze(1).expand(-1, 3), strategies=2
    )
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    allocation = _find(model, grid_increment=0.5, diagnostics=diagnostics)
    torch.testing.assert_close(allocation, torch.ones(2))
    assert diagnostics.algorithm.startswith("Position-ledger")
    assert all(cap == 1.0 for _, cap in model.calls)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "grid_increment"),
        ({"approximate_add_search": True}, "approximate_add_search"),
        ({"allocation_tolerance": 0.01}, "allocation_tolerance"),
    ],
)
def test_position_search_rejects_unsupported_solver_options(kwargs, message):
    model = _PositionModel(lambda f: f.expand(-1, 3))
    with pytest.raises(ValueError, match=message):
        _find(model, **kwargs)


def test_position_search_preserves_selection_constraints():
    model = _PositionModel(
        lambda f: (f[:, 0] * 0.1 + f[:, 1] * 0.2).unsqueeze(1).expand(-1, 3),
        strategies=2,
    )
    allocation = _find(model, grid_increment=0.5, max_strategies=1)
    torch.testing.assert_close(allocation, torch.tensor([0.0, 1.0]))


def test_position_search_rejects_day_blocks_and_mismatched_dimensions():
    model = _PositionModel(
        lambda f: f.sum(dim=1, keepdim=True).expand(-1, 3), strategies=2
    )
    with pytest.raises(ValueError, match="trade-level blocks"):
        _find(model, grid_increment=0.5, block_list=[(0, 0, 1)])
    with pytest.raises(ValueError, match="dimensions"):
        capital_allocation.find_optimal_capital_allocation(
            torch.zeros((4, 2)), alpha=0.0, grid_increment=0.5, position_model=model
        )


def test_position_selection_uses_proxy_correlations():
    model = _PositionModel(
        lambda f: f.sum(dim=1, keepdim=True).expand(-1, 3), strategies=2
    )
    correlated = torch.tensor([[0.1, 0.1], [-0.1, -0.1], [0.2, 0.2]])
    allocation = capital_allocation.find_optimal_capital_allocation(
        correlated,
        alpha=0.0,
        grid_increment=0.5,
        device="cpu",
        position_model=model,
        max_correlation=0.5,
    )
    assert torch.count_nonzero(allocation).item() == 1


def test_position_refinement_scores_the_same_account_model():
    model = _PositionModel(lambda f: (0.2 - (f - 0.375).square()).expand(-1, 3))
    allocation = _find(model, grid_increment=0.5, refinement_runs=2)
    torch.testing.assert_close(allocation, torch.tensor([0.375]))


def test_position_search_rejects_bankruptcy():
    model = _PositionModel(lambda f: torch.cat((f * 2, f * -2, f), dim=1))
    allocation = _find(model, grid_increment=0.25)
    assert allocation.item() < 0.5


def test_position_drawdown_objective_and_growth_floor():
    model = _PositionModel(lambda f: f * torch.tensor([0.2, -0.1, 0.1]))
    allocation = _find(
        model,
        grid_increment=0.25,
        objective="minimize_add",
        CMGR_limit=0.04,
        calendar_days=31,
    )
    torch.testing.assert_close(allocation, torch.tensor([0.25]))
    with pytest.raises(ValueError, match="CMGR_limit"):
        _find(model, grid_increment=0.5, CMGR_limit=10.0, calendar_days=31)


def test_position_minimum_drawdown_without_growth_returns_cash():
    model = _PositionModel(lambda f: f.expand(-1, 3))
    allocation = _find(model, objective="minimize_add")
    torch.testing.assert_close(allocation, torch.zeros(1))
    assert not model.calls


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_position_bootstrap_diagnostic_uses_actual_path(monkeypatch, alpha):
    model = _PositionModel(lambda f: f * torch.tensor([0.4, -0.1, 0.3]))
    indices = torch.tensor([[0, 1, 2], [1, 1, 1], [2, 0, 1]])
    samples = []

    def fixed_samples(*_args):
        samples.append(indices)
        return indices

    monkeypatch.setattr(capital_allocation, "_sample_bootstrap_indices", fixed_samples)
    diagnostics = capital_allocation.CapitalAllocationDiagnostics()
    allocation = _find(
        model,
        alpha=alpha,
        grid_increment=0.25,
        bootstrap_on=True,
        bootstrap_runs=3,
        bootstrap_length=3,
        diagnostics=diagnostics,
    )
    daily = model.path_function(allocation.unsqueeze(0))[0]
    equity = torch.cumprod(1.0 + daily[indices], dim=1)
    peak = torch.cummax(torch.cat((torch.ones(3, 1), equity), dim=1), dim=1).values[
        :, 1:
    ]
    drawdowns = -torch.sqrt(((equity / peak - 1.0) ** 2).mean(dim=1))
    expected = torch.quantile(drawdowns, 0.1).item()
    assert diagnostics.bootstrap_average_drawdown == pytest.approx(expected, abs=1e-6)
    assert len(samples) == 1


def test_position_historical_add_limit_is_applied_to_actual_paths():
    model = _PositionModel(lambda f: f * torch.tensor([0.4, -0.3, 0.4]))
    allocation = _find(model, grid_increment=0.25, ADD_limit=-0.06)
    torch.testing.assert_close(allocation, torch.tensor([0.25]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_position_cuda_compiled_bootstrap_refinement_matches_cpu(monkeypatch):
    """Real ledgers and compiled bootstrap scoring agree across CPU and CUDA."""
    dates = (
        date(2026, 10, 5),
        date(2026, 10, 6),
        date(2026, 10, 7),
        date(2026, 10, 8),
        date(2026, 10, 9),
        date(2026, 10, 12),
    )

    def trade(strategy, opened, closed, ror):
        return PositionTrade(
            strategy,
            datetime.combine(dates[opened], time(10)),
            datetime.combine(dates[closed], time(16)),
            ror,
        )

    model = PositionModel(
        dates,
        ("A", "B"),
        (
            trade(0, 0, 1, 0.2),
            trade(1, 0, 1, -0.1),
            trade(0, 2, 3, -0.3),
            trade(1, 2, 4, 0.08),
            trade(1, 3, 5, 0.12),
            trade(0, 4, 5, 0.35),
        ),
    )
    seed = 1729
    indices = torch.randint(6, (17, 8), generator=torch.Generator().manual_seed(seed))
    sampled_devices = []

    def shared_indices(time_count, runs, length, device, bootstrap_seed):
        assert (time_count, runs, length, bootstrap_seed) == (6, 17, 8, seed)
        sampled_devices.append(torch.device(device).type)
        # CPU and CUDA generators intentionally use different algorithms.
        return indices.to(device)

    monkeypatch.setattr(capital_allocation, "_sample_bootstrap_indices", shared_indices)
    results = []
    for device in ("cpu", "cuda:0"):
        diagnostics = capital_allocation.CapitalAllocationDiagnostics()
        allocation = capital_allocation.find_optimal_capital_allocation(
            model.returns_tensor(),
            alpha=0.02,
            grid_increment=0.25,
            device=device,
            bootstrap_on=True,
            bootstrap_runs=17,
            bootstrap_length=8,
            bootstrap_seed=seed,
            percentile=10.0,
            max_total_allocation=0.75,
            refinement_runs=1,
            refinement_divisor=2.0,
            ADD_limit=-0.05,
            diagnostics=diagnostics,
            position_model=model,
        )
        results.append((allocation.cpu(), diagnostics))

    assert sampled_devices == ["cpu", "cuda"]
    assert torch.count_nonzero(results[0][0]).item() > 0
    assert bool(torch.any(torch.remainder(results[0][0], 0.25) != 0.0))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][1].bootstrap_average_drawdown is not None
    assert results[1][1].bootstrap_average_drawdown == pytest.approx(
        results[0][1].bootstrap_average_drawdown, abs=2e-6
    )
    assert results[0][1].bootstrap_average_drawdown >= -0.05
    assert results[0][1].algorithm == results[1][1].algorithm
    assert results[0][1].algorithm.startswith("Position-ledger")
