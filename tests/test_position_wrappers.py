"""Trade-ledger integration with portfolio reporting and walk-forward folds."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import torch

from ifera.allocation import capital_allocation, portfolio_allocation
from ifera.settings import settings


def _csv(tmp_path: Path, rows: list[tuple[str, str, float]]) -> None:
    directory = tmp_path / "results" / "portfolios" / "positions"
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=["Opened", "Closed", "ROR"], orient="row").write_csv(
        directory / "strategy.csv"
    )


def test_csv_reports_fixed_entry_risk_and_extends_through_close(tmp_path, monkeypatch):
    """One trade compounds to its entry risk times total ROR without daily resizing."""
    _csv(tmp_path, [("2025-01-06 10:00:00", "2025-01-08 15:00:00", 30.0)])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    seen = []

    def optimizer(returns, **kwargs):
        seen.append(kwargs["position_model"])
        return torch.tensor([0.5], dtype=returns.dtype)

    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )
    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "positions",
        alpha=0,
        grid_increment=0.25,
        device="cpu",
        show_plot=False,
    )

    assert seen[0] is not None
    assert result.statistics.end_date == date(2025, 1, 8)
    assert result.statistics.ending_equity == pytest.approx(1.15)
    assert result.daily_results["equity"].to_list() == pytest.approx([1.05, 1.10, 1.15])
    assert result.unrealized_pnl_method == "linear"
    assert result.trade_results is not None
    assert result.trade_results.height == 1


@pytest.mark.parametrize("method", ["linear", "close_only"])
def test_walk_forward_purges_future_outcomes_and_keeps_entry_allocation(
    tmp_path, monkeypatch, method
):
    """Training excludes future closes and later folds never resize existing trades."""
    _csv(
        tmp_path,
        [
            ("2025-01-06 10:00:00", "2025-01-07 15:00:00", 10.0),
            ("2025-01-10 10:00:00", "2025-01-14 15:00:00", 900.0),
            ("2025-01-13 10:00:00", "2025-01-21 15:00:00", 40.0),
        ],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    training_models = []

    def optimizer(returns, **kwargs):
        training_models.append(kwargs["position_model"])
        return torch.tensor(
            [0.5 if len(training_models) == 1 else 0.0], dtype=returns.dtype
        )

    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )
    result = portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "positions",
        alpha=0,
        grid_increment=0.25,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        device="cpu",
        show_plot=False,
        unrealized_pnl_method=method,
    )

    assert len(training_models) == 2
    assert len(training_models[0].trades) == 1
    assert training_models[0].trades[0].return_on_risk == pytest.approx(0.1)
    assert len(training_models[1].trades) == 0
    assert result.statistics.start_date == date(2025, 1, 13)
    assert result.statistics.end_date == date(2025, 1, 21)
    assert result.statistics.ending_equity == pytest.approx(1.2)
    assert result.trade_results is not None
    assert result.trade_results.height == 1
    if method == "close_only":
        assert result.daily_results["equity"].to_list()[:-1] == pytest.approx([1.0] * 5)


@pytest.mark.parametrize(
    "option", ["max_strategies", "min_correlation", "max_correlation"]
)
def test_carried_positions_reject_walk_forward_support_restrictions(
    tmp_path, monkeypatch, option
):
    """Hard support changes cannot silently liquidate trades with unknown interim values."""
    _csv(tmp_path, [("2025-01-06 10:00:00", "2025-01-21 15:00:00", 20.0)])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    with pytest.raises(ValueError, match="carried positions.*strategy-selection"):
        portfolio_allocation.walk_forward_capital_allocation_from_csv(
            "positions",
            alpha=0,
            grid_increment=0.25,
            training_weeks=1,
            embargo_weeks=0,
            simulation_weeks=1,
            device="cpu",
            show_plot=False,
            **{option: 1 if option == "max_strategies" else 0.2},
        )


def test_csv_real_position_optimizer_can_minimize_drawdown_with_growth_floor(
    tmp_path, monkeypatch
):
    """The ledger feeds the same CMGR constraint and report equity as optimization."""
    _csv(
        tmp_path,
        [
            ("2025-01-06 10:00:00", "2025-01-08 15:00:00", 30.0),
            ("2025-01-10 10:00:00", "2025-01-10 15:00:00", -10.0),
        ],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "positions",
        alpha=0,
        objective="minimize_add",
        CMGR_limit=0.05,
        grid_increment=0.1,
        returns_dtype=pl.Float64,
        device="cpu",
        show_plot=False,
    )
    assert result.statistics.cmgr >= 0.05
    assert "Position-ledger" in result.optimization_algorithm_counts[0][0]


def test_walk_forward_ema_sizes_new_entries_without_resizing_carried_trades(
    tmp_path, monkeypatch
):
    """EMA updates entry fractions, while existing original-risk amounts stay fixed."""
    _csv(
        tmp_path,
        [
            ("2025-01-06 10:00:00", "2025-01-07 15:00:00", 10.0),
            ("2025-01-13 10:00:00", "2025-01-22 15:00:00", 20.0),
            ("2025-01-21 10:00:00", "2025-01-22 15:00:00", 10.0),
        ],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    candidates = iter((0.4, 0.2))

    def optimizer(returns, **_kwargs):
        return torch.tensor([next(candidates)], dtype=returns.dtype)

    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )
    result = portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "positions",
        alpha=0,
        grid_increment=0.1,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        allocation_alpha=0.5,
        device="cpu",
        show_plot=False,
    )
    assert result.folds[1].calculated_allocation.item() == pytest.approx(0.2)
    assert result.folds[1].applied_allocation.item() == pytest.approx(0.3)
    assert result.trade_results is not None
    trades = result.trade_results
    assert trades["allocation"].to_list() == pytest.approx([0.4, 0.3])
    # Jan 13-22 has seven market sessions, five before the second trade opens.
    equity_at_entry = 1.0 + 5 * 0.4 * 0.2 / 7
    assert trades["risk"].to_list() == pytest.approx([0.4, 0.3 * equity_at_entry])
    assert result.statistics.ending_equity == pytest.approx(
        1.0 + 0.4 * 0.2 + 0.3 * equity_at_entry * 0.1
    )


@pytest.mark.parametrize("returns_dtype", [pl.Float32, pl.Float64])
def test_position_wrapper_preserves_returns_dtype(tmp_path, monkeypatch, returns_dtype):
    """Unit-risk proxy tensors use the dtype requested by the CSV caller."""
    _csv(tmp_path, [("2025-01-06 10:00:00", "2025-01-08 15:00:00", 20.0)])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    def optimizer(returns, **_kwargs):
        expected = torch.float64 if returns_dtype == pl.Float64 else torch.float32
        assert returns.dtype == expected
        return torch.zeros(1, dtype=expected)

    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )
    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "positions",
        alpha=0,
        grid_increment=0.25,
        returns_dtype=returns_dtype,
        device="cpu",
        show_plot=False,
    )
    assert result.daily_results.schema["strategy"] == returns_dtype
    assert result.statistics.ending_equity == 1.0


def test_position_method_validation(tmp_path, monkeypatch):
    """Typos in the mark convention must not silently choose another model."""
    _csv(tmp_path, [("2025-01-06 10:00:00", "2025-01-08 15:00:00", 20.0)])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    with pytest.raises(ValueError, match="unrealized_pnl_method"):
        portfolio_allocation.find_optimal_capital_allocation_from_csv(
            "positions", alpha=0, unrealized_pnl_method="unknown", show_plot=False
        )
