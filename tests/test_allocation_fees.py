"""Fee-adjusted return propagation through daily and position-ledger wrappers."""

from pathlib import Path

import polars as pl
import pytest
import torch

from ifera.allocation import (
    capital_allocation,
    portfolio_allocation,
    walk_forward_capital_allocation_from_csv,
)
from ifera.settings import settings


def _portfolio(tmp_path: Path, multiday: bool, with_ror: bool = False) -> Path:
    directory = tmp_path / "results" / "portfolios" / "fees"
    directory.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "Opened": ["2024-01-08 10:00:00", "2024-01-16 10:00:00"],
            "Closed": (
                ["2024-01-10 16:00:00", "2024-01-19 16:00:00"]
                if multiday
                else ["2024-01-08 16:00:00", "2024-01-16 16:00:00"]
            ),
            "Symbol": ["SPX", "SPX"],
            "Exp": ["2024-01-12", "2024-01-19"],
            "Legs": [
                "long put 4,900 / short put 4,950",
                "long put 4,900 / short put 4,950 / short call 5,000 / long call 5,050",
            ],
            "Status": ["expired", "closed"],
            "Quantity": [2, 3],
            "P/L": [100.0, 100.0],
            "Risk": [1000.0, 1000.0],
        }
    )
    if with_ror:
        frame = frame.with_columns(pl.lit(999.0).alias("ROR"))
    path = directory / "strategy.csv"
    frame.write_csv(path)
    return path


def _capture_optimizer(monkeypatch):
    calls = []

    def optimize(returns, **kwargs):
        calls.append((returns.clone(), kwargs))
        return torch.tensor([0.5], dtype=returns.dtype)

    monkeypatch.setattr(capital_allocation, "find_optimal_capital_allocation", optimize)
    return calls


@pytest.mark.parametrize("multiday", [False, True])
@pytest.mark.parametrize("returns_dtype", [pl.Float32, pl.Float64])
def test_csv_fees_adjust_optimizer_inputs_and_reported_equity(
    tmp_path, monkeypatch, multiday, returns_dtype
):
    """Fees count each leg and contract, with a single charge for expired trades."""
    _portfolio(tmp_path, multiday)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    calls = _capture_optimizer(monkeypatch)

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "fees",
        alpha=0.0,
        grid_increment=0.1,
        fees_per_leg=1.25,
        returns_dtype=returns_dtype,
        device="cpu",
        show_plot=False,
    )

    returns, kwargs = calls[0]
    expected_dtype = torch.float64 if returns_dtype == pl.Float64 else torch.float32
    assert returns.dtype == expected_dtype
    assert torch.nan_to_num(returns).sum().item() == pytest.approx(0.095 + 0.07)
    assert result.statistics.ending_equity == pytest.approx(1.0475 * 1.035)
    if multiday:
        assert [trade.return_on_risk for trade in kwargs["position_model"].trades] == (
            pytest.approx([0.095, 0.07])
        )
        assert result.trade_results is not None
        assert result.trade_results["pnl"].sum() == pytest.approx(1.0475 * 1.035 - 1)
    else:
        assert "position_model" not in kwargs


@pytest.mark.parametrize("multiday", [False, True])
def test_walk_forward_uses_net_returns_for_training_and_simulation(
    tmp_path, monkeypatch, multiday
):
    _portfolio(tmp_path, multiday)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    calls = _capture_optimizer(monkeypatch)

    result = walk_forward_capital_allocation_from_csv(
        "fees",
        alpha=0.0,
        grid_increment=0.1,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        fees_per_leg=1.25,
        device="cpu",
        show_plot=False,
    )

    assert len(calls) == len(result.folds) == 1
    assert torch.nan_to_num(calls[0][0]).sum().item() == pytest.approx(0.095)
    assert result.statistics.ending_equity == pytest.approx(1.035)
    if multiday:
        assert result.trade_results is not None
        assert result.trade_results["pnl"].to_list() == pytest.approx([0.035])


@pytest.mark.parametrize("multiday", [False, True])
def test_zero_fees_recomputes_returns_from_pl_and_risk(tmp_path, monkeypatch, multiday):
    """Accounting columns, not stale or rounded ROR, are authoritative even at zero fees."""
    _portfolio(tmp_path, multiday, with_ror=True)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    calls = _capture_optimizer(monkeypatch)
    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "fees", alpha=0.0, grid_increment=0.1, device="cpu", show_plot=False
    )
    assert torch.nan_to_num(calls[0][0]).sum().item() == pytest.approx(0.2)
    assert result.statistics.ending_equity == pytest.approx(1.05**2)


@pytest.mark.parametrize("multiday", [False, True])
def test_fees_are_not_charged_for_an_unallocated_strategy(
    tmp_path, monkeypatch, multiday
):
    _portfolio(tmp_path, multiday)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        lambda returns, **_kwargs: torch.zeros(1, dtype=returns.dtype),
    )
    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "fees", alpha=0.0, fees_per_leg=10.0, device="cpu", show_plot=False
    )
    assert result.statistics.ending_equity == 1.0


@pytest.mark.parametrize("walk_forward", [False, True])
@pytest.mark.parametrize("fee", [-0.01, float("inf"), float("nan")])
def test_wrappers_reject_invalid_fees_before_loading(
    tmp_path, monkeypatch, walk_forward, fee
):
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    wrapper = (
        walk_forward_capital_allocation_from_csv
        if walk_forward
        else capital_allocation.find_optimal_capital_allocation_from_csv
    )
    kwargs = (
        {"training_weeks": 1, "embargo_weeks": 0, "simulation_weeks": 1}
        if walk_forward
        else {}
    )
    with pytest.raises(ValueError, match="fees_per_leg"):
        wrapper("missing", alpha=0.0, fees_per_leg=fee, show_plot=False, **kwargs)


@pytest.mark.parametrize("multiday", [False, True])
def test_nonzero_fees_cannot_silently_use_legacy_ror(tmp_path, monkeypatch, multiday):
    path = _portfolio(tmp_path, multiday)
    pl.read_csv(path).drop("P/L", "Risk").with_columns(
        pl.lit(10.0).alias("ROR")
    ).write_csv(path)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    with pytest.raises(ValueError, match="P/L|Risk"):
        portfolio_allocation.find_optimal_capital_allocation_from_csv(
            "fees", alpha=0.0, fees_per_leg=1.0, show_plot=False
        )


@pytest.mark.parametrize("multiday", [False, True])
def test_optimizer_chooses_cash_when_fees_make_all_trades_unprofitable(
    tmp_path, monkeypatch, multiday
):
    """Net returns affect the real optimizer, not only displayed statistics."""
    _portfolio(tmp_path, multiday)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "fees",
        alpha=0.0,
        grid_increment=0.25,
        fees_per_leg=100.0,
        device="cpu",
        show_plot=False,
    )
    assert result.allocation.tolist() == [0.0]
    assert result.statistics.ending_equity == 1.0
