"""Winning-strategy reports use an ADD-constrained capital allocation."""

from datetime import date
from unittest.mock import MagicMock

import polars as pl
import pytest
import torch

from ifera.strategy_optimization import _reporting, optimize_strategy_from_csv
from tests.test_strategy_optimization import strategy_market

_DATES = (
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
)


def test_allocated_report_forwards_single_strategy_and_requested_parameters(
    monkeypatch,
):
    raw = torch.tensor([10.0, 0.0, -20.0, 0.0], dtype=torch.float32)
    original = raw.clone()
    active = torch.tensor([True, True, True, False])
    allocator = MagicMock(return_value=torch.tensor([0.2], dtype=torch.float64))
    monkeypatch.setattr(_reporting, "find_optimal_capital_allocation", allocator)

    daily, statistics, allocation = _reporting.allocated_strategy_statistics(
        _DATES[:4], raw, active
    )

    allocator.assert_called_once()
    inputs = allocator.call_args.args[0]
    assert inputs.shape == (4, 1)
    assert inputs.dtype == torch.float64
    torch.testing.assert_close(inputs[:3, 0], raw[:3].double() / 100.0)
    assert inputs[1, 0] == 0.0  # An executed breakeven trade remains active.
    assert torch.isnan(inputs[3, 0])  # Only the no-trade session becomes NaN.
    kwargs = allocator.call_args.kwargs
    assert kwargs["alpha"] == 0.0
    assert kwargs["grid_increment"] == 0.001
    assert kwargs["bootstrap_on"] is False
    assert kwargs["refinement_runs"] == 0
    assert kwargs["ADD_limit"] == -0.1
    assert kwargs["dtype"] == torch.float64
    assert torch.device(kwargs["device"]) == raw.device
    assert allocation == 0.2
    assert daily["date"].to_list() == list(_DATES[:4])
    assert daily["combined_return"].to_list() == pytest.approx([0.02, 0.0, -0.04, 0.0])
    assert daily["active"].to_list() == [True, True, True, False]
    assert statistics.active_days == 3
    assert statistics.win_rate == pytest.approx(1.0 / 3.0)
    torch.testing.assert_close(raw, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_allocation_keeps_a_full_loss_trade_solvent_and_respects_add(dtype):
    raw = torch.tensor([50.0, 50.0, -100.0, 50.0, 0.0], dtype=dtype)
    original = raw.clone()
    active = torch.tensor([True, True, True, True, False])

    daily, statistics, allocation = _reporting.allocated_strategy_statistics(
        _DATES, raw, active
    )

    assert 0.0 < allocation < 1.0
    assert statistics.average_drawdown >= -0.1 - 1e-8
    assert statistics.average_drawdown < -0.09
    assert statistics.max_drawdown > -1.0
    assert statistics.ending_equity > 1.0
    assert daily["date"].to_list() == list(_DATES)
    assert statistics.market_days == len(_DATES)
    assert statistics.active_days == 4
    assert daily["combined_return"].to_list() == pytest.approx(
        [0.5 * allocation, 0.5 * allocation, -allocation, 0.5 * allocation, 0.0]
    )
    expected_equity = (1.0 + 0.5 * allocation) ** 3 * (1.0 - allocation)
    assert statistics.ending_equity == pytest.approx(expected_equity)
    assert daily["equity"].item(-1) == pytest.approx(expected_equity)
    torch.testing.assert_close(raw, original)


def test_all_filtered_strategy_retains_original_calendar_and_cash_equity():
    daily, statistics, allocation = _reporting.allocated_strategy_statistics(
        _DATES, torch.zeros(5), torch.zeros(5, dtype=torch.bool)
    )

    assert allocation == 0.0
    assert daily["date"].to_list() == list(_DATES)
    assert daily["combined_return"].to_list() == [0.0] * 5
    assert daily["equity"].to_list() == [1.0] * 5
    assert daily["active"].to_list() == [False] * 5
    assert statistics.active_days == 0
    assert statistics.market_days == 5
    assert statistics.average_drawdown == 0.0
    assert statistics.ending_equity == 1.0


def test_allocated_report_preserves_slippage_loss_beyond_original_risk():
    raw = torch.tensor([75.0, 75.0, -125.0, 75.0, 0.0], dtype=torch.float64)
    active = torch.tensor([True, True, True, True, False])

    daily, statistics, allocation = _reporting.allocated_strategy_statistics(
        _DATES, raw, active
    )

    assert 0.0 < allocation < 0.8
    assert statistics.average_drawdown >= -0.1 - 1e-8
    assert statistics.ending_equity > 1.0
    assert daily["combined_return"].item(2) == pytest.approx(-1.25 * allocation)
    expected_equity = (1.0 + 0.75 * allocation) ** 3 * (1.0 - 1.25 * allocation)
    assert statistics.ending_equity == pytest.approx(expected_equity)
    assert raw[2] == -125.0


def test_zero_allocation_makes_executed_losing_strategy_inactive(monkeypatch):
    allocator = MagicMock(return_value=torch.zeros(1, dtype=torch.float64))
    monkeypatch.setattr(_reporting, "find_optimal_capital_allocation", allocator)

    daily, statistics, allocation = _reporting.allocated_strategy_statistics(
        _DATES[:3],
        torch.tensor([-100.0, 0.0, -50.0]),
        torch.tensor([True, True, True]),
    )

    assert allocation == 0.0
    assert daily["active"].to_list() == [False] * 3
    assert daily["combined_return"].to_list() == [0.0] * 3
    assert statistics.active_days == 0
    assert statistics.win_rate == 0.0


@pytest.mark.parametrize(
    "dates,returns,active,error",
    [
        ((), torch.zeros(0), torch.zeros(0, dtype=torch.bool), ValueError),
        (_DATES[:1], torch.zeros(2), torch.ones(2, dtype=torch.bool), ValueError),
        (_DATES[:1], torch.zeros(1), torch.ones(1), TypeError),
        (_DATES[:1], torch.zeros(1, 1), torch.ones(1, dtype=torch.bool), ValueError),
        (
            (_DATES[1], _DATES[0]),
            torch.zeros(2),
            torch.ones(2, dtype=torch.bool),
            ValueError,
        ),
        (
            _DATES[:1],
            torch.tensor([float("nan")]),
            torch.tensor([False]),
            ValueError,
        ),
    ],
)
def test_invalid_allocated_statistics_input_fails_before_allocation(
    monkeypatch, dates, returns, active, error
):
    allocator = MagicMock(side_effect=AssertionError("Invalid input reached allocator"))
    monkeypatch.setattr(_reporting, "find_optimal_capital_allocation", allocator)

    with pytest.raises(error):
        _reporting.allocated_strategy_statistics(dates, returns, active)

    allocator.assert_not_called()


def test_facade_reports_allocated_statistics_without_changing_trades(
    strategy_market, tmp_path, capsys, monkeypatch
):
    data, market = strategy_market
    features = tmp_path / "features.yaml"
    features.write_text("features:\n  clock:\n    template: time\n", encoding="utf-8")
    original = data.path.read_bytes()
    allocator = MagicMock(wraps=_reporting.find_optimal_capital_allocation)
    monkeypatch.setattr(_reporting, "find_optimal_capital_allocation", allocator)

    class OfflineSource:
        def load(self, _data, lookback_days=400):
            return market

    def metric(returns):
        assert returns.shape[1] == 3
        return (returns[:, 0] - returns[:, 2] + 1.0).clamp_min(1e-6)

    result = optimize_strategy_from_csv(
        "sample",
        "strategy",
        features_file=features,
        market_data_source=OfflineSource(),
        metric=metric,
        optimizer_parameters={
            "max_population": 2,
            "generations": 0,
            "max_conjunctions": 1,
            "device": "cpu",
            "seed": 7,
        },
        output_csv=tmp_path / "optimized.csv",
        show_progress=False,
        show_plot=False,
    )

    # This losing strategy should be held entirely in cash, rather than lose
    # half its equity merely because its CSV reports returns per unit of risk.
    assert result.statistics.average_drawdown >= -0.1
    allocator.assert_called_once()
    assert result.allocation == 0.0
    assert result.daily_results["combined_return"].to_list() == [0.0, 0.0, 0.0]
    assert result.daily_results["equity"].to_list() == [1.0, 1.0, 1.0]
    assert result.statistics.market_days == 3
    assert result.statistics.active_days == 0
    assert result.optimization.objective_value == 76.0
    assert result.objective_history == (76.0,)
    assert result.trades.equals(data.raw)
    assert pl.read_csv(result.output_csv).equals(data.raw)
    assert data.path.read_bytes() == original
    assert "Capital allocation: 0.00%" in capsys.readouterr().out
