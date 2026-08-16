"""Tests for loading, reporting, and plotting portfolio allocations."""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import matplotlib
import polars as pl
import pytest
import torch

import ifera.allocation.capital_allocation as capital_allocation
import ifera.allocation.portfolio_allocation as portfolio_allocation
from ifera.settings import settings

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position


def _portfolio_directory(tmp_path: Path, portfolio_name: str = "portfolio") -> Path:
    """Create and return a temporary portfolio directory."""
    directory = tmp_path / "results" / "portfolios" / portfolio_name
    directory.mkdir(parents=True)
    return directory


def _write_strategy(
    path: Path, opened: list[str], returns_on_risk: list[float]
) -> None:
    """Write the subset of the backtest CSV schema needed by the wrapper."""
    pl.DataFrame({"Opened": opened, "ROR": returns_on_risk}).write_csv(path)


def _fixed_optimizer(
    allocation: list[float], captured: dict[str, object] | None = None
):
    """Return a stand-in optimizer which optionally records all arguments."""

    def optimizer(
        returns: torch.Tensor,
        alpha: float,
        grid_increment: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        bootstrap_on: bool = False,
        bootstrap_runs: int = 1024,
        bootstrap_length: int = 256,
        percentile: float = 10.0,
        ADD_limit: float | None = None,
        max_total_allocation: float = 1.0,
        refinement_runs: int = 0,
        refinement_divisor: float = 2.0,
        diagnostics: capital_allocation.CapitalAllocationDiagnostics | None = None,
    ) -> torch.Tensor:
        if captured is not None:
            captured.update(
                {
                    "returns": returns.clone(),
                    "alpha": alpha,
                    "grid_increment": grid_increment,
                    "device": device,
                    "dtype": dtype,
                    "bootstrap_on": bootstrap_on,
                    "bootstrap_runs": bootstrap_runs,
                    "bootstrap_length": bootstrap_length,
                    "percentile": percentile,
                    "ADD_limit": ADD_limit,
                    "max_total_allocation": max_total_allocation,
                    "refinement_runs": refinement_runs,
                    "refinement_divisor": refinement_divisor,
                    "diagnostics": diagnostics,
                }
            )
        allocation_tensor = torch.tensor(
            allocation,
            device=torch.device("cpu") if device is None else device,
            dtype=returns.dtype if dtype is None else dtype,
        )
        if diagnostics is not None:
            diagnostics.bootstrap_average_drawdown = -0.123 if bootstrap_on else None
        return allocation_tensor

    return optimizer


def _sequential_optimizer(
    allocations: list[list[float]], captured: list[dict[str, object]]
):
    """Return successive allocations while retaining every training invocation."""

    def optimizer(
        returns: torch.Tensor,
        alpha: float,
        grid_increment: float,
        **kwargs: object,
    ) -> torch.Tensor:
        call_index = len(captured)
        captured.append(
            {
                "returns": returns.clone(),
                "alpha": alpha,
                "grid_increment": grid_increment,
                **kwargs,
            }
        )
        dtype = kwargs.get("dtype") or returns.dtype
        device = kwargs.get("device") or returns.device
        return torch.tensor(allocations[call_index], dtype=dtype, device=device)

    return optimizer


def _write_complete_market_strategy(
    path: Path,
    start_date: datetime.date,
    end_date: datetime.date,
    multiplier: float = 1.0,
) -> tuple[list[datetime.date], dict[datetime.date, float]]:
    """Write one observation per NYSE session with date-identifiable returns."""
    market_dates = portfolio_allocation._nyse_market_dates(start_date, end_date)[
        "date"
    ].to_list()
    fractional_returns = {
        market_date: multiplier * (index + 1) / 100.0
        for index, market_date in enumerate(market_dates)
    }
    _write_strategy(
        path,
        [f"{market_date.isoformat()} 10:00:00" for market_date in market_dates],
        [fractional_returns[market_date] * 100.0 for market_date in market_dates],
    )
    return market_dates, fractional_returns


def test_wrapper_aligns_nyse_sessions_and_forwards_optimizer_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """CSV columns align by date while holidays are omitted and empty sessions remain."""
    directory = _portfolio_directory(tmp_path, "calendar_case")
    _write_strategy(
        directory / "zeta.csv",
        ["Dec 23, 2024 10:15am"],
        [10.0],
    )
    _write_strategy(
        directory / "alpha.csv",
        ["Dec 26, 2024 11:30am"],
        [20.0],
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([0.25, 0.75], captured),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "calendar_case",
        alpha=0.7,
        grid_increment=0.3,
        returns_dtype=pl.Float64,
        device="cpu",
        dtype=torch.float32,
        bootstrap_on=True,
        bootstrap_runs=17,
        bootstrap_length=23,
        percentile=12.5,
        ADD_limit=-0.15,
        max_total_allocation=1.5,
        refinement_runs=3,
        refinement_divisor=4.0,
        show_plot=False,
    )

    expected_returns = torch.tensor(
        [
            [math.nan, 0.10],
            [math.nan, math.nan],
            [0.20, math.nan],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(captured["returns"], expected_returns, equal_nan=True)
    assert captured["alpha"] == 0.7
    assert captured["grid_increment"] == 0.3
    assert captured["device"] == "cpu"
    assert captured["dtype"] is torch.float32
    assert captured["bootstrap_on"] is True
    assert captured["bootstrap_runs"] == 17
    assert captured["bootstrap_length"] == 23
    assert captured["percentile"] == 12.5
    assert captured["ADD_limit"] == -0.15
    assert captured["max_total_allocation"] == 1.5
    assert captured["refinement_runs"] == 3
    assert captured["refinement_divisor"] == 4.0
    assert isinstance(
        captured["diagnostics"], capital_allocation.CapitalAllocationDiagnostics
    )
    assert result.strategy_names == ("alpha", "zeta")
    assert torch.equal(result.allocation.cpu(), torch.tensor([0.25, 0.75]))
    assert result.daily_results["date"].to_list() == [
        datetime.date(2024, 12, 23),
        datetime.date(2024, 12, 24),
        datetime.date(2024, 12, 26),
    ]
    assert result.daily_results["combined_return"].to_list() == pytest.approx(
        [0.075, 0.0, 0.05]
    )
    assert result.daily_results["active"].to_list() == [True, False, True]
    assert result.statistics.active_days == 2
    assert result.statistics.win_rate == pytest.approx(1.0)
    assert result.statistics.bootstrap_average_drawdown == pytest.approx(-0.123)

    output = capsys.readouterr().out
    for text in (
        "calendar_case",
        "alpha",
        "zeta",
        "CAGR",
        "CMGR",
        "Win",
        "ADD",
        "ADD (bootstrap)",
        "MaxDD",
        "MaxDD duration",
        "Sharpe",
        "Sortino",
    ):
        assert text.casefold() in output.casefold()
    output_lines = [line.strip() for line in output.splitlines()]
    cagr_line = next(
        index for index, line in enumerate(output_lines) if line.startswith("CAGR:")
    )
    assert output_lines[cagr_line + 1].startswith("CMGR:")
    add_line = next(
        index for index, line in enumerate(output_lines) if line.startswith("ADD:")
    )
    assert output_lines[add_line + 1].startswith("ADD (bootstrap):")


def test_wrapper_defaults_csv_returns_to_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """CSV return storage and optimizer input default to single precision."""
    directory = _portfolio_directory(tmp_path, "default_returns_dtype")
    _write_strategy(
        directory / "strategy.csv",
        ["Jan 13, 2025 10:00am", "Jan 14, 2025 10:00am"],
        [12.5, -6.25],
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([1.0], captured),
    )

    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "default_returns_dtype",
        alpha=0.0,
        grid_increment=1.0,
        device="cpu",
        show_plot=False,
    )

    assert isinstance(captured["returns"], torch.Tensor)
    assert captured["returns"].dtype is torch.float32
    assert result.daily_results.schema["strategy"] == pl.Float32
    assert result.allocation.dtype is torch.float32


def test_wrapper_rejects_nonfloating_returns_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The CSV dtype cannot discard fractions or silently promote after division."""
    directory = _portfolio_directory(tmp_path, "invalid_returns_dtype")
    _write_strategy(directory / "strategy.csv", ["Jan 13, 2025 10:00am"], [1.0])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match="returns_dtype|Float32|Float64"):
        portfolio_allocation.find_optimal_capital_allocation_from_csv(
            "invalid_returns_dtype",
            alpha=0.0,
            grid_increment=1.0,
            returns_dtype=pl.Int64,
            device="cpu",
            show_plot=False,
        )


def test_wrapper_reports_pairwise_diversification_from_inactive_zero_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Diversification correlations include zero returns on inactive sessions."""
    directory = _portfolio_directory(tmp_path, "diversification_case")
    _write_strategy(
        directory / "alpha.csv",
        ["Jan 13, 2025 10:00am", "Jan 15, 2025 10:00am"],
        [10.0, -10.0],
    )
    _write_strategy(
        directory / "beta.csv",
        [
            "Jan 13, 2025 10:00am",
            "Jan 14, 2025 10:00am",
            "Jan 15, 2025 10:00am",
            "Jan 16, 2025 10:00am",
            "Jan 17, 2025 10:00am",
        ],
        [10.0, 20.0, -10.0, -10.0, -10.0],
    )
    _write_strategy(
        directory / "contrarian.csv",
        ["Jan 13, 2025 10:00am", "Jan 15, 2025 10:00am"],
        [-10.0, 10.0],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([0.2, 0.3, 0.5]),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "diversification_case", 0.0, 1.0, device="cpu", show_plot=False
    )

    diversification = result.diversification
    assert diversification.strategy_count == 3
    assert diversification.allocated_strategy_count == 3
    assert diversification.total_pair_count == 3
    assert diversification.valid_pair_count == 3
    torch.testing.assert_close(
        diversification.correlation_matrix,
        torch.tensor(
            [[1.0, 0.5, -1.0], [0.5, 1.0, -0.5], [-1.0, -0.5, 1.0]],
            dtype=torch.float64,
        ),
    )
    assert diversification.highest_correlation == pytest.approx(0.5)
    assert diversification.highest_correlation_pair == ("alpha", "beta")
    assert diversification.lowest_negative_correlation == pytest.approx(-1.0)
    assert diversification.lowest_negative_correlation_pair == (
        "alpha",
        "contrarian",
    )
    assert diversification.mean_absolute_correlation == pytest.approx(2.0 / 3.0)
    assert diversification.mean_correlation == pytest.approx(-1.0 / 3.0)
    assert diversification.allocation_concentration == pytest.approx(0.38)
    assert diversification.effective_strategy_count == pytest.approx(1.0 / 0.38)

    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    section_line = output_lines.index("Portfolio robustness / diversification:")
    statistics_line = output_lines.index("Combined portfolio statistics:")
    assert section_line > statistics_line
    assert any(
        line.startswith("Highest pairwise correlation:")
        and "0.500" in line
        and "alpha" in line
        and "beta" in line
        for line in output_lines[section_line:]
    )
    assert any(
        line.startswith("Lowest negative correlation:")
        and "-1.000" in line
        and "alpha" in line
        and "contrarian" in line
        for line in output_lines[section_line:]
    )
    assert any(
        line.startswith("Mean absolute correlation:") and "0.667" in line
        for line in output_lines[section_line:]
    )
    assert any(
        line.startswith("Mean pairwise correlation:") and "-0.333" in line
        for line in output_lines[section_line:]
    )
    assert any(
        line.startswith("Valid pairwise correlations:") and "3/3" in line
        for line in output_lines[section_line:]
    )
    assert any(
        line.startswith("Effective strategy count:") and "2.632" in line
        for line in output_lines[section_line:]
    )
    assert any(
        line.startswith("Allocated strategies:") and "3/3" in line
        for line in output_lines[section_line:]
    )
    assert any(
        line.startswith("Allocation concentration (HHI):") and "0.380" in line
        for line in output_lines[section_line:]
    )


def test_diversification_excludes_undefined_constant_pairs_and_has_no_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Constant strategies do not poison valid summaries or invent correlations."""
    directory = _portfolio_directory(tmp_path, "constant_correlation_case")
    opened = [
        "Jan 13, 2025 10:00am",
        "Jan 14, 2025 10:00am",
        "Jan 15, 2025 10:00am",
    ]
    _write_strategy(directory / "constant.csv", opened, [5.0, 5.0, 5.0])
    _write_strategy(directory / "left.csv", opened, [10.0, 0.0, -10.0])
    _write_strategy(directory / "right.csv", opened, [20.0, 0.0, -20.0])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([0.0, 0.5, 0.5]),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "constant_correlation_case", 0.0, 1.0, device="cpu", show_plot=False
    )

    diversification = result.diversification
    assert diversification.strategy_count == 3
    assert diversification.allocated_strategy_count == 2
    assert diversification.total_pair_count == 3
    assert diversification.valid_pair_count == 1
    torch.testing.assert_close(
        diversification.correlation_matrix,
        torch.tensor(
            [
                [math.nan, math.nan, math.nan],
                [math.nan, 1.0, 1.0],
                [math.nan, 1.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        equal_nan=True,
    )
    assert diversification.highest_correlation == pytest.approx(1.0)
    assert diversification.highest_correlation_pair == ("left", "right")
    assert diversification.lowest_negative_correlation is None
    assert diversification.lowest_negative_correlation_pair is None
    assert diversification.mean_absolute_correlation == pytest.approx(1.0)
    assert diversification.mean_correlation == pytest.approx(1.0)
    assert diversification.allocation_concentration == pytest.approx(0.5)
    assert diversification.effective_strategy_count == pytest.approx(2.0)

    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert "Lowest negative correlation: N/A" in output_lines
    assert "Valid pairwise correlations: 1/3" in output_lines


def test_single_strategy_diversification_reports_unavailable_pairwise_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A one-strategy portfolio has concentration data but no pairwise statistics."""
    directory = _portfolio_directory(tmp_path, "single_diversification_case")
    _write_strategy(
        directory / "only.csv",
        ["Jan 13, 2025 10:00am", "Jan 14, 2025 10:00am"],
        [1.0, -1.0],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([1.0]),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "single_diversification_case", 0.0, 1.0, device="cpu", show_plot=False
    )

    diversification = result.diversification
    assert diversification.strategy_count == 1
    assert diversification.allocated_strategy_count == 1
    assert diversification.total_pair_count == 0
    assert diversification.valid_pair_count == 0
    torch.testing.assert_close(
        diversification.correlation_matrix,
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    assert diversification.highest_correlation is None
    assert diversification.highest_correlation_pair is None
    assert diversification.lowest_negative_correlation is None
    assert diversification.lowest_negative_correlation_pair is None
    assert diversification.mean_absolute_correlation is None
    assert diversification.mean_correlation is None
    assert diversification.allocation_concentration == pytest.approx(1.0)
    assert diversification.effective_strategy_count == pytest.approx(1.0)

    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert "Highest pairwise correlation: N/A" in output_lines
    assert "Lowest negative correlation: N/A" in output_lines
    assert "Mean absolute correlation: N/A" in output_lines
    assert "Mean pairwise correlation: N/A" in output_lines
    assert "Valid pairwise correlations: 0/0" in output_lines


def test_zero_allocation_has_pairwise_correlations_but_no_concentration_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Pairwise statistics remain meaningful when the optimizer allocates nothing."""
    directory = _portfolio_directory(tmp_path, "zero_allocation_case")
    opened = ["Jan 13, 2025 10:00am", "Jan 14, 2025 10:00am"]
    _write_strategy(directory / "first.csv", opened, [1.0, -1.0])
    _write_strategy(directory / "second.csv", opened, [2.0, -2.0])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([0.0, 0.0]),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "zero_allocation_case", 0.0, 1.0, device="cpu", show_plot=False
    )

    diversification = result.diversification
    assert diversification.strategy_count == 2
    assert diversification.allocated_strategy_count == 0
    assert diversification.valid_pair_count == 1
    torch.testing.assert_close(
        diversification.correlation_matrix,
        torch.ones((2, 2), dtype=torch.float64),
    )
    assert diversification.highest_correlation == pytest.approx(1.0)
    assert diversification.allocation_concentration is None
    assert diversification.effective_strategy_count == 0.0

    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert "Allocated strategies: 0/2" in output_lines
    assert "Allocation concentration (HHI): N/A" in output_lines
    assert "Effective strategy count: 0.000" in output_lines


@pytest.mark.parametrize("scale", [1e150, 1e200, 1e-200])
def test_strategy_correlations_are_stable_across_extreme_finite_scales(scale: float):
    """Pearson coefficients remain scale invariant without overflow or underflow."""
    returns = torch.tensor(
        [[-scale, -2.0 * scale], [0.0, 0.0], [scale, 2.0 * scale]],
        dtype=torch.float64,
    )

    correlation_matrix = portfolio_allocation._strategy_correlation_matrix(returns)

    torch.testing.assert_close(
        correlation_matrix,
        torch.ones((2, 2), dtype=torch.float64),
    )


def test_result_contains_expected_combined_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Reported statistics use compounded equity and initial-wealth drawdowns."""
    directory = _portfolio_directory(tmp_path, "statistics_case")
    _write_strategy(
        directory / "strategy.csv",
        [
            "Jan 13, 2025 10:00am",
            "Jan 14, 2025 10:00am",
            "Jan 15, 2025 10:00am",
            "Jan 16, 2025 10:00am",
        ],
        [10.0, 0.0, -5.0, 2.0],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([1.0]),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "statistics_case",
        alpha=0.0,
        grid_increment=1.0,
        device="cpu",
        show_plot=False,
    )

    expected_equity = [1.1, 1.1, 1.045, 1.0659]
    expected_add = -math.sqrt((0.05**2 + 0.031**2) / 4.0)
    statistics = result.statistics
    assert result.daily_results["equity"].to_list() == pytest.approx(expected_equity)
    assert statistics.total_return == pytest.approx(0.0659)
    assert statistics.cagr == pytest.approx(1.0659 ** (365.25 / 4.0) - 1.0)
    assert statistics.cmgr == pytest.approx(1.0659 ** (365.25 / (12.0 * 4.0)) - 1.0)
    assert statistics.win_rate == pytest.approx(0.5)
    assert statistics.average_drawdown == pytest.approx(expected_add)
    assert statistics.bootstrap_average_drawdown is None
    assert statistics.max_drawdown == pytest.approx(-0.05)
    assert statistics.max_drawdown_duration == 2
    assert statistics.annualized_volatility > 0.0
    assert statistics.sharpe_ratio > 0.0
    assert statistics.sortino_ratio > 0.0
    assert statistics.calmar_ratio > 0.0
    assert statistics.profit_factor == pytest.approx(2.4)
    assert statistics.best_day == pytest.approx(0.10)
    assert statistics.worst_day == pytest.approx(-0.05)
    assert statistics.active_days == 4
    assert statistics.exposure == pytest.approx(1.0)
    assert "ADD (bootstrap)" not in capsys.readouterr().out


def test_wrapper_displays_equity_curve_on_log_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The displayed optimal equity curve has a logarithmic vertical axis."""
    directory = _portfolio_directory(tmp_path, "plot_case")
    _write_strategy(
        directory / "strategy.csv",
        ["Jan 6, 2025 10:00am", "Jan 7, 2025 10:00am"],
        [2.0, 3.0],
    )
    observed_scales: list[str] = []
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([1.0]),
    )
    monkeypatch.setattr(
        plt, "show", lambda: observed_scales.append(plt.gca().get_yscale())
    )

    capital_allocation.find_optimal_capital_allocation_from_csv(
        "plot_case",
        alpha=0.0,
        grid_increment=1.0,
        device="cpu",
        show_plot=True,
    )

    assert observed_scales == ["log"]
    plt.close("all")


def test_walk_forward_uses_rolling_week_windows_embargo_and_partial_final_week(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Each fold retrains on a fixed window and clips only its final simulation."""
    directory = _portfolio_directory(tmp_path, "walk_forward_windows")
    all_dates, alpha_returns = _write_complete_market_strategy(
        directory / "alpha.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 2, 19),
    )
    _, beta_returns = _write_complete_market_strategy(
        directory / "beta.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 2, 19),
        multiplier=2.0,
    )
    optimizer_calls: list[dict[str, object]] = []
    progress_calls: list[tuple[int, dict[str, object]]] = []

    def recording_tqdm(iterable, **kwargs):
        items = tuple(iterable)
        progress_calls.append((len(items), kwargs))
        return items

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _sequential_optimizer([[1.0, 0.0], [0.0, 1.0]], optimizer_calls),
    )
    monkeypatch.setattr(portfolio_allocation, "tqdm", recording_tqdm, raising=False)

    result = capital_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_windows",
        alpha=0.4,
        grid_increment=0.5,
        training_weeks=2,
        embargo_weeks=1,
        simulation_weeks=2,
        returns_dtype=pl.Float64,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=7,
        bootstrap_length=8,
        percentile=9.0,
        ADD_limit=-0.2,
        max_total_allocation=1.5,
        refinement_runs=2,
        refinement_divisor=3.0,
        show_plot=False,
    )

    assert isinstance(result, portfolio_allocation.WalkForwardPortfolioAllocationResult)
    assert result.strategy_names == ("alpha", "beta")
    assert len(optimizer_calls) == 2
    expected_training_ranges = (
        (datetime.date(2025, 1, 6), datetime.date(2025, 1, 17)),
        (datetime.date(2025, 1, 21), datetime.date(2025, 1, 31)),
    )
    for call, (start_date, end_date) in zip(optimizer_calls, expected_training_ranges):
        training_dates = [
            market_date
            for market_date in all_dates
            if start_date <= market_date <= end_date
        ]
        expected_returns = torch.tensor(
            [
                [alpha_returns[market_date], beta_returns[market_date]]
                for market_date in training_dates
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(call["returns"], expected_returns)
        assert call["alpha"] == 0.4
        assert call["grid_increment"] == 0.5
        assert call["bootstrap_on"] is True
        assert call["bootstrap_runs"] == 7
        assert call["bootstrap_length"] == 8
        assert call["percentile"] == 9.0
        assert call["ADD_limit"] == -0.2
        assert call["max_total_allocation"] == 1.5
        assert call["refinement_runs"] == 2
        assert call["refinement_divisor"] == 3.0

    assert len(result.folds) == 2
    first_fold, second_fold = result.folds
    assert first_fold.training_start_date == datetime.date(2025, 1, 6)
    assert first_fold.training_end_date == datetime.date(2025, 1, 17)
    assert first_fold.simulation_start_date == datetime.date(2025, 1, 27)
    assert first_fold.simulation_end_date == datetime.date(2025, 2, 7)
    torch.testing.assert_close(
        first_fold.allocation, torch.tensor([1.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        first_fold.calculated_allocation,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        first_fold.applied_allocation,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert second_fold.training_start_date == datetime.date(2025, 1, 21)
    assert second_fold.training_end_date == datetime.date(2025, 1, 31)
    assert second_fold.simulation_start_date == datetime.date(2025, 2, 10)
    assert second_fold.simulation_end_date == datetime.date(2025, 2, 19)
    torch.testing.assert_close(
        second_fold.allocation, torch.tensor([0.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        second_fold.calculated_allocation,
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        second_fold.applied_allocation,
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )

    expected_simulation_dates = [
        market_date
        for market_date in all_dates
        if datetime.date(2025, 1, 27) <= market_date <= datetime.date(2025, 2, 19)
    ]
    expected_combined_returns = [
        (
            alpha_returns[market_date]
            if market_date <= datetime.date(2025, 2, 7)
            else beta_returns[market_date]
        )
        for market_date in expected_simulation_dates
    ]
    assert result.daily_results["date"].to_list() == expected_simulation_dates
    assert result.daily_results["combined_return"].to_list() == pytest.approx(
        expected_combined_returns
    )
    expected_equity: list[float] = []
    current_equity = 1.0
    for combined_return in expected_combined_returns:
        current_equity *= 1.0 + combined_return
        expected_equity.append(current_equity)
    assert result.daily_results["equity"].to_list() == pytest.approx(expected_equity)
    assert result.statistics.start_date == datetime.date(2025, 1, 27)
    assert result.statistics.end_date == datetime.date(2025, 2, 19)
    assert result.statistics.market_days == len(expected_simulation_dates)
    assert result.statistics.bootstrap_average_drawdown is None
    assert result.diversification.highest_correlation == pytest.approx(1.0)
    expected_concentration = (10.0 / 17.0) ** 2 + (7.0 / 17.0) ** 2
    assert result.diversification.allocation_concentration == pytest.approx(
        expected_concentration
    )

    assert len(progress_calls) == 1
    progress_length, progress_kwargs = progress_calls[0]
    assert progress_kwargs.get("total", progress_length) == 2
    output = capsys.readouterr().out
    for statistic_name in (
        "CAGR",
        "CMGR",
        "Win rate",
        "ADD",
        "MaxDD",
        "Sharpe ratio",
        "Sortino ratio",
        "Portfolio robustness / diversification",
    ):
        assert statistic_name in output
    assert "ADD (bootstrap)" not in output


def test_walk_forward_allocation_alpha_recursively_smooths_applied_allocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """EMA uses the prior applied allocation for returns and aggregate reporting."""
    directory = _portfolio_directory(tmp_path, "walk_forward_allocation_alpha")
    all_dates, alpha_returns = _write_complete_market_strategy(
        directory / "alpha.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 31),
    )
    _, beta_returns = _write_complete_market_strategy(
        directory / "beta.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 31),
        multiplier=2.0,
    )
    calculated = ([1.0, 0.0], [0.0, 1.0], [0.8, 0.2])
    optimizer_calls: list[dict[str, object]] = []
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _sequential_optimizer([list(item) for item in calculated], optimizer_calls),
    )

    result = capital_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_allocation_alpha",
        alpha=0.0,
        grid_increment=0.1,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        returns_dtype=pl.Float64,
        allocation_alpha=0.25,
        device="cpu",
        show_plot=False,
    )

    expected_applied = (
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.75, 0.25], dtype=torch.float64),
        torch.tensor([0.7625, 0.2375], dtype=torch.float64),
    )
    assert len(result.folds) == 3
    for fold, expected_calculated, expected_smoothed in zip(
        result.folds, calculated, expected_applied, strict=True
    ):
        torch.testing.assert_close(
            fold.calculated_allocation,
            torch.tensor(expected_calculated, dtype=torch.float64),
        )
        torch.testing.assert_close(fold.applied_allocation, expected_smoothed)
        # The original result field remains an alias for the allocation actually used.
        torch.testing.assert_close(fold.allocation, expected_smoothed)

    simulation_periods = (
        (datetime.date(2025, 1, 13), datetime.date(2025, 1, 17)),
        (datetime.date(2025, 1, 21), datetime.date(2025, 1, 24)),
        (datetime.date(2025, 1, 27), datetime.date(2025, 1, 31)),
    )
    expected_returns: list[float] = []
    expected_dates: list[datetime.date] = []
    simulation_day_counts: list[int] = []
    for (period_start, period_end), applied in zip(
        simulation_periods, expected_applied, strict=True
    ):
        period_dates = [
            market_date
            for market_date in all_dates
            if period_start <= market_date <= period_end
        ]
        simulation_day_counts.append(len(period_dates))
        expected_dates.extend(period_dates)
        expected_returns.extend(
            applied[0].item() * alpha_returns[market_date]
            + applied[1].item() * beta_returns[market_date]
            for market_date in period_dates
        )

    assert result.daily_results["date"].to_list() == expected_dates
    assert result.daily_results["combined_return"].to_list() == pytest.approx(
        expected_returns
    )
    expected_equity = math.prod(1.0 + value for value in expected_returns)
    assert result.statistics.ending_equity == pytest.approx(expected_equity)
    expected_average = sum(
        allocation * day_count
        for allocation, day_count in zip(
            expected_applied, simulation_day_counts, strict=True
        )
    ) / sum(simulation_day_counts)
    torch.testing.assert_close(result.average_allocation, expected_average)
    expected_concentration = float(torch.square(expected_average).sum().item())
    assert result.diversification.allocation_concentration == pytest.approx(
        expected_concentration
    )


def test_walk_forward_excludes_an_initial_partial_calendar_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A dataset beginning after an available Monday session starts next week."""
    directory = _portfolio_directory(tmp_path, "walk_forward_partial_start")
    _, returns_by_date = _write_complete_market_strategy(
        directory / "strategy.csv",
        datetime.date(2025, 1, 7),
        datetime.date(2025, 1, 31),
    )
    optimizer_calls: list[dict[str, object]] = []
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _sequential_optimizer([[1.0], [1.0]], optimizer_calls),
    )

    result = portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_partial_start",
        alpha=0.0,
        grid_increment=1.0,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        device="cpu",
        show_plot=False,
    )

    expected_first_training_dates = [
        datetime.date(2025, 1, 13),
        datetime.date(2025, 1, 14),
        datetime.date(2025, 1, 15),
        datetime.date(2025, 1, 16),
        datetime.date(2025, 1, 17),
    ]
    torch.testing.assert_close(
        optimizer_calls[0]["returns"],
        torch.tensor(
            [
                [returns_by_date[market_date]]
                for market_date in expected_first_training_dates
            ],
            dtype=torch.float32,
        ),
    )
    assert result.folds[0].training_start_date == datetime.date(2025, 1, 13)
    assert result.daily_results["date"].item(0) == datetime.date(2025, 1, 21)
    assert all(
        market_date >= datetime.date(2025, 1, 21)
        for market_date in result.daily_results["date"]
    )


def test_walk_forward_keeps_holiday_shortened_initial_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Starting on Tuesday after a Monday NYSE holiday is a complete week."""
    directory = _portfolio_directory(tmp_path, "walk_forward_holiday_start")
    _, returns_by_date = _write_complete_market_strategy(
        directory / "strategy.csv",
        datetime.date(2025, 1, 21),
        datetime.date(2025, 1, 31),
    )
    optimizer_calls: list[dict[str, object]] = []
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _sequential_optimizer([[1.0]], optimizer_calls),
    )

    result = portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_holiday_start",
        alpha=0.0,
        grid_increment=1.0,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        device="cpu",
        show_plot=False,
    )

    expected_training_dates = [
        datetime.date(2025, 1, 21),
        datetime.date(2025, 1, 22),
        datetime.date(2025, 1, 23),
        datetime.date(2025, 1, 24),
    ]
    torch.testing.assert_close(
        optimizer_calls[0]["returns"],
        torch.tensor(
            [[returns_by_date[market_date]] for market_date in expected_training_dates],
            dtype=torch.float32,
        ),
    )
    assert len(result.folds) == 1
    assert result.folds[0].training_start_date == datetime.date(2025, 1, 21)
    assert result.folds[0].training_end_date == datetime.date(2025, 1, 24)
    assert result.folds[0].simulation_start_date == datetime.date(2025, 1, 27)


def test_walk_forward_bootstrap_draws_fresh_indices_for_every_training_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Independent optimizer calls create a new bootstrap sample for each fold."""
    directory = _portfolio_directory(tmp_path, "walk_forward_bootstrap")
    _write_complete_market_strategy(
        directory / "strategy.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 31),
    )
    randint_calls: list[tuple[int, tuple[int, ...]]] = []
    original_randint = torch.randint

    def recording_randint(high, size, **kwargs):
        randint_calls.append((high, size))
        return original_randint(high, size, **kwargs)

    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(torch, "randint", recording_randint)

    result = portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_bootstrap",
        alpha=0.1,
        grid_increment=1.0,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        device="cpu",
        bootstrap_on=True,
        bootstrap_runs=2,
        bootstrap_length=3,
        show_plot=False,
    )

    assert randint_calls == [(4, (2, 3)), (5, (2, 3)), (4, (2, 3))]
    assert len(result.folds) == 3
    assert result.statistics.bootstrap_average_drawdown is None
    assert "ADD (bootstrap)" not in capsys.readouterr().out


def test_walk_forward_displays_equity_and_stepwise_allocation_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The report plots equity, then every strategy's allocation by simulation."""
    directory = _portfolio_directory(tmp_path, "walk_forward_plot")
    _write_complete_market_strategy(
        directory / "alpha.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 2, 19),
    )
    _write_complete_market_strategy(
        directory / "beta.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 2, 19),
        multiplier=2.0,
    )
    optimizer_calls: list[dict[str, object]] = []
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _sequential_optimizer([[0.25, 0.75], [0.8, 0.2]], optimizer_calls),
    )
    displayed_axes: list[matplotlib.axes.Axes] = []

    def record_displayed_axis() -> None:
        figure = plt.gcf()
        assert len(figure.axes) == 1
        displayed_axes.append(figure.axes[0])

    monkeypatch.setattr(plt, "show", record_displayed_axis)

    portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_plot",
        alpha=0.0,
        grid_increment=0.25,
        training_weeks=2,
        embargo_weeks=1,
        simulation_weeks=2,
        allocation_alpha=0.5,
        device="cpu",
        show_plot=True,
    )

    assert len(displayed_axes) == 2
    equity_axis, allocation_axis = displayed_axes
    assert equity_axis.get_yscale() == "log"

    allocation_lines = allocation_axis.get_lines()
    solid_lines = {
        line.get_label(): line
        for line in allocation_lines
        if line.get_linestyle() == "-"
    }
    dashed_lines = [line for line in allocation_lines if line.get_linestyle() == "--"]
    assert set(solid_lines) == {"alpha", "beta"}
    assert len(dashed_lines) == 2
    assert len({line.get_color() for line in solid_lines.values()}) == 2
    expected_dates = [
        datetime.date(2025, 1, 27),
        datetime.date(2025, 2, 10),
        datetime.date(2025, 2, 20),
    ]
    expected_calculated = {
        "alpha": [0.25, 0.8, 0.8],
        "beta": [0.75, 0.2, 0.2],
    }
    expected_applied = {
        "alpha": [0.25, 0.525, 0.525],
        "beta": [0.75, 0.475, 0.475],
    }
    for strategy_name, solid_line in solid_lines.items():
        calculated_line = next(
            line for line in dashed_lines if line.get_color() == solid_line.get_color()
        )
        for line in (solid_line, calculated_line):
            assert list(line.get_xdata()) == expected_dates
            assert line.get_drawstyle() == "steps-post"
        assert list(solid_line.get_ydata()) == pytest.approx(
            expected_applied[strategy_name]
        )
        assert list(calculated_line.get_ydata()) == pytest.approx(
            expected_calculated[strategy_name]
        )
        assert calculated_line.get_linewidth() < solid_line.get_linewidth()
        calculated_alpha = calculated_line.get_alpha()
        solid_alpha = solid_line.get_alpha()
        assert calculated_alpha is not None
        assert calculated_alpha < (1.0 if solid_alpha is None else solid_alpha)
    assert allocation_axis.get_yscale() == "linear"
    assert allocation_axis.get_ylabel() == "Allocation"
    assert "walk_forward_plot" in allocation_axis.get_title()
    legend = allocation_axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["alpha", "beta"]
    assert "%" in allocation_axis.yaxis.get_major_formatter()(0.25)
    plt.close("all")


def test_walk_forward_show_plot_false_displays_no_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Disabling plots suppresses both equity and allocation figures."""
    directory = _portfolio_directory(tmp_path, "walk_forward_no_plot")
    _write_complete_market_strategy(
        directory / "strategy.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 17),
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([1.0]),
    )
    displayed_figures: list[object] = []
    monkeypatch.setattr(plt, "show", lambda: displayed_figures.append(plt.gcf()))

    portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "walk_forward_no_plot",
        alpha=0.0,
        grid_increment=1.0,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        device="cpu",
        show_plot=False,
    )

    assert displayed_figures == []
    plt.close("all")


@pytest.mark.parametrize(
    ("parameter_name", "invalid_value", "error_type"),
    [
        ("training_weeks", True, TypeError),
        ("training_weeks", 1.0, TypeError),
        ("training_weeks", 0, ValueError),
        ("training_weeks", -1, ValueError),
        ("embargo_weeks", True, TypeError),
        ("embargo_weeks", 1.0, TypeError),
        ("embargo_weeks", -1, ValueError),
        ("simulation_weeks", True, TypeError),
        ("simulation_weeks", 1.0, TypeError),
        ("simulation_weeks", 0, ValueError),
        ("simulation_weeks", -1, ValueError),
    ],
)
def test_walk_forward_rejects_invalid_week_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameter_name: str,
    invalid_value: object,
    error_type: type[Exception],
):
    """Walk-forward durations are integer whole weeks with documented bounds."""
    directory = _portfolio_directory(tmp_path, "walk_forward_invalid_weeks")
    _write_complete_market_strategy(
        directory / "strategy.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 17),
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    parameters: dict[str, object] = {
        "training_weeks": 1,
        "embargo_weeks": 0,
        "simulation_weeks": 1,
    }
    parameters[parameter_name] = invalid_value

    with pytest.raises(error_type, match=parameter_name):
        portfolio_allocation.walk_forward_capital_allocation_from_csv(
            "walk_forward_invalid_weeks",
            alpha=0.0,
            grid_increment=1.0,
            device="cpu",
            show_plot=False,
            **parameters,
        )


@pytest.mark.parametrize(
    ("invalid_value", "error_type"),
    [
        (True, TypeError),
        ("0.5", TypeError),
        (None, TypeError),
        (-0.01, ValueError),
        (1.01, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
    ],
)
def test_walk_forward_rejects_invalid_allocation_alpha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: object,
    error_type: type[Exception],
):
    """Allocation smoothing is a finite real coefficient from zero through one."""
    directory = _portfolio_directory(tmp_path, "walk_forward_invalid_alpha")
    _write_complete_market_strategy(
        directory / "strategy.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 17),
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(error_type, match="allocation_alpha"):
        portfolio_allocation.walk_forward_capital_allocation_from_csv(
            "walk_forward_invalid_alpha",
            alpha=0.0,
            grid_increment=1.0,
            training_weeks=1,
            embargo_weeks=0,
            simulation_weeks=1,
            allocation_alpha=invalid_value,  # type: ignore[arg-type]
            device="cpu",
            show_plot=False,
        )


def test_walk_forward_wrappers_forward_allocation_alpha(
    monkeypatch: pytest.MonkeyPatch,
):
    """Both public lazy facades preserve the requested smoothing coefficient."""
    import ifera.allocation.walk_forward_allocation as walk_forward_allocation

    sentinel = object()
    internal_calls: list[dict[str, object]] = []
    facade_calls: list[dict[str, object]] = []

    def fake_internal(**kwargs: object):
        internal_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        walk_forward_allocation,
        "walk_forward_capital_allocation_from_csv",
        fake_internal,
    )
    result = portfolio_allocation.walk_forward_capital_allocation_from_csv(
        "forward_alpha",
        alpha=0.0,
        grid_increment=1.0,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        returns_dtype=pl.Float64,
        allocation_alpha=0.375,
        show_plot=False,
    )
    assert result is sentinel
    assert internal_calls[0]["allocation_alpha"] == 0.375
    assert internal_calls[0]["returns_dtype"] == pl.Float64

    def fake_facade(**kwargs: object):
        facade_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        portfolio_allocation,
        "walk_forward_capital_allocation_from_csv",
        fake_facade,
    )
    result = capital_allocation.walk_forward_capital_allocation_from_csv(
        "forward_alpha",
        alpha=0.0,
        grid_increment=1.0,
        training_weeks=1,
        embargo_weeks=0,
        simulation_weeks=1,
        returns_dtype=pl.Float64,
        allocation_alpha=0.625,
        show_plot=False,
    )
    assert result is sentinel
    assert facade_calls[0]["allocation_alpha"] == 0.625
    assert facade_calls[0]["returns_dtype"] == pl.Float64


def test_walk_forward_rejects_dataset_without_a_simulation_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """At least one session must remain after the initial training and embargo."""
    directory = _portfolio_directory(tmp_path, "walk_forward_no_simulation")
    _write_complete_market_strategy(
        directory / "strategy.csv",
        datetime.date(2025, 1, 6),
        datetime.date(2025, 1, 10),
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match="simulation|walk.forward|period"):
        portfolio_allocation.walk_forward_capital_allocation_from_csv(
            "walk_forward_no_simulation",
            alpha=0.0,
            grid_increment=1.0,
            training_weeks=1,
            embargo_weeks=0,
            simulation_weeks=1,
            device="cpu",
            show_plot=False,
        )


def test_direct_wrapper_accepts_case_insensitive_columns_and_optimizes_disjoint_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The direct public API preserves disjoint NaNs and accepts CSV header casing."""
    directory = _portfolio_directory(tmp_path, "direct_case")
    pl.DataFrame({"oPeNeD": ["Jan 13, 2025 10:00am"], "RoR": [10.0]}).write_csv(
        directory / "left.CSV"
    )
    pl.DataFrame({"OPENED": ["Jan 14, 2025 10:00am"], "ror": [20.0]}).write_csv(
        directory / "right.csv"
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        "direct_case",
        alpha=0.0,
        grid_increment=1.0,
        device="cpu",
        show_plot=False,
    )

    assert result.strategy_names == ("left", "right")
    assert torch.equal(result.allocation, torch.tensor([1.0, 1.0], dtype=torch.float32))
    assert result.daily_results["combined_return"].to_list() == pytest.approx(
        [0.1, 0.2]
    )


def test_zero_return_active_day_counts_but_zero_weight_strategy_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Activity follows observations in allocated strategies, not nonzero P/L."""
    directory = _portfolio_directory(tmp_path, "activity_case")
    _write_strategy(directory / "allocated.csv", ["Jan 13, 2025 10:00am"], [0.0])
    _write_strategy(directory / "unallocated.csv", ["Jan 14, 2025 10:00am"], [20.0])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        capital_allocation,
        "find_optimal_capital_allocation",
        _fixed_optimizer([1.0, 0.0]),
    )

    result = capital_allocation.find_optimal_capital_allocation_from_csv(
        "activity_case", 0.0, 1.0, device="cpu", show_plot=False
    )

    assert result.daily_results["active"].to_list() == [True, False]
    assert result.statistics.active_days == 1
    assert result.statistics.win_rate == 0.0
    assert result.statistics.exposure == pytest.approx(0.5)


def test_drawdown_reporting_remains_finite_when_equity_overflows():
    """Drawdown uses log equity so an extreme winning path still reports zero risk."""
    synchronized = pl.DataFrame(
        {
            "date": pl.date_range(
                datetime.date(2020, 1, 1),
                datetime.date(2023, 1, 4),
                interval="1d",
                eager=True,
            ),
            "strategy": [1.0] * 1100,
        }
    )

    daily_results = portfolio_allocation._combined_daily_results(
        synchronized, ("strategy",), [1.0]
    )

    assert daily_results["equity"].is_infinite().any()
    assert not daily_results["drawdown"].is_nan().any()
    assert daily_results["drawdown"].min() == 0.0


def test_wrapper_rejects_missing_portfolio_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A missing named portfolio produces an actionable error."""
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing"):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            "missing", 0.0, 1.0, device="cpu", show_plot=False
        )


def test_wrapper_rejects_portfolio_without_csv_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An existing but empty portfolio is not a valid optimizer input."""
    _portfolio_directory(tmp_path, "empty")
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match="CSV|csv"):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            "empty", 0.0, 1.0, device="cpu", show_plot=False
        )


@pytest.mark.parametrize("missing_column", ["Opened", "ROR"])
def test_wrapper_rejects_missing_required_csv_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_column: str,
):
    """Every strategy file must provide both synchronization and return fields."""
    directory = _portfolio_directory(tmp_path, f"missing_{missing_column}")
    data: dict[str, list[str] | list[float]] = {
        "Opened": ["Jan 6, 2025 10:00am"],
        "ROR": [1.0],
    }
    del data[missing_column]
    pl.DataFrame(data).write_csv(directory / "strategy.csv")
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match=missing_column):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            f"missing_{missing_column}",
            0.0,
            1.0,
            device="cpu",
            show_plot=False,
        )


@pytest.mark.parametrize(
    ("opened", "return_on_risk", "error_match"),
    [
        ("not a datetime", 1.0, "Opened|date|datetime|parse"),
        ("Jan 6, 2025 10:00am", math.nan, "ROR|finite|NaN|null"),
        ("Jan 6, 2025 10:00am", math.inf, "ROR|finite|infinite"),
    ],
)
def test_wrapper_rejects_invalid_strategy_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened: str,
    return_on_risk: float,
    error_match: str,
):
    """Invalid dates and nonfinite source returns are rejected before optimization."""
    directory = _portfolio_directory(tmp_path, "invalid_values")
    _write_strategy(directory / "strategy.csv", [opened], [return_on_risk])
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match=error_match):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            "invalid_values", 0.0, 1.0, device="cpu", show_plot=False
        )


def test_wrapper_rejects_empty_strategy_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A header-only strategy CSV cannot define an optimization period."""
    directory = _portfolio_directory(tmp_path, "empty_strategy")
    pl.DataFrame(
        {
            "Opened": pl.Series([], dtype=pl.String),
            "ROR": pl.Series([], dtype=pl.Float64),
        }
    ).write_csv(directory / "strategy.csv")
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match="empty|rows|data"):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            "empty_strategy", 0.0, 1.0, device="cpu", show_plot=False
        )


def test_wrapper_rejects_duplicate_strategy_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A strategy cannot supply two ambiguous returns for the same daily row."""
    directory = _portfolio_directory(tmp_path, "duplicate_dates")
    _write_strategy(
        directory / "strategy.csv",
        ["Jan 13, 2025 10:00am", "Jan 13, 2025 2:00pm"],
        [1.0, 2.0],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match="duplicate|dates"):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            "duplicate_dates", 0.0, 1.0, device="cpu", show_plot=False
        )


def test_wrapper_rejects_nyse_holiday_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Special NYSE closures are not silently treated as market sessions."""
    directory = _portfolio_directory(tmp_path, "closed_session")
    _write_strategy(
        directory / "strategy.csv",
        ["Jan 9, 2025 10:00am"],
        [1.0],
    )
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match="NYSE|market date"):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            "closed_session", 0.0, 1.0, device="cpu", show_plot=False
        )


@pytest.mark.parametrize("portfolio_name", ["", ".", "..", "../outside", "a/b"])
def test_wrapper_rejects_invalid_portfolio_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    portfolio_name: str,
):
    """Portfolio names cannot escape the configured portfolio root."""
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    with pytest.raises(ValueError, match="portfolio_name|portfolio name|name"):
        capital_allocation.find_optimal_capital_allocation_from_csv(
            portfolio_name, 0.0, 1.0, device="cpu", show_plot=False
        )
