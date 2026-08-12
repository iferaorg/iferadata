"""Tests for loading, reporting, and plotting portfolio allocations."""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import matplotlib
import polars as pl
import pytest
import torch

import ifera.capital_allocation as capital_allocation
import ifera.portfolio_allocation as portfolio_allocation
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
    assert torch.equal(result.allocation, torch.tensor([1.0, 1.0], dtype=torch.float64))
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
