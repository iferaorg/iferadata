"""Drawdown-objective forwarding and date-aware growth floors in CSV wrappers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import torch

from ifera.allocation import capital_allocation, portfolio_allocation
from ifera.settings import settings


def _write_portfolio(tmp_path: Path, start: date, stop: date) -> None:
    """Create two synchronized strategies with uniquely identifiable daily returns."""
    directory = tmp_path / "results" / "portfolios" / "objective"
    directory.mkdir(parents=True)
    dates = portfolio_allocation._nyse_market_dates(start, stop)["date"].to_list()
    for index, name in enumerate(("alpha", "beta"), start=1):
        pl.DataFrame(
            {
                "Opened": [f"{day.isoformat()} 10:00:00" for day in dates],
                "ROR": [
                    index * (day_index + 1) / 10 for day_index in range(len(dates))
                ],
            }
        ).write_csv(directory / f"{name}.csv")


@pytest.mark.parametrize("use_facade", [False, True])
def test_csv_forwards_drawdown_objective_with_inclusive_calendar_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_facade: bool
):
    """A CSV growth floor uses observed dates, including intervening nonmarket days."""
    _write_portfolio(tmp_path, date(2025, 1, 2), date(2025, 1, 10))
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    captured: dict[str, object] = {}

    def optimizer(returns: torch.Tensor, **kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return torch.tensor([0.5, 0.0], dtype=returns.dtype)

    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )
    wrapper = (
        capital_allocation.find_optimal_capital_allocation_from_csv
        if use_facade
        else portfolio_allocation.find_optimal_capital_allocation_from_csv
    )
    wrapper(
        "objective",
        alpha=0.0,
        grid_increment=0.1,
        objective="minimize_add",
        CMGR_limit=0.03,
        device="cpu",
        show_plot=False,
    )

    assert captured["objective"] == "minimize_add"
    assert captured["CMGR_limit"] == 0.03
    assert captured["calendar_days"] == 9


@pytest.mark.parametrize("selection", [False, True])
def test_walk_forward_growth_floor_uses_each_training_and_regime_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: bool
):
    """Long first folds, holidays, and background support solves retain local dates."""
    _write_portfolio(tmp_path, date(2025, 1, 6), date(2025, 2, 7))
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        portfolio_allocation, "tqdm", lambda iterable, **_kwargs: iterable
    )
    calls: list[tuple[torch.Tensor, dict[str, object]]] = []
    discovery_count = 0

    def optimizer(returns: torch.Tensor, **kwargs: object) -> torch.Tensor:
        nonlocal discovery_count
        calls.append((returns.clone(), kwargs))
        if returns.shape[1] == 2:
            allocation = [0.5, 0.0] if discovery_count % 2 == 0 else [0.0, 0.5]
            discovery_count += 1
        else:
            allocation = [0.5]
        return torch.tensor(allocation, dtype=returns.dtype)

    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )
    result = capital_allocation.walk_forward_capital_allocation_from_csv(
        "objective",
        alpha=0.0,
        grid_increment=0.1,
        training_weeks=1,
        first_training_weeks=2,
        embargo_weeks=0,
        simulation_weeks=1,
        objective="minimize_add",
        CMGR_limit=0.03,
        allocation_alpha=0.5,
        max_strategies=1 if selection else None,
        device="cpu",
        show_plot=False,
    )

    expected_days = [12, 4, 5]
    assert [
        (fold.training_end_date - fold.training_start_date).days + 1
        for fold in result.folds
    ] == expected_days
    assert [kwargs["calendar_days"] for _, kwargs in calls] == expected_days * (
        2 if selection else 1
    )
    assert all(kwargs["objective"] == "minimize_add" for _, kwargs in calls)
    assert all(kwargs["CMGR_limit"] == 0.03 for _, kwargs in calls)
    if selection:
        for fold_index in range(3):
            original = calls[fold_index][0]
            background = calls[fold_index + 3][0]
            column = 1 if fold_index % 2 == 0 else 0
            torch.testing.assert_close(background, original[:, column : column + 1])


def test_walk_forward_reports_infeasible_background_support_with_fold_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A regime unable to meet a growth floor fails explicitly with local dates."""
    _write_portfolio(tmp_path, date(2025, 1, 6), date(2025, 2, 7))
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        portfolio_allocation, "tqdm", lambda iterable, **_kwargs: iterable
    )
    discovery_count = 0

    def optimizer(returns: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        nonlocal discovery_count
        if returns.shape[1] == 1:
            raise ValueError("No allocation satisfies the CMGR_limit growth floor")
        allocation = [0.5, 0.0] if discovery_count % 2 == 0 else [0.0, 0.5]
        discovery_count += 1
        return torch.tensor(allocation, dtype=returns.dtype)

    monkeypatch.setattr(
        capital_allocation, "find_optimal_capital_allocation", optimizer
    )
    with pytest.raises(
        ValueError,
        match=r"background support \(1,\).*2025-01-06.*2025-01-17.*CMGR_limit",
    ):
        capital_allocation.walk_forward_capital_allocation_from_csv(
            "objective",
            alpha=0.0,
            grid_increment=0.1,
            training_weeks=1,
            first_training_weeks=2,
            embargo_weeks=0,
            simulation_weeks=1,
            objective="minimize_add",
            CMGR_limit=0.03,
            allocation_alpha=0.5,
            max_strategies=1,
            device="cpu",
            show_plot=False,
        )


@pytest.mark.parametrize("returns_dtype", [pl.Float32, pl.Float64])
def test_csv_real_optimizer_meets_reported_cmgr_with_less_drawdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returns_dtype
):
    """The actual risk optimizer and report agree on a 30-calendar-day growth floor."""
    directory = tmp_path / "results" / "portfolios" / "risk_target"
    directory.mkdir(parents=True)
    pl.DataFrame(
        {
            "Opened": [
                "2025-01-02 10:00:00",
                "2025-01-10 10:00:00",
                "2025-01-21 10:00:00",
                "2025-01-31 10:00:00",
            ],
            "ROR": [20.0, -10.0, 20.0, -10.0],
        }
    ).write_csv(directory / "strategy.csv")
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    common = dict(
        portfolio_name="risk_target",
        alpha=0.0,
        grid_increment=0.1,
        device="cpu",
        show_plot=False,
        returns_dtype=returns_dtype,
    )
    risk_result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        **common, objective="minimize_add", CMGR_limit=0.025
    )
    growth_result = portfolio_allocation.find_optimal_capital_allocation_from_csv(
        **common
    )

    assert risk_result.statistics.cmgr >= 0.025
    assert float(risk_result.allocation[0]) == pytest.approx(0.2)
    assert float(risk_result.allocation[0]) < float(growth_result.allocation[0])
    assert (
        risk_result.statistics.average_drawdown
        > growth_result.statistics.average_drawdown
    )
    daily_log_growth = float(risk_result.daily_results["log_equity"].item(-1))
    target_log_growth = torch.tensor(0.025, dtype=torch.float64).log1p().item() * (
        30 / 365.25 * 12
    )
    assert daily_log_growth >= target_log_growth
