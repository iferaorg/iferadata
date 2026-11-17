"""Statistics reuse allocation conventions and preserve original realized P/L."""

from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from ifera.strategy_optimization._reporting import (
    plot_objective_history,
    print_strategy_statistics,
    strategy_statistics,
)


def test_strategy_statistics_include_nontrading_days_and_percent_conversion(capsys):
    dates = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    frame, statistics = strategy_statistics(
        dates,
        torch.tensor([10.0, 0.0, -5.0], dtype=torch.float64),
        torch.tensor([True, False, True]),
    )
    assert frame["combined_return"].to_list() == pytest.approx([0.1, 0.0, -0.05])
    assert frame["equity"].to_list() == pytest.approx([1.1, 1.1, 1.045])
    assert statistics.market_days == 3
    assert statistics.active_days == 2
    assert statistics.win_rate == 0.5
    assert statistics.average_daily_return == pytest.approx(0.05 / 3.0)
    print_strategy_statistics(statistics)
    output = capsys.readouterr().out
    assert "ADD:" in output
    assert "Active days: 2/3" in output


def test_strategy_statistics_empty_winning_strategy():
    frame, statistics = strategy_statistics(
        (date(2024, 1, 2),), torch.zeros(1), torch.tensor([False])
    )
    assert frame["equity"].item() == 1.0
    assert statistics.active_days == 0
    assert statistics.win_rate == 0.0
    assert statistics.average_drawdown == 0.0
    assert statistics.annualized_volatility == 0.0


def test_statistics_total_loss_beyond_original_risk_preserves_arithmetic():
    frame, statistics = strategy_statistics(
        (date(2024, 1, 2), date(2024, 1, 3)),
        torch.tensor([-125.0, 25.0], dtype=torch.float64),
        torch.tensor([True, True]),
        annualization_periods=2,
    )
    assert frame["equity"].to_list() == [0.0, 0.0]
    assert frame["combined_return"].to_list() == [-1.25, 0.25]
    assert statistics.ending_equity == 0.0
    assert statistics.cagr == -1.0
    assert statistics.average_drawdown == -1.0
    assert statistics.max_drawdown == -1.0
    assert statistics.worst_day == -1.25
    assert statistics.average_daily_return == -0.5
    assert statistics.profit_factor == pytest.approx(0.2)
    assert statistics.annualized_volatility == pytest.approx(1.5)
    assert statistics.sortino_ratio == pytest.approx(-0.8)


@pytest.mark.parametrize(
    "dates,returns,active,error",
    [
        ((), torch.zeros(0), torch.zeros(0, dtype=torch.bool), ValueError),
        (
            (date(2024, 1, 2),),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            ValueError,
        ),
        ((date(2024, 1, 2),), torch.zeros(1), torch.zeros(1), TypeError),
        ((date(2024, 1, 2),), torch.zeros(1, 1), torch.tensor([False]), ValueError),
        (
            (date(2024, 1, 3), date(2024, 1, 2)),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            ValueError,
        ),
        (
            (date(2024, 1, 2), date(2024, 1, 2)),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            ValueError,
        ),
    ],
)
def test_invalid_statistics_input(dates, returns, active, error):
    with pytest.raises(error):
        strategy_statistics(dates, returns, active)


def test_objective_history_plot_has_generation_and_value_axes():
    figure = plot_objective_history((1.0, 1.25, 1.5))
    axes = figure.axes[0]
    assert list(axes.lines[0].get_xdata()) == [0, 1, 2]
    assert list(axes.lines[0].get_ydata()) == [1.0, 1.25, 1.5]
    assert axes.get_xlabel() == "Generation"
    assert axes.get_ylabel() == "Best objective value"
    plt.close(figure)


@pytest.mark.parametrize("history", [(), (0.0,), (-1.0,), (float("nan"),)])
def test_invalid_objective_history(history):
    with pytest.raises(ValueError):
        plot_objective_history(history)
