"""Daily marked equity and entry sizing for multi-day portfolio positions."""

from datetime import date, datetime, timedelta

import pytest
import torch

from ifera.allocation._position_model import PositionModel, PositionTrade

DATES = (date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7))


def _trade(strategy, opening_day, closing_day, ror, blockers=(), hour=10):
    return PositionTrade(
        strategy,
        datetime.combine(DATES[opening_day], datetime.min.time()).replace(hour=hour),
        datetime.combine(DATES[closing_day], datetime.min.time()).replace(hour=16),
        ror,
        blockers,
    )


def test_linear_marks_keep_original_trade_size_and_exact_terminal_result():
    model = PositionModel(DATES, ("A",), (_trade(0, 0, 2, 0.6),))
    allocation = torch.tensor([[0.5]], dtype=torch.float64)

    returns = model.evaluate(allocation, 1.0)

    torch.testing.assert_close(
        returns, torch.tensor([[0.1, 0.1 / 1.1, 0.1 / 1.2]], dtype=torch.float64)
    )
    assert (1 + returns).prod().item() == pytest.approx(1.3)


def test_self_overlap_sizes_new_trade_from_marked_start_of_day_equity():
    model = PositionModel(DATES, ("A",), (_trade(0, 0, 2, 0.6), _trade(0, 1, 2, -0.2)))

    simulation = model.simulate(torch.full((3, 1), 0.25, dtype=torch.float64), 1.0)

    assert simulation.trade_results["risk"].to_list() == pytest.approx([0.25, 0.2625])
    assert simulation.trade_results["entry_equity"].to_list() == pytest.approx(
        [1, 1.05]
    )
    assert (1 + simulation.daily_returns).prod().item() == pytest.approx(1.0975)
    assert simulation.active.tolist() == [True, True, True]


def test_entry_budget_skips_full_overlapping_trade_and_reports_reason():
    model = PositionModel(DATES, ("A",), (_trade(0, 0, 2, 0.0), _trade(0, 1, 2, 0.8)))

    simulation = model.simulate(torch.full((3, 1), 0.6), 1.0)

    assert simulation.trade_results["status"].to_list() == ["executed", "capital"]
    assert simulation.trade_results["risk"].to_list() == pytest.approx([0.6, 0])
    torch.testing.assert_close(simulation.daily_returns, torch.zeros(3))


def test_closed_trade_releases_budget_before_later_entry_on_same_day():
    first = PositionTrade(0, datetime(2026, 10, 5, 10), datetime(2026, 10, 6, 11), 0.2)
    second = _trade(0, 1, 2, 0.2, hour=12)
    model = PositionModel(DATES, ("A",), (first, second))

    simulation = model.simulate(torch.full((3, 1), 0.8, dtype=torch.float64), 1.0)

    assert simulation.trade_results["status"].to_list() == ["executed", "executed"]
    assert simulation.trade_results["risk"].to_list() == pytest.approx([0.8, 0.864])


def test_only_executed_still_open_trades_block_later_entries():
    model = PositionModel(
        DATES,
        ("A", "B", "C"),
        (_trade(0, 0, 2, 0), _trade(1, 1, 2, 0, (0,)), _trade(2, 2, 2, 0.4, (1,))),
    )
    allocations = torch.tensor([[0.2, 0.2, 0.2], [0, 0.2, 0.2]])

    returns = model.evaluate(allocations, 1.0)

    torch.testing.assert_close(returns, torch.tensor([[0, 0, 0.08], [0, 0, 0]]))
    simulation = model.simulate(allocations[0].expand(3, -1), 1.0)
    assert simulation.trade_results["status"].to_list() == [
        "executed",
        "blocked",
        "executed",
    ]


def test_zero_allocation_at_next_fold_does_not_resize_or_close_existing_trade():
    model = PositionModel(DATES, ("A",), (_trade(0, 0, 2, 0.6), _trade(0, 1, 2, 0.2)))

    simulation = model.simulate(torch.tensor([[0.5], [0.0], [0.0]]), 1.0)

    assert (1 + simulation.daily_returns).prod().item() == pytest.approx(1.3)
    assert simulation.trade_results["status"].to_list() == [
        "executed",
        "zero_allocation",
    ]


def test_close_only_marks_and_total_loss_are_supported():
    model = PositionModel(
        DATES, ("A",), (_trade(0, 0, 2, -1.0),), valuation_method="close_only"
    )

    returns = model.evaluate(torch.tensor([[0.5], [1.0]]), 1.0)

    torch.testing.assert_close(returns, torch.tensor([[0, 0, -0.5], [0, 0, -1.0]]))
    with pytest.raises(ValueError, match="equity|insolvent"):
        model.simulate(torch.ones(3, 1), 1.0)


def test_zero_duration_trades_close_immediately_without_consuming_later_budget():
    timestamp = datetime(2026, 10, 5, 10)
    model = PositionModel(
        DATES,
        ("A", "B"),
        (
            PositionTrade(0, timestamp, timestamp, 0.2),
            PositionTrade(1, timestamp, timestamp, 0.1, (0,)),
        ),
    )

    simulation = model.simulate(torch.ones(3, 2), 1.0)

    assert simulation.trade_results["status"].to_list() == ["executed", "executed"]
    assert simulation.daily_returns[0].item() == pytest.approx(0.3)


def test_returns_proxy_sums_self_overlap_and_keeps_inactive_nan():
    model = PositionModel(
        DATES,
        ("A", "B"),
        (_trade(0, 0, 2, 0.6), _trade(0, 1, 2, -0.2), _trade(1, 2, 2, 0)),
    )

    proxy = model.returns_tensor(dtype=torch.float64)

    torch.testing.assert_close(
        proxy,
        torch.tensor(
            [[0.2, float("nan")], [0.1, float("nan")], [0.1, 0]], dtype=torch.float64
        ),
        equal_nan=True,
    )


def test_training_window_purges_unclosed_trades_before_building_marks():
    model = PositionModel(
        DATES,
        ("A", "B"),
        (_trade(0, 0, 2, 3.0), _trade(1, 1, 1, 0.2, (0,))),
    )

    window = model.window(DATES[0], DATES[2])

    assert window.dates == DATES[:2]
    assert len(window.trades) == 1
    assert window.trades[0].blockers == ()
    torch.testing.assert_close(
        window.returns_tensor(),
        torch.tensor([[float("nan"), float("nan")], [float("nan"), 0.2]]),
        equal_nan=True,
    )


def test_truncated_simulation_window_preserves_original_mark_denominator():
    model = PositionModel(DATES, ("A",), (_trade(0, 0, 2, 0.6),))

    window = model.window(DATES[0], DATES[2], closed_only=False)
    simulation = window.simulate(torch.full((2, 1), 0.5), 1.0)

    assert (1 + simulation.daily_returns).prod().item() == pytest.approx(1.2)
    assert simulation.trade_results["pnl"].item() == pytest.approx(0.2)


def test_subset_remaps_strategies_and_blocker_trade_identities():
    model = PositionModel(
        DATES,
        ("A", "B", "C"),
        (
            _trade(0, 0, 2, 0.1),
            _trade(1, 1, 2, 0.1, (0,)),
            _trade(2, 2, 2, 0.2, (0, 1)),
        ),
    )

    subset = model.subset((1, 2))

    assert subset.strategy_names == ("B", "C")
    assert [trade.strategy_index for trade in subset.trades] == [0, 1]
    assert [trade.blockers for trade in subset.trades] == [(), (0,)]


@pytest.mark.parametrize("method", ["geometric", "", "Linear"])
def test_invalid_mark_method_is_rejected(method):
    with pytest.raises(ValueError, match="valuation_method"):
        PositionModel(DATES, ("A",), (), valuation_method=method)


@pytest.mark.parametrize("allocation", [float("nan"), float("inf"), -0.1])
def test_invalid_allocations_are_rejected(allocation):
    model = PositionModel(DATES, ("A",), ())
    with pytest.raises(ValueError, match="finite|nonnegative"):
        model.evaluate(torch.tensor([[allocation]]), 1.0)


def test_empty_trade_model_and_memory_batch_budget():
    model = PositionModel(DATES, ("A",), ())
    simulation = model.simulate(torch.zeros(3, 1), 1.0)
    torch.testing.assert_close(simulation.daily_returns, torch.zeros(3))
    assert simulation.active.tolist() == [False, False, False]
    assert simulation.trade_results.is_empty()
    assert model.candidate_batch_size(memory_budget_bytes=1) == 1
    assert model.candidate_batch_size(memory_budget_bytes=4096) > 1


def test_zero_entry_budget_is_a_valid_all_cash_simulation():
    model = PositionModel(DATES, ("A",), (_trade(0, 0, 2, 0.6),))
    simulation = model.simulate(torch.zeros(3, 1), 0.0)
    torch.testing.assert_close(simulation.daily_returns, torch.zeros(3))
    assert simulation.trade_results["status"].to_list() == ["zero_allocation"]


def test_overflowing_daily_return_is_invalid_even_when_equity_stays_finite():
    dates = tuple(DATES[0] + timedelta(days=index) for index in range(32))
    trades = [
        PositionTrade(
            0,
            datetime.combine(dates[0], datetime.min.time()),
            datetime.combine(dates[-1], datetime.min.time()).replace(hour=16),
            1e30,
        )
    ]
    trades.extend(
        PositionTrade(
            1,
            datetime.combine(day, datetime.min.time()).replace(hour=10),
            datetime.combine(day, datetime.min.time()).replace(hour=16),
            -1.0,
        )
        for day in dates[1:]
    )
    model = PositionModel(
        dates, ("A", "B"), tuple(trades), valuation_method="close_only"
    )
    allocations = torch.tensor([0.1, 0.9]).expand(len(dates), -1)
    with pytest.raises(ValueError, match="nonfinite|equity"):
        model.simulate(allocations, 1e35)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("holding_days", [5, 7, 12])
@pytest.mark.parametrize("other_open_position", [False, True])
def test_full_terminal_loss_is_insolvent_despite_daily_mark_roundoff(
    dtype, holding_days, other_open_position
):
    dates = tuple(DATES[0] + timedelta(days=index) for index in range(holding_days + 1))
    first = PositionTrade(
        0,
        datetime.combine(dates[0], datetime.min.time()),
        datetime.combine(dates[-2], datetime.min.time()).replace(hour=16),
        -1.0,
    )
    trades = [first]
    if other_open_position:
        trades.append(
            PositionTrade(
                1,
                first.opened.replace(hour=10),
                datetime.combine(dates[-1], datetime.min.time()),
                0.0,
            )
        )
    model = PositionModel(dates, ("A", "B"), tuple(trades))
    allocation = torch.tensor([[1.0, 0.1]], dtype=dtype)

    returns = model.evaluate(allocation, 1.1)

    assert returns[0, holding_days - 1].item() <= -1.0
    with pytest.raises(ValueError, match="equity"):
        model.simulate(allocation.expand(len(dates), -1), 1.1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_positive_terminal_equity_remains_valid(dtype):
    dates = tuple(DATES[0] + timedelta(days=index) for index in range(7))
    model = PositionModel(
        dates,
        ("A",),
        (
            PositionTrade(
                0,
                datetime.combine(dates[0], datetime.min.time()),
                datetime.combine(dates[-1], datetime.min.time()),
                -1.0,
            ),
        ),
    )
    allocation = torch.tensor([[1.0 - 1e-7]], dtype=dtype)

    simulation = model.simulate(allocation.expand(len(dates), -1), 1.0)

    assert torch.all(simulation.daily_returns > -1)
    assert (1 + simulation.daily_returns).prod().item() == pytest.approx(
        1 - allocation.item(), rel=0.1, abs=1e-12
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["linear", "close_only"])
def test_cuda_candidate_and_simulation_paths_match_cpu(dtype, method):
    model = PositionModel(
        DATES,
        ("A", "B", "C"),
        (
            _trade(0, 0, 2, 0.3),
            _trade(0, 1, 2, -0.2),
            _trade(1, 1, 2, 0.4, (0,)),
            _trade(2, 1, 1, 0.2),
            _trade(1, 2, 2, 0.3, (0,)),
        ),
        valuation_method=method,
    )
    allocations = torch.tensor(
        [[0.6, 0.2, 0.3], [0.25, 0.2, 0.75], [0, 0.4, 0.7], [0.2, 0, 0.1]],
        dtype=dtype,
    )
    expected = model.evaluate(allocations, 1.0)

    actual = model.evaluate(allocations.cuda(), 1.0)
    simulation = model.simulate(allocations[0].expand(len(DATES), -1).cuda(), 1.0)
    cpu_simulation = model.simulate(allocations[0].expand(len(DATES), -1), 1.0)

    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(simulation.daily_returns.cpu(), actual[0].cpu())
    torch.testing.assert_close(simulation.active.cpu(), cpu_simulation.active)
    assert (
        simulation.trade_results["status"].to_list()
        == cpu_simulation.trade_results["status"].to_list()
    )
    assert {"executed", "capital", "blocked"} <= set(
        simulation.trade_results["status"].to_list()
    )
    for column in ("entry_equity", "risk", "pnl"):
        assert simulation.trade_results[column].to_list() == pytest.approx(
            cpu_simulation.trade_results[column].to_list(), rel=1e-6, abs=1e-8
        )
