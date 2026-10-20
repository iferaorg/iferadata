"""Differential checks for the fused CUDA position ledger."""

from datetime import date, datetime, timedelta

import pytest
import torch

from ifera.allocation._position_cuda import prepare_position_evaluator
from ifera.allocation import _position_cuda
from ifera.allocation._position_model import PositionModel, PositionTrade


def _model(method="linear"):
    dates = tuple(date(2026, 10, 5) + timedelta(days=day) for day in range(5))

    def trade(strategy, opened, closed, result, blockers=(), opening_hour=10):
        return PositionTrade(
            strategy,
            datetime.combine(dates[opened], datetime.min.time()).replace(
                hour=opening_hour
            ),
            datetime.combine(dates[closed], datetime.min.time()).replace(hour=16),
            result,
            blockers,
        )

    trades = (
        trade(0, 0, 3, 0.6),
        trade(1, 0, 2, -0.3, (0,), 11),
        trade(0, 1, 4, -0.2),
        trade(2, 2, 2, 0.4, (1,)),
        PositionTrade(1, datetime(2026, 10, 8, 16), datetime(2026, 10, 8, 16), 0.1),
        trade(2, 4, 4, -1.0),
    )
    return PositionModel(dates, ("A", "B", "C"), trades, method)


def test_unsupported_devices_and_dtypes_return_no_backend():
    model = _model()
    assert prepare_position_evaluator(model, torch.device("cpu"), torch.float32) is None
    assert (
        prepare_position_evaluator(model, torch.device("cuda"), torch.float16) is None
    )


def test_missing_triton_returns_no_backend(monkeypatch):
    monkeypatch.setattr(_position_cuda, "find_spec", lambda _name: None)
    assert (
        prepare_position_evaluator(_model(), torch.device("cuda"), torch.float32)
        is None
    )


def test_hip_returns_no_backend(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test_hip")
    assert (
        prepare_position_evaluator(_model(), torch.device("cuda"), torch.float32)
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["linear", "close_only"])
@pytest.mark.parametrize("cap", [0.0, 0.8, 2.0])
def test_fused_ledger_matches_eager_with_blocks_overlap_and_noncontiguous_input(
    dtype, method, cap
):
    model = _model(method)
    generator = torch.Generator().manual_seed(42)
    allocations = torch.rand((3, 131), generator=generator, dtype=dtype).cuda().t()
    allocations[0] = 0
    allocations[1] = torch.tensor([0.0, 0.5, 0.3], device="cuda", dtype=dtype)
    evaluate = prepare_position_evaluator(model, allocations.device, dtype)
    assert evaluate is not None

    actual = evaluate(allocations, cap)
    expected = model._run_ledger(allocations.unsqueeze(1), cap, False).daily_returns

    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.shape == (131, 5)
    assert actual.dtype == dtype
    torch.testing.assert_close(evaluate(allocations[:1], cap), expected[:1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bankruptcy_and_infinite_equity_keep_eager_sentinel(dtype):
    dates = tuple(date(2026, 10, 5) + timedelta(days=day) for day in range(3))
    trades = (
        PositionTrade(0, datetime(2026, 10, 5, 10), datetime(2026, 10, 5, 16), -1),
        PositionTrade(1, datetime(2026, 10, 6, 10), datetime(2026, 10, 6, 16), 1e308),
        PositionTrade(0, datetime(2026, 10, 7, 10), datetime(2026, 10, 7, 16), 0.5),
    )
    model = PositionModel(dates, ("A", "B"), trades)
    allocations = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda", dtype=dtype)
    evaluate = prepare_position_evaluator(model, allocations.device, dtype)
    assert evaluate is not None

    actual = evaluate(allocations, 1.0)
    expected = model._run_ledger(allocations.unsqueeze(1), 1.0, False).daily_returns

    torch.testing.assert_close(actual, expected)
    assert actual[0].tolist() == [-1.0, -1.0, -1.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("variant", ["empty_trades", "empty_days", "truncated"])
def test_empty_and_truncated_models(variant):
    original = _model()
    if variant == "empty_trades":
        model = PositionModel(original.dates, original.strategy_names, ())
    elif variant == "empty_days":
        model = PositionModel((), original.strategy_names, ())
    else:
        model = original.window(date(2026, 10, 5), date(2026, 10, 8), closed_only=False)
    allocations = torch.tensor([[0.25, 0.2, 0.5]], device="cuda")
    evaluate = prepare_position_evaluator(model, allocations.device, allocations.dtype)
    assert evaluate is not None

    actual = evaluate(allocations, 1.0)
    expected = model._run_ledger(allocations.unsqueeze(1), 1.0, False).daily_returns

    torch.testing.assert_close(actual, expected)
    assert evaluate(allocations[:0], 1.0).shape == (0, len(model.dates))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_many_mixed_sign_marks_and_near_capacity_admission(dtype):
    dates = tuple(date(2026, 10, 5) + timedelta(days=day) for day in range(5))
    trade_returns = (0.9, -0.7, 0.4, -0.3, 0.6, -0.5, 0.2, -0.1)
    trades = tuple(
        PositionTrade(
            index,
            datetime(2026, 10, 5, 10, index),
            datetime(2026, 10, 9, 16),
            result,
        )
        for index, result in enumerate(trade_returns)
    ) + (PositionTrade(0, datetime(2026, 10, 6, 12), datetime(2026, 10, 7, 16), 0.2),)
    model = PositionModel(dates, tuple(str(index) for index in range(8)), trades)
    allocations = torch.full((33, 8), 0.1125, device="cuda", dtype=dtype)
    allocations[:, 0] = 0.1111411248298617 + torch.linspace(
        -1e-5, 1e-5, 33, device="cuda", dtype=dtype
    )
    evaluate = prepare_position_evaluator(model, allocations.device, dtype)
    assert evaluate is not None

    actual = evaluate(allocations, 1.0)
    ledger = model._run_ledger(allocations.unsqueeze(1), 1.0, False)
    expected = ledger.daily_returns
    assert (ledger.risks[:, -1] > 0).any()
    assert (ledger.risks[:, -1] == 0).any()

    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
