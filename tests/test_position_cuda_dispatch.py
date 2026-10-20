"""CUDA evaluator caching stays local to immutable position-model instances."""

from datetime import date, datetime
from types import SimpleNamespace
import sys

import pytest
import torch

from ifera.allocation._position_model import PositionModel, PositionTrade


def _model():
    return PositionModel(
        (date(2025, 1, 6), date(2025, 1, 7)),
        ("A",),
        (PositionTrade(0, datetime(2025, 1, 6, 10), datetime(2025, 1, 7, 16), 0.2),),
    )


def test_cpu_position_evaluator_does_not_prepare_cuda_backend(monkeypatch):
    def prepare(*_args):
        raise AssertionError("CPU evaluation must not prepare a GPU backend")

    monkeypatch.setitem(
        sys.modules,
        "ifera.allocation._position_cuda",
        SimpleNamespace(prepare_position_evaluator=prepare),
    )
    result = _model().evaluate(torch.tensor([[0.5]]), 1.0)
    assert (1 + result).prod().item() == pytest.approx(1.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("available", [True, False])
def test_cuda_evaluator_prepares_once_per_model_and_dtype(monkeypatch, available):
    prepared = []
    evaluated = []

    def prepare(model, device, dtype):
        prepared.append((id(model), device, dtype))

        def evaluate(allocations, cap):
            evaluated.append((allocations.shape[0], cap))
            return allocations.new_zeros((allocations.shape[0], len(model.dates)))

        return evaluate if available else None

    monkeypatch.setitem(
        sys.modules,
        "ifera.allocation._position_cuda",
        SimpleNamespace(prepare_position_evaluator=prepare),
    )
    model = _model()
    for size, cap in ((1, 1.0), (3, 0.8)):
        result = model.evaluate(torch.full((size, 1), 0.5, device="cuda"), cap)
        if available:
            assert torch.count_nonzero(result).item() == 0
        else:
            assert (1 + result[0]).prod().item() == pytest.approx(1.1)
    assert len(prepared) == 1
    model.evaluate(torch.zeros((1, 1), dtype=torch.float64, device="cuda"), 1.0)
    assert len(prepared) == 2
    model.window(date(2025, 1, 6), date(2025, 1, 8)).evaluate(
        torch.zeros((1, 1), device="cuda"), 1.0
    )
    assert len(prepared) == 3
    assert len(evaluated) == (4 if available else 0)
