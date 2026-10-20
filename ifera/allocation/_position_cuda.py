"""Prepare optional fused CUDA evaluators for constant-allocation position grids.

Only immutable schedules are retained between calls. Candidate state is private
to each evaluation, and the owning model controls prepared-evaluator lifetime.
The regular Torch ledger remains the CPU and unsupported-dtype implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ._position_model import PositionModel


def prepare_position_evaluator(
    model: PositionModel, device: torch.device, dtype: torch.dtype
) -> Callable[[torch.Tensor, float], torch.Tensor] | None:
    """Pack a model once, returning a CUDA evaluator or an explicit eager fallback."""
    if (
        device.type != "cuda"
        or dtype not in (torch.float32, torch.float64)
        or torch.version.hip is not None
        or find_spec("triton") is None
    ):
        return None
    # Importing this module on CPU must not require a Triton installation.
    # pylint: disable-next=import-outside-toplevel
    from ._position_kernel import (
        run_position_kernel,
    )

    buffers = _pack_schedule(model, device, dtype)
    day_count, trade_count = len(model.dates), len(model.trades)

    def evaluate(allocations: torch.Tensor, cap: float) -> torch.Tensor:
        returns = allocations.new_empty((allocations.shape[0], day_count))
        if not returns.numel():
            return returns
        # Transposing the logical trade ledger coalesces each candidate-lane load.
        risks = allocations.new_zeros((trade_count, allocations.shape[0]))
        with torch.cuda.device(allocations.device):
            run_position_kernel(allocations, returns, risks, buffers, cap)
        return returns

    return evaluate


def _pack_schedule(
    model: PositionModel, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    """Flatten timestamp events, active blockers and daily marks into reusable buffers."""
    # This private backend deliberately shares the model's validated event schedule.
    # pylint: disable=protected-access
    day_offsets = [0]
    event_trades: list[int] = []
    event_openings: list[int] = []
    for events in model._events:
        for is_opening, index in events:
            event_trades.append(index)
            event_openings.append(int(is_opening))
            trade = model.trades[index]
            if is_opening and trade.opened == trade.closed:
                event_trades.append(index)
                event_openings.append(0)
        day_offsets.append(len(event_trades))
    blocker_offsets = [0]
    blockers: list[int] = []
    for timing in model._timings:
        blockers.extend(timing.active_blockers)
        blocker_offsets.append(len(blockers))
    mark_offsets = [0]
    mark_trades: list[int] = []
    mark_values: list[float] = []
    for indices, marks in model._open_marks:
        mark_trades.extend(indices)
        mark_values.extend(marks)
        mark_offsets.append(len(mark_trades))
    integer_buffers = (
        day_offsets,
        event_trades,
        event_openings,
        [trade.strategy_index for trade in model.trades],
        blocker_offsets,
        blockers,
        mark_offsets,
        mark_trades,
    )
    floating_buffers = (
        [trade.return_on_risk for trade in model.trades],
        mark_values,
    )
    return tuple(
        torch.tensor(values, device=device, dtype=torch.int64)
        for values in integer_buffers
    ) + tuple(
        torch.tensor(values, device=device, dtype=dtype) for values in floating_buffers
    )
