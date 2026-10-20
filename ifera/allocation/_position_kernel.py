"""One-launch CUDA position ledger; imported only for supported CUDA evaluations.

Each lane follows one candidate's sequential account, while all lanes share the
event schedule. Runtime loops avoid compilation proportional to trade count.
FMA is disabled and divisions are correctly rounded to retain ledger boundaries.
"""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _position_kernel(  # pylint: disable=too-many-statements
    allocations,
    returns,
    risks,
    day_offsets,
    event_trades,
    event_openings,
    strategies,
    blocker_offsets,
    blockers,
    mark_offsets,
    mark_trades,
    trade_returns,
    mark_values,
    batch,
    days,
    row_stride,
    strategy_stride,
    cap: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    """Apply timestamp events and daily marks without host/device round trips."""
    lane = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = lane < batch
    value_dtype: tl.constexpr = allocations.dtype.element_ty
    equity = tl.full((block_size,), 1, value_dtype)
    realized = tl.full((block_size,), 1, value_dtype)
    reserved = tl.full((block_size,), 0, value_dtype)
    open_count = tl.full((block_size,), 0, tl.int32)
    alive = tl.full((block_size,), True, tl.int1)
    for day in range(days):
        event_start = tl.load(day_offsets + day)
        event_stop = tl.load(day_offsets + day + 1)
        for event in range(event_start, event_stop):
            index = tl.load(event_trades + event)
            opening = tl.load(event_openings + event)
            if opening:
                strategy = tl.load(strategies + index)
                weight = tl.load(
                    allocations + lane * row_stride + strategy * strategy_stride,
                    valid,
                    other=0,
                )
                requested = weight * equity
                blocked = tl.full((block_size,), False, tl.int1)
                blocker_start = tl.load(blocker_offsets + index)
                blocker_stop = tl.load(blocker_offsets + index + 1)
                for blocker_offset in range(blocker_start, blocker_stop):
                    blocker = tl.load(blockers + blocker_offset)
                    blocker_risk = tl.load(risks + blocker * batch + lane, valid, 0)
                    blocked = blocked | (blocker_risk > 0)
                budget = equity * cap
                tolerance = 8 * epsilon * tl.abs(budget)
                fits = requested + reserved <= budget + tolerance
                accepted = (weight > 0) & alive & ~blocked & fits
                risk = tl.where(accepted, requested, 0)
                tl.store(risks + index * batch + lane, risk, valid)
                reserved = reserved + risk
                open_count = open_count + accepted.to(tl.int32)
            else:
                risk = tl.load(risks + index * batch + lane, valid, 0)
                reserved = tl.maximum(reserved - risk, 0)
                open_count = open_count - (risk > 0).to(tl.int32)
                trade_return = tl.load(trade_returns + index)
                realized = realized + risk * trade_return
                reserved = tl.where(open_count == 0, 0, reserved)
        mark_start = tl.load(mark_offsets + day)
        mark_stop = tl.load(mark_offsets + day + 1)
        marked_pnl = tl.full((block_size,), 0, value_dtype)
        for mark in range(mark_start, mark_stop):
            index = tl.load(mark_trades + mark)
            mark_value = tl.load(mark_values + mark)
            risk = tl.load(risks + index * batch + lane, valid, 0)
            marked_pnl = marked_pnl + risk * mark_value
        next_equity = realized + marked_pnl
        daily_pnl = next_equity - equity
        denominator = tl.where(alive, equity, 1)
        daily_return = libdevice.div_rn(daily_pnl, denominator)
        finite = (
            (tl.abs(next_equity) < float("inf"))
            & (tl.abs(daily_pnl) < float("inf"))
            & (tl.abs(daily_return) < float("inf"))
        )
        tl.store(
            returns + lane * days + day,
            tl.where(alive & finite, daily_return, -1),
            valid,
        )
        alive = alive & finite & (next_equity > 0)
        equity = tl.where(alive, next_equity, 0)


def run_position_kernel(
    allocations: torch.Tensor,
    returns: torch.Tensor,
    risks: torch.Tensor,
    buffers: tuple[torch.Tensor, ...],
    cap: float,
) -> None:
    """Launch one warp per candidate tile with precise, non-fused arithmetic."""
    block_size = 32
    # Triton's launch wrapper accepts additional compiler options beyond the kernel signature.
    launch = cast(
        Any, _position_kernel[(triton.cdiv(allocations.shape[0], block_size),)]
    )
    launch(
        allocations,
        returns,
        risks,
        *buffers,
        allocations.shape[0],
        returns.shape[1],
        allocations.stride(0),
        allocations.stride(1),
        cap,
        torch.finfo(allocations.dtype).eps,
        block_size,
        num_warps=1,
        enable_fp_fusion=False,
    )
