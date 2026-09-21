"""Candidate-specific trade blocking for capital-allocation searches."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass

import torch

Block = tuple[int, int, int]
BlockList = Sequence[Sequence[int]]


@dataclass(frozen=True)
class AllocationBlocks:
    """Device tensors representing OR-grouped strategy-day block relations."""

    day_indices: torch.Tensor
    blocked_indices: torch.Tensor
    blocker_incidence: torch.Tensor

    @property
    def group_count(self) -> int:
        """Return the number of unique blocked strategy-day groups."""
        return self.day_indices.numel()


def normalize_block_list(
    block_list: BlockList | None,
    time_count: int,
    strategy_count: int,
) -> tuple[Block, ...]:
    """Validate and deterministically deduplicate public block triples."""
    if block_list is None:
        return ()
    if isinstance(block_list, (str, bytes)) or not isinstance(block_list, Sequence):
        raise TypeError("block_list must be a sequence of three-integer sequences")

    normalized: set[Block] = set()
    for row_index, row in enumerate(block_list):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"block_list[{row_index}] must be a three-integer sequence")
        if len(row) != 3:
            raise ValueError(
                f"block_list[{row_index}] must contain exactly three values"
            )
        values: list[int] = []
        for value_index, value in enumerate(row):
            if isinstance(value, bool):
                raise TypeError(
                    f"block_list[{row_index}][{value_index}] must be an integer"
                )
            try:
                values.append(operator.index(value))
            except TypeError as exc:
                raise TypeError(
                    f"block_list[{row_index}][{value_index}] must be an integer"
                ) from exc
        day_index = values[0]
        blocked = values[1]
        blocked_by = values[2]
        if not 0 <= day_index < time_count:
            raise ValueError(f"block_list[{row_index}] day index is out of range")
        if not 0 <= blocked < strategy_count:
            raise ValueError(f"block_list[{row_index}] blocked index is out of range")
        if not 0 <= blocked_by < strategy_count:
            raise ValueError(
                f"block_list[{row_index}] blocked_by index is out of range"
            )
        if blocked == blocked_by:
            raise ValueError("a strategy cannot block itself")
        normalized.add((day_index, blocked, blocked_by))
    return tuple(sorted(normalized))


def build_allocation_blocks(
    blocks: tuple[Block, ...],
    strategy_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AllocationBlocks | None:
    """Group block triples by strategy-day and build their blocker incidence."""
    if not blocks:
        return None

    grouped: dict[tuple[int, int], set[int]] = {}
    for day_index, blocked, blocked_by in blocks:
        grouped.setdefault((day_index, blocked), set()).add(blocked_by)
    group_keys = sorted(grouped)
    day_indices = torch.tensor(
        [day_index for day_index, _ in group_keys],
        device=device,
        dtype=torch.int64,
    )
    blocked_indices = torch.tensor(
        [blocked for _, blocked in group_keys],
        device=device,
        dtype=torch.int64,
    )
    blocker_rows: list[int] = []
    blocker_columns: list[int] = []
    for group_index, key in enumerate(group_keys):
        blockers = sorted(grouped[key])
        blocker_rows.extend(blockers)
        blocker_columns.extend([group_index] * len(blockers))
    blocker_incidence = torch.zeros(
        (strategy_count, len(group_keys)), device=device, dtype=dtype
    )
    blocker_incidence[
        torch.tensor(blocker_rows, device=device, dtype=torch.int64),
        torch.tensor(blocker_columns, device=device, dtype=torch.int64),
    ] = 1.0
    return AllocationBlocks(day_indices, blocked_indices, blocker_incidence)


def effective_portfolio_returns(
    allocations: torch.Tensor,
    transposed_returns: torch.Tensor,
    block_day_indices: torch.Tensor | None = None,
    blocked_strategy_indices: torch.Tensor | None = None,
    blocker_incidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return portfolio paths after applying allocation-triggered trade blocks."""
    portfolio_returns = allocations @ transposed_returns
    if block_day_indices is None:
        return portfolio_returns
    if blocked_strategy_indices is None or blocker_incidence is None:
        raise RuntimeError("incomplete allocation block tensors")

    blocker_is_active = (
        allocations.ne(0.0).to(dtype=allocations.dtype) @ blocker_incidence
    ).gt(0.0)
    blocked_contributions = (
        allocations[:, blocked_strategy_indices]
        * transposed_returns[blocked_strategy_indices, block_day_indices]
    )
    corrections = blocker_is_active.to(dtype=allocations.dtype) * blocked_contributions
    scatter_indices = block_day_indices.unsqueeze(0).expand(allocations.shape[0], -1)
    return portfolio_returns.scatter_add(1, scatter_indices, -corrections)


def remap_block_list(
    blocks: Sequence[Block], strategy_indices: Sequence[int]
) -> tuple[Block, ...]:
    """Filter blocks to a strategy subset and remap their strategy indices."""
    remapping = {
        strategy_index: index for index, strategy_index in enumerate(strategy_indices)
    }
    return tuple(
        (day_index, remapping[blocked], remapping[blocked_by])
        for day_index, blocked, blocked_by in blocks
        if blocked in remapping and blocked_by in remapping
    )


def slice_block_list(
    blocks: Sequence[Block], start_index: int, stop_index: int
) -> tuple[Block, ...]:
    """Filter blocks to a half-open day range and rebase its day indices."""
    return tuple(
        (day_index - start_index, blocked, blocked_by)
        for day_index, blocked, blocked_by in blocks
        if start_index <= day_index < stop_index
    )
