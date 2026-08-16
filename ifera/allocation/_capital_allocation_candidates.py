"""Valid-only prefix generation for constrained capital-allocation grids."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import torch

from ._strategy_selection import StrategySelectionConstraints


class _ExpansionFrame(NamedTuple):
    """State needed to resume one vectorized prefix expansion."""

    prefixes: torch.Tensor
    clique_usage: torch.Tensor
    active_counts: torch.Tensor | None
    column: int
    cumulative_counts: torch.Tensor
    child_counts: torch.Tensor
    next_child: int
    child_count: int


def constrained_allocation_unit_batches(
    allocation_unit_limit: int,
    strategy_count: int,
    clique_count: int,
    cliques_by_strategy: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    clique_unit_limits: torch.Tensor | None = None,
    selection_constraints: StrategySelectionConstraints | None = None,
    positive_unit_starts: tuple[int, ...] | None = None,
) -> Iterator[torch.Tensor]:
    """Yield full scoring batches under capital and support constraints."""
    fragments = _constrained_allocation_unit_fragments(
        allocation_unit_limit,
        strategy_count,
        clique_count,
        cliques_by_strategy,
        device,
        batch_size,
        clique_unit_limits,
        selection_constraints,
        positive_unit_starts,
    )
    return _coalesce_candidate_batches(fragments, batch_size)


def _constrained_allocation_unit_fragments(
    allocation_unit_limit: int,
    strategy_count: int,
    clique_count: int,
    cliques_by_strategy: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    clique_unit_limits: torch.Tensor | None = None,
    selection_constraints: StrategySelectionConstraints | None = None,
    positive_unit_starts: tuple[int, ...] | None = None,
) -> Iterator[torch.Tensor]:
    """Yield grid batches under capital, support-count, and pair constraints."""
    integer_limit = torch.iinfo(torch.int64).max
    clique_unit_limits = _resolve_clique_unit_limits(
        allocation_unit_limit,
        clique_count,
        clique_unit_limits,
        device,
        integer_limit,
    )

    positive_starts_tensor = None
    active_counts = None
    if selection_constraints is not None:
        if positive_unit_starts is None:
            positive_unit_starts = (1,) * strategy_count
        if len(positive_unit_starts) != strategy_count or any(
            value not in (0, 1) for value in positive_unit_starts
        ):
            raise ValueError(
                "positive_unit_starts must contain one 0 or 1 per strategy"
            )
        positive_starts_tensor = torch.tensor(
            positive_unit_starts, device=device, dtype=torch.int64
        )
        active_counts = torch.zeros((1,), device=device, dtype=torch.int64)

    safe_batch_size = min(batch_size, integer_limit // (allocation_unit_limit + 1))
    prefixes = torch.empty((1, 0), device=device, dtype=torch.int64)
    clique_usage = torch.zeros((1, clique_count), device=device, dtype=torch.int64)
    column = 0
    expansion_stack: list[_ExpansionFrame] = []

    while True:
        if column == strategy_count:
            yield prefixes
        else:
            expansion = _make_constrained_expansion(
                prefixes,
                clique_usage,
                active_counts,
                column,
                allocation_unit_limit,
                clique_unit_limits,
                cliques_by_strategy,
                selection_constraints,
                positive_unit_starts,
                positive_starts_tensor,
            )
            if expansion.child_count > 0:
                prefixes, clique_usage, active_counts, next_child = (
                    _constrained_child_batch(
                        expansion,
                        cliques_by_strategy,
                        positive_unit_starts,
                        safe_batch_size,
                    )
                )
                if next_child < expansion.child_count:
                    expansion_stack.append(expansion._replace(next_child=next_child))
                column += 1
                continue

        if not expansion_stack:
            return
        expansion = expansion_stack.pop()
        prefixes, clique_usage, active_counts, next_child = _constrained_child_batch(
            expansion,
            cliques_by_strategy,
            positive_unit_starts,
            safe_batch_size,
        )
        if next_child < expansion.child_count:
            expansion_stack.append(expansion._replace(next_child=next_child))
        column = expansion.column + 1


def _coalesce_candidate_batches(
    fragments: Iterator[torch.Tensor], batch_size: int
) -> Iterator[torch.Tensor]:
    """Combine adjacent DFS leaf fragments without changing candidate order."""
    pending: list[torch.Tensor] = []
    pending_count = 0
    for fragment in fragments:
        pending.append(fragment)
        pending_count += fragment.shape[0]
        if pending_count < batch_size:
            continue
        combined = torch.cat(pending, dim=0) if len(pending) > 1 else pending[0]
        yield combined[:batch_size]
        remainder = combined[batch_size:]
        pending = [remainder] if remainder.shape[0] else []
        pending_count = remainder.shape[0]
    if pending:
        yield torch.cat(pending, dim=0) if len(pending) > 1 else pending[0]


def _resolve_clique_unit_limits(
    allocation_unit_limit: int,
    clique_count: int,
    clique_unit_limits: torch.Tensor | None,
    device: torch.device,
    integer_limit: int,
) -> torch.Tensor:
    """Validate and materialize the vector of overlap-clique budgets."""
    if allocation_unit_limit >= integer_limit:
        raise ValueError("the allocation limit is too large to index")
    if clique_unit_limits is None:
        clique_unit_limits = torch.full(
            (clique_count,), allocation_unit_limit, device=device, dtype=torch.int64
        )
    if clique_unit_limits.shape != (clique_count,):
        raise ValueError("clique_unit_limits must have one value per clique")
    return clique_unit_limits


def _make_constrained_expansion(
    prefixes: torch.Tensor,
    clique_usage: torch.Tensor,
    active_counts: torch.Tensor | None,
    column: int,
    allocation_unit_limit: int,
    clique_unit_limits: torch.Tensor,
    cliques_by_strategy: list[torch.Tensor],
    selection_constraints: StrategySelectionConstraints | None,
    positive_unit_starts: tuple[int, ...] | None,
    positive_starts_tensor: torch.Tensor | None,
) -> _ExpansionFrame:
    """Prepare child counts for one valid-only prefix expansion."""
    constraint_indices = cliques_by_strategy[column]
    if constraint_indices.numel() == 0:
        limits = torch.full(
            (prefixes.shape[0],),
            allocation_unit_limit,
            device=prefixes.device,
            dtype=torch.int64,
        )
    else:
        used = clique_usage[:, constraint_indices]
        remaining = clique_unit_limits[constraint_indices].unsqueeze(0) - used
        limits = torch.amin(remaining, dim=1)
        limits.clamp_max_(allocation_unit_limit)

    child_counts = limits + 1
    if selection_constraints is not None:
        if (
            active_counts is None
            or positive_unit_starts is None
            or positive_starts_tensor is None
        ):
            raise RuntimeError("strategy-selection prefix state is unavailable")
        can_activate = active_counts < selection_constraints.maximum_active
        conflict_indices = selection_constraints.prior_conflicts[column]
        if conflict_indices.numel() > 0:
            conflict_starts = positive_starts_tensor[conflict_indices]
            conflicting = torch.any(
                prefixes[:, conflict_indices] >= conflict_starts.unsqueeze(0), dim=1
            )
            can_activate &= ~conflicting
        if positive_unit_starts[column] == 0:
            child_counts = torch.where(
                can_activate, child_counts, torch.zeros_like(child_counts)
            )
        else:
            child_counts = torch.where(
                can_activate, child_counts, child_counts.clamp_max(1)
            )

    cumulative_counts = torch.cumsum(child_counts, dim=0)
    child_count = int(cumulative_counts[-1].item())
    return _ExpansionFrame(
        prefixes,
        clique_usage,
        active_counts,
        column,
        cumulative_counts,
        child_counts,
        0,
        child_count,
    )


def _constrained_child_batch(
    expansion: _ExpansionFrame,
    cliques_by_strategy: list[torch.Tensor],
    positive_unit_starts: tuple[int, ...] | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """Expand the next bounded child range from a saved prefix state."""
    stop = min(expansion.next_child + batch_size, expansion.child_count)
    flat_indices = torch.arange(
        expansion.next_child,
        stop,
        device=expansion.prefixes.device,
        dtype=torch.int64,
    )
    parent_indices = torch.searchsorted(
        expansion.cumulative_counts, flat_indices, right=True
    )
    values = (
        flat_indices
        - expansion.cumulative_counts[parent_indices]
        + expansion.child_counts[parent_indices]
    )
    child_prefixes = torch.cat(
        (expansion.prefixes[parent_indices], values.unsqueeze(1)), dim=1
    )
    child_usage = expansion.clique_usage[parent_indices]
    constraint_indices = cliques_by_strategy[expansion.column]
    if constraint_indices.numel() > 0:
        child_usage[:, constraint_indices] = child_usage[
            :, constraint_indices
        ] + values.unsqueeze(1)

    child_active_counts = None
    if expansion.active_counts is not None:
        if positive_unit_starts is None:
            raise RuntimeError("strategy-selection positivity is unavailable")
        child_active_counts = expansion.active_counts[parent_indices] + (
            values >= positive_unit_starts[expansion.column]
        ).to(dtype=torch.int64)
    return child_prefixes, child_usage, child_active_counts, stop
