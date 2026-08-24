"""Support-region helpers for the adaptive spatial ADD search."""

from __future__ import annotations

import itertools

import torch

from ._strategy_selection import StrategySelectionConstraints

_MAX_SUPPORT_ROOTS = 128
_MAX_SUPPORT_COMBINATIONS = 100_000


def maximal_support_masks(
    strategy_count: int,
    constraints: StrategySelectionConstraints | None,
) -> tuple[int, ...] | None:
    """Enumerate a capped cover by maximal selection-compatible supports."""
    all_mask = (1 << strategy_count) - 1
    if constraints is None:
        return (all_mask,)
    maximum = min(constraints.maximum_active, strategy_count)
    if maximum <= 0:
        return (0,)
    conflict_masks = _symmetric_conflict_masks(constraints, strategy_count)
    results: list[int] = []
    examined = 0
    indices = range(strategy_count)
    for size in range(maximum, 0, -1):
        for support in itertools.combinations(indices, size):
            examined += 1
            if examined > _MAX_SUPPORT_COMBINATIONS:
                return None
            mask = sum(1 << strategy for strategy in support)
            if not _support_is_compatible(mask, conflict_masks):
                continue
            if any(mask & result == mask for result in results):
                continue
            results.append(mask)
            if len(results) > _MAX_SUPPORT_ROOTS:
                return None
    return tuple(results) if results else (0,)


def root_box(
    support_mask: int,
    strategy_count: int,
    maximum: float,
    transposed_returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the zero-to-maximum box for one legal support."""
    lower = torch.zeros(
        strategy_count,
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    upper = torch.tensor(
        [
            maximum if support_mask & (1 << index) else 0.0
            for index in range(strategy_count)
        ],
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    return lower, upper


def allocation_is_feasible(
    allocation: torch.Tensor,
    transposed_returns: torch.Tensor,
    maximum: float,
    maximal_cliques: list[tuple[int, ...]],
    constraints: StrategySelectionConstraints | None,
) -> bool:
    """Validate an optional incumbent before it can tighten growth pruning."""
    if not _allocation_tensor_is_valid(allocation, transposed_returns, maximum):
        return False
    for clique in maximal_cliques:
        indices = torch.tensor(clique, device=allocation.device, dtype=torch.int64)
        if float(allocation.index_select(0, indices).sum().item()) > maximum:
            return False
    if constraints is None:
        return True
    support = allocation > 0.0
    if int(torch.count_nonzero(support).item()) > constraints.maximum_active:
        return False
    for strategy, prior_conflicts in enumerate(constraints.prior_conflicts):
        if not bool(support[strategy].item()):
            continue
        conflicts = prior_conflicts.to(device=allocation.device)
        if conflicts.numel() and bool(torch.any(support[conflicts]).item()):
            return False
    return True


def _allocation_tensor_is_valid(
    allocation: torch.Tensor,
    transposed_returns: torch.Tensor,
    maximum: float,
) -> bool:
    """Return whether the candidate has valid tensor structure and coordinates."""
    return (
        isinstance(allocation, torch.Tensor)
        and allocation.shape == (transposed_returns.shape[0],)
        and allocation.device == transposed_returns.device
        and allocation.dtype == transposed_returns.dtype
        and bool(torch.isfinite(allocation).all().item())
        and not bool(torch.any(allocation < 0.0).item())
        and not bool(torch.any(allocation > maximum).item())
    )


def _symmetric_conflict_masks(
    constraints: StrategySelectionConstraints, strategy_count: int
) -> tuple[int, ...]:
    """Expand compact prior conflicts into symmetric CPU integer masks."""
    masks = [0] * strategy_count
    for strategy, conflicts in enumerate(constraints.prior_conflicts):
        for prior in conflicts.detach().to(device="cpu").tolist():
            masks[strategy] |= 1 << prior
            masks[prior] |= 1 << strategy
    return tuple(masks)


def _support_is_compatible(mask: int, conflict_masks: tuple[int, ...]) -> bool:
    """Return whether a support contains no forbidden pair."""
    remaining = mask
    while remaining:
        bit = remaining & -remaining
        strategy = bit.bit_length() - 1
        if conflict_masks[strategy] & mask:
            return False
        remaining ^= bit
    return True
