"""Overlap-clique constraints for reusable trading-strategy capital."""

from __future__ import annotations

from collections.abc import Iterator

import torch


def find_maximal_overlap_cliques(active: torch.Tensor) -> list[tuple[int, ...]]:
    """Return nontrivial maximal cliques of pairwise-overlapping strategies."""
    strategy_count = active.shape[1]
    if strategy_count < 2:
        return []
    if bool(torch.all(active).item()):
        return [tuple(range(strategy_count))]

    activity = active.to(dtype=torch.float32)
    overlap = (activity.transpose(0, 1) @ activity) > 0.0
    overlap.fill_diagonal_(False)
    adjacency = overlap.to(device="cpu")
    neighbor_masks = [
        sum(1 << neighbor for neighbor, connected in enumerate(row) if connected)
        for row in adjacency.tolist()
    ]
    all_strategies = (1 << strategy_count) - 1
    if all(
        neighbors == all_strategies ^ (1 << strategy)
        for strategy, neighbors in enumerate(neighbor_masks)
    ):
        return [tuple(range(strategy_count))]

    maximal_cliques: list[tuple[int, ...]] = []
    _collect_maximal_cliques(
        0,
        all_strategies,
        0,
        neighbor_masks,
        maximal_cliques,
    )
    return maximal_cliques


def _collect_maximal_cliques(
    clique: int,
    candidates: int,
    excluded: int,
    neighbor_masks: list[int],
    maximal_cliques: list[tuple[int, ...]],
) -> None:
    """Collect maximal cliques with an iterative Bron-Kerbosch search."""
    search_stack = [(clique, candidates, excluded)]
    while search_stack:
        current_clique, current_candidates, current_excluded = search_stack.pop()
        if current_candidates == 0 and current_excluded == 0:
            if current_clique.bit_count() > 1:
                maximal_cliques.append(
                    tuple(
                        strategy
                        for strategy in range(len(neighbor_masks))
                        if current_clique & (1 << strategy)
                    )
                )
            continue

        pivot_options = current_candidates | current_excluded
        if pivot_options:
            pivot = max(
                _iter_set_bits(pivot_options),
                key=lambda strategy: (
                    current_candidates & neighbor_masks[strategy]
                ).bit_count(),
            )
            extensions = current_candidates & ~neighbor_masks[pivot]
        else:
            extensions = current_candidates

        child_states = []
        for strategy in _iter_set_bits(extensions):
            strategy_bit = 1 << strategy
            neighbors = neighbor_masks[strategy]
            child_states.append(
                (
                    current_clique | strategy_bit,
                    current_candidates & neighbors,
                    current_excluded & neighbors,
                )
            )
            current_candidates &= ~strategy_bit
            current_excluded |= strategy_bit
        search_stack.extend(reversed(child_states))


def _iter_set_bits(mask: int) -> Iterator[int]:
    """Yield the indices of set bits in ascending order."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def clique_indices_by_strategy(
    maximal_cliques: list[tuple[int, ...]],
    strategy_count: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return constraint indices containing each strategy."""
    constraint_indices: list[list[int]] = [[] for _ in range(strategy_count)]
    for clique_index, clique in enumerate(maximal_cliques):
        for strategy in clique:
            constraint_indices[strategy].append(clique_index)
    return [
        torch.tensor(indices, device=device, dtype=torch.int64)
        for indices in constraint_indices
    ]
