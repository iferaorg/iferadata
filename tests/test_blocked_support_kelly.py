"""Tests for tolerance-bounded Kelly searches across blocked support regions."""

from __future__ import annotations

import torch

from ifera.allocation._blocked_support_kelly import (
    MAX_EXACT_SUPPORTS,
    blocked_returns_for_support,
    enumerate_exact_support_masks,
    try_maximize_blocked_support_kelly,
)
from ifera.allocation._capital_allocation_blocks import build_allocation_blocks
from ifera.allocation._strategy_selection import StrategySelectionConstraints


def _constraints(
    strategy_count: int,
    maximum_active: int,
    conflicts: tuple[tuple[int, int], ...] = (),
) -> StrategySelectionConstraints:
    """Build the compact production representation of support conflicts."""
    prior: list[list[int]] = [[] for _ in range(strategy_count)]
    for first, second in conflicts:
        lower, upper = sorted((first, second))
        prior[upper].append(lower)
    return StrategySelectionConstraints(
        maximum_active,
        tuple(torch.tensor(indices, dtype=torch.int64) for indices in prior),
    )


def test_exact_support_enumeration_respects_selection_and_floor_budgets():
    """Only correlation-compatible, cardinality-safe, funded masks are emitted."""
    masks = enumerate_exact_support_masks(
        strategy_count=3,
        max_total_allocation=0.2,
        maximal_cliques=[(0, 1)],
        selection_constraints=_constraints(3, 2, conflicts=((1, 2),)),
        minimum_active_allocation=0.1,
    )

    assert masks is not None
    assert set(masks) == {0, 1, 2, 3, 4, 5}


def test_exact_support_enumeration_aborts_at_the_shared_cap():
    """The cap includes cash and is checked before any optimizer work."""
    seven_strategy_masks = enumerate_exact_support_masks(7, 1.0, [], None, 1e-6)
    eight_strategy_masks = enumerate_exact_support_masks(8, 1.0, [], None, 1e-6)

    assert seven_strategy_masks is not None
    assert len(seven_strategy_masks) == MAX_EXACT_SUPPORTS == 128
    assert eight_strategy_masks is None


def test_support_fixed_returns_apply_multiple_blockers_with_or_semantics():
    """A support activates a grouped block once regardless of blocker count."""
    transposed_returns = torch.tensor(
        [[0.4, 0.2], [0.1, 0.0], [0.2, 0.0]], dtype=torch.float64
    )
    blocks = build_allocation_blocks(
        ((0, 0, 1), (0, 0, 2)),
        strategy_count=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert blocks is not None

    unblocked = blocked_returns_for_support(transposed_returns, blocks, 0b001)
    blocked_once = blocked_returns_for_support(transposed_returns, blocks, 0b011)
    blocked_twice = blocked_returns_for_support(transposed_returns, blocks, 0b111)

    torch.testing.assert_close(unblocked, transposed_returns)
    assert blocked_once[0, 0].item() == 0.0
    assert blocked_twice[0, 0].item() == 0.0
    torch.testing.assert_close(blocked_once[1:], transposed_returns[1:])


def test_blocked_support_kelly_uses_the_floor_to_capture_blocking_benefit():
    """The best region can retain a blocker exactly at the positive floor."""
    transposed_returns = torch.tensor([[-0.5, 1.0], [-0.01, 0.0]], dtype=torch.float64)
    blocks = build_allocation_blocks(
        ((0, 0, 1),),
        strategy_count=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert blocks is not None

    outcome = try_maximize_blocked_support_kelly(
        transposed_returns,
        max_total_allocation=1.0,
        maximal_cliques=[(0, 1)],
        selection_constraints=None,
        allocation_blocks=blocks,
        allocation_tolerance=0.1,
    )

    assert outcome is not None
    assert outcome.support_mask == 0b11
    torch.testing.assert_close(
        outcome.allocation,
        torch.tensor([0.9, 0.1], dtype=torch.float64),
        atol=1e-8,
        rtol=0.0,
    )
    assert len(outcome.candidates) == 4


def test_blocked_support_kelly_preserves_disjoint_component_capital_reuse():
    """Independent overlap components each retain their own allocation budget."""
    transposed_returns = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.3],
        ],
        dtype=torch.float64,
    )
    blocks = build_allocation_blocks(
        ((1, 0, 1),),
        strategy_count=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert blocks is not None

    outcome = try_maximize_blocked_support_kelly(
        transposed_returns,
        1.0,
        [(0, 1)],
        None,
        blocks,
        0.1,
    )

    assert outcome is not None
    torch.testing.assert_close(
        outcome.allocation,
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        atol=1e-8,
        rtol=0.0,
    )
    assert outcome.allocation.sum().item() == 2.0


def test_blocked_support_kelly_requests_grid_for_unsupported_geometry():
    """Intersecting overlap cliques remain outside the Frank--Wolfe fast path."""
    returns = torch.full((3, 4), 0.1, dtype=torch.float64)
    blocks = build_allocation_blocks(
        ((0, 0, 1),), 3, torch.device("cpu"), torch.float64
    )
    assert blocks is not None

    outcome = try_maximize_blocked_support_kelly(
        returns,
        1.0,
        [(0, 1), (1, 2)],
        None,
        blocks,
        0.1,
    )

    assert outcome is None


def test_blocked_support_kelly_requests_grid_for_an_insolvent_lower_corner():
    """A region is not discarded when its lower corner cannot seed Frank--Wolfe."""
    returns = torch.tensor([[-2.0, 0.1], [0.0, 0.0]], dtype=torch.float64)
    blocks = build_allocation_blocks(
        ((1, 0, 1),),
        2,
        torch.device("cpu"),
        torch.float64,
    )
    assert blocks is not None

    outcome = try_maximize_blocked_support_kelly(
        returns,
        1.0,
        [],
        None,
        blocks,
        0.6,
    )

    assert outcome is None
