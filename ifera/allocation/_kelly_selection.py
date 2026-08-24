"""Exact support selection around the continuous Kelly optimizer."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import torch

from ._kelly_optimizer import (
    KellyFeasibleSet,
    KellyOptimizationResult,
    maximize_kelly_growth,
)
from ._strategy_selection import StrategySelectionConstraints


@dataclass(frozen=True)
class _KellyComponent:
    """One independently budgeted component of the overlap graph."""

    indices: tuple[int, ...]
    index_tensor: torch.Tensor
    feasible_set: KellyFeasibleSet
    strategy_mask: int


@dataclass(frozen=True)
class _Relaxation:
    """A solved continuous relaxation for one branch-and-bound node."""

    allocation: torch.Tensor
    mean_log_growth: float
    upper_bound: float


class _KellyConvergenceError(RuntimeError):
    """Signal that a continuous relaxation is unsafe to use or bound."""


@torch.no_grad()
def maximize_kelly_with_selection(
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    maximal_cliques: list[tuple[int, ...]],
    selection_constraints: StrategySelectionConstraints | None,
    allocation_tolerance: float | None = None,
) -> torch.Tensor | None:
    """Return the selected raw-Kelly optimum, or ``None`` when unsupported.

    A continuous Kelly relaxation can be separated when the maximal overlap
    cliques do not share strategies. Each clique has its own simplex budget;
    uncovered strategies are mutually disjoint one-strategy components. Correlation
    conflicts and the active-strategy cap are then imposed exactly with a
    best-bound branch-and-bound search over strategy exclusions.

    Overlapping maximal cliques describe a more general packing polytope. Its
    linear minimization oracle is intentionally left to the allocation grid,
    so this function reports that geometry as unsupported.
    """
    strategy_count = transposed_returns.shape[0]
    components = _resolve_components(
        strategy_count, maximal_cliques, transposed_returns.device
    )
    if components is None:
        return None

    solver = _RelaxationSolver(
        transposed_returns,
        max_total_allocation,
        components,
        allocation_tolerance,
    )
    root_mask = (1 << strategy_count) - 1
    try:
        root = solver.solve(root_mask)
        if selection_constraints is None:
            return root.allocation

        _validate_selection_constraints(selection_constraints, strategy_count)
        conflict_masks = _symmetric_conflict_masks(
            selection_constraints, strategy_count
        )
        if all(len(component.indices) == 1 for component in components):
            return _select_independent_strategies(
                transposed_returns,
                root.allocation,
                selection_constraints.maximum_active,
                conflict_masks,
            )
        return _best_bound_selection(
            root_mask,
            root,
            solver,
            selection_constraints.maximum_active,
            conflict_masks,
        )
    except _KellyConvergenceError:
        return None


class _RelaxationSolver:
    """Solve and cache separable continuous relaxations."""

    def __init__(
        self,
        transposed_returns: torch.Tensor,
        max_total_allocation: float,
        components: tuple[_KellyComponent, ...],
        allocation_tolerance: float | None,
    ) -> None:
        self._transposed_returns = transposed_returns
        self._max_total_allocation = max_total_allocation
        self._components = components
        self._allocation_tolerance = allocation_tolerance
        self._component_returns = tuple(
            transposed_returns.index_select(0, component.index_tensor)
            for component in components
        )
        self._component_cache: dict[tuple[int, int], KellyOptimizationResult] = {}
        self._node_cache: dict[int, _Relaxation] = {}

    def solve(self, allowed_mask: int) -> _Relaxation:
        """Return the relaxation under a global strategy-allowance mask."""
        cached = self._node_cache.get(allowed_mask)
        if cached is not None:
            return cached

        allocation = torch.zeros(
            self._transposed_returns.shape[0],
            device=self._transposed_returns.device,
            dtype=self._transposed_returns.dtype,
        )
        component_upper_bound = 0.0
        for component_index, component in enumerate(self._components):
            local_mask = allowed_mask & component.strategy_mask
            cache_key = (component_index, local_mask)
            result = self._component_cache.get(cache_key)
            if result is None:
                local_allowed = torch.tensor(
                    [bool(allowed_mask & (1 << index)) for index in component.indices],
                    device=self._transposed_returns.device,
                    dtype=torch.bool,
                )
                result = maximize_kelly_growth(
                    self._component_returns[component_index],
                    self._max_total_allocation,
                    feasible_set=component.feasible_set,
                    allowed_mask=local_allowed,
                    allocation_tolerance=self._allocation_tolerance,
                )
                if not result.converged:
                    raise _KellyConvergenceError
                self._component_cache[cache_key] = result
            allocation[component.index_tensor] = result.allocation
            component_upper_bound += result.upper_bound

        mean_log_growth = _mean_log_growth(self._transposed_returns, allocation)
        relaxation = _Relaxation(
            allocation,
            mean_log_growth,
            max(component_upper_bound, mean_log_growth),
        )
        self._node_cache[allowed_mask] = relaxation
        return relaxation


def _resolve_components(
    strategy_count: int,
    maximal_cliques: list[tuple[int, ...]],
    device: torch.device,
) -> tuple[_KellyComponent, ...] | None:
    """Resolve disjoint simplex cliques and their isolated box component."""
    if not maximal_cliques:
        return tuple(
            _make_component((strategy,), "simplex", device)
            for strategy in range(strategy_count)
        )

    components: list[_KellyComponent] = []
    assigned_mask = 0
    for clique in maximal_cliques:
        clique_mask = _indices_mask(clique, strategy_count)
        if clique_mask is None or clique_mask & assigned_mask:
            return None
        assigned_mask |= clique_mask
        components.append(_make_component(clique, "simplex", device))

    components.extend(
        _make_component((strategy,), "simplex", device)
        for strategy in range(strategy_count)
        if not assigned_mask & (1 << strategy)
    )
    return tuple(components)


def _make_component(
    indices: tuple[int, ...],
    feasible_set: KellyFeasibleSet,
    device: torch.device,
) -> _KellyComponent:
    """Materialize one component and its two useful index representations."""
    return _KellyComponent(
        indices,
        torch.tensor(indices, device=device, dtype=torch.int64),
        feasible_set,
        sum(1 << strategy for strategy in indices),
    )


def _indices_mask(indices: tuple[int, ...], strategy_count: int) -> int | None:
    """Encode unique in-range indices, or report malformed geometry."""
    mask = 0
    for strategy in indices:
        if strategy < 0 or strategy >= strategy_count:
            return None
        bit = 1 << strategy
        if mask & bit:
            return None
        mask |= bit
    return mask or None


def _validate_selection_constraints(
    constraints: StrategySelectionConstraints,
    strategy_count: int,
) -> None:
    """Reject inconsistent internal support constraints clearly."""
    if not 0 <= constraints.maximum_active <= strategy_count:
        raise ValueError("maximum_active must be between zero and strategy count")
    if len(constraints.prior_conflicts) != strategy_count:
        raise ValueError("prior_conflicts must have one tensor per strategy")


def _symmetric_conflict_masks(
    constraints: StrategySelectionConstraints,
    strategy_count: int,
) -> tuple[int, ...]:
    """Expand the compact prior-conflict tensors to symmetric bit masks."""
    masks = [0] * strategy_count
    for strategy, prior_conflicts in enumerate(constraints.prior_conflicts):
        for prior in prior_conflicts.to(device="cpu").tolist():
            if not 0 <= prior < strategy or prior >= strategy_count:
                raise ValueError("prior_conflicts must contain prior strategy indices")
            masks[strategy] |= 1 << prior
            masks[prior] |= 1 << strategy
    return tuple(masks)


def _best_bound_selection(
    root_mask: int,
    root: _Relaxation,
    solver: _RelaxationSolver,
    maximum_active: int,
    conflict_masks: tuple[int, ...],
) -> torch.Tensor:
    """Globally impose support constraints using relaxation upper bounds."""
    best_allocation = torch.zeros_like(root.allocation)
    best_score = 0.0  # Cash is always feasible and has zero log growth.
    serial = itertools.count()
    queue: list[tuple[float, int, int, _Relaxation]] = [
        (-root.upper_bound, next(serial), root_mask, root)
    ]
    visited = {root_mask}

    while queue:
        negative_bound, _, allowed_mask, relaxation = heapq.heappop(queue)
        if -negative_bound <= best_score:
            continue

        support_mask = _positive_support_mask(relaxation.allocation)
        conflict = _first_support_conflict(support_mask, conflict_masks)
        if conflict is not None:
            branch_strategies = conflict
        elif support_mask.bit_count() > maximum_active:
            branch_strategies = _cardinality_branch_strategies(
                relaxation.allocation,
                support_mask,
                maximum_active + 1,
            )
        else:
            if relaxation.mean_log_growth > best_score:
                best_score = relaxation.mean_log_growth
                best_allocation = relaxation.allocation
            continue

        for strategy in branch_strategies:
            child_mask = allowed_mask & ~(1 << strategy)
            if child_mask in visited:
                continue
            visited.add(child_mask)
            child = solver.solve(child_mask)
            if child.upper_bound > best_score:
                heapq.heappush(
                    queue,
                    (-child.upper_bound, next(serial), child_mask, child),
                )
    return best_allocation


def _select_independent_strategies(
    transposed_returns: torch.Tensor,
    unrestricted_allocation: torch.Tensor,
    maximum_active: int,
    conflict_masks: tuple[int, ...],
) -> torch.Tensor:
    """Solve cardinality-constrained disjoint Kelly as weighted selection."""
    strategy_growth = torch.log1p(
        unrestricted_allocation.unsqueeze(1) * transposed_returns
    ).mean(dim=1)
    positive = torch.nonzero(strategy_growth > 0.0, as_tuple=False).flatten()
    ordered = tuple(
        sorted(
            positive.to(device="cpu").tolist(),
            key=lambda strategy: (-float(strategy_growth[strategy].item()), strategy),
        )
    )
    if maximum_active == 0 or not ordered:
        return torch.zeros_like(unrestricted_allocation)

    weights = tuple(float(strategy_growth[strategy].item()) for strategy in ordered)
    best_score = 0.0
    best_support = 0
    # position, selected support, forbidden strategies, count, score
    stack = [(0, 0, 0, 0, 0.0)]
    while stack:
        position, support, forbidden, count, score = stack.pop()
        if score > best_score:
            best_score, best_support = score, support
        remaining_slots = maximum_active - count
        if remaining_slots == 0 or position == len(ordered):
            continue

        optimistic = score
        used_slots = 0
        for later_position in range(position, len(ordered)):
            strategy = ordered[later_position]
            if forbidden & (1 << strategy):
                continue
            optimistic += weights[later_position]
            used_slots += 1
            if used_slots == remaining_slots:
                break
        if optimistic <= best_score:
            continue

        strategy = ordered[position]
        stack.append((position + 1, support, forbidden, count, score))
        strategy_bit = 1 << strategy
        if not forbidden & strategy_bit:
            stack.append(
                (
                    position + 1,
                    support | strategy_bit,
                    forbidden | conflict_masks[strategy],
                    count + 1,
                    score + weights[position],
                )
            )

    selected = torch.tensor(
        [
            bool(best_support & (1 << strategy))
            for strategy in range(len(conflict_masks))
        ],
        device=unrestricted_allocation.device,
        dtype=torch.bool,
    )
    return unrestricted_allocation.masked_fill(~selected, 0.0)


def _positive_support_mask(allocation: torch.Tensor) -> int:
    """Encode the exact positive support of an allocation."""
    mask = 0
    positive = torch.nonzero(allocation > 0.0, as_tuple=False).flatten()
    for strategy in positive.to(device="cpu").tolist():
        mask |= 1 << strategy
    return mask


def _first_support_conflict(
    support_mask: int,
    conflict_masks: tuple[int, ...],
) -> tuple[int, int] | None:
    """Return one conflicting pair in the current relaxed support."""
    remaining = support_mask
    while remaining:
        bit = remaining & -remaining
        strategy = bit.bit_length() - 1
        later_conflicts = conflict_masks[strategy] & support_mask & ~((bit << 1) - 1)
        if later_conflicts:
            other_bit = later_conflicts & -later_conflicts
            return strategy, other_bit.bit_length() - 1
        remaining ^= bit
    return None


def _cardinality_branch_strategies(
    allocation: torch.Tensor,
    support_mask: int,
    branch_count: int,
) -> tuple[int, ...]:
    """Choose K+1 supported strategies whose exclusions cover every K-support."""
    supported = torch.tensor(
        list(_iter_set_bits(support_mask)),
        device=allocation.device,
        dtype=torch.int64,
    )
    values = allocation.index_select(0, supported)
    positions = torch.topk(values, branch_count).indices
    return tuple(supported.index_select(0, positions).to(device="cpu").tolist())


def _iter_set_bits(mask: int):
    """Yield indices of set bits in ascending order."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _mean_log_growth(
    transposed_returns: torch.Tensor, allocation: torch.Tensor
) -> float:
    """Evaluate the combined relaxation allocation in its original space."""
    wealth = 1.0 + allocation @ transposed_returns
    if not bool(torch.all(wealth > 0.0).item()):
        raise RuntimeError("the continuous Kelly relaxation became insolvent")
    return float(torch.log(wealth).mean().item())
