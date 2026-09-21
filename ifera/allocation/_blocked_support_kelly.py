"""Continuous raw-Kelly search over fixed trade-blocking support regions."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

from ._capital_allocation_blocks import AllocationBlocks
from ._kelly_optimizer import maximize_kelly_growth
from ._strategy_selection import StrategySelectionConstraints

MAX_EXACT_SUPPORTS = 128
BLOCKED_SUPPORT_KELLY_ALGORITHM = (
    "Blocking-aware exact-support continuous raw Kelly (away-step Frank-Wolfe)"
)
_BOUND_ROUNDING_ULPS = 16.0


@dataclass(frozen=True)
class BlockedSupportKellyCandidate:
    """The raw-Kelly optimum within one tolerance-bounded exact support."""

    support_mask: int
    allocation: torch.Tensor
    mean_log_growth: float


@dataclass(frozen=True)
class BlockedSupportKellyOutcome:
    """The global raw-Kelly winner and all reusable support candidates."""

    allocation: torch.Tensor
    mean_log_growth: float
    support_mask: int
    candidates: tuple[BlockedSupportKellyCandidate, ...]
    minimum_active_allocation: float
    algorithm: str = BLOCKED_SUPPORT_KELLY_ALGORITHM

    @property
    def support_masks(self) -> tuple[int, ...]:
        """Return every searched exact support in deterministic order."""
        return tuple(candidate.support_mask for candidate in self.candidates)


def enumerate_exact_support_masks(  # pylint: disable=too-many-arguments
    strategy_count: int,
    max_total_allocation: float,
    maximal_cliques: list[tuple[int, ...]],
    selection_constraints: StrategySelectionConstraints | None,
    minimum_active_allocation: float,
    *,
    maximum_supports: int = MAX_EXACT_SUPPORTS,
) -> tuple[int, ...] | None:
    """Enumerate every legal exact support, or request a grid at the cap.

    Cash counts toward ``maximum_supports``. Support construction stops as soon
    as the cap would be exceeded, before any return matrices or Kelly solves are
    materialized. Pair conflicts, active-strategy cardinality, and the capital
    needed by each support's positive floor are enforced during expansion.
    """
    count = _validate_strategy_count(strategy_count)
    maximum = _finite_nonnegative(max_total_allocation, "max_total_allocation")
    minimum = _finite_positive(minimum_active_allocation, "minimum_active_allocation")
    support_limit = _positive_integer(maximum_supports, "maximum_supports")
    clique_masks, clique_indices_by_strategy = _resolve_clique_masks(
        count, maximal_cliques
    )
    maximum_active, conflict_masks = _resolve_selection_constraints(
        count, selection_constraints
    )
    tolerance = _budget_rounding_tolerance(maximum)

    supports: list[int] = []

    def expand(first_strategy: int, support_mask: int, active_count: int) -> bool:
        supports.append(support_mask)
        if len(supports) > support_limit:
            return False
        if active_count >= maximum_active:
            return True

        for strategy in range(first_strategy, count):
            strategy_bit = 1 << strategy
            if conflict_masks[strategy] & support_mask:
                continue
            if minimum > maximum + tolerance:
                continue
            funded = True
            for clique_index in clique_indices_by_strategy[strategy]:
                active_in_clique = (
                    support_mask & clique_masks[clique_index]
                ).bit_count()
                if (active_in_clique + 1) * minimum > maximum + tolerance:
                    funded = False
                    break
            if funded and not expand(
                strategy + 1,
                support_mask | strategy_bit,
                active_count + 1,
            ):
                return False
        return True

    if not expand(0, 0, 0):
        return None
    return tuple(sorted(supports))


def blocked_returns_for_support(
    transposed_returns: torch.Tensor,
    allocation_blocks: AllocationBlocks,
    support_mask: int,
) -> torch.Tensor:
    """Return a linear return matrix for one fixed allocation support.

    Within an exact support, blocker activation no longer depends on allocation
    magnitude. Every strategy-day group having at least one blocker in the
    support is therefore zeroed once, preserving the public OR semantics.
    """
    strategy_count, _ = _validate_transposed_returns(transposed_returns)
    mask = _validate_support_mask(support_mask, strategy_count)
    _validate_allocation_blocks(allocation_blocks, transposed_returns)
    support = torch.tensor(
        [bool(mask & (1 << strategy)) for strategy in range(strategy_count)],
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    active_groups = (support @ allocation_blocks.blocker_incidence).gt(0.0)
    if not bool(torch.any(active_groups).item()):
        return transposed_returns

    effective_returns = transposed_returns.clone()
    group_indices = torch.nonzero(active_groups, as_tuple=False).flatten()
    blocked = allocation_blocks.blocked_indices.index_select(0, group_indices)
    days = allocation_blocks.day_indices.index_select(0, group_indices)
    effective_returns[blocked, days] = 0.0
    return effective_returns


@torch.no_grad()
def try_maximize_blocked_support_kelly(  # pylint: disable=too-many-arguments
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    maximal_cliques: list[tuple[int, ...]],
    selection_constraints: StrategySelectionConstraints | None,
    allocation_blocks: AllocationBlocks,
    allocation_tolerance: float,
) -> BlockedSupportKellyOutcome | None:
    """Try a complete tolerance-bounded blocked-support raw-Kelly search.

    ``None`` is an all-or-nothing request for the existing complete grid. It is
    returned when support enumeration exceeds :data:`MAX_EXACT_SUPPORTS`, the
    original overlap cliques are not separable components, a positive lower
    corner is insolvent, or any Frank--Wolfe solve does not converge. No support
    is silently discarded in those cases.
    """
    strategy_count, _ = _validate_transposed_returns(transposed_returns)
    maximum = _finite_nonnegative(max_total_allocation, "max_total_allocation")
    tolerance = _finite_positive(allocation_tolerance, "allocation_tolerance")
    _validate_allocation_blocks(allocation_blocks, transposed_returns)
    components = _partitioned_overlap_components(strategy_count, maximal_cliques)
    if components is None:
        return None

    minimum = _representable_minimum(tolerance, transposed_returns)
    if minimum is None:
        return None
    support_masks = enumerate_exact_support_masks(
        strategy_count,
        maximum,
        maximal_cliques,
        selection_constraints,
        minimum,
    )
    if support_masks is None:
        return None

    cash = torch.zeros(
        strategy_count,
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    candidates = [BlockedSupportKellyCandidate(0, cash, 0.0)]
    best_candidate = candidates[0]
    for support_mask in support_masks:
        if support_mask == 0:
            continue
        effective_returns = blocked_returns_for_support(
            transposed_returns, allocation_blocks, support_mask
        )
        candidate = _solve_exact_support(
            effective_returns,
            support_mask,
            components,
            maximum,
            minimum,
            tolerance,
        )
        if candidate is None:
            return None
        candidates.append(candidate)
        if candidate.mean_log_growth > best_candidate.mean_log_growth:
            best_candidate = candidate

    return BlockedSupportKellyOutcome(
        best_candidate.allocation,
        best_candidate.mean_log_growth,
        best_candidate.support_mask,
        tuple(candidates),
        minimum,
    )


def _solve_exact_support(  # pylint: disable=too-many-arguments
    effective_returns: torch.Tensor,
    support_mask: int,
    components: tuple[tuple[int, ...], ...],
    maximum: float,
    minimum: float,
    allocation_tolerance: float,
) -> BlockedSupportKellyCandidate | None:
    """Solve one closed support region by translating each simplex component."""
    allocation = torch.zeros(
        effective_returns.shape[0],
        device=effective_returns.device,
        dtype=effective_returns.dtype,
    )
    for component in components:
        active_indices = tuple(
            strategy for strategy in component if support_mask & (1 << strategy)
        )
        if not active_indices:
            continue
        index_tensor = torch.tensor(
            active_indices, device=effective_returns.device, dtype=torch.int64
        )
        local_returns = effective_returns.index_select(0, index_tensor)
        floor_allocation = torch.full(
            (len(active_indices),),
            minimum,
            device=effective_returns.device,
            dtype=effective_returns.dtype,
        )
        base_wealth = 1.0 + floor_allocation @ local_returns
        if not bool(torch.all(base_wealth > 0.0).item()):
            return None

        residual_budget = maximum - len(active_indices) * minimum
        budget_tolerance = _budget_rounding_tolerance(maximum)
        if residual_budget < -budget_tolerance:
            return None
        residual_budget = max(0.0, residual_budget)
        local_allocation = floor_allocation
        if residual_budget > 0.0:
            residual_returns = local_returns / base_wealth.unsqueeze(0)
            result = maximize_kelly_growth(
                residual_returns,
                residual_budget,
                feasible_set="simplex",
                allocation_tolerance=allocation_tolerance,
            )
            if not result.converged:
                return None
            local_allocation = floor_allocation + result.allocation
            local_allocation = _repair_component_budget(
                local_allocation, minimum, maximum
            )
        allocation[index_tensor] = local_allocation

    portfolio_returns = allocation @ effective_returns
    if not bool(torch.all(portfolio_returns > -1.0).item()):
        return None
    score = float(torch.log1p(portfolio_returns).mean().item())
    return BlockedSupportKellyCandidate(support_mask, allocation, score)


def _repair_component_budget(
    allocation: torch.Tensor, minimum: float, maximum: float
) -> torch.Tensor:
    """Remove only floating-point overshoot while preserving positive floors."""
    total = float(allocation.sum().item())
    if total <= maximum:
        return allocation
    floor_total = allocation.numel() * minimum
    residual_budget = max(0.0, maximum - floor_total)
    residual = torch.clamp_min(allocation - minimum, 0.0)
    residual_total = float(residual.sum().item())
    if residual_total == 0.0:
        return allocation
    return torch.full_like(allocation, minimum) + residual * (
        residual_budget / residual_total
    )


def _partitioned_overlap_components(
    strategy_count: int, maximal_cliques: list[tuple[int, ...]]
) -> tuple[tuple[int, ...], ...] | None:
    """Return independent simplex components, rejecting intersecting cliques."""
    clique_masks, _ = _resolve_clique_masks(strategy_count, maximal_cliques)
    assigned = 0
    components: list[tuple[int, ...]] = []
    for clique, clique_mask in zip(maximal_cliques, clique_masks, strict=True):
        if assigned & clique_mask:
            return None
        assigned |= clique_mask
        components.append(tuple(clique))
    components.extend(
        (strategy,)
        for strategy in range(strategy_count)
        if not assigned & (1 << strategy)
    )
    return tuple(components)


def _resolve_clique_masks(
    strategy_count: int, maximal_cliques: list[tuple[int, ...]]
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Validate cliques and return masks plus memberships by strategy."""
    if not isinstance(maximal_cliques, list):
        raise TypeError("maximal_cliques must be a list of strategy tuples")
    masks: list[int] = []
    memberships: list[list[int]] = [[] for _ in range(strategy_count)]
    for clique_index, clique in enumerate(maximal_cliques):
        if not isinstance(clique, tuple) or not clique:
            raise ValueError("maximal_cliques must contain nonempty tuples")
        mask = 0
        for strategy in clique:
            index = _strategy_index(strategy, strategy_count, "maximal clique")
            bit = 1 << index
            if mask & bit:
                raise ValueError("maximal cliques must not repeat a strategy")
            mask |= bit
            memberships[index].append(clique_index)
        masks.append(mask)
    return tuple(masks), tuple(tuple(values) for values in memberships)


def _resolve_selection_constraints(
    strategy_count: int,
    constraints: StrategySelectionConstraints | None,
) -> tuple[int, tuple[int, ...]]:
    """Expand compact prior conflicts to CPU bit masks."""
    if constraints is None:
        return strategy_count, (0,) * strategy_count
    if not isinstance(constraints, StrategySelectionConstraints):
        raise TypeError(
            "selection_constraints must be StrategySelectionConstraints or None"
        )
    maximum_active = constraints.maximum_active
    if not isinstance(maximum_active, int) or isinstance(maximum_active, bool):
        raise TypeError("maximum_active must be an integer")
    if not 0 <= maximum_active <= strategy_count:
        raise ValueError("maximum_active must be between zero and strategy count")
    if len(constraints.prior_conflicts) != strategy_count:
        raise ValueError("prior_conflicts must have one tensor per strategy")

    masks = [0] * strategy_count
    for strategy, conflicts in enumerate(constraints.prior_conflicts):
        if not isinstance(conflicts, torch.Tensor) or conflicts.ndim != 1:
            raise TypeError("prior_conflicts must contain one-dimensional tensors")
        for prior_value in conflicts.detach().to(device="cpu").tolist():
            prior = _strategy_index(prior_value, strategy_count, "prior conflict")
            if prior >= strategy:
                raise ValueError("prior conflicts must contain earlier strategies")
            masks[strategy] |= 1 << prior
            masks[prior] |= 1 << strategy
    return maximum_active, tuple(masks)


def _validate_allocation_blocks(
    allocation_blocks: AllocationBlocks, transposed_returns: torch.Tensor
) -> None:
    """Validate the tensor representation expected from the public normalizer."""
    if not isinstance(allocation_blocks, AllocationBlocks):
        raise TypeError("allocation_blocks must be an AllocationBlocks instance")
    days = allocation_blocks.day_indices
    blocked = allocation_blocks.blocked_indices
    incidence = allocation_blocks.blocker_incidence
    group_count = days.numel()
    if days.dtype != torch.int64 or blocked.dtype != torch.int64:
        raise TypeError("allocation block indices must have dtype int64")
    if days.shape != (group_count,) or blocked.shape != (group_count,):
        raise ValueError("allocation block indices must be one-dimensional")
    if incidence.shape != (transposed_returns.shape[0], group_count):
        raise ValueError("blocker incidence shape does not match the returns")
    if (
        days.device != transposed_returns.device
        or blocked.device != transposed_returns.device
        or incidence.device != transposed_returns.device
    ):
        raise ValueError("allocation blocks and returns must use the same device")
    if incidence.dtype != transposed_returns.dtype:
        raise TypeError("blocker incidence and returns must use the same dtype")
    if group_count and (
        bool(torch.any(days < 0).item())
        or bool(torch.any(days >= transposed_returns.shape[1]).item())
        or bool(torch.any(blocked < 0).item())
        or bool(torch.any(blocked >= transposed_returns.shape[0]).item())
    ):
        raise ValueError("allocation block indices are out of range")


def _validate_transposed_returns(returns: torch.Tensor) -> tuple[int, int]:
    """Validate the finite calculation tensor required by continuous Kelly."""
    if not isinstance(returns, torch.Tensor):
        raise TypeError("transposed_returns must be a torch.Tensor")
    if returns.ndim != 2 or min(returns.shape) <= 0:
        raise ValueError("transposed_returns must have two nonempty dimensions")
    if returns.dtype not in (torch.float32, torch.float64):
        raise TypeError("transposed_returns must have dtype float32 or float64")
    if not bool(torch.isfinite(returns).all().item()):
        raise ValueError("transposed_returns must contain only finite values")
    return returns.shape[0], returns.shape[1]


def _validate_support_mask(value: int, strategy_count: int) -> int:
    """Return an in-range arbitrary-width Python support mask."""
    if isinstance(value, bool):
        raise TypeError("support_mask must be an integer")
    try:
        mask = operator.index(value)
    except TypeError as exc:
        raise TypeError("support_mask must be an integer") from exc
    if not 0 <= mask < 1 << strategy_count:
        raise ValueError("support_mask contains an out-of-range strategy")
    return mask


def _representable_minimum(
    minimum: float, transposed_returns: torch.Tensor
) -> float | None:
    """Round the support floor upward in the calculation dtype."""
    value = torch.tensor(
        minimum, device=transposed_returns.device, dtype=transposed_returns.dtype
    )
    if not bool(torch.isfinite(value).item()):
        return None
    if float(value.item()) < minimum:
        infinity = torch.full_like(value, torch.inf)
        value = torch.nextafter(value, infinity)
    result = float(value.item())
    return result if math.isfinite(result) and result > 0.0 else None


def _validate_strategy_count(value: int) -> int:
    """Return a positive strategy count."""
    if isinstance(value, bool):
        raise TypeError("strategy_count must be an integer")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise TypeError("strategy_count must be an integer") from exc
    if count <= 0:
        raise ValueError("strategy_count must be positive")
    return count


def _strategy_index(value: int, strategy_count: int, name: str) -> int:
    """Return one validated strategy index."""
    if isinstance(value, bool):
        raise TypeError(f"{name} indices must be integers")
    try:
        index = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} indices must be integers") from exc
    if not 0 <= index < strategy_count:
        raise ValueError(f"{name} index is out of range")
    return index


def _positive_integer(value: int, name: str) -> int:
    """Return a positive integer."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _finite_nonnegative(value: float, name: str) -> float:
    """Return a finite nonnegative real number."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: float, name: str) -> float:
    """Return a finite positive real number."""
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _budget_rounding_tolerance(maximum: float) -> float:
    """Return a small CPU-side allowance for support-budget comparisons."""
    return _BOUND_ROUNDING_ULPS * math.ulp(max(1.0, maximum))
