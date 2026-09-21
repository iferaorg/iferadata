"""Hard-ADD searches over allocation-blocking exact-support regions."""

# Support-wise numerical calls intentionally share argument-forwarding shapes
# with the ordinary ADD orchestration.
# pylint: disable=duplicate-code

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._add_branch_and_bound import try_exact_support_add_branch_and_bound
from ._blocked_support_kelly import (
    BlockedSupportKellyCandidate,
    BlockedSupportKellyOutcome,
    blocked_returns_for_support,
)
from ._capital_allocation_blocks import AllocationBlocks
from ._capital_allocation_drawdown import (
    bootstrap_average_drawdown,
    negative_rms_drawdown,
)

MAX_EXACT_SUPPORT_ADD_BRANCH_SUPPORTS = 32
BLOCKED_SUPPORT_ADD_PRECHECK_ALGORITHM = (
    "Blocking-aware exact-support raw Kelly + exact ADD feasibility precheck"
)
BLOCKED_SUPPORT_APPROXIMATE_ADD_ALGORITHM = (
    "Blocking-aware exact-support approximate shifted-Kelly-ray ADD bisection"
)
BLOCKED_SUPPORT_ADD_BRANCH_ALGORITHM = (
    "Blocking-aware exact-support adaptive spatial ADD branch-and-bound"
)
BootstrapDrawdownCalculator = Callable[
    [torch.Tensor, torch.Tensor, float], torch.Tensor
]


@dataclass(frozen=True)
class BlockedSupportAddOutcome:
    """A completed blocked-support ADD result or a request for the full grid."""

    allocation: torch.Tensor | None
    support_mask: int
    bootstrap_average_drawdown: torch.Tensor | None
    algorithm: str | None
    completed: bool
    pilot_ran: bool


@torch.no_grad()
def try_blocked_support_add_search(  # pylint: disable=too-many-arguments,too-many-locals
    transposed_returns: torch.Tensor,
    raw_outcome: BlockedSupportKellyOutcome,
    max_total_allocation: float,
    maximal_cliques: list[tuple[int, ...]],
    allocation_blocks: AllocationBlocks,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    approximate_add_search: bool,
    allocation_tolerance: float,
    grid_steps: int | None,
    *,
    maximum_branch_supports: int = MAX_EXACT_SUPPORT_ADD_BRANCH_SUPPORTS,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator = (
        bootstrap_average_drawdown
    ),
) -> BlockedSupportAddOutcome:
    """Optimize growth with ADD inside every already-solved exact support.

    The global raw-Kelly winner is first checked because, when feasible, it is
    also the exact constrained winner. A binding constraint uses either the
    opt-in shifted-ray approximation or a lower-bounded spatial search for
    every non-cash support. All drawdown calculations receive the caller's one
    shared bootstrap-index tensor.
    """
    raw_effective_returns = blocked_returns_for_support(
        transposed_returns, allocation_blocks, raw_outcome.support_mask
    )
    raw_drawdown = _allocation_drawdown(
        raw_outcome.allocation,
        raw_effective_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )
    if bool((raw_drawdown >= add_limit).item()):
        return _completed_outcome(
            raw_outcome.allocation,
            raw_outcome.support_mask,
            raw_drawdown,
            bootstrap_indices,
            BLOCKED_SUPPORT_ADD_PRECHECK_ALGORITHM,
            False,
        )

    if approximate_add_search:
        return _approximate_support_rays(
            transposed_returns,
            raw_outcome,
            allocation_blocks,
            add_limit,
            bootstrap_indices,
            percentile,
            allocation_tolerance,
            bootstrap_drawdown_calculator,
        )

    noncash_support_count = len(raw_outcome.candidates) - 1
    if noncash_support_count > maximum_branch_supports:
        return _fallback_outcome(False)
    return _spatial_support_searches(
        transposed_returns,
        raw_outcome,
        max_total_allocation,
        maximal_cliques,
        allocation_blocks,
        add_limit,
        bootstrap_indices,
        percentile,
        allocation_tolerance,
        grid_steps,
        bootstrap_drawdown_calculator,
    )


def _approximate_support_rays(  # pylint: disable=too-many-arguments
    transposed_returns: torch.Tensor,
    raw_outcome: BlockedSupportKellyOutcome,
    allocation_blocks: AllocationBlocks,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator,
) -> BlockedSupportAddOutcome:
    """Compare floor-to-Kelly segments without leaving their exact supports."""
    best = raw_outcome.candidates[0]
    best_drawdown = torch.zeros(
        (), device=transposed_returns.device, dtype=transposed_returns.dtype
    )
    for raw_candidate in raw_outcome.candidates[1:]:
        effective_returns = blocked_returns_for_support(
            transposed_returns, allocation_blocks, raw_candidate.support_mask
        )
        candidate, drawdown = _constrain_shifted_kelly_ray(
            raw_candidate,
            effective_returns,
            raw_outcome.minimum_active_allocation,
            add_limit,
            bootstrap_indices,
            percentile,
            allocation_tolerance,
            bootstrap_drawdown_calculator,
        )
        if candidate is None or drawdown is None:
            continue
        score = _mean_log_growth(candidate, effective_returns)
        if score > best.mean_log_growth:
            best = BlockedSupportKellyCandidate(
                raw_candidate.support_mask, candidate, score
            )
            best_drawdown = drawdown
    return _completed_outcome(
        best.allocation,
        best.support_mask,
        best_drawdown,
        bootstrap_indices,
        BLOCKED_SUPPORT_APPROXIMATE_ADD_ALGORITHM,
        False,
    )


def _constrain_shifted_kelly_ray(  # pylint: disable=too-many-arguments
    raw_candidate: BlockedSupportKellyCandidate,
    effective_returns: torch.Tensor,
    minimum: float,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Bisect the segment from the support floor to its raw-Kelly winner."""
    raw_drawdown = _allocation_drawdown(
        raw_candidate.allocation,
        effective_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )
    if bool((raw_drawdown >= add_limit).item()):
        return raw_candidate.allocation, raw_drawdown

    support = _support_tensor(
        raw_candidate.support_mask,
        raw_candidate.allocation.numel(),
        raw_candidate.allocation.device,
    )
    lower_allocation = torch.zeros_like(raw_candidate.allocation)
    lower_allocation[support] = minimum
    lower_drawdown = _allocation_drawdown(
        lower_allocation,
        effective_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )
    if bool((lower_drawdown < add_limit).item()):
        return None, None

    direction = raw_candidate.allocation - lower_allocation
    maximum_width = float(torch.amax(torch.abs(direction)).item())
    lower_scale = 0.0
    upper_scale = 1.0
    feasible_allocation = lower_allocation
    feasible_drawdown = lower_drawdown
    for _ in range(_bisection_iterations(maximum_width, allocation_tolerance)):
        midpoint = (lower_scale + upper_scale) * 0.5
        candidate = lower_allocation + midpoint * direction
        drawdown = _allocation_drawdown(
            candidate,
            effective_returns,
            bootstrap_indices,
            percentile,
            bootstrap_drawdown_calculator,
        )
        if bool((drawdown >= add_limit).item()):
            lower_scale = midpoint
            feasible_allocation = candidate
            feasible_drawdown = drawdown
        else:
            upper_scale = midpoint
    return feasible_allocation, feasible_drawdown


def _spatial_support_searches(  # pylint: disable=too-many-arguments,too-many-locals
    transposed_returns: torch.Tensor,
    raw_outcome: BlockedSupportKellyOutcome,
    max_total_allocation: float,
    maximal_cliques: list[tuple[int, ...]],
    allocation_blocks: AllocationBlocks,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float,
    grid_steps: int | None,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator,
) -> BlockedSupportAddOutcome:
    """Run the exact lower-bounded spatial solver for every non-cash support."""
    best = raw_outcome.candidates[0]
    best_drawdown = torch.zeros(
        (), device=transposed_returns.device, dtype=transposed_returns.dtype
    )
    pilot_ran = False
    strategy_count = transposed_returns.shape[0]
    for raw_candidate in raw_outcome.candidates[1:]:
        support = _support_tensor(
            raw_candidate.support_mask, strategy_count, transposed_returns.device
        )
        lower = torch.zeros_like(raw_candidate.allocation)
        lower[support] = raw_outcome.minimum_active_allocation
        upper = torch.zeros_like(raw_candidate.allocation)
        upper[support] = max_total_allocation
        effective_returns = blocked_returns_for_support(
            transposed_returns, allocation_blocks, raw_candidate.support_mask
        )
        outcome = try_exact_support_add_branch_and_bound(
            transposed_returns=effective_returns,
            root_lower=lower,
            root_upper=upper,
            max_total_allocation=max_total_allocation,
            add_limit=add_limit,
            maximal_cliques=maximal_cliques,
            bootstrap_indices=bootstrap_indices,
            percentile=percentile,
            allocation_tolerance=allocation_tolerance,
            grid_steps=grid_steps,
            bootstrap_drawdown_calculator=bootstrap_drawdown_calculator,
        )
        pilot_ran |= outcome.pilot_nodes > 0
        if not outcome.completed:
            return _fallback_outcome(pilot_ran)
        if (
            outcome.allocation is not None
            and outcome.incumbent_mean_log_growth > best.mean_log_growth
        ):
            best = BlockedSupportKellyCandidate(
                raw_candidate.support_mask,
                outcome.allocation,
                outcome.incumbent_mean_log_growth,
            )
            best_drawdown = (
                outcome.bootstrap_average_drawdown
                if outcome.bootstrap_average_drawdown is not None
                else _allocation_drawdown(
                    outcome.allocation,
                    effective_returns,
                    bootstrap_indices,
                    percentile,
                    bootstrap_drawdown_calculator,
                )
            )
    return _completed_outcome(
        best.allocation,
        best.support_mask,
        best_drawdown,
        bootstrap_indices,
        BLOCKED_SUPPORT_ADD_BRANCH_ALGORITHM,
        pilot_ran,
    )


def _allocation_drawdown(
    allocation: torch.Tensor,
    transposed_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator,
) -> torch.Tensor:
    """Calculate historical or shared-bootstrap ADD for one fixed support."""
    portfolio_returns = allocation @ transposed_returns
    if not bool(torch.all(portfolio_returns > -1.0).item()):
        return torch.full(
            (), -torch.inf, device=allocation.device, dtype=allocation.dtype
        )
    if bootstrap_indices is not None:
        return bootstrap_drawdown_calculator(
            portfolio_returns.unsqueeze(0), bootstrap_indices, percentile
        )[0]
    return negative_rms_drawdown(torch.log1p(portfolio_returns), dim=0)


def _mean_log_growth(
    allocation: torch.Tensor, transposed_returns: torch.Tensor
) -> float:
    """Return normalized log growth for candidate comparison."""
    portfolio_returns = allocation @ transposed_returns
    if not bool(torch.all(portfolio_returns > -1.0).item()):
        return -math.inf
    return float(torch.log1p(portfolio_returns).mean().item())


def _support_tensor(
    support_mask: int, strategy_count: int, device: torch.device
) -> torch.Tensor:
    """Materialize a fixed support as a Boolean device tensor."""
    return torch.tensor(
        [bool(support_mask & (1 << strategy)) for strategy in range(strategy_count)],
        device=device,
        dtype=torch.bool,
    )


def _bisection_iterations(maximum_width: float, allocation_tolerance: float) -> int:
    """Return enough shifted-ray iterations for the allocation resolution."""
    if maximum_width <= allocation_tolerance:
        return 0
    return math.ceil(math.log2(maximum_width / allocation_tolerance))


def _completed_outcome(
    allocation: torch.Tensor,
    support_mask: int,
    drawdown: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    algorithm: str,
    pilot_ran: bool,
) -> BlockedSupportAddOutcome:
    """Build a completed support-wise ADD result."""
    return BlockedSupportAddOutcome(
        allocation,
        support_mask,
        drawdown if bootstrap_indices is not None else None,
        algorithm,
        True,
        pilot_ran,
    )


def _fallback_outcome(pilot_ran: bool) -> BlockedSupportAddOutcome:
    """Ask the caller to run the original complete allocation grid."""
    return BlockedSupportAddOutcome(None, 0, None, None, False, pilot_ran)
