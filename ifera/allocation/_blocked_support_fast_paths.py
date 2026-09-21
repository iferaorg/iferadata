"""Orchestration for continuous searches with allocation-triggered blocks."""

# Bootstrap setup intentionally mirrors the unblocked fast-path orchestrator.
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._blocked_support_add import try_blocked_support_add_search
from ._blocked_support_kelly import (
    blocked_returns_for_support,
    try_maximize_blocked_support_kelly,
)
from ._capital_allocation_blocks import AllocationBlocks
from ._capital_allocation_diagnostics import (
    CapitalAllocationDiagnostics,
    finalize_allocation_diagnostics,
    finalize_continuous_diagnostics,
    prepare_diagnostics,
    record_algorithm,
    resolve_grid_bootstrap_parameters,
)
from ._capital_allocation_drawdown import negative_rms_drawdown
from ._capital_allocation_memory import validate_bootstrap_diagnostic_memory
from ._strategy_selection import StrategySelectionConstraints

BootstrapCalculator = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
BootstrapSampler = Callable[[int, int, int, torch.device, int | None], torch.Tensor]
GridStepResolver = Callable[[float], int]


@dataclass(frozen=True)
class BlockedFastPathOutcome:
    """A fast allocation or bootstrap state reusable by the complete grid."""

    allocation: torch.Tensor | None
    bootstrap_runs: int
    bootstrap_length: int
    percentile: float
    bootstrap_diagnostics_enabled: bool
    bootstrap_needed: bool
    track_bootstrap_drawdown: bool
    bootstrap_indices: torch.Tensor | None
    pilot_ran: bool
    bootstrap_prepared: bool


@torch.no_grad()
def try_blocked_fast_path(  # pylint: disable=too-many-arguments,too-many-locals
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    maximal_cliques: list[tuple[int, ...]],
    selection_constraints: StrategySelectionConstraints | None,
    allocation_blocks: AllocationBlocks,
    add_limit: float | None,
    target_dtype: torch.dtype,
    bootstrap_on: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
    percentile: float,
    bootstrap_seed: int | None,
    approximate_add_search: bool,
    allocation_tolerance: float,
    grid_increment: float | None,
    diagnostics: CapitalAllocationDiagnostics | None,
    sampler: BootstrapSampler,
    grid_step_resolver: GridStepResolver,
    bootstrap_calculator: BootstrapCalculator,
) -> BlockedFastPathOutcome:
    """Try all exact supports before requesting one original grid search."""
    raw_outcome = try_maximize_blocked_support_kelly(
        transposed_returns,
        max_total_allocation,
        maximal_cliques,
        selection_constraints,
        allocation_blocks,
        allocation_tolerance,
    )
    if raw_outcome is None:
        return _empty_outcome()

    if add_limit is None:
        allocation = _cast_support_allocation(
            raw_outcome.allocation,
            raw_outcome.support_mask,
            raw_outcome.minimum_active_allocation,
            target_dtype,
            max_total_allocation,
            maximal_cliques,
        )
        if allocation is None:
            return _empty_outcome()
        record_algorithm(diagnostics, raw_outcome.algorithm)
        allocation = finalize_continuous_diagnostics(
            allocation,
            diagnostics,
            bootstrap_on,
            bootstrap_runs,
            bootstrap_length,
            percentile,
            bootstrap_seed,
            transposed_returns,
            len(maximal_cliques),
            allocation_blocks,
        )
        return BlockedFastPathOutcome(
            allocation,
            1,
            1,
            10.0,
            False,
            False,
            False,
            None,
            False,
            False,
        )

    (
        bootstrap_on_value,
        runs,
        length,
        percentile_value,
        seed,
    ) = resolve_grid_bootstrap_parameters(
        bootstrap_on,
        bootstrap_runs,
        bootstrap_length,
        percentile,
        bootstrap_seed,
        diagnostics is not None and diagnostics.request_bootstrap_average_drawdown,
        True,
    )
    diagnostics_enabled, bootstrap_needed, track_drawdown = prepare_diagnostics(
        diagnostics, bootstrap_on_value, True
    )
    bootstrap_indices = None
    if bootstrap_needed:
        strategy_count, time_count = transposed_returns.shape
        validate_bootstrap_diagnostic_memory(
            time_count,
            strategy_count,
            len(maximal_cliques),
            transposed_returns.dtype,
            transposed_returns.device,
            runs,
            length,
        )
        bootstrap_indices = sampler(
            time_count,
            runs,
            length,
            transposed_returns.device,
            seed,
        )

    grid_steps = _optional_grid_steps(grid_increment, grid_step_resolver)
    add_outcome = try_blocked_support_add_search(
        transposed_returns,
        raw_outcome,
        max_total_allocation,
        maximal_cliques,
        allocation_blocks,
        add_limit,
        bootstrap_indices,
        percentile_value,
        approximate_add_search,
        allocation_tolerance,
        grid_steps,
        bootstrap_drawdown_calculator=bootstrap_calculator,
    )
    allocation = None
    cast_bootstrap_drawdown = None
    if add_outcome.completed and add_outcome.allocation is not None:
        allocation = _cast_support_allocation(
            add_outcome.allocation,
            add_outcome.support_mask,
            raw_outcome.minimum_active_allocation,
            target_dtype,
            max_total_allocation,
            maximal_cliques,
        )
        if allocation is not None:
            cast_drawdown = _cast_add_drawdown(
                allocation,
                add_outcome.support_mask,
                transposed_returns,
                allocation_blocks,
                bootstrap_indices,
                percentile_value,
                bootstrap_calculator,
            )
            if bool((cast_drawdown < add_limit).item()):
                allocation = None
            elif bootstrap_indices is not None:
                cast_bootstrap_drawdown = cast_drawdown
        if allocation is not None:
            if add_outcome.algorithm is None:
                raise RuntimeError("a completed blocked ADD search has no algorithm")
            record_algorithm(diagnostics, add_outcome.algorithm)
            allocation = finalize_allocation_diagnostics(
                allocation,
                diagnostics,
                bootstrap_indices if diagnostics_enabled else None,
                transposed_returns,
                percentile_value,
                cast_bootstrap_drawdown,
                bootstrap_calculator,
                allocation_blocks,
            )
    return BlockedFastPathOutcome(
        allocation,
        runs,
        length,
        percentile_value,
        diagnostics_enabled,
        bootstrap_needed,
        track_drawdown,
        bootstrap_indices,
        add_outcome.pilot_ran,
        True,
    )


def _cast_support_allocation(
    allocation: torch.Tensor,
    support_mask: int,
    minimum: float,
    target_dtype: torch.dtype,
    maximum: float,
    maximal_cliques: list[tuple[int, ...]],
) -> torch.Tensor | None:
    """Cast a result upward at its support floor and recheck capital budgets."""
    cast = allocation.to(dtype=target_dtype)
    if support_mask:
        support = torch.tensor(
            [
                bool(support_mask & (1 << strategy))
                for strategy in range(allocation.numel())
            ],
            device=allocation.device,
            dtype=torch.bool,
        )
        target_minimum = torch.tensor(
            minimum, device=allocation.device, dtype=target_dtype
        )
        if not bool(torch.isfinite(target_minimum).item()):
            return None
        calculation_minimum = float(target_minimum.to(dtype=allocation.dtype).item())
        if calculation_minimum < minimum:
            target_minimum = torch.nextafter(
                target_minimum, torch.full_like(target_minimum, torch.inf)
            )
        if not bool((target_minimum > 0.0).item()):
            return None
        cast[support] = torch.maximum(cast[support], target_minimum)
        cast[~support] = 0.0

    calculation = cast.to(dtype=allocation.dtype)
    rounding = 32.0 * torch.finfo(calculation.dtype).eps * max(1.0, maximum)
    if bool(torch.any(calculation < 0.0).item()) or bool(
        torch.any(calculation > maximum + rounding).item()
    ):
        return None
    for clique in maximal_cliques:
        indices = torch.tensor(clique, device=allocation.device, dtype=torch.int64)
        if float(calculation.index_select(0, indices).sum().item()) > (
            maximum + rounding
        ):
            return None
    return cast


def _cast_add_drawdown(  # pylint: disable=too-many-arguments
    allocation: torch.Tensor,
    support_mask: int,
    transposed_returns: torch.Tensor,
    allocation_blocks: AllocationBlocks,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_calculator: BootstrapCalculator,
) -> torch.Tensor:
    """Return output-dtype ADD so rounding can be checked and reported once."""
    effective_returns = blocked_returns_for_support(
        transposed_returns, allocation_blocks, support_mask
    )
    portfolio_returns = (
        allocation.to(dtype=transposed_returns.dtype) @ effective_returns
    )
    if not bool(torch.all(portfolio_returns > -1.0).item()):
        return torch.full(
            (), -torch.inf, device=allocation.device, dtype=transposed_returns.dtype
        )
    if bootstrap_indices is not None:
        drawdown = bootstrap_calculator(
            portfolio_returns.unsqueeze(0), bootstrap_indices, percentile
        )[0]
    else:
        drawdown = negative_rms_drawdown(torch.log1p(portfolio_returns), dim=0)
    return drawdown


def _optional_grid_steps(
    grid_increment: float | None, resolver: GridStepResolver
) -> int | None:
    """Resolve a valid grid only when it may guide a spatial pilot."""
    if grid_increment is None:
        return None
    try:
        return resolver(grid_increment)
    except (TypeError, ValueError):
        return None


def _empty_outcome() -> BlockedFastPathOutcome:
    """Return unsupported state without prematurely validating grid controls."""
    return BlockedFastPathOutcome(
        None, 1, 1, 10.0, False, False, False, None, False, False
    )
