"""Orchestration for ADD-constrained Kelly shortcuts."""

# Its bootstrap setup intentionally mirrors the continuous-winner diagnostic.
# pylint: disable=duplicate-code

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._add_constrained_kelly import (
    AddConstrainedKellyResult,
    cast_result_to_feasible_dtype,
    try_add_constrained_kelly,
)
from ._add_branch_and_bound import AddBranchAndBoundOutcome
from ._capital_allocation_diagnostics import (
    CapitalAllocationDiagnostics,
    finalize_allocation_diagnostics,
    prepare_diagnostics,
    record_algorithm,
    resolve_grid_bootstrap_parameters,
)
from ._capital_allocation_memory import validate_bootstrap_diagnostic_memory
from ._strategy_selection import StrategySelectionConstraints

BootstrapCalculator = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
BootstrapSampler = Callable[[int, int, int, torch.device, int | None], torch.Tensor]
BranchSolver = Callable[..., AddBranchAndBoundOutcome | None]
GridStepResolver = Callable[[float], int]


@dataclass(frozen=True)
class AddFastPathOutcome:
    """A fast allocation or bootstrap state reusable by the grid fallback."""

    allocation: torch.Tensor | None
    bootstrap_runs: int
    bootstrap_length: int
    percentile: float
    bootstrap_diagnostics_enabled: bool
    bootstrap_needed: bool
    track_bootstrap_drawdown: bool
    bootstrap_indices: torch.Tensor | None


@dataclass(frozen=True)
class AddBranchPathOutcome:
    """A completed result or spatial-pilot state reusable by the grid."""

    allocation: torch.Tensor | None
    search_outcome: AddBranchAndBoundOutcome | None
    pilot_ran: bool


@torch.no_grad()
def try_add_fast_path(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    transposed_returns: torch.Tensor,
    raw_allocation: torch.Tensor | None,
    max_total_allocation: float,
    add_limit: float,
    selection_constraints: StrategySelectionConstraints | None,
    target_dtype: torch.dtype,
    bootstrap_on: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
    percentile: float,
    bootstrap_seed: int | None,
    approximate_add_search: bool,
    allocation_tolerance: float,
    diagnostics: CapitalAllocationDiagnostics | None,
    clique_count: int,
    sampler: BootstrapSampler,
    bootstrap_calculator: BootstrapCalculator,
) -> AddFastPathOutcome:
    """Try exact/approximate ADD routes and retain shared bootstrap state."""
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
            clique_count,
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

    result = try_add_constrained_kelly(
        transposed_returns,
        raw_allocation,
        max_total_allocation,
        add_limit,
        selection_constraints,
        bootstrap_indices,
        percentile_value,
        approximate_add_search,
        allocation_tolerance,
        bootstrap_drawdown_calculator=bootstrap_calculator,
    )
    allocation = None
    if result is not None:
        result = cast_result_to_feasible_dtype(
            result,
            target_dtype,
            transposed_returns,
            add_limit,
            bootstrap_indices,
            percentile_value,
            allocation_tolerance,
            bootstrap_drawdown_calculator=bootstrap_calculator,
        )
        record_algorithm(diagnostics, result.algorithm)
        allocation = finalize_allocation_diagnostics(
            result.allocation,
            diagnostics,
            bootstrap_indices if diagnostics_enabled else None,
            transposed_returns,
            percentile_value,
            result.bootstrap_average_drawdown,
            bootstrap_calculator,
        )
    return AddFastPathOutcome(
        allocation,
        runs,
        length,
        percentile_value,
        diagnostics_enabled,
        bootstrap_needed,
        track_drawdown,
        bootstrap_indices,
    )


@torch.no_grad()
def try_add_branch_path(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    transposed_returns: torch.Tensor,
    raw_allocation: torch.Tensor | None,
    max_total_allocation: float,
    add_limit: float,
    maximal_cliques: list[tuple[int, ...]],
    selection_constraints: StrategySelectionConstraints | None,
    grid_increment: float | None,
    target_dtype: torch.dtype,
    allocation_tolerance: float,
    diagnostics: CapitalAllocationDiagnostics | None,
    fast_outcome: AddFastPathOutcome,
    branch_solver: BranchSolver,
    grid_step_resolver: GridStepResolver,
    bootstrap_calculator: BootstrapCalculator,
) -> AddBranchPathOutcome:
    """Run the adaptive spatial pilot and finalize a completed result."""
    grid_steps = None
    if grid_increment is not None:
        try:
            grid_steps = grid_step_resolver(grid_increment)
        except (TypeError, ValueError):
            # A completed spatial search does not use the grid setting. Normal
            # grid validation reports the error only if the pilot falls back.
            pass
    outcome = branch_solver(
        transposed_returns=transposed_returns,
        raw_allocation=raw_allocation,
        max_total_allocation=max_total_allocation,
        add_limit=add_limit,
        maximal_cliques=maximal_cliques,
        selection_constraints=selection_constraints,
        bootstrap_indices=fast_outcome.bootstrap_indices,
        percentile=fast_outcome.percentile,
        allocation_tolerance=allocation_tolerance,
        grid_steps=grid_steps,
        bootstrap_drawdown_calculator=bootstrap_calculator,
    )
    if outcome is None:
        return AddBranchPathOutcome(None, None, True)
    pilot_ran = getattr(outcome, "pilot_nodes", 1) > 0
    if outcome.allocation is None:
        return AddBranchPathOutcome(None, outcome, pilot_ran)

    result = AddConstrainedKellyResult(
        outcome.allocation,
        outcome.bootstrap_average_drawdown,
        False,
        outcome.algorithm,
    )
    result = cast_result_to_feasible_dtype(
        result,
        target_dtype,
        transposed_returns,
        add_limit,
        fast_outcome.bootstrap_indices,
        fast_outcome.percentile,
        allocation_tolerance,
        bootstrap_drawdown_calculator=bootstrap_calculator,
    )
    record_algorithm(diagnostics, result.algorithm)
    allocation = finalize_allocation_diagnostics(
        result.allocation,
        diagnostics,
        (
            fast_outcome.bootstrap_indices
            if fast_outcome.bootstrap_diagnostics_enabled
            else None
        ),
        transposed_returns,
        fast_outcome.percentile,
        result.bootstrap_average_drawdown,
        bootstrap_calculator,
    )
    return AddBranchPathOutcome(allocation, outcome, pilot_ran)


@torch.no_grad()
def initialize_grid_incumbent(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    strategy_count: int,
    target_device: torch.device,
    target_dtype: torch.dtype,
    calculation_dtype: torch.dtype,
    track_bootstrap_drawdown: bool,
    branch_outcome: AddBranchAndBoundOutcome | None,
    transposed_returns: torch.Tensor,
    add_limit: float | None,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float,
    bootstrap_calculator: BootstrapCalculator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Initialize grid winners, preserving a feasible spatial-pilot incumbent."""
    best_score = torch.full(
        (), -torch.inf, device=target_device, dtype=calculation_dtype
    )
    best_allocation = torch.zeros(
        strategy_count, device=target_device, dtype=target_dtype
    )
    best_bootstrap_drawdown = (
        torch.full((), math.nan, device=target_device, dtype=calculation_dtype)
        if track_bootstrap_drawdown
        else None
    )
    if branch_outcome is None:
        return best_score, best_allocation, best_bootstrap_drawdown
    if add_limit is None:
        raise RuntimeError("ADD pilot state exists without a hard ADD limit")

    result = AddConstrainedKellyResult(
        branch_outcome.incumbent_allocation,
        branch_outcome.bootstrap_average_drawdown,
        False,
        branch_outcome.algorithm,
    )
    result = cast_result_to_feasible_dtype(
        result,
        target_dtype,
        transposed_returns,
        add_limit,
        bootstrap_indices,
        percentile,
        allocation_tolerance,
        bootstrap_drawdown_calculator=bootstrap_calculator,
    )
    best_allocation = result.allocation
    calculation_incumbent = best_allocation.to(dtype=calculation_dtype)
    best_score = torch.log1p(calculation_incumbent @ transposed_returns).sum()
    if track_bootstrap_drawdown:
        if result.bootstrap_average_drawdown is None:
            raise RuntimeError("ADD pilot did not retain bootstrap diagnostics")
        best_bootstrap_drawdown = result.bootstrap_average_drawdown
    return best_score, best_allocation, best_bootstrap_drawdown
