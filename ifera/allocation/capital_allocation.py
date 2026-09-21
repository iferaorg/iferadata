"""Find capital allocations with continuous Kelly and modified-Kelly searches."""

# Public imports below are API re-exports rather than local dependencies.
# pylint: disable=duplicate-code,too-many-lines,unused-import

from __future__ import annotations

import math
from collections.abc import Iterator
import torch

from ._add_constrained_kelly import (
    validate_approximate_add_search as _validate_approximate_add_search,
)
from ._add_branch_and_bound import (
    PILOT_GRID_FALLBACK_ALGORITHM as _PILOT_GRID_FALLBACK_ALGORITHM,
    try_add_branch_and_bound as _try_add_branch_and_bound,
)
from ._capital_allocation_facades import (
    find_optimal_capital_allocation_from_csv,  # pylint: disable=unused-import
    walk_forward_capital_allocation_from_csv,  # pylint: disable=unused-import
)
from ._capital_allocation_fast_paths import (
    initialize_grid_incumbent as _initialize_grid_incumbent,
    try_add_branch_path as _try_add_branch_path,
    try_add_fast_path as _try_add_fast_path,
)
from ._capital_allocation_candidates import (
    constrained_allocation_unit_batches as _constrained_allocation_unit_batches,
)
from ._capital_allocation_blocks import (
    AllocationBlocks as _AllocationBlocks,
    BlockList,
    build_allocation_blocks as _build_allocation_blocks,
    effective_portfolio_returns as _effective_portfolio_returns,
    normalize_block_list as _normalize_block_list,
)
from ._blocked_support_fast_paths import (
    try_blocked_fast_path as _try_blocked_fast_path,
)
from ._capital_allocation_diagnostics import (
    CapitalAllocationDiagnostics,
    finalize_allocation_diagnostics as _finalize_allocation_diagnostics,
    finalize_continuous_diagnostics as _finalize_continuous_diagnostics,
    prepare_diagnostics as _prepare_diagnostics,
    record_algorithm as _record_algorithm,
    resolve_grid_bootstrap_parameters as _resolve_grid_bootstrap_parameters,
    sample_bootstrap_indices as _sample_bootstrap_indices,
)
from ._capital_allocation_drawdown import (
    bootstrap_average_drawdown as _bootstrap_average_drawdown,
    negative_rms_drawdown as _negative_rms_drawdown,
)
from ._capital_allocation_memory import (
    candidate_batch_size as _candidate_batch_size,
    search_candidate_batch_size as _search_candidate_batch_size,
)
from ._capital_allocation_parameters import (
    SUPPORTED_DTYPES as _SUPPORTED_DTYPES,
    resolve_allocation_unit_limit as _resolve_allocation_unit_limit,
    resolve_allocation_tolerance as _resolve_allocation_tolerance,
    resolve_calculation_dtype as _resolve_calculation_dtype,
    resolve_device as _resolve_device,
    resolve_grid_steps as _resolve_grid_steps,
    validate_add_limit as _validate_add_limit,
    validate_alpha as _validate_alpha,
    validate_max_total_allocation as _validate_max_total_allocation,
    validate_returns as _validate_returns,
)
from ._capital_allocation_refinement import (
    clique_unit_limits as _refinement_clique_unit_limits,
    grid_lower_bounds as _refinement_grid_lower_bounds,
    units_to_allocations as _refinement_units_to_allocations,
    validate_parameters as _validate_refinement_parameters,
    validate_final_increment as _validate_final_refinement_increment,
)
from ._kelly_selection import (
    maximize_kelly_with_selection as _maximize_kelly_with_selection,
)
from ._overlap_constraints import (
    clique_indices_by_strategy as _clique_indices_by_strategy,
    find_maximal_overlap_cliques as _find_maximal_overlap_cliques,
)
from ._strategy_selection import (
    StrategySelectionConstraints,
    resolve_strategy_selection_constraints as _resolve_strategy_selection_constraints,
)

_MAX_CANDIDATES_PER_BATCH = 1_000_000
_CONTINUOUS_KELLY_ALGORITHM = "Continuous raw Kelly (away-step Frank-Wolfe)"
_RAW_KELLY_GRID_ALGORITHM = "Raw-Kelly allocation grid fallback"
_ADD_GRID_ALGORITHM = "ADD-constrained allocation grid search"
_MODIFIED_GRID_ALGORITHM = "Drawdown-penalized allocation grid search"
_MODIFIED_ADD_GRID_ALGORITHM = "Drawdown-penalized ADD-constrained grid search"
# Torch's public stubs omit the compiler configuration context decorator.
_COMPILER_CONFIG_PATCH = getattr(torch.compiler.config, "patch")


@torch.no_grad()
def find_optimal_capital_allocation(  # pylint: disable=too-many-branches,too-many-statements
    returns: torch.Tensor,
    alpha: float,
    grid_increment: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    bootstrap_on: bool = False,
    bootstrap_runs: int = 1024,
    bootstrap_length: int = 256,
    percentile: float = 10.0,
    max_total_allocation: float = 1.0,
    refinement_runs: int = 0,
    refinement_divisor: float = 2.0,
    diagnostics: CapitalAllocationDiagnostics | None = None,
    ADD_limit: float | None = None,  # pylint: disable=invalid-name
    max_strategies: int | None = None,
    min_correlation: float | None = None,
    max_correlation: float | None = None,
    bootstrap_seed: int | None = None,
    approximate_add_search: bool = False,
    allocation_tolerance: float | None = None,
    block_list: BlockList | None = None,
) -> torch.Tensor:
    """Return the best long-only allocation under overlap capital constraints.

    ``returns`` must have shape ``(time, strategies)`` and contain simple daily
    returns on risk. NaN marks a day on which a strategy is inactive and is
    replaced by zero for the objective calculation. Rows with no active return
    should be included so their time underwater contributes to the drawdown
    penalty.

    For every candidate allocation ``f``, the function maximizes

    ``sum_t(log(1 + returns[t] @ f)) + alpha * T * log(1 + ADD(f))``,

    where ``ADD`` is the negative root-mean-square drawdown of the compounded
    portfolio equity curve. Initial wealth is included when finding each day's
    running peak. Unallocated capital is treated as cash with zero return. When
    bootstrapping is enabled, ADD is the requested lower percentile across
    fixed-length resampled paths. Each sampled day selects a whole time row
    with replacement, preserving contemporaneous strategy returns, and the
    same paths are used for every candidate. Growth always uses the original
    series.

    ``ADD_limit`` rejects candidates below an inclusive ADD floor. It uses
    historical ADD when bootstrapping is off and the selected percentile ADD
    when bootstrapping is on, independently of ``alpha``.

    Two strategies are disjoint when they have no day on which both returns are
    non-NaN. An allocation must sum to at most ``max_total_allocation`` within
    every maximal clique of pairwise non-disjoint strategies. Thus ordinary
    overlapping strategies share the capital budget, while disjoint strategies
    may reuse it and the global sum of allocations may exceed that budget.

    Optional strategy-selection limits constrain the positive support of an
    allocation. ``max_strategies`` caps its size, while every pair in the
    support must satisfy the inclusive correlation bounds. Correlations are
    calculated after replacing inactive NaNs with zero. Undefined correlations
    are incompatible whenever either bound is specified. Candidate prefixes
    are pruned before scoring, so each valid allocation is evaluated once. As
    with allocation-vector refinement generally, local refinements follow only
    the current winner; they do not independently refine every possible support.

    ``block_list`` contains ``(day_idx, blocked, blocked_by)`` triples. When a
    candidate gives ``blocked_by`` any exactly nonzero allocation, ``blocked``
    contributes zero return on that day. Multiple blockers for the same
    strategy-day are combined with OR semantics. Because activation changes
    discontinuously at exactly zero, a nonempty block list uses the grid-search
    route; continuous Kelly, fixed-ray, and spatial ADD solvers are not valid
    across those support boundaries.

    When ``alpha`` is zero and ``ADD_limit`` is unset, the objective is concave.
    That raw Kelly case is solved continuously with an away-step Frank--Wolfe
    method, including exact branch-and-bound for active-support constraints.
    It supports a simplex, fully disjoint strategies, and disjoint unions of
    complete-overlap components. Intersecting maximal cliques retain the grid fallback.
    Grid and refinement settings are ignored when the continuous route succeeds.
    Bootstrap affects only a requested diagnostic for the final allocation.
    With a hard ADD limit, the continuous winner is first tested for feasibility;
    when it passes, it remains the exact global winner. Problems whose feasible
    support has at most one strategy are solved exactly by comparing vectorized
    scalar Kelly and ADD bisections. ``approximate_add_search=True`` can instead
    scale a multidimensional full-Kelly winner along its fixed allocation ray.
    Otherwise, a binding multidimensional ADD constraint starts with a bounded
    spatial branch-and-bound pilot. Concavity tangents bound growth and interval
    paths bound historical or shared-bootstrap ADD over each allocation box. A
    cheap grid-size estimate plus observed pilot pruning selects continuation
    only when it is likely to beat the vectorized grid; large bootstrap workloads
    bypass the serial pilot. A completed spatial search may therefore make the
    grid optional, while an unpromising pilot reuses its feasible incumbent in
    the full grid. Drawdown-penalized objectives retain the grid search.

    For grid searches, the requested increment is reduced, when necessary, to the largest
    reciprocal of an integer that does not exceed it. For example, ``0.3``
    becomes ``0.25``. Valid integer allocation points are generated directly
    under the overlap-clique budgets and evaluated in vectorized batches,
    without constructing an invalid Cartesian product. If the maximum
    allocation is not on the grid, the largest grid point below it is used.

    Optional refinement runs search progressively smaller local grids. Each
    run divides the preceding increment by ``refinement_divisor`` while keeping
    the initial number of points per strategy. Its per-strategy window is
    centered on the best preceding allocation when possible and shifted at
    zero or ``max_total_allocation`` so it is never truncated. The preceding
    optimum is always one of the new points. After a strict improvement, that
    refinement level is searched again at the same increment around the new
    optimum, until a pass leaves the allocation unchanged. This can traverse
    multiple local windows, but it does not guarantee the global fine-grid
    optimum across a score valley. Points shared with an earlier grid are
    deliberately rescored: their fraction falls exponentially with the strategy
    count, while caching them would add synchronization, memory, and
    floating-point lattice bookkeeping. The fixed point count is per strategy;
    feasible vector counts can still vary as local windows and clique budgets
    change.

    Args:
        returns: Two-dimensional float16, bfloat16, float32, or float64 tensor
            shaped ``(time, strategies)``.
        alpha: Finite, nonnegative weight of the drawdown penalty.
        grid_increment: Requested allocation increment in ``(0, 1]`` for a
            grid search. It is optional when a continuous raw-Kelly or spatial
            ADD search completes without the grid.
        device: Evaluation and output device. Defaults to ``cuda:0`` when CUDA
            is available and to CPU otherwise.
        dtype: Float16, bfloat16, float32, or float64 dtype of the returned
            allocation. Defaults to the dtype of ``returns``. Objective
            calculations use the common dtype of the returns and allocations,
            promoted to at least float32.
        bootstrap_on: Whether to bootstrap the ADD calculation. Defaults to
            ``False``.
        bootstrap_runs: Number of resampled paths. Defaults to ``1024``.
        bootstrap_length: Number of sampled days in every bootstrap path,
            independent of the original series length. Defaults to ``256``.
        percentile: Lower percentile of the bootstrap ADD distribution, in
            percent from ``0`` to ``100``. Defaults to ``10.0``. Since ADD is
            nonpositive, a lower percentile is more pessimistic.
        max_total_allocation: Finite, nonnegative allocation cap for every
            overlap clique. Values above ``1.0`` allow margin or other leverage.
            Defaults to ``1.0``.
        refinement_runs: Number of progressively finer local search levels
            after the initial search. Each level is recentered until its best
            allocation is unchanged. Defaults to ``0``.
        refinement_divisor: Finite factor greater than ``1`` by which the grid
            increment is divided for every refinement. Defaults to ``2.0``.
        diagnostics: Optional mutable result populated with the winning
            candidate's bootstrap ADD. Defaults to ``None``.
        ADD_limit: Optional inclusive ADD floor in ``[-1, 0]``. For example,
            ``-0.1`` requires ADD of negative ten percent or better.
        max_strategies: Optional positive limit on nonzero strategy allocations.
        min_correlation: Optional inclusive minimum pairwise correlation in
            ``[-1, 1]`` for strategies with nonzero allocations.
        max_correlation: Optional inclusive maximum pairwise correlation in
            ``[-1, 1]`` for strategies with nonzero allocations.
        bootstrap_seed: Optional nonnegative int64 seed. Reusing it regenerates
            the same bootstrap row indices without retaining the index tensor.
        approximate_add_search: Whether an otherwise grid-bound ADD constraint
            may use the approximate full-Kelly fixed-ray bisection. Defaults to
            ``False``.
        allocation_tolerance: Optional positive allocation-space stopping
            resolution shared by continuous Kelly and spatial ADD searches.
            Frank--Wolfe uses it as a maximum final-update coordinate movement;
            branch-and-bound uses it as a maximum unresolved box width. These
            are practical stopping rules, not optimizer-distance certificates
            for flat or nonconvex feasible regions. Its dtype-specific default
            preserves the previous solver precision.
        block_list: Optional sequence of ``(day_idx, blocked, blocked_by)``
            integer triples. Indices refer to the corresponding dimensions of
            ``returns``. A nonzero allocation to ``blocked_by`` suppresses the
            blocked strategy's return on that day.

    Returns:
        A one-dimensional tensor containing the optimal strategy allocations.

    Raises:
        TypeError: If a tensor dtype or parameter type is unsupported.
        ValueError: If an input value or shape is invalid, or if the grid cannot
            be safely indexed or evaluated within the search memory budget.
    """
    _validate_returns(returns)
    normalized_blocks = _normalize_block_list(
        block_list, returns.shape[0], returns.shape[1]
    )
    alpha_value, add_limit_value = _validate_alpha(alpha), _validate_add_limit(
        ADD_limit
    )
    approximate_add_search_value = _validate_approximate_add_search(
        approximate_add_search
    )
    max_total_allocation_value = _validate_max_total_allocation(max_total_allocation)
    target_dtype = returns.dtype if dtype is None else dtype
    if (
        not isinstance(target_dtype, torch.dtype)
        or target_dtype not in _SUPPORTED_DTYPES
    ):
        raise TypeError("dtype must be float16, bfloat16, float32, or float64")

    target_device = _resolve_device(device)
    selection_constraints = _resolve_strategy_selection_constraints(
        returns,
        max_strategies,
        min_correlation,
        max_correlation,
        target_device,
    )
    calculation_dtype = _resolve_calculation_dtype(returns.dtype, target_dtype)
    allocation_tolerance_value = _resolve_allocation_tolerance(
        allocation_tolerance, calculation_dtype
    )
    strategy_count = returns.shape[1]
    calculation_returns = returns.to(device=target_device, dtype=calculation_dtype)
    maximal_cliques = _find_maximal_overlap_cliques(~torch.isnan(calculation_returns))
    calculation_returns = torch.nan_to_num(calculation_returns, nan=0.0)
    transposed_returns = calculation_returns.transpose(0, 1).contiguous()
    del calculation_returns
    allocation_blocks = _build_allocation_blocks(
        normalized_blocks,
        strategy_count,
        target_device,
        calculation_dtype,
    )

    # Diagnostics themselves remain relevant in every mode, even when no
    # drawdown calculation is requested. Grid, bootstrap, and refinement
    # controls are intentionally not validated unless that work is needed.
    _prepare_diagnostics(diagnostics, False, False)
    continuous_allocation = (
        _maximize_kelly_with_selection(
            transposed_returns,
            max_total_allocation_value,
            maximal_cliques,
            selection_constraints,
            allocation_tolerance_value,
        )
        if alpha_value == 0.0 and allocation_blocks is None
        else None
    )
    if add_limit_value is None and continuous_allocation is not None:
        _record_algorithm(diagnostics, _CONTINUOUS_KELLY_ALGORITHM)
        return _finalize_continuous_diagnostics(
            continuous_allocation.to(dtype=target_dtype),
            diagnostics,
            bootstrap_on,
            bootstrap_runs,
            bootstrap_length,
            percentile,
            bootstrap_seed,
            transposed_returns,
            len(maximal_cliques),
        )

    blocked_fast_outcome = None
    if alpha_value == 0.0 and allocation_blocks is not None:
        blocked_fast_outcome = _try_blocked_fast_path(
            transposed_returns,
            max_total_allocation_value,
            maximal_cliques,
            selection_constraints,
            allocation_blocks,
            add_limit_value,
            target_dtype,
            bootstrap_on,
            bootstrap_runs,
            bootstrap_length,
            percentile,
            bootstrap_seed,
            approximate_add_search_value,
            allocation_tolerance_value,
            grid_increment,
            diagnostics,
            _sample_bootstrap_indices,
            _resolve_grid_steps,
            _bootstrap_average_drawdown,
        )
        if blocked_fast_outcome.allocation is not None:
            return blocked_fast_outcome.allocation

    add_fast_outcome = None
    add_branch_outcome = None
    add_branch_pilot_ran = False
    if alpha_value == 0.0 and add_limit_value is not None and allocation_blocks is None:
        add_fast_outcome = _try_add_fast_path(
            transposed_returns,
            continuous_allocation,
            max_total_allocation_value,
            add_limit_value,
            selection_constraints,
            target_dtype,
            bootstrap_on,
            bootstrap_runs,
            bootstrap_length,
            percentile,
            bootstrap_seed,
            approximate_add_search_value,
            allocation_tolerance_value,
            diagnostics,
            len(maximal_cliques),
            _sample_bootstrap_indices,
            _bootstrap_average_drawdown,
        )
        if add_fast_outcome.allocation is not None:
            return add_fast_outcome.allocation

        branch_path = _try_add_branch_path(
            transposed_returns,
            continuous_allocation,
            max_total_allocation_value,
            add_limit_value,
            maximal_cliques,
            selection_constraints,
            grid_increment,
            target_dtype,
            allocation_tolerance_value,
            diagnostics,
            add_fast_outcome,
            _try_add_branch_and_bound,
            _resolve_grid_steps,
            _bootstrap_average_drawdown,
        )
        if branch_path.allocation is not None:
            return branch_path.allocation
        add_branch_outcome = branch_path.search_outcome
        add_branch_pilot_ran = branch_path.pilot_ran

    if grid_increment is None:
        unsupported_context = (
            "the requested objective, overlap geometry, or trade blocking"
            if allocation_blocks is not None
            else "the requested objective or overlap geometry"
        )
        raise ValueError(
            "grid_increment is required when the continuous allocation solvers "
            f"cannot handle {unsupported_context}"
        )
    drawdown_required = alpha_value > 0.0 or add_limit_value is not None
    bootstrap_on_value = False
    bootstrap_runs_value = 1
    bootstrap_length_value = 1
    percentile_value = 10.0
    bootstrap_seed_value = None
    blocked_bootstrap_prepared = (
        blocked_fast_outcome is not None and blocked_fast_outcome.bootstrap_prepared
    )
    if add_fast_outcome is None and not blocked_bootstrap_prepared:
        (
            bootstrap_on_value,
            bootstrap_runs_value,
            bootstrap_length_value,
            percentile_value,
            bootstrap_seed_value,
        ) = _resolve_grid_bootstrap_parameters(
            bootstrap_on,
            bootstrap_runs,
            bootstrap_length,
            percentile,
            bootstrap_seed,
            diagnostics is not None and diagnostics.request_bootstrap_average_drawdown,
            drawdown_required,
        )
    refinement_runs_value, refinement_divisor_value = _validate_refinement_parameters(
        refinement_runs, refinement_divisor
    )
    grid_steps = _resolve_grid_steps(grid_increment)
    initial_increment = 1.0 / grid_steps
    _validate_final_refinement_increment(
        initial_increment, refinement_runs_value, refinement_divisor_value
    )
    allocation_unit_limit = _resolve_allocation_unit_limit(
        max_total_allocation_value, grid_steps
    )
    if add_fast_outcome is None and not blocked_bootstrap_prepared:
        (
            bootstrap_diagnostics_enabled,
            bootstrap_needed,
            track_bootstrap_drawdown,
        ) = _prepare_diagnostics(
            diagnostics,
            bootstrap_on_value,
            drawdown_required,
        )
        bootstrap_indices = None
    elif add_fast_outcome is not None:
        bootstrap_runs_value = add_fast_outcome.bootstrap_runs
        bootstrap_length_value = add_fast_outcome.bootstrap_length
        percentile_value = add_fast_outcome.percentile
        bootstrap_diagnostics_enabled = add_fast_outcome.bootstrap_diagnostics_enabled
        bootstrap_needed = add_fast_outcome.bootstrap_needed
        track_bootstrap_drawdown = add_fast_outcome.track_bootstrap_drawdown
        bootstrap_indices = add_fast_outcome.bootstrap_indices
    else:
        if blocked_fast_outcome is None:
            raise RuntimeError("blocked bootstrap state was unexpectedly discarded")
        bootstrap_runs_value = blocked_fast_outcome.bootstrap_runs
        bootstrap_length_value = blocked_fast_outcome.bootstrap_length
        percentile_value = blocked_fast_outcome.percentile
        bootstrap_diagnostics_enabled = (
            blocked_fast_outcome.bootstrap_diagnostics_enabled
        )
        bootstrap_needed = blocked_fast_outcome.bootstrap_needed
        track_bootstrap_drawdown = blocked_fast_outcome.track_bootstrap_drawdown
        bootstrap_indices = blocked_fast_outcome.bootstrap_indices
    if alpha_value == 0.0 and add_limit_value is None:
        grid_algorithm = _RAW_KELLY_GRID_ALGORITHM
    elif alpha_value == 0.0:
        grid_algorithm = _ADD_GRID_ALGORITHM
    elif add_limit_value is None:
        grid_algorithm = _MODIFIED_GRID_ALGORITHM
    else:
        grid_algorithm = _MODIFIED_ADD_GRID_ALGORITHM
    if add_branch_pilot_ran or (
        blocked_fast_outcome is not None and blocked_fast_outcome.pilot_ran
    ):
        grid_algorithm = _PILOT_GRID_FALLBACK_ALGORITHM
    if allocation_blocks is not None:
        grid_algorithm = f"Blocking-aware {grid_algorithm}"
    _record_algorithm(diagnostics, grid_algorithm)
    ordinary_simplex = strategy_count == 1 or (
        len(maximal_cliques) == 1 and len(maximal_cliques[0]) == strategy_count
    )
    fully_disjoint = strategy_count > 1 and not maximal_cliques
    cliques_by_strategy = _clique_indices_by_strategy(
        maximal_cliques, strategy_count, target_device
    )
    uses_prefix_expansion = (
        selection_constraints is not None
        or (not ordinary_simplex and not fully_disjoint)
        or (refinement_runs_value > 0 and bool(maximal_cliques))
    )
    batch_size = _search_candidate_batch_size(
        returns.shape[0],
        strategy_count,
        len(maximal_cliques),
        calculation_dtype,
        target_device,
        uses_prefix_expansion,
        bootstrap_needed,
        drawdown_required,
        bootstrap_runs_value,
        bootstrap_length_value,
        _MAX_CANDIDATES_PER_BATCH,
        _candidate_batch_size,
        block_group_count=(
            0 if allocation_blocks is None else allocation_blocks.group_count
        ),
    )
    if add_fast_outcome is None and not blocked_bootstrap_prepared and bootstrap_needed:
        bootstrap_indices = _sample_bootstrap_indices(
            returns.shape[0],
            bootstrap_runs_value,
            bootstrap_length_value,
            target_device,
            bootstrap_seed_value,
        )

    allocation_unit_batches = _initial_allocation_unit_batches(
        allocation_unit_limit,
        strategy_count,
        maximal_cliques,
        cliques_by_strategy,
        ordinary_simplex,
        fully_disjoint,
        target_device,
        batch_size,
        selection_constraints,
    )

    best_score, best_allocation, best_bootstrap_drawdown = _initialize_grid_incumbent(
        strategy_count,
        target_device,
        target_dtype,
        calculation_dtype,
        track_bootstrap_drawdown,
        add_branch_outcome,
        transposed_returns,
        add_limit_value,
        bootstrap_indices,
        percentile_value,
        allocation_tolerance_value,
        _bootstrap_average_drawdown,
    )

    allocation_batches = (
        _units_to_allocations(allocation_units, grid_steps, target_dtype)
        for allocation_units in allocation_unit_batches
    )
    best_score, best_allocation, best_bootstrap_drawdown = _update_best_allocation(
        allocation_batches,
        transposed_returns,
        alpha_value,
        bootstrap_indices,
        percentile_value,
        best_score,
        best_allocation,
        best_bootstrap_drawdown,
        track_bootstrap_drawdown,
        add_limit_value,
        allocation_blocks,
    )
    if allocation_unit_limit == 0:
        return _finalize_allocation_diagnostics(
            best_allocation,
            diagnostics,
            bootstrap_indices,
            transposed_returns,
            percentile_value,
            best_bootstrap_drawdown,
            _bootstrap_average_drawdown,
            allocation_blocks,
        )

    refinement_increment = initial_increment
    for _ in range(refinement_runs_value):
        refinement_increment /= refinement_divisor_value
        while True:
            refinement_center = best_allocation
            previous_unit_indices, allocation_unit_batches = (
                _refined_allocation_unit_batches(
                    refinement_center,
                    allocation_unit_limit,
                    refinement_increment,
                    max_total_allocation_value,
                    maximal_cliques,
                    cliques_by_strategy,
                    target_device,
                    batch_size,
                    target_dtype,
                    selection_constraints,
                )
            )

            allocation_batches = (
                _refinement_units_to_allocations(
                    allocation_units,
                    refinement_center,
                    previous_unit_indices,
                    refinement_increment,
                    target_dtype,
                )
                for allocation_units in allocation_unit_batches
            )
            best_score, best_allocation, best_bootstrap_drawdown = (
                _update_best_allocation(
                    allocation_batches,
                    transposed_returns,
                    alpha_value,
                    bootstrap_indices,
                    percentile_value,
                    best_score,
                    best_allocation,
                    best_bootstrap_drawdown,
                    track_bootstrap_drawdown,
                    add_limit_value,
                    allocation_blocks,
                )
            )
            if torch.equal(best_allocation, refinement_center):
                break

    return _finalize_allocation_diagnostics(
        best_allocation,
        diagnostics,
        bootstrap_indices if bootstrap_diagnostics_enabled else None,
        transposed_returns,
        percentile_value,
        best_bootstrap_drawdown,
        _bootstrap_average_drawdown,
        allocation_blocks,
    )


def _refined_allocation_unit_batches(
    refinement_center: torch.Tensor,
    allocation_unit_limit: int,
    refinement_increment: float,
    max_total_allocation: float,
    maximal_cliques: list[tuple[int, ...]],
    cliques_by_strategy: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    target_dtype: torch.dtype,
    selection_constraints: StrategySelectionConstraints | None,
) -> tuple[torch.Tensor, Iterator[torch.Tensor]]:
    """Build one local grid and its valid-only candidate generator."""
    strategy_count = refinement_center.numel()
    _, previous_unit_indices = _refinement_grid_lower_bounds(
        refinement_center,
        allocation_unit_limit,
        refinement_increment,
        max_total_allocation,
    )
    clique_unit_limits = None
    if maximal_cliques:
        clique_unit_limits = _refinement_clique_unit_limits(
            refinement_center,
            previous_unit_indices,
            maximal_cliques,
            allocation_unit_limit,
            refinement_increment,
            max_total_allocation,
        )
    if not maximal_cliques and selection_constraints is None:
        return previous_unit_indices, _cartesian_allocation_unit_batches(
            allocation_unit_limit, strategy_count, device, batch_size
        )

    positive_unit_starts = None
    if selection_constraints is not None:
        first_points = _refinement_units_to_allocations(
            torch.zeros((1, strategy_count), device=device, dtype=torch.int64),
            refinement_center,
            previous_unit_indices,
            refinement_increment,
            target_dtype,
        )[0]
        positive_unit_starts = tuple(
            0 if value > 0.0 else 1 for value in first_points.to(device="cpu").tolist()
        )
    return previous_unit_indices, _constrained_allocation_unit_batches(
        allocation_unit_limit,
        strategy_count,
        len(maximal_cliques),
        cliques_by_strategy,
        device,
        batch_size,
        clique_unit_limits=clique_unit_limits,
        selection_constraints=selection_constraints,
        positive_unit_starts=positive_unit_starts,
    )


def _initial_allocation_unit_batches(
    allocation_unit_limit: int,
    strategy_count: int,
    maximal_cliques: list[tuple[int, ...]],
    cliques_by_strategy: list[torch.Tensor],
    ordinary_simplex: bool,
    fully_disjoint: bool,
    device: torch.device,
    batch_size: int,
    selection_constraints: StrategySelectionConstraints | None = None,
) -> Iterator[torch.Tensor]:
    """Return the fastest valid-only generator for the initial global grid."""
    selection_is_redundant = (
        selection_constraints is not None
        and ordinary_simplex
        and selection_constraints.maximum_active
        >= min(strategy_count, allocation_unit_limit)
        and all(
            conflicts.numel() == 0
            for conflicts in selection_constraints.prior_conflicts
        )
    )
    if selection_constraints is not None and not selection_is_redundant:
        return _constrained_allocation_unit_batches(
            allocation_unit_limit,
            strategy_count,
            len(maximal_cliques),
            cliques_by_strategy,
            device,
            batch_size,
            selection_constraints=selection_constraints,
            positive_unit_starts=(1,) * strategy_count,
        )
    if ordinary_simplex:
        return _simplex_allocation_unit_batches(
            allocation_unit_limit, strategy_count, device, batch_size
        )
    if fully_disjoint:
        return _cartesian_allocation_unit_batches(
            allocation_unit_limit, strategy_count, device, batch_size
        )
    return _constrained_allocation_unit_batches(
        allocation_unit_limit,
        strategy_count,
        len(maximal_cliques),
        cliques_by_strategy,
        device,
        batch_size,
    )


def _simplex_allocation_unit_batches(
    allocation_unit_limit: int,
    strategy_count: int,
    device: torch.device,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    """Yield batches from the ordinary simplex without invalid candidates."""
    candidate_count = math.comb(allocation_unit_limit + strategy_count, strategy_count)
    if candidate_count > torch.iinfo(torch.int64).max:
        raise ValueError("the allocation grid has too many candidates to index")
    rank_boundaries = _make_rank_boundaries(
        allocation_unit_limit, strategy_count, device
    )
    for start in range(0, candidate_count, batch_size):
        stop = min(start + batch_size, candidate_count)
        ranks = torch.arange(start, stop, device=device, dtype=torch.int64)
        yield _unrank_allocation_units(ranks, rank_boundaries, strategy_count)


def _cartesian_allocation_unit_batches(
    allocation_unit_limit: int,
    strategy_count: int,
    device: torch.device,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    """Yield all-disjoint grid points by decoding mixed-radix ranks."""
    base = allocation_unit_limit + 1
    candidate_count = base**strategy_count
    if candidate_count > torch.iinfo(torch.int64).max:
        raise ValueError("the allocation grid has too many candidates to index")

    for start in range(0, candidate_count, batch_size):
        stop = min(start + batch_size, candidate_count)
        work = torch.arange(start, stop, device=device, dtype=torch.int64)
        allocation_units = torch.empty(
            (stop - start, strategy_count), device=device, dtype=torch.int64
        )
        for column in range(strategy_count - 1, -1, -1):
            allocation_units[:, column] = torch.remainder(work, base)
            work.div_(base, rounding_mode="floor")
        yield allocation_units


def _make_rank_boundaries(
    allocation_unit_limit: int, strategy_count: int, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build binomial boundaries used to unrank simplex-grid candidates."""
    # Weak compositions correspond to nondecreasing separator positions.
    # These cumulative binomial counts let searchsorted unrank them in batches.
    boundaries = []
    for column in range(strategy_count):
        remaining_columns = strategy_count - column - 1
        degree = remaining_columns + 1
        descending = torch.tensor(
            [
                math.comb(allocation_unit_limit - value + degree, degree)
                for value in range(allocation_unit_limit + 2)
            ],
            device=device,
            dtype=torch.int64,
        )
        boundaries.append((descending, -descending))
    return boundaries


def _unrank_allocation_units(
    ranks: torch.Tensor,
    rank_boundaries: list[tuple[torch.Tensor, torch.Tensor]],
    strategy_count: int,
) -> torch.Tensor:
    """Map consecutive ranks directly to valid integer simplex points."""
    residual_ranks = ranks.clone()
    lower_bound = torch.zeros_like(ranks)
    allocation_units = torch.empty(
        (ranks.shape[0], strategy_count), device=ranks.device, dtype=torch.int64
    )

    for column, (descending, ascending_negative) in enumerate(rank_boundaries):
        base = descending[lower_bound]
        target = base - residual_ranks
        selected = torch.searchsorted(ascending_negative, -target, right=True) - 1
        residual_ranks -= base - descending[selected]
        allocation_units[:, column] = selected - lower_bound
        lower_bound = selected

    return allocation_units


def _units_to_allocations(
    allocation_units: torch.Tensor, grid_steps: int, dtype: torch.dtype
) -> torch.Tensor:
    """Convert exact integer grid units to allocation fractions."""
    intermediate_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    allocations = allocation_units.to(dtype=intermediate_dtype)
    allocations /= grid_steps
    return allocations.to(dtype=dtype)


def _update_best_allocation(
    allocation_batches: Iterator[torch.Tensor],
    transposed_returns: torch.Tensor,
    alpha: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    best_score: torch.Tensor,
    best_allocation: torch.Tensor,
    best_bootstrap_drawdown: torch.Tensor | None,
    track_bootstrap_drawdown: bool,
    add_limit: float | None,
    allocation_blocks: _AllocationBlocks | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Score allocation batches and retain only strict improvements."""
    for allocations in allocation_batches:
        calculation_allocations = allocations.to(dtype=transposed_returns.dtype)
        if track_bootstrap_drawdown:
            bootstrap_drawdown_output: list[torch.Tensor | None] = []
            scores = _allocation_scores(
                calculation_allocations,
                transposed_returns,
                alpha,
                bootstrap_indices=bootstrap_indices,
                percentile=percentile,
                bootstrap_drawdown_output=bootstrap_drawdown_output,
                add_limit=add_limit,
                **_allocation_block_score_arguments(allocation_blocks),
            )
            if len(bootstrap_drawdown_output) != 1:
                raise RuntimeError("Scoring did not return bootstrap diagnostics")
            batch_bootstrap_drawdowns = bootstrap_drawdown_output[0]
        else:
            scores = _allocation_scores(
                calculation_allocations,
                transposed_returns,
                alpha,
                bootstrap_indices=bootstrap_indices,
                percentile=percentile,
                add_limit=add_limit,
                **_allocation_block_score_arguments(allocation_blocks),
            )
            batch_bootstrap_drawdowns = None
        batch_score, batch_index = torch.max(scores, dim=0)
        use_batch_result = batch_score > best_score
        best_score = torch.where(use_batch_result, batch_score, best_score)
        best_allocation = torch.where(
            use_batch_result, allocations[batch_index], best_allocation
        )
        if track_bootstrap_drawdown:
            if best_bootstrap_drawdown is None or batch_bootstrap_drawdowns is None:
                raise RuntimeError("Bootstrap drawdown diagnostics are unavailable")
            best_bootstrap_drawdown = torch.where(
                use_batch_result,
                batch_bootstrap_drawdowns[batch_index],
                best_bootstrap_drawdown,
            )
    return best_score, best_allocation, best_bootstrap_drawdown


# CUDA Inductor's map-to-loop lowering extracts a scalar loop index. Scope
# scalar capture to scoring calls so the compiler can trace that operation.
@_COMPILER_CONFIG_PATCH(capture_scalar_outputs=True)
@torch.compile()
def _allocation_scores(
    allocations: torch.Tensor,
    transposed_returns: torch.Tensor,
    alpha: float,
    bootstrap_indices: torch.Tensor | None = None,
    percentile: float = 10.0,
    bootstrap_drawdown_output: list[torch.Tensor | None] | None = None,
    add_limit: float | None = None,
    block_day_indices: torch.Tensor | None = None,
    blocked_strategy_indices: torch.Tensor | None = None,
    blocker_incidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the modified Kelly objective for a batch of allocations."""
    portfolio_returns = _effective_portfolio_returns(
        allocations,
        transposed_returns,
        block_day_indices,
        blocked_strategy_indices,
        blocker_incidence,
    )
    finite = torch.isfinite(portfolio_returns).all(dim=1)
    valid = finite & (torch.amin(portfolio_returns, dim=1) > -1.0)
    portfolio_returns[~valid] = 0.0

    drawdown_required = alpha != 0.0 or add_limit is not None
    average_drawdown = None
    if drawdown_required and bootstrap_indices is not None:
        average_drawdown = _bootstrap_average_drawdown(
            portfolio_returns, bootstrap_indices, percentile
        )

    portfolio_returns.log1p_()
    growth = portfolio_returns.sum(dim=1)
    if drawdown_required and average_drawdown is None:
        average_drawdown = _negative_rms_drawdown(portfolio_returns, dim=1)
    scores = growth
    if alpha != 0.0:
        if average_drawdown is None:
            raise RuntimeError("Drawdown is unavailable for the penalty")
        duration = portfolio_returns.shape[1]
        scores = scores + alpha * duration * torch.log1p(average_drawdown)
    if add_limit is not None:
        if average_drawdown is None:
            raise RuntimeError("Drawdown is unavailable for ADD_limit")
        scores = scores.masked_fill(average_drawdown < add_limit, -torch.inf)
    scores = scores.nan_to_num(nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
    scores = scores.masked_fill(~valid, -torch.inf)
    if bootstrap_drawdown_output is not None:
        bootstrap_drawdowns = (
            average_drawdown if bootstrap_indices is not None else None
        )
        bootstrap_drawdown_output.append(bootstrap_drawdowns)
    return scores


def _allocation_block_score_arguments(
    allocation_blocks: _AllocationBlocks | None,
) -> dict[str, torch.Tensor | None]:
    """Return keyword tensors consumed by compiled candidate scoring."""
    if allocation_blocks is None:
        return {}
    return {
        "block_day_indices": allocation_blocks.day_indices,
        "blocked_strategy_indices": allocation_blocks.blocked_indices,
        "blocker_incidence": allocation_blocks.blocker_incidence,
    }
