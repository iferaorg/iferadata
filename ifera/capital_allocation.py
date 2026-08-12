"""Find capital allocations with a drawdown-penalized Kelly grid search."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

import torch

from ifera._capital_allocation_diagnostics import (
    CapitalAllocationDiagnostics,
    finalize_allocation_diagnostics as _finalize_allocation_diagnostics,
    prepare_diagnostics as _prepare_diagnostics,
    validate_bootstrap_parameters as _validate_bootstrap_parameters,
)
from ifera._capital_allocation_memory import (
    candidate_batch_size as _candidate_batch_size,
)
from ifera._capital_allocation_parameters import (
    validate_add_limit as _validate_add_limit,
    validate_alpha as _validate_alpha,
    validate_max_total_allocation as _validate_max_total_allocation,
)
from ifera._capital_allocation_refinement import (
    clique_unit_limits as _refinement_clique_unit_limits,
    grid_lower_bounds as _refinement_grid_lower_bounds,
    units_to_allocations as _refinement_units_to_allocations,
    validate_parameters as _validate_refinement_parameters,
    validate_final_increment as _validate_final_refinement_increment,
)

if TYPE_CHECKING:
    from ifera.portfolio_allocation import PortfolioAllocationResult

_BOOTSTRAP_RUN_CHUNK_SIZE = 16
_MAX_CANDIDATES_PER_BATCH = 1_000_000
_RECIPROCAL_ULP_TOLERANCE = 4
_SUPPORTED_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)


class _ExpansionFrame(NamedTuple):
    """State needed to resume one vectorized prefix expansion."""

    prefixes: torch.Tensor
    clique_usage: torch.Tensor
    column: int
    cumulative_counts: torch.Tensor
    child_counts: torch.Tensor
    next_child: int
    child_count: int


@torch.no_grad()
def find_optimal_capital_allocation(
    returns: torch.Tensor,
    alpha: float,
    grid_increment: float,
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
) -> torch.Tensor:
    """Return the best long-only allocation on an overlap-constrained grid.

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

    The requested grid increment is reduced, when necessary, to the largest
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
        grid_increment: Requested allocation increment in ``(0, 1]``.
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

    Returns:
        A one-dimensional tensor containing the optimal strategy allocations.

    Raises:
        TypeError: If a tensor dtype or parameter type is unsupported.
        ValueError: If an input value or shape is invalid, or if the grid cannot
            be safely indexed or evaluated within the search memory budget.
    """
    _validate_returns(returns)
    alpha_value, add_limit_value = _validate_alpha(alpha), _validate_add_limit(
        ADD_limit
    )
    (
        bootstrap_on_value,
        bootstrap_runs_value,
        bootstrap_length_value,
        percentile_value,
    ) = _validate_bootstrap_parameters(
        bootstrap_on, bootstrap_runs, bootstrap_length, percentile
    )
    max_total_allocation_value = _validate_max_total_allocation(max_total_allocation)
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
    target_dtype = returns.dtype if dtype is None else dtype
    if (
        not isinstance(target_dtype, torch.dtype)
        or target_dtype not in _SUPPORTED_DTYPES
    ):
        raise TypeError("dtype must be float16, bfloat16, float32, or float64")

    target_device = _resolve_device(device)
    calculation_dtype = _resolve_calculation_dtype(returns.dtype, target_dtype)
    strategy_count = returns.shape[1]
    calculation_returns = returns.to(device=target_device, dtype=calculation_dtype)
    maximal_cliques = _find_maximal_overlap_cliques(~torch.isnan(calculation_returns))
    calculation_returns = torch.nan_to_num(calculation_returns, nan=0.0)
    transposed_returns = calculation_returns.transpose(0, 1).contiguous()
    del calculation_returns
    (
        bootstrap_diagnostics_enabled,
        bootstrap_needed,
        track_bootstrap_drawdown,
    ) = _prepare_diagnostics(
        diagnostics,
        bootstrap_on_value,
        alpha_value > 0.0 or add_limit_value is not None,
    )
    ordinary_simplex = strategy_count == 1 or (
        len(maximal_cliques) == 1 and len(maximal_cliques[0]) == strategy_count
    )
    fully_disjoint = strategy_count > 1 and not maximal_cliques
    cliques_by_strategy = _clique_indices_by_strategy(
        maximal_cliques, strategy_count, target_device
    )
    uses_prefix_expansion = (not ordinary_simplex and not fully_disjoint) or (
        refinement_runs_value > 0 and bool(maximal_cliques)
    )
    batch_size = min(
        _MAX_CANDIDATES_PER_BATCH,
        _candidate_batch_size(
            returns.shape[0],
            strategy_count,
            len(maximal_cliques),
            calculation_dtype,
            target_device,
            uses_prefix_expansion,
            bootstrap_runs_value if bootstrap_needed else 0,
            bootstrap_length_value if bootstrap_needed else 0,
        ),
    )
    bootstrap_indices = None
    if bootstrap_needed:
        bootstrap_indices = torch.randint(
            returns.shape[0],
            (bootstrap_runs_value, bootstrap_length_value),
            device=target_device,
            dtype=torch.int64,
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
    )

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
        )

    refinement_increment = initial_increment
    for _ in range(refinement_runs_value):
        refinement_increment /= refinement_divisor_value
        while True:
            refinement_center = best_allocation
            _, previous_unit_indices = _refinement_grid_lower_bounds(
                refinement_center,
                allocation_unit_limit,
                refinement_increment,
                max_total_allocation_value,
            )
            if maximal_cliques:
                clique_unit_limits = _refinement_clique_unit_limits(
                    refinement_center,
                    previous_unit_indices,
                    maximal_cliques,
                    allocation_unit_limit,
                    refinement_increment,
                    max_total_allocation_value,
                )
                allocation_unit_batches = _constrained_allocation_unit_batches(
                    allocation_unit_limit,
                    strategy_count,
                    len(maximal_cliques),
                    cliques_by_strategy,
                    target_device,
                    batch_size,
                    clique_unit_limits=clique_unit_limits,
                )
            else:
                allocation_unit_batches = _cartesian_allocation_unit_batches(
                    allocation_unit_limit, strategy_count, target_device, batch_size
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
    )


def find_optimal_capital_allocation_from_csv(
    portfolio_name: str,
    alpha: float,
    grid_increment: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    bootstrap_on: bool = False,
    bootstrap_runs: int = 1024,
    bootstrap_length: int = 256,
    percentile: float = 10.0,
    max_total_allocation: float = 1.0,
    annualization_periods: int = 252,
    risk_free_rate: float = 0.0,
    show_plot: bool = True,
    refinement_runs: int = 0,
    refinement_divisor: float = 2.0,
    ADD_limit: float | None = None,  # pylint: disable=invalid-name
) -> PortfolioAllocationResult:
    """Load a named CSV portfolio, optimize it, and report the result.

    The reporting implementation is imported lazily so the tensor-only search
    does not require Polars, the NYSE calendar, or Matplotlib at import time.
    See :func:`ifera.portfolio_allocation.find_optimal_capital_allocation_from_csv`
    for the complete behavior and return-value documentation.
    """
    from ifera.portfolio_allocation import (  # pylint: disable=import-outside-toplevel
        find_optimal_capital_allocation_from_csv as find_from_csv,
    )

    return find_from_csv(
        portfolio_name,
        alpha,
        grid_increment,
        device,
        dtype,
        bootstrap_on,
        bootstrap_runs,
        bootstrap_length,
        percentile,
        max_total_allocation,
        annualization_periods,
        risk_free_rate,
        show_plot,
        refinement_runs,
        refinement_divisor,
        ADD_limit=ADD_limit,
    )


def _validate_returns(returns: torch.Tensor) -> None:
    """Validate the return-series tensor."""
    if not isinstance(returns, torch.Tensor):
        raise TypeError("returns must be a torch.Tensor")
    if returns.ndim != 2:
        raise ValueError("returns must have shape (time, strategies)")
    if returns.shape[0] == 0 or returns.shape[1] == 0:
        raise ValueError("returns must contain at least one time and one strategy")
    if returns.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "returns must have a float16, bfloat16, float32, or float64 dtype"
        )
    if bool(torch.isinf(returns).any().item()):
        raise ValueError("returns must not contain infinite values")


def _resolve_grid_steps(grid_increment: float) -> int:
    """Return the integer number of intervals for the adjusted grid."""
    try:
        increment = float(grid_increment)
    except (TypeError, ValueError) as exc:
        raise TypeError("grid_increment must be a real number") from exc
    if not math.isfinite(increment) or not 0.0 < increment <= 1.0:
        raise ValueError("grid_increment must be finite and in (0, 1]")

    reciprocal = 1.0 / increment
    if not math.isfinite(reciprocal):
        raise ValueError("grid_increment is too small")
    nearest_integer = round(reciprocal)
    reciprocal_error = abs(reciprocal - nearest_integer)
    if reciprocal_error <= _RECIPROCAL_ULP_TOLERANCE * math.ulp(reciprocal):
        return nearest_integer
    return math.ceil(reciprocal)


def _resolve_allocation_unit_limit(max_total_allocation: float, grid_steps: int) -> int:
    """Return the largest integer grid-unit cap within the requested maximum."""
    scaled_allocation = max_total_allocation * grid_steps
    if not math.isfinite(scaled_allocation):
        raise ValueError("max_total_allocation is too large for the grid")
    nearest_integer = round(scaled_allocation)
    allocation_error = abs(scaled_allocation - nearest_integer)
    if allocation_error <= _RECIPROCAL_ULP_TOLERANCE * math.ulp(scaled_allocation):
        allocation_unit_limit = nearest_integer
    else:
        allocation_unit_limit = math.floor(scaled_allocation)
    if allocation_unit_limit >= torch.iinfo(torch.int64).max:
        raise ValueError("max_total_allocation is too large to index")
    return allocation_unit_limit


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Return the requested device or the preferred default device."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _resolve_calculation_dtype(
    returns_dtype: torch.dtype, allocation_dtype: torch.dtype
) -> torch.dtype:
    """Choose a common calculation dtype with at least float32 precision."""
    calculation_dtype = torch.promote_types(returns_dtype, allocation_dtype)
    if torch.empty((), dtype=calculation_dtype).element_size() < 4:
        return torch.float32
    return calculation_dtype


def _find_maximal_overlap_cliques(active: torch.Tensor) -> list[tuple[int, ...]]:
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


def _clique_indices_by_strategy(
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


def _initial_allocation_unit_batches(
    allocation_unit_limit: int,
    strategy_count: int,
    maximal_cliques: list[tuple[int, ...]],
    cliques_by_strategy: list[torch.Tensor],
    ordinary_simplex: bool,
    fully_disjoint: bool,
    device: torch.device,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    """Return the fastest valid-only generator for the initial global grid."""
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


def _constrained_allocation_unit_batches(
    allocation_unit_limit: int,
    strategy_count: int,
    clique_count: int,
    cliques_by_strategy: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    clique_unit_limits: torch.Tensor | None = None,
) -> Iterator[torch.Tensor]:
    """Yield valid batches under arbitrary overlap-clique constraints."""
    integer_limit = torch.iinfo(torch.int64).max
    if allocation_unit_limit >= integer_limit:
        raise ValueError("the allocation limit is too large to index")
    if clique_unit_limits is None:
        clique_unit_limits = torch.full(
            (clique_count,), allocation_unit_limit, device=device, dtype=torch.int64
        )
    if clique_unit_limits.shape != (clique_count,):
        raise ValueError("clique_unit_limits must have one value per clique")
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
                column,
                allocation_unit_limit,
                clique_unit_limits,
                cliques_by_strategy,
            )
            prefixes, clique_usage, next_child = _constrained_child_batch(
                expansion, cliques_by_strategy, safe_batch_size
            )
            if next_child < expansion.child_count:
                expansion_stack.append(expansion._replace(next_child=next_child))
            column += 1
            continue

        if not expansion_stack:
            return
        expansion = expansion_stack.pop()
        prefixes, clique_usage, next_child = _constrained_child_batch(
            expansion, cliques_by_strategy, safe_batch_size
        )
        if next_child < expansion.child_count:
            expansion_stack.append(expansion._replace(next_child=next_child))
        column = expansion.column + 1


def _make_constrained_expansion(
    prefixes: torch.Tensor,
    clique_usage: torch.Tensor,
    column: int,
    allocation_unit_limit: int,
    clique_unit_limits: torch.Tensor,
    cliques_by_strategy: list[torch.Tensor],
) -> _ExpansionFrame:
    """Prepare counts for one valid-only prefix expansion."""
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
    cumulative_counts = torch.cumsum(child_counts, dim=0)
    child_count = int(cumulative_counts[-1].item())
    return _ExpansionFrame(
        prefixes,
        clique_usage,
        column,
        cumulative_counts,
        child_counts,
        0,
        child_count,
    )


def _constrained_child_batch(
    expansion: _ExpansionFrame,
    cliques_by_strategy: list[torch.Tensor],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
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
    return child_prefixes, child_usage, stop


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


def _allocation_scores(
    allocations: torch.Tensor,
    transposed_returns: torch.Tensor,
    alpha: float,
    bootstrap_indices: torch.Tensor | None = None,
    percentile: float = 10.0,
    bootstrap_drawdown_output: list[torch.Tensor | None] | None = None,
    add_limit: float | None = None,
) -> torch.Tensor:
    """Evaluate the modified Kelly objective for a batch of allocations."""
    portfolio_returns = allocations @ transposed_returns
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


def _bootstrap_average_drawdown(
    portfolio_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor,
    percentile: float,
) -> torch.Tensor:
    """Return the lower-percentile ADD across shared resampled paths."""
    run_count = bootstrap_indices.shape[0]
    run_drawdowns = torch.empty(
        (portfolio_returns.shape[0], run_count),
        device=portfolio_returns.device,
        dtype=portfolio_returns.dtype,
    )
    for start in range(0, run_count, _BOOTSTRAP_RUN_CHUNK_SIZE):
        stop = min(start + _BOOTSTRAP_RUN_CHUNK_SIZE, run_count)
        sampled_log_returns = portfolio_returns[:, bootstrap_indices[start:stop]]
        sampled_log_returns.log1p_()
        run_drawdowns[:, start:stop] = _negative_rms_drawdown(
            sampled_log_returns, dim=2
        )
    return torch.quantile(
        run_drawdowns,
        percentile / 100.0,
        dim=1,
        interpolation="linear",
    )


def _negative_rms_drawdown(
    portfolio_log_returns: torch.Tensor, dim: int
) -> torch.Tensor:
    """Return negative RMS fractional drawdown along a time dimension."""
    log_equity = portfolio_log_returns.cumsum_(dim=dim)
    running_peak = torch.cummax(log_equity, dim=dim).values
    running_peak.clamp_min_(0.0)
    log_equity.sub_(running_peak).expm1_()
    average_drawdown = log_equity.square_().mean(dim=dim).sqrt_().neg_()
    average_drawdown.clamp_min_(-1.0)
    return average_drawdown
