"""Fast Kelly searches with a hard average-drawdown constraint."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._capital_allocation_drawdown import (
    bootstrap_average_drawdown,
    negative_rms_drawdown,
)
from ._strategy_selection import StrategySelectionConstraints

_FLOAT32_SCALE_TOLERANCE = 1e-6
_FLOAT64_SCALE_TOLERANCE = 1e-9
RAW_PRECHECK_ALGORITHM = "Continuous raw Kelly + exact ADD feasibility precheck"
SCALAR_ALGORITHM = "Exact scalar ADD-constrained Kelly (bisection)"
FIXED_RAY_ALGORITHM = "Approximate full-Kelly fixed-ray ADD bisection"
_OUTPUT_ADJUSTMENT_SUFFIX = " + output-dtype feasibility adjustment"
BootstrapDrawdownCalculator = Callable[
    [torch.Tensor, torch.Tensor, float], torch.Tensor
]


@dataclass(frozen=True)
class AddConstrainedKellyResult:
    """An allocation found without the multidimensional allocation grid."""

    allocation: torch.Tensor
    bootstrap_average_drawdown: torch.Tensor | None
    exact: bool
    algorithm: str


def validate_approximate_add_search(value: bool) -> bool:
    """Return a validated fixed-ray approximation switch."""
    if not isinstance(value, bool):
        raise TypeError("approximate_add_search must be a bool")
    return value


@torch.no_grad()
def cast_result_to_feasible_dtype(
    result: AddConstrainedKellyResult,
    target_dtype: torch.dtype,
    transposed_returns: torch.Tensor,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float | None = None,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator = (
        bootstrap_average_drawdown
    ),
) -> AddConstrainedKellyResult:
    """Cast a fast result without allowing output rounding below the ADD floor."""
    if result.allocation.dtype == target_dtype:
        return result

    cast_allocation = result.allocation.to(dtype=target_dtype)
    cast_drawdown = _allocation_drawdowns(
        cast_allocation.to(dtype=transposed_returns.dtype).unsqueeze(0),
        transposed_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )[0]
    if bool((cast_drawdown >= add_limit).item()):
        return _make_result(
            cast_allocation,
            cast_drawdown,
            bootstrap_indices,
            result.exact,
            result.algorithm,
        )

    lower = 0.0
    higher = 1.0
    feasible_allocation = torch.zeros_like(cast_allocation)
    feasible_drawdown = torch.zeros_like(cast_drawdown)
    maximum_width = float(torch.amax(torch.abs(result.allocation)).item())
    for _ in range(
        _bisection_iterations(
            transposed_returns.dtype, maximum_width, allocation_tolerance
        )
    ):
        midpoint = (lower + higher) * 0.5
        candidate = (midpoint * result.allocation).to(dtype=target_dtype)
        candidate_drawdown = _allocation_drawdowns(
            candidate.to(dtype=transposed_returns.dtype).unsqueeze(0),
            transposed_returns,
            bootstrap_indices,
            percentile,
            bootstrap_drawdown_calculator,
        )[0]
        if bool((candidate_drawdown >= add_limit).item()):
            lower = midpoint
            feasible_allocation = candidate
            feasible_drawdown = candidate_drawdown
        else:
            higher = midpoint
    return _make_result(
        feasible_allocation,
        feasible_drawdown,
        bootstrap_indices,
        result.exact,
        result.algorithm + _OUTPUT_ADJUSTMENT_SUFFIX,
    )


@torch.no_grad()
def try_add_constrained_kelly(
    transposed_returns: torch.Tensor,
    raw_allocation: torch.Tensor | None,
    max_total_allocation: float,
    add_limit: float,
    selection_constraints: StrategySelectionConstraints | None,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    approximate_add_search: bool,
    allocation_tolerance: float | None = None,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator = (
        bootstrap_average_drawdown
    ),
) -> AddConstrainedKellyResult | None:
    """Try exact special cases, then an optional fixed-Kelly-ray approximation.

    The caller supplies one set of bootstrap indices for the whole search. A
    ``None`` result means that the general multidimensional allocation grid is
    still required.
    """
    use_approximation = validate_approximate_add_search(approximate_add_search)
    if raw_allocation is not None:
        raw_drawdown = _allocation_drawdowns(
            raw_allocation.unsqueeze(0),
            transposed_returns,
            bootstrap_indices,
            percentile,
            bootstrap_drawdown_calculator,
        )[0]
        if bool((raw_drawdown >= add_limit).item()):
            return _make_result(
                raw_allocation,
                raw_drawdown,
                bootstrap_indices,
                True,
                RAW_PRECHECK_ALGORITHM,
            )

    strategy_count = transposed_returns.shape[0]
    if _support_is_at_most_one(strategy_count, selection_constraints):
        if (
            selection_constraints is not None
            and selection_constraints.maximum_active == 0
        ):
            return _cash_result(transposed_returns, bootstrap_indices, SCALAR_ALGORITHM)
        return _maximize_over_singleton_supports(
            transposed_returns,
            max_total_allocation,
            add_limit,
            bootstrap_indices,
            percentile,
            bootstrap_drawdown_calculator,
            allocation_tolerance,
        )

    if use_approximation and raw_allocation is not None:
        allocations, drawdowns = _constrain_allocation_rays(
            raw_allocation.unsqueeze(0),
            transposed_returns,
            add_limit,
            bootstrap_indices,
            percentile,
            bootstrap_drawdown_calculator,
            allocation_tolerance,
        )
        return _make_result(
            allocations[0],
            drawdowns[0],
            bootstrap_indices,
            exact=False,
            algorithm=FIXED_RAY_ALGORITHM,
        )
    return None


def _make_result(
    allocation: torch.Tensor,
    drawdown: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    exact: bool,
    algorithm: str,
) -> AddConstrainedKellyResult:
    """Keep a bootstrap ADD only when it can be reused for diagnostics."""
    bootstrap_drawdown = drawdown if bootstrap_indices is not None else None
    return AddConstrainedKellyResult(allocation, bootstrap_drawdown, exact, algorithm)


def _cash_result(
    transposed_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    algorithm: str,
) -> AddConstrainedKellyResult:
    """Return the feasible no-risk allocation."""
    cash = torch.zeros(
        transposed_returns.shape[0],
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    cash_drawdown = torch.zeros(
        (), device=transposed_returns.device, dtype=transposed_returns.dtype
    )
    return _make_result(
        cash, cash_drawdown, bootstrap_indices, exact=True, algorithm=algorithm
    )


def _support_is_at_most_one(
    strategy_count: int,
    constraints: StrategySelectionConstraints | None,
) -> bool:
    """Return whether every feasible allocation has at most one active asset."""
    if strategy_count <= 1:
        return True
    if constraints is None:
        return False
    if constraints.maximum_active <= 1:
        return True
    return all(
        prior_conflicts.numel() == strategy
        for strategy, prior_conflicts in enumerate(constraints.prior_conflicts)
    )


def _maximize_over_singleton_supports(
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator,
    allocation_tolerance: float | None,
) -> AddConstrainedKellyResult:
    """Compare the exact drawdown-constrained optimum on every singleton ray."""
    raw_fractions = _singleton_kelly_fractions(
        transposed_returns, max_total_allocation, allocation_tolerance
    )
    raw_allocations = torch.diag(raw_fractions)
    allocations, drawdowns = _constrain_allocation_rays(
        raw_allocations,
        transposed_returns,
        add_limit,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
        allocation_tolerance,
    )
    portfolio_returns = allocations @ transposed_returns
    growth = torch.log1p(portfolio_returns).sum(dim=1)
    best_growth, best_strategy = torch.max(growth, dim=0)
    if bool((best_growth > 0.0).item()):
        return _make_result(
            allocations[best_strategy],
            drawdowns[best_strategy],
            bootstrap_indices,
            exact=True,
            algorithm=SCALAR_ALGORITHM,
        )

    return _cash_result(transposed_returns, bootstrap_indices, SCALAR_ALGORITHM)


def _singleton_kelly_fractions(
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    allocation_tolerance: float | None,
) -> torch.Tensor:
    """Solve all independent one-dimensional raw-Kelly problems in parallel."""
    strategy_count = transposed_returns.shape[0]
    zero = torch.zeros(
        strategy_count,
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    if max_total_allocation == 0.0:
        return zero

    maximum = torch.full_like(zero, max_total_allocation)
    negative_returns = transposed_returns < 0.0
    solvency_limits = torch.where(
        negative_returns,
        -1.0 / transposed_returns,
        torch.full_like(transposed_returns, torch.inf),
    ).amin(dim=1)
    upper = torch.minimum(maximum, solvency_limits)
    initial_derivative = transposed_returns.sum(dim=1)

    upper_wealth = 1.0 + upper.unsqueeze(1) * transposed_returns
    upper_is_solvent = torch.all(upper_wealth > 0.0, dim=1)
    safe_upper_wealth = torch.where(
        upper_is_solvent.unsqueeze(1),
        upper_wealth,
        torch.ones_like(upper_wealth),
    )
    upper_derivative = (transposed_returns / safe_upper_wealth).sum(dim=1)
    use_upper = (
        (initial_derivative > 0.0) & upper_is_solvent & (upper_derivative >= 0.0)
    )
    find_root = (initial_derivative > 0.0) & ~use_upper

    lower = zero.clone()
    higher = upper
    for _ in range(
        _bisection_iterations(
            transposed_returns.dtype,
            max_total_allocation,
            allocation_tolerance,
        )
    ):
        midpoint = (lower + higher) * 0.5
        wealth = 1.0 + midpoint.unsqueeze(1) * transposed_returns
        solvent = torch.all(wealth > 0.0, dim=1)
        safe_wealth = torch.where(solvent.unsqueeze(1), wealth, torch.ones_like(wealth))
        derivative = (transposed_returns / safe_wealth).sum(dim=1)
        move_lower = find_root & solvent & (derivative > 0.0)
        lower = torch.where(move_lower, midpoint, lower)
        higher = torch.where(find_root & ~move_lower, midpoint, higher)
    return torch.where(use_upper, upper, lower)


def _constrain_allocation_rays(
    raw_allocations: torch.Tensor,
    transposed_returns: torch.Tensor,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator,
    allocation_tolerance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale allocation rays to their largest drawdown-feasible points."""
    raw_drawdowns = _allocation_drawdowns(
        raw_allocations,
        transposed_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )
    constrained = raw_drawdowns < add_limit
    if not bool(torch.any(constrained).item()):
        return raw_allocations, raw_drawdowns

    constrained_indices = torch.nonzero(constrained, as_tuple=False).flatten()
    constrained_raw = raw_allocations.index_select(0, constrained_indices)
    lower = torch.zeros_like(constrained_indices, dtype=raw_drawdowns.dtype)
    higher = torch.ones_like(lower)
    lower_drawdowns = torch.zeros_like(lower)

    maximum_width = float(torch.amax(torch.abs(constrained_raw)).item())
    for _ in range(
        _bisection_iterations(
            transposed_returns.dtype, maximum_width, allocation_tolerance
        )
    ):
        midpoint = (lower + higher) * 0.5
        midpoint_allocations = midpoint.unsqueeze(1) * constrained_raw
        midpoint_drawdowns = _allocation_drawdowns(
            midpoint_allocations,
            transposed_returns,
            bootstrap_indices,
            percentile,
            bootstrap_drawdown_calculator,
        )
        midpoint_is_feasible = midpoint_drawdowns >= add_limit
        move_lower = midpoint_is_feasible
        lower = torch.where(move_lower, midpoint, lower)
        lower_drawdowns = torch.where(move_lower, midpoint_drawdowns, lower_drawdowns)
        higher = torch.where(~midpoint_is_feasible, midpoint, higher)

    allocations = raw_allocations.clone()
    allocations[constrained_indices] = lower.unsqueeze(1) * constrained_raw
    drawdowns = raw_drawdowns.clone()
    drawdowns[constrained_indices] = lower_drawdowns
    return allocations, drawdowns


def _allocation_drawdowns(
    allocations: torch.Tensor,
    transposed_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_drawdown_calculator: BootstrapDrawdownCalculator,
) -> torch.Tensor:
    """Calculate historical or shared-bootstrap ADD for allocation rows."""
    portfolio_returns = allocations @ transposed_returns
    valid = torch.isfinite(portfolio_returns).all(dim=1) & (
        torch.amin(portfolio_returns, dim=1) > -1.0
    )
    safe_returns = torch.where(
        valid.unsqueeze(1), portfolio_returns, torch.zeros_like(portfolio_returns)
    )
    if bootstrap_indices is not None:
        drawdowns = bootstrap_drawdown_calculator(
            safe_returns, bootstrap_indices, percentile
        )
    else:
        drawdowns = negative_rms_drawdown(safe_returns.log1p_(), dim=1)
    return drawdowns.masked_fill(~valid, -torch.inf)


def _bisection_iterations(
    dtype: torch.dtype,
    maximum_width: float = 1.0,
    allocation_tolerance: float | None = None,
) -> int:
    """Return iterations needed for the requested absolute allocation width."""
    tolerance = allocation_tolerance or (
        _FLOAT64_SCALE_TOLERANCE if dtype == torch.float64 else _FLOAT32_SCALE_TOLERANCE
    )
    if maximum_width <= tolerance:
        return 0
    return math.ceil(math.log2(maximum_width / tolerance))
