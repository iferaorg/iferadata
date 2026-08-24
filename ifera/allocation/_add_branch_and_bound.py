"""Adaptive spatial search for Kelly growth under a hard ADD floor.

The search is deliberately optional.  It first spends a bounded pilot budget
and returns a fallback outcome when interval bounds do not prune allocation
boxes quickly enough to beat the existing vectorized grid.  Every bootstrap
comparison uses the caller's single shared index tensor.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ._capital_allocation_drawdown import (
    bootstrap_average_drawdown,
    negative_rms_drawdown,
)
from ._add_branch_and_bound_support import (
    allocation_is_feasible as _allocation_is_feasible,
    maximal_support_masks as _maximal_support_masks,
    root_box as _root_box,
)
from ._strategy_selection import StrategySelectionConstraints

BRANCH_AND_BOUND_ALGORITHM = "Adaptive spatial ADD branch-and-bound"
PILOT_GRID_FALLBACK_ALGORITHM = "ADD branch-and-bound pilot + allocation grid fallback"

_MAX_PILOT_SUPPORTS = 32
_PILOT_NODE_BUDGET = 64
_MAX_SEARCH_NODES = 100_000
_GRID_ADVANTAGE_FACTOR = 0.65
_BOUND_ROUNDING_ULPS = 16.0
_BOOTSTRAP_PILOT_ELEMENT_BUDGET = 8_000_000
_MAX_BOOTSTRAP_PILOT_RUNS = 256


@dataclass(frozen=True)
class AddBranchAndBoundOutcome:
    """A completed allocation or a feasible incumbent for grid fallback.

    ``allocation`` is non-``None`` only when the spatial search completed.
    ``incumbent_allocation`` is always feasible and lets the caller preserve
    useful pilot work while still evaluating its complete initial grid.
    """

    allocation: torch.Tensor | None
    incumbent_allocation: torch.Tensor
    incumbent_mean_log_growth: float
    bootstrap_average_drawdown: torch.Tensor | None
    algorithm: str
    completed: bool
    pilot_nodes: int


@dataclass(frozen=True)
class _BoxNode:
    """One support-specific allocation box in the best-bound queue."""

    lower: torch.Tensor
    upper: torch.Tensor
    growth_upper_bound: float
    add_lower_bound: float
    add_upper_bound: float
    depth: int


@dataclass
class _Incumbent:
    """Mutable best feasible allocation shared by pilot and continuation."""

    allocation: torch.Tensor
    mean_log_growth: float
    drawdown: torch.Tensor


class _NodeEvaluation(NamedTuple):
    """A bounded node and an optional feasible representative."""

    node: _BoxNode | None
    candidate: torch.Tensor | None
    candidate_growth: float
    candidate_drawdown: torch.Tensor | None


@torch.no_grad()
def try_add_branch_and_bound(  # pylint: disable=too-many-arguments,too-many-locals
    transposed_returns: torch.Tensor,
    raw_allocation: torch.Tensor | None,
    max_total_allocation: float,
    add_limit: float,
    maximal_cliques: list[tuple[int, ...]],
    selection_constraints: StrategySelectionConstraints | None,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float,
    grid_steps: int | None,
    *,
    bootstrap_drawdown_calculator=bootstrap_average_drawdown,
) -> AddBranchAndBoundOutcome:
    """Pilot and optionally complete an ADD-constrained spatial search.

    The objective is mean log growth; multiplying it by the fixed observation
    count would not change the winner.  ``allocation_tolerance`` is the maximum
    coordinate width of a resolved leaf.  A ``None`` allocation in the outcome
    recommends the existing grid and carries a feasible incumbent for it.
    """
    _validate_inputs(
        transposed_returns,
        max_total_allocation,
        add_limit,
        percentile,
        allocation_tolerance,
        grid_steps,
    )
    support_masks = _maximal_support_masks(
        transposed_returns.shape[0], selection_constraints
    )
    incumbent = _cash_incumbent(transposed_returns)
    if support_masks is None or len(support_masks) > _MAX_PILOT_SUPPORTS:
        return _fallback_outcome(incumbent, bootstrap_indices, 0)

    estimated_grid_candidates = _estimate_grid_candidates(
        transposed_returns.shape[0],
        max_total_allocation,
        maximal_cliques,
        selection_constraints,
        grid_steps,
    )
    pilot_node_budget = _pilot_node_budget(estimated_grid_candidates, bootstrap_indices)
    if len(support_masks) > pilot_node_budget:
        return _fallback_outcome(incumbent, bootstrap_indices, 0)

    if (
        raw_allocation is not None
        and bootstrap_indices is None
        and _allocation_is_feasible(
            raw_allocation,
            transposed_returns,
            max_total_allocation,
            maximal_cliques,
            selection_constraints,
        )
    ):
        _consider_ray_incumbent(
            raw_allocation,
            incumbent,
            transposed_returns,
            add_limit,
            bootstrap_indices,
            percentile,
            allocation_tolerance,
            bootstrap_drawdown_calculator,
        )

    queue: list[tuple[float, int, _BoxNode]] = []
    serial = itertools.count()
    evaluated_nodes = 0
    surviving_children = 0
    split_nodes = 0
    for support_mask in support_masks:
        lower, upper = _root_box(
            support_mask,
            transposed_returns.shape[0],
            max_total_allocation,
            transposed_returns,
        )
        evaluation = _evaluate_box(
            lower,
            upper,
            0,
            transposed_returns,
            max_total_allocation,
            maximal_cliques,
            add_limit,
            bootstrap_indices,
            percentile,
            incumbent.mean_log_growth,
            bootstrap_drawdown_calculator,
        )
        evaluated_nodes += 1
        _accept_evaluation(evaluation, incumbent)
        if evaluation.node is not None:
            heapq.heappush(
                queue,
                (-evaluation.node.growth_upper_bound, next(serial), evaluation.node),
            )
        if bootstrap_indices is None:
            _consider_ray_incumbent(
                _feasible_box_midpoint(
                    lower, upper, maximal_cliques, max_total_allocation
                ),
                incumbent,
                transposed_returns,
                add_limit,
                bootstrap_indices,
                percentile,
                allocation_tolerance,
                bootstrap_drawdown_calculator,
            )

    while queue and evaluated_nodes < pilot_node_budget:
        survived, evaluated = _expand_best_node(
            queue,
            serial,
            incumbent,
            transposed_returns,
            max_total_allocation,
            maximal_cliques,
            add_limit,
            bootstrap_indices,
            percentile,
            allocation_tolerance,
            bootstrap_drawdown_calculator,
        )
        if evaluated == 0:
            continue
        split_nodes += 1
        surviving_children += survived
        evaluated_nodes += evaluated

    if not queue:
        return _completed_outcome(incumbent, bootstrap_indices, evaluated_nodes)
    if not _pilot_is_promising(
        queue,
        split_nodes,
        surviving_children,
        evaluated_nodes,
        estimated_grid_candidates,
        allocation_tolerance,
        bootstrap_indices is not None,
    ):
        return _fallback_outcome(incumbent, bootstrap_indices, evaluated_nodes)

    while queue and evaluated_nodes < _MAX_SEARCH_NODES:
        _, evaluated = _expand_best_node(
            queue,
            serial,
            incumbent,
            transposed_returns,
            max_total_allocation,
            maximal_cliques,
            add_limit,
            bootstrap_indices,
            percentile,
            allocation_tolerance,
            bootstrap_drawdown_calculator,
        )
        evaluated_nodes += evaluated
    if queue:
        return _fallback_outcome(incumbent, bootstrap_indices, evaluated_nodes)
    return _completed_outcome(incumbent, bootstrap_indices, evaluated_nodes)


def allocation_box_drawdown_bounds(
    lower: torch.Tensor,
    upper: torch.Tensor,
    transposed_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    *,
    bootstrap_drawdown_calculator=bootstrap_average_drawdown,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return outward-padded lower and upper ADD bounds for an allocation box.

    The daily linear-return extrema are relaxed to the containing axis-aligned
    box.  Applying the path drawdown operator to their log returns remains
    one-sided because every drawdown is the minimum of cumulative segment
    returns. Bootstrap quantiles preserve those bounds by monotonicity. These
    statements are exact in real arithmetic; the implementation adds an
    operation-count-scaled allowance for floating-point reductions.
    """
    center = (lower + upper) * 0.5
    radius = (upper - lower) * 0.5
    center_returns = center @ transposed_returns
    return_radius = radius @ transposed_returns.abs()
    lower_returns = center_returns - return_radius
    upper_returns = center_returns + return_radius
    scalar = torch.empty((), device=lower.device, dtype=transposed_returns.dtype)
    if bool(torch.any(upper_returns <= -1.0).item()):
        negative_infinity = scalar.fill_(-torch.inf)
        return negative_infinity, negative_infinity.clone()

    lower_is_solvent = not bool(torch.any(lower_returns <= -1.0).item())
    safe_lower_returns = (
        lower_returns if lower_is_solvent else torch.zeros_like(lower_returns)
    )
    bound_drawdowns = _path_average_drawdowns(
        torch.stack((safe_lower_returns, upper_returns)),
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )
    lower_add = bound_drawdowns[0] if lower_is_solvent else scalar.fill_(-1.0)
    upper_add = bound_drawdowns[1]
    path_length = (
        transposed_returns.shape[1]
        if bootstrap_indices is None
        else bootstrap_indices.shape[1]
    )
    padding = _rounding_padding(
        transposed_returns.dtype,
        transposed_returns.shape[0] + path_length,
    )
    lower_add = (lower_add - padding).clamp_min(-1.0)
    upper_add = (upper_add + padding).clamp_max(0.0)
    return lower_add, upper_add


def tighten_allocation_box(
    lower: torch.Tensor,
    upper: torch.Tensor,
    maximal_cliques: list[tuple[int, ...]],
    max_total_allocation: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Intersect coordinate bounds with every overlap-clique capital budget."""
    tightened_upper = upper.clone()
    tolerance = _rounding_padding(lower.dtype) * max(1.0, max_total_allocation)
    for clique in maximal_cliques:
        indices = torch.tensor(clique, device=lower.device, dtype=torch.int64)
        clique_lower = lower.index_select(0, indices)
        lower_sum = float(clique_lower.sum().item())
        if lower_sum > max_total_allocation + tolerance:
            return None
        slack = max(0.0, max_total_allocation - lower_sum)
        clique_upper = tightened_upper.index_select(0, indices)
        clique_upper = torch.minimum(clique_upper, clique_lower + slack)
        tightened_upper[indices] = clique_upper
    if bool(torch.any(tightened_upper + tolerance < lower).item()):
        return None
    tightened_upper = torch.maximum(tightened_upper, lower)
    return lower, tightened_upper


def _validate_inputs(
    transposed_returns: torch.Tensor,
    maximum: float,
    add_limit: float,
    percentile: float,
    allocation_tolerance: float,
    grid_steps: int | None,
) -> None:
    """Reject malformed direct calls to the internal public seam."""
    if not isinstance(transposed_returns, torch.Tensor):
        raise TypeError("transposed_returns must be a torch.Tensor")
    if transposed_returns.ndim != 2 or min(transposed_returns.shape) <= 0:
        raise ValueError("transposed_returns must have two nonempty dimensions")
    if transposed_returns.dtype not in (torch.float32, torch.float64):
        raise TypeError("transposed_returns must have dtype float32 or float64")
    if not bool(torch.isfinite(transposed_returns).all().item()):
        raise ValueError("transposed_returns must contain only finite values")
    for value, name in (
        (maximum, "max_total_allocation"),
        (allocation_tolerance, "allocation_tolerance"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        invalid_sign = (
            float(value) < 0.0
            if name == "max_total_allocation"
            else float(value) <= 0.0
        )
        if not math.isfinite(float(value)) or invalid_sign:
            qualifier = "nonnegative" if name == "max_total_allocation" else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
    if not -1.0 <= add_limit <= 0.0:
        raise ValueError("add_limit must be in [-1, 0]")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if grid_steps is not None and grid_steps <= 0:
        raise ValueError("grid_steps must be positive or None")


def _evaluate_box(  # pylint: disable=too-many-arguments
    lower: torch.Tensor,
    upper: torch.Tensor,
    depth: int,
    transposed_returns: torch.Tensor,
    maximum: float,
    maximal_cliques: list[tuple[int, ...]],
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    incumbent_growth: float,
    bootstrap_drawdown_calculator,
) -> _NodeEvaluation:
    """Bound one box and evaluate one capital-feasible representative."""
    tightened = tighten_allocation_box(lower, upper, maximal_cliques, maximum)
    if tightened is None:
        return _NodeEvaluation(None, None, -math.inf, None)
    lower, upper = tightened
    add_lower, add_upper = allocation_box_drawdown_bounds(
        lower,
        upper,
        transposed_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator=bootstrap_drawdown_calculator,
    )
    add_lower_value = float(add_lower.item())
    add_upper_value = float(add_upper.item())
    if add_upper_value < add_limit:
        return _NodeEvaluation(None, None, -math.inf, None)

    candidate = _feasible_box_midpoint(lower, upper, maximal_cliques, maximum)
    growth_upper = _tangent_growth_upper_bound(
        candidate,
        lower,
        upper,
        transposed_returns,
        maximal_cliques,
        maximum,
    )
    if growth_upper <= incumbent_growth:
        return _NodeEvaluation(None, None, -math.inf, None)
    candidate_growth = _mean_log_growth(candidate, transposed_returns)
    feasible_candidate = None
    candidate_drawdown = None
    if candidate_growth > incumbent_growth:
        candidate_drawdown = _allocation_drawdown(
            candidate,
            transposed_returns,
            bootstrap_indices,
            percentile,
            bootstrap_drawdown_calculator,
        )
        if bool((candidate_drawdown >= add_limit).item()):
            feasible_candidate = candidate
    node = _BoxNode(
        lower,
        upper,
        growth_upper,
        add_lower_value,
        add_upper_value,
        depth,
    )
    return _NodeEvaluation(
        node, feasible_candidate, candidate_growth, candidate_drawdown
    )


def _feasible_box_midpoint(
    lower: torch.Tensor,
    upper: torch.Tensor,
    maximal_cliques: list[tuple[int, ...]],
    maximum: float,
) -> torch.Tensor:
    """Return a midpoint contracted only as much as clique budgets require."""
    increments = (upper - lower) * 0.5
    scale = 1.0
    for clique in maximal_cliques:
        indices = torch.tensor(clique, device=lower.device, dtype=torch.int64)
        used = float(lower.index_select(0, indices).sum().item())
        increment = float(increments.index_select(0, indices).sum().item())
        if increment > 0.0:
            scale = min(scale, max(0.0, maximum - used) / increment)
    return lower + min(1.0, scale) * increments


def _tangent_growth_upper_bound(
    representative: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    transposed_returns: torch.Tensor,
    maximal_cliques: list[tuple[int, ...]],
    maximum: float,
) -> float:
    """Return a concavity tangent bound over a safe polytope relaxation."""
    tangent = _solvent_ray_point(representative, transposed_returns)
    wealth = 1.0 + tangent @ transposed_returns
    growth = torch.log(wealth).mean()
    gradient = (transposed_returns / wealth.unsqueeze(0)).mean(dim=1)
    linear_corner = _linear_maximizer(gradient, lower, upper, maximal_cliques, maximum)
    upper_bound = growth + gradient @ (linear_corner - tangent)
    padding = _rounding_padding(
        transposed_returns.dtype,
        transposed_returns.shape[0] + transposed_returns.shape[1],
    )
    return float(upper_bound.item()) + padding


def _linear_maximizer(
    coefficients: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    maximal_cliques: list[tuple[int, ...]],
    maximum: float,
) -> torch.Tensor:
    """Maximize a linear form over partitioned cliques or a safe box.

    Pairwise-disjoint maximal cliques form independent bounded simplexes, for
    which a fractional-knapsack fill is exact.  Intersecting cliques use the
    containing coordinate box; this is looser but remains a valid upper bound.
    """
    assigned = 0
    for clique in maximal_cliques:
        clique_mask = sum(1 << strategy for strategy in clique)
        if assigned & clique_mask:
            return torch.where(coefficients >= 0.0, upper, lower)
        assigned |= clique_mask

    maximizer = lower.clone()
    for clique in maximal_cliques:
        indices = torch.tensor(clique, device=lower.device, dtype=torch.int64)
        local_coefficients = coefficients.index_select(0, indices)
        local_widths = (upper - lower).index_select(0, indices)
        residual = max(
            0.0,
            maximum - float(lower.index_select(0, indices).sum().item()),
        )
        order = torch.argsort(local_coefficients, descending=True)
        for local_position in order.detach().to(device="cpu").tolist():
            coefficient = float(local_coefficients[local_position].item())
            if coefficient <= 0.0 or residual <= 0.0:
                break
            increment = min(residual, float(local_widths[local_position].item()))
            maximizer[indices[local_position]] += increment
            residual -= increment

    uncovered = [
        strategy
        for strategy in range(coefficients.numel())
        if not assigned & (1 << strategy)
    ]
    if uncovered:
        indices = torch.tensor(uncovered, device=lower.device, dtype=torch.int64)
        independent_values = torch.where(
            coefficients.index_select(0, indices) >= 0.0,
            upper.index_select(0, indices),
            lower.index_select(0, indices),
        )
        maximizer[indices] = independent_values
    return maximizer


def _solvent_ray_point(
    allocation: torch.Tensor, transposed_returns: torch.Tensor
) -> torch.Tensor:
    """Move toward cash only when the requested tangent point is insolvent."""
    portfolio_returns = allocation @ transposed_returns
    if bool(torch.all(portfolio_returns > -1.0).item()):
        return allocation
    negative = portfolio_returns < 0.0
    boundary = torch.where(
        negative,
        -1.0 / portfolio_returns,
        torch.full_like(portfolio_returns, torch.inf),
    ).amin()
    scale = min(1.0, max(0.0, float(boundary.item()) * 0.5))
    return allocation * scale


def _expand_best_node(  # pylint: disable=too-many-arguments
    queue: list[tuple[float, int, _BoxNode]],
    serial,
    incumbent: _Incumbent,
    transposed_returns: torch.Tensor,
    maximum: float,
    maximal_cliques: list[tuple[int, ...]],
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float,
    bootstrap_drawdown_calculator,
) -> tuple[int, int]:
    """Split the best unresolved box and enqueue its surviving children."""
    _, _, node = heapq.heappop(queue)
    if node.growth_upper_bound <= incumbent.mean_log_growth:
        return 0, 0
    widths = node.upper - node.lower
    maximum_width, split_dimension = torch.max(widths, dim=0)
    if float(maximum_width.item()) <= allocation_tolerance:
        return 0, 0
    dimension = int(split_dimension.item())
    midpoint = (node.lower[dimension] + node.upper[dimension]) * 0.5
    if bool(
        (midpoint <= node.lower[dimension]).item()
        or (midpoint >= node.upper[dimension]).item()
    ):
        return 0, 0

    survived = 0
    evaluated = 0
    for upper_child in (False, True):
        child_lower = node.lower.clone()
        child_upper = node.upper.clone()
        if upper_child:
            child_lower[dimension] = midpoint
        else:
            child_upper[dimension] = midpoint
        evaluation = _evaluate_box(
            child_lower,
            child_upper,
            node.depth + 1,
            transposed_returns,
            maximum,
            maximal_cliques,
            add_limit,
            bootstrap_indices,
            percentile,
            incumbent.mean_log_growth,
            bootstrap_drawdown_calculator,
        )
        evaluated += 1
        _accept_evaluation(evaluation, incumbent)
        if evaluation.node is not None:
            survived += 1
            heapq.heappush(
                queue,
                (-evaluation.node.growth_upper_bound, next(serial), evaluation.node),
            )
    return survived, evaluated


def _accept_evaluation(evaluation: _NodeEvaluation, incumbent: _Incumbent) -> None:
    """Promote a feasible representative when its growth strictly improves."""
    if (
        evaluation.candidate is not None
        and evaluation.candidate_drawdown is not None
        and evaluation.candidate_growth > incumbent.mean_log_growth
    ):
        incumbent.allocation = evaluation.candidate
        incumbent.mean_log_growth = evaluation.candidate_growth
        incumbent.drawdown = evaluation.candidate_drawdown


def _consider_ray_incumbent(  # pylint: disable=too-many-arguments
    direction: torch.Tensor,
    incumbent: _Incumbent,
    transposed_returns: torch.Tensor,
    add_limit: float,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    allocation_tolerance: float,
    bootstrap_drawdown_calculator,
) -> None:
    """Optimize growth then ADD feasibility along one cash-to-portfolio ray."""
    raw_scale = _kelly_ray_scale(direction, transposed_returns)
    if raw_scale <= 0.0:
        return
    raw = direction * raw_scale
    raw_drawdown = _allocation_drawdown(
        raw,
        transposed_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )
    if bool((raw_drawdown >= add_limit).item()):
        candidate, drawdown = raw, raw_drawdown
    else:
        lower_scale, upper_scale = 0.0, raw_scale
        candidate = torch.zeros_like(direction)
        drawdown = torch.zeros_like(raw_drawdown)
        maximum_coordinate = max(float(direction.abs().amax().item()), 1.0)
        iteration_limit = 64 if direction.dtype == torch.float64 else 32
        for _ in range(iteration_limit):
            if (upper_scale - lower_scale) * maximum_coordinate <= allocation_tolerance:
                break
            midpoint = (lower_scale + upper_scale) * 0.5
            if midpoint in (lower_scale, upper_scale):
                break
            midpoint_allocation = direction * midpoint
            midpoint_drawdown = _allocation_drawdown(
                midpoint_allocation,
                transposed_returns,
                bootstrap_indices,
                percentile,
                bootstrap_drawdown_calculator,
            )
            if bool((midpoint_drawdown >= add_limit).item()):
                lower_scale = midpoint
                candidate = midpoint_allocation
                drawdown = midpoint_drawdown
            else:
                upper_scale = midpoint
    growth = _mean_log_growth(candidate, transposed_returns)
    if growth > incumbent.mean_log_growth:
        incumbent.allocation = candidate
        incumbent.mean_log_growth = growth
        incumbent.drawdown = drawdown


def _kelly_ray_scale(
    direction: torch.Tensor, transposed_returns: torch.Tensor
) -> float:
    """Return the raw-Kelly growth maximizer between cash and ``direction``."""
    ray_returns = direction @ transposed_returns
    initial_derivative = float(ray_returns.mean().item())
    if initial_derivative <= 0.0:
        return 0.0
    negative = ray_returns < 0.0
    solvency = torch.where(
        negative,
        -1.0 / ray_returns,
        torch.full_like(ray_returns, torch.inf),
    ).amin()
    upper = min(
        1.0, float(solvency.item()) * (1.0 - 8.0 * torch.finfo(ray_returns.dtype).eps)
    )
    upper_wealth = 1.0 + upper * ray_returns
    upper_derivative = float((ray_returns / upper_wealth).mean().item())
    if upper_derivative >= 0.0:
        return upper
    lower = 0.0
    for _ in range(64 if ray_returns.dtype == torch.float64 else 32):
        midpoint = (lower + upper) * 0.5
        derivative = float((ray_returns / (1.0 + midpoint * ray_returns)).mean().item())
        if derivative > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def _allocation_drawdown(
    allocation: torch.Tensor,
    transposed_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_drawdown_calculator,
) -> torch.Tensor:
    """Evaluate exact historical or shared-bootstrap ADD for one allocation."""
    portfolio_returns = allocation @ transposed_returns
    if not bool(torch.all(portfolio_returns > -1.0).item()):
        return torch.full(
            (), -torch.inf, device=allocation.device, dtype=allocation.dtype
        )
    return _path_average_drawdown(
        portfolio_returns,
        bootstrap_indices,
        percentile,
        bootstrap_drawdown_calculator,
    )


def _path_average_drawdown(
    portfolio_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_drawdown_calculator,
) -> torch.Tensor:
    """Apply the production ADD definition to one daily-return sequence."""
    if bootstrap_indices is not None:
        return bootstrap_drawdown_calculator(
            portfolio_returns.unsqueeze(0), bootstrap_indices, percentile
        )[0]
    log_returns = portfolio_returns.log1p()
    return negative_rms_drawdown(log_returns, dim=0)


def _path_average_drawdowns(
    portfolio_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor | None,
    percentile: float,
    bootstrap_drawdown_calculator,
) -> torch.Tensor:
    """Apply the production ADD definition to a batch of return paths."""
    if bootstrap_indices is not None:
        return bootstrap_drawdown_calculator(
            portfolio_returns, bootstrap_indices, percentile
        )
    log_returns = portfolio_returns.log1p()
    return negative_rms_drawdown(log_returns, dim=1)


def _mean_log_growth(
    allocation: torch.Tensor, transposed_returns: torch.Tensor
) -> float:
    """Return normalized log growth, or negative infinity when insolvent."""
    wealth = 1.0 + allocation @ transposed_returns
    if not bool(torch.all(wealth > 0.0).item()):
        return -math.inf
    return float(torch.log(wealth).mean().item())


def _cash_incumbent(transposed_returns: torch.Tensor) -> _Incumbent:
    """Return the always-feasible zero-growth incumbent."""
    allocation = torch.zeros(
        transposed_returns.shape[0],
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    drawdown = torch.zeros(
        (), device=transposed_returns.device, dtype=transposed_returns.dtype
    )
    return _Incumbent(allocation, 0.0, drawdown)


def _pilot_is_promising(
    queue: list[tuple[float, int, _BoxNode]],
    split_nodes: int,
    surviving_children: int,
    evaluated_nodes: int,
    estimated_grid_candidates: int | None,
    allocation_tolerance: float,
    bootstrap_enabled: bool,
) -> bool:
    """Project continuation cost from observed pruning and frontier widths."""
    if estimated_grid_candidates is None:
        return True
    if split_nodes < 4:
        return False
    effective_branching = surviving_children / split_nodes
    if effective_branching >= 1.95:
        return False
    projected_nodes = float(evaluated_nodes)
    for _, _, node in queue:
        remaining_depth = _remaining_box_depth(node, allocation_tolerance)
        projected_nodes += _project_descendants(effective_branching, remaining_depth)
        if projected_nodes > estimated_grid_candidates:
            return False
    relative_node_cost = 3.0 if bootstrap_enabled else 2.0
    return (
        projected_nodes * relative_node_cost
        <= estimated_grid_candidates * _GRID_ADVANTAGE_FACTOR
    )


def _pilot_node_budget(
    estimated_grid_candidates: int | None,
    bootstrap_indices: torch.Tensor | None,
) -> int:
    """Limit the pilot by grid size and serial bootstrap path work."""
    if estimated_grid_candidates is None:
        budget = _PILOT_NODE_BUDGET
    elif estimated_grid_candidates < 800:
        return 0
    else:
        budget = min(
            _PILOT_NODE_BUDGET,
            max(8, math.ceil(estimated_grid_candidates * 0.01)),
        )
    if bootstrap_indices is None:
        return budget
    # Each box evaluates two interval paths and, at most, one incumbent path.
    # Eager node evaluation is serial, whereas the grid scorer batches candidates;
    # large bootstrap workloads should therefore bypass the pilot altogether.
    if bootstrap_indices.shape[0] > _MAX_BOOTSTRAP_PILOT_RUNS:
        return 0
    per_node_elements = 3 * bootstrap_indices.numel()
    if per_node_elements <= 0:
        return 0
    return min(budget, _BOOTSTRAP_PILOT_ELEMENT_BUDGET // per_node_elements)


def _remaining_box_depth(node: _BoxNode, allocation_tolerance: float) -> int:
    """Estimate binary coordinate splits needed along one surviving path."""
    depth = 0
    for width in (node.upper - node.lower).detach().to(device="cpu").tolist():
        if width > allocation_tolerance:
            depth += math.ceil(math.log2(width / allocation_tolerance))
    return depth


def _project_descendants(branching: float, depth: int) -> float:
    """Return a capped geometric descendant projection."""
    if depth <= 0:
        return 1.0
    if branching < 1.0:
        return min(depth + 1.0, 1.0 / max(1e-6, 1.0 - branching))
    if branching <= 1.05:
        return depth + 1.0
    exponent = min(depth + 1, 64)
    return min(1e30, (branching**exponent - 1.0) / (branching - 1.0))


def _estimate_grid_candidates(
    strategy_count: int,
    maximum: float,
    maximal_cliques: list[tuple[int, ...]],
    constraints: StrategySelectionConstraints | None,
    grid_steps: int | None,
) -> int | None:
    """Return a cheap conservative upper estimate of initial grid candidates."""
    if grid_steps is None:
        return None
    units = math.floor(maximum * grid_steps + 8.0 * math.ulp(maximum * grid_steps))
    ordinary_simplex = (
        strategy_count == 1
        or len(maximal_cliques) == 1
        and len(maximal_cliques[0]) == strategy_count
    )
    if ordinary_simplex:
        estimate = math.comb(units + strategy_count, strategy_count)
    elif not maximal_cliques:
        estimate = (units + 1) ** strategy_count
    else:
        estimate = (units + 1) ** strategy_count
        for clique in maximal_cliques:
            constrained = math.comb(units + len(clique), len(clique))
            relaxed = constrained * (units + 1) ** (strategy_count - len(clique))
            estimate = min(estimate, relaxed)
    if constraints is not None:
        maximum_active = min(constraints.maximum_active, strategy_count)
        support_estimate = sum(
            math.comb(strategy_count, active) * units**active
            for active in range(maximum_active + 1)
        )
        estimate = min(estimate, support_estimate)
    return estimate


def _rounding_padding(dtype: torch.dtype, operation_count: int = 1) -> float:
    """Return an operation-scaled outward allowance for reduction roundoff."""
    return min(
        1.0,
        _BOUND_ROUNDING_ULPS * torch.finfo(dtype).eps * max(1, operation_count),
    )


def _completed_outcome(
    incumbent: _Incumbent,
    bootstrap_indices: torch.Tensor | None,
    pilot_nodes: int,
) -> AddBranchAndBoundOutcome:
    """Build a successful spatial-search outcome."""
    bootstrap_drawdown = incumbent.drawdown if bootstrap_indices is not None else None
    return AddBranchAndBoundOutcome(
        incumbent.allocation,
        incumbent.allocation,
        incumbent.mean_log_growth,
        bootstrap_drawdown,
        BRANCH_AND_BOUND_ALGORITHM,
        True,
        pilot_nodes,
    )


def _fallback_outcome(
    incumbent: _Incumbent,
    bootstrap_indices: torch.Tensor | None,
    pilot_nodes: int,
) -> AddBranchAndBoundOutcome:
    """Build an outcome that asks the caller to run its complete grid."""
    bootstrap_drawdown = incumbent.drawdown if bootstrap_indices is not None else None
    return AddBranchAndBoundOutcome(
        None,
        incumbent.allocation,
        incumbent.mean_log_growth,
        bootstrap_drawdown,
        PILOT_GRID_FALLBACK_ALGORITHM,
        False,
        pilot_nodes,
    )
