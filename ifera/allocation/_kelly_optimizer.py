"""Projection-free optimization of the concave Kelly growth objective."""

# Pylint's repository-wide similarity check flags small, unrelated tensor
# initializers and argument-forwarding blocks against this numerical module.
# pylint: disable=duplicate-code

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Literal

import torch

from ._capital_allocation_parameters import validate_allocation_tolerance

KellyFeasibleSet = Literal["simplex", "box"]

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_DEFAULT_FLOAT32_TOLERANCE = 1e-6
_DEFAULT_FLOAT64_TOLERANCE = 1e-9
_DEFAULT_MAX_ITERATIONS = 10_000


@dataclass(frozen=True)
class KellyOptimizationResult:
    """A feasible allocation and its Frank-Wolfe optimality certificate."""

    allocation: torch.Tensor
    mean_log_growth: float
    frank_wolfe_gap: float
    iterations: int
    converged: bool

    @property
    def upper_bound(self) -> float:
        """Return the concavity upper bound on the optimal mean log growth."""
        return self.mean_log_growth + self.frank_wolfe_gap


@torch.no_grad()
def maximize_kelly_growth(
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    feasible_set: KellyFeasibleSet,
    allowed_mask: torch.Tensor | None = None,
    *,
    tolerance: float | None = None,
    allocation_tolerance: float | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> KellyOptimizationResult:
    """Maximize mean log growth over a simplex or a nonnegative box.

    ``transposed_returns`` has shape ``(strategies, time)`` and must already
    contain finite calculation values. A simplex constrains the sum of all
    allocations to ``max_total_allocation``. A box applies that upper bound to
    each strategy independently. ``allowed_mask`` fixes excluded strategies at
    exactly zero, which lets a caller reuse this solver inside branch-and-bound.

    The away-step Frank-Wolfe method keeps every iterate feasible and uses a
    safeguarded exact line search. The reported Frank-Wolfe gap gives

    ``optimal_mean_growth <= mean_log_growth + frank_wolfe_gap``.

    The objective is normalized by the number of observations so ``tolerance``
    has the same meaning for differently sized training windows.
    ``allocation_tolerance`` is the shared allocation-space search resolution.
    Frank--Wolfe stops after an exact line-search update moves every allocation
    coordinate by no more than that threshold. This makes the rule invariant to
    the scale of the returns, but it remains a practical iterate-resolution
    rule rather than a distance certificate when the optimum is non-unique or
    the objective is not strongly concave.
    """
    uses_default_tolerance = tolerance is None and allocation_tolerance is None
    (
        maximum,
        resolved_tolerance,
        iteration_limit,
        allowed,
        resolved_allocation_tolerance,
    ) = _validate_inputs(
        transposed_returns,
        max_total_allocation,
        feasible_set,
        allowed_mask,
        tolerance,
        allocation_tolerance,
        max_iterations,
    )
    strategy_count = transposed_returns.shape[0]
    zero = torch.zeros(
        strategy_count,
        device=transposed_returns.device,
        dtype=transposed_returns.dtype,
    )
    if maximum == 0.0 or not bool(torch.any(allowed).item()):
        return KellyOptimizationResult(zero, 0.0, 0.0, 0, True)

    atoms: dict[int, torch.Tensor] = {0: zero}
    weights: dict[int, float] = {0: 1.0}
    allocation = zero
    _, gradient, wealth = _growth_state(transposed_returns, allocation)
    if uses_default_tolerance:
        _, initial_atom = _linear_oracle(gradient, allowed, maximum, feasible_set)
        initial_gap = _directional_gain(gradient, initial_atom, allocation)
        if initial_gap > 0.0:
            resolved_tolerance = min(
                resolved_tolerance,
                initial_gap * 1e-5,
            )
    completed_steps = 0
    allocation_resolution_reached = False

    for _ in range(iteration_limit):
        forward_key, forward_atom = _linear_oracle(
            gradient, allowed, maximum, feasible_set
        )
        forward_gap = _directional_gain(gradient, forward_atom, allocation)
        if forward_gap <= 0.0:
            allocation_resolution_reached = True
            break
        if resolved_allocation_tolerance is None and forward_gap <= resolved_tolerance:
            break

        away_key, away_atom, away_gain = _away_atom(
            atoms, weights, gradient, allocation
        )
        use_away = away_gain > forward_gap and weights[away_key] < 1.0
        if use_away:
            direction = allocation - away_atom
            away_weight = weights[away_key]
            maximum_step = away_weight / (1.0 - away_weight)
        else:
            direction = forward_atom - allocation
            maximum_step = 1.0

        directional_returns = direction @ transposed_returns
        step = _optimal_line_step(
            wealth,
            directional_returns,
            maximum_step,
            transposed_returns.dtype,
        )
        if step <= 0.0:
            break
        allocation_displacement = float(torch.amax(torch.abs(step * direction)).item())

        if use_away:
            _apply_away_step(weights, away_key, step, atoms)
        else:
            atoms.setdefault(forward_key, forward_atom)
            _apply_forward_step(weights, forward_key, step, atoms)
        allocation = _allocation_from_atoms(atoms, weights)
        _, gradient, wealth = _growth_state(transposed_returns, allocation)
        completed_steps += 1
        if (
            resolved_allocation_tolerance is not None
            and allocation_displacement <= resolved_allocation_tolerance
        ):
            allocation_resolution_reached = True
            break

    return _finalize_result(
        transposed_returns,
        allocation,
        allowed,
        maximum,
        feasible_set,
        resolved_tolerance,
        resolved_allocation_tolerance,
        completed_steps,
        allocation_resolution_reached,
    )


def _finalize_result(
    transposed_returns: torch.Tensor,
    iterative_allocation: torch.Tensor,
    allowed: torch.Tensor,
    maximum: float,
    feasible_set: KellyFeasibleSet,
    tolerance: float,
    allocation_tolerance: float | None,
    completed_steps: int,
    allocation_resolution_reached: bool,
) -> KellyOptimizationResult:
    """Clean an iterate without weakening its final optimality certificate."""
    allocation = _clean_allocation(
        iterative_allocation,
        allowed,
        maximum,
        feasible_set,
        allocation_tolerance if allocation_tolerance is not None else tolerance,
        transposed_returns,
    )
    score, gradient, _ = _growth_state(transposed_returns, allocation)
    _, final_atom = _linear_oracle(gradient, allowed, maximum, feasible_set)
    final_gap = _directional_gain(gradient, final_atom, allocation)
    if final_gap > tolerance and not torch.equal(allocation, iterative_allocation):
        allocation = iterative_allocation
        score, gradient, _ = _growth_state(transposed_returns, allocation)
        _, final_atom = _linear_oracle(gradient, allowed, maximum, feasible_set)
        final_gap = _directional_gain(gradient, final_atom, allocation)
    score_value = float(score.item())
    return KellyOptimizationResult(
        allocation,
        score_value,
        final_gap,
        completed_steps,
        (
            final_gap <= tolerance
            if allocation_tolerance is None
            else allocation_resolution_reached or final_gap == 0.0
        ),
    )


def _validate_inputs(
    transposed_returns: torch.Tensor,
    max_total_allocation: float,
    feasible_set: str,
    allowed_mask: torch.Tensor | None,
    tolerance: float | None,
    allocation_tolerance: float | None,
    max_iterations: int,
) -> tuple[float, float, int, torch.Tensor, float | None]:
    """Validate inputs and materialize the allowed-strategy mask."""
    if not isinstance(transposed_returns, torch.Tensor):
        raise TypeError("transposed_returns must be a torch.Tensor")
    if transposed_returns.ndim != 2 or min(transposed_returns.shape) <= 0:
        raise ValueError("transposed_returns must have nonempty strategy and time axes")
    if transposed_returns.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("transposed_returns must have dtype float32 or float64")
    if not bool(torch.isfinite(transposed_returns).all().item()):
        raise ValueError("transposed_returns must contain only finite values")
    if feasible_set not in ("simplex", "box"):
        raise ValueError("feasible_set must be 'simplex' or 'box'")

    maximum = _finite_nonnegative(max_total_allocation, "max_total_allocation")
    resolved_allocation_tolerance = validate_allocation_tolerance(allocation_tolerance)
    resolved_tolerance = _resolve_tolerance(tolerance, transposed_returns.dtype)
    iteration_limit = _positive_integer(max_iterations, "max_iterations")
    allowed = _resolve_allowed_mask(allowed_mask, transposed_returns)
    return (
        maximum,
        resolved_tolerance,
        iteration_limit,
        allowed,
        resolved_allocation_tolerance,
    )


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


def _resolve_tolerance(value: float | None, dtype: torch.dtype) -> float:
    """Return a finite positive normalized-growth tolerance."""
    if value is None:
        return (
            _DEFAULT_FLOAT64_TOLERANCE
            if dtype == torch.float64
            else _DEFAULT_FLOAT32_TOLERANCE
        )
    if isinstance(value, bool):
        raise TypeError("tolerance must be a real number or None")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("tolerance must be a real number or None") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return result


def _positive_integer(value: int, name: str) -> int:
    """Return a strictly positive integer."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _resolve_allowed_mask(
    value: torch.Tensor | None, transposed_returns: torch.Tensor
) -> torch.Tensor:
    """Return a Boolean strategy mask on the calculation device."""
    strategy_count = transposed_returns.shape[0]
    if value is None:
        return torch.ones(
            strategy_count, device=transposed_returns.device, dtype=torch.bool
        )
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError("allowed_mask must be a Boolean tensor or None")
    if value.shape != (strategy_count,):
        raise ValueError("allowed_mask must have one value per strategy")
    return value.to(device=transposed_returns.device)


def _growth_state(
    transposed_returns: torch.Tensor, allocation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean log growth, its gradient, and current wealth multipliers."""
    wealth = 1.0 + allocation @ transposed_returns
    if not bool(torch.all(wealth > 0.0).item()):
        raise RuntimeError("the Kelly optimizer produced an insolvent allocation")
    reciprocal_wealth = torch.reciprocal(wealth)
    score = torch.log(wealth).mean()
    gradient = torch.mv(transposed_returns, reciprocal_wealth)
    gradient /= transposed_returns.shape[1]
    return score, gradient, wealth


def _linear_oracle(
    gradient: torch.Tensor,
    allowed: torch.Tensor,
    maximum: float,
    feasible_set: KellyFeasibleSet,
) -> tuple[int, torch.Tensor]:
    """Maximize the linearized objective over a simplex or box."""
    atom = torch.zeros_like(gradient)
    if feasible_set == "simplex":
        masked_gradient = gradient.masked_fill(~allowed, -torch.inf)
        strategy = int(torch.argmax(masked_gradient).item())
        if float(masked_gradient[strategy].item()) <= 0.0:
            return 0, atom
        atom[strategy] = maximum
        return 1 << strategy, atom

    positive = allowed & (gradient > 0.0)
    atom[positive] = maximum
    return _positive_mask_key(positive), atom


def _positive_mask_key(mask: torch.Tensor) -> int:
    """Encode a Boolean atom as an arbitrary-width Python bit mask."""
    key = 0
    for strategy in torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist():
        key |= 1 << strategy
    return key


def _directional_gain(
    gradient: torch.Tensor, atom: torch.Tensor, allocation: torch.Tensor
) -> float:
    """Return a nonnegative Frank-Wolfe directional gain."""
    return max(float(torch.dot(gradient, atom - allocation).item()), 0.0)


def _away_atom(
    atoms: dict[int, torch.Tensor],
    weights: dict[int, float],
    gradient: torch.Tensor,
    allocation: torch.Tensor,
) -> tuple[int, torch.Tensor, float]:
    """Return the active atom with the smallest linearized objective."""
    keys = tuple(weights)
    active_atoms = torch.stack([atoms[key] for key in keys])
    linear_values = torch.mv(active_atoms, gradient)
    away_position = int(torch.argmin(linear_values).item())
    away_key = keys[away_position]
    current_linear_value = torch.dot(gradient, allocation)
    away_gain = max(
        float((current_linear_value - linear_values[away_position]).item()), 0.0
    )
    return away_key, atoms[away_key], away_gain


def _optimal_line_step(
    wealth: torch.Tensor,
    directional_returns: torch.Tensor,
    maximum_step: float,
    dtype: torch.dtype,
) -> float:
    """Maximize log growth on a feasible line segment by safeguarded Newton."""
    upper = _solvent_upper_step(wealth, directional_returns, maximum_step)
    if upper <= 0.0:
        return 0.0
    upper_derivative = _line_derivative(wealth, directional_returns, upper)
    if upper_derivative >= 0.0:
        return upper

    lower = 0.0
    step = upper / 2.0
    relative_tolerance = 8.0 * torch.finfo(dtype).eps
    for _ in range(64):
        denominator = wealth + step * directional_returns
        ratios = directional_returns / denominator
        derivative = float(torch.mean(ratios).item())
        if derivative > 0.0:
            lower = step
        else:
            upper = step
        if upper - lower <= relative_tolerance * max(1.0, abs(step)):
            break

        curvature = -float(torch.mean(ratios.square()).item())
        newton_step = step - derivative / curvature if curvature < 0.0 else math.nan
        step = (
            newton_step
            if math.isfinite(newton_step) and lower < newton_step < upper
            else (lower + upper) / 2.0
        )
    return (lower + upper) / 2.0


def _solvent_upper_step(
    wealth: torch.Tensor,
    directional_returns: torch.Tensor,
    maximum_step: float,
) -> float:
    """Return the largest requested step strictly before any zero wealth."""
    zero = torch.zeros((), device=wealth.device, dtype=wealth.dtype)
    upper = torch.tensor(maximum_step, device=wealth.device, dtype=wealth.dtype)
    decreasing = directional_returns < 0.0
    if bool(torch.any(decreasing).item()):
        boundaries = -wealth[decreasing] / directional_returns[decreasing]
        first_boundary = torch.amin(boundaries)
        if bool((first_boundary <= upper).item()):
            upper = torch.nextafter(first_boundary, zero)

    for _ in range(64):
        if bool(torch.all(wealth + upper * directional_returns > 0.0).item()):
            return max(float(upper.item()), 0.0)
        upper = torch.nextafter(upper, zero)
    return 0.0


def _line_derivative(
    wealth: torch.Tensor, directional_returns: torch.Tensor, step: float
) -> float:
    """Return the derivative of mean log growth along one direction."""
    denominator = wealth + step * directional_returns
    return float(torch.mean(directional_returns / denominator).item())


def _apply_forward_step(
    weights: dict[int, float],
    forward_key: int,
    step: float,
    atoms: dict[int, torch.Tensor],
) -> None:
    """Update the active convex decomposition after a forward step."""
    for key in tuple(weights):
        weights[key] *= 1.0 - step
    weights[forward_key] = weights.get(forward_key, 0.0) + step
    _normalize_active_weights(weights, atoms)


def _apply_away_step(
    weights: dict[int, float],
    away_key: int,
    step: float,
    atoms: dict[int, torch.Tensor],
) -> None:
    """Update the active convex decomposition after an away step."""
    away_weight = weights[away_key]
    maximum_step = away_weight / (1.0 - away_weight)
    for key in tuple(weights):
        weights[key] *= 1.0 + step
    weights[away_key] -= step
    if math.isclose(step, maximum_step, rel_tol=1e-12, abs_tol=1e-15):
        weights[away_key] = 0.0
    _normalize_active_weights(weights, atoms)


def _normalize_active_weights(
    weights: dict[int, float], atoms: dict[int, torch.Tensor]
) -> None:
    """Drop roundoff-sized atoms and keep a normalized convex combination."""
    removed_weight = 0.0
    for key in tuple(weights):
        if weights[key] <= 1e-14:
            removed_weight += max(weights.pop(key), 0.0)
            if key != 0:
                atoms.pop(key)
    if removed_weight > 0.0:
        if 0 not in atoms:
            first_atom = next(iter(atoms.values()))
            atoms[0] = torch.zeros_like(first_atom)
        weights[0] = weights.get(0, 0.0) + removed_weight
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total


def _allocation_from_atoms(
    atoms: dict[int, torch.Tensor], weights: dict[int, float]
) -> torch.Tensor:
    """Reconstruct the feasible allocation from its active atoms."""
    keys = tuple(weights)
    active_atoms = torch.stack([atoms[key] for key in keys])
    active_weights = torch.tensor(
        [weights[key] for key in keys],
        device=active_atoms.device,
        dtype=active_atoms.dtype,
    )
    return active_weights @ active_atoms


def _clean_allocation(
    allocation: torch.Tensor,
    allowed: torch.Tensor,
    maximum: float,
    feasible_set: KellyFeasibleSet,
    tolerance: float,
    transposed_returns: torch.Tensor,
) -> torch.Tensor:
    """Return a numerically clean, feasible, and solvent allocation."""
    cleaned = allocation.clone()
    cleaned[~allowed] = 0.0
    cutoff = maximum * max(8.0 * torch.finfo(cleaned.dtype).eps, tolerance * 0.1)
    cleaned[cleaned <= cutoff] = 0.0
    cleaned.clamp_(min=0.0, max=maximum)
    if feasible_set == "simplex":
        total = float(cleaned.sum().item())
        if total > maximum:
            cleaned *= maximum / total
    if bool(torch.all(1.0 + cleaned @ transposed_returns > 0.0).item()):
        return cleaned
    return allocation
