"""Strategy-correlation and active-support constraints for allocation searches."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

_CORRELATION_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class StrategySelectionConstraints:
    """Prefix-search constraints for active strategy supports."""

    maximum_active: int
    prior_conflicts: tuple[torch.Tensor, ...]


def validate_strategy_selection_parameters(
    max_strategies: int | None,
    min_correlation: float | None,
    max_correlation: float | None,
) -> tuple[int | None, float | None, float | None]:
    """Return validated optional strategy-selection parameters."""
    maximum = _validate_max_strategies(max_strategies)
    minimum_correlation = _validate_correlation_bound(
        min_correlation, "min_correlation"
    )
    maximum_correlation = _validate_correlation_bound(
        max_correlation, "max_correlation"
    )
    if (
        minimum_correlation is not None
        and maximum_correlation is not None
        and minimum_correlation > maximum_correlation
    ):
        raise ValueError("min_correlation must not exceed max_correlation")
    return maximum, minimum_correlation, maximum_correlation


def resolve_strategy_selection_constraints(
    returns: torch.Tensor,
    max_strategies: int | None,
    min_correlation: float | None,
    max_correlation: float | None,
    device: torch.device,
) -> StrategySelectionConstraints | None:
    """Build effective cardinality and prior-pair conflict constraints."""
    maximum, minimum_correlation, maximum_correlation = (
        validate_strategy_selection_parameters(
            max_strategies, min_correlation, max_correlation
        )
    )
    strategy_count = returns.shape[1]
    maximum_active = strategy_count if maximum is None else min(maximum, strategy_count)
    conflicts = torch.zeros(
        (strategy_count, strategy_count), dtype=torch.bool, device="cpu"
    )
    if minimum_correlation is not None or maximum_correlation is not None:
        correlations = strategy_correlation_matrix(returns)
        compatible = torch.isfinite(correlations)
        if minimum_correlation is not None:
            compatible &= correlations >= minimum_correlation
        if maximum_correlation is not None:
            compatible &= correlations <= maximum_correlation
        conflicts = ~compatible
        conflicts.fill_diagonal_(False)

    if maximum_active == strategy_count and not bool(conflicts.any().item()):
        return None
    prior_conflicts = tuple(
        torch.nonzero(conflicts[strategy, :strategy], as_tuple=False)
        .flatten()
        .to(device=device, dtype=torch.int64)
        for strategy in range(strategy_count)
    )
    return StrategySelectionConstraints(maximum_active, prior_conflicts)


def strategy_correlation_matrix(returns: torch.Tensor) -> torch.Tensor:
    """Return CPU float64 Pearson correlations after replacing NaNs with zero."""
    filled_returns = torch.nan_to_num(
        returns.detach().to(device="cpu", dtype=torch.float64), nan=0.0
    )
    column_scales = filled_returns.abs().amax(dim=0)
    safe_scales = torch.where(
        column_scales > 0.0, column_scales, torch.ones_like(column_scales)
    )
    scaled_returns = filled_returns / safe_scales
    centered_returns = scaled_returns - scaled_returns.mean(dim=0, keepdim=True)
    norms = centered_returns.square().sum(dim=0).sqrt()
    cross_products = centered_returns.transpose(0, 1) @ centered_returns
    denominators = torch.outer(norms, norms)
    correlation_matrix = torch.full_like(cross_products, math.nan)
    nonconstant = torch.amax(scaled_returns, dim=0) > torch.amin(scaled_returns, dim=0)
    defined = (denominators > 0.0) & torch.outer(nonconstant, nonconstant)
    correlation_matrix[defined] = cross_products[defined] / denominators[defined]
    correlation_matrix.clamp_(-1.0, 1.0)
    correlation_matrix[correlation_matrix.abs() <= _CORRELATION_ZERO_TOLERANCE] = 0.0
    return correlation_matrix


def _validate_max_strategies(value: int | None) -> int | None:
    """Return an optional positive active-strategy limit."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("max_strategies must be an integer or None")
    try:
        maximum = operator.index(value)
    except TypeError as exc:
        raise TypeError("max_strategies must be an integer or None") from exc
    if maximum <= 0:
        raise ValueError("max_strategies must be positive")
    return maximum


def _validate_correlation_bound(value: float | None, name: str) -> float | None:
    """Return one optional finite Pearson-correlation bound."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number or None")
    try:
        bound = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number or None") from exc
    if not math.isfinite(bound) or not -1.0 <= bound <= 1.0:
        raise ValueError(f"{name} must be finite and in [-1, 1]")
    return bound
