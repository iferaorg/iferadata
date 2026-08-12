"""Validation helpers for capital-allocation search parameters."""

from __future__ import annotations

import math


def validate_alpha(alpha: float) -> float:
    """Return ``alpha`` as a validated Python float."""
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError("alpha must be a real number") from exc
    if not math.isfinite(alpha_value) or alpha_value < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    return alpha_value


def validate_add_limit(add_limit: float | None) -> float | None:
    """Return an optional inclusive ADD floor as a validated Python float."""
    if add_limit is None:
        return None
    if isinstance(add_limit, bool):
        raise TypeError("ADD_limit must be a real number or None")
    try:
        add_limit_value = float(add_limit)
    except (TypeError, ValueError) as exc:
        raise TypeError("ADD_limit must be a real number or None") from exc
    if not math.isfinite(add_limit_value) or not -1.0 <= add_limit_value <= 0.0:
        raise ValueError("ADD_limit must be finite and in [-1, 0]")
    return add_limit_value


def validate_max_total_allocation(max_total_allocation: float) -> float:
    """Return a validated nonnegative allocation cap."""
    if isinstance(max_total_allocation, bool):
        raise TypeError("max_total_allocation must be a real number")
    try:
        max_total_allocation_value = float(max_total_allocation)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_total_allocation must be a real number") from exc
    if (
        not math.isfinite(max_total_allocation_value)
        or max_total_allocation_value < 0.0
    ):
        raise ValueError("max_total_allocation must be finite and nonnegative")
    return max_total_allocation_value
