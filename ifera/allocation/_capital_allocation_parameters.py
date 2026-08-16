"""Validation helpers for capital-allocation search parameters."""

from __future__ import annotations

import math

import torch

SUPPORTED_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
_RECIPROCAL_ULP_TOLERANCE = 4


def validate_returns(returns: torch.Tensor) -> None:
    """Validate a return-series tensor."""
    if not isinstance(returns, torch.Tensor):
        raise TypeError("returns must be a torch.Tensor")
    if returns.ndim != 2:
        raise ValueError("returns must have shape (time, strategies)")
    if returns.shape[0] == 0 or returns.shape[1] == 0:
        raise ValueError("returns must contain at least one time and one strategy")
    if returns.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            "returns must have a float16, bfloat16, float32, or float64 dtype"
        )
    if bool(torch.isinf(returns).any().item()):
        raise ValueError("returns must not contain infinite values")


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


def resolve_device(device: torch.device | str | None) -> torch.device:
    """Return the requested device or the preferred default device."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_calculation_dtype(
    returns_dtype: torch.dtype, allocation_dtype: torch.dtype
) -> torch.dtype:
    """Choose a common calculation dtype with at least float32 precision."""
    calculation_dtype = torch.promote_types(returns_dtype, allocation_dtype)
    if torch.empty((), dtype=calculation_dtype).element_size() < 4:
        return torch.float32
    return calculation_dtype


def resolve_grid_steps(grid_increment: float) -> int:
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


def resolve_allocation_unit_limit(max_total_allocation: float, grid_steps: int) -> int:
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
