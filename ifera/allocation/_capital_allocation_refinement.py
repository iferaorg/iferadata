"""Grid geometry helpers for incremental capital-allocation refinement."""

from __future__ import annotations

import math
import operator

import torch

_ROUNDING_ULPS = 4.0
_VALIDATION_ELEMENT_BUDGET = 1_000_000


def validate_parameters(
    refinement_runs: int, refinement_divisor: float
) -> tuple[int, float]:
    """Return validated refinement settings."""
    if isinstance(refinement_runs, bool):
        raise TypeError("refinement_runs must be an integer")
    try:
        runs_value = operator.index(refinement_runs)
    except TypeError as exc:
        raise TypeError("refinement_runs must be an integer") from exc
    if runs_value < 0:
        raise ValueError("refinement_runs must be nonnegative")

    if isinstance(refinement_divisor, bool):
        raise TypeError("refinement_divisor must be a real number")
    try:
        divisor_value = float(refinement_divisor)
    except (TypeError, ValueError) as exc:
        raise TypeError("refinement_divisor must be a real number") from exc
    if not math.isfinite(divisor_value) or divisor_value <= 1.0:
        raise ValueError("refinement_divisor must be finite and greater than 1.0")
    return runs_value, divisor_value


def validate_final_increment(
    initial_increment: float, refinement_runs: int, refinement_divisor: float
) -> None:
    """Reject a requested refinement whose increment underflows."""
    try:
        divisor_power = refinement_divisor**refinement_runs
    except OverflowError as exc:
        raise ValueError("refined grid increment is too small") from exc
    refinement_increment = initial_increment / divisor_power
    if not math.isfinite(refinement_increment) or refinement_increment <= 0.0:
        raise ValueError("refined grid increment is too small")


def grid_lower_bounds(
    previous_allocation: torch.Tensor,
    allocation_unit_limit: int,
    refinement_increment: float,
    max_total_allocation: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shifted local-grid origins and the prior point's unit indices."""
    dtype_epsilon = float(torch.finfo(previous_allocation.dtype).eps)
    allocation_scale = max(1.0, max_total_allocation)
    allocation_tolerance = _ROUNDING_ULPS * dtype_epsilon * allocation_scale

    desired_start = -(allocation_unit_limit // 2)
    lower_bounds = []
    previous_indices = []
    for previous_value in (
        previous_allocation.detach().to(device="cpu", dtype=torch.float64).tolist()
    ):
        if (
            not -allocation_tolerance
            <= previous_value
            <= (max_total_allocation + allocation_tolerance)
        ):
            raise RuntimeError("previous allocation exceeds the allocation bounds")
        minimum_offset = min(
            _ceil_with_tolerance(-previous_value / refinement_increment, 0.0), 0
        )
        maximum_offset = max(
            _floor_with_tolerance(
                (max_total_allocation - previous_value) / refinement_increment,
                0.0,
            ),
            0,
        )
        minimum_start = max(minimum_offset, -allocation_unit_limit)
        maximum_start = min(maximum_offset - allocation_unit_limit, 0)
        if minimum_start > maximum_start:
            raise ValueError("refined grid cannot fit within the allocation bounds")
        start = min(max(desired_start, minimum_start), maximum_start)
        lower_bound = previous_value + start * refinement_increment
        if abs(lower_bound) <= allocation_tolerance:
            lower_bound = 0.0
        lower_bounds.append(lower_bound)
        previous_indices.append(-start)

    lower_bounds_tensor = torch.tensor(
        lower_bounds,
        device=previous_allocation.device,
        dtype=torch.float64,
    )
    previous_indices_tensor = torch.tensor(
        previous_indices,
        device=previous_allocation.device,
        dtype=torch.int64,
    )
    _validate_output_grid(
        previous_allocation,
        previous_indices_tensor,
        allocation_unit_limit,
        refinement_increment,
        max_total_allocation,
        allocation_tolerance,
    )
    return lower_bounds_tensor, previous_indices_tensor


def _validate_output_grid(
    previous_allocation: torch.Tensor,
    previous_indices: torch.Tensor,
    allocation_unit_limit: int,
    refinement_increment: float,
    max_total_allocation: float,
    allocation_tolerance: float,
) -> None:
    """Check that the output dtype preserves distinct, bounded grid points."""
    if allocation_unit_limit == 0:
        return
    intermediate_dtype = (
        torch.float64 if previous_allocation.dtype == torch.float64 else torch.float32
    )
    centers = previous_allocation.to(dtype=intermediate_dtype)
    upper_offsets = allocation_unit_limit - previous_indices
    last_points = (
        centers + upper_offsets.to(dtype=intermediate_dtype) * refinement_increment
    ).to(dtype=previous_allocation.dtype)
    _validate_distinct_points(
        centers,
        previous_indices,
        allocation_unit_limit,
        refinement_increment,
        previous_allocation.dtype,
    )

    first_points = (
        centers - previous_indices.to(dtype=intermediate_dtype) * refinement_increment
    ).to(dtype=previous_allocation.dtype)
    if bool(torch.any(first_points < -allocation_tolerance).item()) or bool(
        torch.any(last_points > max_total_allocation + allocation_tolerance).item()
    ):
        raise RuntimeError("refined grid exceeds the allocation bounds")


def _validate_distinct_points(
    centers: torch.Tensor,
    previous_indices: torch.Tensor,
    allocation_unit_limit: int,
    refinement_increment: float,
    output_dtype: torch.dtype,
) -> None:
    """Verify every adjacent local-grid point remains distinct after casting."""
    strategy_count = centers.numel()
    units_per_chunk = max(1, _VALIDATION_ELEMENT_BUDGET // strategy_count)
    intermediate_dtype = centers.dtype
    for start in range(0, allocation_unit_limit, units_per_chunk):
        stop = min(start + units_per_chunk, allocation_unit_limit)
        units = torch.arange(
            start,
            stop + 1,
            device=centers.device,
            dtype=torch.int64,
        ).unsqueeze(1)
        offsets = units - previous_indices.unsqueeze(0)
        points = (
            centers.unsqueeze(0)
            + offsets.to(dtype=intermediate_dtype) * refinement_increment
        ).to(dtype=output_dtype)
        if bool(torch.any(points[1:] <= points[:-1]).item()):
            raise ValueError("refined grid increment is too small for the output dtype")


def _floor_with_tolerance(value: float, tolerance: float) -> int:
    """Floor a grid-unit value while accepting near-integer roundoff."""
    nearest_integer = round(value)
    if abs(value - nearest_integer) <= max(tolerance, math.ulp(value) * _ROUNDING_ULPS):
        return nearest_integer
    return math.floor(value)


def _ceil_with_tolerance(value: float, tolerance: float) -> int:
    """Ceil a grid-unit value while accepting near-integer roundoff."""
    nearest_integer = round(value)
    if abs(value - nearest_integer) <= max(tolerance, math.ulp(value) * _ROUNDING_ULPS):
        return nearest_integer
    return math.ceil(value)


def clique_unit_limits(
    previous_allocation: torch.Tensor,
    previous_unit_indices: torch.Tensor,
    maximal_cliques: list[tuple[int, ...]],
    allocation_unit_limit: int,
    refinement_increment: float,
    max_total_allocation: float,
) -> torch.Tensor:
    """Return residual integer budgets for refined overlap cliques."""
    previous_values = (
        previous_allocation.detach().to(device="cpu", dtype=torch.float64).tolist()
    )
    previous_indices = previous_unit_indices.to(device="cpu").tolist()
    dtype_epsilon = float(torch.finfo(previous_allocation.dtype).eps)
    integer_limit = torch.iinfo(torch.int64).max
    unit_limits = []
    for clique in maximal_cliques:
        previous_sum = sum(previous_values[strategy] for strategy in clique)
        allocation_tolerance = (
            _ROUNDING_ULPS
            * dtype_epsilon
            * max(1.0, max_total_allocation, abs(previous_sum))
            * len(clique)
        )
        slack = max_total_allocation - previous_sum
        if slack < -allocation_tolerance:
            raise RuntimeError("previous allocation violates an overlap constraint")
        slack = max(slack, 0.0)
        extra_units = _floor_with_tolerance(
            slack / refinement_increment,
            0.0,
        )
        previous_usage = sum(previous_indices[strategy] for strategy in clique)
        maximum_usage = allocation_unit_limit * len(clique)
        clique_limit = min(previous_usage + max(extra_units, 0), maximum_usage)
        if clique_limit >= integer_limit:
            raise ValueError("refined allocation grid is too large to index")
        unit_limits.append(clique_limit)
    return torch.tensor(
        unit_limits,
        device=previous_allocation.device,
        dtype=torch.int64,
    )


def units_to_allocations(
    allocation_units: torch.Tensor,
    previous_allocation: torch.Tensor,
    previous_unit_indices: torch.Tensor,
    refinement_increment: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert local units to allocations relative to the exact prior point."""
    intermediate_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    allocation_offsets = allocation_units - previous_unit_indices
    allocations = allocation_offsets.to(dtype=intermediate_dtype)
    allocations.mul_(refinement_increment)
    allocations.add_(previous_allocation.to(dtype=intermediate_dtype))
    return allocations.to(dtype=dtype)
