"""Diagnostic output populated by the capital-allocation search."""

# The fast-path orchestrator deliberately mirrors the one-winner bootstrap
# memory validation performed here.
# pylint: disable=duplicate-code

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._capital_allocation_blocks import (
    AllocationBlocks,
    effective_portfolio_returns,
)
from ._capital_allocation_drawdown import bootstrap_average_drawdown
from ._capital_allocation_memory import validate_bootstrap_diagnostic_memory
from ._capital_allocation_parameters import validate_bootstrap_seed


@dataclass
class CapitalAllocationDiagnostics:
    """Optional details retained while selecting the optimal allocation."""

    bootstrap_average_drawdown: float | None = None
    algorithm: str | None = None
    request_bootstrap_average_drawdown: bool = True


def prepare_diagnostics(
    diagnostics: CapitalAllocationDiagnostics | None,
    bootstrap_on: bool,
    drawdown_required: bool,
) -> tuple[bool, bool, bool]:
    """Validate diagnostics and return bootstrap calculation modes."""
    if diagnostics is not None and not isinstance(
        diagnostics, CapitalAllocationDiagnostics
    ):
        raise TypeError("diagnostics must be a CapitalAllocationDiagnostics instance")
    if diagnostics is not None:
        diagnostics.bootstrap_average_drawdown = None
        diagnostics.algorithm = None
    scoring_enabled = bootstrap_on and drawdown_required
    diagnostics_enabled = (
        bootstrap_on
        and diagnostics is not None
        and diagnostics.request_bootstrap_average_drawdown
    )
    return (
        diagnostics_enabled,
        scoring_enabled or diagnostics_enabled,
        scoring_enabled and diagnostics is not None,
    )


def finalize_allocation_diagnostics(
    best_allocation: torch.Tensor,
    diagnostics: CapitalAllocationDiagnostics | None,
    bootstrap_indices: torch.Tensor | None,
    transposed_returns: torch.Tensor,
    percentile: float,
    best_bootstrap_drawdown: torch.Tensor | None,
    bootstrap_calculator: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    allocation_blocks: AllocationBlocks | None = None,
) -> torch.Tensor:
    """Populate requested diagnostics without repeating candidate bootstraps."""
    if diagnostics is None:
        return best_allocation
    bootstrap_drawdown = best_bootstrap_drawdown
    if bootstrap_indices is not None and bootstrap_drawdown is None:
        calculation_allocation = best_allocation.to(
            dtype=transposed_returns.dtype
        ).unsqueeze(0)
        portfolio_returns = effective_portfolio_returns(
            calculation_allocation,
            transposed_returns,
            None if allocation_blocks is None else allocation_blocks.day_indices,
            None if allocation_blocks is None else allocation_blocks.blocked_indices,
            None if allocation_blocks is None else allocation_blocks.blocker_incidence,
        )
        bootstrap_drawdown = bootstrap_calculator(
            portfolio_returns, bootstrap_indices, percentile
        )[0]
    diagnostics.bootstrap_average_drawdown = (
        None if bootstrap_drawdown is None else float(bootstrap_drawdown.item())
    )
    return best_allocation


def finalize_continuous_diagnostics(
    best_allocation: torch.Tensor,
    diagnostics: CapitalAllocationDiagnostics | None,
    bootstrap_on: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
    percentile: float,
    bootstrap_seed: int | None,
    transposed_returns: torch.Tensor,
    clique_count: int,
    allocation_blocks: AllocationBlocks | None = None,
) -> torch.Tensor:
    """Optionally bootstrap the one continuous winner for report diagnostics."""
    if diagnostics is None or not diagnostics.request_bootstrap_average_drawdown:
        return best_allocation
    if not isinstance(bootstrap_on, bool):
        raise TypeError("bootstrap_on must be a bool")
    if not bootstrap_on:
        return best_allocation

    (
        _,
        bootstrap_runs_value,
        bootstrap_length_value,
        percentile_value,
    ) = validate_bootstrap_parameters(
        bootstrap_on, bootstrap_runs, bootstrap_length, percentile
    )
    bootstrap_seed_value = validate_bootstrap_seed(bootstrap_seed)
    strategy_count, time_count = transposed_returns.shape
    validate_bootstrap_diagnostic_memory(
        time_count,
        strategy_count,
        clique_count,
        transposed_returns.dtype,
        transposed_returns.device,
        bootstrap_runs_value,
        bootstrap_length_value,
    )
    bootstrap_indices = sample_bootstrap_indices(
        time_count,
        bootstrap_runs_value,
        bootstrap_length_value,
        transposed_returns.device,
        bootstrap_seed_value,
    )
    return finalize_allocation_diagnostics(
        best_allocation,
        diagnostics,
        bootstrap_indices,
        transposed_returns,
        percentile_value,
        None,
        bootstrap_average_drawdown,
        allocation_blocks,
    )


def record_algorithm(
    diagnostics: CapitalAllocationDiagnostics | None, algorithm: str
) -> None:
    """Retain the algorithm selected by one optimizer invocation."""
    if diagnostics is not None:
        diagnostics.algorithm = algorithm


def sample_bootstrap_indices(
    time_count: int,
    bootstrap_runs: int,
    bootstrap_length: int,
    device: torch.device,
    bootstrap_seed: int | None,
) -> torch.Tensor:
    """Sample bootstrap rows, optionally from a reproducible local generator."""
    size = (bootstrap_runs, bootstrap_length)
    if bootstrap_seed is None:
        return torch.randint(time_count, size, device=device, dtype=torch.int64)
    generator = torch.Generator(device=device).manual_seed(bootstrap_seed)
    return torch.randint(
        time_count,
        size,
        device=device,
        dtype=torch.int64,
        generator=generator,
    )


def resolve_grid_bootstrap_parameters(
    bootstrap_on: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
    percentile: float,
    bootstrap_seed: int | None,
    diagnostics_requested: bool,
    drawdown_required: bool,
) -> tuple[bool, int, int, float, int | None]:
    """Resolve only bootstrap controls that can affect scoring or output."""
    if not drawdown_required and not diagnostics_requested:
        return False, 1, 1, 10.0, None
    if not drawdown_required and not isinstance(bootstrap_on, bool):
        raise TypeError("bootstrap_on must be a bool")
    if not drawdown_required and not bootstrap_on:
        return False, 1, 1, 10.0, None

    enabled, runs, length, resolved_percentile = validate_bootstrap_parameters(
        bootstrap_on,
        bootstrap_runs,
        bootstrap_length,
        percentile,
    )
    return (
        enabled,
        runs,
        length,
        resolved_percentile,
        validate_bootstrap_seed(bootstrap_seed),
    )


def validate_bootstrap_parameters(
    bootstrap_on: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
    percentile: float,
) -> tuple[bool, int, int, float]:
    """Return validated bootstrap settings."""
    if not isinstance(bootstrap_on, bool):
        raise TypeError("bootstrap_on must be a bool")
    runs_value = _positive_integer(bootstrap_runs, "bootstrap_runs")
    length_value = _positive_integer(bootstrap_length, "bootstrap_length")
    if isinstance(percentile, bool):
        raise TypeError("percentile must be a real number")
    try:
        percentile_value = float(percentile)
    except (TypeError, ValueError) as exc:
        raise TypeError("percentile must be a real number") from exc
    if not math.isfinite(percentile_value) or not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be finite and in [0, 100]")
    return bootstrap_on, runs_value, length_value, percentile_value


def _positive_integer(value: int, name: str) -> int:
    """Return a strictly positive integer parameter."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        integer_value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if integer_value <= 0:
        raise ValueError(f"{name} must be positive")
    return integer_value
