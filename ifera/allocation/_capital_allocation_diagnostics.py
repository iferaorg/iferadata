"""Diagnostic output populated by the capital-allocation search."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class CapitalAllocationDiagnostics:
    """Optional details retained while selecting the optimal allocation."""

    bootstrap_average_drawdown: float | None = None


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
    scoring_enabled = bootstrap_on and drawdown_required
    diagnostics_enabled = bootstrap_on and diagnostics is not None
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
) -> torch.Tensor:
    """Populate requested diagnostics without repeating candidate bootstraps."""
    if diagnostics is None:
        return best_allocation
    bootstrap_drawdown = best_bootstrap_drawdown
    if bootstrap_indices is not None and bootstrap_drawdown is None:
        calculation_allocation = best_allocation.to(
            dtype=transposed_returns.dtype
        ).unsqueeze(0)
        portfolio_returns = calculation_allocation @ transposed_returns
        bootstrap_drawdown = bootstrap_calculator(
            portfolio_returns, bootstrap_indices, percentile
        )[0]
    diagnostics.bootstrap_average_drawdown = (
        None if bootstrap_drawdown is None else float(bootstrap_drawdown.item())
    )
    return best_allocation


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
