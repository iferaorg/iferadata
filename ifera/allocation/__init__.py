"""Capital-allocation optimization, reporting, and walk-forward simulation."""

# pyright: reportUnsupportedDunderAll=false
# pylint: disable=undefined-all-variable

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CapitalAllocationDiagnostics": (
        "ifera.allocation._capital_allocation_diagnostics",
        "CapitalAllocationDiagnostics",
    ),
    "PortfolioAllocationResult": (
        "ifera.allocation.portfolio_allocation",
        "PortfolioAllocationResult",
    ),
    "PortfolioDiversification": (
        "ifera.allocation.portfolio_allocation",
        "PortfolioDiversification",
    ),
    "PortfolioStatistics": (
        "ifera.allocation.portfolio_allocation",
        "PortfolioStatistics",
    ),
    "WalkForwardFold": (
        "ifera.allocation.portfolio_allocation",
        "WalkForwardFold",
    ),
    "WalkForwardPortfolioAllocationResult": (
        "ifera.allocation.portfolio_allocation",
        "WalkForwardPortfolioAllocationResult",
    ),
    "find_optimal_capital_allocation": (
        "ifera.allocation.capital_allocation",
        "find_optimal_capital_allocation",
    ),
    "find_optimal_capital_allocation_from_csv": (
        "ifera.allocation.portfolio_allocation",
        "find_optimal_capital_allocation_from_csv",
    ),
    "walk_forward_capital_allocation_from_csv": (
        "ifera.allocation.portfolio_allocation",
        "walk_forward_capital_allocation_from_csv",
    ),
}

__all__ = [
    "CapitalAllocationDiagnostics",
    "PortfolioAllocationResult",
    "PortfolioDiversification",
    "PortfolioStatistics",
    "WalkForwardFold",
    "WalkForwardPortfolioAllocationResult",
    "find_optimal_capital_allocation",
    "find_optimal_capital_allocation_from_csv",
    "walk_forward_capital_allocation_from_csv",
]


def __getattr__(name: str) -> Any:
    """Resolve public names lazily to keep tensor-only imports lightweight."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes, including lazy public exports."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
