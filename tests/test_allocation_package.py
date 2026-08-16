"""Public API tests for the allocation package."""

from __future__ import annotations

from importlib.util import find_spec

import ifera.allocation as allocation
from ifera.allocation import (
    CapitalAllocationDiagnostics,
    PortfolioAllocationResult,
    PortfolioDiversification,
    PortfolioStatistics,
    WalkForwardFold,
    WalkForwardPortfolioAllocationResult,
    find_optimal_capital_allocation,
    find_optimal_capital_allocation_from_csv,
    walk_forward_capital_allocation_from_csv,
)
from ifera.allocation._capital_allocation_diagnostics import (
    CapitalAllocationDiagnostics as DiagnosticsImplementation,
)
from ifera.allocation.capital_allocation import (
    find_optimal_capital_allocation as OptimizerImplementation,
)
from ifera.allocation import portfolio_allocation
from ifera.allocation.portfolio_allocation import (
    PortfolioAllocationResult as PortfolioResultImplementation,
    PortfolioDiversification as DiversificationImplementation,
    PortfolioStatistics as StatisticsImplementation,
    WalkForwardFold as WalkForwardFoldImplementation,
    WalkForwardPortfolioAllocationResult as WalkForwardResultImplementation,
)

PUBLIC_API = {
    "CapitalAllocationDiagnostics",
    "PortfolioAllocationResult",
    "PortfolioDiversification",
    "PortfolioStatistics",
    "WalkForwardFold",
    "WalkForwardPortfolioAllocationResult",
    "find_optimal_capital_allocation",
    "find_optimal_capital_allocation_from_csv",
    "walk_forward_capital_allocation_from_csv",
}


def test_allocation_package_exposes_complete_public_api_directly():
    assert set(allocation.__all__) == PUBLIC_API
    assert CapitalAllocationDiagnostics is DiagnosticsImplementation
    assert PortfolioAllocationResult is PortfolioResultImplementation
    assert PortfolioDiversification is DiversificationImplementation
    assert PortfolioStatistics is StatisticsImplementation
    assert WalkForwardFold is WalkForwardFoldImplementation
    assert WalkForwardPortfolioAllocationResult is WalkForwardResultImplementation
    assert find_optimal_capital_allocation is OptimizerImplementation
    assert (
        find_optimal_capital_allocation_from_csv
        is portfolio_allocation.find_optimal_capital_allocation_from_csv
    )
    assert (
        walk_forward_capital_allocation_from_csv
        is portfolio_allocation.walk_forward_capital_allocation_from_csv
    )


def test_allocation_modules_are_not_kept_at_legacy_package_paths():
    legacy_modules = (
        "ifera.capital_allocation",
        "ifera.portfolio_allocation",
        "ifera.walk_forward_allocation",
        "ifera._capital_allocation_diagnostics",
        "ifera._capital_allocation_drawdown",
        "ifera._capital_allocation_memory",
        "ifera._capital_allocation_parameters",
        "ifera._capital_allocation_refinement",
    )

    assert all(find_spec(module_name) is None for module_name in legacy_modules)
