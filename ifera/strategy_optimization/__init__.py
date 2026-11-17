"""Evolve entry filters and minute exit rules for single options strategies."""

# pyright: reportUnsupportedDunderAll=false
# pylint: disable=undefined-all-variable

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "OptimizationMode": ("._evaluation", "OptimizationMode"),
    "StrategyOptimizationResult": (".optimizer", "StrategyOptimizationResult"),
    "optimize_strategy_from_csv": (".optimizer", "optimize_strategy_from_csv"),
    "apply_strategy_conditions": ("._replay", "apply_strategy_conditions"),
    "FeatureSpec": ("._features", "FeatureSpec"),
    "load_features": ("._features", "load_features"),
    "EventCalendar": ("._events", "EventCalendar"),
    "EventYear": ("._event_sources", "EventYear"),
    "EventCalendarError": ("._event_sources", "EventCalendarError"),
    "OracleMarketDataSource": ("._data", "OracleMarketDataSource"),
    "annualized_return": (".metrics", "annualized_return"),
    "sortino": (".metrics", "sortino"),
    "return_over_add": (".metrics", "return_over_add"),
    "with_scarcity_penalty": (".metrics", "with_scarcity_penalty"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public exports lazily, following the allocation package."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in package introspection."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
