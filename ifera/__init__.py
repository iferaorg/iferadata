"""Lazy package exports for the ``ifera`` package."""

# pyright: reportUnsupportedDunderAll=false
# pylint: disable=undefined-all-variable

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseInstrumentConfig": ("ifera.config", "BaseInstrumentConfig"),
    "InstrumentConfig": ("ifera.config", "InstrumentConfig"),
    "ConfigManager": ("ifera.config", "ConfigManager"),
    "BrokerConfig": ("ifera.config", "BrokerConfig"),
    "load_data": ("ifera.data_loading", "load_data"),
    "load_data_tensor": ("ifera.data_loading", "load_data_tensor"),
    "process_data": ("ifera.data_processing", "process_data"),
    "aggregate_large_quote_file": (
        "ifera.data_processing",
        "aggregate_large_quote_file",
    ),
    "settings": ("ifera.settings", "settings"),
    "sma": ("ifera.series", "sma"),
    "ema": ("ifera.series", "ema"),
    "ema_slow": ("ifera.series", "ema_slow"),
    "ffill": ("ifera.series", "ffill"),
    "rtr": ("ifera.series", "rtr"),
    "artr": ("ifera.series", "artr"),
    "masked_sma": ("ifera.masked_series", "masked_sma"),
    "masked_ema": ("ifera.masked_series", "masked_ema"),
    "masked_rtr": ("ifera.masked_series", "masked_rtr"),
    "masked_artr": ("ifera.masked_series", "masked_artr"),
    "InstrumentData": ("ifera.data_models", "InstrumentData"),
    "DataManager": ("ifera.data_models", "DataManager"),
    "MarketSimulatorIntraday": (
        "ifera.market_simulator",
        "MarketSimulatorIntraday",
    ),
    "PolicyBase": ("ifera.policies", "PolicyBase"),
    "TradingPolicy": ("ifera.policies", "TradingPolicy"),
    "AlwaysOpenPolicy": ("ifera.policies", "AlwaysOpenPolicy"),
    "OpenOncePolicy": ("ifera.policies", "OpenOncePolicy"),
    "ArtrStopLossPolicy": ("ifera.policies", "ArtrStopLossPolicy"),
    "InitialArtrStopLossPolicy": (
        "ifera.policies",
        "InitialArtrStopLossPolicy",
    ),
    "ScaledArtrMaintenancePolicy": (
        "ifera.policies",
        "ScaledArtrMaintenancePolicy",
    ),
    "PercentGainMaintenancePolicy": (
        "ifera.policies",
        "PercentGainMaintenancePolicy",
    ),
    "AlwaysFalseDonePolicy": ("ifera.policies", "AlwaysFalseDonePolicy"),
    "SingleTradeDonePolicy": ("ifera.policies", "SingleTradeDonePolicy"),
    "clone_trading_policy_for_devices": (
        "ifera.policies",
        "clone_trading_policy_for_devices",
    ),
    "SingleMarketEnv": ("ifera.environments", "SingleMarketEnv"),
    "MultiGPUSingleMarketEnv": (
        "ifera.environments",
        "MultiGPUSingleMarketEnv",
    ),
    "FileManager": ("ifera.file_manager", "FileManager"),
    "Scheme": ("ifera.enums", "Scheme"),
    "Source": ("ifera.enums", "Source"),
    "RuleType": ("ifera.file_manager", "RuleType"),
    "calculate_rollover": ("ifera.data_processing", "calculate_rollover"),
    "list_s3_objects": ("ifera.s3_utils", "list_s3_objects"),
    "download_s3_file": ("ifera.s3_utils", "download_s3_file"),
    "upload_s3_file": ("ifera.s3_utils", "upload_s3_file"),
    "delete_s3_file": ("ifera.s3_utils", "delete_s3_file"),
    "ThreadSafeCache": ("ifera.decorators", "ThreadSafeCache"),
    "calculate_expiration": ("ifera.date_utils", "calculate_expiration"),
    "ExpirationRule": ("ifera.enums", "ExpirationRule"),
    "check_s3_file_exists": ("ifera.s3_utils", "check_s3_file_exists"),
    "rename_s3_file": ("ifera.s3_utils", "rename_s3_file"),
    "parse_trade_log": ("ifera.optionalpha", "parse_trade_log"),
    "parse_filter_log": ("ifera.optionalpha", "parse_filter_log"),
    "get_filters": ("ifera.optionalpha", "get_filters"),
    "prepare_splits": ("ifera.optionalpha", "prepare_splits"),
    "Split": ("ifera.optionalpha", "Split"),
    "SplitGenerator": ("ifera.optionalpha", "SplitGenerator"),
}

__all__ = [
    "BaseInstrumentConfig",
    "InstrumentConfig",
    "ConfigManager",
    "BrokerConfig",
    "load_data",
    "load_data_tensor",
    "process_data",
    "aggregate_large_quote_file",
    "settings",
    "sma",
    "ema",
    "ema_slow",
    "ffill",
    "rtr",
    "artr",
    "masked_sma",
    "masked_ema",
    "masked_rtr",
    "masked_artr",
    "InstrumentData",
    "DataManager",
    "MarketSimulatorIntraday",
    "TradingPolicy",
    "AlwaysOpenPolicy",
    "ArtrStopLossPolicy",
    "InitialArtrStopLossPolicy",
    "ScaledArtrMaintenancePolicy",
    "AlwaysFalseDonePolicy",
    "SingleTradeDonePolicy",
    "SingleMarketEnv",
    "MultiGPUSingleMarketEnv",
    "FileManager",
    "Scheme",
    "Source",
    "RuleType",
    "calculate_rollover",
    "list_s3_objects",
    "download_s3_file",
    "upload_s3_file",
    "delete_s3_file",
    "ThreadSafeCache",
    "calculate_expiration",
    "ExpirationRule",
    "check_s3_file_exists",
    "rename_s3_file",
    "parse_trade_log",
    "parse_filter_log",
    "get_filters",
    "prepare_splits",
    "Split",
    "SplitGenerator",
]


def __getattr__(name: str) -> Any:
    """Resolve exported names lazily to keep package import side effects minimal."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes, including lazy exports."""

    return sorted(set(globals()) | set(_LAZY_EXPORTS))
