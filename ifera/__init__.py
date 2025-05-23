"""
ifera
"""

from .config import BrokerConfig, ConfigManager, InstrumentConfig
from .enums import Scheme, Source
from .data_loading import load_data, load_data_tensor
from .data_models import InstrumentData, DataManager
from .data_processing import (
    aggregate_large_quote_file,
    process_data,
    calculate_rollover,
)
from .file_manager import FileManager, RuleType
from .market_simulator import MarketSimulatorIntraday
from .masked_series import masked_artr, masked_ema, masked_rtr, masked_sma
from .policies import *
from .series import artr, ema, ema_slow, ffill, rtr, sma
from .settings import settings
from .s3_utils import list_s3_objects, download_s3_file, upload_s3_file, delete_s3_file

__all__ = [
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
    "FileManager",
    "Scheme",
    "Source",
    "RuleType",
    "calculate_rollover",
    "list_s3_objects",
    "download_s3_file",
    "upload_s3_file",
    "delete_s3_file",
]
