"""
Data models for financial instruments.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

SECONDS_IN_DAY = 86400


def to_camel(string: str) -> str:
    """Convert snake_case strings to camelCase."""
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class BaseInstrumentConfig(BaseModel):
    """
    Pydantic v2 model for an instrument's base configuration.
    """

    # JSON -> Model Fields
    symbol: str
    description: str
    currency: str
    type: str
    interval: str
    trading_start: pd.Timedelta = Field(..., alias="tradingStart")
    trading_end: pd.Timedelta = Field(..., alias="tradingEnd")
    liquid_start: pd.Timedelta = Field(..., alias="liquidStart")
    liquid_end: pd.Timedelta = Field(..., alias="liquidEnd")
    regular_start: pd.Timedelta = Field(..., alias="regularStart")
    regular_end: pd.Timedelta = Field(..., alias="regularEnd")
    contract_multiplier: int = Field(..., alias="contractMultiplier")
    tick_size: float = Field(..., alias="tickSize")
    remove_dates: Optional[List[datetime.date]] = Field(
        None, alias="removeDates", validate_default=True
    )
    last_update: Optional[float] = Field(default=None)
    # Derived Fields
    time_step: pd.Timedelta = pd.Timedelta(0)
    end_time: pd.Timedelta = pd.Timedelta(0)
    total_steps: int = 0

    @field_validator(
        "trading_start",
        "trading_end",
        "liquid_start",
        "liquid_end",
        "regular_start",
        "regular_end",
        mode="before",
    )
    @classmethod
    def parse_timedelta(cls, value):
        """Parse timedelta from string value."""
        try:
            return pd.to_timedelta(value)
        except Exception as exc:
            raise ValueError(
                f"Error parsing timedelta from value {value}: {exc}"
            ) from exc

    @field_validator("remove_dates", mode="before")
    @classmethod
    def parse_remove_dates(cls, value):
        """Parse remove_dates from string values."""
        if value is None:
            return None
        try:
            return [pd.to_datetime(date_str).date() for date_str in value]
        except Exception as exc:
            raise ValueError(f"Error parsing remove_dates: {exc}") from exc

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "BaseInstrumentConfig":
        """Compute derived fields after validation."""
        try:
            if self.interval is None:
                raise ValueError("Interval is required.")
            self.time_step = pd.to_timedelta(self.interval)
            if self.trading_start is None or self.trading_end is None:
                raise ValueError("Both trading_start and trading_end are required.")
            all_steps = pd.timedelta_range(
                start=pd.Timedelta(0), end=pd.Timedelta(days=1), freq=self.time_step
            )
            self.end_time = all_steps[
                all_steps < self.trading_end - self.trading_start
            ][-1]
            total_seconds = self.end_time.total_seconds()
            step_seconds = self.time_step.total_seconds()
            if step_seconds <= 0:
                raise ValueError("Invalid time_step: must be positive.")
            self.total_steps = int(total_seconds / step_seconds) + 1
            if self.last_update is None:
                self.last_update = -float("inf")
        except Exception as exc:
            raise ValueError(f"Error computing derived fields: {exc}") from exc
        return self

    model_config = {
        "arbitrary_types_allowed": True,  # allow pandas.Timedelta
        "alias_generator": to_camel,  # snake_case -> camelCase
        "populate_by_name": True,  # allow field population by pythonic names
    }


class BrokerInstrumentConfig(BaseModel):
    """
    Pydantic v2 model for broker-specific instrument configuration.
    """

    instrument_symbol: str = Field(..., alias="instrumentSymbol")
    broker_symbol: str = Field(..., alias="brokerSymbol")
    margin: float
    commission: float
    min_commission: float = Field(..., alias="minCommission")
    max_commission_pct: float = Field(..., alias="maxCommissionPct")
    slippage: float
    min_slippage: float = Field(..., alias="minSlippage")
    reference_price: float = Field(..., alias="referencePrice")

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }


class InstrumentConfig(BaseInstrumentConfig, BrokerInstrumentConfig):
    """
    Combined instrument configuration with both base and broker-specific data.
    """

    broker_name: str = Field("", description="Name of the broker")
    parent_config: Optional["InstrumentConfig"] = Field(None, exclude=True)

    model_config = {
        "arbitrary_types_allowed": True,
        "alias_generator": to_camel,
        "populate_by_name": True,
    }


class BrokerConfig(BaseModel):
    """
    Pydantic v2 model for broker configuration.
    """

    name: str
    instruments: Dict[str, BrokerInstrumentConfig]

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }


class ConfigManager:
    """
    Loads and manages instrument configurations from JSON.
    Implemented as a singleton to ensure only one instance exists.
    """

    _instance = None  # Singleton instance

    def __new__(
        cls,
        instruments_filename: str = "data/instruments.json",
        brokers_filename: str = "data/brokers.json",
    ):
        _, _ = instruments_filename, brokers_filename
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        instruments_filename: str = "data/instruments.json",
        brokers_filename: str = "data/brokers.json",
    ):
        """Initialize with configuration filenames (once)."""
        if getattr(self, "_initialized", False):
            return

        self.instruments_filename = instruments_filename
        self.brokers_filename = brokers_filename
        self.last_instruments_update: Optional[float] = None
        self.last_brokers_update: Optional[float] = None
        self.instruments_data: Dict[str, Dict[str, Any]] = {}
        self.brokers_data: Dict[str, Dict[str, Any]] = {}
        # Cache for InstrumentConfig instances
        self._config_cache: Dict[Tuple[str, str], InstrumentConfig] = {}
        # Cache for derived configs with different intervals
        self._derived_cache: Dict[Tuple[str, str, str], InstrumentConfig] = {}
        self._load_data()
        self._initialized = True

    def _load_data(self):
        """Load data from the JSON files."""
        # Load instruments data
        try:
            with open(self.instruments_filename, "r", encoding="utf-8") as f:
                self.instruments_data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Configuration file '{self.instruments_filename}' not found."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from '{self.instruments_filename}': {e}"
            ) from e

        # Load brokers data
        try:
            with open(self.brokers_filename, "r", encoding="utf-8") as f:
                self.brokers_data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Configuration file '{self.brokers_filename}' not found."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from '{self.brokers_filename}': {e}"
            ) from e

        try:
            path = Path(self.instruments_filename)
            self.last_instruments_update = path.stat().st_mtime
            path = Path(self.brokers_filename)
            self.last_brokers_update = path.stat().st_mtime
        except Exception as e:
            raise OSError(f"Error accessing configuration files: {e}") from e

        # Clear caches when data is reloaded
        self._config_cache.clear()
        self._derived_cache.clear()

    def reload_if_updated(self):
        """Reload configs if files have been modified."""
        try:
            instruments_mtime = Path(self.instruments_filename).stat().st_mtime
            brokers_mtime = Path(self.brokers_filename).stat().st_mtime
        except Exception as e:
            raise OSError(f"Error accessing configuration files: {e}") from e

        if (
            self.last_instruments_update is None
            or instruments_mtime > self.last_instruments_update
            or self.last_brokers_update is None
            or brokers_mtime > self.last_brokers_update
        ):
            self._load_data()

    def get_base_instrument_config(self, instrument_key: str) -> BaseInstrumentConfig:
        """Get base configuration for a specific instrument."""
        self.reload_if_updated()
        try:
            instrument_dict = self.instruments_data[instrument_key]
        except KeyError as e:
            raise KeyError(
                f"Instrument key '{instrument_key}' not found in '{self.instruments_filename}'."
            ) from e
        try:
            return BaseInstrumentConfig(
                **instrument_dict, last_update=self.last_instruments_update
            )
        except Exception as e:
            raise ValueError(
                f"Error creating BaseInstrumentData for '{instrument_key}': {e}"
            ) from e

    def get_broker_config(self, broker_name: str) -> BrokerConfig:
        """Get configuration for a specific broker."""
        self.reload_if_updated()
        try:
            broker_dict = self.brokers_data[broker_name]
        except KeyError as e:
            raise KeyError(
                f"Broker configuration '{broker_name}' not found in '{self.brokers_filename}'."
            ) from e
        try:
            return BrokerConfig(**broker_dict)
        except Exception as e:
            raise ValueError(
                f"Error creating BrokerData for '{broker_name}': {e}"
            ) from e

    def get_config(self, broker_name: str, instrument_key: str) -> InstrumentConfig:
        """Get combined configuration for a specific instrument and broker."""
        self.reload_if_updated()

        # Check if we have a cached config for this combination
        cache_key = (broker_name, instrument_key)
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Get base instrument configuration
        base_config = self.get_base_instrument_config(instrument_key)

        # Get broker configuration and find matching instrument
        broker_config = self.get_broker_config(broker_name)
        try:
            broker_instrument_config = broker_config.instruments[base_config.symbol]
        except KeyError as e:
            raise KeyError(
                f"No broker configuration found for instrument '{base_config.symbol}' "
                f"with broker '{broker_name}'"
            ) from e

        # Combine the configurations
        combined_dict = {
            **base_config.model_dump(),
            **broker_instrument_config.model_dump(),
            "last_update": max(
                self.last_instruments_update or float("-inf"),
                self.last_brokers_update or float("-inf"),
            ),
            "broker_name": broker_name,
        }

        # Create and cache the config
        config = InstrumentConfig(**combined_dict)
        self._config_cache[cache_key] = config

        return config

    def create_derived_config(
        self, parent_config: InstrumentConfig, new_interval: str
    ) -> InstrumentConfig:
        """
        Create a derived InstrumentConfig with a different interval.

        Args:
            parent_config: The parent InstrumentConfig
            new_interval: The new interval string (e.g., '5m', '1h', '1d')

        Returns:
            A new InstrumentConfig with updated interval and derived fields

        Raises:
            ValueError: If the new interval doesn't meet requirements
        """
        # Check if we already have this derived config in the cache
        cache_key = (parent_config.broker_name, parent_config.symbol, new_interval)
        if cache_key in self._derived_cache:
            return self._derived_cache[cache_key]

        # Convert intervals to seconds for validation
        parent_step_seconds = parent_config.time_step.total_seconds()
        new_time_step = pd.to_timedelta(new_interval)
        new_step_seconds = new_time_step.total_seconds()

        # Validate interval relationships
        if new_step_seconds < parent_step_seconds:
            raise ValueError(
                f"Child interval ({new_interval}) must be greater than or equal to "
                f"parent interval ({parent_config.interval})"
            )

        if new_step_seconds % parent_step_seconds != 0:
            raise ValueError(
                f"Child step seconds ({new_step_seconds}) must be an integer multiple of "
                f"parent step seconds ({parent_step_seconds})"
            )

        # For intervals less than a day, ensure they divide evenly into a day
        if new_step_seconds < SECONDS_IN_DAY:
            if SECONDS_IN_DAY % new_step_seconds != 0:
                raise ValueError(
                    f"For intervals less than a day, the interval ({new_interval}) "
                    f"must divide evenly into {SECONDS_IN_DAY} seconds"
                )
        # For intervals greater than a day, ensure they're multiples of a day
        elif new_step_seconds % SECONDS_IN_DAY != 0:
            raise ValueError(
                f"For intervals greater than a day, the interval ({new_interval}) "
                f"must be an integer multiple of a day ({SECONDS_IN_DAY} seconds)"
            )

        # Create a copy of the parent config dictionary and update with new values
        config_dict = parent_config.model_dump()
        config_dict["interval"] = new_interval

        # Create new config with updated interval
        child_config = InstrumentConfig(**config_dict)

        # Set parent reference (excluded from serialization)
        child_config.parent_config = parent_config

        # Cache the derived config
        self._derived_cache[cache_key] = child_config

        return child_config

    def clear_cache(self):
        """Clear all configuration caches."""
        self._config_cache.clear()
        self._derived_cache.clear()
