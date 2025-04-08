"""
Data models for financial instruments.
"""

import datetime
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from .decorators import singleton

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

    expiration_date: Optional[datetime.date] = Field(None, alias="expirationDate")
    rollover_date: Optional[datetime.date] = Field(None, alias="rolloverDate")
    contract_code: Optional[str] = Field(None, alias="contractCode")

    parent_config: Optional["BaseInstrumentConfig"] = Field(None, exclude=True)

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


@singleton
class ConfigManager:
    """
    Loads and manages instrument configurations from JSON.
    Implemented as a singleton to ensure only one instance exists.
    """

    def __init__(
        self,
        instruments_filename: str = "data/instruments.json",
        brokers_filename: str = "data/brokers.json",
    ):
        self.instruments_filename = instruments_filename
        self.brokers_filename = brokers_filename
        self.last_instruments_update: Optional[float] = None
        self.last_brokers_update: Optional[float] = None
        self.instruments_data: Dict[str, Dict[str, Any]] = {}
        self.brokers_data: Dict[str, Dict[str, Any]] = {}
        self._base_config_cache: Dict[str, BaseInstrumentConfig] = {}
        self._config_cache: Dict[Tuple[str, str], InstrumentConfig] = {}
        self._base_derived_cache: Dict[
            Tuple[str, Optional[str], Optional[datetime.date]], BaseInstrumentConfig
        ] = {}
        self._derived_cache: Dict[
            Tuple[str, str, Optional[str], Optional[datetime.date]], InstrumentConfig
        ] = {}
        self._load_data()

    def _load_data(self):
        """Load data from the JSON files."""

        with open(self.instruments_filename, "r", encoding="utf-8") as f:
            self.instruments_data = yaml.safe_load(f) or {}
        with open(self.brokers_filename, "r", encoding="utf-8") as f:
            self.brokers_data = yaml.safe_load(f) or {}

        self.last_instruments_update = Path(self.instruments_filename).stat().st_mtime
        self.last_brokers_update = Path(self.brokers_filename).stat().st_mtime

        # Clear caches when data is reloaded
        self._base_config_cache.clear()
        self._base_derived_cache.clear()
        self._config_cache.clear()
        self._derived_cache.clear()

    def reload_if_updated(self):
        """Reload configs if files have been modified."""
        instruments_mtime = Path(self.instruments_filename).stat().st_mtime
        brokers_mtime = Path(self.brokers_filename).stat().st_mtime

        if instruments_mtime > (self.last_instruments_update or 0) or brokers_mtime > (
            self.last_brokers_update or 0
        ):
            self._load_data()

    def get_base_instrument_config(self, instrument_key: str) -> BaseInstrumentConfig:
        """Get base configuration for a specific instrument."""
        self.reload_if_updated()

        if instrument_key in self._base_config_cache:
            return self._base_config_cache[instrument_key]

        try:
            instrument_dict = self.instruments_data[instrument_key]
        except KeyError as e:
            raise KeyError(f"Instrument key '{instrument_key}' not found.") from e

        # Merge with template if specified
        if "template" in instrument_dict:
            template_name = instrument_dict["template"]
            try:
                template_dict = self.instruments_data["templates"][template_name]
            except KeyError:
                raise KeyError(f"Template '{template_name}' not found.")
            combined_dict = {**template_dict, **instrument_dict}
            combined_dict.pop("template", None)
        else:
            combined_dict = instrument_dict

        config = BaseInstrumentConfig(
            **combined_dict, last_update=self.last_instruments_update
        )

        self._base_config_cache[instrument_key] = config
        return config

    def get_broker_config(self, broker_name: str) -> BrokerConfig:
        """Get configuration for a specific broker."""
        self.reload_if_updated()
        try:
            broker_dict = self.brokers_data[broker_name]
        except KeyError:
            raise KeyError(f"Broker '{broker_name}' not found.")

        instruments_dict = {}
        defaults = broker_dict.get("defaults", {})
        for symbol, instr in broker_dict["instruments"].items():
            instruments_dict[symbol] = BrokerInstrumentConfig(**{**defaults, **instr})

        return BrokerConfig(name=broker_dict["name"], instruments=instruments_dict)

    def _get_config_from_base(
        self, base_config: BaseInstrumentConfig, broker_name: str
    ) -> InstrumentConfig:
        """Get configuration for a specific instrument and broker."""

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

        if base_config.contract_code is not None:
            config.broker_symbol = (
                f"{broker_instrument_config.broker_symbol}{base_config.contract_code}"
            )

        return config

    def get_config(self, broker_name: str, instrument_key: str) -> InstrumentConfig:
        """Get combined configuration for a specific instrument and broker."""
        self.reload_if_updated()

        # Check if we have a cached config for this combination
        cache_key = (broker_name, instrument_key)
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Get base instrument configuration
        base_config = self.get_base_instrument_config(instrument_key)

        # Create and cache the config
        config = self._get_config_from_base(
            base_config=base_config, broker_name=broker_name
        )

        self._config_cache[cache_key] = config

        return config

    def create_derived_base_config(
        self,
        parent_config: InstrumentConfig,
        new_interval: Optional[str] = None,
        expiration_date: Optional[datetime.date] = None,
    ) -> BaseInstrumentConfig:
        """
        Create a derived BaseInstrumentConfig with a new interval and/or expiration date.

        Args:
            parent_config: The parent BaseInstrumentConfig
            new_interval: Optional new interval string (e.g., '5m', '1h')
            expiration_date: Optional expiration date for futures individual contracts

        Returns:
            A new BaseInstrumentConfig with updated fields

        Raises:
            ValueError: If neither new_interval nor expiration_date is provided,
                        or if expiration_date is used with non-futures instruments
        """
        if new_interval is None and expiration_date is None:
            raise ValueError(
                "At least one of new_interval or expiration_date must be provided."
            )

        # Check if we already have this derived config in the cache
        cache_key = (parent_config.symbol, new_interval, expiration_date)
        if cache_key in self._derived_cache:
            return self._base_derived_cache[cache_key]

        config_dict = parent_config.model_dump()

        if new_interval is not None:
            config_dict["interval"] = new_interval
            time_step = pd.to_timedelta(new_interval)
            parent_step_seconds = parent_config.time_step.total_seconds()
            new_step_seconds = time_step.total_seconds()

            # Validate interval (adapted from original logic)
            if new_step_seconds < parent_step_seconds:
                raise ValueError(
                    f"Child interval ({new_interval}) must be >= parent interval ({parent_config.interval})"
                )
            if new_step_seconds % parent_step_seconds != 0:
                raise ValueError(
                    f"Child interval ({new_interval}) must be a multiple of parent interval"
                )
            if (
                new_step_seconds < SECONDS_IN_DAY
                and SECONDS_IN_DAY % new_step_seconds != 0
            ):
                raise ValueError(
                    f"Interval ({new_interval}) must divide evenly into a day"
                )
            elif (
                new_step_seconds > SECONDS_IN_DAY
                and new_step_seconds % SECONDS_IN_DAY != 0
            ):
                raise ValueError(
                    f"Interval ({new_interval}) must be a multiple of a day"
                )

            # Recompute derived fields
            all_steps = pd.timedelta_range(
                start=pd.Timedelta(0), end=pd.Timedelta(days=1), freq=time_step
            )
            end_time = all_steps[
                all_steps < parent_config.trading_end - parent_config.trading_start
            ][-1]
            total_seconds = end_time.total_seconds()
            step_seconds = time_step.total_seconds()

            if step_seconds <= 0:
                raise ValueError("Invalid time_step: must be positive.")

            total_steps = int(total_seconds / step_seconds) + 1
            config_dict["time_step"] = time_step
            config_dict["end_time"] = end_time
            config_dict["total_steps"] = total_steps

        if expiration_date is not None:
            if parent_config.type != "futures":
                raise ValueError(
                    "expiration_date can only be set for futures instruments."
                )

            config_dict["expiration_date"] = expiration_date
            # Compute contract_code (e.g., "M24" for June 2024)
            month_code = "FGHJKMNQUVXZ"[
                expiration_date.month - 1
            ]  # F=Jan, M=Jun, Z=Dec
            year_code = str(expiration_date.year % 100).zfill(
                2
            )  # Last two digits of year
            config_dict["contract_code"] = month_code + year_code

        # Create child config
        child_config = BaseInstrumentConfig(**config_dict)
        child_config.parent_config = parent_config

        # Cache the derived config
        self._base_derived_cache[cache_key] = child_config

        return child_config

    def create_derived_config(
        self,
        parent_config: InstrumentConfig,
        new_interval: Optional[str] = None,
        expiration_date: Optional[datetime.date] = None,
    ) -> InstrumentConfig:
        """
        Create a derived InstrumentConfig with a new interval and/or expiration date.

        Args:
            parent_config: The parent InstrumentConfig
            new_interval: Optional new interval string (e.g., '5m', '1h')
            expiration_date: Optional expiration date for futures individual contracts

        Returns:
            A new InstrumentConfig with updated fields

        Raises:
            ValueError: If neither new_interval nor expiration_date is provided
            KeyError: If broker configuration is missing
        """
        if new_interval is None and expiration_date is None:
            raise ValueError(
                "At least one of new_interval or expiration_date must be provided."
            )

        # Check if we already have this derived config in the cache
        cache_key = (
            parent_config.broker_name,
            parent_config.symbol,
            new_interval,
            expiration_date,
        )

        if cache_key in self._derived_cache:
            return self._derived_cache[cache_key]

        base_config = self.create_derived_base_config(
            parent_config=parent_config,
            new_interval=new_interval,
            expiration_date=expiration_date,
        )

        config = self._get_config_from_base(
            base_config=base_config, broker_name=parent_config.broker_name
        )

        self._derived_cache[cache_key] = config

        return config

    def get_config_from_base(
        self, base_config: BaseInstrumentConfig, broker_name: str
    ) -> InstrumentConfig:
        """
        Get configuration for a specific instrument and broker.

        Args:
            base_config: The base configuration for the instrument
            broker_name: The name of the broker

        Returns:
            An InstrumentConfig object with combined fields

        Raises:
            KeyError: If no broker configuration is found for the instrument
        """

        if base_config.parent_config is None:
            cache_key = (
                base_config.symbol,
                broker_name,
            )

            if cache_key not in self._config_cache:
                config = self._get_config_from_base(
                    base_config=base_config, broker_name=broker_name
                )
                self._config_cache[cache_key] = config

            return self._config_cache[cache_key]
        else:
            cache_key = (
                base_config.parent_config.symbol,
                base_config.parent_config.interval,
                broker_name,
                base_config.expiration_date,
            )

            if cache_key not in self._derived_cache:
                config = self._get_config_from_base(
                    base_config=base_config, broker_name=broker_name
                )
                self._derived_cache[cache_key] = config

            return self._derived_cache[cache_key]

    def clear_cache(self):
        """Clear all configuration caches."""
        self._base_config_cache.clear()
        self._base_derived_cache.clear()
        self._config_cache.clear()
        self._derived_cache.clear()
