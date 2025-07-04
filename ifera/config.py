"""
Data models for financial instruments.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datetime
import yaml

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
    start_date: datetime.date = Field(..., alias="startDate")
    days_of_week: List[int] = Field(..., alias="daysOfWeek", validate_default=True)
    rollover_time: Optional[pd.Timedelta] = Field(
        None, alias="rolloverTime", validate_default=True
    )
    rollover_vol_alpha: Optional[float] = Field(
        None, alias="rolloverVolAlpha", validate_default=True
    )
    rollover_max_days: Optional[int] = Field(
        None, alias="rolloverMaxDays", validate_default=True
    )
    traded_months: Optional[str] = Field(
        None, alias="tradedMonths", validate_default=True
    )
    last_trading_day_rule: Optional[str] = Field(
        None, alias="lastTradeDayRule", validate_default=True
    )
    first_notice_day_rule: Optional[str] = Field(
        None, alias="firstNoticeDayRule", validate_default=True
    )
    asset_class: Optional[str] = Field(None, alias="assetClass", validate_default=True)
    last_update: Optional[float] = Field(default=None)

    # Derived Fields
    time_step: pd.Timedelta = pd.Timedelta(0)
    end_time: pd.Timedelta = pd.Timedelta(0)
    rollover_offset: int = 0
    total_steps: int = 0

    expiration_date: Optional[datetime.date] = Field(None, alias="expirationDate")
    first_notice_date: Optional[datetime.date] = Field(None, alias="firstNoticeDate")
    rollover_date: Optional[datetime.date] = Field(None, alias="rolloverDate")
    contract_code: Optional[str] = Field(None, alias="contractCode")

    parent_config: Optional["BaseInstrumentConfig"] = Field(None, exclude=True)

    def __hash__(self):
        return id(self)

    @property
    def file_symbol(self) -> str:
        """Return the file name component based on presence of contract_code."""
        if self.contract_code:
            return f"{self.symbol}-{self.contract_code}"
        return self.symbol

    @field_validator(
        "trading_start",
        "trading_end",
        "liquid_start",
        "liquid_end",
        "regular_start",
        "regular_end",
        "rollover_time",
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

    @field_validator("start_date", mode="before")
    @classmethod
    def parse_start_date(cls, value):
        """Parse start_date from string value."""
        try:
            return pd.to_datetime(value)
        except Exception as exc:
            raise ValueError(f"Error parsing start_date: {exc}") from exc

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

    @field_validator("days_of_week", mode="before")
    @classmethod
    def parse_days_of_week(cls, value):
        """Parse days_of_week from string values."""
        if isinstance(value, list):
            return [int(day) for day in value]
        else:
            raise ValueError("days_of_week must be a list of integers.")

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

            if self.rollover_time is not None:
                if self.rollover_time < self.trading_start:
                    raise ValueError(
                        "Rollover time must be greater than trading start time."
                    )
                if self.rollover_time > self.trading_end:
                    raise ValueError(
                        "Rollover time must be less than trading end time."
                    )
                self.rollover_offset = int(
                    (self.rollover_time - self.trading_start).total_seconds()
                )

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
    Loads and manages instrument configurations from YAML.
    Implemented as a singleton to ensure only one instance exists.
    """

    def __init__(
        self,
        instruments_filename: str = "data/instruments.yml",
        brokers_filename: str = "data/brokers.yml",
    ):
        self.instruments_filename = instruments_filename
        self.brokers_filename = brokers_filename
        self.last_instruments_update: Optional[float] = None
        self.last_brokers_update: Optional[float] = None
        self.instruments_data: Dict[str, Dict[str, Any]] = {}
        self.brokers_data: Dict[str, Dict[str, Any]] = {}
        self._base_config_cache: Dict[
            Tuple[str, str, Optional[str]], BaseInstrumentConfig
        ] = {}
        self._config_cache: Dict[
            Tuple[str, str, str, Optional[str]], InstrumentConfig
        ] = {}
        self._base_derived_cache: Dict[
            Tuple[str, Optional[str], Optional[str]], BaseInstrumentConfig
        ] = {}
        self._load_data()

    def _load_data(self):
        """Load data from the YAML files."""

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

    def reload_if_updated(self):
        """Reload configs if files have been modified."""
        instruments_mtime = Path(self.instruments_filename).stat().st_mtime
        brokers_mtime = Path(self.brokers_filename).stat().st_mtime

        if instruments_mtime > (self.last_instruments_update or 0) or brokers_mtime > (
            self.last_brokers_update or 0
        ):
            self._load_data()

    def _find_parent_interval(
        self, allowed_intervals: List[str], interval: str
    ) -> Optional[str]:
        """Return the best parent interval from allowed_intervals for the requested one."""
        requested_td = pd.to_timedelta(interval)
        candidates: List[Tuple[pd.Timedelta, str]] = []
        for allowed in allowed_intervals:
            allowed_td = pd.to_timedelta(allowed)
            if (
                allowed_td < requested_td
                and requested_td.total_seconds() % allowed_td.total_seconds() == 0
            ):
                candidates.append((allowed_td, allowed))

        if not candidates:
            return None

        return max(candidates, key=lambda x: x[0])[1]

    def get_base_instrument_config(
        self, symbol: str, interval: str, contract_code: Optional[str] = None
    ) -> BaseInstrumentConfig:
        """Get base configuration for a specific instrument, interval and optional contract code."""
        self.reload_if_updated()

        cache_key = (symbol, interval, contract_code)
        if cache_key in self._base_config_cache:
            return self._base_config_cache[cache_key]

        try:
            instrument_dict = self.instruments_data["instruments"][symbol]
        except KeyError as e:
            raise KeyError(f"Instrument symbol '{symbol}' not found.") from e

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

        # Check if the interval is allowed or can be derived from an allowed interval
        allowed_intervals = combined_dict.get("intervals", [])
        if interval not in allowed_intervals:
            parent_interval = self._find_parent_interval(allowed_intervals, interval)
            if parent_interval is None:
                raise ValueError(
                    f"Interval '{interval}' not allowed for instrument '{symbol}'. "
                    f"Allowed intervals: {allowed_intervals}"
                )
            parent_config = self.get_base_instrument_config(symbol, parent_interval)
            config = self.create_derived_base_config(
                parent_config=parent_config,
                new_interval=interval,
                contract_code=contract_code,
            )
            self._base_config_cache[cache_key] = config
            return config

        # Remove 'intervals' as it's not part of BaseInstrumentConfig
        combined_dict.pop("intervals", None)

        # Set the interval
        combined_dict["interval"] = interval

        base_config = BaseInstrumentConfig(
            **combined_dict, last_update=self.last_instruments_update
        )
        # Cache the base config without contract code
        self._base_config_cache[(symbol, interval, None)] = base_config

        if contract_code is not None:
            config = self.create_derived_base_config(
                parent_config=base_config, contract_code=contract_code
            )
        else:
            config = base_config

        self._base_config_cache[cache_key] = config
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

        # Create the config
        config = InstrumentConfig(**combined_dict)

        if base_config.contract_code is not None:
            config.broker_symbol = (
                f"{broker_instrument_config.broker_symbol}{base_config.contract_code}"
            )

        return config

    def get_config(
        self,
        broker_name: str,
        symbol: str,
        interval: str,
        contract_code: Optional[str] = None,
    ) -> InstrumentConfig:
        """Get combined configuration for a specific instrument, interval, broker and optional contract code."""
        self.reload_if_updated()

        base_config = self.get_base_instrument_config(
            symbol=symbol,
            interval=interval,
            contract_code=contract_code,
        )

        return self.get_config_from_base(
            base_config=base_config, broker_name=broker_name
        )

    def create_derived_base_config(
        self,
        parent_config: BaseInstrumentConfig,
        new_interval: Optional[str] = None,
        contract_code: Optional[str] = None,
    ) -> BaseInstrumentConfig:
        """
        Create a derived BaseInstrumentConfig with a new interval and/or contract code.

        Args:
            parent_config: The parent BaseInstrumentConfig
            new_interval: Optional new interval string (e.g., '5m', '1h')
            contract_code: Optional contract code for futures individual contracts (e.g., 'M24')

        Returns:
            A new BaseInstrumentConfig with updated fields

        Raises:
            ValueError: If neither new_interval nor contract_code is provided,
                        or if contract_code is used with non-futures instruments
        """
        if new_interval is None and contract_code is None:
            raise ValueError(
                "At least one of new_interval or contract_code must be provided."
            )

        # Check if we already have this derived config in the cache
        cache_interval = new_interval if new_interval else parent_config.interval
        cache_contract_code = (
            contract_code if contract_code else parent_config.contract_code
        )
        cache_key = (parent_config.symbol, cache_interval, cache_contract_code)
        if cache_key in self._base_derived_cache:
            return self._base_derived_cache[cache_key]

        config_dict = parent_config.model_dump()

        if new_interval is not None:
            config_dict["interval"] = new_interval
            time_step = pd.to_timedelta(new_interval)
            parent_step_seconds = parent_config.time_step.total_seconds()
            new_step_seconds = time_step.total_seconds()

            # Validate interval
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

        if contract_code is not None:
            if parent_config.type != "futures":
                raise ValueError(
                    "contract_code can only be set for futures instruments."
                )
            config_dict["contract_code"] = contract_code
            config_dict["type"] = "futures_individual"
            # Do not set expiration_date; it will be calculated elsewhere

        # Create child config
        child_config = BaseInstrumentConfig(**config_dict)
        child_config.parent_config = parent_config

        # Cache the derived config
        self._base_derived_cache[cache_key] = child_config

        return child_config

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
        cache_key = (
            broker_name,
            base_config.symbol,
            base_config.interval,
            base_config.contract_code,
        )
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        config = self._get_config_from_base(
            base_config=base_config, broker_name=broker_name
        )

        self._config_cache[cache_key] = config
        return config

    def clear_cache(self):
        """Clear all configuration caches."""
        self._base_config_cache.clear()
        self._base_derived_cache.clear()
        self._config_cache.clear()
