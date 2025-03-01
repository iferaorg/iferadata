"""
Data models for financial instruments.
"""
import datetime
import json
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

def to_camel(string: str) -> str:
    """Convert snake_case strings to camelCase."""
    parts = string.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

class BaseInstrumentData(BaseModel):
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
    time_step: Optional[pd.Timedelta] = None
    end_time: Optional[pd.Timedelta] = None
    total_steps: Optional[int] = None

    @field_validator(
        "trading_start",
        "trading_end",
        "liquid_start",
        "liquid_end",
        "regular_start",
        "regular_end",
        mode="before"
    )
    @classmethod
    def parse_timedelta(cls, value):
        """Parse timedelta from string value."""
        try:
            return pd.to_timedelta(value)
        except Exception as exc:
            raise ValueError(f"Error parsing timedelta from value {value}: {exc}") from exc

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
    def compute_derived_fields(self) -> "BaseInstrumentData":
        """Compute derived fields after validation."""
        try:
            if self.interval is None:
                raise ValueError("Interval is required.")
            self.time_step = pd.to_timedelta(self.interval)
            if self.trading_start is None or self.trading_end is None:
                raise ValueError("Both trading_start and trading_end are required.")
            self.end_time = self.trading_end - self.trading_start - self.time_step
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
        "alias_generator": to_camel,      # snake_case -> camelCase
        "populate_by_name": True,         # allow field population by pythonic names
    }

class BrokerInstrumentData(BaseModel):
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

class InstrumentData(BaseInstrumentData, BrokerInstrumentData):
    """
    Combined instrument configuration with both base and broker-specific data.
    """
    pass

class BrokerData(BaseModel):
    """
    Pydantic v2 model for broker configuration.
    """
    name: str
    instruments: Dict[str, BrokerInstrumentData]

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }

class InstrumentConfig:
    """Loads and manages instrument configurations from JSON."""
    def __init__(
        self,
        instruments_filename: str = "data/instruments.json",
        brokers_filename: str = "data/brokers.json"
    ):
        """Initialize with configuration filenames."""
        self.instruments_filename = instruments_filename
        self.brokers_filename = brokers_filename
        self.last_instruments_update: Optional[float] = None
        self.last_brokers_update: Optional[float] = None
        self.instruments_data = {}
        self.brokers_data = {}
        self._load_data()

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

    def reload_if_updated(self):
        """Reload configs if files have been modified."""
        try:
            instruments_mtime = Path(self.instruments_filename).stat().st_mtime
            brokers_mtime = Path(self.brokers_filename).stat().st_mtime
        except Exception as e:
            raise OSError(f"Error accessing configuration files: {e}") from e

        if (self.last_instruments_update is None 
            or instruments_mtime > self.last_instruments_update
            or self.last_brokers_update is None
            or brokers_mtime > self.last_brokers_update):
            self._load_data()

    def get_base_instrument_config(self, instrument_key: str) -> BaseInstrumentData:
        """Get base configuration for a specific instrument."""
        self.reload_if_updated()
        try:
            instrument_dict = self.instruments_data[instrument_key]
        except KeyError as e:
            raise KeyError(
                f"Instrument configuration '{instrument_key}' not found in '{self.instruments_filename}'."
            ) from e
        try:
            return BaseInstrumentData(**instrument_dict, last_update=self.last_instruments_update)
        except Exception as e:
            raise ValueError(
                f"Error creating BaseInstrumentData for '{instrument_key}': {e}"
            ) from e

    def get_broker_config(self, broker_name: str) -> BrokerData:
        """Get configuration for a specific broker."""
        self.reload_if_updated()
        try:
            broker_dict = self.brokers_data[broker_name]
        except KeyError as e:
            raise KeyError(
                f"Broker configuration '{broker_name}' not found in '{self.brokers_filename}'."
            ) from e
        try:
            return BrokerData(**broker_dict)
        except Exception as e:
            raise ValueError(
                f"Error creating BrokerData for '{broker_name}': {e}"
            ) from e

    def get_config(self, broker_name: str, instrument_key: str) -> InstrumentData:
        """Get combined configuration for a specific instrument and broker."""
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
            "last_update": max(self.last_instruments_update or float("-inf"),
                             self.last_brokers_update or float("-inf"))
        }
        
        return InstrumentData(**combined_dict)
