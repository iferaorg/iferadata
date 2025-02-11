"""
Data models for financial instruments.
"""
import datetime
import json
from pathlib import Path
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

def to_camel(string: str) -> str:
    """Convert snake_case strings to camelCase."""
    parts = string.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

class InstrumentData(BaseModel):
    """
    Pydantic v2 model for an instrument's configuration.
    """
    # JSON -> Model Fields
    symbol: str
    currency: str
    type: str
    broker: str
    interval: str
    trading_start: pd.Timedelta = Field(..., alias="tradingStart")
    trading_end: pd.Timedelta = Field(..., alias="tradingEnd")
    liquid_start: pd.Timedelta = Field(..., alias="liquidStart")
    liquid_end: pd.Timedelta = Field(..., alias="liquidEnd")
    regular_start: pd.Timedelta = Field(..., alias="regularStart")
    regular_end: pd.Timedelta = Field(..., alias="regularEnd")
    contract_multiplier: int = Field(..., alias="contractMultiplier")
    tick_size: float = Field(..., alias="tickSize")
    margin: float
    commission: float
    min_commission: float = Field(..., alias="minCommission")
    max_commission_pct: float = Field(..., alias="maxCommissionPct")
    slippage: float
    min_slippage: float = Field(..., alias="minSlippage")
    reference_price: float = Field(..., alias="referencePrice")
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
    def compute_derived_fields(self) -> "InstrumentData":
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

class InstrumentConfig:
    """Loads and manages instrument configurations from JSON."""

    def __init__(self, filename: str = "data/instruments.json"):
        """Initialize with configuration filename."""
        self.filename = filename
        self.last_update: Optional[float] = None
        self.data = {}
        self._load_data()

    def _load_data(self):
        """Load data from the JSON file."""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Configuration file '{self.filename}' not found."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from '{self.filename}': {e}"
            ) from e
        try:
            path = Path(self.filename)
            self.last_update = path.stat().st_mtime
        except Exception as e:
            raise OSError(f"Error accessing file '{self.filename}': {e}") from e

    def reload_if_updated(self):
        """Reload config if file has been modified."""
        try:
            current_mtime = Path(self.filename).stat().st_mtime
        except Exception as e:
            raise OSError(f"Error accessing file '{self.filename}': {e}") from e
        if self.last_update is None or current_mtime > self.last_update:
            self._load_data()

    def get_config(self, instrument_key: str) -> InstrumentData:
        """Get configuration for a specific instrument."""
        self.reload_if_updated()
        try:
            instrument_dict = self.data[instrument_key]
        except KeyError as e:
            raise KeyError(
                f"Instrument configuration '{instrument_key}' not found in '{self.filename}'."
            ) from e
        try:
            return InstrumentData(**instrument_dict, last_update=self.last_update)
        except Exception as e:
            raise ValueError(
                f"Error creating InstrumentData for '{instrument_key}': {e}"
            ) from e
