"""
Financial instrument data handling and processing module.

This module provides classes and utilities for loading, managing, and processing
financial instrument data. It supports operations on time-series financial data,
including handling missing values and calculating technical indicators such as
Average Relative True Range (ARTR).

The module leverages PyTorch tensors and mask operations for efficient
computation on both CPU and GPU devices.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from einops import repeat, rearrange

from .config import InstrumentConfig
from .data_loading import load_data_tensor
from .masked_series import masked_artr


class InstrumentData:
    """
    Class for loading and processing financial instrument data.

    This class handles the loading, storage, and computation on financial instrument
    data. It provides lazy loading of data, masking of invalid values, and methods
    for calculating technical indicators such as Average Relative True Range (ARTR).
    The class automatically optimizes memory usage and computation based on the
    available device (CPU/GPU).

    Attributes:
        instrument (InstrumentConfig): Configuration for the financial instrument
        zipfile (bool): Whether the data is stored in a zip file
        dtype (torch.dtype): Data type for the tensor
        device (torch.device): Device to store and process the data on

    Examples:
        >>> from ifera.config import InstrumentConfig
        >>> config = InstrumentConfig(symbol="AAPL")
        >>> data = InstrumentData(config)
        >>> atr = data.artr(alpha=0.1)
    """

    def __init__(
        self,
        instrument: InstrumentConfig,
        zipfile: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        sentinel: Optional[object] = None,
    ) -> None:
        if sentinel is not DataManager()._sentinel:
            raise PermissionError(
                "InstrumentData must be created via DataManager.get_instrument_data"
            )

        self.instrument = instrument
        self.dtype = dtype
        self.zipfile = zipfile
        self.device = (
            device
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self._data: None | torch.Tensor = None
        self._chunk_size = 0
        # Store ARTR data and mask separately
        self._artr_data: None | torch.Tensor = None
        self._artr_mask: None | torch.Tensor = None
        self._alpha = 1.0 / 14
        self._acrossday = True

    @property
    def data(self) -> torch.Tensor:
        """Read-only property that lazily loads data on first access."""
        if self._data is None:
            if self.instrument.parent_config is not None:
                self._data = self._aggregate_from_parent(self.instrument.parent_config)
            else:
                self._data = load_data_tensor(
                    self.instrument,
                    zipfile=self.zipfile,
                    dtype=self.dtype,
                    device=self.device,
                )
        return self._data

    # TODO: Handle time steps larger than 1 day
    def _aggregate_from_parent(self, parent_config: InstrumentConfig) -> torch.Tensor:
        """Aggregate data from parent data."""
        multiplier = int(
            self.instrument.time_step.total_seconds()
            // parent_config.time_step.total_seconds()
        )
        if multiplier <= 0:
            raise ValueError(
                "The parent time step must be a multiple of the child time step."
            )
        manager = DataManager()
        parent_data = manager.get_instrument_data(
            parent_config,
            self.zipfile,
            self.dtype,
            self.device,
        ).data

        parent_steps = parent_data.shape[1]

        if parent_steps % multiplier != 0:
            padding = (parent_steps // multiplier + 1) * multiplier - parent_steps
            padding_data = torch.zeros(
                (parent_data.shape[0], parent_data.shape[2]),
                dtype=parent_data.dtype,
                device=parent_data.device,
            )
            padding_data[:, 0:4] = parent_data[
                :, -1, 3, None
            ]  # Open, High, Low, Close <- Last Close
            padding_data = repeat(padding_data, "d c -> d g c", g=padding)
            parent_data = torch.cat([parent_data, padding_data], dim=1)

        parent_data = rearrange(parent_data, "d (g n) c -> d g c n", n=multiplier)
        result = torch.zeros(
            (parent_data.shape[0], self.instrument.total_steps, parent_data.shape[2]),
            dtype=parent_data.dtype,
            device=parent_data.device,
        )
        result[:, :, 0] = parent_data[:, :, 0, 0]  # Open
        result[:, :, 1] = parent_data[:, :, 1].max(dim=-1).values  # High
        result[:, :, 2] = parent_data[:, :, 2].min(dim=-1).values  # Low
        result[:, :, 3] = parent_data[:, :, 3, -1]  # Close
        result[:, :, 4] = parent_data[:, :, 4].sum(dim=-1)  # Volume

        return result

    @property
    def artr_alpha(self) -> float:
        """Alpha parameter for ARTR calculation."""
        return self._alpha

    @artr_alpha.setter
    def artr_alpha(self, value: float) -> None:
        """Set alpha parameter and clear cached ARTR values."""
        self._alpha = value
        self._artr_data = None
        self._artr_mask = None

    @property
    def artr_acrossday(self) -> bool:
        """Flag indicating whether to calculate ARTR across days."""
        return self._acrossday

    @artr_acrossday.setter
    def artr_acrossday(self, value: bool) -> None:
        """Set acrossday flag and clear cached ARTR values."""
        self._acrossday = value
        self._artr_data = None
        self._artr_mask = None

    @property
    def valid_mask(self) -> torch.Tensor:
        """Mask indicating valid data points."""
        return self.data[..., -1].to(torch.int) != 0

    @property
    def chunk_size(self) -> int:
        """Calculate the chunk size for the ARTR calculation.

        Returns the largest power of 2 that is less than
        sqrt(device_memory / (10 * dtype_size))
        """
        if self._chunk_size == 0:
            # Get total memory of the device in bytes
            if self.device.type == "cuda":
                device_memory_bytes = torch.cuda.get_device_properties(
                    self.device
                ).total_memory
            else:
                # For CPU, use a reasonable default (8 GB)
                device_memory_bytes = 8 * 1024 * 1024 * 1024

            # Get size of the data type in bytes
            dtype_size_bytes = torch.tensor([], dtype=self.dtype).element_size()

            # Calculate according to formula
            value = device_memory_bytes / (10 * dtype_size_bytes)
            sqrt_value = math.sqrt(value)

            # Find the largest power of 2 less than the calculated value
            self._chunk_size = 2 ** int(math.log2(sqrt_value))

        return self._chunk_size

    @property
    def artr(self) -> torch.Tensor:
        """Average Relative True Range (ATR) data."""
        if self._artr_data is None:
            mask = self.valid_mask
            self._artr_data = masked_artr(
                self.data,
                mask,
                alpha=self._alpha,
                acrossday=self._acrossday,
                chunk_size=self.chunk_size,
            )

        return self._artr_data

    def get_prev_indices(self, date_idx, time_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the indices of the previous time step for the given indices.

        Parameters
        ----------
        date_idx : torch.Tensor
            Batch of date indices
        time_idx : torch.Tensor
            Batch of time indices

        Returns
        -------
        prev_date_idx : torch.Tensor
            Batch of previous date indices
        prev_time_idx : torch.Tensor
            Batch of previous time indices
        """
        prev_date_idx = date_idx.clone()
        prev_time_idx = time_idx - 1
        prev_date_idx[prev_time_idx < 0] -= 1
        prev_time_idx[prev_time_idx < 0] += self.instrument.total_steps
        return prev_date_idx, prev_time_idx

    def get_next_indices(self, date_idx, time_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the indices of the next time step for the given indices.

        Parameters
        ----------
        date_idx : torch.Tensor
            Batch of date indices
        time_idx : torch.Tensor
            Batch of time indices

        Returns
        -------
        next_date_idx : torch.Tensor
            Batch of next date indices
        next_time_idx : torch.Tensor
            Batch of next time indices
        """
        next_date_idx = date_idx.clone()
        next_time_idx = time_idx + 1
        next_date_idx[next_time_idx >= self.instrument.total_steps] += 1
        next_time_idx[next_time_idx >= self.instrument.total_steps] = 0
        return next_date_idx, next_time_idx

    def convert_indices(
        self, source: "InstrumentData", date_idx, time_idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert indices from another InstrumentData instance to the current instance.

        Parameters
        ----------
        source : InstrumentData
            Source InstrumentData instance
        date_idx : torch.Tensor
            Batch of date indices
        time_idx : torch.Tensor
            Batch of time indices

        Returns
        -------
        converted_date_idx : torch.Tensor
            Batch of converted date indices
        converted_time_idx : torch.Tensor
            Batch of converted time indices
        """
        if source.instrument.symbol != self.instrument.symbol:
            raise ValueError("Cannot convert indices between different symbols.")
        if source.instrument.trading_start != self.instrument.trading_start:
            raise ValueError(
                "Cannot convert indices between different trading start times."
            )

        time_ratio = self.instrument.time_step / source.instrument.time_step

        # Make sure the time ratio is an integer
        if not time_ratio.is_integer():
            raise ValueError("Time step ratio must be an integer.")

        tr = int(time_ratio)

        # Get the next indices from the source, then convert them to the current instance
        # Then get the previous indices from the current instance
        # This is to ensure that the last full bar is used for the conversion
        next_date_idx, next_time_idx = source.get_next_indices(date_idx, time_idx)

        return self.get_prev_indices(next_date_idx, next_time_idx // tr)


class DataManager:
    """
    Manages and caches instrument data instances.
    Implemented as a singleton to ensure only one instance exists.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the data manager (once)."""
        if getattr(self, "_initialized", False):
            return

        # Cache for InstrumentData instances
        # Key: (broker_name, instrument_symbol, interval, zipfile, dtype, device_type)
        self._data_cache: Dict[
            Tuple[str, str, str, bool, torch.dtype, str], InstrumentData
        ] = {}
        self._sentinel = object()
        self._initialized = True

    def get_instrument_data(
        self,
        instrument_config: InstrumentConfig,
        zipfile: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> InstrumentData:
        """
        Get an InstrumentData instance for the specified configuration.
        Uses cached instance if available, otherwise creates a new one.

        Args:
            instrument_config: The instrument configuration
            zipfile: Whether the data is stored in a zip file
            dtype: Data type for the tensor
            device: Device to store and process the data on

        Returns:
            An InstrumentData instance
        """
        # Normalize device
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # Create cache key
        cache_key = (
            instrument_config.broker_name,
            instrument_config.symbol,
            instrument_config.interval,
            zipfile,
            dtype,
            device.type,
        )

        # Check if we have a cached instance
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # Create a new instance and cache it
        data = InstrumentData(
            instrument=instrument_config,
            zipfile=zipfile,
            dtype=dtype,
            device=device,
            sentinel=self._sentinel,
        )
        self._data_cache[cache_key] = data
        return data

    def clear_cache(self):
        """Clear the entire data cache."""
        self._data_cache.clear()
