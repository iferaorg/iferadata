"""
Financial instrument data handling and processing module.

This module provides classes and utilities for loading, managing, and processing
financial instrument data. It supports operations on time-series financial data,
including handling missing values and calculating technical indicators such as
Average Relative True Range (ARTR).

The module leverages PyTorch tensors and masked operations for efficient
computation on both CPU and GPU devices.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from einops import repeat
from torch.masked import MaskedTensor, masked_tensor

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
    ) -> None:
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
        self._artr: None | MaskedTensor = None
        self._alpha = 1.0 / 14
        self._acrossday = True

    @property
    def data(self) -> torch.Tensor:
        """Read-only property that lazily loads data on first access."""
        if self._data is None:
            self._data = load_data_tensor(
                self.instrument,
                zipfile=self.zipfile,
                dtype=self.dtype,
                device=self.device,
            )
        return self._data

    @property
    def artr_alpha(self) -> float:
        """Alpha parameter for ARTR calculation."""
        return self._alpha

    @artr_alpha.setter
    def artr_alpha(self, value: float) -> None:
        """Set alpha parameter and clear cached ARTR values."""
        self._alpha = value
        self._artr = None

    @property
    def valid_mask(self) -> torch.Tensor:
        """Mask indicating valid data points."""
        vol_mask = self.data[..., -1].to(torch.int) != 0
        return repeat(vol_mask, "... -> ... n", n=self.data.shape[-1])

    @property
    def maked_data(self) -> MaskedTensor:
        """Masked data tensor."""
        return masked_tensor(self.data, self.valid_mask)

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
        if self._artr is None:
            self._artr = masked_artr(
                self.maked_data,
                alpha=self._alpha,
                acrossday=self._acrossday,
                chunk_size=self.chunk_size,
            )

        artr = self._artr.get_data()

        if isinstance(artr, torch.Tensor):
            return artr

        # If the data is not a tensor, raise an error
        raise ValueError("The data is not a tensor")


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
        )
        self._data_cache[cache_key] = data
        return data

    def clear_cache(self):
        """Clear the entire data cache."""
        self._data_cache.clear()

    def remove_from_cache(
        self,
        instrument_config: InstrumentConfig,
        zipfile: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        """
        Remove a specific instance from the cache.

        Args:
            instrument_config: The instrument configuration
            zipfile: Whether the data is stored in a zip file
            dtype: Data type for the tensor
            device: Device to store and process the data on
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

        # Remove from cache if exists
        if cache_key in self._data_cache:
            del self._data_cache[cache_key]
