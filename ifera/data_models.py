"""
Financial instrument data handling and processing module.

This module provides classes and utilities for loading, managing, and processing
financial instrument data. It supports operations on time-series financial data,
including handling missing values and calculating technical indicators such as
Average Relative True Range (ARTR).

The module leverages PyTorch tensors and mask operations for efficient
computation on both CPU and GPU devices.
"""

import datetime as dt
import math
from typing import Optional, Tuple

import torch
import yaml

from .config import BaseInstrumentConfig
from .data_loading import load_data_tensor
from .decorators import ThreadSafeCache, singleton
from .enums import Scheme, Source
from .file_manager import FileManager, refresh_instrument_file
from .file_utils import make_path
from .masked_series import masked_artr
from .url_utils import make_url


def calculate_chunk_size(device: torch.device, dtype: torch.dtype) -> int:
    """Calculate the chunk size for the ARTR calculation.

    Returns the largest power of 2 that is less than
    sqrt(device_memory / (10 * dtype_size))
    """
    # if self._chunk_size == 0:
    # Get total memory of the device in bytes
    if device.type == "cuda":
        device_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    else:
        # For CPU, use a reasonable default (8 GB)
        device_memory_bytes = 8 * 1024 * 1024 * 1024

    # Get size of the data type in bytes
    dtype_size_bytes = torch.tensor([], dtype=dtype).element_size()

    # Calculate according to formula
    value = device_memory_bytes / (10 * dtype_size_bytes)
    sqrt_value = math.sqrt(value)

    # Find the largest power of 2 less than the calculated value
    chunk_size = 2 ** int(math.log2(sqrt_value))

    return chunk_size


class InstrumentData:
    """
    Class for loading and processing financial instrument data.

    This class handles the loading, storage, and computation on financial instrument
    data. It provides lazy loading of data, masking of invalid values, and methods
    for calculating technical indicators such as Average Relative True Range (ARTR).
    The class automatically optimizes memory usage and computation based on the
    available device (CPU/GPU).

    Attributes:
        instrument (BaseInstrumentConfig): Configuration for the financial instrument
        zipfile (bool): Whether the data is stored in a zip file
        dtype (torch.dtype): Data type for the tensor
        device (torch.device): Device to store and process the data on

    Examples:
        >>> from ifera.config import BaseInstrumentConfig
        >>> config = BaseInstrumentConfig(symbol="AAPL")
        >>> data = InstrumentData(config)
        >>> atr = data.artr(alpha=0.1)
    """

    def __init__(
        self,
        instrument: BaseInstrumentConfig,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        sentinel: Optional[object] = None,
        backadjust: bool = False,
    ) -> None:
        if sentinel is not DataManager()._sentinel:
            raise PermissionError(
                "InstrumentData must be created via DataManager.get_instrument_data"
            )

        self.instrument = instrument
        self.dtype = dtype
        self.device = (
            device
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.backadjust = backadjust
        self.source = Source.TENSOR_BACKADJUSTED if self.backadjust else Source.TENSOR

        self._load_data()
        self._load_multiplier()

        # Store ARTR data
        self._artr_data: torch.Tensor = torch.tensor(
            [], dtype=self.dtype, device=self.device
        )
        self._alpha = 1.0 / 14
        self._acrossday = True

    def _load_data(self) -> None:
        """Load data tensor from the file."""
        refresh_instrument_file(
            instrument=self.instrument,
            scheme=Scheme.FILE,
            source=self.source,
            reset=False,
        )
        self._data = load_data_tensor(
            self.instrument,
            dtype=self.dtype,
            device=self.device,
            strip_date_time=False,
            source=self.source,
        )

    def _load_multiplier(self) -> None:
        """Load or initialize the contract multiplier tensor."""
        if not self.backadjust:
            self._multiplier = torch.tensor(
                [[1.0]], dtype=self.dtype, device=self.device
            ).expand(self._data.shape[0], self._data.shape[1])
            return

        url = make_url(
            Scheme.FILE,
            Source.META,
            self.instrument.type,
            "rollover",
            self.instrument.symbol,
        )
        FileManager().refresh_file(url)

        path = make_path(
            Source.META,
            self.instrument.type,
            "rollover",
            self.instrument.symbol,
        )
        with path.open("r", encoding="utf-8") as fh:
            rollover_data = yaml.safe_load(fh) or []

        trade_dates = self._data[:, :, 2].to(torch.long)
        offset_times = self._data[:, :, 3].to(torch.long)

        if not rollover_data:
            self._multiplier = torch.ones(
                trade_dates.shape, dtype=self.dtype, device=self.device
            )
            return

        entries = sorted(
            (
                (
                    dt.date.fromisoformat(e["start_date"]).toordinal(),
                    float(e["multiplier"]),
                )
                for e in rollover_data
            ),
            key=lambda x: x[0],
        )

        base_mult = entries[0][1]
        mult_tensor = torch.full(
            trade_dates.shape, base_mult, dtype=self.dtype, device=self.device
        )
        rollover_offset = int(self.instrument.rollover_offset)

        for start_ord, mult in entries[1:]:
            mask = (trade_dates > start_ord) | (
                (trade_dates == start_ord) & (offset_times >= rollover_offset)
            )
            mult_tensor = torch.where(
                mask,
                torch.tensor(mult, dtype=self.dtype, device=self.device),
                mult_tensor,
            )

        self._multiplier = mult_tensor

    @property
    def data(self) -> torch.Tensor:
        """Returns the data tensor excluding date and time columns."""
        return self._data[:, :, 4:]

    @property
    def data_full(self) -> torch.Tensor:
        """Full data tensor including date and time columns."""
        return self._data

    @property
    def multiplier(self) -> torch.Tensor:
        """Contract multiplier for each bar."""
        return self._multiplier

    @property
    def artr_alpha(self) -> float:
        """Alpha parameter for ARTR calculation."""
        return self._alpha

    @property
    def artr_acrossday(self) -> bool:
        """Flag indicating whether to calculate ARTR across days."""
        return self._acrossday

    @property
    def valid_mask(self) -> torch.Tensor:
        """Mask indicating valid data points."""
        return self.data[..., -1].to(torch.int) != 0

    @property
    def artr(self) -> torch.Tensor:
        """Average Relative True Range (ARTR) data."""
        return self._artr_data

    def calculate_artr(self, alpha: float, acrossday: bool) -> torch.Tensor:
        """Calculate and store the ARTR using the provided hyperparameters."""
        self._alpha = alpha
        self._acrossday = acrossday
        mask = self.valid_mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artr_data = masked_artr(
            self.data.to(device),
            mask.to(device),
            alpha=self._alpha,
            acrossday=self._acrossday,
            chunk_size=calculate_chunk_size(device=device, dtype=self.dtype),
        )
        self._artr_data = artr_data.to(self.device)
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
        prev_time_idx = time_idx - 1
        mask = prev_time_idx < 0
        prev_date_idx = date_idx - mask.long()
        prev_time_idx = torch.where(
            mask, self.instrument.total_steps - 1, prev_time_idx
        )
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
        next_time_idx = time_idx + 1
        mask = next_time_idx >= self.instrument.total_steps
        next_date_idx = date_idx + mask.long()
        next_time_idx = torch.where(mask, 0, next_time_idx)
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


@singleton
class DataManager:
    """
    Manages and caches instrument data instances.
    Implemented as a singleton to ensure only one instance exists.
    """

    def __init__(self):
        self._sentinel = object()

    @ThreadSafeCache
    def get_instrument_data(
        self,
        instrument_config: BaseInstrumentConfig,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        backadjust: bool = False,
    ) -> InstrumentData:
        """
        Get an InstrumentData instance for the specified configuration.
        Uses cached instance if available, otherwise creates a new one.

        Args:
            instrument_config: The instrument configuration
            zipfile: Whether the data is stored in a zip file
            dtype: Data type for the tensor
            device: Device to store and process the data on
            backadjust: Whether to use backadjusted data

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

        # Create a new instance and cache it
        data = InstrumentData(
            instrument=instrument_config,
            dtype=dtype,
            device=device,
            sentinel=self._sentinel,
            backadjust=backadjust,
        )

        return data

    def clear_cache(self):
        """Clear the entire data cache."""
        self.get_instrument_data.cache_clear()  # type: ignore
