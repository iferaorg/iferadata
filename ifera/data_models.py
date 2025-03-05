from typing import Optional
import math

import torch
from torch.masked import MaskedTensor, masked_tensor
from einops import repeat

from .config import InstrumentConfig
from .data_loading import load_data_tensor
from .masked_series import masked_artr


class InstrumentData:
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
        self._data = None
        self._chunk_size = None

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
        if self._chunk_size is None:
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
            self._chunk_size = int(2 ** int(math.log2(sqrt_value)))

        return self._chunk_size

    def artr(self, alpha: float = 1.0 / 14.0, acrossday: bool = False) -> torch.Tensor:
        """Average Relative True Range (ATR) data."""
        ma = masked_artr(
            self.maked_data,
            alpha=alpha,
            acrossday=acrossday,
            chunk_size=self.chunk_size,
        )

        artr = ma.get_data()

        if isinstance(artr, torch.Tensor):
            return artr

        # If the data is not a tensor, raise an error
        raise ValueError("The data is not a tensor")
