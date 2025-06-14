"""
File system utilities for the ifera package.
"""

from pathlib import Path
import torch
import gzip

from .config import BaseInstrumentConfig
from .enums import Source, extension_map
from .settings import settings


def make_path(
    source: Source,
    file_type: str,
    interval: str,
    symbol: str,
    remove_file: bool = False,
) -> Path:
    """Generate a path to a data file."""
    path = Path(settings.DATA_FOLDER, source.value, file_type, interval, symbol)
    path = path.with_suffix(extension_map[source])

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Error creating directories for path {path.parent}: {e}") from e

    if remove_file:
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            raise OSError(f"Error removing file {path}: {e}") from e

    return path


def make_instrument_path(
    source: Source,
    instrument: BaseInstrumentConfig,
    remove_file: bool = False,
) -> Path:
    """Generate a path to a data file."""
    file_name = instrument.file_symbol
    return make_path(
        source,
        instrument.type,
        instrument.interval,
        file_name,
        remove_file=remove_file,
    )


def write_tensor_to_gzip(file_name: str, tensor: torch.Tensor) -> None:
    """Save a tensor to a gzip-compressed file."""
    with gzip.open(file_name, "wb") as f:
        torch.save(tensor, f)  # type: ignore


def read_tensor_from_gzip(
    file_name: str, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Load a tensor from a gzip-compressed file."""
    with gzip.open(file_name, "rb") as f:
        return torch.load(f, map_location=device, weights_only=True)  # type: ignore
