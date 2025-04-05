"""
File system utilities for the ifera package.
"""

from pathlib import Path

from .config import BaseInstrumentConfig
from .enums import Source, extension_map
from .settings import settings


def make_path(
    source: Source,
    type: str,
    interval: str,
    symbol: str,
    remove_file: bool = False,
) -> Path:
    """Generate a path to a data file."""
    path = Path(settings.DATA_FOLDER, source.value, type, interval, symbol)
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
    file_name = instrument.symbol
    return make_path(
        source,
        instrument.type,
        instrument.interval,
        file_name,
        remove_file=remove_file,
    )
