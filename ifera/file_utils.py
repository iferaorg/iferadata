"""
File system utilities for the ifera package.
"""

from pathlib import Path
from typing import Optional

from .config import BaseInstrumentConfig
from .settings import settings


def make_path(
    source: str,
    type: str,
    interval: str,
    symbol: str,
    zipfile: bool,
    remove_file: bool = False,
) -> Path:
    """Generate a path to a data file."""
    path = Path(settings.DATA_FOLDER, source, type, interval, symbol)
    path = path.with_suffix(".zip" if zipfile else ".csv")

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
    source: str,
    instrument: BaseInstrumentConfig,
    remove_file: bool = False,
    zipfile: bool = True,
) -> Path:
    """Generate a path to a data file."""
    file_name = instrument.symbol
    return make_path(
        source,
        instrument.type,
        instrument.interval,
        file_name,
        zipfile=zipfile,
        remove_file=remove_file,
    )
