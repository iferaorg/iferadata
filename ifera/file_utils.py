"""
File system utilities for the ifera package.
"""

from pathlib import Path
from typing import Optional
from .models import InstrumentData
from .settings import settings


def make_path(
    raw: bool,
    instrument: InstrumentData,
    remove_file: bool = False,
    special_interval: Optional[str] = None,
    zipfile: bool = True,
) -> Path:
    """Generate a path to a data file."""
    source = "raw" if raw else "processed"
    interval = special_interval if special_interval is not None else instrument.interval
    file_name = instrument.symbol
    path = Path(settings.DATA_FOLDER, source, instrument.type, interval, file_name)
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
