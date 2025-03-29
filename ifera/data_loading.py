"""
Functions for loading and processing financial data.
"""

import zipfile as zip_module  # Renamed to avoid parameter conflict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm

from .config import InstrumentConfig, BaseInstrumentConfig
from .file_utils import make_instrument_path
from .file_manager import refresh_file


def count_lines(file_path: str, is_zip: bool = False) -> int:
    """Count number of lines in a file, handling both regular and zip files."""
    if is_zip:
        with zip_module.ZipFile(file_path, "r") as z:
            # Get the first file in the zip
            filename = z.namelist()[0]
            with z.open(filename) as f:
                return sum(1 for _ in f)
    else:
        with open(file_path, "rb") as f:
            return sum(1 for _ in f)


def read_csv_with_progress(
    file_path: str, read_csv_kwargs: Dict[str, Any], zipfile: bool
) -> pd.DataFrame:
    """Read a CSV file with progress tracking, handling both regular and zip files."""
    total_lines = count_lines(file_path, zipfile)
    chunksize = max(1000, min(100000, total_lines // 100))
    read_csv_kwargs["chunksize"] = chunksize
    chunks = []
    desc = f"Loading data from {file_path}"
    with tqdm(total=total_lines, unit="lines", desc=desc) as pbar:
        for chunk in pd.read_csv(file_path, **read_csv_kwargs):
            chunks.append(chunk)
            pbar.update(len(chunk))
    df = pd.concat(chunks, ignore_index=True)
    return df


def load_data(
    raw: bool,
    instrument: BaseInstrumentConfig,
    dtype: str = "float32",
    reset: bool = False,
    zipfile: bool = True,
) -> pd.DataFrame:
    """Load data from CSV files."""
    source = "raw" if raw else "processed"
    refresh_file(
        scheme="file",
        source=source,
        instrument=instrument,
        zipfile=zipfile,
        reset=reset,
    )
    file_path = make_instrument_path(
        source=source, instrument=instrument, zipfile=zipfile
    )

    read_csv_kwargs: Dict[str, Any] = {}

    if raw:
        read_csv_kwargs = {
            "header": None,
            "parse_dates": False,
            "names": ["date", "time", "open", "high", "low", "close", "volume"],
            "dtype": {
                "open": dtype,
                "high": dtype,
                "low": dtype,
                "close": dtype,
                "volume": "int32",
            },
        }
    else:
        # pylint: disable=duplicate-code
        # Use a different variable name to avoid redefinition
        read_csv_kwargs = {
            "header": None,
            "parse_dates": False,
            "names": [
                "date",
                "time",
                "trade_date",
                "offset_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
            "dtype": {
                "open": dtype,
                "high": dtype,
                "low": dtype,
                "close": dtype,
                "volume": "int32",
                "date": "int32",
                "time": "int32",
                "trade_date": "int32",
                "offset_time": "int32",
            },
        }
        # pylint: enable=duplicate-code

    if zipfile:
        read_csv_kwargs["compression"] = "zip"
    try:
        df = read_csv_with_progress(str(file_path), read_csv_kwargs, zipfile)
    except Exception as e:
        raise ValueError(f"Error reading CSV file at {file_path}: {e}") from e

    if raw:
        try:
            df["date_time"] = pd.to_datetime(df["date"] + " " + df["time"])
        except Exception as e:
            raise ValueError(
                "Error converting 'date' and 'time' columns to datetime"
            ) from e

        df = df.drop(columns=["date", "time"], inplace=False).set_index("date_time")
    return df


def torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    """Convert torch dtype to numpy dtype."""
    return torch.empty((), dtype=dtype).numpy().dtype


def load_data_tensor(
    instrument: InstrumentConfig,
    reset: bool = False,
    zipfile: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load processed data as a PyTorch tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = torch_dtype_to_numpy_dtype(dtype).name
    df: pd.DataFrame = load_data(
        raw=False, instrument=instrument, dtype=dtype_str, reset=reset, zipfile=zipfile
    )
    np_array = df.to_numpy()
    tensor = torch.as_tensor(np_array, dtype=dtype, device=device)
    tensor = rearrange(tensor, "(d t) c -> d t c", t=instrument.total_steps)

    return tensor[
        ..., 4:
    ].clone()  # Skip the first 4 columns (date, time, trade_date, offset_time)
