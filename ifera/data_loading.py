"""
Functions for loading and processing financial data.
"""

import zipfile as zip_module  # Renamed to avoid parameter conflict
from typing import Any, Dict, Optional

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .config import BaseInstrumentConfig
from .enums import Source
from .file_utils import make_instrument_path, read_tensor_from_gzip


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
) -> pl.DataFrame:
    """Read a CSV file with progress tracking, handling both regular and zip files."""
    total_lines = count_lines(file_path, zipfile)
    desc = f"Loading data from {file_path}"
    
    # Polars doesn't support chunk reading, so we'll read the whole file at once
    # but still show a progress bar
    with tqdm(total=total_lines, unit="lines", desc=desc) as pbar:
        # Remove chunksize if present since polars doesn't support it
        read_csv_kwargs_copy = {k: v for k, v in read_csv_kwargs.items() if k != 'chunksize'}
        df = pl.read_csv(file_path, **read_csv_kwargs_copy)
        pbar.update(len(df))
    return df


def load_data(
    raw: bool,
    instrument: BaseInstrumentConfig,
    dtype: str = "float32",
    zipfile: bool = True,
) -> pl.DataFrame:
    """Load data from CSV files."""
    source = Source.RAW if raw else Source.PROCESSED
    file_path = make_instrument_path(source=source, instrument=instrument)

    read_csv_kwargs: Dict[str, Any] = {}

    # Map pandas dtype names to polars dtype classes
    pl_dtype_map = {
        "float32": pl.Float32,
        "float64": pl.Float64,
        "int32": pl.Int32,
        "int64": pl.Int64,
    }
    
    pl_dtype = pl_dtype_map.get(dtype, pl.Float32)

    if raw:
        read_csv_kwargs = {
            "has_header": False,
            "new_columns": ["date", "time", "open", "high", "low", "close", "volume"],
            "schema_overrides": {
                "open": pl_dtype,
                "high": pl_dtype,
                "low": pl_dtype,
                "close": pl_dtype,
                "volume": pl.Int32,
            },
        }
    else:
        # pylint: disable=duplicate-code
        # Use a different variable name to avoid redefinition
        read_csv_kwargs = {
            "has_header": False,
            "new_columns": [
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
            "schema_overrides": {
                "open": pl_dtype,
                "high": pl_dtype,
                "low": pl_dtype,
                "close": pl_dtype,
                "volume": pl.Int32,
                "date": pl.Int32,
                "time": pl.Int32,
                "trade_date": pl.Int32,
                "offset_time": pl.Int32,
            },
        }
        # pylint: enable=duplicate-code

    # Polars uses infer_schema_length instead of compression for zip files
    try:
        df = read_csv_with_progress(str(file_path), read_csv_kwargs, zipfile)
    except Exception as e:
        raise ValueError(f"Error reading CSV file at {file_path}: {e}") from e

    if raw:
        try:
            # Convert date and time columns to datetime
            df = df.with_columns([
                (pl.col("date") + " " + pl.col("time")).str.to_datetime().alias("date_time")
            ])
        except Exception as e:
            raise ValueError(
                "Error converting 'date' and 'time' columns to datetime"
            ) from e

        # Drop date and time columns (polars doesn't have index like pandas)
        df = df.drop(["date", "time"])
    return df


def torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    """Convert torch dtype to numpy dtype."""
    return torch.empty((), dtype=dtype).numpy().dtype


def load_data_tensor(
    instrument: BaseInstrumentConfig,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    strip_date_time: bool = True,
    source: Source = Source.TENSOR,
) -> torch.Tensor:
    """Load processed data as a PyTorch tensor using dependency-based file refreshing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the tensor from the local file
    file_path = make_instrument_path(source=source, instrument=instrument)

    try:
        tensor = read_tensor_from_gzip(str(file_path), device=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Tensor file not found at {file_path}: {e}") from e

    tensor = tensor.to(dtype=dtype)

    if strip_date_time:
        tensor = tensor[:, :, 4:].clone()  # Skip first 4 columns

    return tensor
