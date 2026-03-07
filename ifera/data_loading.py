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
    dtype_mapping = {
        "float32": pl.Float32,
        "float64": pl.Float64,
        "int32": pl.Int32,
        "int64": pl.Int64,
        "str": pl.String,
    }
    schema_overrides = {
        col: dtype_mapping.get(str(dtype), dtype)
        for col, dtype in read_csv_kwargs.get("dtype", {}).items()
    }
    has_header = read_csv_kwargs.get("header", "infer") is not None
    new_columns = read_csv_kwargs.get("names")
    desc = f"Loading data from {file_path}"
    with tqdm(total=total_lines, unit="lines", desc=desc) as pbar:
        if zipfile:
            with zip_module.ZipFile(file_path, "r") as z_file:
                filename = z_file.namelist()[0]
                with z_file.open(filename) as csv_file:
                    data = csv_file.read()
            df = pl.read_csv(
                data,
                has_header=has_header,
                new_columns=new_columns,
                schema_overrides=schema_overrides,
            )
        else:
            df = pl.read_csv(
                file_path,
                has_header=has_header,
                new_columns=new_columns,
                schema_overrides=schema_overrides,
            )
        pbar.update(total_lines)
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
            df = df.with_columns(
                pl.concat_str([pl.col("date"), pl.col("time")], separator=" ")
                .str.to_datetime(strict=False)
                .alias("date_time")
            )
        except Exception as e:
            raise ValueError(
                "Error converting 'date' and 'time' columns to datetime"
            ) from e

        df = df.drop(["date", "time"]).select(
            ["date_time", "open", "high", "low", "close", "volume"]
        )
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
