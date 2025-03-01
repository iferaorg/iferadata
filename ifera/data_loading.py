"""
Functions for loading and processing financial data.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import zipfile as zip_module  # Renamed to avoid parameter conflict
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from .models import InstrumentData
from .settings import settings
from .s3_utils import (
    download_s3_file,
    upload_s3_file,
    check_s3_file_exists,
    get_s3_last_modified,
)
from .data_processing import process_data
from .file_utils import make_path


def make_s3_key(instrument: InstrumentData, zipfile: bool) -> str:
    """Build an S3 key for the instrument data file."""
    extension = ".zip" if zipfile else ".csv"
    return f"{instrument.type}/{instrument.interval}/{instrument.symbol}{extension}"


def try_download_processed_file(
    instrument: InstrumentData,
    target_path: str,
    threshold: float,
    reset: bool,
    zipfile: bool,
) -> bool:
    """Attempt to download an up-to-date processed file from S3."""
    processed_s3_key = make_s3_key(instrument, zipfile)
    s3_bucket = settings.S3_BUCKET_PROCESSED
    if not check_s3_file_exists(s3_bucket, processed_s3_key):
        print(f"S3 processed file s3://{s3_bucket}/{processed_s3_key} does not exist.")
        return False
    try:
        s3_processed_timestamp = get_s3_last_modified(s3_bucket, processed_s3_key)
    except Exception as e:
        msg = (
            f"Error retrieving processed S3 metadata for instrument {instrument.symbol}"
        )
        raise RuntimeError(msg) from e
    if s3_processed_timestamp < threshold or reset:
        print(f"S3 processed file s3://{s3_bucket}/{processed_s3_key} is stale.")
        return False
    print(
        f"S3 processed file s3://{s3_bucket}/{processed_s3_key} is up-to-date. "
        "Downloading..."
    )
    try:
        download_s3_file(s3_bucket, processed_s3_key, target_path)
    except Exception as e:
        msg = f"Error downloading processed file from S3 for instrument {instrument.symbol}"
        raise RuntimeError(msg) from e
    return True


def ensure_raw_data(instrument: InstrumentData, zipfile: bool, reset: bool) -> Path:
    """Ensure raw data file is available locally and up-to-date."""
    try:
        raw_path = make_path(raw=True, instrument=instrument, zipfile=zipfile)
    except Exception as e:
        msg = f"Error generating raw file path for instrument {instrument.symbol}"
        raise RuntimeError(msg) from e
    # Check staleness if file exists and reset not requested
    if raw_path.exists() and not reset:
        try:
            raw_s3_key = make_s3_key(instrument, zipfile)
            s3_timestamp = get_s3_last_modified(settings.S3_BUCKET, raw_s3_key)
        except Exception as e:
            msg = f"Error retrieving raw S3 metadata for instrument {instrument.symbol}"
            raise RuntimeError(msg) from e
        local_mtime = raw_path.stat().st_mtime
        if local_mtime < s3_timestamp:
            print(f"Local raw file {raw_path} is stale. Re-downloading raw data...")
            reset = True
        else:
            print(f"Local raw file {raw_path} is up-to-date.")
    if not raw_path.exists() or reset:
        print(
            f"Raw file {raw_path} missing or reset requested. Downloading raw data..."
        )
        try:
            raw_s3_key = make_s3_key(instrument, zipfile)
            download_s3_file(
                settings.S3_BUCKET, raw_s3_key, str(raw_path)
            )  # Convert Path to str
        except Exception as e:
            msg = f"Error downloading raw data for instrument {instrument.symbol}"
            raise RuntimeError(msg) from e
    else:
        print(f"Raw file {raw_path} is available.")
    return raw_path


def ensure_processed_data(
    instrument: InstrumentData, zipfile: bool, reset: bool
) -> Path:
    """Ensure processed data file is available locally and up-to-date."""
    try:
        local_path = make_path(raw=False, instrument=instrument, zipfile=zipfile)
    except Exception as e:
        msg = f"Error generating processed file path for instrument {instrument.symbol}"
        raise RuntimeError(msg) from e
    try:
        raw_s3_key = make_s3_key(instrument, zipfile)
        raw_s3_timestamp = get_s3_last_modified(settings.S3_BUCKET, raw_s3_key)
    except Exception as e:
        msg = f"Error retrieving raw S3 metadata for instrument {instrument.symbol}"
        raise RuntimeError(msg) from e
    # Ensure we have valid timestamps to compare
    threshold = raw_s3_timestamp
    if instrument.last_update is not None:
        threshold = max(raw_s3_timestamp, instrument.last_update)
    if local_path.exists():
        local_mtime = local_path.stat().st_mtime
        if local_mtime >= threshold and not reset:
            print(f"Local processed file {local_path} is up-to-date.")
            return local_path
        print(f"Local processed file {local_path} is stale.")
    else:
        print(f"Local processed file {local_path} does not exist.")
    if try_download_processed_file(
        instrument, str(local_path), threshold, reset, zipfile
    ):
        return local_path
    print(f"Reprocessing raw data for instrument {instrument.symbol}.")
    # Check if raw file exists before processing
    raw_path = make_path(raw=True, instrument=instrument, zipfile=zipfile)
    raw_existed = raw_path.exists()
    try:
        raw_df = load_data(
            raw=True, instrument=instrument, dtype="float64", zipfile=zipfile
        )
        process_data(raw_df, instrument, zipfile)
        # Clean up raw file if it didn't exist before
        if not raw_existed and raw_path.exists():
            print(f"Cleaning up temporary raw file {raw_path}")
            raw_path.unlink()
    except Exception as e:
        # Clean up raw file even on error if it didn't exist before
        if not raw_existed and raw_path.exists():
            print(f"Cleaning up temporary raw file {raw_path} after error")
            raw_path.unlink()
        msg = f"Error processing data for instrument {instrument.symbol}"
        raise RuntimeError(msg) from e
    processed_s3_key = make_s3_key(instrument, zipfile)
    s3_bucket = settings.S3_BUCKET_PROCESSED
    print(f"Uploading processed file to S3: s3://{s3_bucket}/{processed_s3_key}")
    try:
        upload_s3_file(s3_bucket, processed_s3_key, str(local_path))
    except Exception as e:
        msg = f"Error uploading processed file to S3 for instrument {instrument.symbol}"
        raise RuntimeError(msg) from e
    return local_path


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
    instrument: InstrumentData,
    dtype: str = "float32",
    reset: bool = False,
    zipfile: bool = True,
) -> pd.DataFrame:
    """Load data from CSV files."""
    try:
        if raw:
            file_path = ensure_raw_data(instrument, zipfile, reset)
        else:
            file_path = ensure_processed_data(instrument, zipfile, reset)
    except Exception as e:
        msg = f"Error ensuring data availability for instrument {instrument.symbol}"
        raise RuntimeError(msg) from e

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
    instrument: InstrumentData,
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
    return tensor
