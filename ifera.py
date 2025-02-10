import json
import os

from pathlib import Path
from typing import Optional, List
import datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
import boto3
import torch

from settings import settings

SECONDS_IN_DAY = 86400

def to_camel(string: str) -> str:
    """
    Convert snake_case strings to camelCase.
    Used as an alias generator for Pydantic.
    """
    parts = string.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])


class InstrumentData(BaseModel):
    """
    Pydantic v2 model for an instrument's configuration,
    including aliases from camelCase -> snake_case, time-based fields,
    and derived fields computed after validation.
    """

    # -----------------------
    # JSON -> Model Fields
    # -----------------------
    symbol: str
    currency: str
    type: str
    broker: str
    interval: str

    trading_start: pd.Timedelta = Field(..., alias="tradingStart")
    trading_end: pd.Timedelta = Field(..., alias="tradingEnd")
    liquid_start: pd.Timedelta = Field(..., alias="liquidStart")
    liquid_end: pd.Timedelta = Field(..., alias="liquidEnd")
    regular_start: pd.Timedelta = Field(..., alias="regularStart")
    regular_end: pd.Timedelta = Field(..., alias="regularEnd")

    contract_multiplier: int = Field(..., alias="contractMultiplier")
    tick_size: float = Field(..., alias="tickSize")

    margin: float
    commission: float
    min_commission: float = Field(..., alias="minCommission")
    max_commission_pct: float = Field(..., alias="maxCommissionPct")
    slippage: float
    min_slippage: float = Field(..., alias="minSlippage")
    reference_price: float = Field(..., alias="referencePrice")

    remove_dates: Optional[List[datetime.date]] = Field(
        None, alias="removeDates", validate_default=True
    )

    # Additional field for file modification time
    last_update: float = -float("inf")

    # -----------------------
    # Derived Fields
    # -----------------------
    time_step: Optional[pd.Timedelta] = None
    end_time: Optional[pd.Timedelta] = None
    total_steps: Optional[int] = None

    # -----------------------
    # Field Validators (v2)
    # -----------------------
    @field_validator(
        "trading_start", 
        "trading_end", 
        "liquid_start", 
        "liquid_end",
        "regular_start",
        "regular_end",
        mode="before"
    )
    @classmethod
    def parse_timedelta(cls, value):
        """
        Convert a time string (or similar) to a pandas.Timedelta
        before assignment to the model field.
        """
        try:
            return pd.to_timedelta(value)
        except Exception as exc:
            raise ValueError(f"Error parsing timedelta from value {value}: {exc}") from exc

    @field_validator("remove_dates", mode="before")
    @classmethod
    def parse_remove_dates(cls, value):
        """
        Convert each date string in removeDates to a datetime.date object.
        """
        if value is None:
            return None
        try:
            return [pd.to_datetime(date_str).date() for date_str in value]
        except Exception as exc:
            raise ValueError(f"Error parsing remove_dates: {exc}") from exc

    # -----------------------
    # Model Validator (v2)
    # -----------------------
    @model_validator(mode="after")
    @classmethod
    def compute_derived_fields(cls, model: "InstrumentData") -> "InstrumentData":
        """
        Compute additional fields after all base fields are parsed:
          - time_step: derived from `interval`.
          - end_time: calculated as liquid_end - trading_start - time_step.
          - total_steps: number of steps between trading_start and trading_end.
        """
        try:
            if model.interval is None:
                raise ValueError("Interval is required.")

            # Convert interval string -> Timedelta
            model.time_step = pd.to_timedelta(model.interval)

            if model.trading_start is None or model.trading_end is None:
                raise ValueError("Both trading_start and trading_end are required.")

            model.end_time = model.trading_end - model.trading_start - model.time_step

            total_seconds = model.end_time.total_seconds()
            step_seconds = model.time_step.total_seconds()
            if step_seconds <= 0:
                raise ValueError("Invalid time_step: must be positive.")

            model.total_steps = int(total_seconds / step_seconds) + 1

        except Exception as exc:
            raise ValueError(f"Error computing derived fields: {exc}") from exc

        return model

    # -----------------------
    # Pydantic v2 Config
    # -----------------------
    model_config = {
        "arbitrary_types_allowed": True,  # allow pandas.Timedelta
        "alias_generator": to_camel,      # snake_case -> camelCase
        "populate_by_name": True,         # allow field population by pythonic names
    }


class InstrumentConfig:
    """
    InstrumentConfig class

    Loads instrument configuration from a JSON file and provides methods to retrieve individual
    instrument configurations as InstrumentData instances.
    """
    def __init__(self, filename: str = "data/instruments.json"):
        self.filename = filename
        self.last_update: Optional[float] = None
        self.data = {}
        self._load_data()

    def _load_data(self):
        """
        Load data from the JSON file, updating self.data and self.last_update.
        """
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file '{self.filename}' not found.") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from '{self.filename}': {e}") from e

        try:
            path = Path(self.filename)
            self.last_update = path.stat().st_mtime
        except Exception as e:
            raise OSError(f"Error accessing file '{self.filename}': {e}") from e

    def reload_if_updated(self):
        """
        Check if the configuration file has been updated on disk.
        If so, reload the configuration data.
        """
        try:
            current_mtime = Path(self.filename).stat().st_mtime
        except Exception as e:
            raise OSError(f"Error accessing file '{self.filename}': {e}") from e
        if self.last_update is None or current_mtime > self.last_update:
            self._load_data()

    def get_config(self, instrument_key: str) -> InstrumentData:
        """
        Retrieve the instrument configuration for a given instrument key.

        Parameters
        ----------
        instrument_key : str
            The instrument key, e.g. "BTCUSD@bitfinex:1m".

        Returns
        -------
        InstrumentData
            A validated and enriched InstrumentData instance containing the configuration.
        """
        self.reload_if_updated()
        try:
            instrument_dict = self.data[instrument_key]
        except KeyError as e:
            raise KeyError(f"Instrument configuration '{instrument_key}' not found in '{self.filename}'.") from e

        try:
            # Create an InstrumentData instance and include the file's last update time.
            instrument_config = InstrumentData(**instrument_dict, last_update=self.last_update)
        except Exception as e:
            raise ValueError(f"Error creating InstrumentData for '{instrument_key}': {e}") from e
        return instrument_config


# =============================================================================
# Path generation
# =============================================================================
def make_path(
    raw: bool,
    instrument: "InstrumentData",
    remove_file: bool = False,
    special_interval: Optional[str] = None,
    zipfile: bool = True,
) -> Path:
    """
    Generate a path to a data file (CSV or zipped CSV).

    Parameters
    ----------
    raw : bool
        If True, generate a path under the "raw" folder; otherwise, under "processed".
    instrument : InstrumentData
        Instrument configuration.
    remove_file : bool, optional
        If True, remove any existing file at this path.
    special_interval : str, optional
        A special interval to override instrument.interval.
    zipfile : bool, optional
        If True, the file extension will be ".zip"; otherwise, ".csv".

    Returns
    -------
    Path
        The full local file path.
    """
    source = "raw" if raw else "processed"
    interval = special_interval if special_interval is not None else instrument.interval
    file_name = instrument.symbol
    path = Path(settings.DATA_FOLDER, source, instrument.type, interval, file_name)
    path = path.with_suffix(".zip" if zipfile else ".csv")

    # Ensure parent directories exist.
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


# =============================================================================
# S3 Download/Upload Helpers
# =============================================================================
def download_s3_file(
    bucket: str,
    key: str,
    target_path: Path
) -> None:
    """
    Download a file from S3 to the specified local target path.

    Parameters
    ----------
    bucket : str
        The name of the S3 bucket.
    key : str
        The S3 key (path) of the file to download.
    target_path : Path
        The full local file path where the file should be saved.
    """
    try:
        s3 = boto3.client('s3')
    except Exception as e:
        raise RuntimeError("Error initializing S3 client") from e

    try:
        os.makedirs(target_path.parent, exist_ok=True)
    except Exception as e:
        raise OSError(f"Error creating directories for {target_path.parent}: {e}") from e

    try:
        s3.download_file(bucket, key, str(target_path))
    except Exception as e:
        raise RuntimeError(f"Error downloading file from S3 (bucket='{bucket}', key='{key}')") from e


def upload_s3_file(bucket: str, key: str, local_path: Path) -> str:
    """
    Upload a file from the local directory to S3, preserving folder structure.

    Parameters
    ----------
    bucket : str
        The name of the S3 bucket.
    key : str
        The S3 key (destination path) for the uploaded file.
    local_path : Path
        The local file path of the file to upload.

    Returns
    -------
    str
        The S3 key where the file was uploaded.
    """
    try:
        s3 = boto3.client('s3')
    except Exception as e:
        raise RuntimeError("Error initializing S3 client") from e

    try:
        s3.upload_file(str(local_path), bucket, key)
    except Exception as e:
        raise RuntimeError(f"Error uploading file to S3 (bucket='{bucket}', key='{key}', local_path='{local_path}')") from e

    return key


def check_s3_file_exists(bucket_name: str, key: str) -> bool:
    """
    Check if a file with the given key exists in the specified S3 bucket.

    Parameters
    ----------
    bucket_name : str
        The S3 bucket name.
    key : str
        The S3 key (path) of the file.

    Returns
    -------
    bool
        True if the file exists; False otherwise.
    """
    try:
        s3_client = boto3.client("s3")
    except Exception as e:
        raise RuntimeError("Error initializing S3 client") from e

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=key,
            MaxKeys=1
        )
    except Exception as e:
        raise RuntimeError(f"Error listing objects in S3 bucket '{bucket_name}' with prefix '{key}'") from e

    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'] == key:
                return True

    return False


def make_s3_key(instrument: "InstrumentData", zipfile: bool) -> str:
    """
    Build an S3 key for the instrument data file.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    zipfile : bool
        If True, use a ".zip" extension; otherwise, ".csv".

    Returns
    -------
    str
        S3 key in the format "<type>/<interval>/<symbol><extension>".
    """
    extension = ".zip" if zipfile else ".csv"
    return f"{instrument.type}/{instrument.interval}/{instrument.symbol}{extension}"


def get_s3_last_modified(bucket: str, key: str) -> float:
    """
    Retrieve the last modified timestamp for an S3 object.

    Parameters
    ----------
    bucket : str
        The S3 bucket name.
    key : str
        The S3 key of the object.

    Returns
    -------
    float
        The last modified time as a Unix timestamp.
    """
    try:
        s3 = boto3.client("s3")
        response = s3.head_object(Bucket=bucket, Key=key)
        return response["LastModified"].timestamp()
    except Exception as e:
        raise RuntimeError(f"Error retrieving S3 metadata for s3://{bucket}/{key}") from e


def torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    """
    Convert a torch dtype to the equivalent numpy dtype.
    """
    return torch.empty((), dtype=dtype).numpy().dtype


# =============================================================================
# High-Level Data Functions
# =============================================================================
def download_data(instrument: "InstrumentData", target_path: Path, zipfile: bool) -> None:
    """
    Download raw data for the given instrument from S3.

    The raw data file is stored in S3 under the key:
        "<instrument.type>/<instrument.interval>/<instrument.symbol>.zip"
    The file is saved locally at the specified target path.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    target_path : Path
        The local path where the downloaded file should be saved.
    zipfile : bool
        Whether the file is zipped.
    """
    try:
        s3_key = make_s3_key(instrument, zipfile)
        print(f"Downloading S3 file: s3://{settings.S3_BUCKET}/{s3_key}")
        download_s3_file(settings.S3_BUCKET, s3_key, target_path)
        print(f"Downloaded file saved to: {target_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download data for instrument {instrument.symbol}") from e


def try_download_processed_file(
    instrument: "InstrumentData", target_path: Path, threshold: float, reset: bool, zipfile: bool
) -> bool:
    """
    Attempt to download an up-to-date processed file from S3.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    target_path : Path
        The local path where the processed file should be saved.
    threshold : float
        The staleness threshold (Unix timestamp) that the processed file must meet.
    reset : bool
        If True, force reprocessing.
    zipfile : bool
        Whether the file is zipped.

    Returns
    -------
    bool
        True if an up-to-date processed file was downloaded; False otherwise.
    """
    processed_s3_key = make_s3_key(instrument, zipfile)

    if not check_s3_file_exists(settings.S3_BUCKET_PROCESSED, processed_s3_key):
        print(f"S3 processed file s3://{settings.S3_BUCKET_PROCESSED}/{processed_s3_key} does not exist.")
        return False

    try:
        s3_processed_timestamp = get_s3_last_modified(settings.S3_BUCKET_PROCESSED, processed_s3_key)
    except Exception as e:
        raise RuntimeError(f"Error retrieving processed S3 metadata for instrument {instrument.symbol}") from e

    if s3_processed_timestamp < threshold or reset:
        print(f"S3 processed file s3://{settings.S3_BUCKET_PROCESSED}/{processed_s3_key} is stale.")
        return False

    print(f"S3 processed file s3://{settings.S3_BUCKET_PROCESSED}/{processed_s3_key} is up-to-date. Downloading...")
    try:
        download_s3_file(settings.S3_BUCKET_PROCESSED, processed_s3_key, target_path)
    except Exception as e:
        raise RuntimeError(f"Error downloading processed file from S3 for instrument {instrument.symbol}") from e
    return True


def ensure_raw_data(instrument: "InstrumentData", zipfile: bool, reset: bool) -> Path:
    """
    Ensure that the raw data file is available locally and is up-to-date.

    If the local raw file does not exist, is stale compared to the S3 version, 
    or if a reset is requested, this function downloads the raw data from S3.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    zipfile : bool
        Whether the file is zipped.
    reset : bool
        If True, force re-download of raw data.

    Returns
    -------
    Path
        The local file path for the raw data.
    """
    try:
        raw_path = make_path(raw=True, instrument=instrument, zipfile=zipfile)
    except Exception as e:
        raise RuntimeError(f"Error generating raw file path for instrument {instrument.symbol}") from e

    # Check staleness if the file exists and reset is not already requested.
    if raw_path.exists() and not reset:
        try:
            raw_s3_key = make_s3_key(instrument, zipfile)
            s3_timestamp = get_s3_last_modified(settings.S3_BUCKET, raw_s3_key)
        except Exception as e:
            raise RuntimeError(f"Error retrieving raw S3 metadata for instrument {instrument.symbol}") from e

        local_mtime = raw_path.stat().st_mtime
        if local_mtime < s3_timestamp:
            print(f"Local raw file {raw_path} is stale (local mtime: {local_mtime} < S3 timestamp: {s3_timestamp}). Re-downloading raw data...")
            reset = True
        else:
            print(f"Local raw file {raw_path} is up-to-date.")

    # If the file doesn't exist or if a reset is required, download it.
    if not raw_path.exists() or reset:
        print(f"Raw file {raw_path} missing or reset requested. Downloading raw data...")
        try:
            download_data(instrument=instrument, target_path=raw_path, zipfile=zipfile)
        except Exception as e:
            raise RuntimeError(f"Error downloading raw data for instrument {instrument.symbol}") from e
    else:
        print(f"Raw file {raw_path} is available.")

    return raw_path


def ensure_processed_data(instrument: "InstrumentData", zipfile: bool, reset: bool) -> Path:
    """
    Ensure that the processed data file is available locally and up-to-date.

    The logic is:
      1. Determine a staleness threshold based on the raw S3 file's timestamp and instrument.last_update.
      2. If the local processed file exists and is fresh, return it.
      3. Otherwise, attempt to download an up-to-date processed file from S3.
      4. If that fails, reprocess the raw data, upload it to S3, and return the local file.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    zipfile : bool
        Whether the file is zipped.
    reset : bool
        If True, force reprocessing.

    Returns
    -------
    Path
        The local file path for the processed data.
    """
    try:
        local_path = make_path(raw=False, instrument=instrument, zipfile=zipfile)
    except Exception as e:
        raise RuntimeError(f"Error generating processed file path for instrument {instrument.symbol}") from e

    try:
        raw_s3_key = make_s3_key(instrument, zipfile)
        raw_s3_timestamp = get_s3_last_modified(settings.S3_BUCKET, raw_s3_key)
    except Exception as e:
        raise RuntimeError(f"Error retrieving raw S3 metadata for instrument {instrument.symbol}") from e

    # The staleness threshold is the later of the raw file's timestamp and instrument.last_update.
    threshold = max(raw_s3_timestamp, instrument.last_update)
    if local_path.exists():
        local_mtime = local_path.stat().st_mtime
        if local_mtime >= threshold and not reset:
            print(f"Local processed file {local_path} is up-to-date.")
            return local_path
        print(f"Local processed file {local_path} is stale.")
    else:
        print(f"Local processed file {local_path} does not exist.")

    # Try to download an up-to-date processed file from S3.
    if try_download_processed_file(instrument, local_path, threshold, reset, zipfile):
        return local_path

    # If no up-to-date S3 file is available, reprocess the raw data.
    print(f"Reprocessing raw data for instrument {instrument.symbol}.")
    try:
        process_data(instrument, zipfile)
    except Exception as e:
        raise RuntimeError(f"Error processing data for instrument {instrument.symbol}") from e

    processed_s3_key = make_s3_key(instrument, zipfile)

    print(f"Uploading processed file to S3: s3://{settings.S3_BUCKET_PROCESSED}/{processed_s3_key}")
    try:
        upload_s3_file(settings.S3_BUCKET_PROCESSED, processed_s3_key, local_path)
    except Exception as e:
        raise RuntimeError(f"Error uploading processed file to S3 for instrument {instrument.symbol}") from e

    return local_path


# =============================================================================
# Main load_data() Function
# =============================================================================
def load_data(
    raw: bool,
    instrument: InstrumentData,
    dtype: str = "float32",
    reset: bool = False,
    zipfile: bool = True,
) -> pd.DataFrame:
    """
    Load data from CSV (or zipped CSV) files.

    For raw data, this function ensures the local file is available (downloading it if needed).
    For processed data, the function:
      1. Checks if a local processed file exists and is fresh.
      2. If not, it attempts to download an up-to-date processed file from S3.
      3. Otherwise, it reprocesses the raw data, uploads the processed file to S3, and then uses it.

    Parameters
    ----------
    raw : bool
        If True, load raw data; otherwise, load processed data.
    instrument : InstrumentData
        Instrument configuration object.
    dtype : str, optional
        Data type for numerical columns. Default is "float32".
    reset : bool, optional
        If True, force reprocessing. Default is False.
    zipfile : bool, optional
        If True, files are zipped (.zip) and will be read with compression="zip". Default is False.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the loaded data.
    """
    try:
        if raw:
            file_path = ensure_raw_data(instrument, zipfile, reset)
        else:
            file_path = ensure_processed_data(instrument, zipfile, reset)
    except Exception as e:
        raise RuntimeError(f"Error ensuring data availability for instrument {instrument.symbol}") from e

    print(f"Loading data from {file_path}")

    if raw:
        columns = ["date", "time", "open", "high", "low", "close", "volume"]
        dtypes = {"open": dtype, "high": dtype, "low": dtype, "close": dtype, "volume": "int32"}
    else:
        columns = ["date", "time", "trade_date", "offset_time", "open", "high", "low", "close", "volume"]
        dtypes = {
            "open": dtype,
            "high": dtype,
            "low": dtype,
            "close": dtype,
            "volume": "int32",
            "date": "int32",
            "time": "int32",
            "trade_date": "int32",
            "offset_time": "int32",
        }

    read_csv_kwargs = {
        "header": None,
        "parse_dates": False,
        "names": columns,
        "dtype": dtypes,
    }
    if zipfile:
        read_csv_kwargs["compression"] = "zip"

    try:
        df: pd.DataFrame = pd.read_csv(file_path, **read_csv_kwargs)
    except Exception as e:
        raise ValueError(f"Error reading CSV file at {file_path}: {e}") from e

    if raw:
        try:
            df["date_time"] = pd.to_datetime(df["date"] + " " + df["time"])
        except Exception as e:
            raise ValueError("Error converting 'date' and 'time' columns to datetime") from e
        df = df.drop(columns=["date", "time"]).set_index("date_time")

    return df


def process_data(instrument: InstrumentData, zipfile: bool) -> None:
    """
    Process raw data and save to a CSV (or zipped CSV) file.

    This function performs the following steps:
      1. Loads the raw data (using the `zipfile` parameter as needed).
      2. Computes additional date/time columns (including trading offset and trade_date).
      3. Removes rows corresponding to any dates in instrument.removeDates (if specified).
      4. Groups the data by trade_date and adds missing rows for each expected time step.
      5. Filters rows outside the trading hours and days with an incomplete number of time steps.
      6. Computes ordinal and time columns.
      7. Saves the processed data to a file using make_path (with the proper file extension).

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    zipfile : bool, optional
        If True, both input and output files use a .zip extension and are read/written with compression.
        Default is False.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If any step in the processing or saving of data fails.
    """
    try:
        # Retrieve timing parameters from the instrument configuration.
        time_step = instrument.time_step
        start_time_offset = instrument.trading_start
        start_time = pd.Timedelta(0)
        end_time = instrument.end_time
        total_steps = instrument.total_steps
        offset_seconds = start_time_offset.total_seconds()

        # Load raw data with the zipfile parameter.
        df = load_data(raw=True, instrument=instrument, dtype="float64", zipfile=zipfile)

        # Compute 'date' and 'time' columns from the DataFrame index.
        dt_index = pd.to_datetime(df.index)
        df["date"] = pd.to_datetime(dt_index.date)
        df["time"] = pd.to_timedelta(
            dt_index.hour * 3600 + dt_index.minute * 60 + dt_index.second, unit="s"
        )

        # Calculate offset_time relative to the trading start offset.
        df["offset_time"] = df["time"] - start_time_offset
        # Remove any day component from offset_time.
        df["offset_time"] = df["offset_time"].apply(
            lambda x: x - pd.to_timedelta(x.days, unit="d")
        )

        # Compute trade_date (e.g. for futures, trading may start on the previous day).
        df["trade_date"] = pd.to_datetime((dt_index - start_time_offset).date)

        # Remove rows for dates that are in instrument.removeDates (if any).
        if instrument.remove_dates is not None:
            df = df[~df["trade_date"].isin(instrument.remove_dates)]

        # Group by trade_date and add missing rows for each time step.
        df = (
            df.groupby("trade_date")
            .apply(
                add_missing_rows,
                start_time=start_time,
                end_time=end_time,
                time_step=time_step,
            )
            .reset_index(drop=True)
        )

        # Filter out rows outside trading hours.
        df = df[(df["offset_time"] >= start_time) & (df["offset_time"] <= end_time)]

        # Filter out days that do not have the expected number of rows.
        df = df.groupby("trade_date").filter(lambda x: x["open"].count() == total_steps)

        # Sort by trade_date and offset_time.
        df.sort_values(["trade_date", "offset_time"], inplace=True)

        # Compute ordinal and time-based columns.
        df["ord_trade_date"] = df["trade_date"].map(lambda x: x.toordinal()).astype("int32")
        df["time_seconds"] = (
            (df["offset_time"].map(lambda x: x.total_seconds()) + offset_seconds)
            .mod(SECONDS_IN_DAY)
            .astype("int32")
        )
        df["offset_time_seconds"] = df["offset_time"].map(lambda x: x.total_seconds()).astype("int32")
        df["ord_date"] = df["ord_trade_date"] + (
            (df["offset_time_seconds"] + offset_seconds) // SECONDS_IN_DAY
        ).astype("int32")

        # Select and reorder the final columns.
        df = df[
            [
                "ord_date",
                "time_seconds",
                "ord_trade_date",
                "offset_time_seconds",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].reset_index(drop=True)

        # Save to CSV. When zipfile is True, pass compression="zip" so that the output is a zipped file.
        try:
            output_path = make_path(raw=False, instrument=instrument, remove_file=True, zipfile=zipfile)
            if zipfile:
                df.to_csv(output_path, header=False, index=False, compression="zip")
            else:
                df.to_csv(output_path, header=False, index=False)
            print(f"Processed data saved to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving processed file for instrument {instrument.symbol}: {e}") from e

    except Exception as e:
        raise RuntimeError(f"Error processing data for instrument {instrument.symbol}: {e}") from e


def add_missing_rows(
    group: pd.Series | pd.DataFrame,
    start_time: pd.Timedelta,
    end_time: pd.Timedelta,
    time_step: pd.Timedelta,
) -> pd.DataFrame:
    """
    This function takes a DataFrame group and a time range defined by start_time, end_time, and time_step.
    It creates a new DataFrame that includes all time steps in the given range, not just those present in the input DataFrame.
    For each new row (time step) added, it fills in missing data as follows:
      - 'open', 'high', 'low', 'close' columns are filled with the previous 'close' value.
      - 'volume' is filled with 0.
    The function returns the new DataFrame with added rows and filled missing data.
    """
    all_time_steps = pd.timedelta_range(start=start_time, end=end_time, freq=time_step)

    # Create a DataFrame with all time steps for the day.
    all_time_step_rows = pd.DataFrame(
        {
            "trade_date": group["trade_date"].iloc[0],
            "offset_time": all_time_steps,
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": np.nan,
            "volume": np.nan,
        }
    )

    # Merge the all_time_step_rows with the original group, ensuring every time step is present.
    merged = pd.merge(
        group,
        all_time_step_rows,
        on=["trade_date", "offset_time"],
        how="outer",
        suffixes=("", "_y"),
        sort=True,
    )[group.columns]

    # Forward-fill missing 'close' values from previous rows.
    merged["close"] = merged["close"].ffill()
    # For any missing 'close' values at the beginning (which ffill won't fill), use the first available 'open' value.
    first_open = group["open"].iloc[0]
    merged["close"] = merged["close"].fillna(first_open)

    # Back-fill missing values in 'open', 'high', 'low', and 'close' using available data from the right.
    merged[["open", "high", "low", "close"]] = merged[["open", "high", "low", "close"]].bfill(axis=1)
    merged["volume"] = merged["volume"].fillna(0).astype("int32")

    return merged


def load_data_tensor(
    instrument: "InstrumentData", 
    reset: bool = False, 
    zipfile: bool = True, 
    dtype: torch.dtype = torch.float32, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Load a processed data file for the given instrument and convert it into a PyTorch tensor.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration object.
    reset : bool, optional
        If True, force reprocessing/download of data even if a local file exists. Default is False.
    zipfile : bool, optional
        Whether the file on disk is zipped. Default is True.
    dtype : torch.dtype, optional
        The torch data type for the returned tensor. Default is torch.float32.
    device : torch.device, optional
        The device on which to create the tensor. If None, uses CUDA if available, else CPU.

    Returns
    -------
    torch.Tensor
        A tensor containing the processed data.
    """
    # Auto-detect device if not provided.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert the torch dtype to a string (e.g. "float32") for load_data.
    dtype_str = torch_dtype_to_numpy_dtype(dtype).name

    # Load the processed data as a pandas DataFrame.
    # Here, raw=False indicates we want processed data.
    df: pd.DataFrame = load_data(
        raw=False,
        instrument=instrument,
        dtype=dtype_str,
        reset=reset,
        zipfile=zipfile
    )

    # Convert the DataFrame to a numpy array.
    # (Assuming the DataFrame contains only numeric data.)
    np_array = df.to_numpy()

    # Convert the numpy array into a torch tensor on the requested device and dtype.
    tensor = torch.as_tensor(np_array, dtype=dtype, device=device)
    return tensor
