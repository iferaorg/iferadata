import torch
import time
from einops import rearrange
from .file_manager import FileManager
from .s3_utils import download_s3_file, upload_s3_file, make_s3_key
from .settings import settings
from .data_loading import load_data
from .data_processing import process_data
from .config import ConfigManager
from .enums import Scheme, Source, extension_map
from .url_utils import make_url
from .file_utils import make_path


def download_file(
    source: str, type: str, interval: str, symbol: str, ext: str, reset: bool
) -> None:
    """
    Download a file from S3 if it is not up to date.
    The file is expected to be in a specific directory structure.
    The S3 file is expected to exist and up-to-date.
    """
    # Convert string source to Source enum
    source_enum = Source(source)

    if f".{ext}" != extension_map[source_enum]:
        raise ValueError(
            f"Extension '{ext}' does not match the expected extension for source '{source_enum.value}'."
        )

    path = f"{source}/{type}/{interval}/{symbol}{extension_map[source_enum]}"

    file_path = make_path(source_enum, type, interval, symbol)
    file_url = make_url(Scheme.FILE, source_enum, type, interval, symbol)
    fm = FileManager()

    if reset or not fm.is_up_to_date(file_url):
        bucket = settings.S3_BUCKET
        download_s3_file(bucket, path, str(file_path))


def process_raw_file(type: str, interval: str, symbol: str, reset: bool) -> None:
    """
    Process a raw file from S3 if it is not up to date.
    The file is expected to be in a specific directory structure.
    The S3 file is expected to exist and up-to-date.
    """

    file_url = make_url(Scheme.FILE, Source.PROCESSED, type, interval, symbol)
    fm = FileManager()

    if reset or not fm.is_up_to_date(file_url):
        raw_file_path = make_path(Source.RAW, type, interval, symbol)
        raw_file_exists = raw_file_path.exists()
        cm = ConfigManager()
        instrument = cm.get_base_instrument_config(f"{symbol}:{interval}")

        df = load_data(raw=True, instrument=instrument)
        process_data(df, instrument=instrument, zipfile=True)
        # Upload the processed file to S3
        bucket = settings.S3_BUCKET
        key = make_s3_key(Source.PROCESSED, instrument, zipfile=True)
        file_path = make_path(Source.PROCESSED, type, interval, symbol)
        upload_s3_file(bucket, key, str(file_path))
        # Touch the local file to update its timestamp
        file_path.touch()

        # Remove the raw file if it was not already present
        if not raw_file_exists:
            raw_file_path.unlink(missing_ok=True)


def process_tensor_file(type: str, interval: str, symbol: str, reset: bool) -> None:
    """
    Process the processed file to generate the tensor file and upload to S3.
    """
    fm = FileManager()
    tensor_file_url = make_url(Scheme.FILE, Source.TENSOR, type, interval, symbol)

    if reset or not fm.is_up_to_date(tensor_file_url):
        cm = ConfigManager()
        instrument = cm.get_base_instrument_config(f"{symbol}:{interval}")
        processed_file_path = make_path(Source.PROCESSED, type, interval, symbol)
        processed_file_exists = processed_file_path.exists()
        # Load processed DataFrame
        df = load_data(raw=False, instrument=instrument, zipfile=True)
        # Convert to tensor
        tensor = torch.as_tensor(df.to_numpy(), dtype=torch.float32)
        tensor = rearrange(tensor, "(d t) c -> d t c", t=instrument.total_steps)
        tensor = tensor[..., 4:].clone()  # Skip first 4 columns
        # Save locally
        tensor_file_path = make_path(Source.TENSOR, type, interval, symbol)
        torch.save(tensor, str(tensor_file_path))
        # Upload to S3
        bucket = settings.S3_BUCKET
        s3_key = f"tensor/{type}/{interval}/{symbol}.pt"
        upload_s3_file(bucket, s3_key, str(tensor_file_path))
        # Touch the local file to update its timestamp
        tensor_file_path.touch()
        
        # Remove the processed file if it was not already present
        if not processed_file_exists:
            processed_file_path.unlink(missing_ok=True)
