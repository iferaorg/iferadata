from .file_manager import FileManager
from .s3_utils import download_s3_file, upload_s3_file, make_s3_key
from .settings import settings
from .data_loading import load_data
from .data_processing import process_data
from .config import ConfigManager
from .enums import Scheme, Source
from .url_utils import make_url
from .file_utils import make_path


def download_file(
    source: str, type: str, interval: str, symbol: str, reset: bool
) -> None:
    """
    Download a file from S3 if it is not up to date.
    The file is expected to be in a specific directory structure.
    The S3 file is expected to exist and up-to-date.
    """
    path = f"{source}/{type}/{interval}/{symbol}.zip"

    # Convert string source to Source enum
    source_enum = Source(source)

    file_path = make_path(source_enum, type, interval, symbol, zipfile=True)
    file_url = make_url(Scheme.FILE, source_enum, type, interval, symbol, zipfile=True)
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

    file_path = make_path(Source.PROCESSED, type, interval, symbol, zipfile=True)
    file_url = make_url(
        Scheme.FILE, Source.PROCESSED, type, interval, symbol, zipfile=True
    )
    fm = FileManager()
    cm = ConfigManager()
    raw_file_path = make_path(Source.RAW, type, interval, symbol, zipfile=True)
    raw_file_exists = raw_file_path.exists()
    instrument = cm.get_base_instrument_config(f"{symbol}:{interval}")

    if reset or not fm.is_up_to_date(file_url):
        df = load_data(raw=True, instrument=instrument)
        process_data(df, instrument=instrument, zipfile=True)
        # Upload the processed file to S3
        bucket = settings.S3_BUCKET
        key = make_s3_key(Source.PROCESSED, instrument, zipfile=True)
        upload_s3_file(bucket, key, str(file_path))

        # Remove the raw file if it was not already present
        if not raw_file_exists:
            raw_file_path.unlink(missing_ok=True)
