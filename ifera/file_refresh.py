import torch
import pathlib as pl
from einops import rearrange
from .s3_utils import download_s3_file, upload_s3_file, make_s3_key
from .data_loading import load_data
from .data_processing import process_data
from .config import ConfigManager
from .enums import Source, extension_map
from .file_utils import make_path


def download_file(source: str, type: str, interval: str, symbol: str, ext: str) -> None:
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

    path = f"{source}/{type}/{interval}/{symbol}.{ext}"
    file_path = make_path(source_enum, type, interval, symbol)
    download_s3_file(path, str(file_path))


def upload_file(source: str, type: str, interval: str, symbol: str, ext: str) -> None:
    """
    Upload a file to S3 if it is not up to date.
    The file is expected to be in a specific directory structure.
    The S3 file is expected to exist and up-to-date.
    """
    # Convert string source to Source enum
    source_enum = Source(source)

    if f".{ext}" != extension_map[source_enum]:
        raise ValueError(
            f"Extension '{ext}' does not match the expected extension for source '{source_enum.value}'."
        )

    path = f"{source}/{type}/{interval}/{symbol}.{ext}"
    file_path = make_path(source_enum, type, interval, symbol)
    upload_s3_file(path, str(file_path))

    # Touch the local file to update its timestamp
    pl.Path(file_path).touch()


def process_raw_file(
    type: str,
    interval: str,
    symbol: str,
    contract_code: str = "",
) -> None:
    """
    Process a raw file from S3 if it is not up to date.
    The file is expected to be in a specific directory structure.
    The S3 file is expected to exist and up-to-date.
    """
    _ = type

    cm = ConfigManager()
    instrument = cm.get_base_instrument_config(symbol, interval)

    if contract_code:
        instrument = cm.create_derived_base_config(
            instrument, contract_code=contract_code
        )

    df = load_data(raw=True, instrument=instrument)
    process_data(df, instrument=instrument, zipfile=True)


def process_tensor_file(
    type: str,
    interval: str,
    symbol: str,
    contract_code: str = "",
) -> None:
    """
    Process the processed file to generate the tensor file and upload to S3.
    """
    cm = ConfigManager()
    instrument = cm.get_base_instrument_config(symbol, interval)

    if contract_code:
        instrument = cm.create_derived_base_config(
            instrument, contract_code=contract_code
        )

    # Load processed DataFrame
    df = load_data(raw=False, instrument=instrument, zipfile=True)

    # Convert to tensor
    tensor = torch.as_tensor(df.to_numpy(), dtype=torch.float32)
    tensor = rearrange(tensor, "(d t) c -> d t c", t=instrument.total_steps)
    tensor = tensor[..., 4:].clone()  # Skip first 4 columns

    # Save locally
    file_name = f"{symbol}-{contract_code}" if contract_code else symbol
    tensor_file_path = make_path(Source.TENSOR, type, interval, file_name)
    torch.save(tensor, str(tensor_file_path))
