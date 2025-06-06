import torch
import pathlib as pl
import yaml
import datetime as dt
import time
from tqdm import tqdm
from einops import rearrange
from .url_utils import contract_notice_and_expiry, make_url
from .s3_utils import download_s3_file, upload_s3_file
from .data_loading import load_data
from .data_processing import process_data
from .config import ConfigManager
from .enums import Source, extension_map, Scheme, ExpirationRule
from .file_utils import make_path
from .config import ConfigManager
from .file_manager import FileManager
from .date_utils import calculate_expiration


def download_file(
    source: str,
    type: str,
    interval: str,
    symbol: str,
    ext: str,
    contract_code: str = "",
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

    if contract_code:
        symbol = f"{symbol}-{contract_code}"

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

    target_path = make_url(Scheme.S3, source_enum, type, interval, symbol)
    fm = FileManager()
    dep_last_modified = fm.dependencies_max_last_modified(
        target_path, scheme_filter=Scheme.S3
    )

    # Sleep for 1 second to ensure the file created timestamp is updated if necessary
    if (
        dep_last_modified
        and (dt.datetime.now(tz=dt.timezone.utc) - dep_last_modified).total_seconds()
        < 1
    ):
        time.sleep(1)

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
    # tensor = tensor[..., 4:].clone()  # Skip first 4 columns

    # Save locally
    file_name = f"{symbol}-{contract_code}" if contract_code else symbol
    tensor_file_path = make_path(Source.TENSOR, type, interval, file_name)
    torch.save(tensor, str(tensor_file_path))


def process_futures_metadata_raw(symbol: str, contract_codes: list[str]) -> None:
    """
    Fetch ``First Notice`` and ``Expiration`` dates for a set of futures
    contracts (``symbol`` + ``contract_code``) from Barchart and save
    them in a YAML file:

        meta/futures/dates/{symbol}.yml

    The YAML structure is::

        F10:
          first_notice_date: 2010-12-31
          expiration_date:   2011-01-27
        G10:
          first_notice_date: …
          expiration_date:   …

    Parameters
    ----------
    symbol
        Root futures symbol, e.g. ``"GC"``.
    contract_codes
        List of contract codes such as ``["F10", "G10", …]`` (month letter
        + 2-digit year).
    """
    dates: dict[str, dict[str, str | None]] = {}
    cm = ConfigManager()
    instrument = cm.get_config(broker_name="barchart", symbol=symbol, interval="30m")
    broker_symbol = instrument.broker_symbol

    print(f"Fetching contract dates for {symbol} ({broker_symbol})...")

    for code in tqdm(contract_codes):
        full_symbol = f"{broker_symbol}{code}"  # e.g. "GCF10"
        first_notice, expiration = contract_notice_and_expiry(full_symbol)

        dates[code] = {
            "first_notice_date": first_notice.isoformat() if first_notice else None,
            "expiration_date": expiration.isoformat() if expiration else None,
        }

    file_path = make_path(Source.META, "futures", "dates_raw", symbol)

    with open(file_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            dates,
            fh,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
        )


def process_futures_metadata(symbol: str) -> None:
    """
    Fetch ``First Notice`` and ``Expiration`` dates for a set of futures
    contracts (``symbol`` + ``contract_code``) from Barchart and save
    them in a YAML file:

        meta/futures/dates/{symbol}.yml

    The YAML structure is::

        F10:
          first_notice_date: 2010-12-31
          expiration_date:   2011-01-27
        G10:
          first_notice_date: …
          expiration_date:   …

    Parameters
    ----------
    symbol
        Root futures symbol, e.g. ``"GC"``.
    contract_codes
        List of contract codes such as ``["F10", "G10", …]`` (month letter
        + 2-digit year).
    """
    raw_yml_path = make_path(Source.META, "futures", "dates_raw", symbol)
    if not raw_yml_path.exists():
        raise FileNotFoundError(
            f"Raw metadata file for {symbol} not found at {raw_yml_path}. "
            "Please run process_futures_metadata_raw first."
        )

    cm = ConfigManager()
    instrument = cm.get_base_instrument_config(symbol, "30m")

    with open(raw_yml_path, "r", encoding="utf-8") as fh:
        dates = yaml.safe_load(fh)

        for code, date_info in dates.items():
            expiration = date_info.get("expiration_date")
            full_symbol = f"{symbol}-{code}"  # e.g. "GC-F10"
            contract_instrument = cm.create_derived_base_config(
                instrument, contract_code=code
            )

            # Overwrite first_notice_date with calculated value
            if contract_instrument.first_notice_day_rule is not None:
                first_notice_date = calculate_expiration(
                    month_code=contract_instrument.contract_code,  # type: ignore
                    rule=ExpirationRule(contract_instrument.first_notice_day_rule),
                    asset_class=contract_instrument.asset_class,
                )
            else:
                first_notice_date = None

            dates[code]["first_notice_date"] = (
                first_notice_date.isoformat() if first_notice_date else None
            )

            # Overwrite expiration_date with calculated value if not found
            if expiration is None:
                print(
                    f"Warning: Expiration date for {full_symbol} not found. Using last date in data."
                )
                expiration = calculate_expiration(
                    month_code=contract_instrument.contract_code,  # type: ignore
                    rule=ExpirationRule(contract_instrument.last_trading_day_rule),
                    asset_class=contract_instrument.asset_class,
                )

                dates[code]["expiration_date"] = (
                    expiration.isoformat() if expiration else None
                )

    file_path = make_path(Source.META, "futures", "dates", symbol)

    with open(file_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            dates,
            fh,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
        )
