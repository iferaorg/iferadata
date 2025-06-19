"""Utility functions for refreshing and processing local data files."""

import datetime as dt
import pathlib as pl
import time

import torch
import yaml
from einops import rearrange
from tqdm import tqdm

from .url_utils import contract_notice_and_expiry, make_url
from .s3_utils import download_s3_file, upload_s3_file, check_s3_file_exists
from .data_loading import load_data, load_data_tensor
from .data_processing import process_data, calculate_rollover
from .config import ConfigManager
from .enums import Source, extension_map, Scheme, ExpirationRule
from .file_utils import make_path, write_tensor_to_gzip
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
    write_tensor_to_gzip(str(tensor_file_path), tensor)


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
    file_path = make_path(Source.META, "futures", "dates_raw", symbol)

    s3_key = f"{Source.META.value}/futures/dates_raw/{symbol}.yml"

    if check_s3_file_exists(s3_key):
        download_s3_file(s3_key, str(file_path))
        with open(file_path, "r", encoding="utf-8") as fh:
            dates = yaml.safe_load(fh)

    print(f"Fetching contract dates for {symbol} ({broker_symbol})...")

    for code in tqdm(contract_codes):
        if code in dates:
            continue  # Skip if already processed

        full_symbol = f"{broker_symbol}{code}"  # e.g. "GCF10"
        first_notice, expiration = contract_notice_and_expiry(full_symbol)

        dates[code] = {
            "first_notice_date": first_notice.isoformat() if first_notice else None,
            "expiration_date": expiration.isoformat() if expiration else None,
        }

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


def calculate_rollover_spec(symbol: str, contract_codes: list[str]) -> None:
    """Generate a YAML rollover specification for a futures symbol.

    This loads metadata and tensors for each contract, determines when to roll
    from one contract to the next and calculates the multiplier required to
    stitch prices together.

    The resulting file is written to ``meta/futures/rollover/{symbol}.yml`` and
    contains a list of dictionaries with the keys ``"start_date"``,
    ``"contract_code"`` and ``"multiplier"``.

    Parameters
    ----------
    symbol
        Root futures symbol, e.g. ``"GC"``.
    contract_codes
        List of contract codes such as ``["F10", "G10", …]``.
    """

    dates_path = make_path(Source.META, "futures", "dates", symbol)
    if not dates_path.exists():
        raise FileNotFoundError(
            f"Metadata file for {symbol} not found at {dates_path}. "
            "Please run process_futures_metadata first."
        )

    with open(dates_path, "r", encoding="utf-8") as fh:
        dates = yaml.safe_load(fh)

    cm = ConfigManager()
    base_instrument = cm.get_base_instrument_config(symbol, "30m")

    contract_tensors = []
    contract_instruments = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for code in contract_codes:
        if code not in dates:
            print(f"Warning: No metadata found for {symbol}-{code}. Skipping.")
            continue

        contract_instrument = cm.create_derived_base_config(
            base_instrument, contract_code=code
        )

        if dates[contract_instrument.contract_code]["first_notice_date"] is not None:
            contract_instrument.first_notice_date = dt.date.fromisoformat(
                dates[contract_instrument.contract_code]["first_notice_date"]
            )

        if dates[contract_instrument.contract_code]["expiration_date"] is not None:
            contract_instrument.expiration_date = dt.date.fromisoformat(
                dates[contract_instrument.contract_code]["expiration_date"]
            )

        contract_tensor = load_data_tensor(
            instrument=contract_instrument,
            dtype=torch.float64,
            device=device,
            strip_date_time=False,
        )

        contract_instruments.append(contract_instrument)
        contract_tensors.append(contract_tensor)

    # Sort both contract_instruments and contract_tensors by expiration date
    sorted_indices = sorted(
        range(len(contract_instruments)),
        key=lambda k: contract_instruments[k].expiration_date,
    )
    contract_instruments_sorted = [contract_instruments[i] for i in sorted_indices]
    contract_tensors_sorted = [contract_tensors[i] for i in sorted_indices]

    ratios, active_idx, all_dates_ord = calculate_rollover(
        contract_instruments_sorted, contract_tensors_sorted
    )

    # Get the indices where rollover occurs (non-NaN values in ratios), plus the first index
    rollover_indices = torch.nonzero(~ratios.isnan(), as_tuple=True)[0]
    rollover_indices = torch.cat(
        (
            torch.tensor(
                [0], device=rollover_indices.device, dtype=rollover_indices.dtype
            ),
            rollover_indices,
        ),
        dim=0,
    )

    # Calculate the cumulative product of the ratios backwards
    ratios_tmp = torch.cat(
        (
            ratios.nan_to_num(1.0),
            torch.ones(1, device=ratios.device, dtype=ratios.dtype),
        ),
        dim=0,
    )
    ratios_tmp = torch.flip(ratios_tmp, dims=[0])
    ratios_cump = torch.cumprod(ratios_tmp, dim=0)
    ratios_cump = torch.flip(ratios_cump, dims=[0]).to(torch.float32)

    rollover_spec = []

    for active, pos, ratio in zip(
        active_idx[rollover_indices].cpu().numpy(),
        rollover_indices.cpu().numpy(),
        ratios_cump[rollover_indices + 1].cpu().numpy(),
    ):
        contract_code = contract_instruments_sorted[active].contract_code
        rollover_spec.append(
            {
                "start_date": dt.date.fromordinal(all_dates_ord[pos]),
                "contract_code": contract_code,
                "multiplier": ratio.item(),
            }
        )

    # Save the rollover spec to a YAML file
    rollover_spec_path = make_path(Source.META, "futures", "rollover", symbol)
    with open(rollover_spec_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            rollover_spec,
            fh,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def process_futures_backadjusted_tensor(
    symbol: str, interval: str, contract_codes: list[str] | None = None
) -> None:
    """Create a back-adjusted futures tensor based on a rollover specification.

    Parameters
    ----------
    symbol : str
        Root futures symbol.
    interval : str
        Interval of the desired tensor (e.g. ``"30m"``).
    contract_codes : list[str] | None, optional
        List of contract codes. This argument is optional as the required
        contracts can be inferred from the rollover specification.
    """
    rollover_spec_path = make_path(Source.META, "futures", "rollover", symbol)
    if not rollover_spec_path.exists():
        raise FileNotFoundError(
            f"Rollover specification for {symbol} not found at {rollover_spec_path}."
        )

    with open(rollover_spec_path, "r", encoding="utf-8") as fh:
        rollover_spec = yaml.safe_load(fh)

    if not isinstance(rollover_spec, list) or not rollover_spec:
        raise ValueError("Rollover specification is empty or malformed")

    cm = ConfigManager()
    base_instrument = cm.get_base_instrument_config(symbol, interval)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trade_date_col = 2
    offset_col = 3
    price_slice = slice(4, 8)
    rollover_offset = base_instrument.rollover_offset

    segments: list[torch.Tensor] = []

    for idx, entry in enumerate(rollover_spec):
        code = entry["contract_code"]
        if contract_codes is not None and code not in contract_codes:
            continue
        multiplier = float(entry["multiplier"])
        start_ord = dt.date.fromisoformat(str(entry["start_date"])).toordinal()

        contract_instrument = cm.create_derived_base_config(
            base_instrument, contract_code=code
        )
        tens = load_data_tensor(
            instrument=contract_instrument,
            dtype=torch.float32,
            device=device,
            strip_date_time=False,
        )
        tens = rearrange(tens, "d t c -> (d t) c")
        trade_date = tens[:, trade_date_col]
        offset_time = tens[:, offset_col]

        if idx == 0:
            start_mask = trade_date >= start_ord
        else:
            start_mask = (trade_date > start_ord) | (
                (trade_date == start_ord) & (offset_time >= rollover_offset)
            )

        if idx + 1 < len(rollover_spec):
            next_start_ord = dt.date.fromisoformat(
                str(rollover_spec[idx + 1]["start_date"])
            ).toordinal()
            end_mask = (trade_date < next_start_ord) | (
                (trade_date == next_start_ord) & (offset_time < rollover_offset)
            )
            mask = start_mask & end_mask
        else:
            mask = start_mask

        segment = tens[mask].clone()
        if segment.numel() == 0:
            continue
        segment[:, price_slice] *= multiplier
        segments.append(segment)

    if not segments:
        raise ValueError(f"No data found for symbol {symbol}")

    combined = torch.cat(segments, dim=0)
    steps = base_instrument.total_steps
    if combined.shape[0] % steps != 0:
        raise RuntimeError(
            "Combined tensor length is not divisible by instrument steps"
        )
    result = rearrange(combined, "(d t) c -> d t c", t=steps)

    tensor_file_path = make_path(
        Source.TENSOR, "futures_backadjusted", interval, symbol
    )
    write_tensor_to_gzip(str(tensor_file_path), result)
