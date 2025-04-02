from .enums import Scheme, Source
from .file_utils import make_path
from .config import BaseInstrumentConfig


def make_url(
    scheme: Scheme,
    source: Source,
    instrument_type: str,
    interval: str,
    symbol: str,
    zipfile: bool = True,
) -> str:
    """Generate a URL to a data file."""
    if scheme == Scheme.FILE:
        path = make_path(source, instrument_type, interval, symbol, zipfile=zipfile)
        return f"{scheme.value}:{path}"

    if source == Source.TENSOR:
        file_name = f"{symbol}.pt"
    else:
        file_name = f"{symbol}.zip" if zipfile else f"{symbol}.csv"

    url = f"{scheme.value}:{source.value}/{instrument_type}/{interval}/{file_name}"
    return url


def make_instrument_url(
    scheme: Scheme,
    source: Source,
    instrument: BaseInstrumentConfig,
    zipfile: bool = True,
) -> str:
    """Generate a URL to a data file."""
    return make_url(
        scheme,
        source,
        instrument.type,
        instrument.interval,
        instrument.symbol,
        zipfile=zipfile,
    )
