from .enums import Scheme, Source, extension_map
from .file_utils import make_path
from .config import BaseInstrumentConfig


def make_url(
    scheme: Scheme,
    source: Source,
    instrument_type: str,
    interval: str,
    symbol: str,
) -> str:
    """Generate a URL to a data file."""
    if scheme == Scheme.FILE:
        path = make_path(source, instrument_type, interval, symbol)
        return f"{scheme.value}:{path}"

    file_name = f"{symbol}{extension_map[source]}"

    url = f"{scheme.value}:{source.value}/{instrument_type}/{interval}/{file_name}"
    return url


def make_instrument_url(
    scheme: Scheme,
    source: Source,
    instrument: BaseInstrumentConfig,
) -> str:
    """Generate a URL to a data file."""
    return make_url(
        scheme,
        source,
        instrument.type,
        instrument.interval,
        instrument.symbol,
    )
