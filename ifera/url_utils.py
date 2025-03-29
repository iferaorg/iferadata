from .file_utils import make_path
from .config import BaseInstrumentConfig
from .settings import settings
from typing import Optional


def make_url(
    scheme: str,
    source: str,
    instrument_type: str,
    interval: str,
    symbol: str,
    zipfile: bool = True,
) -> str:
    """Generate a URL to a data file."""
    if scheme == "file":
        path = make_path(source, instrument_type, interval, symbol, zipfile=zipfile)
        return f"{scheme}:{path}"

    file_name = f"{symbol}.zip" if zipfile else f"{symbol}.csv"
    netloc = settings.S3_BUCKET if scheme == "//s3/" else ""
    url = f"{scheme}:{netloc}{source}/{instrument_type}/{interval}/{file_name}"
    return url


def make_instrument_url(
    scheme: str,
    source: str,
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
