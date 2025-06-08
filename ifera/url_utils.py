import re
import datetime as dt
from typing import Tuple, Optional
import time

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparse

from .enums import Scheme, Source, extension_map
from .file_utils import make_path


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


def _extract_date(label: str, soup: BeautifulSoup) -> Optional[dt.date]:
    """
    Find the first element whose text *contains* ``label`` and grab the text in
    its next sibling node (works for <td>, <th>, <dt>, <div>, etc.).
    Falls back to a regex search through the whole document.
    """
    # 1️⃣ DOM-based search
    node = soup.find(string=lambda s: s and label in s) # type: ignore
    if node:
        sib = node.parent and node.parent.find_next_sibling()
        if sib and sib.get_text(strip=True):
            return _clean_date(sib.get_text())

    # 2️⃣ Regex fall-back (robust to layout changes)
    m = re.search(rf"{re.escape(label)}\s*([0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{2,4}})",
                  soup.get_text(" ", strip=True), flags=re.I)
    return _clean_date(m.group(1)) if m else None


def _clean_date(text: str) -> Optional[dt.date]:
    if not text:
        return None
    # Drop trailing notes such as "(expired)" or "(est)".
    date_str = text.split()[0]
    return dtparse.parse(date_str, dayfirst=False).date()


def contract_notice_and_expiry(symbol: str, max_retries: int = 3) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    """
    Parameters
    ----------
    symbol : str
        Futures contract symbol as it appears in the Barchart URL
        (e.g. ``"CLF11"`` for Crude Oil WTI Jan '11).
    max_retries : int, optional
        Maximum number of retries for fetching the URL (default is 3).

    Returns
    -------
    first_notice : datetime.date or None
    expiration    : datetime.date or None
    """
    url = f"https://www.barchart.com/futures/quotes/{symbol}"
    retries = 0
    resp = None

    while retries < max_retries:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                break
        except requests.RequestException as e:
            if retries == max_retries - 1:
                raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries.") from e
        except Exception as e:
            if retries == max_retries - 1:
                print(f"Unexpected error fetching {url}: {e}")
                raise
        retries += 1
        if retries < max_retries:
            time.sleep(1)  # Wait 1 second before retrying
    else:
        if resp is None:
            raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries.")
        if resp.status_code != 404:
            print(f"Failed to fetch {url}: HTTP {resp.status_code}")
        return None, None
        

    soup = BeautifulSoup(resp.text, "html.parser")

    first_notice = _extract_date("First Notice Date", soup)
    expiration   = _extract_date("Expiration Date", soup)
    return first_notice, expiration
