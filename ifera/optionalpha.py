"""
Module for parsing Option Alpha trade log HTML grids.

This module provides functionality to parse HTML-formatted trade log grids
from Option Alpha and convert them into pandas DataFrames.
"""

import re
from datetime import datetime, time
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup


def parse_trade_log(html_string: str) -> pd.DataFrame:
    """
    Parse an Option Alpha trade log HTML grid and return a pandas DataFrame.

    This function extracts trade information from an HTML grid containing
    trade log data. The grid includes columns for symbol, trade type,
    date, time range, status, risk, and profit/loss.

    Parameters
    ----------
    html_string : str
        HTML string containing the trade log grid data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - symbol: str, e.g., "SPX"
        - trade_type: str, e.g., "Long Call"
        - date: datetime, parsed date
        - start_time: datetime.time, e.g., time(15, 47)
        - end_time: datetime.time, e.g., time(16, 0)
        - status: str, e.g., "Expired"
        - risk: float, dollar amount of risk (0 if missing)
        - profit: float, profit/loss in dollars (0 if missing)

        The DataFrame will not contain any NaN values.

    Raises
    ------
    ValueError
        If the HTML string is empty or cannot be parsed.

    Examples
    --------
    >>> html = '<grid>...</grid>'
    >>> df = parse_trade_log(html)
    >>> df.columns
    Index(['symbol', 'trade_type', 'date', 'start_time', 'end_time', 'status', 'risk', 'profit'])
    """
    if not html_string or not html_string.strip():
        raise ValueError("HTML string cannot be empty")

    soup = BeautifulSoup(html_string, "html.parser")

    # Find all row elements
    rows = soup.find_all("row")

    if not rows:
        raise ValueError("No trade rows found in HTML")

    data = []

    for row in rows:
        try:
            # Extract symbol and type from the first cell (description)
            symbol_cell = row.find("div", class_="cell symbol")
            if not symbol_cell:
                continue

            symbol_spans = symbol_cell.find_all("span")
            if len(symbol_spans) >= 2:
                symbol = symbol_spans[0].get_text(strip=True)
                trade_type = symbol_spans[1].get_text(strip=True)
            else:
                # Fallback if structure is different
                text = symbol_cell.get_text(strip=True)
                parts = text.split(None, 1)
                if len(parts) == 2:
                    symbol = parts[0]
                    trade_type = parts[1]
                else:
                    symbol = text
                    trade_type = ""

            # Extract date and time from closeTime cell
            close_time_cell = row.find("div", class_="closeTime")
            if not close_time_cell:
                continue

            time_divs = close_time_cell.find_all("div", class_="clip")
            if len(time_divs) >= 2:
                date_str = time_divs[0].get_text(strip=True)
                time_range = time_divs[1].get_text(strip=True)

                # Parse time range (e.g., "3:47pm → 4:00pm")
                time_parts = time_range.split("→")
                if len(time_parts) == 2:
                    start_time_str = time_parts[0].strip()
                    end_time_str = time_parts[1].strip()
                else:
                    start_time_str = ""
                    end_time_str = ""
            else:
                date_str = ""
                start_time_str = ""
                end_time_str = ""

            # Extract status
            status_cell = row.find("div", class_="status")
            if status_cell:
                status_span = status_cell.find("span", class_="lbl")
                status = status_span.get_text(strip=True) if status_span else ""
            else:
                status = ""

            # Extract risk (dollar amount)
            risk_cell = row.find("div", class_="risk")
            risk = _extract_dollar_amount(risk_cell)

            # Extract P/L (dollar amount)
            pnl_cell = row.find("div", class_="pnl")
            profit = _extract_dollar_amount(pnl_cell)

            data.append(
                {
                    "symbol": symbol,
                    "trade_type": trade_type,
                    "date": date_str,
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "status": status,
                    "risk": risk,
                    "profit": profit,
                }
            )

        except (AttributeError, KeyError, IndexError):
            # Skip rows that fail to parse
            continue

    if not data:
        raise ValueError("No valid trade data could be extracted from HTML")

    # Create DataFrame and ensure no NaN values
    df = pd.DataFrame(data)

    # Fill any potential NaN values with appropriate defaults
    df["symbol"] = df["symbol"].fillna("")
    df["trade_type"] = df["trade_type"].fillna("")
    df["date"] = df["date"].fillna("")
    df["start_time"] = df["start_time"].fillna("")
    df["end_time"] = df["end_time"].fillna("")
    df["status"] = df["status"].fillna("")
    df["risk"] = df["risk"].fillna(0)
    df["profit"] = df["profit"].fillna(0)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

    # Convert time columns to datetime.time objects
    df["start_time"] = df["start_time"].apply(_parse_time)
    df["end_time"] = df["end_time"].apply(_parse_time)

    return df


def _parse_time(time_str: str) -> Optional[time]:
    """
    Parse a time string in 12-hour format to a datetime.time object.

    Parameters
    ----------
    time_str : str
        Time string in format like "3:47pm" or "4:00pm"

    Returns
    -------
    datetime.time or None
        Parsed time object, or None if parsing fails

    Examples
    --------
    >>> _parse_time("3:47pm")
    datetime.time(15, 47)
    >>> _parse_time("4:00pm")
    datetime.time(16, 0)
    >>> _parse_time("")
    None
    """
    if not time_str or time_str.strip() == "":
        return None

    try:
        # Parse time in 12-hour format (e.g., "3:47pm")
        dt = datetime.strptime(time_str.strip(), "%I:%M%p")
        return dt.time()
    except ValueError:
        return None


def _extract_dollar_amount(cell) -> float:
    """
    Extract dollar amount from a cell element.

    Parameters
    ----------
    cell : bs4.element.Tag or None
        BeautifulSoup tag element containing the dollar amount.

    Returns
    -------
    float
        The extracted dollar amount, or 0 if not found or cannot be parsed.

    Examples
    --------
    >>> from bs4 import BeautifulSoup
    >>> html = '<div><span class="val">$1,234</span></div>'
    >>> cell = BeautifulSoup(html, 'html.parser').find('div')
    >>> _extract_dollar_amount(cell)
    1234.0
    """
    if cell is None:
        return 0.0

    val_span = cell.find("span", class_="val")
    if not val_span:
        return 0.0

    text = val_span.get_text(strip=True)
    if not text:
        return 0.0

    # Remove dollar sign and commas, handle negative values
    # Pattern: optional minus, dollar sign, digits with optional commas
    pattern = r"^(-?)\$?([\d,]+(?:\.\d+)?)$"
    match = re.match(pattern, text)

    if match:
        sign = match.group(1)
        number_str = match.group(2).replace(",", "")
        value = float(number_str)
        return -value if sign == "-" else value

    return 0.0
