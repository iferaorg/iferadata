"""
Module for parsing Option Alpha trade log HTML grids.

This module provides functionality to parse HTML-formatted trade log grids
from Option Alpha and convert them into pandas DataFrames.
"""

import re

import pandas as pd
from bs4 import BeautifulSoup


def parse_trade_log(html_string: str) -> pd.DataFrame:
    """
    Parse an Option Alpha trade log HTML grid and return a pandas DataFrame.

    This function extracts trade information from an HTML grid containing
    trade log data. The grid includes columns for description (symbol and
    trade type), date, time range, status, risk, and P/L.

    Parameters
    ----------
    html_string : str
        HTML string containing the trade log grid data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - description: str, e.g., "SPX Long Call"
        - date: str, e.g., "Jan 10, 2022"
        - start_time: str, e.g., "3:47pm"
        - end_time: str, e.g., "4:00pm"
        - status: str, e.g., "Expired"
        - risk: float, dollar amount of risk (0 if missing)
        - pl: float, profit/loss in dollars (0 if missing)

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
    Index(['description', 'date', 'start_time', 'end_time', 'status', 'risk', 'pl'])
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
                description = f"{symbol} {trade_type}"
            else:
                # Fallback if structure is different
                description = symbol_cell.get_text(strip=True)

            # Extract date and time from closeTime cell
            close_time_cell = row.find("div", class_="closeTime")
            if not close_time_cell:
                continue

            time_divs = close_time_cell.find_all("div", class_="clip")
            if len(time_divs) >= 2:
                date = time_divs[0].get_text(strip=True)
                time_range = time_divs[1].get_text(strip=True)

                # Parse time range (e.g., "3:47pm → 4:00pm")
                time_parts = time_range.split("→")
                if len(time_parts) == 2:
                    start_time = time_parts[0].strip()
                    end_time = time_parts[1].strip()
                else:
                    start_time = ""
                    end_time = ""
            else:
                date = ""
                start_time = ""
                end_time = ""

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
            pl = _extract_dollar_amount(pnl_cell)

            data.append(
                {
                    "description": description,
                    "date": date,
                    "start_time": start_time,
                    "end_time": end_time,
                    "status": status,
                    "risk": risk,
                    "pl": pl,
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
    df["description"] = df["description"].fillna("")
    df["date"] = df["date"].fillna("")
    df["start_time"] = df["start_time"].fillna("")
    df["end_time"] = df["end_time"].fillna("")
    df["status"] = df["status"].fillna("")
    df["risk"] = df["risk"].fillna(0)
    df["pl"] = df["pl"].fillna(0)

    return df


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
