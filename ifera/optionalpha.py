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
    df["start_time"] = df["start_time"].apply(_parse_time) # type: ignore
    df["end_time"] = df["end_time"].apply(_parse_time)  # type: ignore

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


def parse_filter_log(html_string: str) -> pd.DataFrame:
    """
    Parse an Option Alpha filter log HTML and return a pandas DataFrame.

    This function extracts filter information from an HTML grid containing
    filter log data. The grid includes columns for date, filter type, and
    an optional description.

    Parameters
    ----------
    html_string : str
        HTML string containing the filter log grid data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - date: datetime, parsed date
        - filter_type: str, the type of filter applied
        - description: str, optional description (empty string if not present)

        The DataFrame will not contain any NaN values.

    Raises
    ------
    ValueError
        If the HTML string is empty or cannot be parsed.

    Examples
    --------
    >>> html = '<div class="rows"><div class="flex">...</div></div>'
    >>> df = parse_filter_log(html)
    >>> df.columns
    Index(['date', 'filter_type', 'description'])
    """
    if not html_string or not html_string.strip():
        raise ValueError("HTML string cannot be empty")

    soup = BeautifulSoup(html_string, "html.parser")

    # Find all filter entries (div elements with class="flex")
    entries = soup.find_all("div", class_="flex")

    if not entries:
        raise ValueError("No filter entries found in HTML")

    data = []

    for entry in entries:
        try:
            # Extract the date from the first child div with width:12rem
            date_div = entry.find("div", style=lambda s: bool(s and "width:12rem" in s))
            if not date_div:
                continue

            date_str = date_div.get_text(strip=True)

            # Extract the filter type and description from the second div with flex:1
            content_div = entry.find("div", style=lambda s: bool(s and "flex:1" in s))
            if not content_div:
                continue

            # The content div might have a desc tag for the description
            desc_tag = content_div.find("desc")
            if desc_tag:
                # Filter type is the text before the desc tag
                filter_type = (
                    content_div.get_text(strip=True)
                    .replace(desc_tag.get_text(strip=True), "")
                    .strip()
                )
                description = desc_tag.get_text(strip=True)
            else:
                # No description, just the filter type
                filter_type = content_div.get_text(strip=True)
                description = ""

            data.append(
                {
                    "date": date_str,
                    "filter_type": filter_type,
                    "description": description,
                }
            )

        except (AttributeError, KeyError, IndexError):
            # Skip entries that fail to parse
            continue

    if not data:
        raise ValueError("No valid filter data could be extracted from HTML")

    # Create DataFrame and ensure no NaN values
    df = pd.DataFrame(data)

    # Fill any potential NaN values with appropriate defaults for string columns
    df["filter_type"] = df["filter_type"].fillna("")
    df["description"] = df["description"].fillna("")

    # Convert date column to datetime, filtering out any invalid dates
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

    # Drop rows where date conversion failed (resulting in NaT)
    # This ensures the guarantee of no NaN/NaT values
    initial_count = len(df)
    df = df.dropna(subset=["date"])
    if len(df) < initial_count:
        # Some dates failed to parse, but we continue with valid entries
        pass

    if len(df) == 0:
        raise ValueError("No valid filter data with parseable dates could be extracted")

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

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


FILTERS_FOLDER = 'data/results/option_alpha/filters/'


def parse_simple_filter(file_name: str) -> pd.DataFrame | None:
    try:
        with open(file_name, 'r') as f:
            html = f.read()

        df = parse_filter_log(html)
        df['filter'] = 1

    except FileNotFoundError:
        return None

    return df[['date', 'filter']]


def parse_simple_indicator(file_name: str) -> pd.DataFrame | None:
    try:
        with open(file_name, 'r') as f:
            html = f.read()

        df = parse_filter_log(html)
        df['indicator'] = df['description'].astype(float)

    except FileNotFoundError:
        return None

    return df[['date', 'indicator']]

def parse_range_with(prefix: str) -> pd.DataFrame | None:
    file_name = f'{FILTERS_FOLDER}{prefix}-RANGE_WIDTH.txt'
    try:
        with open(file_name, 'r') as f:
            html = f.read()

        df = parse_filter_log(html)

    except FileNotFoundError:
        return None

    def _parse_range_width(description):
        match = re.search(r'Opening Range: ([\d,]+\.\d+) - ([\d,]+\.\d+)', description)
        if match:
            min_val = float(match.group(1).replace(',', ''))
            max_val = float(match.group(2).replace(',', ''))
            return (max_val - min_val) / max_val if max_val != 0 else 0
        else:
            return None

    df['range_width'] = df['description'].apply(_parse_range_width)
    
    return df[['date', 'range_width']]


def get_filters(prefix: str) -> pd.DataFrame:
    dfs = []

    range_df = parse_range_with(prefix)
    if range_df is not None:
        dfs.append(range_df)
        
    simple_filters = ['FIRST_BO', 'SKIP_CPI', 'SKIP_EOM', 'SKIP_EOQ', 'SKIP_FM', 'SKIP_FOMC', 'SKIP_FW', 'SKIP_ME', 'SKIP_PAY', 'SKIP_PCE', 'SKIP_PPI', 'SKIP_TW']
    for filter_name in simple_filters:
        filter_df = parse_simple_filter(f'{FILTERS_FOLDER}{prefix}-{filter_name}.txt')
        if filter_df is not None:
            filter_df = filter_df.rename(columns={'filter': f'{filter_name.lower()}'})
            dfs.append(filter_df)

    indicator_names = ['ADX_14']
    for indicator_name in indicator_names:
        indicator_df = parse_simple_indicator(f'{FILTERS_FOLDER}{prefix}-{indicator_name}.txt')
        if indicator_df is not None:
            indicator_df = indicator_df.rename(columns={'indicator': f'{indicator_name.lower()}'})
            dfs.append(indicator_df)

    if dfs:
        from functools import reduce
        df_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
        # Replace NaN with 0 for filter columns
        for col in df_merged.columns:
            if col != 'date':
                df_merged[col] = df_merged[col].fillna(0)
        return df_merged
    else:
        return pd.DataFrame()