"""
Module for parsing Option Alpha trade log HTML grids.

This module provides functionality to parse HTML-formatted trade log grids
from Option Alpha and convert them into pandas DataFrames.
"""

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, time
from typing import Iterable, NamedTuple, Optional
from functools import reduce

import pandas as pd
import torch
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from tqdm.rich import tqdm


@dataclass
class SplitTensorState:
    """
    Tensor-based representation of split state for efficient batch operations.

    This class stores split information in tensor format to minimize context
    switching between tensors and Python objects during split generation.

    Attributes
    ----------
    masks : torch.Tensor
        2D boolean tensor of shape (n_splits, n_samples) containing split masks
    split_objects : list[Split]
        Corresponding Split objects (created lazily only when needed)
    """

    masks: torch.Tensor
    split_objects: list["Split"]


class FilterInfo(NamedTuple):
    """
    Information about a filter that produces a split mask.

    Attributes
    ----------
    filter_idx : int
        Column index in the filters DataFrame
    filter_name : str
        Name of the filter column
    threshold : float
        Threshold value for the split
    direction : str
        Direction of the split: "left" (<= threshold) or "right" (>= threshold)
    """

    filter_idx: int
    filter_name: str
    threshold: float
    direction: str


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
        A DataFrame with a DatetimeIndex (named 'date') and columns:
        - symbol: str, e.g., "SPX"
        - trade_type: str, e.g., "Long Call"
        - start_time: datetime.time, e.g., time(15, 47)
        - end_time: datetime.time, e.g., time(16, 0)
        - status: str, e.g., "Expired"
        - risk: float, dollar amount of risk (0 if missing)
        - profit: float, profit/loss in dollars (0 if missing)

        The DataFrame will not contain any NaN values.
        The index is a pd.DatetimeIndex containing unique dates.

    Raises
    ------
    ValueError
        If the HTML string is empty or cannot be parsed, or if duplicate
        dates are found in the trade log.

    Examples
    --------
    >>> html = '<grid>...</grid>'
    >>> df = parse_trade_log(html)
    >>> df.columns
    Index(['symbol', 'trade_type', 'start_time', 'end_time', 'status', 'risk', 'profit'])
    >>> isinstance(df.index, pd.DatetimeIndex)
    True
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
    df["start_time"] = df["start_time"].apply(_parse_time)  # type: ignore
    df["end_time"] = df["end_time"].apply(_parse_time)  # type: ignore

    # Check for duplicate dates and raise error if found
    duplicates = df[df["date"].duplicated(keep=False)]
    if len(duplicates) > 0:
        unique_dup_dates = duplicates["date"].drop_duplicates().tolist()  # type: ignore
        raise ValueError(f"Duplicate dates found in trade log: " f"{unique_dup_dates}")

    # Set date as the index
    df = df.set_index("date")
    df.index.name = "date"

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


FILTERS_FOLDER = "data/results/option_alpha/filters/"


def _read_filter_file(file_name: str) -> pd.DataFrame | None:
    """
    Read and parse a filter log file.

    Parameters
    ----------
    file_name : str
        Path to the filter log file

    Returns
    -------
    pd.DataFrame | None
        Parsed filter log DataFrame, or None if file not found
    """
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            html = f.read()
        return parse_filter_log(html)
    except FileNotFoundError:
        return None
    except Exception as e:
        raise ValueError(f"Error reading filter file '{file_name}': {e}") from e


def _parse_description_with_regex(
    df: pd.DataFrame, column_name: str, regex_pattern: str, parse_func=None
) -> pd.DataFrame:
    """
    Parse description column using a regex pattern and create a new column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'description' column
    column_name : str
        Name of the new column to create
    regex_pattern : str
        Regex pattern to match in the description
    parse_func : callable, optional
        Function to parse the matched value. If None, extracts first group as float.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new column added and duplicates eliminated
    """

    def default_parse(description):
        match = re.search(regex_pattern, description)
        if match:
            val = float(match.group(1).replace(",", ""))
            return val
        return None

    parser = parse_func if parse_func is not None else default_parse
    df[column_name] = df["description"].apply(parser)
    df = _check_and_eliminate_duplicates(df, column_name)
    return df[["date", column_name]]  # type: ignore


def _check_and_eliminate_duplicates(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Check for duplicate dates and eliminate them if values are the same.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'date' column
    column_name : str
        Name of the value column to check for consistency across duplicates

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed

    Raises
    ------
    ValueError
        If duplicate dates have different values for the specified column
    """
    if "date" not in df.columns:
        return df

    duplicates = df[df["date"].duplicated(keep=False)]
    if len(duplicates) > 0:
        # Check if duplicate dates have the same values
        unique_dates = duplicates["date"].drop_duplicates().tolist()  # type: ignore
        for date in unique_dates:
            date_rows = df[df["date"] == date]
            # Check if all values for this date are the same
            if len(date_rows[column_name].unique()) > 1:  # type: ignore
                raise ValueError(
                    f"Duplicate date {date} found with different values for '{column_name}': "
                    f"{date_rows[column_name].tolist()}"
                )

        # Remove duplicates, keeping first occurrence
        df = df.drop_duplicates(subset=["date"], keep="first")

    return df


def parse_simple_filter(file_name: str) -> pd.DataFrame | None:
    df = _read_filter_file(file_name)
    if df is None:
        return None

    df["filter"] = 1
    df = _check_and_eliminate_duplicates(df, "filter")
    return df[["date", "filter"]]  # type: ignore


def parse_simple_indicator(file_name: str) -> pd.DataFrame | None:
    df = _read_filter_file(file_name)
    if df is None:
        return None

    # Fill empty descriptions with 0
    df["description"] = df["description"].replace("", "0")
    df["indicator"] = df["description"].astype(float)
    df = _check_and_eliminate_duplicates(df, "indicator")
    return df[["date", "indicator"]]  # type: ignore


def parse_moving_average(file_name: str) -> pd.DataFrame | None:
    df = _read_filter_file(file_name)
    if df is None:
        return None

    def _parse_moving_average(description):
        match = re.search(
            r"Price: \$(\-?[\d,]+\.\d+), [SE]MA: \$(\-?[\d,]+\.\d+)", description
        )
        if match:
            price = float(match.group(1).replace(",", ""))
            ma = float(match.group(2).replace(",", ""))
            return int(price > ma)
        return None

    df["moving_average"] = df["description"].apply(_parse_moving_average)
    df = _check_and_eliminate_duplicates(df, "moving_average")
    return df[["date", "moving_average"]]  # type: ignore


def parse_range_with(prefix: str) -> pd.DataFrame | None:
    file_name = f"{FILTERS_FOLDER}{prefix}-RANGE_WIDTH.txt"
    df = _read_filter_file(file_name)
    if df is None:
        return None

    def _parse_range_width(description):
        match = re.search(r"Opening Range: ([\d,]+\.\d+) - ([\d,]+\.\d+)", description)
        if match:
            min_val = float(match.group(1).replace(",", ""))
            max_val = float(match.group(2).replace(",", ""))
            return (max_val - min_val) / max_val if max_val != 0 else 0
        return None

    df["range_width"] = df["description"].apply(_parse_range_width)
    df = _check_and_eliminate_duplicates(df, "range_width")
    return df[["date", "range_width"]]  # type: ignore


def parse_gex(file_name: str) -> pd.DataFrame | None:
    df = _read_filter_file(file_name)
    if df is None:
        return None
    return _parse_description_with_regex(df, "gex", r":\s+(\-?[\d,]+(?:\.\d+)?)")


def parse_change_percent(prefix: str) -> pd.DataFrame | None:
    file_name = f"{FILTERS_FOLDER}{prefix}-CHANGE_PERCENT.txt"
    df = _read_filter_file(file_name)
    if df is None:
        return None
    return _parse_description_with_regex(
        df, "change_percent", r"Below min: (\-?[\d,]+\.\d+)"
    )


def parse_change_stdev(prefix: str) -> pd.DataFrame | None:
    file_name = f"{FILTERS_FOLDER}{prefix}-CHANGE_STDEV.txt"
    df = _read_filter_file(file_name)
    if df is None:
        return None
    return _parse_description_with_regex(
        df, "change_stdev", r"Change Std Devs: (\-?[\d,]+\.\d+)"
    )


def parse_gap(prefix: str) -> pd.DataFrame | None:
    file_name = f"{FILTERS_FOLDER}{prefix}-GAP.txt"
    df = _read_filter_file(file_name)
    if df is None:
        return None
    return _parse_description_with_regex(df, "gap", r"Gap: (\-?[\d,]+\.\d+)")


def parse_open_change(prefix: str) -> pd.DataFrame | None:
    file_name = f"{FILTERS_FOLDER}{prefix}-OPEN_CHANGE.txt"
    df = _read_filter_file(file_name)
    if df is None:
        return None
    return _parse_description_with_regex(
        df, "open_change", r"Open Chg %: (\-?[\d,]+\.\d+)"
    )


def parse_vixc(prefix: str) -> pd.DataFrame | None:
    file_name = f"{FILTERS_FOLDER}{prefix}-VIXC.txt"
    df = _read_filter_file(file_name)
    if df is None:
        return None
    return _parse_description_with_regex(df, "vixc", r"VIX Change: (\-?[\d,]+\.\d+)")


def get_filters(prefix: str) -> pd.DataFrame:
    """
    Get and merge filter data for a given prefix.

    This function reads various filter files (range width, skip filters, indicators)
    for a given prefix, eliminates duplicate dates from each filter, and merges them
    into a single DataFrame.

    Parameters
    ----------
    prefix : str
        The prefix for filter file names (e.g., "SPX-ORB-L")

    Returns
    -------
    pd.DataFrame
        A DataFrame with a DatetimeIndex (named 'date') and columns for each filter.
        If no filter files are found, returns an empty DataFrame with a DatetimeIndex.
        Missing values are filled with 0.

    Notes
    -----
    - Duplicate dates in individual filters are eliminated. If duplicates have different
      values, a ValueError is raised.
    - Filter files are loaded from FILTERS_FOLDER.
    - The function attempts to load: RANGE_WIDTH, FIRST_BO, SKIP_CPI, SKIP_EOM,
      SKIP_EOQ, SKIP_FM, SKIP_FOMC, SKIP_FW, SKIP_ME, SKIP_PAY, SKIP_PCE,
      SKIP_PPI, SKIP_TW, and ADX_14.
    """
    dfs = []

    # Simple filters
    simple_filter_names = [
        "FIRST_BO",
        "SKIP_CPI",
        "SKIP_EOM",
        "SKIP_EOQ",
        "SKIP_FM",
        "SKIP_FOMC",
        "SKIP_FW",
        "SKIP_ME",
        "SKIP_PAY",
        "SKIP_PCE",
        "SKIP_PPI",
        "SKIP_TW",
    ]
    for filter_name in simple_filter_names:
        filter_df = parse_simple_filter(f"{FILTERS_FOLDER}{prefix}-{filter_name}.txt")
        if filter_df is not None:
            filter_df = filter_df.rename(columns={"filter": f"{filter_name.lower()}"})
            dfs.append(filter_df)

    # Range width
    range_df = parse_range_with(prefix)
    if range_df is not None:
        dfs.append(range_df)

    # Metric filters with prefix parameter
    metric_filters = [
        ("change_percent", parse_change_percent),
        ("change_stdev", parse_change_stdev),
        ("gap", parse_gap),
        ("open_change", parse_open_change),
        ("vixc", parse_vixc),
    ]
    for _column_name, parse_func in metric_filters:
        metric_df = parse_func(prefix)
        if metric_df is not None:
            dfs.append(metric_df)

    # Indicators
    indicator_names = [
        "ADX_14",
        "CCI_20",
        "CMO_9",
        "IVR",
        "MACD_12_26_9",
        "MOMENTUM_10",
        "RSI_14",
        "STOCH_14_3_3",
        "STOCH_RSI_14_14_3_3",
        "VIX",
    ]
    for indicator_name in indicator_names:
        indicator_df = parse_simple_indicator(
            f"{FILTERS_FOLDER}{prefix}-{indicator_name}.txt"
        )
        if indicator_df is not None:
            indicator_df = indicator_df.rename(
                columns={"indicator": f"{indicator_name.lower()}"}
            )
            dfs.append(indicator_df)

    # Moving averages
    moving_average_names = [
        "SMA_10",
        "SMA_20",
        "SMA_30",
        "SMA_50",
        "SMA_100",
        "SMA_200",
        "EMA_10",
        "EMA_20",
        "EMA_30",
        "EMA_50",
        "EMA_100",
        "EMA_200",
    ]
    for ma in moving_average_names:
        ma_df = parse_moving_average(f"{FILTERS_FOLDER}{prefix}-{ma}.txt")
        if ma_df is not None:
            ma_df = ma_df.rename(columns={"moving_average": f"{ma.lower()}"})
            dfs.append(ma_df)

    file_names = []
    for file in os.listdir(FILTERS_FOLDER):
        if file.startswith(f"{prefix}-GEX-") and file.endswith(".txt"):
            file_names.append(os.path.join(FILTERS_FOLDER, file))

    for file_name in file_names:
        gex_df = parse_gex(file_name)
        if gex_df is not None:
            # Extract suffix from file name for column naming
            suffix = (
                file_name.split(f"{prefix}-GEX-")[-1]
                .replace(".txt", "")
                .replace("-", "_")
                .lower()
            )
            gex_df = gex_df.rename(columns={"gex": f"gex_{suffix}"})
            dfs.append(gex_df)

    if dfs:
        df_merged = reduce(
            lambda left, right: pd.merge(left, right, on="date", how="outer"), dfs
        )
        # Replace NaN with 0 for filter columns
        for col in df_merged.columns:
            if col != "date":
                df_merged[col] = df_merged[col].fillna(0)

        # Set date as the index
        df_merged = df_merged.set_index("date")
        df_merged.index.name = "date"

        return df_merged

    # Return empty DataFrame with DatetimeIndex
    return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))


class Split:
    """
    Represents a split condition for filtering data.

    A single split can now represent multiple filters that result in the same mask.
    Splits can be either depth 1 (original) or child splits created from parent combinations.

    Attributes
    ----------
    mask : torch.Tensor
        1-D bool tensor representing the row-mask of the split
    filters : list[FilterInfo]
        List of filters that result in this mask. Each FilterInfo contains:
        filter_idx, filter_name, threshold, and direction.
        Empty for child splits.
    parents : list[set[Split]]
        List of sets of parent Split objects. Each set represents the parent
        splits that were combined with logical AND to create this child split.
        Empty for depth 1 (original) splits.
    score : float | None
        Score assigned to the split by the score_func. Initialized to None.
    """

    score: float | None
    max_conjunctions_print = 3

    def __init__(
        self,
        mask: torch.Tensor,
        filters: list[FilterInfo],
        parents: list[set["Split"]],
    ):
        """
        Initialize a Split object.

        Parameters
        ----------
        mask : torch.Tensor
            1-D bool tensor representing the row-mask of the split
        filters : list[FilterInfo]
            List of filters that result in this mask
        parents : list[set[Split]]
            List of sets of parent Split objects
        """
        self.mask = mask
        self.filters = filters
        self.parents = parents
        self.score = None

    def __eq__(self, other):
        """
        Compare Split objects by object identity.

        Parameters
        ----------
        other : Split
            Another Split object to compare with

        Returns
        -------
        bool
            True if the objects are the same instance
        """
        return self is other

    def __hash__(self):
        """
        Return hash of the Split object based on object identity.

        Returns
        -------
        int
            Hash value based on object id
        """
        return id(self)

    def _get_dnf_terms(self) -> list[list[tuple[int, str, str, float]]]:
        """
        Get the disjunctive normal form (DNF) representation of this split.

        Returns a list of conjunctions, where each conjunction is a list of
        (filter_idx, filter_name, operator, threshold) tuples.

        Returns
        -------
        list[list[tuple[int, str, str, float]]]
            List of conjunctions in DNF
        """
        # Base case: depth 1 split with filters
        if len(self.filters) > 0:
            # Multiple filters in the same split represent OR relationship
            result = []
            for filter_info in self.filters:
                operator = "<=" if filter_info.direction == "left" else ">="
                result.append(
                    [
                        (
                            filter_info.filter_idx,
                            filter_info.filter_name,
                            operator,
                            filter_info.threshold,
                        )
                    ]
                )
            return result

        # Recursive case: child split with parents
        if len(self.parents) > 0:
            all_conjunctions = []
            # Multiple parent sets represent OR relationship
            for parent_set in self.parents:
                # Get DNF from each parent in the set
                parent_dnfs = [
                    parent._get_dnf_terms()  # pylint: disable=protected-access
                    for parent in sorted(parent_set, key=id)
                ]

                # Combine all parent DNFs with AND using the distributive property
                # Start with the first parent's DNF terms
                combined_terms = parent_dnfs[0]

                # Iteratively combine with each subsequent parent's DNF
                for parent_dnf in parent_dnfs[1:]:
                    new_combined = []
                    for existing_term in combined_terms:
                        for new_term in parent_dnf:
                            # Combine the two terms (both are lists of filter conditions)
                            merged_term = existing_term + new_term
                            new_combined.append(merged_term)
                    combined_terms = new_combined

                # Add all combined terms from this parent list to the result
                all_conjunctions.extend(combined_terms)

            return all_conjunctions

        # Empty split (shouldn't happen, but handle gracefully)
        return []

    def _merge_conjunction_terms(
        self, conjunction: list[tuple[int, str, str, float]]
    ) -> list[tuple[int, str, str, float]]:
        """
        Merge terms with the same filter and direction within a conjunction.

        For left direction (<=), keeps the lower threshold.
        For right direction (>=), keeps the higher threshold.

        Parameters
        ----------
        conjunction : list[tuple[int, str, str, float]]
            List of (filter_idx, filter_name, operator, threshold) tuples

        Returns
        -------
        list[tuple[int, str, str, float]]
            Merged list of filter terms
        """
        if len(conjunction) <= 1:
            return conjunction

        # Group terms by (filter_idx, filter_name, operator)
        # Key: (filter_idx, filter_name, operator), Value: list of thresholds
        groups: dict[tuple[int, str, str], list[float]] = {}
        for filter_idx, filter_name, operator, threshold in conjunction:
            key = (filter_idx, filter_name, operator)
            if key not in groups:
                groups[key] = []
            groups[key].append(threshold)

        # Merge terms: for each group, select the appropriate threshold
        merged = []
        for (filter_idx, filter_name, operator), thresholds in groups.items():
            if operator == "<=":
                # For left direction, use the minimum (most restrictive for <=)
                selected_threshold = min(thresholds)
            else:  # operator == ">="
                # For right direction, use the maximum (most restrictive for >=)
                selected_threshold = max(thresholds)
            merged.append((filter_idx, filter_name, operator, selected_threshold))

        return merged

    def __str__(self) -> str:
        """
        Return a string representation of the split in DNF.

        Returns
        -------
        str
            String representation with filters in disjunctive normal form
        """
        dnf_terms = self._get_dnf_terms()

        if not dnf_terms:
            header = "Split filters:"
        else:
            header = "Split filters:"

        # Add score and sample count to header if available
        sample_count = int(self.mask.sum().item())
        header_parts = [header]
        if self.score is not None:
            header_parts.append(f" (score: {self.score:.4f}, samples: {sample_count})")
        else:
            header_parts.append(f" (samples: {sample_count})")
        header = "".join(header_parts)

        if not dnf_terms:
            return f"{header}\n - (empty)"

        lines = [header]
        seen_conjunctions = set()

        for conjunction in dnf_terms:
            # Merge terms with the same filter and direction
            merged_conjunction = self._merge_conjunction_terms(conjunction)

            # Sort terms by filter_idx to create canonical ordering
            sorted_conjunction = sorted(merged_conjunction, key=lambda x: x[0])

            # Create a hashable representation for deduplication
            conjunction_tuple = tuple(sorted_conjunction)

            # Skip if we've already seen this conjunction
            if conjunction_tuple in seen_conjunctions:
                continue
            seen_conjunctions.add(conjunction_tuple)

            # Format each filter in the conjunction
            filter_strs = []
            for _, filter_name, operator, threshold in sorted_conjunction:
                filter_strs.append(f"({filter_name} {operator} {threshold:.4g})")

            # Join with " & " for compactness
            conjunction_str = " & ".join(filter_strs)
            lines.append(f" - {conjunction_str}")

            if len(seen_conjunctions) >= self.max_conjunctions_print:
                break

        if len(seen_conjunctions) > self.max_conjunctions_print:
            lines.append(
                f" - ... ({len(seen_conjunctions) - self.max_conjunctions_print} more)"
            )

        return "\n".join(lines)


def _deduplicate_parent_sets(parent_sets: list[set[Split]]) -> list[set[Split]]:
    """
    Remove duplicate parent sets while preserving order.

    Parameters
    ----------
    parent_sets : list[set[Split]]
        Parent sets that may contain duplicates

    Returns
    -------
    list[set[Split]]
        Unique parent sets
    """
    unique_sets: list[set[Split]] = []
    seen: set[frozenset[Split]] = set()

    for parent_set in parent_sets:
        frozen_set = frozenset(parent_set)
        if frozen_set in seen:
            continue
        seen.add(frozen_set)
        unique_sets.append(set(parent_set))

    return unique_sets


def _compute_child_parent_sets(parent_a: Split, parent_b: Split) -> list[set[Split]]:
    """
    Compute parent sets for a child split from two parent splits.

    Parameters
    ----------
    parent_a : Split
        First parent split
    parent_b : Split
        Second parent split

    Returns
    -------
    list[set[Split]]
        List of parent sets containing only depth 1 splits
    """

    def _get_depth_one_parent_sets(split: Split) -> list[set[Split]]:
        if len(split.parents) > 0:
            return split.parents
        return [{split}]

    parent_sets_a = _get_depth_one_parent_sets(parent_a)
    parent_sets_b = _get_depth_one_parent_sets(parent_b)

    combined_sets: list[set[Split]] = []
    for parent_set_a in parent_sets_a:
        for parent_set_b in parent_sets_b:
            combined_sets.append(parent_set_a | parent_set_b)

    return _deduplicate_parent_sets(combined_sets)


def _align_filters_with_trades(
    filters_df: pd.DataFrame, trades_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Align filters DataFrame with trades DataFrame.

    Remove rows from filters_df that don't exist in trades_df and verify that
    every row in trades_df has a matching row in filters_df.

    Parameters
    ----------
    filters_df : pd.DataFrame
        DataFrame with DatetimeIndex containing filter columns
    trades_df : pd.DataFrame
        DataFrame with DatetimeIndex containing trade data

    Returns
    -------
    pd.DataFrame
        Aligned filters DataFrame with the same index as trades_df

    Raises
    ------
    ValueError
        If there are dates in trades_df without corresponding dates in filters_df
    """
    # Remove rows from filters_df that don't exist in trades_df
    filters_df = filters_df.loc[filters_df.index.isin(trades_df.index)].copy()

    # Verify that every row in trades_df has a matching row in filters_df
    missing_dates = trades_df.index.difference(filters_df.index)
    if len(missing_dates) > 0:
        raise ValueError(
            f"Trades dataframe contains dates not found in filters dataframe: "
            f"{missing_dates.tolist()}"
        )

    # Reindex filters to match trades exactly
    return filters_df.reindex(trades_df.index)


def _add_computed_columns(
    filters_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    spread_width: int,
    left_only_filters: list[str],
    right_only_filters: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Add computed columns to filters DataFrame.

    Adds reward_per_risk, premium, weekday filters (is_monday, is_tuesday, etc.),
    and open_minutes columns.

    Parameters
    ----------
    filters_df : pd.DataFrame
        DataFrame with DatetimeIndex containing filter columns
    trades_df : pd.DataFrame
        DataFrame with DatetimeIndex containing trade data (risk, profit, start_time)
    spread_width : int
        Width of the option spread in points
    left_only_filters : list[str]
        List of filter column names that should only generate left splits

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Updated filters DataFrame and updated left_only_filters list
    """
    # Add reward_per_risk column
    filters_df["reward_per_risk"] = (
        spread_width * 100 - trades_df["risk"]
    ) / trades_df["risk"]

    # Add premium column
    filters_df["premium"] = spread_width * 100 - trades_df["risk"]

    # Add weekday filters based on date
    weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    for i, day in enumerate(weekday_names):
        filters_df[f"is_{day}"] = (filters_df.index.dayofweek == i).astype(int)  # type: ignore

    # Add open_minutes based on start_time (hours*60+minutes)
    if "start_time" in trades_df.columns:
        filters_df["open_minutes"] = trades_df["start_time"].apply(
            lambda t: t.hour * 60 + t.minute if t is not None else 0
        )
    else:
        # Default to 0 if start_time column doesn't exist
        filters_df["open_minutes"] = 0

    # Append weekday filters to left_only_filters (they can only be excluded)
    # Create a new list to avoid mutating the input parameter
    weekday_filter_names = [f"is_{day}" for day in weekday_names]
    updated_left_only_filters = list(left_only_filters) + weekday_filter_names
    updated_right_only_filters = list(right_only_filters) + ["reward_per_risk"]

    return filters_df, updated_left_only_filters, updated_right_only_filters


def _select_split_indices(
    unique_vals: torch.Tensor,
    col_tensor: torch.Tensor,
    max_splits_per_filter: int | None,
    min_samples: int,
    direction: str,
) -> list[int]:
    """
    Select split indices for a filter column in a specific direction.

    If max_splits_per_filter is set and we have more possible splits than allowed,
    select splits that distribute samples evenly. Otherwise, use all possible splits.
    Only creates splits where the selected side has at least min_samples samples.

    Parameters
    ----------
    unique_vals : torch.Tensor
        Sorted unique values in the filter column
    col_tensor : torch.Tensor
        Tensor of all values in the filter column
    max_splits_per_filter : int | None
        Maximum number of splits per direction, or None for all splits
    min_samples : int
        Minimum number of samples required on the selected side of the split
    direction : str
        Direction of the split: "left" or "right"

    Returns
    -------
    list[int]
        List of indices where splits should be created
    """
    # Count samples for each unique value using vectorized operations
    _, sample_counts = torch.unique(col_tensor, return_counts=True)

    # Calculate cumulative counts
    cumsum = torch.cumsum(sample_counts, dim=0)
    total_samples = cumsum[-1].item()

    # Determine valid split indices based on min_samples and direction
    n_possible_splits = len(unique_vals) - 1
    valid_indices = []

    for i in range(n_possible_splits):
        # cumsum[i] is the count of samples <= unique_vals[i]
        # cumsum[-1] - cumsum[i] is the count of samples > unique_vals[i]
        left_count = cumsum[i].item()
        right_count = total_samples - left_count

        if direction == "left" and left_count >= min_samples:
            valid_indices.append(i)
        elif direction == "right" and right_count >= min_samples:
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return []

    if max_splits_per_filter is None or len(valid_indices) <= max_splits_per_filter:
        # Use all valid splits
        return valid_indices

    # Select splits to evenly distribute the valid sample range into buckets
    # We aim to create max_splits_per_filter+1 buckets of roughly equal size
    # in the range [min_samples, total_samples]
    start_count = min_samples
    end_count = total_samples

    # Target size per bucket
    range_size = end_count - start_count
    target_bucket_size = range_size / (max_splits_per_filter + 1)

    # Find the split indices that best achieve even distribution
    selected_indices = []
    for split_num in range(1, max_splits_per_filter + 1):
        target_cumsum = start_count + split_num * target_bucket_size

        # Only consider valid indices
        diffs = torch.full((n_possible_splits,), float("inf"))
        for idx in valid_indices:
            if direction == "left":
                # For left splits, compare cumulative count to target
                diffs[idx] = abs(cumsum[idx].item() - target_cumsum)
            else:
                # For right splits, compare remaining count to target
                # remaining_count = total_samples - cumsum[idx]
                # target_remaining = total_samples - target_cumsum
                remaining_count = total_samples - cumsum[idx].item()
                target_remaining = total_samples - target_cumsum
                diffs[idx] = abs(remaining_count - target_remaining)

        best_idx = int(torch.argmin(diffs).item())

        # Avoid duplicate indices and ensure we have valid candidates
        attempts = 0
        max_attempts = len(valid_indices)
        while (
            best_idx in selected_indices
            and diffs[best_idx] < float("inf")
            and attempts < max_attempts
        ):
            diffs[best_idx] = float("inf")
            best_idx = int(torch.argmin(diffs).item())
            attempts += 1

        if best_idx not in selected_indices and best_idx in valid_indices:
            selected_indices.append(best_idx)

    return sorted(selected_indices)


def _create_splits_for_filter(
    col_idx: int,
    col_name: str,
    filters_df: pd.DataFrame,
    left_only_filters: list[str],
    right_only_filters: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_splits_per_filter: int | None,
    min_samples: int,
    filter_granularities: dict[str, float],
) -> list[Split]:
    """
    Create splits for a single filter column.

    Parameters
    ----------
    col_idx : int
        Index of the column in the filters DataFrame
    col_name : str
        Name of the filter column
    filters_df : pd.DataFrame
        DataFrame containing filter data
    left_only_filters : list[str]
        Filter names that should only generate left splits
    right_only_filters : list[str]
        Filter names that should only generate right splits
    device : torch.device
        PyTorch device for tensors
    dtype : torch.dtype
        PyTorch dtype for tensors
    max_splits_per_filter : int | None
        Maximum number of splits per direction
    min_samples : int
        Minimum number of samples required on each side of the split
    filter_granularities : dict[str, float]
        Dictionary mapping filter names to granularity values

    Returns
    -------
    list[Split]
        List of Split objects for this filter
    """
    # Convert filter column to tensor for masking (original values)
    col_tensor_original = torch.tensor(
        filters_df[col_name].values, dtype=dtype, device=device
    )

    # Check if this filter has a granularity
    granularity = filter_granularities.get(col_name, 0.01)

    splits = []

    # Create left splits if not in right_only_filters
    if col_name not in right_only_filters:
        # Round values UP to nearest granularity multiple for left direction
        rounded_vals = [
            math.ceil(v / granularity) * granularity for v in filters_df[col_name]
        ]
        unique_vals_left = torch.tensor(
            sorted(set(rounded_vals)), dtype=dtype, device=device
        )
        col_tensor_left = torch.tensor(rounded_vals, dtype=dtype, device=device)

        if len(unique_vals_left) > 1:
            left_split_indices = _select_split_indices(
                unique_vals_left,
                col_tensor_left,
                max_splits_per_filter,
                min_samples,
                "left",
            )
            for i in left_split_indices:
                # Calculate threshold from unique values
                threshold_avg = (unique_vals_left[i] + unique_vals_left[i + 1]) / 2.0

                if granularity is not None:
                    # Round threshold DOWN (opposite of left rounding) to nearest granularity
                    threshold = (
                        math.floor(threshold_avg.item() / granularity) * granularity
                    )
                else:
                    threshold = threshold_avg.item()

                # Use original col_tensor for mask
                left_mask = col_tensor_original <= threshold
                splits.append(
                    Split(
                        mask=left_mask,
                        filters=[FilterInfo(col_idx, col_name, threshold, "left")],
                        parents=[],
                    )
                )

    # Create right splits if not in left_only_filters
    if col_name not in left_only_filters:
        # Round values DOWN to nearest granularity multiple for right direction
        rounded_vals = [
            math.floor(v / granularity) * granularity for v in filters_df[col_name]
        ]
        unique_vals_right = torch.tensor(
            sorted(set(rounded_vals)), dtype=dtype, device=device
        )
        col_tensor_right = torch.tensor(rounded_vals, dtype=dtype, device=device)

        if len(unique_vals_right) > 1:
            right_split_indices = _select_split_indices(
                unique_vals_right,
                col_tensor_right,
                max_splits_per_filter,
                min_samples,
                "right",
            )
            for i in right_split_indices:
                # Calculate threshold from unique values
                threshold_avg = (unique_vals_right[i] + unique_vals_right[i + 1]) / 2.0

                if granularity is not None:
                    # Round threshold UP (opposite of right rounding) to nearest granularity
                    threshold = (
                        math.ceil(threshold_avg.item() / granularity) * granularity
                    )
                else:
                    threshold = threshold_avg.item()

                # Use original col_tensor for mask
                right_mask = col_tensor_original >= threshold
                splits.append(
                    Split(
                        mask=right_mask,
                        filters=[FilterInfo(col_idx, col_name, threshold, "right")],
                        parents=[],
                    )
                )

    return splits


@torch.compile()
def _compute_mask_match_counts_for_row(
    mask_row: torch.Tensor,
    all_masks_float: torch.Tensor,
    all_masks_inv_float: torch.Tensor,
) -> torch.Tensor:
    """
    Compute match counts for a single mask against all masks.

    This is a helper function to be called row-by-row to avoid creating
    large n_splits x n_splits tensors.

    Parameters
    ----------
    mask_row : torch.Tensor
        1D float tensor representing a single mask (n_samples,)
    all_masks_float : torch.Tensor
        2D float tensor of all masks (n_splits, n_samples)
    all_masks_inv_float : torch.Tensor
        2D float tensor of inverted masks (n_splits, n_samples)

    Returns
    -------
    torch.Tensor
        1D tensor of match counts (n_splits,)
    """
    # Compute number of positions where masks match
    # match_count = sum of (both True) + sum of (both False)
    mask_row_2d = mask_row.unsqueeze(0)  # Shape: (1, n_samples)
    mask_row_inv = 1 - mask_row_2d

    match_counts = torch.matmul(mask_row_2d, all_masks_float.T) + torch.matmul(
        mask_row_inv, all_masks_inv_float.T
    )
    return match_counts.squeeze(0)


def _find_identical_mask_groups(masks: torch.Tensor) -> torch.Tensor:
    """
    Find groups of identical masks using row-by-row processing.

    Processes masks one at a time to avoid creating large n_splits x n_splits
    tensors that would consume too much memory with 100k+ splits.

    Parameters
    ----------
    masks : torch.Tensor
        2D boolean tensor of shape (n_splits, n_samples)

    Returns
    -------
    torch.Tensor
        1D integer tensor of shape (n_splits,) where each value is the index
        of the first occurrence of that mask pattern. Splits with the same
        value should be merged together.
    """
    n_splits = masks.shape[0]
    n_samples = masks.shape[1]

    # Convert to float for matrix operations
    masks_float = masks.float()
    masks_inv_float = 1 - masks_float

    # For each mask, find the index of its first occurrence
    # Process row-by-row to avoid O(n_splits²) memory usage
    group_ids = torch.zeros(n_splits, dtype=torch.long, device=masks.device)
    processed = torch.zeros(n_splits, dtype=torch.bool, device=masks.device)

    for i in range(n_splits):
        if processed[i]:
            # Already assigned to a group
            continue

        # Compute match counts for mask i against all masks
        match_counts_i = _compute_mask_match_counts_for_row(
            masks_float[i], masks_float, masks_inv_float
        )

        # Find all masks identical to mask i
        identical_indices = (match_counts_i == n_samples).nonzero(as_tuple=True)[0]

        # Assign all identical masks to the same group (using index i)
        for idx in identical_indices:
            group_ids[idx] = i
            processed[idx] = True

    return group_ids


def _merge_identical_splits(splits: list[Split]) -> list[Split]:
    """
    Merge splits with identical masks.

    Parameters
    ----------
    splits : list[Split]
        List of Split objects

    Returns
    -------
    list[Split]
        List of merged Split objects
    """
    if len(splits) == 0:
        return splits

    # Stack all masks into a 2D tensor (n_splits x n_samples)
    all_masks = torch.stack([split.mask for split in splits], dim=0)

    # Find groups of identical masks
    group_ids = _find_identical_mask_groups(all_masks)

    # Get unique group ids and build merged splits
    unique_groups = torch.unique(group_ids, sorted=True)
    merged_splits: list[Split] = []

    for group_id in unique_groups:
        group_id_int = int(group_id.item())
        # Find all splits in this group
        group_mask = group_ids == group_id
        group_indices = group_mask.nonzero(as_tuple=True)[0]

        # Merge filters and parents from all splits in the group
        merged_filters = []
        merged_parents: list[set[Split]] = []
        for idx in group_indices:
            idx_int = int(idx.item())
            merged_filters.extend(splits[idx_int].filters)
            merged_parents.extend(splits[idx_int].parents)

        # Create merged split using the mask from the first split in the group
        merged_split = Split(
            mask=splits[group_id_int].mask,
            filters=merged_filters,
            parents=_deduplicate_parent_sets(merged_parents),
        )
        # Preserve the score from the first split in the group
        merged_split.score = splits[group_id_int].score
        merged_splits.append(merged_split)

    return merged_splits


@torch.compile()
def _compute_exclusion_mask_tensor(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    min_samples: int,
) -> torch.Tensor:
    """
    Compute exclusion mask between two sets of masks (pure tensor operation).

    Parameters
    ----------
    masks_a : torch.Tensor
        2D boolean tensor of shape (n_set_a, n_samples)
    masks_b : torch.Tensor
        2D boolean tensor of shape (n_set_b, n_samples)
    min_samples : int
        Minimum number of samples required in the intersection

    Returns
    -------
    torch.Tensor
        2-D bool tensor (n_set_a x n_set_b) indicating mutual exclusion
    """
    masks_a_float = masks_a.float()
    masks_b_float = masks_b.float()

    # Compute intersection counts between the two sets
    intersection_counts = torch.matmul(masks_a_float, masks_b_float.T)

    # Rule 1: Insufficient intersection means mutually exclusive
    has_sufficient_intersection = intersection_counts >= min_samples
    rule1_mask = ~has_sufficient_intersection

    # Rule 2: Subset relationships
    mask_sums_a = masks_a_float.sum(dim=1)
    mask_sums_b = masks_b_float.sum(dim=1)
    a_subset_of_b = intersection_counts == mask_sums_a.unsqueeze(1)
    b_subset_of_a = intersection_counts == mask_sums_b.unsqueeze(0)
    rule2_mask = a_subset_of_b | b_subset_of_a

    # Combine rules
    exclusion_mask = rule1_mask | rule2_mask

    return exclusion_mask


def _calculate_exclusion_mask(
    parent_set_a: Iterable[Split],
    parent_set_b: Iterable[Split],
    device: torch.device,
    min_samples: int,
    masks_b_stacked: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculate exclusion mask between two separate parent sets.

    Determines which pairs of splits (one from each set) are mutually exclusive
    based on:
    1. Insufficient intersection (overlap has fewer than min_samples samples)
    2. Subset relationship (one split's rows are subset of another's)

    Parameters
    ----------
    parent_set_a : list[Split]
        First set of parent splits
    parent_set_b : list[Split]
        Second set of parent splits
    device : torch.device
        PyTorch device for tensors
    min_samples : int
        Minimum number of samples required in the intersection for a valid combination
    masks_b_stacked : torch.Tensor | None, optional
        Pre-stacked masks for parent_set_b to avoid redundant stacking.
        If None, masks will be stacked from parent_set_b.

    Returns
    -------
    torch.Tensor
        2-D bool tensor (n_set_a x n_set_b) indicating mutual exclusion
    """
    sets_are_same = parent_set_a is parent_set_b
    parent_list_a = list(parent_set_a)
    parent_list_b = parent_list_a if sets_are_same else list(parent_set_b)

    n_set_a = len(parent_list_a)
    n_set_b = len(parent_list_b)

    if n_set_a == 0 or n_set_b == 0:
        return torch.zeros((n_set_a, n_set_b), dtype=torch.bool, device=device)

    # Stack masks for set_a
    masks_a = torch.stack([split.mask for split in parent_list_a], dim=0)

    # Use pre-stacked masks for set_b if provided, otherwise stack them
    if masks_b_stacked is None:
        masks_b = torch.stack([split.mask for split in parent_list_b], dim=0)
    else:
        masks_b = masks_b_stacked

    # Use compiled tensor function
    return _compute_exclusion_mask_tensor(masks_a, masks_b, min_samples)


@torch.compile()
def _compute_child_masks_tensor(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    valid_pairs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute child masks for valid parent pairs (pure tensor operation).

    Parameters
    ----------
    masks_a : torch.Tensor
        2D boolean tensor of shape (n_set_a, n_samples)
    masks_b : torch.Tensor
        2D boolean tensor of shape (n_set_b, n_samples)
    valid_pairs : torch.Tensor
        2D tensor of shape (n_pairs, 2) containing indices of valid parent pairs

    Returns
    -------
    torch.Tensor
        2D boolean tensor of shape (n_pairs, n_samples) containing child masks
    """
    # Extract parent masks for each valid pair and compute AND
    parent_a_masks = masks_a[valid_pairs[:, 0]]
    parent_b_masks = masks_b[valid_pairs[:, 1]]
    child_masks = parent_a_masks & parent_b_masks

    return child_masks


def _generate_child_splits(
    parent_set_a: Iterable[Split],
    parent_set_b: Iterable[Split],
    exclusion_mask: torch.Tensor,
) -> list[Split]:
    """
    Generate child splits from pairs of non-exclusive parent splits from two sets.

    When the two parent sets are the same (depth 2), we only check the upper triangle
    (j > i) to avoid duplicates. When the sets are different (depth > 2), we check
    all combinations but track parent pairs to avoid generating duplicates when
    parent_set_a is a subset of parent_set_b.

    Parameters
    ----------
    parent_set_a : list[Split]
        First set of parent splits
    parent_set_b : list[Split]
        Second set of parent splits
    exclusion_mask : torch.Tensor
        2-D bool tensor (len(parent_set_a) x len(parent_set_b)) indicating which
        pairs are mutually exclusive

    Returns
    -------
    list[Split]
        List of newly created child Split objects
    """
    sets_are_same = parent_set_a is parent_set_b
    parent_list_a = list(parent_set_a)
    parent_list_b = parent_list_a if sets_are_same else list(parent_set_b)

    n_set_a = len(parent_list_a)
    n_set_b = len(parent_list_b)

    if n_set_a == 0 or n_set_b == 0:
        return []

    # Check if the two sets are the same (by object identity)
    # Find valid (non-exclusive) pairs
    if sets_are_same:
        # Same parent sets: only check upper triangle to avoid duplicates
        # Create indices for upper triangle
        i_indices, j_indices = torch.triu_indices(
            n_set_a, n_set_b, offset=1, device=exclusion_mask.device
        )
        # Filter by exclusion mask
        upper_triangle_mask = ~exclusion_mask[i_indices, j_indices]
        valid_i = i_indices[upper_triangle_mask]
        valid_j = j_indices[upper_triangle_mask]
    else:
        # Different parent sets: check all combinations
        # Find all valid pairs
        valid_pairs_mask = ~exclusion_mask
        valid_i, valid_j = valid_pairs_mask.nonzero(as_tuple=True)

        # Track parent pairs to avoid duplicates
        # Use frozenset of Split objects for deduplication
        unique_pairs = {}
        filtered_i = []
        filtered_j = []

        for idx in range(len(valid_i)):
            i = int(valid_i[idx].item())
            j = int(valid_j[idx].item())
            parent_a = parent_list_a[i]
            parent_b = parent_list_b[j]

            # Create a pair identifier using frozenset
            pair = frozenset([id(parent_a), id(parent_b)])

            # Skip if this pair was already used
            if pair in unique_pairs:
                continue

            unique_pairs[pair] = True
            filtered_i.append(i)
            filtered_j.append(j)

        # Convert back to tensors
        if len(filtered_i) > 0:
            valid_i = torch.tensor(
                filtered_i, dtype=torch.long, device=exclusion_mask.device
            )
            valid_j = torch.tensor(
                filtered_j, dtype=torch.long, device=exclusion_mask.device
            )
        else:
            valid_i = torch.tensor([], dtype=torch.long, device=exclusion_mask.device)
            valid_j = torch.tensor([], dtype=torch.long, device=exclusion_mask.device)

    # If no valid pairs, return empty list
    if len(valid_i) == 0:
        return []

    # Stack parent masks
    masks_a = torch.stack([split.mask for split in parent_list_a], dim=0)
    masks_b = torch.stack([split.mask for split in parent_list_b], dim=0)

    # Create tensor of valid pairs
    valid_pairs = torch.stack([valid_i, valid_j], dim=1)

    # Compute child masks using compiled function
    child_masks = _compute_child_masks_tensor(masks_a, masks_b, valid_pairs)

    # Create Split objects
    child_splits = []
    for idx in range(len(valid_i)):
        i = int(valid_i[idx].item())
        j = int(valid_j[idx].item())
        child_splits.append(
            Split(
                mask=child_masks[idx],
                filters=[],
                parents=_compute_child_parent_sets(parent_list_a[i], parent_list_b[j]),
            )
        )

    return child_splits


@torch.compile()
def _find_redundant_mask_indices(
    new_masks: torch.Tensor,
    old_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Find indices of new masks that are redundant (identical to old masks).

    Parameters
    ----------
    new_masks : torch.Tensor
        2D boolean tensor of shape (n_new, n_samples)
    old_masks : torch.Tensor
        2D boolean tensor of shape (n_old, n_samples)

    Returns
    -------
    torch.Tensor
        1D boolean tensor of shape (n_new,) where True means the mask should be kept
    """
    # Convert to float for matrix operations
    new_masks_float = new_masks.float()
    old_masks_float = old_masks.float()

    n_samples = new_masks.shape[1]

    # Compute pairwise equality: new_masks[i] == old_masks[j]
    # Two masks are equal if they match at all positions
    match_counts = torch.matmul(new_masks_float, old_masks_float.T) + torch.matmul(
        1 - new_masks_float, (1 - old_masks_float).T
    )
    mask_equality = match_counts == n_samples

    # Find new splits that have no matching old split
    has_match = mask_equality.any(dim=1)
    keep_mask = ~has_match

    return keep_mask


def _remove_redundant_splits(
    new_splits: list[Split], old_splits: list[Split]
) -> list[Split]:
    """
    Remove new splits that have identical masks to old splits.

    Parameters
    ----------
    new_splits : list[Split]
        List of newly created Split objects
    old_splits : list[Split]
        List of existing Split objects

    Returns
    -------
    list[Split]
        List of new splits with redundant ones removed
    """
    if len(new_splits) == 0 or len(old_splits) == 0:
        return new_splits

    # Stack masks for vectorized comparison
    new_masks = torch.stack([split.mask for split in new_splits], dim=0)
    old_masks = torch.stack([split.mask for split in old_splits], dim=0)

    # Use compiled function to find which splits to keep
    keep_mask = _find_redundant_mask_indices(new_masks, old_masks)
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    # Return only new splits that don't match any old split
    return [new_splits[i] for i in keep_indices.tolist()]


@torch.compile()
def _compute_scores_tensor(
    y: torch.Tensor, masks: torch.Tensor, score_func
) -> torch.Tensor:
    """
    Compute scores for all masks using the score function (compilable wrapper).

    Parameters
    ----------
    y : torch.Tensor
        1-D tensor of target values (n_samples)
    masks : torch.Tensor
        2D boolean tensor of shape (n_splits, n_samples)
    score_func : callable
        Function that takes y and masks and returns scores

    Returns
    -------
    torch.Tensor
        1-D tensor of scores (n_splits,)
    """
    return score_func(y, masks)


def _score_splits(splits: list[Split], y: torch.Tensor, score_func) -> None:
    """
    Score splits using the provided score function.

    Updates the score attribute of each split in place.

    Parameters
    ----------
    splits : list[Split]
        List of Split objects to score
    y : torch.Tensor
        1-D tensor of target values (n_samples)
    score_func : callable
        Function that takes y (n_samples) and masks (batch_size, n_samples)
        and returns a float tensor of scores (batch_size)
    """
    if len(splits) == 0:
        return

    # Stack all masks into a 2D tensor (batch_size x n_samples)
    masks = torch.stack([split.mask for split in splits], dim=0)

    # Call score_func to get scores for all splits (use compiled version if possible)
    try:
        scores = _compute_scores_tensor(y, masks, score_func)
    except Exception:  # pylint: disable=broad-except
        # Fallback to non-compiled version if compilation fails
        scores = score_func(y, masks)

    # Assign scores to splits
    for i, split in enumerate(splits):
        split.score = float(scores[i].item())


@torch.compile()
def _select_top_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Select indices of top k scores (pure tensor operation).

    Parameters
    ----------
    scores : torch.Tensor
        1-D tensor of scores
    k : int
        Number of top scores to select

    Returns
    -------
    torch.Tensor
        1-D tensor of indices of top k scores
    """
    if len(scores) <= k:
        # Return all indices if we have fewer than k scores
        return torch.arange(len(scores), dtype=torch.long, device=scores.device)

    # Get top k indices
    _, top_indices = torch.topk(scores, k, largest=True, sorted=True)
    return top_indices


def _keep_top_n_splits(splits: list[Split], keep_best_n: int) -> list[Split]:
    """
    Keep only the top n splits based on their scores.

    Parameters
    ----------
    splits : list[Split]
        List of Split objects with scores assigned
    keep_best_n : int
        Number of top-scoring splits to keep

    Returns
    -------
    list[Split]
        List of the top n splits sorted by score (descending)
    """
    if len(splits) <= keep_best_n:
        return splits.copy()

    # Extract scores as tensor (treating None as -inf)
    scores_list = [s.score if s.score is not None else float("-inf") for s in splits]
    scores = torch.tensor(scores_list, dtype=torch.float32)

    # Use compiled function to get top k indices
    top_indices = _select_top_k_indices(scores, keep_best_n)

    # Return splits in top-k order
    return [splits[i] for i in top_indices.tolist()]


def _print_splits_for_depth(
    depth: int, splits: list[Split], score_func: Optional[object] = None
) -> None:
    """
    Print splits for a given depth using rich formatting.

    Parameters
    ----------
    depth : int
        The depth level being printed
    splits : list[Split]
        List of Split objects to print
    score_func : Optional[object], optional
        Score function (used to determine if splits should be sorted by score)
    """
    console = Console()

    # Sort splits by score if score_func is provided
    if score_func is not None:
        splits_to_print = sorted(
            splits,
            key=lambda s: s.score if s.score is not None else float("-inf"),
            reverse=True,
        )
    else:
        splits_to_print = splits

    # Create header
    console.print()
    console.rule(f"[bold blue]Depth {depth}[/bold blue]", style="blue")
    console.print()

    # Print each split
    for split in splits_to_print:
        split_str = str(split)
        console.print(Panel(split_str, border_style="cyan", padding=(0, 1)))

    console.print()


def _create_cv_indices(
    n_samples: int, n_folds: int, n_repeats: int, device: torch.device
) -> tuple[torch.Tensor, int, int, int]:
    """
    Create cross-validation indices for n-fold CV with k repeats.

    DEPRECATED: This function uses random k-fold CV which causes look-ahead bias
    in time-series data. Use _create_purged_ts_cv_splits instead for time-series
    data like trading strategies.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    n_folds : int
        Number of folds for cross-validation
    n_repeats : int
        Number of times to repeat the cross-validation
    device : torch.device
        PyTorch device for tensors

    Returns
    -------
    tuple[torch.Tensor, int, int, int]
        - randperm: Tensor of shape (n_repeats, n_samples_padded) with random indices
        - n_samples_padded: Padded number of samples divisible by n_folds
        - n_samples_train: Number of samples in training set
        - n_samples_valid: Number of samples in validation set
    """
    # Calculate n_samples_padded: smallest int >= n_samples divisible by n_folds
    n_samples_padded = ((n_samples + n_folds - 1) // n_folds) * n_folds

    # Calculate train and validation set sizes
    n_samples_valid = n_samples_padded // n_folds
    n_samples_train = n_samples_padded - n_samples_valid

    # Create k batches of random permutations
    # Generate permutations for the full range up to n_samples_padded
    randperm = torch.zeros(
        (n_repeats, n_samples_padded), dtype=torch.long, device=device
    )
    for k in range(n_repeats):
        # Create random permutation of the padded range
        perm = torch.randperm(n_samples_padded, device=device)
        randperm[k] = perm

    return randperm, n_samples_padded, n_samples_train, n_samples_valid


def _create_cv_splits(
    y: torch.Tensor,
    randperm: torch.Tensor,
    n_folds: int,
    n_samples_padded: int,
    n_samples_train: int,
    n_samples_valid: int,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create train/validation splits for cross-validation.

    DEPRECATED: This function uses random k-fold CV which causes look-ahead bias
    in time-series data. Use _create_purged_ts_cv_splits instead for time-series
    data like trading strategies.

    Parameters
    ----------
    y : torch.Tensor
        1-D tensor of target values (n_samples,)
    randperm : torch.Tensor
        Random permutation tensor of shape (n_repeats, n_samples_padded)
    n_folds : int
        Number of folds
    n_samples_padded : int
        Padded number of samples
    n_samples_train : int
        Number of samples in training set
    n_samples_valid : int
        Number of samples in validation set
    n_samples : int
        Original number of samples

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - y_train: Tensor of shape (n_repeats, n_folds, n_samples_train)
        - y_valid: Tensor of shape (n_repeats, n_folds, n_samples_valid)
    """
    n_repeats = randperm.shape[0]
    device = y.device
    dtype = y.dtype

    # Pad y with zeros if needed
    if n_samples_padded > n_samples:
        y_padded = torch.cat(
            [y, torch.zeros(n_samples_padded - n_samples, dtype=dtype, device=device)]
        )
    else:
        y_padded = y

    # Initialize output tensors
    y_train = torch.zeros(
        (n_repeats, n_folds, n_samples_train), dtype=dtype, device=device
    )
    y_valid = torch.zeros(
        (n_repeats, n_folds, n_samples_valid), dtype=dtype, device=device
    )

    # Create train/validation splits for each repeat and fold
    for k in range(n_repeats):
        perm_k = randperm[k]
        y_shuffled = y_padded[perm_k]

        for fold in range(n_folds):
            # Validation indices for this fold
            valid_start = fold * n_samples_valid
            valid_end = valid_start + n_samples_valid

            # Create validation set
            y_valid[k, fold] = y_shuffled[valid_start:valid_end]

            # Create training set (all except validation fold)
            if fold == 0:
                y_train[k, fold] = y_shuffled[valid_end:]
            elif fold == n_folds - 1:
                y_train[k, fold] = y_shuffled[:valid_start]
            else:
                y_train[k, fold] = torch.cat(
                    [y_shuffled[:valid_start], y_shuffled[valid_end:]]
                )

    return y_train, y_valid


def _create_purged_ts_cv_splits(
    n_samples: int,
    n_splits: int,
    n_repeats: int,
    purge_pct: float,
    embargo_pct: float,
    min_train_pct: float,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int]:
    """
    Create repeated purged time-series cross-validation splits.

    This function creates time-ordered splits with purging and embargo to prevent
    look-ahead bias in time-series data. Unlike random k-fold CV, this method:
    - Respects chronological order (train always before test)
    - Applies purging (removes observations right after training set)
    - Applies embargo (extra gap before test set starts)
    - Repeats with different random start offsets

    The method is based on Marcos López de Prado's Combinatorial Purged Cross-Validation
    and is widely used in quantitative finance to avoid overfitting.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    n_splits : int
        Number of splits (folds) per repeat
    n_repeats : int
        Number of times to repeat with different random offsets
    purge_pct : float
        Percentage of test set size to purge from the end of training set (0.0 to 1.0)
    embargo_pct : float
        Percentage of test set size for additional gap between train end and test start.
        This is in addition to the purge period. (0.0 to 1.0)
    min_train_pct : float
        Minimum training set size as percentage of total samples (0.0 to 1.0)
    device : torch.device
        PyTorch device for tensors

    Returns
    -------
    tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int]
        - splits: List of (train_indices, test_indices) tuples for all valid splits
        - max_train_size: Maximum training set size across all splits
        - max_test_size: Maximum test set size across all splits
    """
    test_size = n_samples // n_splits
    min_train_samples = int(n_samples * min_train_pct)
    purge_samples = max(1, int(test_size * purge_pct))
    embargo_samples = int(test_size * embargo_pct)

    # Total gap between training end and test start is purge + embargo
    total_gap = purge_samples + embargo_samples

    splits: list[tuple[torch.Tensor, torch.Tensor]] = []
    max_train_size = 0
    max_test_size = 0

    for _ in range(n_repeats):
        # Random start offset so repeats don't overlap exactly
        # Offset is within the first test_size positions
        if test_size > 1:
            offset = int(torch.randint(0, test_size, (1,), device=device).item())
        else:
            offset = 0

        for split_idx in range(n_splits):
            # Test set: fixed length, moves forward chronologically
            test_start = offset + split_idx * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                continue  # Skip incomplete final fold

            # Training set: everything before test, minus total gap (purge + embargo)
            train_end = max(0, test_start - total_gap)

            # Ensure minimum training period
            if train_end < min_train_samples:
                continue

            # Create index tensors
            train_indices = torch.arange(0, train_end, device=device)
            test_indices = torch.arange(test_start, test_end, device=device)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
                max_train_size = max(max_train_size, len(train_indices))
                max_test_size = max(max_test_size, len(test_indices))

    return splits, max_train_size, max_test_size


def _create_purged_ts_masks(
    n_samples: int,
    cv_splits: list[tuple[torch.Tensor, torch.Tensor]],
    max_train_size: int,
    max_test_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create boolean masks for purged time-series CV splits.

    This function creates fixed-size tensors for efficient vectorized operations
    by padding variable-length splits to the maximum size.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    cv_splits : list[tuple[torch.Tensor, torch.Tensor]]
        List of (train_indices, test_indices) tuples from _create_purged_ts_cv_splits
    max_train_size : int
        Maximum training set size across all splits
    max_test_size : int
        Maximum test set size across all splits
    device : torch.device
        PyTorch device for tensors

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        - train_masks: Boolean tensor of shape (n_buckets, n_samples) for training
        - test_masks: Boolean tensor of shape (n_buckets, n_samples) for testing
        - train_valid_masks: Boolean tensor of shape (n_buckets, max_train_size) indicating
          which positions in the padded train sets contain valid data
        - test_valid_masks: Boolean tensor of shape (n_buckets, max_test_size) indicating
          which positions in the padded test sets contain valid data
    """
    n_buckets = len(cv_splits)

    # Create full-size masks for indexing original data
    train_masks = torch.zeros((n_buckets, n_samples), dtype=torch.bool, device=device)
    test_masks = torch.zeros((n_buckets, n_samples), dtype=torch.bool, device=device)

    # Create validity masks for padded arrays
    train_valid_masks = torch.zeros(
        (n_buckets, max_train_size), dtype=torch.bool, device=device
    )
    test_valid_masks = torch.zeros(
        (n_buckets, max_test_size), dtype=torch.bool, device=device
    )

    for bucket_idx, (train_idx, test_idx) in enumerate(cv_splits):
        # Set masks for actual data positions
        train_masks[bucket_idx, train_idx] = True
        test_masks[bucket_idx, test_idx] = True

        # Set validity masks for padded arrays
        train_valid_masks[bucket_idx, : len(train_idx)] = True
        test_valid_masks[bucket_idx, : len(test_idx)] = True

    return train_masks, test_masks, train_valid_masks, test_valid_masks


def _evaluate_filters(
    depth_1_splits: list[Split],
    y: torch.Tensor,
    score_func,
    filter_eval_folds: int,
    filter_eval_repeats: int,
    verbose: str = "no",
    min_score_improvement: float | None = None,
    purge_pct: float = 0.1,
    embargo_pct: float = 0.0,
    min_train_pct: float = 0.2,
) -> dict[tuple[str, str], float]:
    """
    Evaluate filters using repeated purged time-series cross-validation.

    This function evaluates each filter+direction combination by computing score
    improvements via purged time-series CV. Unlike random k-fold CV, this method
    respects chronological order and applies purging to prevent look-ahead bias:
    - Training data is always before test data (no future leakage)
    - A purge period removes observations right after training to kill autocorrelation
    - Multiple repeats with different random offsets provide robust evaluation

    For each split, it finds the best threshold on the training set and evaluates
    it on the test set, comparing against a baseline (all-true mask).

    Parameters
    ----------
    depth_1_splits : list[Split]
        List of depth 1 Split objects to evaluate
    y : torch.Tensor
        1-D tensor of target values (n_samples,)
    score_func : callable
        Function that takes y and masks and returns scores
    filter_eval_folds : int
        Number of time-series splits (folds) per repeat
    filter_eval_repeats : int
        Number of times to repeat with different random offsets
    verbose : str, optional
        Controls printing. Default is "no". If not "no", prints the evaluation table.
    min_score_improvement : float | None, optional
        Threshold for filtering. Used to calculate summary statistics for filters
        above threshold. Default is None.
    purge_pct : float, optional
        Percentage of test set size to purge from end of training set (0.0 to 1.0).
        Default is 0.1 (10%). Prevents autocorrelation leakage.
    embargo_pct : float, optional
        Percentage of test set size for additional gap after test (0.0 to 1.0).
        Default is 0.0. Used for extra safety in high-frequency data.
    min_train_pct : float, optional
        Minimum training set size as percentage of total samples (0.0 to 1.0).
        Default is 0.2 (20%). Ensures sufficient training data.

    Returns
    -------
    dict[tuple[str, str], float]
        Dictionary mapping (filter_name, direction) to average score improvement
    """
    from rich.table import Table

    console = Console()
    n_samples = len(y)
    device = y.device

    # Step 1: Sort depth_1_splits into dict keyed by filter_id + direction
    filter_splits_dict: dict[tuple[str, str], list[Split]] = {}
    for split in depth_1_splits:
        if len(split.filters) > 0:
            # Get the first filter (depth 1 splits should have only one filter)
            filter_info = split.filters[0]
            key = (filter_info.filter_name, filter_info.direction)
            if key not in filter_splits_dict:
                filter_splits_dict[key] = []
            filter_splits_dict[key].append(split)

    if len(filter_splits_dict) == 0:
        return {}

    # Step 2: Create purged time-series CV splits
    cv_splits, _, _ = _create_purged_ts_cv_splits(
        n_samples,
        filter_eval_folds,
        filter_eval_repeats,
        purge_pct,
        embargo_pct,
        min_train_pct,
        device,
    )

    n_buckets = len(cv_splits)
    if n_buckets == 0:
        # Not enough data for CV splits with the given parameters
        return {}

    # Step 3: Build flat arrays for all splits and their filter group mappings
    all_splits_flat: list[Split] = []
    filter_keys: list[tuple[str, str]] = []
    filter_group_indices: list[int] = []  # Maps each split to its filter group

    for group_idx, (key, splits_for_key) in enumerate(filter_splits_dict.items()):
        filter_keys.append(key)
        for split in splits_for_key:
            all_splits_flat.append(split)
            filter_group_indices.append(group_idx)

    n_total_splits = len(all_splits_flat)
    n_filter_groups = len(filter_keys)

    if n_total_splits == 0:
        return {}

    # Step 4: Pre-compute all split masks stacked
    all_split_masks = torch.stack([s.mask for s in all_splits_flat], dim=0)
    # Shape: (n_total_splits, n_samples)

    # Step 5: Create filter group indices tensor for segmented operations
    filter_group_tensor = torch.tensor(
        filter_group_indices, dtype=torch.long, device=device
    )

    # Step 6: Pre-compute original mask sample counts for all splits
    original_sample_counts = torch.tensor(
        [s.mask.sum().item() for s in all_splits_flat], dtype=torch.long, device=device
    )

    # Step 7: Accumulators for results per filter group
    all_improvements_tensor = torch.zeros(
        (n_filter_groups, n_buckets), dtype=y.dtype, device=device
    )
    all_best_sample_counts_tensor = torch.zeros(
        (n_filter_groups, n_buckets), dtype=torch.long, device=device
    )

    # Step 8: Process all CV buckets
    # For each bucket: score splits on train, find best per group, score on test
    with tqdm(total=n_buckets, desc="Purged TS-CV evaluation") as pbar:
        for bucket_idx in range(n_buckets):
            train_idx, test_idx = cv_splits[bucket_idx]

            # Get y values for train and test sets
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Get split masks intersected with train/test indices
            # train_masks[bucket_idx] is (n_samples,) boolean mask for train indices
            # all_split_masks is (n_total_splits, n_samples)
            # We want (n_total_splits, len(train_idx))
            train_split_masks = all_split_masks[:, train_idx]
            test_split_masks = all_split_masks[:, test_idx]

            # Score all splits on training set at once
            train_scores = score_func(y_train, train_split_masks)
            # Shape: (n_total_splits,)

            # Find best split per filter group using segmented argmax
            group_max_scores = torch.full(
                (n_filter_groups,),
                float("-inf"),
                dtype=train_scores.dtype,
                device=device,
            )
            group_max_scores = group_max_scores.scatter_reduce(
                0, filter_group_tensor, train_scores, reduce="amax", include_self=False
            )

            # Find which split is the best in each group
            is_best_in_group = train_scores == group_max_scores[filter_group_tensor]

            # For ties, find the minimum index per group
            split_indices = torch.arange(n_total_splits, device=device)
            large_val = n_total_splits + 1
            masked_indices = torch.where(
                is_best_in_group,
                split_indices,
                torch.full_like(split_indices, large_val),
            )

            best_split_indices = torch.full(
                (n_filter_groups,), large_val, dtype=torch.long, device=device
            )
            best_split_indices = best_split_indices.scatter_reduce(
                0,
                filter_group_tensor,
                masked_indices,
                reduce="amin",
                include_self=False,
            )

            # Get test scores for best splits only
            best_test_masks = test_split_masks[best_split_indices]
            # Shape: (n_filter_groups, len(test_idx))

            test_scores = score_func(y_test, best_test_masks)
            # Shape: (n_filter_groups,)

            # Calculate base test score (all samples in test set)
            base_test_mask = torch.ones(
                (1, len(test_idx)), dtype=torch.bool, device=device
            )
            base_test_score = score_func(y_test, base_test_mask)[0]

            # Calculate improvements
            improvements = test_scores - base_test_score

            # Store results
            all_improvements_tensor[:, bucket_idx] = improvements
            all_best_sample_counts_tensor[:, bucket_idx] = original_sample_counts[
                best_split_indices
            ]

            pbar.update(1)

    # Step 9: Aggregate results per filter group
    score_improvements: dict[tuple[str, str], float] = {}
    best_split_sample_counts: dict[tuple[str, str], int] = {}

    avg_improvements = all_improvements_tensor.mean(dim=1)  # (n_filter_groups,)
    avg_sample_counts = all_best_sample_counts_tensor.float().mean(
        dim=1
    )  # (n_filter_groups,)

    for group_idx, key in enumerate(filter_keys):
        score_improvements[key] = float(avg_improvements[group_idx].item())
        best_split_sample_counts[key] = int(avg_sample_counts[group_idx].item() + 0.5)

    # Step 10: Print results using rich table (only if verbose is not "no")
    if len(score_improvements) > 0 and verbose != "no":
        # Sort by score improvement (descending)
        sorted_items = sorted(
            score_improvements.items(), key=lambda x: x[1], reverse=True
        )

        table = Table(title="Filter Evaluation Results (Purged Time-Series CV)")
        table.add_column("Filter", style="cyan")
        table.add_column("Direction", style="magenta")
        table.add_column("Avg Score Improvement", style="green", justify="right")
        table.add_column("Samples", style="yellow", justify="right")

        for (filter_name, direction), improvement in sorted_items:
            # Get average sample count from best splits across CV folds
            avg_samples = best_split_sample_counts.get((filter_name, direction), 0)
            table.add_row(
                filter_name, direction, f"{improvement:.6f}", str(avg_samples)
            )

        # Add summary statistics
        table.add_section()

        # Calculate summary statistics
        all_improvements = list(score_improvements.values())
        avg_improvement = sum(all_improvements) / len(all_improvements)

        # Calculate weighted average (weighted by number of splits per filter)
        total_splits = 0
        weighted_sum = 0.0
        for key, improvement in score_improvements.items():
            n_splits = len(filter_splits_dict[key])
            weighted_sum += improvement * n_splits
            total_splits += n_splits
        weighted_avg_improvement = (
            weighted_sum / total_splits if total_splits > 0 else 0.0
        )

        # Add summary rows
        table.add_row(
            "[bold]Average (all filters)[/bold]",
            "",
            f"[bold]{avg_improvement:.6f}[/bold]",
            "",
        )
        table.add_row(
            "[bold]Weighted Avg (all filters)[/bold]",
            "",
            f"[bold]{weighted_avg_improvement:.6f}[/bold]",
            "",
        )

        # Add statistics for filters above threshold if min_score_improvement is set
        if min_score_improvement is not None:
            above_threshold = [
                (key, imp)
                for key, imp in score_improvements.items()
                if imp >= min_score_improvement
            ]

            if len(above_threshold) > 0:
                improvements_above = [imp for _, imp in above_threshold]
                avg_above = sum(improvements_above) / len(improvements_above)

                # Weighted average for above threshold
                weighted_sum_above = 0.0
                total_splits_above = 0
                for key, improvement in above_threshold:
                    n_splits = len(filter_splits_dict[key])
                    weighted_sum_above += improvement * n_splits
                    total_splits_above += n_splits
                weighted_avg_above = (
                    weighted_sum_above / total_splits_above
                    if total_splits_above > 0
                    else 0.0
                )

                table.add_row(
                    f"[bold]Average (>= {min_score_improvement:.6f})[/bold]",
                    "",
                    f"[bold]{avg_above:.6f}[/bold]",
                    "",
                )
                table.add_row(
                    f"[bold]Weighted Avg (>= {min_score_improvement:.6f})[/bold]",
                    "",
                    f"[bold]{weighted_avg_above:.6f}[/bold]",
                    "",
                )

        console.print()
        console.print(table)
        console.print()

    return score_improvements


def prepare_splits(
    trades_df: pd.DataFrame,
    filters_df: pd.DataFrame,
    spread_width: int,
    left_only_filters: list[str],
    right_only_filters: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_splits_per_filter: int | None = None,
    max_depth: int = 1,
    min_samples: int = 1,
    score_func=None,
    keep_best_n: int | None = None,
    verbose: str = "no",
    filter_granularities: dict[str, float] | None = None,
    filter_eval_folds: int = 5,
    filter_eval_repeats: int = 2,
    min_score_improvement: float | None = None,
    purge_pct: float = 0.1,
    embargo_pct: float = 0.0,
    min_train_pct: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, list[Split]]:
    """
    Prepare splits and tensors for Option Alpha trading analysis.

    This function aligns trade and filter data, calculates reward-per-risk metrics,
    identifies potential split points in the data, and converts everything to
    PyTorch tensors for further analysis. Optionally generates child splits up to
    max_depth by combining parent splits with logical AND.

    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame created from parse_trade_log function, with DatetimeIndex
        containing columns: risk, profit, etc.
    filters_df : pd.DataFrame
        DataFrame created from get_filters function, with DatetimeIndex
        containing various filter columns.
    spread_width : int
        Width of the option spread in points (e.g., 20 for a 20-wide spread).
    left_only_filters : list[str]
        List of filter column names that should only generate left splits.
    right_only_filters : list[str]
        List of filter column names that should only generate right splits.
    device : torch.device
        PyTorch device to place tensors on (e.g., torch.device('cpu')).
    dtype : torch.dtype
        PyTorch dtype for tensors (e.g., torch.float32).
    max_splits_per_filter : int | None, optional
        Maximum number of splits per direction (left/right) to create for each filter.
        If None (default), all possible splits are created. When set, splits are
        selected to distribute samples as evenly as possible across buckets while
        respecting the constraint that samples with equal filter values cannot be
        separated. Default is None.
    max_depth : int, optional
        Maximum depth for generating child splits. Default is 1 (only depth 1 splits).
        If greater than 1, child splits are generated by combining parent splits
        with logical AND up to the specified depth. At depth 2, combines depth 1
        splits with each other. At depths > 2, combines previous depth's new splits
        with depth 1 splits.
    min_samples : int, optional
        Minimum number of samples required on each side of a split. Default is 1.
        Only splits with at least min_samples samples on the selected side will be
        created. This applies to both left and right splits independently, so they
        may have different thresholds. Also affects exclusion mask calculation.
    score_func : callable, optional
        Function that takes y (n_samples) and masks (batch_size, n_samples) and
        returns a float tensor containing a score for each batch. Default is None.
    keep_best_n : int | None, optional
        If not None, keep only the top n splits based on their scores. If set,
        score_func must also be provided. Default is None.
    verbose : str, optional
        Controls printing of splits. Default is "no".
        - "no": No printing
        - "best": Print the split with the highest score at the end of each depth
                  (requires score_func to be not None)
        - "all": Print all splits in all_splits list at the end of each depth.
                 If score_func is not None, splits are printed in descending score order.
    filter_granularities : dict[str, float] | None, optional
        Dictionary mapping filter names to granularity values. Default is None (empty dict).
        When generating splits for a filter in this dict, values are rounded to multiples
        of the granularity before selecting split indices:
        - Left direction: round values up to nearest granularity multiple
        - Right direction: round values down to nearest granularity multiple
        - Threshold: rounded in opposite direction after averaging unique values
    filter_eval_folds : int, optional
        Number of folds for cross-validation when evaluating filters. Default is 5.
        Only used if score_func is not None.
    filter_eval_repeats : int, optional
        Number of times to repeat cross-validation when evaluating filters. Default is 2.
        Only used if score_func is not None.
    min_score_improvement : float | None, optional
        If not None, filters with score improvement less than this value will be removed
        from depth_1_splits before merging. Only used if score_func is not None.
        Default is None.
    purge_pct : float, optional
        Percentage of test set size to purge from end of training set during filter
        evaluation CV (0.0 to 1.0). Default is 0.1 (10%). Prevents autocorrelation
        leakage by removing observations right after training set.
    embargo_pct : float, optional
        Additional percentage of test set size for gap between train end and test start
        during filter evaluation CV (0.0 to 1.0). Default is 0.0. Provides extra safety
        for high-frequency data.
    min_train_pct : float, optional
        Minimum training set size as percentage of total samples during filter
        evaluation CV (0.0 to 1.0). Default is 0.2 (20%). Ensures sufficient training
        data for stable evaluation.

    Returns
    -------
    X : torch.Tensor
        2-D tensor of filter values (n_samples x n_features) with added
        reward_per_risk column.
    y : torch.Tensor
        1-D tensor of return on risk (RoR) values calculated as profit / risk.
    splits : list[Split]
        List of Split objects, each representing a potential split condition.

    Raises
    ------
    ValueError
        If there are dates in trades_df without corresponding dates in filters_df.

    Notes
    -----
    - Reward per risk is calculated as: (spread_width * 100 - risk) / risk
    - For each filter column, splits are created at the midpoint between
      consecutive unique values.
    - Left splits include values <= threshold, right splits include values >= threshold.
    - At depth 2, child splits are created by combining depth 1 splits with each other
      (avoiding duplicates using j > i).
    - At depths > 2, child splits are created by combining the previous depth's new
      splits with the original depth 1 splits.
    """
    # Validate parameters
    if keep_best_n is not None and score_func is None:
        raise ValueError("score_func must be provided when keep_best_n is not None")

    if verbose not in ["no", "best", "all"]:
        raise ValueError(
            f"verbose must be one of 'no', 'best', or 'all', got '{verbose}'"
        )

    if verbose == "best" and score_func is None:
        raise ValueError("verbose='best' requires score_func to be not None")

    # Initialize filter_granularities with default values
    if filter_granularities is None:
        filter_granularities = {}

    # Add default granularities for computed columns
    default_granularities = {
        "reward_per_risk": 0.001,
        "premium": 1.0,
        "is_monday": 1,
        "is_tuesday": 1,
        "is_wednesday": 1,
        "is_thursday": 1,
        "is_friday": 1,
        "open_minutes": 5,
    }

    # Merge default granularities with user-provided ones (user values take precedence)
    filter_granularities = {**default_granularities, **filter_granularities}

    # Align filters_df with trades_df
    filters_df = _align_filters_with_trades(filters_df, trades_df)

    # Add computed columns and update left_only_filters
    filters_df, left_only_filters, right_only_filters = _add_computed_columns(
        filters_df, trades_df, spread_width, left_only_filters, right_only_filters
    )

    # Calculate RoR (y) as profit / risk
    # Done early so it can be used for scoring
    y = torch.tensor(  # pylint: disable=invalid-name
        (trades_df["profit"] / trades_df["risk"]).values, dtype=dtype, device=device
    )

    # Find all depth 1 splits
    depth_1_splits: list[Split] = []
    filter_names = filters_df.columns.tolist()

    for col_idx, col_name in enumerate(filter_names):
        filter_splits = _create_splits_for_filter(
            col_idx,
            col_name,
            filters_df,
            left_only_filters,
            right_only_filters,
            device,
            dtype,
            max_splits_per_filter,
            min_samples,
            filter_granularities,
        )
        depth_1_splits.extend(filter_splits)
    print(f"Generated {len(depth_1_splits)} depth 1 splits.")

    # Evaluate filters using cross-validation if score_func is provided
    if score_func is not None:
        # Evaluate filters before scoring to get score improvements
        filter_score_improvements = _evaluate_filters(
            depth_1_splits,
            y,
            score_func,
            filter_eval_folds,
            filter_eval_repeats,
            verbose,
            min_score_improvement,
            purge_pct,
            embargo_pct,
            min_train_pct,
        )

        # Remove splits with insufficient score improvement if min_score_improvement is set
        if min_score_improvement is not None and len(filter_score_improvements) > 0:
            filtered_splits = []
            for split in depth_1_splits:
                # Check if this split should be kept
                should_keep = False
                if len(split.filters) > 0:
                    # Get the first filter (depth 1 splits should have only one filter)
                    filter_info = split.filters[0]
                    key = (filter_info.filter_name, filter_info.direction)
                    if key in filter_score_improvements:
                        if filter_score_improvements[key] >= min_score_improvement:
                            should_keep = True
                else:
                    # Keep splits without filters
                    should_keep = True

                if should_keep:
                    filtered_splits.append(split)

            original_count = len(depth_1_splits)
            depth_1_splits = filtered_splits
            removed_count = original_count - len(depth_1_splits)
            if removed_count > 0:
                print(
                    f"Removed {removed_count} splits with score improvement "
                    f"< {min_score_improvement:.6f}"
                )

    # Merge splits with identical masks
    depth_1_splits = _merge_identical_splits(depth_1_splits)

    # Score depth_1_splits if score_func is provided
    if score_func is not None:
        _score_splits(depth_1_splits, y, score_func)

    # Start with depth 1 splits
    # If keep_best_n is specified, only keep top n in all_splits
    # but keep all depth_1_splits for generating child splits
    if keep_best_n is not None:
        all_splits = _keep_top_n_splits(depth_1_splits, keep_best_n)
    else:
        all_splits = depth_1_splits.copy()

    # Print depth 1 splits if verbose is enabled
    if verbose == "best":
        if len(all_splits) > 0:
            best_split = max(
                all_splits,
                key=lambda s: s.score if s.score is not None else float("-inf"),
            )
            _print_splits_for_depth(1, [best_split], score_func)
    elif verbose == "all":
        _print_splits_for_depth(1, all_splits, score_func)

    # Generate child splits if max_depth > 1
    if max_depth > 1:
        # Track the previous depth's new splits
        # Use all_splits (which contains filtered depth 1 splits if keep_best_n is set)
        previous_depth_splits = all_splits

        # Pre-stack masks for depth_1_splits to avoid redundant stacking in each iteration
        depth_1_masks_stacked = torch.stack(
            [split.mask for split in depth_1_splits], dim=0
        )

        for depth in range(2, max_depth + 1):
            # Calculate exclusion mask between the two parent sets
            # Pass the pre-stacked depth_1_masks to avoid redundant stacking
            exclusion_mask = _calculate_exclusion_mask(
                previous_depth_splits,
                depth_1_splits,
                device,
                min_samples,
                masks_b_stacked=depth_1_masks_stacked,
            )

            # Generate child splits from non-exclusive parent pairs
            new_splits = _generate_child_splits(
                previous_depth_splits,
                depth_1_splits,
                exclusion_mask,
            )
            print(f"Generated {len(new_splits)} depth {depth} splits.")

            # Exit early if no new splits were generated
            if len(new_splits) == 0:
                break

            # Optimization: Score new_splits immediately and work with candidates only
            # This avoids expensive operations on all splits when keep_best_n is set
            if keep_best_n is not None and score_func is not None:
                # Step 1: Score all new splits
                _score_splits(new_splits, y, score_func)

                # Step 2: Sort new splits by descending score
                new_splits.sort(
                    key=lambda s: s.score if s.score is not None else float("-inf"),
                    reverse=True,
                )

                # Step 3-4: Process candidates in batches
                candidate_splits = []
                candidate_start_idx = 0
                batch_size = 2 * keep_best_n

                # Loop to ensure we get enough valid candidates
                while len(candidate_splits) < keep_best_n and candidate_start_idx < len(
                    new_splits
                ):
                    # Select next batch of candidates
                    candidate_end_idx = min(
                        candidate_start_idx + batch_size, len(new_splits)
                    )
                    batch = new_splits[candidate_start_idx:candidate_end_idx]

                    # Remove redundant splits and merge identical ones on this batch
                    batch = _remove_redundant_splits(batch, all_splits)
                    if len(batch) > 0:
                        batch = _merge_identical_splits(batch)
                        candidate_splits.extend(batch)

                    candidate_start_idx = candidate_end_idx

                # Step 5: Add candidate splits to all_splits (not all new_splits)
                all_splits.extend(candidate_splits)

                # Keep only top n all_splits
                all_splits = _keep_top_n_splits(all_splits, keep_best_n)

                # Update previous_depth_splits to only contain kept splits from candidates
                all_splits_set = set(id(s) for s in all_splits)
                previous_depth_splits = [
                    split for split in candidate_splits if id(split) in all_splits_set
                ]

                # Exit early if no new splits were kept
                if len(previous_depth_splits) == 0:
                    break
            else:
                # Original path when keep_best_n is not set or score_func is not provided
                # Remove new splits that have identical masks to any existing splits
                new_splits = _remove_redundant_splits(new_splits, all_splits)

                # Exit early if no new splits remain
                if len(new_splits) == 0:
                    break

                # Merge identical splits among new splits
                new_splits = _merge_identical_splits(new_splits)

                # Exit early if no new splits remain
                if len(new_splits) == 0:
                    break

                # Score new_splits if score_func is provided
                if score_func is not None:
                    _score_splits(new_splits, y, score_func)

                # Add new splits to all_splits
                all_splits.extend(new_splits)

                # Keep only top n all_splits if keep_best_n is specified
                if keep_best_n is not None:
                    all_splits = _keep_top_n_splits(all_splits, keep_best_n)

                    # Update previous_depth_splits to only contain kept splits from new_splits
                    all_splits_set = set(id(s) for s in all_splits)
                    previous_depth_splits = [
                        split for split in new_splits if id(split) in all_splits_set
                    ]

                    # Exit early if no new splits were kept
                    if len(previous_depth_splits) == 0:
                        break
                else:
                    # Update previous_depth_splits for the next iteration
                    previous_depth_splits = new_splits

            # Print splits for this depth if verbose is enabled
            if verbose == "best":
                if len(all_splits) > 0:
                    best_split = max(
                        all_splits,
                        key=lambda s: s.score if s.score is not None else float("-inf"),
                    )
                    _print_splits_for_depth(depth, [best_split], score_func)
            elif verbose == "all":
                _print_splits_for_depth(depth, all_splits, score_func)

    # Convert filters_df to torch tensor X
    X = torch.tensor(
        filters_df.values, dtype=dtype, device=device
    )  # pylint: disable=invalid-name

    return X, y, all_splits
