"""
Module for parsing Option Alpha trade log HTML grids.

This module provides functionality to parse HTML-formatted trade log grids
from Option Alpha and convert them into pandas DataFrames.
"""

import re
from dataclasses import dataclass
from datetime import datetime, time
from typing import NamedTuple, Optional

import pandas as pd
import torch
from bs4 import BeautifulSoup


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
    try:
        with open(file_name, "r") as f:
            html = f.read()

        df = parse_filter_log(html)
        df["filter"] = 1

        # Eliminate duplicate dates
        df = _check_and_eliminate_duplicates(df, "filter")

    except FileNotFoundError:
        return None

    result: pd.DataFrame = df[["date", "filter"]]  # type: ignore
    return result


def parse_simple_indicator(file_name: str) -> pd.DataFrame | None:
    try:
        with open(file_name, "r") as f:
            html = f.read()

        df = parse_filter_log(html)
        df["indicator"] = df["description"].astype(float)

        # Eliminate duplicate dates
        df = _check_and_eliminate_duplicates(df, "indicator")

    except FileNotFoundError:
        return None

    result: pd.DataFrame = df[["date", "indicator"]]  # type: ignore
    return result


def parse_range_with(prefix: str) -> pd.DataFrame | None:
    file_name = f"{FILTERS_FOLDER}{prefix}-RANGE_WIDTH.txt"
    try:
        with open(file_name, "r") as f:
            html = f.read()

        df = parse_filter_log(html)

    except FileNotFoundError:
        return None

    def _parse_range_width(description):
        match = re.search(r"Opening Range: ([\d,]+\.\d+) - ([\d,]+\.\d+)", description)
        if match:
            min_val = float(match.group(1).replace(",", ""))
            max_val = float(match.group(2).replace(",", ""))
            return (max_val - min_val) / max_val if max_val != 0 else 0
        else:
            return None

    df["range_width"] = df["description"].apply(_parse_range_width)

    # Eliminate duplicate dates
    df = _check_and_eliminate_duplicates(df, "range_width")

    result: pd.DataFrame = df[["date", "range_width"]]  # type: ignore
    return result


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

    range_df = parse_range_with(prefix)
    if range_df is not None:
        dfs.append(range_df)

    simple_filters = [
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
    for filter_name in simple_filters:
        filter_df = parse_simple_filter(f"{FILTERS_FOLDER}{prefix}-{filter_name}.txt")
        if filter_df is not None:
            filter_df = filter_df.rename(columns={"filter": f"{filter_name.lower()}"})
            dfs.append(filter_df)

    indicator_names = ["ADX_14"]
    for indicator_name in indicator_names:
        indicator_df = parse_simple_indicator(
            f"{FILTERS_FOLDER}{prefix}-{indicator_name}.txt"
        )
        if indicator_df is not None:
            indicator_df = indicator_df.rename(
                columns={"indicator": f"{indicator_name.lower()}"}
            )
            dfs.append(indicator_df)

    if dfs:
        from functools import reduce

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
    else:
        # Return empty DataFrame with DatetimeIndex
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))


@dataclass
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
    parents : list[tuple[int, int]]
        List of pairs of parent split indices. Each pair represents the two parent
        splits that were combined with logical AND to create this child split.
        Empty for depth 1 (original) splits.
    """

    mask: torch.Tensor
    filters: list[FilterInfo]
    parents: list[tuple[int, int]]


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

    Adds reward_per_risk, weekday filters (is_monday, is_tuesday, etc.),
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

    Returns
    -------
    list[Split]
        List of Split objects for this filter
    """
    # Get sorted unique values
    unique_vals = torch.tensor(
        sorted(filters_df[col_name].unique()), dtype=dtype, device=device
    )

    if len(unique_vals) <= 1:
        # No splits possible for this filter
        return []

    # Convert filter column to tensor for masking
    col_tensor = torch.tensor(filters_df[col_name].values, dtype=dtype, device=device)

    splits = []

    # Create left splits if not in right_only_filters
    if col_name not in right_only_filters:
        left_split_indices = _select_split_indices(
            unique_vals, col_tensor, max_splits_per_filter, min_samples, "left"
        )
        for i in left_split_indices:
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
            left_mask = col_tensor <= threshold
            splits.append(
                Split(
                    mask=left_mask,
                    filters=[FilterInfo(col_idx, col_name, threshold.item(), "left")],
                    parents=[],
                )
            )

    # Create right splits if not in left_only_filters
    if col_name not in left_only_filters:
        right_split_indices = _select_split_indices(
            unique_vals, col_tensor, max_splits_per_filter, min_samples, "right"
        )
        for i in right_split_indices:
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
            right_mask = col_tensor >= threshold
            splits.append(
                Split(
                    mask=right_mask,
                    filters=[FilterInfo(col_idx, col_name, threshold.item(), "right")],
                    parents=[],
                )
            )

    return splits


def _merge_identical_splits(splits: list[Split], device: torch.device) -> list[Split]:
    """
    Merge splits with identical masks.

    Parameters
    ----------
    splits : list[Split]
        List of Split objects
    device : torch.device
        PyTorch device for tensors

    Returns
    -------
    list[Split]
        List of merged Split objects
    """
    if len(splits) == 0:
        return splits

    # Stack all masks into a 2D tensor (n_splits x n_samples)
    all_masks = torch.stack([split.mask for split in splits], dim=0)

    # Convert to float for matrix operations
    all_masks_float = all_masks.float()

    # Compute pairwise equality of masks
    n_splits = len(splits)
    n_samples = all_masks.shape[1]

    # Create a matrix where entry (i,j) is the number of positions where masks match
    match_counts = torch.matmul(all_masks_float, all_masks_float.T) + torch.matmul(
        1 - all_masks_float, (1 - all_masks_float).T
    )

    # Two masks are identical if they match at all positions
    mask_equality = match_counts == n_samples

    # Find groups of identical masks
    merged_splits: list[Split] = []
    processed = torch.zeros(n_splits, dtype=torch.bool, device=device)

    for i in range(n_splits):
        if processed[i]:
            continue

        # Find all splits with identical masks to split i
        identical_indices = mask_equality[i].nonzero(as_tuple=True)[0]

        # Merge filters and parents from all identical splits
        merged_filters = []
        merged_parents = []
        for idx in identical_indices:
            idx_int = int(idx.item())
            merged_filters.extend(splits[idx_int].filters)
            merged_parents.extend(splits[idx_int].parents)
            processed[idx] = True

        # Create a merged split with all filters and parents
        merged_splits.append(
            Split(mask=splits[i].mask, filters=merged_filters, parents=merged_parents)
        )

    return merged_splits


def _calculate_exclusion_mask(
    parent_set_a: list[Split],
    parent_set_b: list[Split],
    device: torch.device,
    min_samples: int,
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

    Returns
    -------
    torch.Tensor
        2-D bool tensor (n_set_a x n_set_b) indicating mutual exclusion
    """
    n_set_a = len(parent_set_a)
    n_set_b = len(parent_set_b)

    if n_set_a == 0 or n_set_b == 0:
        return torch.zeros((n_set_a, n_set_b), dtype=torch.bool, device=device)

    # Stack masks
    masks_a = torch.stack([split.mask for split in parent_set_a], dim=0)
    masks_b = torch.stack([split.mask for split in parent_set_b], dim=0)

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


def _generate_child_splits(
    parent_set_a: list[Split],
    parent_set_b: list[Split],
    exclusion_mask: torch.Tensor,
    parent_set_a_offset: int = 0,
    parent_set_b_offset: int = 0,
) -> list[Split]:
    """
    Generate child splits from pairs of non-exclusive parent splits from two sets.

    When the two parent sets are the same (depth 2), we only check the upper triangle
    (j > i) to avoid duplicates. When the sets are different (depth > 2), we check
    all combinations.

    Parameters
    ----------
    parent_set_a : list[Split]
        First set of parent splits
    parent_set_b : list[Split]
        Second set of parent splits
    exclusion_mask : torch.Tensor
        2-D bool tensor (len(parent_set_a) x len(parent_set_b)) indicating which
        pairs are mutually exclusive
    parent_set_a_offset : int, optional
        Offset to add to parent_set_a indices when recording parent pairs.
        Default is 0.
    parent_set_b_offset : int, optional
        Offset to add to parent_set_b indices when recording parent pairs.
        Default is 0.

    Returns
    -------
    list[Split]
        List of newly created child Split objects
    """
    n_set_a = len(parent_set_a)
    n_set_b = len(parent_set_b)
    child_splits = []

    # Check if the two sets are the same (by object identity)
    sets_are_same = parent_set_a is parent_set_b

    if sets_are_same:
        # Same parent sets: only check upper triangle to avoid duplicates
        for i in range(n_set_a):
            for j in range(i + 1, n_set_b):
                if not exclusion_mask[i, j]:
                    # Create child split by combining masks with logical AND
                    child_mask = parent_set_a[i].mask & parent_set_b[j].mask
                    # Child splits have empty filters list and parent indices
                    child_splits.append(
                        Split(
                            mask=child_mask,
                            filters=[],
                            parents=[
                                (i + parent_set_a_offset, j + parent_set_b_offset)
                            ],
                        )
                    )
    else:
        # Different parent sets: check all combinations
        for i in range(n_set_a):
            for j in range(n_set_b):
                if not exclusion_mask[i, j]:
                    # Create child split by combining masks with logical AND
                    child_mask = parent_set_a[i].mask & parent_set_b[j].mask
                    # Child splits have empty filters list and parent indices
                    child_splits.append(
                        Split(
                            mask=child_mask,
                            filters=[],
                            parents=[
                                (i + parent_set_a_offset, j + parent_set_b_offset)
                            ],
                        )
                    )

    return child_splits


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
    keep_indices = (~has_match).nonzero(as_tuple=True)[0]

    # Return only new splits that don't match any old split
    return [new_splits[i] for i in keep_indices.tolist()]


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
    # Align filters_df with trades_df
    filters_df = _align_filters_with_trades(filters_df, trades_df)

    # Add computed columns and update left_only_filters
    filters_df, left_only_filters, right_only_filters = _add_computed_columns(
        filters_df, trades_df, spread_width, left_only_filters, right_only_filters
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
        )
        depth_1_splits.extend(filter_splits)

    # Merge splits with identical masks
    depth_1_splits = _merge_identical_splits(depth_1_splits, device)

    # Start with depth 1 splits
    all_splits = depth_1_splits.copy()

    # Generate child splits if max_depth > 1
    if max_depth > 1:
        # Track the previous depth's new splits
        previous_depth_splits = depth_1_splits

        for _ in range(2, max_depth + 1):
            # Calculate exclusion mask between the two parent sets
            exclusion_mask = _calculate_exclusion_mask(
                previous_depth_splits, depth_1_splits, device, min_samples
            )

            previous_depth_offset = len(all_splits) - len(previous_depth_splits)

            # Generate child splits from non-exclusive parent pairs
            new_splits = _generate_child_splits(
                previous_depth_splits,
                depth_1_splits,
                exclusion_mask,
                previous_depth_offset,
                0,
            )

            # Remove new splits that have identical masks to any existing splits
            new_splits = _remove_redundant_splits(new_splits, all_splits)

            # Exit early if no new splits remain
            if len(new_splits) == 0:
                break

            # Merge identical splits among new splits
            new_splits = _merge_identical_splits(new_splits, device)

            # Filter out splits with fewer than min_samples
            new_splits = [
                split for split in new_splits if split.mask.sum().item() >= min_samples
            ]

            # Exit early if no new splits remain after filtering
            if len(new_splits) == 0:
                break

            # Add new splits to all_splits
            all_splits.extend(new_splits)

            # Update previous_depth_splits for the next iteration
            previous_depth_splits = new_splits

    # Convert filters_df to torch tensor X
    X = torch.tensor(
        filters_df.values, dtype=dtype, device=device
    )  # pylint: disable=invalid-name

    # Calculate RoR (y) as profit / risk
    y = torch.tensor(  # pylint: disable=invalid-name
        (trades_df["profit"] / trades_df["risk"]).values, dtype=dtype, device=device
    )

    return X, y, all_splits
