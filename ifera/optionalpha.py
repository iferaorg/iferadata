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

    Attributes
    ----------
    mask : torch.Tensor
        1-D bool tensor representing the row-mask of the split
    filters : list[FilterInfo]
        List of filters that result in this mask. Each FilterInfo contains:
        filter_idx, filter_name, threshold, and direction
    """

    mask: torch.Tensor
    filters: list[FilterInfo]


def prepare_splits(
    trades_df: pd.DataFrame,
    filters_df: pd.DataFrame,
    spread_width: int,
    left_only_filters: list[str],
    right_only_filters: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_splits_per_filter: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[Split], torch.Tensor]:
    """
    Prepare splits and tensors for Option Alpha trading analysis.

    This function aligns trade and filter data, calculates reward-per-risk metrics,
    identifies potential split points in the data, and converts everything to
    PyTorch tensors for further analysis.

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

    Returns
    -------
    X : torch.Tensor
        2-D tensor of filter values (n_samples x n_features) with added
        reward_per_risk column.
    y : torch.Tensor
        1-D tensor of return on risk (RoR) values calculated as profit / risk.
    splits : list[Split]
        List of Split objects, each representing a potential split condition.
    splits_exclusion_mask : torch.Tensor
        2-D bool tensor (n_splits x n_splits) indicating which splits are
        mutually exclusive with each other.

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
    - Splits are mutually exclusive if they:
      1. Would result in an empty set when both are applied
      2. One split's rows are a subset of the other's (redundant/overlapping masks)
      3. A split is also marked as exclusive with itself (no point applying same mask twice)
    """
    # Step 1: Align filters_df with trades_df
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
    filters_df = filters_df.reindex(trades_df.index)

    # Step 2: Add reward_per_risk column
    filters_df["reward_per_risk"] = (
        spread_width * 100 - trades_df["risk"]
    ) / trades_df["risk"]

    # Add weekday filters based on date
    filters_df["is_monday"] = (filters_df.index.dayofweek == 0).astype(int)  # type: ignore
    filters_df["is_tuesday"] = (filters_df.index.dayofweek == 1).astype(int)  # type: ignore
    filters_df["is_wednesday"] = (filters_df.index.dayofweek == 2).astype(int)  # type: ignore
    filters_df["is_thursday"] = (filters_df.index.dayofweek == 3).astype(int)  # type: ignore
    filters_df["is_friday"] = (filters_df.index.dayofweek == 4).astype(int)  # type: ignore

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
    left_only_filters = list(left_only_filters) + [
        "is_monday",
        "is_tuesday",
        "is_wednesday",
        "is_thursday",
        "is_friday",
    ]

    # Step 3: Find all splits
    splits: list[Split] = []
    filter_names = filters_df.columns.tolist()

    for col_idx, col_name in enumerate(filter_names):
        # Get sorted unique values
        unique_vals = torch.tensor(
            sorted(filters_df[col_name].unique()), dtype=dtype, device=device
        )

        if len(unique_vals) <= 1:
            # No splits possible for this filter
            continue

        # Convert filter column to tensor for masking
        col_tensor = torch.tensor(
            filters_df[col_name].values, dtype=dtype, device=device
        )

        # Determine which split indices to use
        n_possible_splits = len(unique_vals) - 1

        # If max_splits_per_filter is set and we have more possible splits than allowed,
        # select splits that distribute samples evenly
        if (
            max_splits_per_filter is not None
            and n_possible_splits > max_splits_per_filter
        ):
            # Count samples for each unique value using vectorized operations
            # torch.unique returns sorted values and their counts
            _, sample_counts = torch.unique(col_tensor, return_counts=True)

            # Calculate cumulative counts - this tells us how many samples are <= each unique value
            cumsum = torch.cumsum(sample_counts, dim=0)
            total_samples = cumsum[-1].item()

            # For even distribution with max_splits_per_filter splits, we want to place
            # splits so that we create max_splits_per_filter + 1 buckets of approximately
            # equal size. Target size per bucket:
            target_bucket_size = total_samples / (max_splits_per_filter + 1)

            # Find the split indices that best achieve even distribution
            # For each potential split position k, placing a threshold between unique_vals[k]
            # and unique_vals[k+1] means cumsum[k] samples are on the left
            selected_indices = []
            for split_num in range(1, max_splits_per_filter + 1):
                # Target cumulative count for this split
                target_cumsum = split_num * target_bucket_size

                # Find the index where cumsum is closest to target_cumsum
                # We search among indices 0 to n_possible_splits-1
                diffs = torch.abs(cumsum[:-1] - target_cumsum)
                best_idx = int(torch.argmin(diffs).item())

                # Avoid duplicate indices
                while best_idx in selected_indices and best_idx < n_possible_splits - 1:
                    diffs[best_idx] = float("inf")
                    best_idx = int(torch.argmin(diffs).item())

                if best_idx not in selected_indices:
                    selected_indices.append(best_idx)

            # Sort the selected indices
            selected_indices = sorted(selected_indices)
            split_indices_to_use = selected_indices
        else:
            # Use all possible splits
            split_indices_to_use = list(range(n_possible_splits))

        # Create splits for selected indices
        for i in split_indices_to_use:
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0

            # Create left split (values <= threshold) if not in right_only_filters
            if col_name not in right_only_filters:
                left_mask = col_tensor <= threshold
                splits.append(
                    Split(
                        mask=left_mask,
                        filters=[
                            FilterInfo(col_idx, col_name, threshold.item(), "left")
                        ],
                    )
                )

            # Create right split (values >= threshold) if not in left_only_filters
            if col_name not in left_only_filters:
                right_mask = col_tensor >= threshold
                splits.append(
                    Split(
                        mask=right_mask,
                        filters=[
                            FilterInfo(col_idx, col_name, threshold.item(), "right")
                        ],
                    )
                )

    # Step 3.5: Merge splits with identical masks
    if len(splits) > 0:
        # Stack all masks into a 2D tensor (n_splits x n_samples)
        all_masks = torch.stack([split.mask for split in splits], dim=0)

        # Convert to float for matrix operations
        all_masks_float = all_masks.float()

        # Compute pairwise equality of masks using matrix multiplication
        # Two masks are equal if they have the same True values everywhere
        n_splits = len(splits)
        n_samples = all_masks.shape[1]

        # Create a matrix where entry (i,j) is the number of positions where masks i and j match
        # Matching means both True or both False
        match_counts = torch.matmul(all_masks_float, all_masks_float.T) + torch.matmul(
            1 - all_masks_float, (1 - all_masks_float).T
        )

        # Two masks are identical if they match at all positions
        mask_equality = match_counts == n_samples

        # Find groups of identical masks
        # Use a simple algorithm: for each mask, find all identical masks
        merged_splits: list[Split] = []
        processed = torch.zeros(n_splits, dtype=torch.bool, device=device)

        for i in range(n_splits):
            if processed[i]:
                continue

            # Find all splits with identical masks to split i
            identical_indices = mask_equality[i].nonzero(as_tuple=True)[0]

            # Merge filters from all identical splits
            merged_filters = []
            for idx in identical_indices:
                idx_int = int(idx.item())
                merged_filters.extend(splits[idx_int].filters)
                processed[idx] = True

            # Create a merged split with all filters
            merged_splits.append(
                Split(
                    mask=splits[i].mask,
                    filters=merged_filters,
                )
            )

        splits = merged_splits

    # Step 4: Calculate splits_exclusion_mask (vectorized)
    n_splits = len(splits)
    splits_exclusion_mask = torch.zeros(
        (n_splits, n_splits), dtype=torch.bool, device=device
    )

    if n_splits == 0:
        # No splits, return empty mask
        pass
    else:
        # Stack all masks into a 2D tensor (n_splits x n_samples)
        all_masks = torch.stack([split.mask for split in splits], dim=0)

        # Use matrix multiplication for all rules
        # Convert masks to float for matrix operations
        all_masks_float = all_masks.float()

        # Compute intersection counts: (n_splits, n_splits)
        # intersection_counts[i, j] = number of samples where both mask[i] and mask[j] are True
        intersection_counts = torch.matmul(all_masks_float, all_masks_float.T)

        # Rule 1: If applying both splits results in empty set
        # has_intersection[i, j] = True if intersection_counts[i, j] > 0
        has_intersection = intersection_counts > 0
        # Empty intersection means mutually exclusive
        rule1_mask = ~has_intersection

        # Rule 2: Splits are exclusive if one's rows are a subset of the other's (redundant)
        # This means: all rows in i are also in j OR all rows in j are also in i
        # Split i is subset of j if: intersection_counts[i, j] == mask_sums[i]
        # Split j is subset of i if: intersection_counts[i, j] == mask_sums[j]
        mask_sums = all_masks_float.sum(dim=1)  # (n_splits,)
        # Broadcasting: (n_splits, 1) == (n_splits, n_splits)
        i_subset_of_j = intersection_counts == mask_sums.unsqueeze(1)
        # Broadcasting: (n_splits, n_splits) == (1, n_splits)
        j_subset_of_i = intersection_counts == mask_sums.unsqueeze(0)
        # Mark as exclusive if either is a subset of the other (prevents redundant masks)
        rule2_mask = i_subset_of_j | j_subset_of_i

        # Combine all rules with OR operation
        splits_exclusion_mask = rule1_mask | rule2_mask

        # Splits are exclusive with themselves (no point applying same mask twice)
        splits_exclusion_mask.fill_diagonal_(True)

    # Step 5: Convert filters_df to torch tensor X
    X = torch.tensor(
        filters_df.values, dtype=dtype, device=device
    )  # pylint: disable=invalid-name

    # Step 6: Calculate RoR (y) as profit / risk
    y = torch.tensor(  # pylint: disable=invalid-name
        (trades_df["profit"] / trades_df["risk"]).values, dtype=dtype, device=device
    )

    return X, y, splits, splits_exclusion_mask
