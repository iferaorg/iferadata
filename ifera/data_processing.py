"""
Data processing functionality for financial data.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import BaseInstrumentConfig
from .enums import Source
from .file_utils import make_instrument_path

SECONDS_IN_DAY = 86400


def add_missing_rows(
    group: pd.DataFrame,
    start_time: pd.Timedelta,
    end_time: pd.Timedelta,
    time_step: pd.Timedelta,
) -> pd.DataFrame:
    """Add missing rows for each time step in a group."""
    all_time_steps = pd.timedelta_range(start=start_time, end=end_time, freq=time_step)
    all_time_step_rows = pd.DataFrame(
        {
            "trade_date": group["trade_date"].iloc[0],
            "offset_time": all_time_steps,
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": np.nan,
            "volume": np.nan,
        }
    )
    merged = pd.merge(
        group,
        all_time_step_rows,
        on=["trade_date", "offset_time"],
        how="outer",
        suffixes=("", "_y"),
        sort=True,
    )[group.columns].copy()
    merged["close"] = merged["close"].ffill()
    first_open = group["open"].iloc[0]
    merged["close"] = merged["close"].fillna(first_open)
    merged[["open", "high", "low", "close"]] = merged[
        ["open", "high", "low", "close"]
    ].bfill(axis=1)
    merged["volume"] = merged["volume"].fillna(0, inplace=False).astype("int32")
    return merged


def detect_decimal_places(s: str) -> int:
    """Count decimal places in a string representation of a number."""
    s = s.strip()
    if "." not in s:
        return 0
    return len(s) - s.index(".") - 1


def find_max_decimals_in_file(
    input_file: str,
    chunk_size: int = 1_000_000,
    bid_col_idx: int = 2,
    ask_col_idx: int = 3,
    price_col_idx: int = 4,
) -> int:
    """Find maximum decimal places in numeric columns of a file."""
    max_decimals = 0
    lines_read = 0
    with open(input_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.strip().split(",")
            if len(parts) < price_col_idx + 1:
                continue
            for col_idx in (bid_col_idx, ask_col_idx, price_col_idx):
                max_decimals = max(max_decimals, detect_decimal_places(parts[col_idx]))
            lines_read += 1
            if lines_read >= chunk_size:
                break
    return max_decimals


def count_lines(filename: str, chunk_size: int = 1_000_000) -> int:
    """Count lines in a file by reading in chunks."""
    lines = 0
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            lines += chunk.count(b"\n")
    return lines


def make_float_formatter(max_decimals: int):
    """Create a function that formats floats with specified decimal places."""

    def float_formatter(x):
        if pd.isna(x):
            return ""
        formatted = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
        return formatted

    return float_formatter


def aggregate_by_second(df: pd.DataFrame, max_decimals: int) -> pd.DataFrame:
    """Aggregate data by second, computing OHLCV and VWAP."""
    if df.empty:
        return df
    df["PxSize"] = df["Price"] * df["Size"]
    grouped = df.groupby(["Date", "Time"], as_index=False, sort=False).agg(
        {
            "Bid": ["first", "max", "min", "last"],
            "Ask": ["first", "max", "min", "last"],
            "Price": ["first", "max", "min", "last"],
            "Size": "sum",
            "PxSize": "sum",
        }
    )
    # Convert list of strings to pandas Index to satisfy type checking
    grouped.columns = pd.Index(
        [
            "Date",
            "Time",
            "BidOpen",
            "BidHigh",
            "BidLow",
            "BidClose",
            "AskOpen",
            "AskHigh",
            "AskLow",
            "AskClose",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "PxSizeSum",
        ]
    )
    grouped["PxSizeSum"] = grouped["PxSizeSum"].astype(float)
    grouped["Volume"] = grouped["Volume"].astype(float)
    grouped["Close"] = grouped["Close"].astype(float)
    grouped["VWAP"] = grouped["Close"].astype(float)
    nonzero_mask = grouped["Volume"] != 0
    grouped.loc[nonzero_mask, "VWAP"] = (
        grouped.loc[nonzero_mask, "PxSizeSum"]
        .div(grouped.loc[nonzero_mask, "Volume"])
        .astype(float)
    )
    grouped["Volume"] = grouped["Volume"].astype(int)
    grouped.drop(columns=["PxSizeSum"], inplace=True)
    float_cols = [
        "BidOpen",
        "BidHigh",
        "BidLow",
        "BidClose",
        "AskOpen",
        "AskHigh",
        "AskLow",
        "AskClose",
        "Open",
        "High",
        "Low",
        "Close",
        "VWAP",
    ]
    grouped[float_cols] = grouped[float_cols].round(max_decimals)
    return grouped


def process_chunk(
    chunk: pd.DataFrame, partial_df: pd.DataFrame, max_decimals: int
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process a chunk of data, aggregating complete seconds and carrying over incomplete data."""
    if partial_df is None:
        combined = chunk
    else:
        combined = pd.concat([partial_df, chunk], ignore_index=True)

    if combined.empty:
        return None, partial_df  # type: ignore

    last_dt = combined.iloc[-1]["Date"] + " " + combined.iloc[-1]["Time"]
    mask = (combined["Date"] + " " + combined["Time"]) == last_dt
    to_aggregate = combined[~mask]
    new_partial_df = combined[mask]

    if not to_aggregate.empty:
        aggregated = aggregate_by_second(to_aggregate, max_decimals)
    else:
        aggregated = None

    return aggregated, new_partial_df  # type: ignore


def aggregate_large_quote_file(
    input_file: str, output_file: str, chunksize: int = 1_000_000
):
    """Process a large quote file in chunks, aggregating by second."""
    max_decimals = find_max_decimals_in_file(input_file, chunk_size=chunksize)
    print(f"Detected max decimals: {max_decimals}")
    float_formatter = make_float_formatter(max_decimals)
    total_lines = count_lines(input_file)
    num_chunks = (total_lines + chunksize - 1) // chunksize

    with open(output_file, "w", encoding="utf-8") as f_out:
        header = (
            "Date,Time,"
            "BidOpen,BidHigh,BidLow,BidClose,"
            "AskOpen,AskHigh,AskLow,AskClose,"
            "Open,High,Low,Close,"
            "Volume,VWAP\n"
        )
        f_out.write(header)

    partial_df = None
    pbar = tqdm(total=num_chunks, desc="Processing chunks")
    for chunk in pd.read_csv(
        input_file,
        names=["Date", "Time", "Bid", "Ask", "Price", "Size"],
        header=None,
        chunksize=chunksize,
        dtype={
            "Date": str,
            "Time": str,
            "Bid": float,
            "Ask": float,
            "Price": float,
            "Size": int,
        },
        encoding="utf-8",
    ):
        aggregated, partial_df = process_chunk(chunk, partial_df, max_decimals)  # type: ignore

        if aggregated is not None:
            aggregated.to_csv(
                output_file,
                mode="a",
                header=False,
                index=False,
                float_format=float_formatter,
            )
        pbar.update(1)
    pbar.close()

    if partial_df is not None and not partial_df.empty:
        aggregated = aggregate_by_second(partial_df, max_decimals)
        aggregated.to_csv(
            output_file,
            mode="a",
            header=False,
            index=False,
            float_format=float_formatter,
        )

    print("Aggregation complete.")


def calculate_time_columns(
    df: pd.DataFrame, instrument: BaseInstrumentConfig
) -> pd.DataFrame:
    """Calculate and assign datetime-related columns to the DataFrame."""
    dt_index = pd.to_datetime(df.index)
    df = df.assign(
        date=pd.to_datetime(dt_index.date),
        time=pd.to_timedelta(
            [d.hour * 3600 + d.minute * 60 + d.second for d in dt_index], unit="s"
        ),
        offset_time=pd.Series(index=df.index, dtype="timedelta64[ns]"),
        trade_date=pd.Series(index=df.index, dtype="datetime64[ns]"),
    )
    df["offset_time"] = df["time"] - instrument.trading_start
    df["offset_time"] = df["offset_time"].apply(
        lambda x: x - pd.to_timedelta(x.days, unit="d")
    )
    df["trade_date"] = pd.to_datetime(
        [(d - instrument.trading_start).date() for d in dt_index]
    )
    return df


def process_group(
    group: pd.DataFrame, instrument: BaseInstrumentConfig
) -> pd.DataFrame:
    """Process a group by adding missing rows based on instrument settings."""
    return add_missing_rows(
        group,
        start_time=pd.Timedelta(0),
        end_time=instrument.end_time,
        time_step=instrument.time_step,
    )


def perform_final_calculations(
    df: pd.DataFrame, instrument: BaseInstrumentConfig
) -> pd.DataFrame:
    """Perform final calculations for ordinal dates and time seconds."""
    offset_seconds = instrument.trading_start.total_seconds()
    df = df.assign(
        ord_trade_date=df["trade_date"].map(lambda x: x.toordinal()),
        time_seconds=(
            df["offset_time"]
            .map(lambda x: x.total_seconds())
            .add(offset_seconds)
            .mod(SECONDS_IN_DAY)
        ),
        offset_time_seconds=df["offset_time"].map(lambda x: x.total_seconds()),
    )
    df["ord_date"] = df["ord_trade_date"] + (
        (df["offset_time_seconds"] + offset_seconds) // SECONDS_IN_DAY
    )
    return df


def process_data(
    df: pd.DataFrame, instrument: BaseInstrumentConfig, zipfile: bool
) -> None:
    """
    Process raw data into a standardized format.

    This function transforms raw OHLCV data into a standardized format with proper time offsets
    and date calculations. It handles missing data points and ensures regular intervals.
    """
    try:
        print("Converting datetime columns...")
        df = calculate_time_columns(df, instrument)

        start_date_ts = pd.to_datetime(instrument.start_date)
        df = df[df["trade_date"] >= start_date_ts]

        if instrument.remove_dates is not None:
            remove_dates_ts = pd.to_datetime(instrument.remove_dates)
            df = df[~df["trade_date"].isin(remove_dates_ts)]

        if instrument.days_of_week is not None:
            df = df[df["trade_date"].dt.dayofweek.isin(instrument.days_of_week)]

        print("Processing groups...")
        # Convert the unique dates to a list explicitly with the right type
        unique_dates = df["trade_date"].unique()
        unique_dates_list = [pd.Timestamp(d) for d in unique_dates]

        processed_groups = [
            process_group(df[df["trade_date"] == date], instrument)
            for date in tqdm(unique_dates_list, desc="Processing trade dates")
        ]

        if processed_groups:
            df = pd.concat(processed_groups, ignore_index=True)
        else:
            df = pd.DataFrame()

        df = df[
            (df["offset_time"] >= pd.Timedelta(0))
            & (df["offset_time"] <= instrument.end_time)
        ]

        df = df.groupby("trade_date").filter(
            lambda x: x["open"].count() == instrument.total_steps
        )

        df.sort_values(["trade_date", "offset_time"], inplace=True)

        print("Performing final calculations...")
        df = perform_final_calculations(df, instrument)

        cols = [
            "ord_date",
            "time_seconds",
            "ord_trade_date",
            "offset_time_seconds",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df = df[cols]
        df = df.reset_index(drop=True, inplace=False)  # type: ignore

        print("Saving processed data...")
        output_path = make_instrument_path(
            source=Source.PROCESSED, instrument=instrument, remove_file=True
        )
        if zipfile:
            df.to_csv(str(output_path), header=False, index=False, compression="zip")
        else:
            df.to_csv(str(output_path), header=False, index=False)
        print(f"Processed data saved to {output_path}")

    except Exception as e:
        raise RuntimeError(
            f"Error processing data for instrument {instrument.symbol}: {e}"
        ) from e
