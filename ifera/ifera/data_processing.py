"""
Data processing functionality for financial data.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from .models import InstrumentData
from .file_utils import make_path

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
    )[group.columns]

    merged["close"] = merged["close"].ffill()
    first_open = group["open"].iloc[0]
    merged["close"] = merged["close"].fillna(first_open)

    merged[["open", "high", "low", "close"]] = merged[
        ["open", "high", "low", "close"]
    ].bfill(axis=1)
    merged["volume"] = merged["volume"].fillna(0).astype("int32")
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
    grouped.columns = [
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


def aggregate_large_quote_file(
    input_file: str, output_file: str, chunksize: int = 1_000_000
):
    """Process a large quote file in chunks, aggregating by second."""
    max_decimals = find_max_decimals_in_file(input_file, chunk_size=chunksize)
    print(f"Detected max decimals: {max_decimals}")
    float_formatter = make_float_formatter(max_decimals)
    total_lines = count_lines(input_file)
    num_chunks = total_lines + chunksize - 1 // chunksize  # Removed superfluous parens

    with open(output_file, "w", encoding="utf-8") as f_out:
        header = (
            "Date,Time,"
            "BidOpen,BidHigh,BidLow,BidClose,"
            "AskOpen,AskHigh,AskLow,AskClose,"
            "Open,High,Low,Close,"
            "Volume,VWAP\n"
        )
        f_out.write(header)

    col_names = ["Date", "Time", "Bid", "Ask", "Price", "Size"]
    partial_df = None
    pbar = tqdm(total=num_chunks, desc="Processing chunks")

    for chunk in pd.read_csv(
        input_file,
        names=col_names,
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
        if partial_df is None:
            partial_df = chunk.iloc[:0].copy()

        combined = pd.concat([partial_df, chunk], ignore_index=True)
        if combined.empty:
            partial_df = chunk.iloc[:0].copy()
            continue

        combined["DateTime"] = combined["Date"] + " " + combined["Time"]
        last_dt = combined.iloc[-1]["DateTime"]
        mask_partial = combined["DateTime"] == last_dt
        partial_df = combined[mask_partial].copy()
        to_aggregate = combined[~mask_partial].copy()
        to_aggregate.drop(columns=["DateTime"], inplace=True, errors="ignore")
        partial_df.drop(columns=["DateTime"], inplace=True, errors="ignore")

        if not to_aggregate.empty:
            aggregated = aggregate_by_second(to_aggregate, max_decimals)
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


def process_data(df: pd.DataFrame, instrument: InstrumentData, zipfile: bool) -> None:
    """Process raw data into a standardized format."""
    try:
        params = {
            "time_step": instrument.time_step,
            "start_time_offset": instrument.trading_start,
            "start_time": pd.Timedelta(0),
            "end_time": instrument.end_time,
            "total_steps": instrument.total_steps,
            "offset_seconds": instrument.trading_start.total_seconds(),
        }

        print("Converting datetime columns...")
        # Pre-allocate all columns we'll need
        dt_index = pd.to_datetime(df.index)
        n_rows = len(df)
        
        # Create empty columns with appropriate dtypes
        df = df.assign(
            date=pd.Series(index=df.index, dtype='datetime64[ns]'),
            time=pd.Series(index=df.index, dtype='timedelta64[ns]'),
            offset_time=pd.Series(index=df.index, dtype='timedelta64[ns]'),
            trade_date=pd.Series(index=df.index, dtype='datetime64[ns]'),
            ord_trade_date=pd.Series(index=df.index, dtype='int32'),
            time_seconds=pd.Series(index=df.index, dtype='int32'),
            offset_time_seconds=pd.Series(index=df.index, dtype='int32'),
            ord_date=pd.Series(index=df.index, dtype='int32')
        )

        # Fill datetime-related columns
        print("Calculating time offsets...")
        with tqdm(total=3, desc="Time calculations") as pbar:
            # Convert date/time components
            dates = [d.date() for d in dt_index]
            df['date'].values[:] = pd.to_datetime(dates)
            df['time'].values[:] = pd.to_timedelta(
                [d.hour * 3600 + d.minute * 60 + d.second for d in dt_index], unit="s"
            )
            pbar.update(1)

            # Calculate offset times
            df['offset_time'].values[:] = df['time'] - params["start_time_offset"]
            df['offset_time'].values[:] = df['offset_time'].apply(
                lambda x: x - pd.to_timedelta(x.days, unit="d")
            )
            pbar.update(1)

            # Calculate trade dates
            df['trade_date'].values[:] = pd.to_datetime(
                [(d - params["start_time_offset"]).date() for d in dt_index]
            )
            pbar.update(1)

        if instrument.remove_dates is not None:
            df = df[~df["trade_date"].isin(instrument.remove_dates)]

        # Process each group separately with progress bar
        print("Processing groups...")
        unique_dates = df["trade_date"].unique()
        processed_groups = []
        for date in tqdm(unique_dates, desc="Processing trade dates"):
            group = df[df["trade_date"] == date]
            processed_group = add_missing_rows(
                group,
                start_time=params["start_time"],
                end_time=params["end_time"],
                time_step=params["time_step"],
            )
            processed_groups.append(processed_group)

        if processed_groups:
            df = pd.concat(processed_groups, ignore_index=True)
        else:
            df = pd.DataFrame()

        df = df[
            (df["offset_time"] >= params["start_time"])
            & (df["offset_time"] <= params["end_time"])
        ]
        df = df.groupby("trade_date").filter(
            lambda x: x["open"].count() == params["total_steps"]
        )
        df.sort_values(["trade_date", "offset_time"], inplace=True)

        print("Performing final calculations...")
        with tqdm(total=4, desc="Final calculations") as pbar:
            # Calculate ordinal dates and time values
            df['ord_trade_date'].values[:] = df['trade_date'].map(lambda x: x.toordinal())
            pbar.update(1)
            
            df['time_seconds'].values[:] = (
                df['offset_time'].map(lambda x: x.total_seconds())
                .add(params["offset_seconds"])
                .mod(SECONDS_IN_DAY)
            )
            pbar.update(1)
            
            df['offset_time_seconds'].values[:] = df['offset_time'].map(lambda x: x.total_seconds())
            pbar.update(1)
            
            df['ord_date'].values[:] = df['ord_trade_date'] + (
                (df['offset_time_seconds'] + params["offset_seconds"]) // SECONDS_IN_DAY
            )
            pbar.update(1)

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
        df = df[cols].reset_index(drop=True)

        print("Saving processed data...")
        try:
            output_path = make_path(
                raw=False, instrument=instrument, remove_file=True, zipfile=zipfile
            )
            if zipfile:
                df.to_csv(
                    str(output_path), header=False, index=False, compression="zip"
                )
            else:
                df.to_csv(str(output_path), header=False, index=False)
            print(f"Processed data saved to {output_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error saving processed file for instrument {instrument.symbol}: {e}"
            ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error processing data for instrument {instrument.symbol}: {e}"
        ) from e
