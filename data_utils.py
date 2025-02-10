"""
This module contains utility functions for processing large files of
financial quotes/trades data in (Date, Time, Bid, Ask, Price, Size) format.
"""

import os

import boto3
import pandas as pd
from tqdm import tqdm


###############################################################################
# 1. HELPER FUNCTIONS
###############################################################################

def count_lines(filename: str, chunk_size: int = 1_000_000) -> int:
    """
    Return the total number of lines in `filename` by reading 
    in chunks of `chunk_size` bytes.
    """
    lines = 0
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            lines += chunk.count(b"\n")
    return lines


def detect_decimal_places(s: str) -> int:
    """
    Given a string representation of a number, return how many digits 
    appear after '.'.

      e.g. "123.456" -> 3,  "100" -> 0,  "3.1400" -> 4
    """
    s = s.strip()
    if '.' not in s:
        return 0
    return len(s) - s.index('.') - 1


def find_max_decimals_in_file(
    input_file: str,
    chunk_size: int = 1_000_000,
    bid_col_idx: int = 2,
    ask_col_idx: int = 3,
    price_col_idx: int = 4
) -> int:
    """
    Reads up to `chunk_size` lines from `input_file` (in raw text mode),
    inspects the columns for Bid, Ask, and Price (by index),
    and returns the maximum number of decimal places detected.

    :param input_file: Path to the large input file
    :param chunk_size: Number of lines to read for sampling. Default = 1,000,000
    :param bid_col_idx: 0-based index of the 'Bid' column in the CSV
    :param ask_col_idx: 0-based index of the 'Ask' column in the CSV
    :param price_col_idx: 0-based index of the 'Price' column in the CSV

    :return: The maximum number of decimal places found among Bid, Ask, Price 
             in the sampled lines.
    """
    max_decimals = 0
    lines_read = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            parts = line.strip().split(',')
            if len(parts) < price_col_idx + 1:
                continue  # skip malformed lines

            # Check decimal places for (Bid, Ask, Price)
            for col_idx in (bid_col_idx, ask_col_idx, price_col_idx):
                dec_places = detect_decimal_places(parts[col_idx])
                if dec_places > max_decimals:
                    max_decimals = dec_places

            lines_read += 1
            if lines_read >= chunk_size:
                # Only check up to the first chunk worth of lines
                break

    return max_decimals


def make_float_formatter(max_decimals: int):
    """
    Returns a function that formats floats with up to `max_decimals` places,
    and strips trailing zeros (and any trailing dot if it becomes unnecessary).
    """
    def float_formatter(x):
        if pd.isna(x):
            return ""
        # Format with fixed decimal places, then strip trailing zeros
        formatted = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
        return formatted
    return float_formatter


def aggregate_by_second(df: pd.DataFrame, max_decimals: int) -> pd.DataFrame:
    """
    Given a DataFrame with columns: [Date, Time, Bid, Ask, Price, Size],
    aggregate by (Date, Time) to compute:
      - BidOpen, BidHigh, BidLow, BidClose
      - AskOpen, AskHigh, AskLow, AskClose
      - Open, High, Low, Close (from Price)
      - Volume = sum of Size
      - VWAP = sum(Price * Size) / sum(Size), 
               (use Close if Volume=0 to avoid divide-by-zero).

    Rounds float columns to `max_decimals`.

    Returns a DataFrame in sorted order with one row per (Date, Time).
    """
    if df.empty:
        return df

    # Compute Price*Size for VWAP
    df["PxSize"] = df["Price"] * df["Size"]

    grouped = df.groupby(["Date", "Time"], as_index=False, sort=False).agg({
        "Bid":   ["first", "max", "min", "last"],
        "Ask":   ["first", "max", "min", "last"],
        "Price": ["first", "max", "min", "last"],
        "Size":  "sum",
        "PxSize":"sum",
    })

    grouped.columns = [
        "Date", "Time",
        "BidOpen", "BidHigh", "BidLow", "BidClose",
        "AskOpen", "AskHigh", "AskLow", "AskClose",
        "Open", "High", "Low", "Close",
        "Volume", "PxSizeSum"
    ]

    # Ensure numeric types
    grouped["PxSizeSum"] = grouped["PxSizeSum"].astype(float)
    grouped["Volume"] = grouped["Volume"].astype(float)
    grouped["Close"] = grouped["Close"].astype(float)

    # 1) Initialize VWAP = Close
    grouped["VWAP"] = grouped["Close"].astype(float)

    # 2) For rows where Volume != 0, set VWAP to sum(Price * Size)/Volume
    nonzero_mask = grouped["Volume"] != 0
    grouped.loc[nonzero_mask, "VWAP"] = (
        grouped.loc[nonzero_mask, "PxSizeSum"].astype(float)
        / grouped.loc[nonzero_mask, "Volume"].astype(float)
    ).astype(float)

    # Optionally convert Volume back to int
    grouped["Volume"] = grouped["Volume"].astype(int)

    grouped.drop(columns=["PxSizeSum"], inplace=True)

    # Round the float columns
    float_cols = [
        "BidOpen", "BidHigh", "BidLow", "BidClose",
        "AskOpen", "AskHigh", "AskLow", "AskClose",
        "Open", "High", "Low", "Close", "VWAP"
    ]
    grouped[float_cols] = grouped[float_cols].round(max_decimals)

    return grouped


###############################################################################
# 2. MAIN FUNCTION
###############################################################################

def aggregate_large_quote_file(
    input_file: str,
    output_file: str,
    chunksize: int = 1_000_000
):
    """
    Reads a large text file of quotes/trades in (Date, Time, Bid, Ask, Price, Size) format,
    aggregates them by (Date, Time) second, and writes the aggregated rows to `output_file`.
    Additionally:
      - Detects the max # of decimal digits for (Bid, Ask, Price) from a sample 
        of up to `chunksize` lines (raw read).
      - Applies a custom float formatter to strip trailing zeros.
      - Handles chunk boundaries properly so that lines sharing the last 
        (Date, Time) are not split across multiple aggregates.

    :param input_file: Path to the large input file (no header, comma separated):
                       Date (DD/MM/YY), Time (HH24:MM:SS), Bid, Ask, Price, Size
    :param output_file: Path to the output file (will have aggregated data).
    :param chunksize: Number of lines to process in each chunk. Default 1,000,000.
    """

    # ------------------------------------------------------------------------
    # (A) Detect max decimal places
    # ------------------------------------------------------------------------
    max_decimals = find_max_decimals_in_file(input_file, chunk_size=chunksize)
    print(f"Detected max decimals: {max_decimals}")

    # Create the float formatter
    float_formatter = make_float_formatter(max_decimals)

    # -----------------------------------------------------------------------
    # (B) Count lines in the input file
    # -----------------------------------------------------------------------
    total_lines = count_lines(input_file)
    num_chunks = (total_lines + chunksize - 1) // chunksize

    # ------------------------------------------------------------------------
    # (C) Write the output file header
    # ------------------------------------------------------------------------
    with open(output_file, 'w', encoding='utf-8') as f_out:
        header = (
            "Date,Time,"
            "BidOpen,BidHigh,BidLow,BidClose,"
            "AskOpen,AskHigh,AskLow,AskClose,"
            "Open,High,Low,Close,"
            "Volume,VWAP\n"
        )
        f_out.write(header)

    # ------------------------------------------------------------------------
    # (D) Process the file in chunks
    # ------------------------------------------------------------------------
    col_names = ["Date", "Time", "Bid", "Ask", "Price", "Size"]

    partial_df = None  # for leftover rows at chunk boundaries

    pbar = tqdm(total=num_chunks, desc="Processing chunks")

    # Read the entire file from the beginning again
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
            "Size": int
        },
        encoding="utf-8"
    ):
        # If partial_df is None, build it from chunk's schema
        if partial_df is None:
            partial_df = chunk.iloc[:0].copy()

        # Combine leftover rows from previous iteration with this chunk
        combined = pd.concat([partial_df, chunk], ignore_index=True)
        if combined.empty:
            partial_df = chunk.iloc[:0].copy()
            continue

        # Identify the last date/time in combined
        combined["DateTime"] = combined["Date"] + " " + combined["Time"]
        last_dt = combined.iloc[-1]["DateTime"]

        # The rows that share the last_dt get carried over to partial_df
        mask_partial = (combined["DateTime"] == last_dt)
        partial_df = combined[mask_partial].copy()
        to_aggregate = combined[~mask_partial].copy()

        # Remove the helper column
        to_aggregate.drop(columns=["DateTime"], inplace=True, errors="ignore")
        partial_df.drop(columns=["DateTime"], inplace=True, errors="ignore")

        # Aggregate and append to output
        if not to_aggregate.empty:
            aggregated = aggregate_by_second(to_aggregate, max_decimals)
            aggregated.to_csv(
                output_file,
                mode='a',
                header=False,
                index=False,
                float_format=float_formatter
            )

        pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------------------
    # (D) After reading all chunks, handle leftover rows in partial_df
    # ------------------------------------------------------------------------
    if partial_df is not None and not partial_df.empty:
        aggregated = aggregate_by_second(partial_df, max_decimals)
        aggregated.to_csv(
            output_file,
            mode='a',
            header=False,
            index=False,
            float_format=float_formatter
        )

    print("Aggregation complete.")


def download_s3_file(bucket, path, local_base_dir):
    """
    Download a file from S3 to the local directory maintaining folder structure.
    """
    s3 = boto3.client('s3')
    local_path = os.path.join(local_base_dir, path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, path, local_path)
    return local_path


def upload_s3_file(bucket, path, local_base_dir):
    """
    Upload a file from local directory to S3 maintaining folder structure.
    """
    s3 = boto3.client('s3')
    local_path = os.path.join(local_base_dir, path)
    s3.upload_file(local_path, bucket, path)
    return path


def check_s3_file_exists(bucket_name: str, key: str) -> bool:
    """
    Checks if 'key' exists in the given S3 bucket 'bucket_name' 
    
    :param bucket_name: Name of the S3 bucket
    :param key: Path/Key of the file in the S3 bucket
    :return: True if file exists, False otherwise
    """
    s3_client = boto3.client("s3")

    # List up to 1 object with matching prefix
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=key,
        MaxKeys=1
    )

    # If 'Contents' is in the response, check if the returned key matches exactly
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'] == key:
                return True

    return False

###############################################################################
# EXAMPLE USAGE (comment out in production)
###############################################################################

if __name__ == "__main__":
    print(check_s3_file_exists('kibotdata', 'futures/1min/ECD.zip'))
#     input_file_path = "IVE_tickbidask.txt"
#     output_file_path = "IVE_bidask1sec.txt"
#     aggregate_large_quote_file(input_file_path, output_file_path)
