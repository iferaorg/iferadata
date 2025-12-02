"""
Data processing functionality for financial data.
"""

import datetime
import zipfile as zip_file
from typing import Optional, List, Tuple

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .config import BaseInstrumentConfig, parse_datetime, timedelta_range
from .enums import Source
from .file_utils import make_instrument_path

SECONDS_IN_DAY = 86400


def add_missing_rows(
    group: pl.DataFrame,
    start_time: datetime.timedelta,
    end_time: datetime.timedelta,
    time_step: datetime.timedelta,
) -> pl.DataFrame:
    """Add missing rows for each time step in a group."""
    all_time_steps = timedelta_range(start=start_time, end=end_time, freq=time_step)
    
    # Convert timedeltas to total_seconds for polars
    all_time_steps_seconds = [int(td.total_seconds()) for td in all_time_steps]
    trade_date_val = group["trade_date"][0]
    
    all_time_step_rows = pl.DataFrame({
        "trade_date": [trade_date_val] * len(all_time_steps_seconds),
        "offset_time": all_time_steps_seconds,
        "open": [None] * len(all_time_steps_seconds),
        "high": [None] * len(all_time_steps_seconds),
        "low": [None] * len(all_time_steps_seconds),
        "close": [None] * len(all_time_steps_seconds),
        "volume": [None] * len(all_time_steps_seconds),
    })
    
    # Merge and sort
    merged = group.join(
        all_time_step_rows,
        on=["trade_date", "offset_time"],
        how="full",
        suffix="_y"
    ).sort(["trade_date", "offset_time"])
    
    # Forward fill close
    merged = merged.with_columns([
        pl.col("close").forward_fill().alias("close")
    ])
    
    # Fill remaining nulls in close with first open value
    first_open = group["open"][0]
    merged = merged.with_columns([
        pl.col("close").fill_null(first_open).alias("close")
    ])
    
    # Backward fill open, high, low from close
    merged = merged.with_columns([
        pl.when(pl.col("open").is_null())
        .then(pl.col("close"))
        .otherwise(pl.col("open"))
        .alias("open"),
        
        pl.when(pl.col("high").is_null())
        .then(pl.col("close"))
        .otherwise(pl.col("high"))
        .alias("high"),
        
        pl.when(pl.col("low").is_null())
        .then(pl.col("close"))
        .otherwise(pl.col("low"))
        .alias("low"),
    ])
    
    # Fill volume nulls with 0
    merged = merged.with_columns([
        pl.col("volume").fill_null(0).cast(pl.Int32).alias("volume")
    ])
    
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
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return ""
        formatted = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
        return formatted

    return float_formatter


def aggregate_by_second(df: pl.DataFrame, max_decimals: int) -> pl.DataFrame:
    """Aggregate data by second, computing OHLCV and VWAP."""
    if df.is_empty():
        return df
    
    df = df.with_columns([
        (pl.col("Price") * pl.col("Size")).alias("PxSize")
    ])
    
    grouped = df.group_by(["Date", "Time"], maintain_order=True).agg([
        pl.col("Bid").first().alias("BidOpen"),
        pl.col("Bid").max().alias("BidHigh"),
        pl.col("Bid").min().alias("BidLow"),
        pl.col("Bid").last().alias("BidClose"),
        pl.col("Ask").first().alias("AskOpen"),
        pl.col("Ask").max().alias("AskHigh"),
        pl.col("Ask").min().alias("AskLow"),
        pl.col("Ask").last().alias("AskClose"),
        pl.col("Price").first().alias("Open"),
        pl.col("Price").max().alias("High"),
        pl.col("Price").min().alias("Low"),
        pl.col("Price").last().alias("Close"),
        pl.col("Size").sum().alias("Volume"),
        pl.col("PxSize").sum().alias("PxSizeSum"),
    ])
    
    # Cast to appropriate types
    grouped = grouped.with_columns([
        pl.col("PxSizeSum").cast(pl.Float64),
        pl.col("Volume").cast(pl.Float64),
        pl.col("Close").cast(pl.Float64),
    ])
    
    # Calculate VWAP
    grouped = grouped.with_columns([
        pl.when(pl.col("Volume") != 0)
        .then(pl.col("PxSizeSum") / pl.col("Volume"))
        .otherwise(pl.col("Close"))
        .alias("VWAP")
    ])
    
    # Cast volume to int
    grouped = grouped.with_columns([
        pl.col("Volume").cast(pl.Int64)
    ])
    
    # Drop PxSizeSum
    grouped = grouped.drop("PxSizeSum")
    
    # Round float columns
    float_cols = [
        "BidOpen", "BidHigh", "BidLow", "BidClose",
        "AskOpen", "AskHigh", "AskLow", "AskClose",
        "Open", "High", "Low", "Close", "VWAP",
    ]
    grouped = grouped.with_columns([
        pl.col(col).round(max_decimals).alias(col) for col in float_cols
    ])
    
    return grouped


def process_chunk(
    chunk: pl.DataFrame, partial_df: pl.DataFrame, max_decimals: int
) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Process a chunk of data, aggregating complete seconds and carrying over incomplete data."""
    if partial_df is None:
        combined = chunk
    else:
        combined = pl.concat([partial_df, chunk])

    if combined.is_empty():
        return None, partial_df  # type: ignore

    last_row = combined.tail(1)
    last_dt = last_row["Date"][0] + " " + last_row["Time"][0]
    
    combined = combined.with_columns([
        (pl.col("Date") + " " + pl.col("Time")).alias("datetime_str")
    ])
    
    to_aggregate = combined.filter(pl.col("datetime_str") != last_dt).drop("datetime_str")
    new_partial_df = combined.filter(pl.col("datetime_str") == last_dt).drop("datetime_str")

    if not to_aggregate.is_empty():
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
    
    # Read the entire file with polars (polars is more efficient with large files)
    print("Reading file...")
    df = pl.read_csv(
        input_file,
        has_header=False,
        new_columns=["Date", "Time", "Bid", "Ask", "Price", "Size"],
        schema_overrides={
            "Date": pl.Utf8,
            "Time": pl.Utf8,
            "Bid": pl.Float64,
            "Ask": pl.Float64,
            "Price": pl.Float64,
            "Size": pl.Int64,
        },
    )
    
    # Write header
    with open(output_file, "w", encoding="utf-8") as f_out:
        header = (
            "Date,Time,"
            "BidOpen,BidHigh,BidLow,BidClose,"
            "AskOpen,AskHigh,AskLow,AskClose,"
            "Open,High,Low,Close,"
            "Volume,VWAP\n"
        )
        f_out.write(header)
    
    print("Aggregating by second...")
    aggregated = aggregate_by_second(df, max_decimals)
    
    # Write to CSV
    aggregated.write_csv(output_file, has_header=False, append=True)
    
    print("Aggregation complete.")


def calculate_time_columns(
    df: pl.DataFrame, instrument: BaseInstrumentConfig
) -> pl.DataFrame:
    """Calculate and assign datetime-related columns to the DataFrame."""
    # Assuming df has a 'date_time' column from the raw data loading
    trading_start_seconds = int(instrument.trading_start.total_seconds())
    
    df = df.with_columns([
        # Extract date
        pl.col("date_time").dt.date().alias("date"),
        
        # Calculate time in seconds from start of day
        (pl.col("date_time").dt.hour() * 3600 + 
         pl.col("date_time").dt.minute() * 60 + 
         pl.col("date_time").dt.second()).alias("time_seconds"),
    ])
    
    # Calculate offset_time as seconds from trading_start
    df = df.with_columns([
        (pl.col("time_seconds") - trading_start_seconds).alias("offset_time_seconds")
    ])
    
    # Normalize offset_time to be within 0 to SECONDS_IN_DAY
    df = df.with_columns([
        (pl.col("offset_time_seconds") % SECONDS_IN_DAY).alias("offset_time_seconds")
    ])
    
    # Calculate trade_date (the date adjusted for trading_start)
    df = df.with_columns([
        pl.when(pl.col("time_seconds") < trading_start_seconds)
        .then(pl.col("date") - pl.duration(days=1))
        .otherwise(pl.col("date"))
        .alias("trade_date")
    ])
    
    return df


def process_group(
    group: pl.DataFrame, instrument: BaseInstrumentConfig
) -> pl.DataFrame:
    """Process a group by adding missing rows based on instrument settings."""
    return add_missing_rows(
        group,
        start_time=datetime.timedelta(0),
        end_time=instrument.end_time,
        time_step=instrument.time_step,
    )


def perform_final_calculations(
    df: pd.DataFrame, instrument: BaseInstrumentConfig
) -> pd.DataFrame:
    """Perform final calculations for ordinal dates and time seconds."""
    # Handle empty DataFrame case
    if df.empty:
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "date",
                "time",
                "offset_time",
                "trade_date",
                "ord_date",
                "time_seconds",
                "ord_trade_date",
                "offset_time_seconds",
            ]
        ).astype(
            {
                "open": df.dtypes["open"],
                "high": df.dtypes["high"],
                "low": df.dtypes["low"],
                "close": df.dtypes["close"],
                "volume": df.dtypes["volume"],
                "date": "datetime64[ns]",
                "time": "timedelta64[ns]",
                "offset_time": "timedelta64[ns]",
                "trade_date": "datetime64[ns]",
                "ord_date": "int64",
                "time_seconds": "float64",
                "ord_trade_date": "int64",
                "offset_time_seconds": "float64",
            }
        )

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
    print("Converting datetime columns...")
    df = calculate_time_columns(df, instrument)

    start_date_ts = pd.to_datetime(instrument.start_date)
    df = df[df["trade_date"] >= start_date_ts]

    if instrument.remove_dates is not None and len(instrument.remove_dates) > 0:
        remove_dates_ts = pd.to_datetime(instrument.remove_dates)
        df = df[~df["trade_date"].isin(remove_dates_ts)]

    if instrument.days_of_week is not None and len(instrument.days_of_week) > 0:
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
        if df.empty:
            # Create zip file with empty CSV file containing header
            with zip_file.ZipFile(str(output_path), "w", zip_file.ZIP_DEFLATED) as zf:
                csv_name = output_path.stem + ".csv"
                empty_csv_content = ""
                zf.writestr(csv_name, empty_csv_content)
        else:
            df.to_csv(str(output_path), header=False, index=False, compression="zip")
    else:
        df.to_csv(str(output_path), header=False, index=False)
    print(f"Processed data saved to {output_path}")


def _last_business_day(
    date_: datetime.date,
    trading_days_ord: set[int],
) -> datetime.date:
    """
    Return the most-recent calendar day **≤ `date_`** that is present in the
    dataset ( `trading_days_ord` is a set of ordinal dates that *do* trade).

    Because public-holidays or other outages may remove otherwise-valid
    weekdays, we walk backwards one day at a time until we find a match.

    Parameters
    ----------
    date_ : datetime.date
        Anchor date (inclusive).
    trading_days_ord : set[int]
        All trading days in the dataset (as `date.toordinal()` integers).

    Returns
    -------
    datetime.date
        The last trading day **on or before** `date_`.
    """
    # Check if there are any trading days before the given date
    if not trading_days_ord or date_.toordinal() < min(trading_days_ord):
        return date_

    cur = date_
    while cur.toordinal() not in trading_days_ord:
        cur -= datetime.timedelta(days=1)
    return cur


def _forced_roll_date(instr, trading_days_ord: set[int]) -> int | None:
    """
    Return the ordinal of the last trading day **before**
    min(instr.expiration_date, instr.first_notice_date).

    If either date is missing, fall back to the one that is present.
    If both are missing, return None (no forced roll rule).
    """
    exp_dt = instr.expiration_date
    fnd_dt = instr.first_notice_date

    max_dt_ord = None
    max_dt = None

    if len(trading_days_ord) > 0:
        max_dt_ord = max(trading_days_ord)
        max_dt = datetime.date.fromordinal(max_dt_ord)

    # Nothing to enforce
    if exp_dt is None and fnd_dt is None:
        return max_dt_ord

    # Choose the earlier of the two that exist
    if exp_dt is None:
        anchor = fnd_dt
    elif fnd_dt is None:
        anchor = exp_dt
    else:
        anchor = min(exp_dt, fnd_dt)  # the “critical” date

    if max_dt is not None and anchor > max_dt:
        return max_dt_ord

    # Step back to the previous trading day actually present in the data
    prev_trade_day = _last_business_day(
        anchor - datetime.timedelta(days=1), trading_days_ord
    )
    return prev_trade_day.toordinal()


def calculate_rollover(
    instruments: List[BaseInstrumentConfig],
    data: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Calculate rollover multipliers for futures contracts.

    Args:
        instruments: List of instrument configurations for futures contracts
        data: List of data tensors corresponding to each instrument

    Returns:
        Tuple containing:
        - Combined data tensor with rollover multipliers applied
        - Multiplier tensor showing the rollover adjustments
        - List of indices indicating start of each instrument's data

    Raises:
        ValueError: If instruments list is empty or data is inconsistent
    """
    if not instruments:
        raise ValueError("`instruments` may not be empty")

    # Common dtype / device --------------------------------------------------
    device = data[0].device
    dtype = data[0].dtype

    OFFSET_CH = 3  # offset_time_seconds
    ORD_TRD_CH = 2  # ord_trade_date
    OPEN_CH = 4  # open
    VOL_CH = 8  # volume

    cut_off = instruments[0].rollover_offset
    end_time = int(instruments[0].end_time.total_seconds())
    alpha_raw = instruments[0].rollover_vol_alpha
    alpha = alpha_raw if alpha_raw is not None else 1.0
    start_ord = instruments[0].start_date.toordinal()

    traded_months = instruments[0].traded_months
    index_map = list(range(len(instruments)))

    if traded_months is not None:
        filtered: list[tuple[BaseInstrumentConfig, torch.Tensor]] = []
        index_map = []
        for idx, (inst, tens) in enumerate(zip(instruments, data)):
            if inst.contract_code is None or inst.contract_code[0] in traded_months:
                filtered.append((inst, tens))
                index_map.append(idx)
        if not filtered:
            raise ValueError("No contracts match traded_months")
        instruments, data = zip(*filtered)  # type: ignore
        instruments = list(instruments)
        data = list(data)

    # Pre-compute: per contract  ➜  dict( ord_trade_date → (volume<15:30>, open15:30) )
    contract_day_stats: list[dict[int, tuple[float, float]]] = []

    for tens in tqdm(data, desc="Pre-processing contracts"):
        n_days, _, _ = tens.shape
        stats: dict[int, tuple[float, float]] = {}

        for d in range(n_days):
            # --- basic day-level arrays ---------------------------------------
            ord_td = int(tens[d, 0, ORD_TRD_CH].item())

            if ord_td < start_ord - 1:
                continue

            times_d = tens[d, :, OFFSET_CH]
            vols_d = tens[d, :, VOL_CH]

            # ①  Current-day volume up to (but **not incl.**) 15 : 30
            v_up_to_cutoff = vols_d[times_d < cut_off].sum().item()

            # ②  Previous-trading-day volume **after** 15 : 30
            v_prev_after_cutoff = 0.0
            if d > 0:  # not available on very first day
                times_prev = tens[d - 1, :, OFFSET_CH]
                vols_prev = tens[d - 1, :, VOL_CH]
                v_prev_after_cutoff = (
                    vols_prev[(times_prev >= cut_off) & (times_prev <= end_time)]
                    .sum()
                    .item()
                )

            # ③  Full 24-h liquidity snapshot
            liq_vol = v_prev_after_cutoff + v_up_to_cutoff

            # ④  15 : 30 price (needed for ratio when we roll)
            mask_cutoff = (times_d == cut_off).nonzero(as_tuple=False)
            if mask_cutoff.numel() != 1:
                raise RuntimeError("15 : 30 bar missing (or duplicated) in dataset")
            idx_cutoff: int = int(mask_cutoff.item())
            price_cutoff = tens[d, idx_cutoff, OPEN_CH].item()

            stats[ord_td] = (liq_vol, price_cutoff)

        contract_day_stats.append(stats)

    # Build the master calendar ---------------------------------------------
    all_days_ord_full = sorted(
        {day for stats in contract_day_stats for day in stats.keys()}
    )

    contract_forced_roll: list[int | None] = []

    for inst, tens in zip(instruments, data):
        inst_trading_days_ord = tens[:, 0, ORD_TRD_CH].unique().int().tolist()
        contract_forced_roll.append(_forced_roll_date(inst, set(inst_trading_days_ord)))

    data = []  # free memory
    filtered = []
    all_days_ord = [d for d in all_days_ord_full if d >= start_ord]
    n_days = len(all_days_ord)

    # Output tensors (initialised with NaNs) ---------------------------------
    ratios = torch.full((n_days,), float("nan"), dtype=dtype, device=device)
    active_idx = torch.zeros((n_days,), dtype=torch.int64, device=device)

    # Pick the **first** active contract: highest volume on first calendar day
    first_day = all_days_ord[0]
    vols_first = [stats.get(first_day, (0.0, 0.0))[0] for stats in contract_day_stats]
    current = int(torch.tensor(vols_first).argmax().item())

    # -----------------------------------------------------------------------
    for pos, day in enumerate(all_days_ord):
        active_idx[pos] = current

        if pos == n_days - 1:
            # Last day, no need to roll
            ratios[pos] = float("nan")
            continue

        # Volume today for active contract (0 if not trading any more)
        cur_vol, cur_open = contract_day_stats[current].get(day, (0.0, float("nan")))

        # Forced roll?
        forced_date = contract_forced_roll[current]
        forced = forced_date is not None and day >= forced_date

        # Look-ahead volume: next contract only -----------------------------
        next_idx = current + 1 if current + 1 < len(instruments) else None
        next_vol = 0.0
        if next_idx is not None:
            next_vol, _ = contract_day_stats[next_idx].get(day, (0.0, float("nan")))

        # Decide to roll -----------------------------------------------------
        should_roll = False
        target_idx = current

        if next_idx is not None and day in contract_day_stats[next_idx]:
            if forced:
                should_roll = True
                target_idx = next_idx
            else:
                within_window = True
                rollover_max_days = instruments[current].rollover_max_days
                if forced_date is not None and rollover_max_days is not None:
                    within_window = (forced_date - day) <= rollover_max_days
                if within_window and cur_vol * alpha < next_vol:
                    should_roll = True
                    target_idx = next_idx

        # Execute roll (record the ratio) -----------------------------------
        if should_roll and target_idx != current:
            new_open = contract_day_stats[target_idx][day][1]
            if not np.isnan(cur_open) and cur_open != 0:
                ratios[pos] = new_open / cur_open
            active_idx[pos] = target_idx
            current = target_idx

    if traded_months is not None:
        map_tensor = torch.tensor(index_map, dtype=torch.int64, device=device)
        active_idx = map_tensor[active_idx]

    return ratios, active_idx, all_days_ord
