"""
Data processing functionality for financial data.
"""

import datetime
import zipfile as zip_file
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
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

    if traded_months is not None:
        filtered = [
            (inst, tens)
            for inst, tens in zip(instruments, data)
            if inst.contract_code is None or inst.contract_code[0] in traded_months
        ]
        if not filtered:
            raise ValueError("No contracts match traded_months")
        instruments, data = zip(*filtered)
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

        if next_idx is not None:
            if forced:
                should_roll = True
                target_idx = next_idx
            else:
                within_window = True
                if (
                    forced_date is not None
                    and instruments[current].rollover_max_days is not None
                ):
                    within_window = (forced_date - day) <= instruments[
                        current
                    ].rollover_max_days
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

    return ratios, active_idx, all_days_ord
