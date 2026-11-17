"""Select an explicit strategy date range without dropping idle market sessions."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl

from ._data import StrategyData


def select_strategy_dates(
    data: StrategyData, start_date: date | None = None, end_date: date | None = None
) -> StrategyData:
    """Restrict a ledger and its full market calendar to inclusive date bounds.

    Preserve source row order and fields, then remap all option-leg identities to
    contiguous selected row offsets. Idle sessions at the requested boundaries
    remain in the calendar. Neither input frames nor the source CSV are changed.
    """
    if start_date is None and end_date is None:
        return data
    for name, value in (("start_date", start_date), ("end_date", end_date)):
        if value is not None and (
            not isinstance(value, date) or isinstance(value, datetime)
        ):
            raise TypeError(f"{name} must be a date or None")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    predicate = pl.lit(True)
    opened_date = pl.col("opened").dt.date()
    if start_date is not None:
        predicate &= opened_date >= start_date
    if end_date is not None:
        predicate &= opened_date <= end_date
    selected = data.trades.filter(predicate).sort("trade_index")
    if selected.is_empty():
        raise ValueError("The selected date range contains no strategy trades")
    original_indices = selected["trade_index"].to_list()
    index_mapping = {old: new for new, old in enumerate(original_indices)}
    trades = selected.with_columns(
        pl.col("trade_index").replace_strict(index_mapping).alias("trade_index")
    )
    legs = data.legs.filter(pl.col("trade_index").is_in(original_indices)).with_columns(
        pl.col("trade_index").replace_strict(index_mapping).alias("trade_index")
    )
    dates = tuple(
        session
        for session in data.dates
        if (start_date is None or session >= start_date)
        and (end_date is None or session <= end_date)
    )
    return StrategyData(data.raw[original_indices], trades, legs, dates, data.path)
