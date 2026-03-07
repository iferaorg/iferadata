"""Small pandas-like shim used by tests during the Polars migration."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable

import polars as pl


def _to_datetime(value: Any) -> datetime:
    """Convert common timestamp inputs to a datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported timestamp value: {type(value)!r}")


def _to_date(value: Any) -> date:
    """Convert common date-like values to a date."""
    return _to_datetime(value).date()


class DatetimeIndex(list[date]):
    """Minimal replacement for pandas.DatetimeIndex used in tests."""

    def __init__(self, values: Iterable[Any], name: str | None = None):
        super().__init__(_to_date(value) for value in values)
        self.name = name


def Timestamp(value: Any) -> datetime:
    """Minimal replacement for pandas.Timestamp used in tests."""
    return _to_datetime(value)


def Timedelta(*args: Any, **kwargs: Any) -> timedelta:
    """Minimal replacement for pandas.Timedelta used in tests."""
    return timedelta(*args, **kwargs)


def DataFrame(
    data: dict[str, Any] | None = None,
    index: Iterable[Any] | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Create a Polars DataFrame while supporting pandas-style index input."""
    frame = pl.DataFrame(data or {}, **kwargs)

    if index is not None:
        index_values = [_to_date(value) for value in index]
        if frame.height not in (0, len(index_values)):
            raise ValueError(
                "Length of index does not match number of rows in DataFrame data"
            )
        payload = {"date": index_values, **frame.to_dict(as_series=False)}
        frame = pl.DataFrame(payload)

    if "date" in frame.columns:
        frame = frame.with_columns(pl.col("date").cast(pl.Date, strict=False))

    return frame
