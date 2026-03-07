"""Time parsing and interval helper utilities."""

from __future__ import annotations

import datetime as dt
import re
from typing import Final

_UNIT_SECONDS: Final[dict[str, float]] = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
}
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ns|us|ms|s|m|h|d|w)", re.IGNORECASE
)
_ONE_DAY: Final[dt.timedelta] = dt.timedelta(days=1)
_EPSILON: Final[dt.timedelta] = dt.timedelta(microseconds=1)


def parse_timedelta(value: dt.timedelta | int | float | str) -> dt.timedelta:
    """Parse a duration to ``datetime.timedelta``."""
    if isinstance(value, dt.timedelta):
        return value

    if isinstance(value, (int, float)):
        return dt.timedelta(seconds=float(value))

    if not isinstance(value, str):
        raise ValueError(f"Cannot parse timedelta from value {value!r}")

    text = value.strip()
    if text == "":
        raise ValueError("Cannot parse timedelta from empty string")

    parsed_clock = _parse_clock_timedelta(text)
    if parsed_clock is not None:
        return parsed_clock

    parsed_units = _parse_unit_timedelta(text)
    if parsed_units is not None:
        return parsed_units

    raise ValueError(f"Cannot parse timedelta from value {value!r}")


def parse_date(value: dt.date | dt.datetime | str) -> dt.date:
    """Parse a date-like value to ``datetime.date``."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Cannot parse date from value {value!r}")

    text = value.strip()
    if text == "":
        raise ValueError("Cannot parse date from empty string")

    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        try:
            return dt.datetime.fromisoformat(text).date()
        except ValueError as second_error:
            raise ValueError(
                f"Cannot parse date from value {value!r}"
            ) from second_error


def derive_end_time_and_steps(
    *,
    time_step: dt.timedelta,
    trading_start: dt.timedelta,
    trading_end: dt.timedelta,
) -> tuple[dt.timedelta, int]:
    """
    Compute the latest bar offset and number of bars for a trading window.

    Mirrors the previous behavior where valid offsets satisfy:
    ``offset < trading_end - trading_start`` and ``offset <= 1 day``.
    """
    if time_step <= dt.timedelta(0):
        raise ValueError("Invalid time_step: must be positive.")

    trading_window = trading_end - trading_start
    if trading_window <= dt.timedelta(0):
        return dt.timedelta(0), 1

    max_index_by_day = _ONE_DAY // time_step
    max_index_by_window = (trading_window - _EPSILON) // time_step
    max_index = min(max_index_by_day, max_index_by_window)

    if max_index < 0:
        return dt.timedelta(0), 1

    end_time = max_index * time_step
    total_steps = int(end_time / time_step) + 1
    return end_time, total_steps


def timedelta_is_multiple(
    *, child: dt.timedelta, parent: dt.timedelta, allow_equal: bool = True
) -> bool:
    """Return whether ``child`` is an integer multiple of ``parent``."""
    if parent <= dt.timedelta(0) or child <= dt.timedelta(0):
        return False
    if not allow_equal and child == parent:
        return False

    parent_us = timedelta_to_microseconds(parent)
    child_us = timedelta_to_microseconds(child)
    if parent_us <= 0:
        return False
    return child_us % parent_us == 0


def timedelta_to_microseconds(value: dt.timedelta) -> int:
    """Convert a timedelta to integer microseconds."""
    return value.days * 86_400_000_000 + value.seconds * 1_000_000 + value.microseconds


def _parse_clock_timedelta(text: str) -> dt.timedelta | None:
    sign = 1
    raw = text
    if raw[0] in {"+", "-"}:
        sign = -1 if raw[0] == "-" else 1
        raw = raw[1:]

    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        return None

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2]) if len(parts) == 3 else 0.0
    except ValueError:
        return None

    if minutes < 0 or minutes > 59:
        return None
    if seconds < 0 or seconds >= 60:
        return None

    parsed = dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return sign * parsed


def _parse_unit_timedelta(text: str) -> dt.timedelta | None:
    sign = 1
    raw = text.replace(" ", "")
    if raw[0] in {"+", "-"}:
        sign = -1 if raw[0] == "-" else 1
        raw = raw[1:]

    if raw == "":
        return None

    matches = list(_TOKEN_PATTERN.finditer(raw))
    if not matches:
        return None

    consumed = "".join(match.group(0) for match in matches)
    if consumed.lower() != raw.lower():
        return None

    total_seconds = 0.0
    for match in matches:
        amount = float(match.group("value"))
        unit = match.group("unit").lower()
        total_seconds += amount * _UNIT_SECONDS[unit]

    return dt.timedelta(seconds=sign * total_seconds)
