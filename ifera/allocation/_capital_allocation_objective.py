"""Objective selection and calendar-based minimum growth constraints."""

from __future__ import annotations

import math
import operator
from numbers import Real


def resolve_objective(
    objective: str, cmgr_limit: float | None, calendar_days: int | None
) -> tuple[bool, float | None]:
    """Return the drawdown-only flag and optional minimum historical log growth.

    Calendar days use the inclusive observed date span, as in the portfolio
    report. A nonpositive CMGR limit is inactive. No market-day annualization
    is inferred from a tensor's row count.
    """
    if not isinstance(objective, str):
        raise TypeError("objective must be 'kelly' or 'minimize_add'")
    if objective not in ("kelly", "minimize_add"):
        raise ValueError("objective must be 'kelly' or 'minimize_add'")
    if cmgr_limit is not None:
        if isinstance(cmgr_limit, bool) or not isinstance(cmgr_limit, Real):
            raise TypeError("CMGR_limit must be a real number or None")
        cmgr_limit = float(cmgr_limit)
        if not math.isfinite(cmgr_limit):
            raise ValueError("CMGR_limit must be finite")
    days = _validate_calendar_days(calendar_days)
    minimum_log_growth = None
    if cmgr_limit is not None and cmgr_limit > 0.0:
        if days is None:
            raise ValueError("calendar_days is required when CMGR_limit is positive")
        try:
            monthly_periods = days / 365.25 * 12
            minimum_log_growth = math.log1p(cmgr_limit) * monthly_periods
        except OverflowError as exc:
            raise ValueError("calendar_days is too large") from exc
        if not math.isfinite(minimum_log_growth):
            raise ValueError("CMGR_limit and calendar_days imply excessive log growth")
    return objective == "minimize_add", minimum_log_growth


def _validate_calendar_days(value: int | None) -> int | None:
    """Validate an optional inclusive observed calendar duration."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("calendar_days must be an integer or None")
    try:
        days = operator.index(value)
    except TypeError as exc:
        raise TypeError("calendar_days must be an integer or None") from exc
    if days <= 0:
        raise ValueError("calendar_days must be positive")
    return days
