"""Calculate daily indicators with causal, independently replaceable last bars."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from . import _indicators

if TYPE_CHECKING:
    from ._features import FeatureSpec

_CALCULATORS: dict[str, Callable[..., _indicators.FloatArray]] = {
    "rsi": _indicators.rsi,
    "stoch": _indicators.stochastic,
    "cci": _indicators.cci,
    "adx": _indicators.adx,
    "momentum": _indicators.momentum,
    "macd": _indicators.macd,
    "stoch_rsi": _indicators.stoch_rsi,
    "williams_r": _indicators.williams_r,
    "ultimate": _indicators.ultimate,
    "mfi": _indicators.mfi,
    "chande": _indicators.chande,
}
BOOLEAN_INDICATORS = frozenset(
    ("underlying_above_ma", "ma_comparison", "macd_above_signal")
)
INDICATOR_TEMPLATES = frozenset(_CALCULATORS) | BOOLEAN_INDICATORS


def _above(left: _indicators.FloatArray, right: _indicators.FloatArray):
    """Keep missing Boolean indicators unknown instead of turning them false."""
    return np.where(np.isfinite(left) & np.isfinite(right), left > right, np.nan)


def _ma_values(bars: _indicators.FloatArray, template: str, arguments: dict[str, Any]):
    """Compare complete daily close histories with their moving averages."""
    kind = arguments["ma_type"]
    if template == "underlying_above_ma":
        return _above(
            bars[:, -1, 3],
            _indicators.moving_average(bars, arguments["days"], kind),
        )
    return _above(
        _indicators.moving_average(bars, arguments["days1"], kind),
        _indicators.moving_average(bars, arguments["days2"], kind),
    )


def _daily_values(
    bars: _indicators.FloatArray, template: str, arguments: dict[str, Any]
):
    """Dispatch complete-session kernels, preserving indicator missingness."""
    if template in ("underlying_above_ma", "ma_comparison"):
        return _ma_values(bars, template, arguments)
    if template == "macd_above_signal":
        values = _indicators.macd(bars, **arguments, component="histogram")
        return _above(values, np.zeros_like(values))
    return _CALCULATORS[template](bars, **arguments)


def _snapshot(
    history: _indicators.FloatArray,
    current: _indicators.FloatArray,
    template: str,
    arguments: dict[str, Any],
) -> _indicators.FloatArray:
    """Share recursive historical states across all minutes of a session."""
    if template == "adx":
        return _indicators.adx_snapshot(history, current, **arguments)
    if template in ("macd", "macd_above_signal"):
        options = {**arguments}
        if template == "macd_above_signal":
            options["component"] = "histogram"
        values = _indicators.macd_snapshot(history, current, **options)
        return (
            _above(values, np.zeros_like(values))
            if template == "macd_above_signal"
            else values
        )
    kind = arguments["ma_type"]
    if template == "underlying_above_ma":
        average = _indicators.moving_average_snapshot(
            history, current, arguments["days"], kind
        )
        return _above(current[:, 3], average)
    first = _indicators.moving_average_snapshot(
        history, current, arguments["days1"], kind
    )
    second = _indicators.moving_average_snapshot(
        history, current, arguments["days2"], kind
    )
    return _above(first, second)


def _window_size(template: str, arguments: dict[str, Any]) -> int:
    """Include the comparison bar and all nested smoothing warmup where needed."""
    if template == "ultimate":
        return max(arguments["short"], arguments["medium"], arguments["long"]) + 1
    size = arguments["days"]
    if template in ("momentum", "mfi", "chande"):
        size += 1
    if template in ("stoch", "stoch_rsi"):
        size += arguments["smooth_k"] - 1
        if arguments["component"] == "d":
            size += arguments["smooth_d"] - 1
    if template == "stoch_rsi":
        size += arguments["rsi_days"] - 1
    return size


def indicator_feature(
    observations: pl.DataFrame, history: pl.DataFrame, spec: FeatureSpec
) -> _indicators.FloatArray:
    """Use completed prior sessions and only the current minute's partial OHLCV."""
    required = {"date", "open", "high", "low", "close"}
    if not required <= set(history.columns):
        raise ValueError(f"Daily indicator history requires columns {sorted(required)}")
    history = history.sort("date")
    if history["date"].n_unique() != history.height:
        raise ValueError("Daily indicator history must have unique session dates")
    fields = ["open", "high", "low", "close"]
    current_fields = ["day_open", "day_high", "day_low", "underlying_price"]
    if spec.template == "mfi":
        if "volume" not in history.columns:
            history = history.with_columns(pl.lit(float("nan")).alias("volume"))
        if "day_volume" not in observations.columns:
            observations = observations.with_columns(
                pl.lit(float("nan")).alias("day_volume")
            )
        fields.append("volume")
        current_fields.append("day_volume")
    history_dates = np.asarray(history["date"].to_list(), dtype="datetime64[D]")
    bars = history.select(fields).to_numpy().astype(np.float64)
    observation_dates = np.asarray(
        observations["datetime"].dt.date().to_list(), dtype="datetime64[D]"
    )
    values = np.full(observations.height, np.nan)
    arguments = dict(spec.parameters)
    days_ago = arguments.pop("days_ago", 0)
    current_bars = (
        observations.select(current_fields).to_numpy().astype(np.float64)
        if not days_ago
        else np.empty((0, len(current_fields)), dtype=np.float64)
    )
    recursive = spec.template in {"adx", "macd"} | BOOLEAN_INDICATORS
    for session in np.unique(observation_dates):
        indices = np.flatnonzero(observation_dates == session)
        prior_count = int(np.searchsorted(history_dates, session))
        if days_ago:
            end = max(0, prior_count - days_ago + 1)
            if end:
                values[indices] = _daily_values(
                    bars[None, :end], spec.template, arguments
                )[0]
            continue
        current = current_bars[indices]
        if recursive:
            values[indices] = _snapshot(
                bars[:prior_count], current, spec.template, arguments
            )
            continue
        size = _window_size(spec.template, arguments)
        preceding = bars[max(0, prior_count - size + 1) : prior_count]
        previous = np.broadcast_to(preceding, (len(indices), *preceding.shape))
        combined = np.concatenate((previous, current[:, None]), axis=1)
        values[indices] = _daily_values(combined, spec.template, arguments)
    return values
