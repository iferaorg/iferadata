"""Daily technical indicators with independently replaceable intraday final bars.

Each input row is a separate price history, so its final bar can represent a
minute snapshot without including later prices. RSI follows the requested
14-price convention (13 changes), using mean gains and losses in that window.
ADX and MACD use Wilder and exponential smoothing respectively.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _mean_windows(values: FloatArray, size: int) -> FloatArray:
    """Return full-length simple moving means, with NaN during warmup."""
    result = np.full_like(values, np.nan)
    if values.shape[1] >= size:
        windows = np.lib.stride_tricks.sliding_window_view(values, size, axis=1)
        result[:, size - 1 :] = windows.mean(axis=-1)
    return result


def _exponential_mean(values: FloatArray, size: int, alpha: float) -> FloatArray:
    """Seed a recursive moving mean with the first finite complete window."""
    result = np.full_like(values, np.nan)
    if values.shape[0] == 1:
        running_scalar = math.nan
        for index in range(size - 1, values.shape[1]):
            running_scalar = (
                alpha * float(values[0, index]) + (1.0 - alpha) * running_scalar
                if math.isfinite(running_scalar)
                else float(values[0, index - size + 1 : index + 1].mean())
            )
            result[0, index] = running_scalar
        return result
    running = np.full(values.shape[0], np.nan)
    for index in range(size - 1, values.shape[1]):
        window = values[:, index - size + 1 : index + 1]
        seed = window.mean(axis=1)
        running = np.where(
            np.isfinite(running),
            alpha * values[:, index] + (1.0 - alpha) * running,
            seed,
        )
        result[:, index] = running
    return result


def _advance_mean(
    history: FloatArray, current: FloatArray, size: int, alpha: float, previous: float
) -> FloatArray:
    """Advance a shared historical moving mean for independent current values."""
    if history.size + 1 < size:
        return np.full_like(current, np.nan)
    if math.isfinite(previous):
        return alpha * current + (1.0 - alpha) * previous
    needed = size - 1
    historical_sum = history[-needed:].sum() if needed else 0.0
    return (historical_sum + current) / size


def rsi(bars: FloatArray, days: int) -> FloatArray:
    """Return RSI from exactly ``days`` closing prices (``days - 1`` changes)."""
    if bars.shape[1] < days:
        return np.full(bars.shape[0], np.nan)
    changes = np.diff(bars[:, -days:, 3], axis=1)
    gains = np.maximum(changes, 0.0).mean(axis=1)
    losses = np.maximum(-changes, 0.0).mean(axis=1)
    total = gains + losses
    return np.divide(
        100.0 * gains, total, out=np.full_like(total, 50.0), where=total > 0
    )


def stochastic(
    bars: FloatArray, days: int, smooth_k: int, smooth_d: int, component: str
) -> FloatArray:
    """Return the slow stochastic K or D after its complete smoothing warmup."""
    close = bars[:, :, 3]
    raw = np.full_like(close, np.nan)
    if bars.shape[1] >= days:
        highs = np.lib.stride_tricks.sliding_window_view(bars[:, :, 1], days, axis=1)
        lows = np.lib.stride_tricks.sliding_window_view(bars[:, :, 2], days, axis=1)
        highest = highs.max(axis=-1)
        lowest = lows.min(axis=-1)
        width = highest - lowest
        raw[:, days - 1 :] = np.divide(
            100.0 * (close[:, days - 1 :] - lowest),
            width,
            out=np.full_like(width, 50.0),
            where=width > 0,
        )
    slow_k = _mean_windows(raw, smooth_k)
    series = slow_k if component == "k" else _mean_windows(slow_k, smooth_d)
    return series[:, -1]


def cci(bars: FloatArray, days: int) -> FloatArray:
    """Return the commodity channel index using high, low, and close prices."""
    if bars.shape[1] < days:
        return np.full(bars.shape[0], np.nan)
    typical = bars[:, -days:, 1:].mean(axis=2)
    average = typical.mean(axis=1)
    deviation = np.abs(typical - average[:, None]).mean(axis=1)
    return np.divide(
        typical[:, -1] - average,
        0.015 * deviation,
        out=np.zeros_like(deviation),
        where=deviation > 0,
    )


def _directional_moves(bars: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return true ranges and positive/negative directional price changes."""
    high, low, close = bars[:, :, 1], bars[:, :, 2], bars[:, :, 3]
    up, down = np.diff(high, axis=1), -np.diff(low, axis=1)
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    true_range = np.maximum.reduce(
        (
            high[:, 1:] - low[:, 1:],
            np.abs(high[:, 1:] - close[:, :-1]),
            np.abs(low[:, 1:] - close[:, :-1]),
        )
    )
    return true_range, plus, minus


def _directional_index(
    average_range: FloatArray, plus_average: FloatArray, minus_average: FloatArray
) -> FloatArray:
    """Calculate DX from smoothed ranges and directional changes."""
    plus_di = np.divide(
        plus_average,
        average_range,
        out=np.zeros_like(plus_average),
        where=average_range > 0,
    )
    minus_di = np.divide(
        minus_average,
        average_range,
        out=np.zeros_like(minus_average),
        where=average_range > 0,
    )
    total = plus_di + minus_di
    return np.divide(
        100.0 * np.abs(plus_di - minus_di),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )


def adx(bars: FloatArray, days: int) -> FloatArray:
    """Return Wilder ADX, first available after ``2 * days`` price bars."""
    if bars.shape[1] < 2 * days:
        return np.full(bars.shape[0], np.nan)
    averages = tuple(
        _exponential_mean(values, days, 1.0 / days)
        for values in _directional_moves(bars)
    )
    directional = _directional_index(*averages)
    directional[:, : days - 1] = np.nan
    return _exponential_mean(directional, days, 1.0 / days)[:, -1]


def adx_snapshot(history: FloatArray, current: FloatArray, days: int) -> FloatArray:
    """Compute shared historical ADX state once and advance each minute snapshot."""
    if history.shape[0] + 1 < 2 * days:
        return np.full(current.shape[0], np.nan)
    if days == 1:
        last_bars = np.broadcast_to(history[-1], (current.shape[0], 1, 4))
        return adx(np.concatenate((last_bars, current[:, None]), axis=1), days)
    averages = tuple(
        _exponential_mean(values, days, 1.0 / days)
        for values in _directional_moves(history[None])
    )
    last_bars = np.broadcast_to(history[-1], (current.shape[0], 1, 4))
    changes = _directional_moves(np.concatenate((last_bars, current[:, None]), axis=1))
    advanced = tuple(
        (1.0 - 1.0 / days) * average[0, -1] + change[:, 0] / days
        for average, change in zip(averages, changes)
    )
    historical_dx = _directional_index(*averages)
    historical_dx[:, : days - 1] = np.nan
    current_dx = _directional_index(*advanced)
    previous = _exponential_mean(historical_dx, days, 1.0 / days)[0, -1]
    return _advance_mean(
        historical_dx[0], current_dx, days, 1.0 / days, float(previous)
    )


def momentum(bars: FloatArray, days: int) -> FloatArray:
    """Return the point change from the close ``days`` sessions earlier."""
    if bars.shape[1] <= days:
        return np.full(bars.shape[0], np.nan)
    return bars[:, -1, 3] - bars[:, -days - 1, 3]


def macd(
    bars: FloatArray, fast: int, slow: int, signal: int, component: str
) -> FloatArray:
    """Return the MACD line, signal line, or histogram using SMA-seeded EMAs."""
    close = bars[:, :, 3]
    line = _exponential_mean(close, fast, 2.0 / (fast + 1)) - _exponential_mean(
        close, slow, 2.0 / (slow + 1)
    )
    if component == "line":
        return line[:, -1]
    signal_line = _exponential_mean(line, signal, 2.0 / (signal + 1))
    return (signal_line if component == "signal" else line - signal_line)[:, -1]


def macd_snapshot(
    history: FloatArray,
    current: FloatArray,
    fast: int,
    slow: int,
    signal: int,
    component: str,
) -> FloatArray:
    """Reuse historical EMA states across independent current-minute snapshots."""
    if history.shape[0] + 1 < slow:
        return np.full(current.shape[0], np.nan)
    close = history[:, 3]
    fast_history = _exponential_mean(close[None], fast, 2.0 / (fast + 1))[0]
    slow_history = _exponential_mean(close[None], slow, 2.0 / (slow + 1))[0]
    previous_fast = float(fast_history[-1]) if close.size else math.nan
    previous_slow = float(slow_history[-1]) if close.size else math.nan
    fast_current = _advance_mean(
        close, current[:, 3], fast, 2.0 / (fast + 1), previous_fast
    )
    slow_current = _advance_mean(
        close, current[:, 3], slow, 2.0 / (slow + 1), previous_slow
    )
    line = fast_current - slow_current
    if component == "line":
        return line
    historical_line = fast_history - slow_history
    signal_history = _exponential_mean(
        historical_line[None], signal, 2.0 / (signal + 1)
    )[0]
    previous_signal = float(signal_history[-1]) if signal_history.size else math.nan
    current_signal = _advance_mean(
        historical_line, line, signal, 2.0 / (signal + 1), previous_signal
    )
    return current_signal if component == "signal" else line - current_signal


def stoch_rsi(
    bars: FloatArray,
    days: int = 14,
    rsi_days: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
    component: str = "d",
) -> FloatArray:
    """Return smoothed stochastic RSI using this package's exact-price RSI window.

    RSI uses ``rsi_days`` prices and ``rsi_days - 1`` changes, matching ``rsi``.
    Apply a ``days`` stochastic window to that RSI, then K/D simple smoothing.
    D first needs ``rsi_days + days + smooth_k + smooth_d - 3`` price bars;
    K omits D smoothing. Flat RSI windows return the neutral value 50.
    Unlike TA-Lib, this deliberately retains the package's non-Wilder RSI:
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_STOCHRSI.c
    """
    close = bars[:, :, 3]
    if close.shape[1] < rsi_days:
        return np.full(bars.shape[0], np.nan)
    changes = np.diff(close, axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(changes, rsi_days - 1, axis=1)
    gains = np.maximum(windows, 0.0).mean(axis=-1)
    losses = np.maximum(-windows, 0.0).mean(axis=-1)
    total = gains + losses
    relative = 100.0 * np.divide(
        gains, total, out=np.full_like(total, 0.5), where=total > 0
    )
    relative = np.where(np.isfinite(total), relative, np.nan)
    raw = np.full_like(relative, np.nan)
    if relative.shape[1] >= days:
        ranges = np.lib.stride_tricks.sliding_window_view(relative, days, axis=1)
        highest, lowest = ranges.max(axis=-1), ranges.min(axis=-1)
        width = highest - lowest
        scaled = np.divide(
            100.0 * (relative[:, days - 1 :] - lowest),
            width,
            out=np.full_like(width, 50.0),
            where=width > 0,
        )
        raw[:, days - 1 :] = np.where(np.isfinite(width), scaled, np.nan)
    slow_k = _mean_windows(raw, smooth_k)
    result = slow_k if component == "k" else _mean_windows(slow_k, smooth_d)
    return result[:, -1]


def williams_r(bars: FloatArray, days: int = 14) -> FloatArray:
    """Return Williams %R on [-100, 0], with -50 for a flat high/low window."""
    if bars.shape[1] < days:
        return np.full(bars.shape[0], np.nan)
    window = bars[:, -days:, 1:4]
    highest = window[:, :, 0].max(axis=1)
    lowest = window[:, :, 1].min(axis=1)
    width = highest - lowest
    result = np.divide(
        -100.0 * (highest - window[:, -1, 2]),
        width,
        out=np.full_like(width, -50.0),
        where=width > 0,
    )
    return np.where(np.isfinite(window).all(axis=(1, 2)), result, np.nan)


def ultimate(
    bars: FloatArray, short: int = 7, medium: int = 14, long: int = 28
) -> FloatArray:
    """Return the 4:2:1 weighted buying-pressure / true-range oscillator.

    The longest window needs one extra previous close. Zero true-range windows
    contribute zero, matching TA-Lib's Ultimate Oscillator convention:
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ULTOSC.c
    """
    maximum = max(short, medium, long)
    if bars.shape[1] <= maximum:
        return np.full(bars.shape[0], np.nan)
    window = bars[:, -maximum - 1 :, 1:4]
    previous = window[:, :-1, 2]
    true_low = np.minimum(window[:, 1:, 1], previous)
    true_high = np.maximum(window[:, 1:, 0], previous)
    pressure = window[:, 1:, 2] - true_low
    true_range = true_high - true_low
    result = np.zeros(bars.shape[0], dtype=np.float64)
    for size, weight in ((short, 4.0), (medium, 2.0), (long, 1.0)):
        numerator = pressure[:, -size:].sum(axis=1)
        denominator = true_range[:, -size:].sum(axis=1)
        result += weight * np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    return np.where(np.isfinite(window).all(axis=(1, 2)), result * 100.0 / 7.0, np.nan)


def mfi(bars: FloatArray, days: int = 14) -> FloatArray:
    """Return money-flow index from genuine OHLCV, with neutral flat flows of 50.

    ``days`` money-flow changes require ``days + 1`` prices. The last ``days``
    volumes must all be finite and strictly positive; missing volume returns NaN.
    The comparison bar's volume is unused. Formula reference:
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MFI.c
    """
    if bars.shape[1] <= days or bars.shape[2] < 5:
        return np.full(bars.shape[0], np.nan)
    typical = bars[:, -days - 1 :, 1:4].mean(axis=2)
    volume = bars[:, -days:, 4]
    safe_typical = np.where(np.isfinite(typical), typical, 0.0)
    safe_volume = np.where(np.isfinite(volume) & (volume > 0), volume, 0.0)
    changes = np.diff(safe_typical, axis=1)
    flow = safe_typical[:, 1:] * safe_volume
    positive = np.where(changes > 0, flow, 0.0).sum(axis=1)
    negative = np.where(changes < 0, flow, 0.0).sum(axis=1)
    total = positive + negative
    result = np.divide(
        100.0 * positive,
        total,
        out=np.full_like(total, 50.0),
        where=total > 0,
    )
    valid = (
        np.isfinite(typical).all(axis=1)
        & np.isfinite(volume).all(axis=1)
        & (volume > 0).all(axis=1)
    )
    return np.where(valid, result, np.nan)


def chande(bars: FloatArray, days: int = 9) -> FloatArray:
    """Return Chande momentum oscillator from exactly ``days`` close changes."""
    if bars.shape[1] <= days:
        return np.full(bars.shape[0], np.nan)
    changes = np.diff(bars[:, -days - 1 :, 3], axis=1)
    gains = np.maximum(changes, 0.0).sum(axis=1)
    losses = np.maximum(-changes, 0.0).sum(axis=1)
    total = gains + losses
    result = np.divide(
        100.0 * (gains - losses),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    return np.where(np.isfinite(changes).all(axis=1), result, np.nan)


def _adaptive_mean(values: FloatArray, days: int) -> FloatArray:
    """Return KAMA series with 2/30 smoothing and a previous-close seed.

    A ``days``-change efficiency ratio first exists after ``days + 1`` prices.
    Seeding uses the preceding close as in TA-Lib. Period one is identity:
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_KAMA.c
    """
    if days == 1:
        return np.where(np.isfinite(values), values, np.nan)
    values = np.where(np.isfinite(values), values, np.nan)
    result = np.full_like(values, np.nan)
    if values.shape[1] <= days:
        return result
    changes = np.abs(np.diff(values, axis=1))
    volatility = np.lib.stride_tricks.sliding_window_view(changes, days, axis=1).sum(
        axis=-1
    )
    change = np.abs(values[:, days:] - values[:, :-days])
    efficiency = np.divide(
        change, volatility, out=np.zeros_like(change), where=volatility > 0
    ).clip(0.0, 1.0)
    smoothing = (2.0 / 31.0 + efficiency * (2.0 / 3.0 - 2.0 / 31.0)) ** 2
    valid = np.isfinite(
        np.lib.stride_tricks.sliding_window_view(values, days + 1, axis=1)
    ).all(axis=-1)
    running = np.full(values.shape[0], np.nan)
    for offset, index in enumerate(range(days, values.shape[1])):
        previous = np.where(np.isfinite(running), running, values[:, index - 1])
        running = np.where(
            valid[:, offset],
            previous + smoothing[:, offset] * (values[:, index] - previous),
            np.nan,
        )
        result[:, index] = running
    return result


def _moving_average_type(ma_type: str) -> str:
    """Normalize supported moving-average names before selecting their kernels."""
    kind = ma_type.upper()
    if kind not in ("SMA", "EMA", "TRIMA", "KAMA"):
        raise ValueError("ma_type must be SMA, EMA, TRIMA, or KAMA")
    return kind


def _adaptive_snapshot(close: FloatArray, present: FloatArray, days: int) -> FloatArray:
    """Advance one shared KAMA state across independent current closing prices."""
    if close.size < days or not np.isfinite(close[-days:]).all():
        return np.full_like(present, np.nan)
    series = _adaptive_mean(close[None], days)[0]
    previous = float(series[-1]) if math.isfinite(series[-1]) else float(close[-1])
    safe_present = np.where(np.isfinite(present), present, close[-1])
    volatility = np.abs(np.diff(close[-days:])).sum() + np.abs(safe_present - close[-1])
    change = np.abs(safe_present - close[-days])
    efficiency = np.divide(
        change, volatility, out=np.zeros_like(change), where=volatility > 0
    ).clip(0.0, 1.0)
    smoothing = (2.0 / 31.0 + efficiency * (2.0 / 3.0 - 2.0 / 31.0)) ** 2
    return np.where(
        np.isfinite(present), previous + smoothing * (safe_present - previous), np.nan
    )


def moving_average(bars: FloatArray, days: int, ma_type: str = "SMA") -> FloatArray:
    """Return the final close-based SMA, SMA-seeded EMA, TRIMA, or 2/30 KAMA."""
    kind = _moving_average_type(ma_type)
    if bars.shape[1] < days:
        return np.full(bars.shape[0], np.nan)
    close = bars[:, :, 3]
    if kind == "SMA":
        result = close[:, -days:].mean(axis=1)
    elif kind == "TRIMA":
        weights = np.minimum(np.arange(1, days + 1), np.arange(days, 0, -1))
        result = (close[:, -days:] * weights).sum(axis=1) / weights.sum()
    elif kind == "EMA":
        result = _exponential_mean(close, days, 2.0 / (days + 1))[:, -1]
    else:
        result = _adaptive_mean(close, days)[:, -1]
    return np.where(np.isfinite(result), result, np.nan)


def moving_average_snapshot(
    history: FloatArray, current: FloatArray, days: int, ma_type: str = "SMA"
) -> FloatArray:
    """Reuse shared historical MA state across independent current-minute bars."""
    kind = _moving_average_type(ma_type)
    close, present = history[:, 3], current[:, 3]
    if close.size + 1 < days:
        return np.full_like(present, np.nan)
    if days == 1:
        return np.where(np.isfinite(present), present, np.nan)
    if kind == "SMA":
        result = _advance_mean(close, present, days, 1.0 / days, math.nan)
        return np.where(np.isfinite(result), result, np.nan)
    if kind == "TRIMA":
        weights = np.minimum(np.arange(1, days + 1), np.arange(days, 0, -1))
        result = (np.dot(close[-days + 1 :], weights[:-1]) + present) / weights.sum()
        return np.where(np.isfinite(result), result, np.nan)
    if kind == "EMA":
        series = _exponential_mean(close[None], days, 2.0 / (days + 1))[0]
        previous = float(series[-1]) if close.size else math.nan
        result = _advance_mean(close, present, days, 2.0 / (days + 1), previous)
        return np.where(np.isfinite(result), result, np.nan)
    return _adaptive_snapshot(close, present, days)
