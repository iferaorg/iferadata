"""New oscillator and moving-average calculations remain causal and batched."""

import numpy as np
import pytest

from ifera.strategy_optimization import _indicators as indicators


def _bars(close, volume=None):
    close = np.asarray(close, dtype=np.float64)
    if close.ndim == 1:
        close = close[None]
    values = np.stack((close, close + 1.0, close - 1.0, close), axis=-1)
    if volume is not None:
        volumes = np.broadcast_to(np.asarray(volume, dtype=np.float64), close.shape)
        values = np.concatenate((values, volumes[..., None]), axis=-1)
    return values


def _prices(count):
    index = np.arange(count, dtype=np.float64)
    return 100 + 3 * np.sin(index * 0.7) + index * 0.2


def _stoch_rsi_reference(close, days, rsi_days, smooth_k, smooth_d):
    relative = []
    for index in range(rsi_days - 1, len(close)):
        changes = np.diff(close[index - rsi_days + 1 : index + 1])
        up = sum(max(float(change), 0) for change in changes)
        down = sum(max(float(-change), 0) for change in changes)
        relative.append(100 * up / (up + down) if up + down else 50)
    raw = []
    for index in range(days - 1, len(relative)):
        window = relative[index - days + 1 : index + 1]
        low, high = min(window), max(window)
        raw.append(100 * (relative[index] - low) / (high - low) if high > low else 50)
    slow = [
        np.mean(raw[index - smooth_k + 1 : index + 1])
        for index in range(smooth_k - 1, len(raw))
    ]
    return slow[-1], np.mean(slow[-smooth_d:])


def test_stochastic_rsi_matches_independent_price_window_reference():
    close = _prices(70)
    bars = _bars(np.stack((close, close[::-1])))
    for index in range(2):
        expected_k, expected_d = _stoch_rsi_reference(bars[index, :, 3], 5, 4, 2, 3)
        assert indicators.stoch_rsi(bars[index : index + 1], 5, 4, 2, 3, "k")[
            0
        ] == pytest.approx(expected_k)
        assert indicators.stoch_rsi(bars[index : index + 1], 5, 4, 2, 3, "d")[
            0
        ] == pytest.approx(expected_d)


def test_stochastic_rsi_flat_prices_are_neutral():
    assert indicators.stoch_rsi(_bars(np.full(31, 10.0))).tolist() == [50.0]


def test_stochastic_rsi_monotone_decimal_prices_have_constant_rsi():
    # Tiny roundoff in an all-gain RSI must not become a false stochastic range.
    close = 100.0 + np.arange(40) * 0.1
    assert indicators.stoch_rsi(_bars(close)).tolist() == [50.0]


def test_williams_r_uses_high_low_and_last_close():
    assert indicators.williams_r(_bars([10, 12, 11, 13]), 3)[0] == pytest.approx(-25)
    flat = np.full((2, 14, 4), 20.0)
    assert indicators.williams_r(flat).tolist() == [-50.0, -50.0]


def test_ultimate_oscillator_weights_three_pressure_range_ratios():
    expected = 100 / 7 * (4 * (1 / 2) + 2 * (3 / 5) + 5 / 9)
    assert indicators.ultimate(_bars([10, 11, 10, 12, 11]), 1, 2, 4)[
        0
    ] == pytest.approx(expected)
    assert indicators.ultimate(np.full((1, 29, 4), 20.0))[0] == 0.0


def test_money_flow_uses_volume_and_excludes_comparison_bar_volume():
    bars = _bars([10, 11, 10, 12], [np.nan, 100, 200, 300])
    assert indicators.mfi(bars, 3)[0] == pytest.approx(100 * 4700 / 6700)
    assert indicators.mfi(_bars([10, 11, 12, 13], 100), 3)[0] == 100
    assert indicators.mfi(_bars([13, 12, 11, 10], 100), 3)[0] == 0
    assert indicators.mfi(_bars([10, 10, 10, 10], 100), 3)[0] == 50


@pytest.mark.parametrize("volume", [0.0, -1.0, np.nan, np.inf])
def test_money_flow_requires_positive_finite_window_volume(volume):
    bars = _bars([10, 11, 10, 12], 100)
    bars[0, -2, 4] = volume
    assert np.isnan(indicators.mfi(bars, 3)[0])


def test_money_flow_without_genuine_volume_is_unavailable():
    assert np.isnan(indicators.mfi(_bars(_prices(20)))).all()


def test_chande_uses_the_requested_number_of_consecutive_changes():
    assert indicators.chande(_bars([10, 12, 11, 14]), 3)[0] == pytest.approx(
        100 * 4 / 6
    )
    assert indicators.chande(_bars(np.arange(10.0)), 9)[0] == 100
    assert indicators.chande(_bars(np.arange(10.0)[::-1]), 9)[0] == -100
    assert indicators.chande(_bars(np.full(10, 20.0)), 9)[0] == 0


@pytest.mark.parametrize(
    "calculator,minimum",
    [
        (indicators.stoch_rsi, 31),
        (lambda bars: indicators.stoch_rsi(bars, component="k"), 29),
        (indicators.williams_r, 14),
        (indicators.ultimate, 29),
        (indicators.mfi, 15),
        (indicators.chande, 10),
        (lambda bars: indicators.moving_average(bars, 14, "SMA"), 14),
        (lambda bars: indicators.moving_average(bars, 14, "EMA"), 14),
        (lambda bars: indicators.moving_average(bars, 14, "TRIMA"), 14),
        (lambda bars: indicators.moving_average(bars, 14, "KAMA"), 15),
    ],
)
def test_exact_indicator_warmup(calculator, minimum):
    assert np.isnan(calculator(_bars(_prices(minimum - 1), 100))).all()
    assert np.isfinite(calculator(_bars(_prices(minimum), 100))).all()


@pytest.mark.parametrize(
    "calculator",
    [
        indicators.stoch_rsi,
        indicators.williams_r,
        indicators.ultimate,
        indicators.mfi,
        indicators.chande,
    ],
)
def test_oscillators_batch_independent_snapshots_and_keep_input(calculator):
    history = _bars(_prices(40), 100)[0]
    current = _bars([105.0, 95.0, 115.0], [100, 200, 300])[0]
    histories = np.broadcast_to(history, (current.shape[0], *history.shape))
    bars = np.concatenate((histories, current[:, None]), axis=1)
    original = bars.copy()

    actual = calculator(bars)
    separate = np.concatenate(
        [calculator(bars[index : index + 1]) for index in range(3)]
    )

    assert actual.shape == (3,)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, separate)
    np.testing.assert_array_equal(bars, original)
    changed = bars.copy()
    changed[-1, -1, 3] += 10
    assert calculator(changed)[0] == actual[0]
    bars[0, -1, 3] = np.nan
    assert np.isnan(calculator(bars)[0])


@pytest.mark.parametrize(
    "kind,days,expected",
    [
        ("SMA", 3, 17 / 3),
        ("EMA", 3, 17 / 3),
        ("TRIMA", 3, 21 / 4),
        ("TRIMA", 4, 14 / 3),
    ],
)
def test_moving_averages_have_known_weighting_and_sma_seed(kind, days, expected):
    assert indicators.moving_average(_bars([1, 2, 5, 4, 8]), days, kind)[
        0
    ] == pytest.approx(expected)


def test_kama_uses_efficiency_ratio_squared_smoothing_and_previous_close_seed():
    # A monotone series has efficiency one, so the squared fast smoothing is 4/9.
    assert indicators.moving_average(_bars([1, 2, 3, 4, 5]), 3, "KAMA")[
        0
    ] == pytest.approx(335 / 81)
    assert indicators.moving_average(_bars(np.full(10, 20.0)), 3, "KAMA")[0] == 20


@pytest.mark.parametrize("kind", ["SMA", "EMA", "TRIMA", "KAMA"])
@pytest.mark.parametrize("days", [1, 2, 10])
@pytest.mark.parametrize("history_length", [0, 1, 9, 10, 17, 40])
def test_moving_average_snapshots_match_independent_full_histories(
    kind, days, history_length
):
    history = _bars(_prices(history_length))[0]
    current = _bars([103.0, 98.0, 110.0, 120.0])[0]
    full = np.concatenate(
        (np.broadcast_to(history, (4, *history.shape)), current[:, None]), axis=1
    )
    original_history, original_current = history.copy(), current.copy()

    actual = indicators.moving_average_snapshot(history, current, days, kind)
    expected = indicators.moving_average(full, days, kind)

    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(history, original_history)
    np.testing.assert_array_equal(current, original_current)
    np.testing.assert_allclose(
        indicators.moving_average_snapshot(history, current[::-1], days, kind),
        actual[::-1],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "kind,helper_name", [("EMA", "_exponential_mean"), ("KAMA", "_adaptive_mean")]
)
def test_recursive_moving_average_history_is_computed_once_for_all_minutes(
    monkeypatch, kind, helper_name
):
    history = _bars(_prices(40))[0]
    current = _bars(_prices(200))[0]
    original = getattr(indicators, helper_name)
    calls = []

    def observe(values, *parameters):
        calls.append(values.shape)
        return original(values, *parameters)

    monkeypatch.setattr(indicators, helper_name, observe)
    actual = indicators.moving_average_snapshot(history, current, 10, kind)

    assert calls == [(1, 40)]
    assert actual.shape == (200,)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("kind", ["SMA", "EMA", "TRIMA", "KAMA"])
@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_moving_average_missing_snapshot_is_unavailable(kind, missing):
    history = _bars(_prices(40))[0]
    current = _bars([105.0, missing])[0]
    assert np.isfinite(
        indicators.moving_average_snapshot(history, current, 10, kind)[0]
    )
    assert np.isnan(indicators.moving_average_snapshot(history, current, 10, kind)[1])


def test_moving_average_type_is_case_insensitive_and_rejects_unknown_types():
    bars = _bars(_prices(20))
    np.testing.assert_allclose(
        indicators.moving_average(bars, 10, "sma"),
        indicators.moving_average(bars, 10, "SMA"),
    )
    with pytest.raises(ValueError, match="ma_type"):
        indicators.moving_average(bars, 10, "unknown")
    with pytest.raises(ValueError, match="ma_type"):
        indicators.moving_average_snapshot(bars[0, :-1], bars[:, -1], 10, "unknown")
