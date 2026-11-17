"""Extended feature schemas and calculators preserve causal observation semantics."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from ifera.strategy_optimization import _events, _indicators
from ifera.strategy_optimization._features import (
    FeatureSpec,
    build_features,
    load_features,
)


@pytest.fixture(autouse=True)
def offline_events(monkeypatch):
    def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("Feature integration tests must use preloaded event dates")

    monkeypatch.setattr(_events, "load_event_dates", unexpected_fetch)


def _history(count=80):
    index = np.arange(count, dtype=np.float64)
    close = 100 + np.sin(index * 0.4) * 8 + index * 0.1
    return pl.DataFrame(
        {
            "date": [
                date(2025, 1, 1) + timedelta(days=offset) for offset in range(count)
            ],
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000 + index * 10,
        }
    )


def _observations():
    session = date(2025, 1, 1) + timedelta(days=80)
    return pl.DataFrame(
        {
            "datetime": [
                datetime.combine(session, time(10)),
                datetime.combine(session, time(15)),
            ],
            "underlying_price": [106.0, 114.0],
            "day_open": [102.0, 102.0],
            "day_high": [107.0, 115.0],
            "day_low": [99.0, 98.0],
            "day_volume": [350.0, 700.0],
            "prev_close": [105.0, 105.0],
            "premium": [200.0, 200.0],
            "risk": [800.0, 800.0],
            "reward_risk": [25.0, 25.0],
            "minutes_open": [0.0, 300.0],
            "return_pct": [-20.0, 30.0],
            "return_pct_high": [10.0, 40.0],
            "return_pct_low": [-30.0, -5.0],
            "ror_pct": [-5.0, 15.0],
            "ror_pct_high": [20.0, 25.0],
            "ror_pct_low": [-10.0, -8.0],
            "short_call_delta": [0.3, 0.5],
            "long_call_delta": [0.1, 0.2],
            "short_put_delta": [-0.3, -0.5],
            "long_put_delta": [-0.1, -0.2],
            "position_bid_ask_spread": [0.2, 0.8],
            "position_itm_pct": [-2.0, 3.0],
        }
    )


def _load_one(tmp_path, template, **parameters):
    path = tmp_path / "features.yaml"
    path.write_text(
        yaml.safe_dump(
            {"features": {"selected": {"template": template, **parameters}}}
        ),
        encoding="utf-8",
    )
    return load_features(path)[0]


def _complete_bars(observations, history, *, volume=False):
    columns = ["open", "high", "low", "close"]
    current_columns = ["day_open", "day_high", "day_low", "underlying_price"]
    if volume:
        columns.append("volume")
        current_columns.append("day_volume")
    past = history.select(columns).to_numpy().astype(np.float64)
    current = observations.select(current_columns).to_numpy().astype(np.float64)
    return np.concatenate(
        (np.broadcast_to(past, (len(current), *past.shape)), current[:, None]), axis=1
    )


def test_default_extensions_schema_and_offline_feature_matrix():
    specs = load_features()
    assert len(specs) == 55
    assert all(spec.kind == "bool" for spec in specs[:24])
    assert all(spec.kind == "continuous" for spec in specs[24:])
    assert all(spec.granularity is None for spec in specs[:24])
    assert sum(spec.template == "event_day" for spec in specs) == 13
    names = {spec.name: spec for spec in specs}
    assert names["mfi_14_intra"].parameters["days_ago"] == 0
    assert names["underlying_above_ema_5"].parameters == {"days": 5, "ma_type": "EMA"}
    assert names["ema_5_above_ema_20"].parameters == {
        "days1": 5,
        "days2": 20,
        "ma_type": "EMA",
    }
    events = {name: () for name in _events.EVENT_TYPES}

    actual = build_features(_observations(), _history(), specs, events=events)

    assert actual.shape == (2, 55)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()


@pytest.mark.parametrize("kind", ["SMA", "EMA", "TRIMA", "KAMA"])
@pytest.mark.parametrize("days", [5, 10, 20])
def test_underlying_above_moving_average_uses_raw_close(kind, days):
    observations, history = _observations(), _history()
    spec = FeatureSpec(
        "above",
        "bool",
        "any",
        "underlying_above_ma",
        parameters={"days": days, "ma_type": kind},
    )
    complete = _complete_bars(observations, history)
    average = _indicators.moving_average(complete, days, kind)
    expected = (complete[:, -1, 3] > average).astype(np.float32)

    actual = build_features(observations, history, (spec,))

    np.testing.assert_array_equal(actual[:, 0].numpy(), expected)
    assert spec.needs_daily_history


@pytest.mark.parametrize("kind", ["SMA", "EMA", "TRIMA", "KAMA"])
@pytest.mark.parametrize("first,second", [(5, 10), (5, 20), (10, 20)])
def test_moving_average_comparison_respects_both_periods(kind, first, second):
    observations, history = _observations(), _history()
    spec = FeatureSpec(
        "comparison",
        "bool",
        "any",
        "ma_comparison",
        parameters={"days1": first, "days2": second, "ma_type": kind},
    )
    complete = _complete_bars(observations, history)
    expected = (
        _indicators.moving_average(complete, first, kind)
        > _indicators.moving_average(complete, second, kind)
    ).astype(np.float32)

    np.testing.assert_array_equal(
        build_features(observations, history, (spec,))[:, 0].numpy(), expected
    )


@pytest.mark.parametrize("kind", ["SMA", "EMA", "TRIMA", "KAMA"])
@pytest.mark.parametrize(
    "template,parameters",
    [
        ("underlying_above_ma", {"days": 20}),
        ("ma_comparison", {"days1": 5, "days2": 20}),
    ],
)
def test_moving_average_boolean_warmup_stays_nan(kind, template, parameters):
    spec = FeatureSpec(
        "unavailable",
        "bool",
        "any",
        template,
        parameters={**parameters, "ma_type": kind},
    )
    assert torch.isnan(build_features(_observations(), _history(2), (spec,))).all()


_NEW_INDICATORS = (
    "stoch_rsi",
    "williams_r",
    "ultimate",
    "mfi",
    "chande",
    "macd_above_signal",
)


def _default_spec(template):
    return next(spec for spec in load_features() if spec.template == template)


def _native_indicator(bars, template, parameters):
    arguments = {key: value for key, value in parameters.items() if key != "days_ago"}
    if template == "macd_above_signal":
        histogram = _indicators.macd(bars, **arguments, component="histogram")
        return np.where(np.isfinite(histogram), histogram > 0, np.nan)
    return getattr(_indicators, template)(bars, **arguments)


@pytest.mark.parametrize("template", _NEW_INDICATORS)
@pytest.mark.parametrize("days_ago", [0, 1, 2])
def test_new_indicator_pipeline_matches_causal_bar_history(template, days_ago):
    observations, history = _observations(), _history()
    original = _default_spec(template)
    spec = replace(original, parameters={**original.parameters, "days_ago": days_ago})
    if days_ago:
        columns = ["open", "high", "low", "close"] + (
            ["volume"] if template == "mfi" else []
        )
        end = history.height - days_ago + 1
        bars = history.head(end).select(columns).to_numpy().astype(np.float64)[None]
        expected = np.repeat(
            _native_indicator(bars, template, spec.parameters), observations.height
        )
    else:
        bars = _complete_bars(observations, history, volume=template == "mfi")
        expected = _native_indicator(bars, template, spec.parameters)

    actual = build_features(observations, history, (spec,))

    np.testing.assert_allclose(
        actual[:, 0].numpy(), spec.normalize(expected), rtol=1e-6, atol=1e-7
    )
    assert spec.needs_daily_history
    session = observations["datetime"][0].date()
    future = pl.DataFrame(
        {
            "date": [session, session + timedelta(days=1)],
            "open": [10000.0, 20000.0],
            "high": [10001.0, 20001.0],
            "low": [9999.0, 19999.0],
            "close": [10000.0, 20000.0],
            "volume": [1e8, 2e8],
        }
    )
    torch.testing.assert_close(
        actual, build_features(observations, pl.concat((history, future)), (spec,))
    )
    changed = observations.with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.col("underlying_price") + 100)
        .otherwise(pl.col("underlying_price"))
        .alias("underlying_price")
    )
    torch.testing.assert_close(actual[0], build_features(changed, history, (spec,))[0])
    if days_ago:
        torch.testing.assert_close(actual, build_features(changed, history, (spec,)))


@pytest.mark.parametrize("template", _NEW_INDICATORS)
@pytest.mark.parametrize("days_ago", [0, 1])
def test_new_indicator_warmup_preserves_missingness(template, days_ago):
    original = _default_spec(template)
    spec = replace(original, parameters={**original.parameters, "days_ago": days_ago})
    assert torch.isnan(build_features(_observations(), _history(2), (spec,))).all()


def test_macd_boolean_does_not_turn_missing_current_price_into_false():
    observations = _observations().with_columns(
        pl.Series("underlying_price", [106.0, float("nan")])
    )
    actual = build_features(
        observations, _history(), (_default_spec("macd_above_signal"),)
    )
    assert torch.isfinite(actual[0, 0])
    assert torch.isnan(actual[1, 0])


def test_money_flow_pipeline_uses_current_day_genuine_volume():
    observations, history = _observations(), _history()
    spec = _default_spec("mfi")
    original = build_features(observations, history, (spec,))
    changed = observations.with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.col("day_volume") * 10)
        .otherwise(pl.col("day_volume"))
        .alias("day_volume")
    )
    updated = build_features(changed, history, (spec,))
    assert updated[0, 0] == original[0, 0]
    assert updated[1, 0] != original[1, 0]
    missing = observations.with_columns(pl.Series("day_volume", [350.0, float("nan")]))
    values = build_features(missing, history, (spec,))
    assert torch.isfinite(values[0, 0])
    assert torch.isnan(values[1, 0])


@pytest.mark.parametrize("missing", ["history", "observations", "both"])
def test_money_flow_pipeline_does_not_invent_missing_volume(missing):
    observations, history = _observations(), _history()
    if missing in ("history", "both"):
        history = history.drop("volume")
    if missing in ("observations", "both"):
        observations = observations.drop("day_volume")
    assert torch.isnan(
        build_features(observations, history, (_default_spec("mfi"),))
    ).all()


@pytest.mark.parametrize(
    "template,column", [("return_pct", "return_pct"), ("ror_pct", "ror_pct")]
)
@pytest.mark.parametrize("component", ["last", "high", "low"])
def test_position_return_components_select_matching_observation_columns(
    tmp_path, template, column, component
):
    spec = _load_one(tmp_path, template, component=component)
    name = column if component == "last" else f"{column}_{component}"
    expected = spec.normalize(_observations()[name].to_numpy())
    np.testing.assert_allclose(
        build_features(_observations(), pl.DataFrame(), (spec,))[:, 0].numpy(), expected
    )


@pytest.mark.parametrize("template", ["return_pct", "ror_pct"])
def test_legacy_return_yaml_without_component_defaults_to_last(tmp_path, template):
    spec = _load_one(tmp_path, template)
    assert spec.parameters == {"component": "last"}
    expected = spec.normalize(_observations()[template].to_numpy())
    np.testing.assert_allclose(
        build_features(_observations(), pl.DataFrame(), (spec,))[:, 0].numpy(), expected
    )


def test_position_spread_and_signed_moneyness_normalization():
    observations = pl.DataFrame(
        {
            "datetime": [datetime(2025, 1, 2, 10)] * 3,
            "position_bid_ask_spread": [1.5, 0.0, 15.0],
            "position_itm_pct": [-5.0, 0.0, 5.0],
        }
    )
    defaults = {spec.name: spec for spec in load_features()}
    specs = (defaults["position_bid_ask_spread"], defaults["position_itm_pct"])
    expected = torch.tensor([[0.15, 0.25], [0.0, 0.5], [1.0, 0.75]])
    torch.testing.assert_close(
        build_features(observations, pl.DataFrame(), specs), expected
    )


@pytest.mark.filterwarnings(
    "error:The 'generic' unit for NumPy timedelta is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize(
    "offset,event_date,expected",
    [
        (-1, date(2025, 1, 17), [0.0, 1.0]),
        (0, date(2025, 1, 21), [0.0, 1.0]),
        (1, date(2025, 1, 21), [1.0, 0.0]),
    ],
)
def test_event_features_use_preloaded_dates_and_market_session_offsets(
    offset, event_date, expected
):
    observations = pl.DataFrame(
        {"datetime": [datetime(2025, 1, 17, 10), datetime(2025, 1, 21, 10)]}
    )
    spec = FeatureSpec(
        "event",
        "bool",
        "any",
        "event_day",
        parameters={"event_type": "CPI Release", "days": offset},
    )
    actual = build_features(
        observations,
        pl.DataFrame(),
        (spec,),
        events={"CPI Release": (event_date,)},
    )
    assert actual[:, 0].tolist() == expected


@pytest.mark.parametrize("kind", ["SMA", "EMA", "TRIMA", "KAMA"])
def test_supported_moving_average_types_load_from_yaml(tmp_path, kind):
    assert (
        _load_one(tmp_path, "underlying_above_ma", days=5, ma_type=kind).kind == "bool"
    )
    assert (
        _load_one(tmp_path, "ma_comparison", days1=5, days2=20, ma_type=kind).kind
        == "bool"
    )


@pytest.mark.parametrize(
    "template,parameters,match",
    [
        ("underlying_above_ma", {"days": 5, "ma_type": "WMA"}, "ma_type"),
        ("underlying_above_ma", {"days": 5, "ma_type": None}, "ma_type"),
        ("underlying_above_ma", {"days": 0, "ma_type": "EMA"}, "days"),
        ("underlying_above_ma", {"days": True, "ma_type": "EMA"}, "days"),
        ("ma_comparison", {"days1": -1, "days2": 20, "ma_type": "EMA"}, "days1"),
        ("ma_comparison", {"days1": 5, "days2": 2.5, "ma_type": "EMA"}, "days2"),
        ("williams_r", {"days": 0, "days_ago": 0}, "days"),
        ("ultimate", {"short": 0, "medium": 14, "long": 28, "days_ago": 0}, "short"),
        ("mfi", {"days": 14, "days_ago": -1}, "days_ago"),
        ("chande", {"days": 9.5, "days_ago": 0}, "days"),
        (
            "stoch_rsi",
            {
                "days": 14,
                "rsi_days": 1,
                "smooth_k": 3,
                "smooth_d": 3,
                "component": "d",
                "days_ago": 0,
            },
            "RSI",
        ),
        (
            "stoch_rsi",
            {
                "days": 14,
                "rsi_days": 14,
                "smooth_k": 3,
                "smooth_d": 3,
                "component": "other",
                "days_ago": 0,
            },
            "component",
        ),
        (
            "macd_above_signal",
            {"fast": 26, "slow": 12, "signal": 9, "days_ago": 0},
            "fast",
        ),
        ("return_pct", {"component": "median"}, "component"),
        ("ror_pct", {"component": 2}, "component"),
    ],
)
def test_invalid_feature_extension_parameters_are_rejected(
    tmp_path, template, parameters, match
):
    with pytest.raises(ValueError, match=match):
        _load_one(tmp_path, template, **parameters)


@pytest.mark.parametrize("days", [-11, 11, True, 0.5, "1", None])
def test_event_schema_rejects_invalid_offset_bounds_and_types(tmp_path, days):
    with pytest.raises(ValueError, match="days"):
        _load_one(tmp_path, "event_day", event_type="CPI Release", days=days)


@pytest.mark.parametrize("days", [-10, 0, 10])
def test_event_schema_accepts_both_offset_boundaries(tmp_path, days):
    spec = _load_one(tmp_path, "event_day", event_type="CPI Release", days=days)
    assert spec.parameters["days"] == days


@pytest.mark.parametrize("event_type", ["not an event", 7, None])
def test_event_schema_rejects_unknown_event_types(tmp_path, event_type):
    with pytest.raises(ValueError, match="[Ee]vent"):
        _load_one(tmp_path, "event_day", event_type=event_type, days=0)
