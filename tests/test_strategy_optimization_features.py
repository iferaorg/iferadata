"""Feature schema, indicator conventions, normalization, and lookahead tests."""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from ifera.strategy_optimization import _features, _indicators
from ifera.strategy_optimization._events import canonical_event_type
from ifera.strategy_optimization._features import (
    FeatureSpec,
    build_features,
    load_features,
)


def _history(count=80):
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(count)]
    close = np.arange(count, dtype=float) + 100.0
    return pl.DataFrame(
        {
            "date": days,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _observations():
    return pl.DataFrame(
        {
            "datetime": [datetime(2025, 3, 24, 10, 1), datetime(2025, 3, 24, 12, 2)],
            "underlying_price": [182.0, 183.0],
            "day_open": [181.0, 181.0],
            "day_high": [183.0, 184.0],
            "day_low": [180.0, 180.0],
            "prev_close": [179.0, 179.0],
            "premium": [200.0, 200.0],
            "risk": [800.0, 800.0],
            "reward_risk": [25.0, 25.0],
            "minutes_open": [0.0, 121.0],
            "return_pct": [0.0, -25.0],
            "ror_pct": [0.0, -6.25],
            "return_pct_high": [0.0, 0.0],
            "return_pct_low": [0.0, -25.0],
            "ror_pct_high": [0.0, 0.0],
            "ror_pct_low": [0.0, -6.25],
            "position_bid_ask_spread": [0.2, 0.25],
            "position_itm_pct": [-0.3, 0.2],
            "short_put_delta": [-0.3, -0.5],
            "long_put_delta": [-0.1, -0.2],
        }
    )


def test_default_schema_boolean_order_and_complete_finite_features():
    specs = load_features()
    assert len(specs) == 55
    assert all(spec.kind == "bool" for spec in specs[:24])
    assert all(spec.kind == "continuous" for spec in specs[24:])
    events = {
        canonical_event_type(spec.parameters["event_type"]): ()
        for spec in specs
        if spec.template == "event_day"
    }
    result = build_features(_observations(), _history(), specs, events=events)
    assert result.shape == (2, 55)
    absent = [
        index
        for index, spec in enumerate(specs)
        if spec.name in ("short_call_delta", "long_call_delta", "mfi_14_intra")
    ]
    assert torch.isnan(result[:, absent]).all()
    finite = result[:, [index for index in range(len(specs)) if index not in absent]]
    assert torch.isfinite(finite).all()
    assert ((finite >= 0) & (finite <= 1)).all()
    assert result[:, 0].tolist() == [1.0, 1.0]
    assert result[:, 1:5].sum().item() == 0
    names = [spec.name for spec in specs]
    assert result[0, names.index("time")].item() == pytest.approx(36060 / 86400)
    assert result[0, names.index("position_reward_risk")].item() == pytest.approx(
        25 / 500
    )
    assert result[0, names.index("short_put_delta")].item() == pytest.approx(0.35)


def test_custom_subset_inherits_package_owned_templates(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "features": {
                    "risk": {"template": "position_risk"},
                    "monday": {"template": "is_day", "day": "Monday"},
                },
            }
        ),
        encoding="utf-8",
    )
    specs = load_features(path)
    assert [spec.name for spec in specs] == ["monday", "risk"]
    assert specs[1].max_value == 10000
    assert specs[1].granularity == 1
    result = build_features(_observations(), pl.DataFrame(), specs)
    assert result[:, 1].tolist() == pytest.approx([0.08, 0.08])


def test_packaged_templates_and_feature_instances_are_separate_yaml_files():
    directory = Path(_features.__file__).parent
    templates = yaml.safe_load(
        (directory / "feature_templates.yaml").read_text(encoding="utf-8")
    )
    defaults = yaml.safe_load(
        (directory / "default_features.yaml").read_text(encoding="utf-8")
    )
    assert set(templates) == {"feature_templates"}
    assert set(defaults) == {"features"}
    assert len(defaults["features"]) == 55
    expected = {
        "time": 60,
        "position_premium": 1,
        "position_risk": 1,
        "minutes_since_open": 1,
    }
    for name, metadata in templates["feature_templates"].items():
        if metadata["type"] == "bool":
            assert "granularity" not in metadata
        else:
            assert metadata["granularity"] == expected.get(name, 0.01)
    for spec in load_features():
        assert spec.granularity == (
            None if spec.kind == "bool" else expected.get(spec.template, 0.01)
        )


@pytest.mark.parametrize("granularity", [None, 0.01, 2])
def test_optional_template_granularity_preserves_normalized_observations(granularity):
    metadata = {
        "type": "continuous",
        "category": "entry",
        "min_value": 0,
        "max_value": 10000,
        "parameters": [],
        "granularity": granularity,
    }
    spec = _features._feature_spec(
        "risk", {"template": "position_risk"}, {"position_risk": metadata}
    )
    assert spec.granularity == granularity
    np.testing.assert_allclose(
        spec.normalize(np.asarray([0.125, 800.123])), [0.0000125, 0.0800123]
    )


def test_missing_template_granularity_keeps_existing_positional_parameters():
    parameters = {"days": 14, "days_ago": 0}
    spec = FeatureSpec("rsi", "continuous", "any", "rsi", 0, 100, parameters)
    assert spec.parameters is parameters
    assert spec.granularity is None
    resolved = _features._feature_spec(
        "time",
        {"template": "time"},
        {
            "time": {
                "type": "continuous",
                "category": "any",
                "min_value": 0,
                "max_value": 86400,
                "parameters": [],
            }
        },
    )
    assert resolved.granularity is None


@pytest.mark.parametrize(
    "granularity", [True, False, 0, -0.1, float("nan"), float("inf"), "0.1", []]
)
def test_invalid_granularity_is_rejected_for_configured_and_manual_specs(granularity):
    with pytest.raises(ValueError, match="granularity must be a finite positive"):
        FeatureSpec("x", "continuous", "any", "time", granularity=granularity)
    with pytest.raises(ValueError, match="granularity must be a finite positive"):
        _features._feature_spec(
            "x",
            {"template": "time"},
            {
                "time": {
                    "type": "continuous",
                    "category": "any",
                    "min_value": 0,
                    "max_value": 86400,
                    "parameters": [],
                    "granularity": granularity,
                }
            },
        )


def test_boolean_template_rejects_nonnull_granularity():
    with pytest.raises(ValueError, match="granularity requires continuous type"):
        _features._feature_spec(
            "monday",
            {"template": "is_day", "day": "Monday"},
            {
                "is_day": {
                    "type": "bool",
                    "category": "any",
                    "parameters": ["day"],
                    "granularity": 1,
                }
            },
        )


@pytest.mark.parametrize(
    "configuration,match",
    [
        ({"features": {}}, "At least one feature"),
        ({"features": {"x": {"template": "bad"}}}, "unknown template"),
        ({"features": {"x": {"template": "is_day", "day": "Sunday"}}}, "weekday"),
        ({"features": {"x": {"template": "rsi", "days": 14}}}, "requires parameters"),
        (
            {"features": {"x": {"template": "rsi", "days": 1, "days_ago": 0}}},
            "at least 2",
        ),
        (
            {"features": {"x": {"template": "rsi", "days": True, "days_ago": 0}}},
            "integer",
        ),
        (
            {"features": {"x": {"template": "rsi", "days": 14, "days_ago": -1}}},
            "integer",
        ),
        ({"features": {"x": {"template": "leg_delta", "leg": "middle_put"}}}, "one of"),
        (
            {
                "feature_templates": {"position_risk": {"category": "exit"}},
                "features": {"x": {"template": "position_risk"}},
            },
            "accepts only features",
        ),
        (
            {
                "feature_templates": {"rsi": {"min_value": 100, "max_value": 0}},
                "features": {"x": {"template": "rsi", "days": 14, "days_ago": 0}},
            },
            "accepts only features",
        ),
        (
            {
                "feature_templates": {"rsi": {"max_value": float("nan")}},
                "features": {"x": {"template": "rsi", "days": 14, "days_ago": 0}},
            },
            "accepts only features",
        ),
        (
            {
                "feature_templates": {"rsi": {"type": "number"}},
                "features": {"x": {"template": "rsi", "days": 14, "days_ago": 0}},
            },
            "accepts only features",
        ),
        (
            {"features": {"x": {"template": "time", "unrecognized": 1}}},
            "requires parameters",
        ),
        (
            {"features": {"x": {"template": "time", "granularity": 30}}},
            "requires parameters",
        ),
        ({"features": {"x": {"template": "time"}}, "unknown": {}}, "accepts only"),
    ],
)
def test_invalid_feature_configuration_is_rejected(tmp_path, configuration, match):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_features(path)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"category": "exit"}, "requires category entry"),
        ({"min_value": 100, "max_value": 0}, "increasing"),
        ({"max_value": float("nan")}, "finite"),
        ({"type": "number"}, "type must be"),
        ({"unexpected": 1}, "unknown fields"),
    ],
)
def test_invalid_package_template_metadata_is_rejected(overrides, match):
    metadata = {
        "type": "continuous",
        "category": "entry",
        "min_value": 0,
        "max_value": 10000,
        "parameters": [],
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        _features._feature_spec(
            "risk", {"template": "position_risk"}, {"position_risk": metadata}
        )


def test_rsi_uses_exactly_thirteen_prior_closes_and_current_price():
    history = _history(13)
    observations = _observations().with_columns(
        pl.lit(datetime(2025, 1, 14, 10)).alias("datetime"),
        pl.Series("underlying_price", [111.0, 113.0]),
    )
    spec = FeatureSpec(
        "rsi", "continuous", "any", "rsi", 0, 100, {"days": 14, "days_ago": 0}
    )
    values = build_features(observations, history, (spec,))[:, 0]
    assert values.tolist() == pytest.approx([12 / 13, 1.0])
    warmup = build_features(observations, history.slice(1), (spec,))
    assert warmup.isnan().all()


def test_intraday_indicators_never_use_same_day_final_bar_or_future_history():
    specs = tuple(
        spec
        for spec in load_features()
        if spec.template in ("rsi", "stoch", "cci", "adx", "momentum", "macd")
    )
    history = _history()
    future = pl.DataFrame(
        {
            "date": [date(2025, 3, 24), date(2025, 3, 25)],
            "open": [9000.0, 9000.0],
            "high": [9000.0, 9000.0],
            "low": [9000.0, 9000.0],
            "close": [9000.0, 9000.0],
        }
    )
    before = build_features(_observations(), history, specs)
    after = build_features(_observations(), pl.concat([history, future]), specs)
    torch.testing.assert_close(before, after)


def test_days_ago_uses_only_completed_daily_bars():
    observations = _observations()
    spec = FeatureSpec(
        "momentum",
        "continuous",
        "any",
        "momentum",
        -100,
        100,
        {"days": 10, "days_ago": 2},
    )
    values = build_features(observations, _history(), (spec,))
    assert values[:, 0].tolist() == pytest.approx([0.55, 0.55])
    changed = observations.with_columns(pl.lit(10000.0).alias("underlying_price"))
    torch.testing.assert_close(values, build_features(changed, _history(), (spec,)))


def test_normalization_clips_and_preserves_missing_values():
    spec = FeatureSpec("x", "continuous", "exit", "ror_pct", -100, 100)
    result = spec.normalize(np.asarray([-200.0, 0.0, 200.0, np.nan, np.inf]))
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0, np.nan, np.nan], equal_nan=True)


def test_indicator_reference_values_on_a_linear_trend():
    bars = _history().select("open", "high", "low", "close").to_numpy()[None]
    assert _indicators.rsi(bars, 14)[0] == pytest.approx(100)
    assert _indicators.stochastic(bars, 14, 3, 3, "k")[0] == pytest.approx(
        100 * 14 / 15
    )
    assert _indicators.stochastic(bars, 14, 3, 3, "d")[0] == pytest.approx(
        100 * 14 / 15
    )
    assert _indicators.cci(bars, 20)[0] == pytest.approx(9.5 / (0.015 * 5))
    assert _indicators.adx(bars, 14)[0] == pytest.approx(100)
    assert _indicators.momentum(bars, 10)[0] == pytest.approx(10)
    assert _indicators.macd(bars, 13, 26, 9, "line")[0] == pytest.approx(6.5)
    assert _indicators.macd(bars, 13, 26, 9, "signal")[0] == pytest.approx(6.5)
    assert _indicators.macd(bars, 13, 26, 9, "histogram")[0] == pytest.approx(
        0, abs=1e-12
    )


def test_flat_history_and_indicator_warmup():
    bars = np.full((2, 80, 4), 100.0)
    assert _indicators.rsi(bars, 14).tolist() == [50.0, 50.0]
    assert _indicators.stochastic(bars, 14, 3, 3, "d").tolist() == [50.0, 50.0]
    assert _indicators.cci(bars, 20).tolist() == [0.0, 0.0]
    assert _indicators.adx(bars, 14).tolist() == [0.0, 0.0]
    assert _indicators.macd(bars, 13, 26, 9, "histogram").tolist() == [0.0, 0.0]
    assert np.isnan(_indicators.stochastic(bars[:, :17], 14, 3, 3, "d")).all()
    assert np.isnan(_indicators.cci(bars[:, :19], 20)).all()
    assert np.isnan(_indicators.adx(bars[:, :27], 14)).all()
    assert np.isnan(_indicators.momentum(bars[:, :10], 10)).all()
    assert np.isnan(_indicators.macd(bars[:, :33], 13, 26, 9, "signal")).all()


def test_invalid_observations_fail_with_clear_messages():
    spec = FeatureSpec("risk", "continuous", "entry", "position_risk", 0, 1000)
    with pytest.raises(ValueError, match="require datetime"):
        build_features(pl.DataFrame(), pl.DataFrame(), (spec,))
    with pytest.raises(ValueError, match="require column 'risk'"):
        build_features(_observations().drop("risk"), pl.DataFrame(), (spec,))
    boolean = FeatureSpec(
        "monday", "bool", "any", "is_day", parameters={"day": "Monday"}
    )
    with pytest.raises(ValueError, match="Boolean feature"):
        build_features(_observations(), pl.DataFrame(), (spec, boolean))
    assert build_features(_observations(), pl.DataFrame(), ()).shape == (2, 0)
    assert build_features(_observations().head(0), pl.DataFrame(), (spec,)).shape == (
        0,
        1,
    )


def _random_bars(count, seed):
    """Create varying price bars for recursive-indicator equivalence checks."""
    random = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(random.normal(0, 1.2, count))
    return np.column_stack(
        (
            close + random.normal(0, 0.4, count),
            close + random.uniform(0.5, 3, count),
            close - random.uniform(0.5, 3, count),
            close,
        )
    )


@pytest.mark.parametrize(
    "history_length", [0, 1, 12, 13, 14, 25, 26, 27, 28, 32, 33, 34, 120]
)
@pytest.mark.parametrize("days", [1, 2, 14])
def test_adx_snapshot_matches_complete_independent_histories(history_length, days):
    history = _random_bars(history_length, 16)
    current = _random_bars(17, 42)
    complete = np.concatenate(
        (np.broadcast_to(history, (17, history_length, 4)), current[:, None]), axis=1
    )
    expected = _indicators.adx(complete, days)
    actual = _indicators.adx_snapshot(history, current, days)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    "history_length", [0, 1, 12, 13, 14, 25, 26, 27, 28, 32, 33, 34, 120]
)
@pytest.mark.parametrize("component", ["line", "signal", "histogram"])
def test_macd_snapshot_matches_complete_independent_histories(
    history_length, component
):
    history = _random_bars(history_length, 16)
    current = _random_bars(17, 42)
    complete = np.concatenate(
        (np.broadcast_to(history, (17, history_length, 4)), current[:, None]), axis=1
    )
    expected = _indicators.macd(complete, 13, 26, 9, component)
    actual = _indicators.macd_snapshot(history, current, 13, 26, 9, component)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_bounded_window_feature_calculation_matches_full_histories():
    historical_bars = _random_bars(80, 16)
    history = _history(80).with_columns(
        *[
            pl.Series(column, historical_bars[:, index])
            for index, column in enumerate(("open", "high", "low", "close"))
        ]
    )
    observations = _observations()
    specs = tuple(
        spec
        for spec in load_features()
        if spec.template in ("rsi", "cci", "stoch", "momentum")
    )
    current = observations.select(
        "day_open", "day_high", "day_low", "underlying_price"
    ).to_numpy()
    complete = np.concatenate(
        (np.broadcast_to(historical_bars, (2, 80, 4)), current[:, None]), axis=1
    )
    calculators = {
        "rsi": _indicators.rsi,
        "cci": _indicators.cci,
        "stoch": _indicators.stochastic,
        "momentum": _indicators.momentum,
    }
    expected = []
    for spec in specs:
        arguments = dict(spec.parameters)
        arguments.pop("days_ago")
        expected.append(
            spec.normalize(calculators[spec.template](complete, **arguments))
        )
    actual = build_features(observations, history, specs)
    np.testing.assert_allclose(actual.numpy(), np.column_stack(expected), rtol=1e-6)
