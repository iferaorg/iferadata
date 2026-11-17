"""Historical daily replacements recover indicators without fabricating minutes."""

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest
import torch
import yaml

from ifera.strategy_optimization import _daily_history, optimizer
from ifera.strategy_optimization._data import MarketData
from ifera.strategy_optimization._features import FeatureSpec
from ifera.strategy_optimization._preparation import prepare_strategy
from tests.test_strategy_optimization import strategy_market


def _bars(rows):
    return pl.DataFrame(
        [
            {
                "symbol": "SPX",
                "date": session,
                "open": close - 1,
                "high": close + 5,
                "low": close - 5,
                "close": close,
            }
            for session, close in rows
        ],
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


def _momentum(days=1):
    return FeatureSpec(
        "momentum",
        "continuous",
        "any",
        "momentum",
        -10000,
        10000,
        {"days": days, "days_ago": 0},
    )


def test_daily_replacements_supply_previous_close_gap_and_indicator_history(
    strategy_market,
):
    data, original = strategy_market
    bars = _bars([(date(2024, 1, 5), 4900.0), (date(2024, 1, 9), 4950.0)])
    market = MarketData(original.underlying, original.options, bars)
    features = (
        _momentum(),
        FeatureSpec(
            "rsi", "continuous", "any", "rsi", 0, 100, {"days": 2, "days_ago": 0}
        ),
        FeatureSpec("gap", "continuous", "any", "gap", -10, 10),
        FeatureSpec("change", "continuous", "any", "change_since_prev_close", -10, 10),
    )
    prepared = prepare_strategy(data, market, features, "early_exit")
    values = prepared.tensors.entry_features
    assert torch.isfinite(values).all()
    assert values[:, 0].tolist() == pytest.approx([0.505, 0.5025])
    assert values[:, 1].tolist() == pytest.approx([1.0, 1.0])
    expected_gap = [
        ((4990 / previous - 1) * 100 + 10) / 20 for previous in (4900, 4950)
    ]
    expected_change = [
        ((5000 / previous - 1) * 100 + 10) / 20 for previous in (4900, 4950)
    ]
    assert values[:, 2].tolist() == pytest.approx(expected_gap)
    assert values[:, 3].tolist() == pytest.approx(expected_change)
    assert prepared.exits["datetime"].to_list() == [
        datetime(2024, 1, 8, 13, 1),
        datetime(2024, 1, 8, 13, 2),
        datetime(2024, 1, 10, 13, 1),
        datetime(2024, 1, 10, 13, 2),
    ]


def test_finalized_same_day_bar_changes_only_later_trade_history(strategy_market):
    data, original = strategy_market
    bars = _bars(
        [
            (date(2024, 1, 4), 4880.0),
            (date(2024, 1, 5), 4900.0),
            (date(2024, 1, 8), 5003.0),
            (date(2024, 1, 9), 4950.0),
        ]
    )
    changed = bars.with_columns(
        *[
            pl.when(pl.col("date") == date(2024, 1, 8))
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
            for column, value in (
                ("open", 7000.0),
                ("high", 9000.0),
                ("low", 6000.0),
                ("close", 8000.0),
            )
        ]
    )
    features = (
        _momentum(2),
        FeatureSpec(
            "stoch",
            "continuous",
            "any",
            "stoch",
            0,
            100,
            {"days": 2, "smooth_k": 1, "smooth_d": 1, "component": "k", "days_ago": 0},
        ),
        FeatureSpec("change", "continuous", "any", "change_since_open", -10, 10),
    )
    before = prepare_strategy(
        data,
        MarketData(original.underlying, original.options, bars),
        features,
        "early_exit",
    )
    after = prepare_strategy(
        data,
        MarketData(original.underlying, original.options, changed),
        features,
        "early_exit",
    )
    torch.testing.assert_close(
        before.tensors.entry_features[0], after.tensors.entry_features[0]
    )
    torch.testing.assert_close(
        before.tensors.exit_features[:2], after.tensors.exit_features[:2]
    )
    assert before.tensors.entry_features[1, 0] != after.tensors.entry_features[1, 0]
    assert before.tensors.entry_features[0, 0].item() == pytest.approx(0.506)


@pytest.mark.parametrize(
    "missing_stamp", [datetime(2024, 1, 8, 13), datetime(2024, 1, 8, 13, 1)]
)
def test_daily_bar_never_replaces_a_missing_trade_minute(
    strategy_market, missing_stamp
):
    data, original = strategy_market
    underlying = original.underlying.filter(pl.col("datetime") != missing_stamp)
    market = MarketData(
        underlying,
        original.options,
        _bars(
            [
                (date(2024, 1, 5), 4900.0),
                (date(2024, 1, 8), 5003.0),
                (date(2024, 1, 9), 4950.0),
            ]
        ),
    )
    with pytest.raises(ValueError, match="Missing underlying minute for trade 0"):
        prepare_strategy(data, market, (_momentum(),), "early_exit")


def test_daily_bar_never_replaces_a_missing_option_leg(strategy_market):
    data, original = strategy_market
    quotes = original.options.filter(
        ~(
            (pl.col("trade_index") == 0)
            & (pl.col("datetime") == datetime(2024, 1, 8, 13, 1))
            & (pl.col("direction") == "short")
        )
    )
    market = MarketData(
        original.underlying,
        quotes,
        _bars(
            [
                (date(2024, 1, 5), 4900.0),
                (date(2024, 1, 8), 5003.0),
                (date(2024, 1, 9), 4950.0),
            ]
        ),
    )
    with pytest.raises(ValueError, match="Missing complete option minute for trade 0"):
        prepare_strategy(data, market, (_momentum(),), "early_exit")


def test_daily_replacement_removes_invalid_duplicate_nontrade_history_minutes(
    strategy_market,
):
    data, original = strategy_market
    bad_history = pl.DataFrame(
        {
            "symbol": ["SPX", "SPX"],
            "datetime": [datetime(2024, 1, 5, 9, 31)] * 2,
            "price": [None, -1.0],
        },
        schema=original.underlying.schema,
    )
    market = MarketData(
        pl.concat([original.underlying, bad_history]),
        original.options,
        _bars([(date(2024, 1, 5), 4900.0), (date(2024, 1, 9), 4950.0)]),
    )
    prepared = prepare_strategy(data, market, (_momentum(),), "early_exit")
    assert prepared.tensors.entry_features[:, 0].tolist() == pytest.approx(
        [0.505, 0.5025]
    )


@pytest.mark.parametrize("bad_price", [None, -1.0])
def test_trade_day_invalid_minute_still_fails_with_a_daily_replacement(
    strategy_market, bad_price
):
    data, original = strategy_market
    underlying = original.underlying.with_columns(
        pl.when(pl.col("datetime") == datetime(2024, 1, 8, 13, 1))
        .then(pl.lit(bad_price, dtype=pl.Float64))
        .otherwise(pl.col("price"))
        .alias("price")
    )
    market = MarketData(
        underlying,
        original.options,
        _bars([(date(2024, 1, 8), 5003.0)]),
    )
    with pytest.raises(ValueError, match="Invalid underlying prices"):
        prepare_strategy(data, market, (_momentum(),), "early_exit")


def test_existing_two_field_custom_market_data_stays_offline(
    strategy_market, monkeypatch
):
    data, original = strategy_market
    download = MagicMock(side_effect=AssertionError("Unexpected Yahoo request"))
    monkeypatch.setattr(_daily_history, "fetch_daily_history", download)
    market = MarketData(original.underlying, original.options)
    prepared = prepare_strategy(data, market, (_momentum(),), "entry_filter")
    assert market.daily_history is None
    assert torch.isnan(prepared.tensors.entry_features[0, 0])
    download.assert_not_called()


@pytest.mark.parametrize(
    "template,expected",
    [
        ("rsi", True),
        ("stoch", True),
        ("cci", True),
        ("adx", True),
        ("momentum", True),
        ("macd", True),
        ("gap", True),
        ("change_since_prev_close", True),
        ("is_day", False),
        ("time", False),
        ("change_since_open", False),
        ("leg_delta", False),
        ("position_premium", False),
        ("position_risk", False),
        ("position_reward_risk", False),
        ("minutes_since_open", False),
        ("return_pct", False),
        ("ror_pct", False),
    ],
)
def test_feature_template_history_requirement(template, expected):
    assert (
        FeatureSpec("x", "continuous", "any", template).needs_daily_history is expected
    )


@pytest.mark.parametrize("template,expected", [("time", False), ("gap", True)])
def test_csv_facade_requests_daily_fallback_only_for_history_features(
    strategy_market, tmp_path, monkeypatch, template, expected
):
    data, market = strategy_market
    features = tmp_path / "features.yaml"
    features.write_text(
        yaml.safe_dump({"features": {"selected": {"template": template}}})
    )
    factory = MagicMock()
    factory.return_value.__enter__.return_value.load.return_value = market
    monkeypatch.setattr(optimizer, "OracleMarketDataSource", factory)
    optimizer.optimize_strategy_from_csv(
        "sample",
        "strategy",
        features_file=features,
        optimizer_parameters={
            "max_population": 4,
            "generations": 0,
            "device": "cpu",
            "seed": 4,
        },
        output_csv=tmp_path / "optimized.csv",
        show_progress=False,
        show_plot=False,
        print_report=False,
    )
    factory.assert_called_once_with(fill_daily_history=expected)
    factory.return_value.__enter__.return_value.load.assert_called_once()


def test_csv_facade_preserves_custom_source_without_yahoo_requests(
    strategy_market, tmp_path, monkeypatch
):
    _, market = strategy_market
    download = MagicMock(side_effect=AssertionError("Unexpected Yahoo request"))
    monkeypatch.setattr(_daily_history, "fetch_daily_history", download)
    source = SimpleNamespace(load=MagicMock(return_value=market))
    features = tmp_path / "historical_indicator.yaml"
    features.write_text(
        yaml.safe_dump(
            {
                "features": {
                    "momentum": {"template": "momentum", "days": 10, "days_ago": 0}
                }
            }
        ),
        encoding="utf-8",
    )
    optimizer.optimize_strategy_from_csv(
        "sample",
        "strategy",
        market_data_source=source,
        features_file=features,
        optimizer_parameters={
            "max_population": 4,
            "generations": 0,
            "device": "cpu",
            "seed": 4,
        },
        output_csv=tmp_path / "custom.csv",
        show_progress=False,
        show_plot=False,
        print_report=False,
    )
    source.load.assert_called_once()
    download.assert_not_called()
