"""Position marks, return extrema, and genuine underlying volume are causal."""

from datetime import date, datetime, timedelta
from pathlib import Path

import math
import polars as pl
import pytest
import torch

from ifera.strategy_optimization import _preparation
from ifera.strategy_optimization._data import MarketData, StrategyData
from ifera.strategy_optimization._evaluation import OptimizationMode
from ifera.strategy_optimization._features import FeatureSpec


@pytest.fixture
def position_market():
    """A credit spread has marked returns zero, positive, negative, then positive."""
    opened = datetime(2024, 1, 8, 10)
    trade = {
        "trade_index": 0,
        "opened": opened,
        "closed": opened + timedelta(minutes=4),
        "symbol": "SPX",
        "opening_price": -2.0,
        "quantity": 2,
        "premium": 400.0,
        "risk": 1600.0,
        "reward_risk": 25.0,
        "ror": 25.0,
    }
    legs = pl.DataFrame(
        {
            "trade_index": [0, 0],
            "direction": ["long", "short"],
            "right": ["put", "put"],
            "strike": [4970.0, 4980.0],
        }
    )
    underlying = [
        {
            "symbol": "SPX",
            "datetime": opened.replace(hour=9, minute=30),
            "price": 4990.0,
        }
    ]
    options = []
    for index, combo in enumerate((-2.0, -1.0, -3.0, -0.5, 100.0)):
        instant = opened + timedelta(minutes=index)
        underlying.append(
            {"symbol": "SPX", "datetime": instant, "price": 5000.0 + index}
        )
        for direction, strike, mid, half_width in (
            ("long", 4970.0, 101.0, 0.05),
            ("short", 4980.0, 101.0 - combo, 0.10),
        ):
            options.append(
                {
                    "trade_index": 0,
                    "datetime": instant,
                    "direction": direction,
                    "right": "put",
                    "strike": strike,
                    "bid": mid - half_width,
                    "ask": mid + half_width,
                    "mid": mid,
                    "delta": -0.2,
                }
            )
    market = MarketData(pl.from_dicts(underlying), pl.from_dicts(options))
    data = StrategyData(
        pl.DataFrame(),
        pl.from_dicts([trade]),
        legs,
        (opened.date(),),
        Path("sample.csv"),
    )
    return trade, data, market


def _observations(position_market, slippage=0.0):
    trade, data, market = position_market
    minute, _daily = _preparation._underlying_history(
        market, "SPX", frozenset(data.dates)
    )
    return _preparation._trade_observations(
        trade, data.legs, market, minute, OptimizationMode.EARLY_EXIT, slippage, True
    )


def test_position_spread_uses_signed_combination_bid_and_ask(position_market):
    observations = _observations(position_market)
    assert observations["position_bid_ask_spread"].to_list() == pytest.approx([0.3] * 4)
    assert observations["combo_bid"].to_list() == pytest.approx(
        [-2.15, -1.15, -3.15, -0.65]
    )
    assert observations["combo_ask"].to_list() == pytest.approx(
        [-1.85, -0.85, -2.85, -0.35]
    )


def test_position_spread_does_not_scale_with_quantity_or_entry_direction(
    position_market,
):
    trade, data, market = position_market
    modified = {**trade, "opening_price": 2.0, "quantity": 7}
    observations = _observations((modified, data, market))
    assert observations["position_bid_ask_spread"].to_list() == pytest.approx([0.3] * 4)


def test_entry_spread_feature_loads_quotes_without_leg_delta(
    position_market, monkeypatch
):
    _trade, data, market = position_market
    seen = []

    def capture(observations, _history, features, *, events=None):
        assert events == {}
        seen.append(observations)
        return torch.zeros(observations.height, len(features))

    monkeypatch.setattr(_preparation, "build_features", capture)
    feature = FeatureSpec("spread", "continuous", "any", "position_bid_ask_spread")
    prepared = _preparation.prepare_strategy(data, market, (feature,), "entry_filter")
    assert prepared.tensors.entry_features.shape == (1, 1)
    assert len(seen) == 1
    assert seen[0]["combo_mid"].to_list() == pytest.approx([-2.0])
    assert seen[0]["position_bid_ask_spread"].to_list() == pytest.approx([0.3])


def test_event_calendars_are_preloaded_and_flags_computed_once(
    position_market, monkeypatch
):
    """All minute snapshots share a single prepared daily calendar feature table."""
    _trade, data, market = position_market
    loads, flags = [], []
    original_flags = _preparation.event_flags

    def preload(timestamps, event_types, *, offsets):
        loads.append((timestamps, event_types, offsets))
        return {
            "Monthly Expiration": (data.dates[0], data.dates[0] + timedelta(days=1))
        }

    def counted_flags(*args, **kwargs):
        flags.append(args)
        return original_flags(*args, **kwargs)

    def unexpected_minute_flags(*_args, **_kwargs):
        raise AssertionError(
            "Event flags must not be recalculated for every trade/minute"
        )

    from ifera.strategy_optimization import _features

    monkeypatch.setattr(_preparation, "load_event_dates", preload)
    monkeypatch.setattr(_preparation, "event_flags", counted_flags)
    monkeypatch.setattr(_features, "event_flags", unexpected_minute_flags)
    features = tuple(
        FeatureSpec(
            f"event_{offset}",
            "bool",
            "any",
            "event_day",
            parameters={"event_type": "Monthly Expiration", "days": offset},
        )
        for offset in (0, 1)
    )
    prepared = _preparation.prepare_strategy(data, market, features, "combined")

    assert loads == [(data.dates, ["Monthly Expiration"] * 2, [0, 1])]
    assert len(flags) == 2
    assert prepared.tensors.entry_features.tolist() == [[1.0, 1.0]]
    assert prepared.tensors.exit_features.tolist() == [[1.0, 1.0]] * 3


@pytest.mark.parametrize("slippage", [0.0, 0.2])
def test_return_extrema_include_entry_mark_and_current_mark(position_market, slippage):
    observations = _observations(position_market, slippage)
    offset = slippage * 50.0
    returns = [0.0 - offset, 50.0 - offset, -50.0 - offset, 75.0 - offset]
    assert observations["return_pct"].to_list() == pytest.approx(returns)
    assert observations["return_pct_high"].to_list() == pytest.approx(
        [returns[0], returns[1], returns[1], returns[3]]
    )
    assert observations["return_pct_low"].to_list() == pytest.approx(
        [returns[0], returns[0], returns[2], returns[2]]
    )
    assert observations["ror_pct"].to_list() == pytest.approx(
        [value / 4 for value in returns]
    )
    assert observations["ror_pct_high"].to_list() == pytest.approx(
        [returns[0] / 4, returns[1] / 4, returns[1] / 4, returns[3] / 4]
    )
    assert observations["ror_pct_low"].to_list() == pytest.approx(
        [returns[0] / 4, returns[0] / 4, returns[2] / 4, returns[2] / 4]
    )


def test_return_extrema_never_use_later_or_original_close_marks(position_market):
    trade, data, market = position_market
    before = _observations(position_market)
    modified = market.options.with_columns(
        *[
            pl.when(pl.col("datetime") >= trade["opened"] + timedelta(minutes=3))
            .then(pl.col(column) + 1000.0 * (pl.col("direction") == "long"))
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("bid", "ask", "mid")
        ]
    )
    after = _observations((trade, data, MarketData(market.underlying, modified)))
    columns = ["return_pct_high", "return_pct_low", "ror_pct_high", "ror_pct_low"]
    assert before.select(columns).head(3).equals(after.select(columns).head(3))
    assert before.height == after.height == 4
    assert before["return_pct_high"][-1] == pytest.approx(75.0)


@pytest.mark.parametrize(
    "options, prices, expected",
    [
        ([("call", 100.0)], [90.0, 100.0, 110.0], [-10.0, 0.0, 10.0]),
        ([("put", 100.0)], [90.0, 100.0, 110.0], [10.0, 0.0, -10.0]),
        (
            [("call", 90.0), ("call", 110.0)],
            [80.0, 90.0, 100.0, 120.0],
            [-100 / 9, 0.0, 100 / 9, 100 / 3],
        ),
        (
            [("put", 90.0), ("put", 110.0)],
            [80.0, 110.0, 120.0],
            [300 / 11, 0.0, -100 / 11],
        ),
        (
            [("put", 90.0), ("put", 95.0), ("call", 105.0), ("call", 110.0)],
            [80.0, 95.0, 100.0, 105.0, 120.0],
            [300 / 19, 0.0, -100 / 21, 0.0, 100 / 7],
        ),
        ([("put", 90.0), ("call", 110.0)], [100.0], [-100 / 11]),
    ],
)
def test_position_moneyness_uses_most_itm_option_strike(options, prices, expected):
    legs = pl.DataFrame(
        {
            "right": [right for right, _strike in options],
            "strike": [strike for _right, strike in options],
            "direction": [
                "short" if index % 2 else "long" for index in range(len(options))
            ],
        }
    )
    underlying = pl.DataFrame({"price": prices})
    actual = underlying.select(_preparation._position_moneyness(legs).alias("itm"))
    assert actual["itm"].to_list() == pytest.approx(expected)
    reversed_legs = legs.with_columns(
        pl.when(pl.col("direction") == "long")
        .then(pl.lit("short"))
        .otherwise(pl.lit("long"))
        .alias("direction")
    )
    assert underlying.select(
        _preparation._position_moneyness(reversed_legs)
    ).to_series().to_list() == pytest.approx(expected)


def test_trade_observations_expose_position_moneyness_in_percent(position_market):
    observations = _observations(position_market)
    assert observations["position_itm_pct"].to_list() == pytest.approx(
        [
            100.0 * (4980.0 - value) / 4980.0
            for value in (5000.0, 5001.0, 5002.0, 5003.0)
        ]
    )


def _volume_market(volume=None, supplemental=None):
    underlying = pl.DataFrame(
        {
            "symbol": ["SPX"] * 5,
            "datetime": [
                datetime(2024, 1, 5, 9, 30),
                datetime(2024, 1, 5, 9, 31),
                datetime(2024, 1, 8, 9, 30),
                datetime(2024, 1, 8, 9, 31),
                datetime(2024, 1, 8, 9, 32),
            ],
            "price": [99.0, 100.0, 101.0, 102.0, 103.0],
        }
    )
    if volume is not None:
        underlying = underlying.with_columns(
            pl.Series("volume", volume, dtype=pl.Float64)
        )
    return MarketData(underlying, pl.DataFrame(), supplemental)


def test_underlying_volume_sums_per_session_and_accumulates_only_observed_minutes():
    minute, daily = _preparation._underlying_history(
        _volume_market([2.0, 3.0, 5.0, 7.0, 11.0]), "SPX", frozenset({date(2024, 1, 8)})
    )
    assert daily["volume"].to_list() == [5.0, 23.0]
    assert minute["day_volume"].to_list() == [2.0, 5.0, 5.0, 12.0, 23.0]
    assert minute["volume"].to_list() == [2.0, 3.0, 5.0, 7.0, 11.0]


@pytest.mark.parametrize("unknown", [None, float("nan"), -1.0, float("inf")])
def test_missing_or_invalid_volume_never_becomes_a_partial_session_total(unknown):
    minute, daily = _preparation._underlying_history(
        _volume_market([2.0, 3.0, 5.0, unknown, 11.0]),
        "SPX",
        frozenset({date(2024, 1, 8)}),
    )
    assert daily["volume"][0] == 5.0
    assert math.isnan(daily["volume"][1])
    assert minute["day_volume"][:3].to_list() == [2.0, 5.0, 5.0]
    assert all(math.isnan(value) for value in minute["day_volume"][3:])


def test_supplemental_daily_volume_never_supplies_current_intraday_volume():
    supplemental = pl.DataFrame(
        {
            "symbol": ["SPX", "SPX"],
            "date": [date(2024, 1, 4), date(2024, 1, 8)],
            "open": [97.0, 101.0],
            "high": [100.0, 999.0],
            "low": [96.0, 100.0],
            "close": [99.0, 999.0],
            "volume": [1500.0, 999999.0],
        }
    )
    minute, daily = _preparation._underlying_history(
        _volume_market(supplemental=supplemental), "SPX", frozenset({date(2024, 1, 8)})
    )
    assert daily["volume"][0] == 1500.0
    assert math.isnan(daily["volume"][1])
    assert all(math.isnan(value) for value in minute["day_volume"])
    assert minute.filter(pl.col("date") == date(2024, 1, 8))["day_high"].to_list() == [
        101.0,
        102.0,
        103.0,
    ]


def test_supplemental_bars_without_volume_keep_unknown_historical_volume():
    supplemental = pl.DataFrame(
        {
            "symbol": ["SPX"],
            "date": [date(2024, 1, 4)],
            "open": [97.0],
            "high": [100.0],
            "low": [96.0],
            "close": [99.0],
        }
    )
    _minute, daily = _preparation._underlying_history(
        _volume_market([2.0, 3.0, 5.0, 7.0, 11.0], supplemental),
        "SPX",
        frozenset({date(2024, 1, 8)}),
    )
    assert math.isnan(daily["volume"][0])
    assert daily["volume"][1:].to_list() == [5.0, 23.0]


def test_price_only_data_keeps_ohlc_history_without_fabricating_volume():
    minute, daily = _preparation._underlying_history(
        _volume_market(), "SPX", frozenset({date(2024, 1, 8)})
    )
    assert daily.columns == ["date", "open", "high", "low", "close"]
    assert "volume" not in minute.columns
    assert "day_volume" not in minute.columns
