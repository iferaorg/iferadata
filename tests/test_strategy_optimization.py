"""Strategy DNF evaluation, causal minute preparation, and CSV integration."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest
import torch

from ifera.settings import settings
from ifera.strategy_optimization import optimize_strategy_from_csv
from ifera.strategy_optimization._data import MarketData, load_strategy
from ifera.strategy_optimization._evaluation import (
    OptimizationMode,
    StrategyEvaluator,
    StrategyTensors,
)
from ifera.strategy_optimization._features import FeatureSpec
from ifera.strategy_optimization._output import (
    format_conditions,
    save_strategy,
    winning_trades,
)
from ifera.strategy_optimization._preparation import prepare_strategy


def _bounds(batch, conjunctions, features):
    values = torch.zeros(batch, conjunctions, features, 2)
    values[..., 1] = 1.0
    return values


@pytest.fixture
def observations():
    features = (
        FeatureSpec("is_monday", "bool", "any", "is_day", parameters={"day": "Monday"}),
        FeatureSpec("premium", "continuous", "entry", "position_premium", 0, 1000),
        FeatureSpec("ror", "continuous", "exit", "ror_pct", -100, 100),
    )
    tensors = StrategyTensors(
        entry_features=torch.tensor(
            [[1.0, 0.4, float("nan")], [0.0, 0.6, float("nan")]]
        ),
        exit_features=torch.tensor(
            [
                [1.0, float("nan"), 0.2],
                [1.0, float("nan"), 0.8],
                [0.0, float("nan"), 0.4],
                [0.0, float("nan"), 0.9],
            ]
        ),
        entry_ignored=torch.zeros(2, 3, dtype=torch.bool),
        exit_ignored=torch.zeros(4, 3, dtype=torch.bool),
        exit_trades=torch.tensor([0, 0, 1, 1]),
        exit_ror=torch.tensor([10.0, 20.0, 30.0, 40.0]),
        original_ror=torch.tensor([25.0, -50.0]),
        trade_days=torch.tensor([0, 2]),
        market_days=3,
    )
    return features, tensors


def test_entry_filter_ignores_exit_categories(observations):
    features, tensors = observations
    bounds = _bounds(1, 1, 3)
    bounds[0, 0, 0] = torch.tensor([1.0, 1.0])
    bounds[0, 0, 2] = torch.tensor([0.9, 1.0])
    result = StrategyEvaluator(tensors, features, "entry_filter").evaluate(
        bounds, torch.ones(1, 1, dtype=torch.bool)
    )
    assert result.entered.tolist() == [[True, False]]
    assert result.exit_indices.tolist() == [[-1, -1]]
    assert result.daily_ror.tolist() == [[25.0, 0.0, 0.0]]


def test_early_exit_uses_first_matching_minute_and_ignores_entry_categories(
    observations,
):
    features, tensors = observations
    bounds = _bounds(1, 1, 3)
    bounds[0, 0, 1] = torch.tensor([0.9, 1.0])
    bounds[0, 0, 2] = torch.tensor([0.5, 1.0])
    result = StrategyEvaluator(tensors, features, "early_exit", 1).evaluate(
        bounds, torch.ones(1, 1, dtype=torch.bool)
    )
    assert result.entered.tolist() == [[True, True]]
    assert result.exit_indices.tolist() == [[1, 3]]
    assert result.daily_ror.tolist() == [[20.0, 0.0, 40.0]]


def test_combined_odd_conjunctions_assign_extra_slot_to_entry(observations):
    features, tensors = observations
    bounds = _bounds(1, 3, 3)
    bounds[0, 1, 0] = torch.tensor([1.0, 1.0])
    bounds[0, 2, 2] = torch.tensor([0.5, 1.0])
    result = StrategyEvaluator(tensors, features, "combined").evaluate(
        bounds, torch.tensor([[False, True, True]])
    )
    assert result.entered.tolist() == [[True, False]]
    assert result.exit_indices.tolist() == [[1, -1]]
    assert result.daily_ror.tolist() == [[20.0, 0.0, 0.0]]


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_unrestricted_and_inactive_clause_semantics(observations, mode):
    features, tensors = observations
    bounds = _bounds(2, 3, 3)
    active = torch.tensor([[False, False, False], [True, False, False]])
    result = StrategyEvaluator(tensors, features, mode).evaluate(bounds, active)
    if mode == OptimizationMode.EARLY_EXIT:
        assert result.daily_ror.tolist() == [[25.0, 0.0, -50.0], [10.0, 0.0, 30.0]]
    else:
        assert result.daily_ror.tolist() == [[0.0, 0.0, 0.0], [25.0, 0.0, -50.0]]


def test_combined_single_clause_has_entry_and_no_exit_slots(observations):
    features, tensors = observations
    result = StrategyEvaluator(tensors, features, "combined").evaluate(
        _bounds(1, 1, 3), torch.ones(1, 1, dtype=torch.bool)
    )
    assert result.daily_ror.tolist() == [[25.0, 0.0, -50.0]]


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_chunked_and_batched_evaluation_is_independent_of_chunk_size(
    observations, mode
):
    features, tensors = observations
    bounds = _bounds(3, 3, 3)
    bounds[0, :, 0] = torch.tensor([1.0, 1.0])
    bounds[1, :, 2] = torch.tensor([0.5, 1.0])
    active = torch.tensor(
        [[True, True, True], [True, False, True], [False, False, False]]
    )
    full = StrategyEvaluator(tensors, features, mode, 512).evaluate(bounds, active)
    for chunk in (1, 2, 3):
        evaluator = StrategyEvaluator(tensors, features, mode, chunk)
        result = evaluator.evaluate(bounds, active)
        torch.testing.assert_close(result.entered, full.entered)
        torch.testing.assert_close(result.exit_indices, full.exit_indices)
        torch.testing.assert_close(result.daily_ror, full.daily_ror)
        for index in range(3):
            single = evaluator.evaluate(
                bounds[index : index + 1], active[index : index + 1]
            )
            torch.testing.assert_close(single.daily_ror[0], full.daily_ror[index])


def test_absent_leg_is_ignored_but_missing_indicator_fails_restricted_atom():
    bounds = _bounds(3, 1, 2)
    bounds[0, 0, 0] = torch.tensor([0.7, 0.9])
    bounds[1, 0, 1] = torch.tensor([0.7, 0.9])
    result = StrategyEvaluator._matches(
        bounds,
        torch.ones(3, 1, dtype=torch.bool),
        torch.tensor([[float("nan"), float("nan")]]),
        torch.tensor([[True, False]]),
        torch.tensor([True, True]),
    )
    assert result.tolist() == [[True], [False], [True]]


@pytest.fixture
def strategy_market(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    source = tmp_path / "results" / "portfolios" / "sample" / "strategy.csv"
    source.parent.mkdir(parents=True)
    stamps = [datetime(2024, 1, 8, 13), datetime(2024, 1, 10, 13)]
    raw = pl.DataFrame(
        {
            "Opened": [stamp.strftime("%Y-%m-%d %H:%M:%S") for stamp in stamps],
            "Closed": [
                (stamp + timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S")
                for stamp in stamps
            ],
            "Symbol": ["SPX", "SPX"],
            "Exp": [stamp.date().isoformat() for stamp in stamps],
            "Legs": [
                "long put 4970 / short put 4980",
                "long call 4970 / short call 4980",
            ],
            "Quantity": [2, 3],
            "Status": ["expired", "expired"],
            "P/L": [400.0, -300.0],
            "Risk": [1600.0, 600.0],
            "ROR": [25.0, -50.0],
            "Premium": [400.0, 600.0],
            "Reward/Risk": [25.0, 400.0],
            "Open Price": [2.0, 2.0],
            "Close Price": [0.0, 1.0],
            "Close Bid Price": [0.0, 0.9],
            "Close Ask Price": [0.0, 1.1],
            "Price at Close": [5003.0, 5003.0],
            "Return %": [100.0, -50.0],
            "Custom": ["keep first", "keep second"],
        }
    )
    raw.write_csv(source)
    data = load_strategy("sample", "strategy")
    underlying = []
    options = []
    for index, stamp in enumerate(stamps):
        underlying.append(
            {
                "symbol": "SPX",
                "datetime": stamp.replace(hour=9, minute=30),
                "price": 4990.0,
            }
        )
        for minute in range(4):
            instant = stamp + timedelta(minutes=minute)
            underlying.append(
                {"symbol": "SPX", "datetime": instant, "price": 5000.0 + minute}
            )
            long_mid, short_mid = (
                [(1.0, 3.0), (0.5, 1.5), (0.4, 1.2), (0.2, 0.2)][minute]
                if index == 0
                else [(3.0, 1.0), (4.0, 1.0), (5.0, 1.0), (6.0, 1.0)][minute]
            )
            for direction, strike, mid in (
                ("long", 4970.0, long_mid),
                ("short", 4980.0, short_mid),
            ):
                options.append(
                    {
                        "trade_index": index,
                        "datetime": instant,
                        "direction": direction,
                        "right": "put" if index == 0 else "call",
                        "strike": strike,
                        "bid": mid - 0.05,
                        "ask": mid + 0.05,
                        "mid": mid,
                        "delta": -0.2 if index == 0 else 0.2,
                    }
                )
    market = MarketData(pl.from_dicts(underlying), pl.from_dicts(options))
    return data, market


def _time_feature():
    return (FeatureSpec("time", "continuous", "any", "time", 0.0, 86400.0),)


def test_preparation_excludes_entry_and_original_close_from_early_exit(strategy_market):
    data, market = strategy_market
    prepared = prepare_strategy(data, market, _time_feature(), "early_exit")
    times = prepared.exits["datetime"].to_list()
    assert times == [
        datetime(2024, 1, 8, 13, 1),
        datetime(2024, 1, 8, 13, 2),
        datetime(2024, 1, 10, 13, 1),
        datetime(2024, 1, 10, 13, 2),
    ]
    result = StrategyEvaluator(
        prepared.tensors, prepared.features, "early_exit"
    ).evaluate(_bounds(1, 1, 1), torch.ones(1, 1, dtype=torch.bool))
    assert result.exit_indices.tolist() == [[0, 2]]
    assert result.daily_ror.tolist() == [[12.5, 0.0, 50.0]]


def test_combination_slippage_is_applied_once_for_credit_and_debit_quantities(
    strategy_market,
):
    data, market = strategy_market
    prepared = prepare_strategy(
        data, market, _time_feature(), "early_exit", exit_slippage=0.2
    )
    first = prepared.exits.filter(pl.col("datetime").dt.minute() == 1)
    assert first["liquidation_value"].to_list() == pytest.approx([-1.2, 2.8])
    assert first["profit"].to_list() == pytest.approx([160.0, 240.0])
    assert first["ror_pct"].to_list() == pytest.approx([10.0, 40.0])
    assert first["return_pct"].to_list() == pytest.approx([40.0, 40.0])
    result = StrategyEvaluator(
        prepared.tensors, prepared.features, "early_exit"
    ).evaluate(_bounds(1, 1, 1), torch.ones(1, 1, dtype=torch.bool))
    exported = winning_trades(prepared, result)
    assert exported.columns == data.raw.columns
    assert exported["Custom"].to_list() == data.raw["Custom"].to_list()
    assert exported["Quantity"].to_list() == [2, 3]
    assert exported["P/L"].to_list() == pytest.approx([160.0, 240.0])
    assert exported["ROR"].to_list() == pytest.approx([10.0, 40.0])
    assert exported["Return %"].to_list() == pytest.approx([40.0, 40.0])
    assert exported["Close Price"].to_list() == pytest.approx([1.2, 2.8])
    assert exported["Close Bid Price"].to_list() == pytest.approx([0.9, 2.9])
    assert exported["Close Ask Price"].to_list() == pytest.approx([1.1, 3.1])
    assert exported["Price at Close"].to_list() == [5001.0, 5001.0]
    assert exported["Status"].to_list() == ["closed", "closed"]
    assert exported["Risk"].to_list() == data.raw["Risk"].to_list()
    assert exported["Premium"].to_list() == data.raw["Premium"].to_list()


def test_inactive_exit_preserves_baseline_accounting_despite_requested_slippage(
    strategy_market,
):
    data, market = strategy_market
    prepared = prepare_strategy(
        data, market, _time_feature(), "early_exit", exit_slippage=5.0
    )
    result = StrategyEvaluator(
        prepared.tensors, prepared.features, "early_exit"
    ).evaluate(_bounds(1, 1, 1), torch.zeros(1, 1, dtype=torch.bool))
    assert result.daily_ror.tolist() == [[25.0, 0.0, -50.0]]
    assert winning_trades(prepared, result).equals(data.raw)


def test_all_filtered_trades_keep_output_header_and_schema(strategy_market):
    data, market = strategy_market
    prepared = prepare_strategy(data, market, _time_feature(), "entry_filter")
    result = StrategyEvaluator(
        prepared.tensors, prepared.features, "entry_filter"
    ).evaluate(_bounds(1, 1, 1), torch.zeros(1, 1, dtype=torch.bool))
    exported = winning_trades(prepared, result)
    assert exported.is_empty()
    assert exported.schema == data.raw.schema


def test_missing_minute_quotes_fail_without_forward_fill(strategy_market):
    data, market = strategy_market
    quotes = market.options.filter(
        ~((pl.col("trade_index") == 0) & (pl.col("datetime").dt.minute() == 1))
    )
    with pytest.raises(ValueError, match="Missing complete option minute"):
        prepare_strategy(
            data, MarketData(market.underlying, quotes), _time_feature(), "early_exit"
        )


def test_prepared_indicators_do_not_use_future_same_day_underlying(strategy_market):
    data, market = strategy_market
    features = (
        FeatureSpec(
            "cci", "continuous", "any", "cci", -200, 200, {"days": 2, "days_ago": 0}
        ),
        FeatureSpec(
            "momentum",
            "continuous",
            "any",
            "momentum",
            -100,
            100,
            {"days": 1, "days_ago": 0},
        ),
    )
    original = prepare_strategy(data, market, features, "early_exit")
    changed_underlying = market.underlying.with_columns(
        pl.when(pl.col("datetime") >= datetime(2024, 1, 10, 13, 2))
        .then(9000.0)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    changed = prepare_strategy(
        data, MarketData(changed_underlying, market.options), features, "early_exit"
    )
    torch.testing.assert_close(
        original.tensors.entry_features, changed.tensors.entry_features, equal_nan=True
    )
    torch.testing.assert_close(
        original.tensors.exit_features[:3],
        changed.tensors.exit_features[:3],
        equal_nan=True,
    )
    assert not torch.equal(
        original.tensors.exit_features[3], changed.tensors.exit_features[3]
    )


def test_format_conditions_omits_inactive_unrestricted_and_irrelevant_terms(
    observations,
):
    features, _ = observations
    bounds = _bounds(1, 3, 3)[0]
    bounds[0, 0] = torch.tensor([1.0, 1.0])
    bounds[0, 2] = torch.tensor([0.75, 1.0])
    bounds[1, 1] = torch.tensor([0.5, 1.0])
    bounds[2, 2] = torch.tensor([0.75, 1.0])
    text = format_conditions(
        bounds, torch.tensor([True, False, True]), features, "combined"
    )
    assert text == "Entry filter: (is_monday)\nEarly exit: (ror >= 50)"


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_facade_evolves_exports_and_supplies_full_calendar_to_custom_metric(
    strategy_market, tmp_path, mode
):
    data, market = strategy_market
    feature_file = tmp_path / "features.yaml"
    feature_file.write_text(
        "features:\n  clock:\n    template: time\n", encoding="utf-8"
    )
    original_bytes = data.path.read_bytes()
    seen = []

    class OfflineSource:
        def load(self, loaded, lookback_days=400):
            assert loaded.path == data.path
            assert lookback_days == 120
            return market

    def metric(daily):
        assert daily.ndim == 2
        assert daily.shape[1] == 3
        assert (daily[:, 1] == 0.0).all()
        seen.append(daily.clone())
        return (daily[:, 0] - daily[:, 2] + 1.0).clamp_min(1e-6)

    result = optimize_strategy_from_csv(
        "sample",
        "strategy",
        mode,
        features_file=feature_file,
        metric=metric,
        optimizer_parameters={
            "device": "cpu",
            "max_population": 2,
            "generations": 0,
            "max_conjunctions": 3,
            "seed": 7,
        },
        market_data_source=OfflineSource(),
        lookback_days=120,
        exit_slippage=0.2,
        observation_batch_size=1,
        show_progress=False,
        show_plot=False,
        print_report=False,
    )
    assert seen
    assert result.trades.equals(data.raw)
    assert result.allocation == 0.0
    assert result.daily_results["combined_return"].to_list() == [0.0, 0.0, 0.0]
    assert result.daily_results["strategy_return"].to_list() == [0.25, None, -0.5]
    assert result.objective_history == (76.0,)
    assert result.statistics.market_days == 3
    assert result.statistics.active_days == 0
    assert result.statistics.ending_equity == 1.0
    assert result.figure is None
    assert result.output_csv.exists()
    assert result.output_csv.parent != data.path.parent
    assert pl.read_csv(result.output_csv).equals(data.raw)
    assert data.path.read_bytes() == original_bytes


def test_output_cannot_overwrite_source(strategy_market):
    data, _ = strategy_market
    with pytest.raises(ValueError, match="differ"):
        save_strategy(data.raw, data.path, data.path)
