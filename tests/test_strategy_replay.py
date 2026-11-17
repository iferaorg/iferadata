"""Saved conditions recreate CSV trade decisions and accounting without evolution."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import torch
import yaml

from ifera.algorithms.evolutionary_dnf import EvolutionaryDNFOptimizer
from ifera.strategy_optimization import (
    apply_strategy_conditions,
    optimize_strategy_from_csv,
)
from ifera.strategy_optimization import _features, _reporting, metrics
from ifera.strategy_optimization import optimizer as optimizer_module
from ifera.strategy_optimization._conditions import (
    SavedConditions,
    load_conditions,
    save_conditions,
)
from ifera.strategy_optimization._data import MarketData
from ifera.strategy_optimization._evaluation import OptimizationMode, StrategyEvaluator
from ifera.strategy_optimization._features import FeatureSpec
from ifera.strategy_optimization._output import winning_trades
from ifera.strategy_optimization._preparation import prepare_strategy
from tests.test_strategy_optimization import strategy_market


class _OfflineSource:
    """Return fixture quotes remapped to whichever source rows were selected."""

    def __init__(self, original, market):
        self.original = original
        self.market = market
        self.calls = []
        self.closed = False

    def load(self, data, lookback_days=400):
        self.calls.append((data, lookback_days))
        original_indices = {
            opened: index
            for opened, index in self.original.trades.select(
                "opened", "trade_index"
            ).iter_rows()
        }
        mapping = {
            original_indices[opened]: index
            for opened, index in data.trades.select("opened", "trade_index").iter_rows()
        }
        options = self.market.options.filter(pl.col("trade_index").is_in(list(mapping)))
        options = options.with_columns(pl.col("trade_index").replace_strict(mapping))
        return MarketData(self.market.underlying, options, self.market.daily_history)

    def close(self):
        self.closed = True


def _features_for_manual_rules():
    return (
        FeatureSpec("monday", "bool", "any", "is_day", parameters={"day": "Monday"}),
        FeatureSpec("clock", "continuous", "any", "time", 0, 86400, granularity=60),
    )


def test_optimizer_csv_cannot_overwrite_its_input_feature_yaml(
    strategy_market, tmp_path
):
    data, market = strategy_market
    feature_file = tmp_path / "keep_features.yaml"
    feature_file.write_text(
        "features:\n  clock:\n    template: time\n", encoding="utf-8"
    )
    original = feature_file.read_bytes()
    source = _OfflineSource(data, market)

    with pytest.raises(ValueError, match="feature"):
        optimize_strategy_from_csv(
            "sample",
            "strategy",
            features_file=feature_file,
            output_csv=feature_file,
            market_data_source=source,
            optimizer_parameters={
                "device": "cpu",
                "max_population": 2,
                "generations": 0,
            },
            show_progress=False,
            show_plot=False,
            print_report=False,
        )

    assert source.calls == []
    assert feature_file.read_bytes() == original
    assert not feature_file.with_suffix(".conditions.yaml").exists()


def _manual_saved(mode):
    features = _features_for_manual_rules()
    clauses = 3 if mode == OptimizationMode.COMBINED else 1
    bounds = torch.zeros(clauses, len(features), 2)
    bounds[..., 1] = 1
    if mode == OptimizationMode.EARLY_EXIT:
        bounds[0, 0] = torch.tensor([0.0, 0.0])
        bounds[0, 1, 0] = (13 * 3600 + 2 * 60) / 86400
    else:
        bounds[0, 0] = torch.tensor([1.0, 1.0])
        bounds[0, 1] = torch.tensor([12 / 24, 14 / 24])
        if mode == OptimizationMode.COMBINED:
            bounds[1, 0] = torch.tensor([0.0, 0.0])
            bounds[1, 1, 0] = 14 / 24
            bounds[2, 0] = torch.tensor([1.0, 1.0])
            bounds[2, 1, 0] = (13 * 3600 + 2 * 60) / 86400
    return SavedConditions(
        mode=mode,
        features=features,
        bounds=bounds,
        active=torch.ones(clauses, dtype=torch.bool),
        exit_slippage=0.2,
        lookback_days=37,
    )


def _save(saved, path, source):
    assert save_conditions(saved, path, protected_paths=(source,)) == path
    return path


def _assert_unchanged(path, content, modified_ns):
    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == modified_ns


def _forbid_optimization(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError(
            "Applying saved rules must not optimize, score, or allocate"
        )

    monkeypatch.setattr(EvolutionaryDNFOptimizer, "__init__", unexpected)
    monkeypatch.setattr(EvolutionaryDNFOptimizer, "optimize", unexpected)
    monkeypatch.setattr(_reporting, "allocated_strategy_statistics", unexpected)
    monkeypatch.setattr(optimizer_module, "allocated_strategy_statistics", unexpected)
    monkeypatch.setattr(optimizer_module, "annualized_return", unexpected)
    for name in ("annualized_return", "sortino", "return_over_add"):
        monkeypatch.setattr(metrics, name, unexpected)


def _forbid_packaged_features(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Replay must use the frozen feature metadata")

    monkeypatch.setattr(_features, "load_features", unexpected)
    monkeypatch.setattr(optimizer_module, "load_features", unexpected)
    original_open = Path.open

    def protected_open(path, *arguments, **options):
        if path.name in ("default_features.yaml", "feature_templates.yaml"):
            raise AssertionError("Replay consulted current packaged feature files")
        return original_open(path, *arguments, **options)

    monkeypatch.setattr(Path, "open", protected_open)


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_manual_saved_rules_recreate_nontrivial_trade_ledger(
    strategy_market, tmp_path, monkeypatch, mode
):
    data, market = strategy_market
    saved = _manual_saved(mode)
    condition_file = _save(
        saved, tmp_path / f"manual_{mode.value}.conditions.yaml", data.path
    )
    original, original_mtime = data.path.read_bytes(), data.path.stat().st_mtime_ns
    rules, rules_mtime = condition_file.read_bytes(), condition_file.stat().st_mtime_ns
    prepared = prepare_strategy(data, market, saved.features, mode, saved.exit_slippage)
    evaluation = StrategyEvaluator(prepared.tensors, saved.features, mode).evaluate(
        saved.bounds[None], saved.active[None]
    )
    expected = winning_trades(prepared, evaluation)
    source = _OfflineSource(data, market)
    _forbid_optimization(monkeypatch)
    _forbid_packaged_features(monkeypatch)
    destination = tmp_path / f"manual_{mode.value}.csv"

    output = apply_strategy_conditions(
        str(data.path),
        str(condition_file),
        output_csv=destination,
        market_data_source=source,
        device="cpu",
        observation_batch_size=1,
    )

    assert output == destination
    actual = pl.read_csv(output)
    assert actual.equals(expected)
    assert len(source.calls) == 1
    assert source.calls[0][1] == 37
    assert not source.closed
    if mode == OptimizationMode.ENTRY_FILTER:
        assert actual["Custom"].to_list() == ["keep first"]
        assert actual["P/L"].to_list() == [400.0]
    elif mode == OptimizationMode.EARLY_EXIT:
        assert actual["Custom"].to_list() == ["keep first", "keep second"]
        assert actual["P/L"].to_list() == pytest.approx([400.0, 540.0])
        assert actual["Closed"][1] == "2024-01-10 13:02:00"
    else:
        assert actual["Custom"].to_list() == ["keep first"]
        assert actual["P/L"].to_list() == pytest.approx([200.0])
        assert actual["Closed"][0] == "2024-01-08 13:02:00"
    _assert_unchanged(data.path, original, original_mtime)
    _assert_unchanged(condition_file, rules, rules_mtime)


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_optimizer_automatically_exports_rules_that_recreate_output(
    strategy_market, tmp_path, monkeypatch, mode
):
    data, market = strategy_market
    source = _OfflineSource(data, market)
    feature_file = tmp_path / "selected_features.yaml"
    feature_file.write_text(
        "features:\n  monday:\n    template: is_day\n    day: Monday\n  clock:\n    template: time\n",
        encoding="utf-8",
    )
    original, original_mtime = data.path.read_bytes(), data.path.stat().st_mtime_ns
    result = optimize_strategy_from_csv(
        "sample",
        "strategy",
        mode,
        features_file=feature_file,
        optimizer_parameters={
            "device": "cpu",
            "max_population": 8,
            "generations": 1,
            "max_conjunctions": 3,
            "seed": 17,
        },
        metric=lambda returns: returns.sum(dim=1) + 101,
        exit_slippage=0.2,
        output_csv=tmp_path / f"winner_{mode.value}.csv",
        market_data_source=source,
        lookback_days=37,
        show_progress=False,
        show_plot=False,
        print_report=False,
    )
    assert result.conditions_yaml == result.output_csv.with_suffix(".conditions.yaml")
    assert result.conditions_yaml.is_file()
    saved = load_conditions(result.conditions_yaml)
    assert saved.mode == mode
    assert saved.lookback_days == 37
    assert saved.exit_slippage == 0.2
    assert saved.start_date == data.dates[0]
    assert saved.end_date == data.dates[-1]
    rules, rules_mtime = (
        result.conditions_yaml.read_bytes(),
        result.conditions_yaml.stat().st_mtime_ns,
    )
    _forbid_optimization(monkeypatch)
    _forbid_packaged_features(monkeypatch)

    output = apply_strategy_conditions(
        data.path, result.conditions_yaml, market_data_source=source, device="cpu"
    )

    assert output == tmp_path / f"winner_{mode.value}_recreated.csv"
    assert output.read_bytes() == result.output_csv.read_bytes()
    assert len(source.calls) == 2
    assert source.calls[1][1] == 37
    _assert_unchanged(data.path, original, original_mtime)
    _assert_unchanged(result.conditions_yaml, rules, rules_mtime)


@pytest.mark.parametrize(
    "mode,flags,expected",
    [
        (OptimizationMode.ENTRY_FILTER, [False], "empty"),
        (OptimizationMode.ENTRY_FILTER, [True], "original"),
        (OptimizationMode.EARLY_EXIT, [False], "original"),
        (OptimizationMode.EARLY_EXIT, [True], "early"),
        (OptimizationMode.COMBINED, [False, False], "empty"),
        (OptimizationMode.COMBINED, [True, False], "original"),
        (OptimizationMode.COMBINED, [False, True], "empty"),
        (OptimizationMode.COMBINED, [True, True], "early"),
    ],
)
def test_true_false_rules_replay_with_zero_features(
    strategy_market, tmp_path, monkeypatch, mode, flags, expected
):
    data, market = strategy_market
    saved = SavedConditions(
        mode=mode,
        features=(),
        bounds=torch.empty(len(flags), 0, 2),
        active=torch.tensor(flags),
        exit_slippage=0.2,
    )
    condition_file = _save(saved, tmp_path / "constant.conditions.yaml", data.path)
    document = yaml.safe_load(condition_file.read_text(encoding="utf-8"))
    phases = (
        ["entry_filter", "early_exit"]
        if mode == OptimizationMode.COMBINED
        else ["entry_filter" if mode == OptimizationMode.ENTRY_FILTER else "early_exit"]
    )
    for phase, flag in zip(phases, flags):
        assert document["rules"][phase] == ([{}] if flag else [])
    source = _OfflineSource(data, market)
    _forbid_optimization(monkeypatch)

    output = apply_strategy_conditions(
        data.path, condition_file, market_data_source=source, device="cpu"
    )

    assert output == tmp_path / "constant_recreated.csv"
    actual = pl.read_csv(output, schema_overrides=data.raw.schema)
    if expected == "empty":
        assert actual.is_empty()
        assert actual.columns == data.raw.columns
    elif expected == "original":
        assert actual.equals(data.raw)
    else:
        assert actual["P/L"].to_list() == pytest.approx([160.0, 240.0])
        assert actual["ROR"].to_list() == pytest.approx([10.0, 40.0])
        assert actual["Closed"].to_list() == [
            "2024-01-08 13:01:00",
            "2024-01-10 13:01:00",
        ]


def test_saved_date_window_and_lookback_are_applied_before_loading_market(
    strategy_market, tmp_path, monkeypatch
):
    data, market = strategy_market
    saved = SavedConditions(
        mode=OptimizationMode.ENTRY_FILTER,
        features=(),
        bounds=torch.empty(1, 0, 2),
        active=torch.tensor([True]),
        lookback_days=73,
        start_date=date(2024, 1, 9),
        end_date=date(2024, 1, 10),
    )
    conditions = _save(saved, tmp_path / "selected.conditions.yaml", data.path)
    source = _OfflineSource(data, market)
    _forbid_optimization(monkeypatch)

    output = apply_strategy_conditions(
        data.path, conditions, market_data_source=source, device="cpu"
    )

    selected, lookback = source.calls[0]
    assert lookback == 73
    assert selected.dates == (date(2024, 1, 9), date(2024, 1, 10))
    assert selected.trades["trade_index"].to_list() == [0]
    assert selected.legs["trade_index"].unique().to_list() == [0]
    assert pl.read_csv(output).equals(data.raw.tail(1))


def test_replay_accepts_a_csv_path_outside_the_portfolio_tree(
    strategy_market, tmp_path, monkeypatch
):
    data, market = strategy_market
    external = tmp_path / "independent input" / "external backtest.csv"
    external.parent.mkdir()
    external.write_bytes(data.path.read_bytes())
    content, modified = external.read_bytes(), external.stat().st_mtime_ns
    conditions = _save(
        _manual_saved(OptimizationMode.ENTRY_FILTER),
        tmp_path / "external.conditions.yaml",
        external,
    )
    source = _OfflineSource(data, market)
    _forbid_optimization(monkeypatch)

    output = apply_strategy_conditions(
        external, conditions, market_data_source=source, device="cpu"
    )

    assert source.calls[0][0].path == external
    assert pl.read_csv(output).equals(data.raw.head(1))
    _assert_unchanged(external, content, modified)


@pytest.mark.parametrize("collision", ["raw_csv", "conditions_yaml"])
def test_replay_path_collisions_fail_before_market_io(
    strategy_market, tmp_path, collision
):
    data, market = strategy_market
    conditions = _save(
        _manual_saved(OptimizationMode.COMBINED),
        tmp_path / "protected.conditions.yaml",
        data.path,
    )
    source = _OfflineSource(data, market)
    source_bytes, source_mtime = data.path.read_bytes(), data.path.stat().st_mtime_ns
    rules_bytes, rules_mtime = conditions.read_bytes(), conditions.stat().st_mtime_ns
    destination = data.path if collision == "raw_csv" else conditions

    with pytest.raises(ValueError):
        apply_strategy_conditions(
            data.path,
            conditions,
            output_csv=destination,
            market_data_source=source,
            device="cpu",
        )

    assert not source.calls
    _assert_unchanged(data.path, source_bytes, source_mtime)
    _assert_unchanged(conditions, rules_bytes, rules_mtime)


def test_automatic_rules_sidecar_cannot_overwrite_input_feature_yaml(
    strategy_market, tmp_path
):
    data, market = strategy_market
    destination = tmp_path / "winner.csv"
    feature_file = destination.with_suffix(".conditions.yaml")
    feature_file.write_text(
        "features:\n  clock:\n    template: time\n", encoding="utf-8"
    )
    content, modified = feature_file.read_bytes(), feature_file.stat().st_mtime_ns
    source = _OfflineSource(data, market)

    with pytest.raises(ValueError):
        optimize_strategy_from_csv(
            "sample",
            "strategy",
            features_file=feature_file,
            output_csv=destination,
            market_data_source=source,
            show_progress=False,
            show_plot=False,
            print_report=False,
        )

    assert not source.calls
    assert not destination.exists()
    _assert_unchanged(feature_file, content, modified)


@pytest.mark.parametrize("fail", [False, True])
def test_replay_disposes_owned_oracle_source_on_success_and_failure(
    strategy_market, tmp_path, monkeypatch, fail
):
    data, market = strategy_market
    spec = FeatureSpec(
        "rsi",
        "continuous",
        "any",
        "rsi",
        0,
        100,
        {"days": 14, "days_ago": 0},
        granularity=0.01,
    )
    saved = SavedConditions(
        mode=OptimizationMode.ENTRY_FILTER,
        features=(spec,),
        bounds=torch.tensor([[[0.4, 0.6]]]),
        active=torch.tensor([True]),
        lookback_days=37,
    )
    conditions = _save(saved, tmp_path / "oracle.conditions.yaml", data.path)
    source = _OfflineSource(data, market)
    events = []

    class OwnedOracle:
        def __init__(self, *, fill_daily_history):
            events.append(("created", fill_daily_history))

        def __enter__(self):
            events.append(("entered",))
            return self

        def __exit__(self, error_type, _error, _traceback):
            events.append(("closed", error_type))
            return False

        def load(self, selected, lookback_days=400):
            events.append(("loaded", lookback_days))
            if fail:
                raise RuntimeError("mock Oracle read failed")
            return source.load(selected, lookback_days=lookback_days)

    monkeypatch.setattr(optimizer_module, "OracleMarketDataSource", OwnedOracle)
    _forbid_optimization(monkeypatch)
    if fail:
        with pytest.raises(RuntimeError, match="mock Oracle read failed"):
            apply_strategy_conditions(data.path, conditions, device="cpu")
        assert events[-1] == ("closed", RuntimeError)
    else:
        output = apply_strategy_conditions(data.path, conditions, device="cpu")
        assert output.is_file()
        assert events[-1] == ("closed", None)
    assert events[:3] == [("created", True), ("entered",), ("loaded", 37)]


def test_public_package_exports_replay_helper():
    from ifera import strategy_optimization

    assert strategy_optimization.apply_strategy_conditions is apply_strategy_conditions
    assert "apply_strategy_conditions" in strategy_optimization.__all__
