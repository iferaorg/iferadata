"""Self-contained rule snapshots preserve decisions and reject invalid artifacts."""

from dataclasses import replace
from datetime import date, datetime

import pytest
import torch
import yaml

from ifera.strategy_optimization import _conditions, _features
from ifera.strategy_optimization._conditions import (
    SavedConditions,
    load_conditions,
    save_conditions,
)
from ifera.strategy_optimization._evaluation import (
    OptimizationMode,
    StrategyEvaluator,
    StrategyTensors,
)
from ifera.strategy_optimization._features import FeatureSpec


def _features_and_tensors():
    features = (
        FeatureSpec("monday", "bool", "any", "is_day", parameters={"day": "Monday"}),
        FeatureSpec(
            "unused_event",
            "bool",
            "any",
            "event_day",
            parameters={"event_type": "FOMC Meeting", "days": 0},
        ),
        FeatureSpec(
            "risk", "continuous", "entry", "position_risk", 0, 1000, granularity=1
        ),
        FeatureSpec("gap", "continuous", "any", "gap", -10, 10),
        FeatureSpec(
            "ror",
            "continuous",
            "exit",
            "ror_pct",
            -100,
            100,
            {"component": "last"},
            0.01,
        ),
    )
    tensors = StrategyTensors(
        entry_features=torch.tensor(
            [
                [1, 0, 0.4, 0.2, float("nan")],
                [0, 0, 0.8, 0.7, float("nan")],
                [1, 0, 0.6, 0.9, float("nan")],
            ]
        ),
        exit_features=torch.tensor(
            [
                [1, 0, float("nan"), 0.3, 0.2],
                [1, 0, float("nan"), 0.8, 0.8],
                [0, 0, float("nan"), 0.7, 0.4],
                [0, 0, float("nan"), 0.5, 0.9],
                [1, 0, float("nan"), 0.9, 0.1],
                [1, 0, float("nan"), 0.2, 0.6],
            ]
        ),
        entry_ignored=torch.zeros(3, 5, dtype=torch.bool),
        exit_ignored=torch.zeros(6, 5, dtype=torch.bool),
        exit_trades=torch.tensor([0, 0, 1, 1, 2, 2]),
        exit_ror=torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        original_ror=torch.tensor([5.0, -10.0, 15.0]),
        trade_days=torch.tensor([0, 2, 3]),
        market_days=5,
    )
    return features, tensors


def _saved(mode=OptimizationMode.COMBINED):
    features, _tensors = _features_and_tensors()
    count = 5 if mode == OptimizationMode.COMBINED else 3
    bounds = torch.zeros(count, len(features), 2, dtype=torch.float32)
    bounds[..., 1] = 1
    bounds[0, 0] = torch.tensor([1.0, 1.0])
    bounds[0, 2] = torch.tensor([0.3, 1.0])
    bounds[0, 3] = torch.tensor([0.0, 0.73123455])
    bounds[0, 4] = torch.tensor([0.55, 1.0])
    bounds[1, 1] = torch.tensor([1.0, 1.0])
    bounds[2, 3] = torch.tensor([0.81, 0.93])
    active = torch.tensor([True, False, True] + ([True, False] if count == 5 else []))
    if count == 5:
        bounds[3, 2] = torch.tensor([0.0, 0.4])
        bounds[3, 3] = torch.tensor([0.3123456, 0.89])
        bounds[3, 4] = torch.tensor([0.55, 1.0])
    return SavedConditions(
        mode, features, bounds, active, 0.05, 200, date(2024, 1, 8), date(2024, 1, 12)
    )


def _tensors_for(features):
    original_features, tensors = _features_and_tensors()
    indices = [
        [feature.name for feature in original_features].index(feature.name)
        for feature in features
    ]
    return replace(
        tensors,
        entry_features=tensors.entry_features[:, indices],
        exit_features=tensors.exit_features[:, indices],
        entry_ignored=tensors.entry_ignored[:, indices],
        exit_ignored=tensors.exit_ignored[:, indices],
    )


def _write(path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _document(tmp_path):
    path = save_conditions(_saved(), tmp_path / "rules.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", list(OptimizationMode))
def test_sparse_roundtrip_preserves_entry_exits_and_daily_returns(tmp_path, mode):
    saved = _saved(mode)
    original = saved.bounds.clone()
    path = save_conditions(saved, tmp_path / "rules.yaml")
    document = yaml.safe_load(path.read_text())
    restored = load_conditions(path)
    before = StrategyEvaluator(
        _tensors_for(saved.features), saved.features, mode
    ).evaluate(saved.bounds[None], saved.active[None])
    after = StrategyEvaluator(
        _tensors_for(restored.features), restored.features, mode
    ).evaluate(restored.bounds[None], restored.active[None])

    assert document["version"] == 1
    assert document["threshold_units"] == "normalized"
    assert "unused_event" not in document["features"]
    assert restored.mode == mode
    assert restored.exit_slippage == 0.05
    assert restored.lookback_days == 200
    assert restored.start_date == date(2024, 1, 8)
    assert restored.end_date == date(2024, 1, 12)
    assert restored.bounds.device.type == restored.active.device.type == "cpu"
    assert restored.bounds.dtype == torch.float32
    assert restored.active.dtype == torch.bool
    for field in ("entered", "exit_indices", "daily_ror"):
        torch.testing.assert_close(
            getattr(after, field), getattr(before, field), rtol=0, atol=0
        )
    torch.testing.assert_close(saved.bounds, original, rtol=0, atol=0)
    if mode == OptimizationMode.COMBINED:
        assert restored.bounds.shape[0] == 4
        assert restored.active.tolist() == [True, True, True, False]
        assert len(document["rules"]["entry_filter"]) == 2
        assert len(document["rules"]["early_exit"]) == 1
    else:
        absent = (
            "early_exit" if mode == OptimizationMode.ENTRY_FILTER else "entry_filter"
        )
        assert document["rules"][absent] is None


@pytest.mark.parametrize("mode", list(OptimizationMode))
@pytest.mark.parametrize("enabled", [False, True])
def test_true_false_and_inactive_rules_need_no_feature_calculators(
    tmp_path, mode, enabled
):
    saved = _saved(mode)
    bounds = torch.zeros_like(saved.bounds)
    bounds[..., 1] = 1
    active = torch.full_like(saved.active, enabled)
    saved = replace(saved, bounds=bounds, active=active)
    path = save_conditions(saved, tmp_path / "constant.yaml")
    document = yaml.safe_load(path.read_text())
    restored = load_conditions(path)
    assert restored.features == ()
    assert document["features"] == {}
    assert restored.bounds.shape[1] == 0
    for phase in document["rules"].values():
        if phase is not None:
            assert all(clause == {} for clause in phase)
            assert bool(phase) is enabled
    before = StrategyEvaluator(
        _tensors_for(saved.features), saved.features, mode
    ).evaluate(saved.bounds[None], saved.active[None])
    after = StrategyEvaluator(
        _tensors_for(restored.features), restored.features, mode
    ).evaluate(restored.bounds[None], restored.active[None])
    torch.testing.assert_close(before.daily_ror, after.daily_ror, rtol=0, atol=0)


def test_float32_non_grid_endpoints_are_bit_exact_even_with_large_raw_offset(tmp_path):
    feature = FeatureSpec("offset", "continuous", "any", "gap", 1e20, 1e20 + 1e6)
    lower = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    upper = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    bounds = torch.stack((lower, upper)).reshape(1, 1, 2)
    saved = SavedConditions(
        OptimizationMode.ENTRY_FILTER, (feature,), bounds, torch.tensor([True])
    )
    restored = load_conditions(save_conditions(saved, tmp_path / "exact.yaml"))
    assert torch.equal(restored.bounds, bounds)
    assert restored.features[0].min_value == feature.min_value
    assert restored.features[0].max_value == feature.max_value


def test_boolean_false_and_endpoint_restrictions_keep_sentinel_meaning(tmp_path):
    features = _saved().features
    bounds = torch.zeros(2, len(features), 2)
    bounds[..., 1] = 1
    bounds[0, 0] = 0
    bounds[0, 3] = 0
    bounds[1, 0] = 1
    bounds[1, 3] = 1
    saved = SavedConditions(
        OptimizationMode.ENTRY_FILTER, features, bounds, torch.ones(2, dtype=torch.bool)
    )
    path = save_conditions(saved, tmp_path / "endpoints.yaml")
    document = yaml.safe_load(path.read_text())
    assert document["rules"]["entry_filter"] == [
        {"monday": False, "gap": {"max": 0.0}},
        {"monday": True, "gap": {"min": 1.0}},
    ]
    restored = load_conditions(path)
    assert restored.bounds[:, 0].tolist() == [[0, 0], [1, 1]]
    assert restored.bounds[:, 1].tolist() == [[0, 0], [1, 1]]


def test_saved_snapshots_ignore_changed_live_template_defaults_and_sort_booleans(
    tmp_path, monkeypatch
):
    document = _document(tmp_path)
    document["features"] = dict(reversed(list(document["features"].items())))
    document["human_readable"] = "Informational text may change without changing rules"
    monkeypatch.setattr(
        _features,
        "load_features",
        lambda *_args, **_kwargs: pytest.fail(
            "Live template defaults must not be read"
        ),
    )
    loaded = load_conditions(_write(tmp_path / "snapshot.yaml", document))
    assert loaded.features[0].kind == "bool"
    assert (
        next(feature for feature in loaded.features if feature.name == "risk").max_value
        == 1000
    )
    assert next(
        feature for feature in loaded.features if feature.name == "ror"
    ).parameters == {"component": "last"}


def test_snapshot_parameter_defaults_are_resolved_without_template_yaml(tmp_path):
    document = _document(tmp_path)
    document["features"]["ror"]["parameters"] = {}
    loaded = load_conditions(_write(tmp_path / "defaults.yaml", document))
    assert next(
        feature for feature in loaded.features if feature.name == "ror"
    ).parameters == {"component": "last"}


@pytest.mark.parametrize("version", [True, 1.0, 2, "1", None])
def test_invalid_versions_are_rejected(tmp_path, version):
    document = _document(tmp_path)
    document["version"] = version
    with pytest.raises(ValueError, match="version"):
        load_conditions(_write(tmp_path / "bad.yaml", document))


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        (None, "unknown", 1, "exactly fields"),
        (None, "threshold_units", "raw", "threshold_units"),
        (None, "human_readable", 1, "human_readable"),
        (None, "mode", "unknown", "valid OptimizationMode"),
        ("execution", "unknown", 1, "Execution settings"),
        ("execution", "lookback_days", 0, "lookback_days"),
        ("execution", "lookback_days", True, "lookback_days"),
        ("execution", "exit_slippage", -0.1, "exit_slippage"),
        ("execution", "start_date", "2025-01-01", "start_date"),
        ("rules", "entry_filter", None, "must be a list"),
        ("rules", "unknown", [], "exactly fields"),
    ],
)
def test_invalid_document_fields_are_rejected(tmp_path, section, key, value, match):
    document = _document(tmp_path)
    (document if section is None else document[section])[key] = value
    with pytest.raises((ValueError, TypeError), match=match):
        load_conditions(_write(tmp_path / "bad.yaml", document))


@pytest.mark.parametrize(
    "feature,key,value,match",
    [
        ("gap", "template", "unknown", "Unknown saved feature template"),
        ("gap", "type", "bool", "continuous type"),
        ("gap", "category", "invalid", "category"),
        ("gap", "max_value", -10, "increasing"),
        ("gap", "max_value", float("inf"), "finite"),
        ("gap", "parameters", {"unexpected": 1}, "requires parameters"),
        ("gap", "unknown", 1, "exactly fields"),
        ("monday", "granularity", 1, "continuous type"),
    ],
)
def test_invalid_feature_snapshots_are_rejected(tmp_path, feature, key, value, match):
    document = _document(tmp_path)
    document["features"][feature][key] = value
    with pytest.raises(ValueError, match=match):
        load_conditions(_write(tmp_path / "bad.yaml", document))


@pytest.mark.parametrize("value", [0, 1, "true", {"min": 1}])
def test_boolean_rules_require_actual_boolean_literals(tmp_path, value):
    document = _document(tmp_path)
    document["rules"]["entry_filter"][0]["monday"] = value
    with pytest.raises(ValueError, match="true or false"):
        load_conditions(_write(tmp_path / "bad.yaml", document))


@pytest.mark.parametrize(
    "value", [True, float("nan"), float("inf"), -0.01, 1.01, "0.5"]
)
def test_continuous_endpoints_require_finite_normalized_numbers(tmp_path, value):
    document = _document(tmp_path)
    document["rules"]["entry_filter"][0]["gap"] = {"min": value}
    with pytest.raises(ValueError, match="finite numbers"):
        load_conditions(_write(tmp_path / "bad.yaml", document))


@pytest.mark.parametrize("value", [{}, {"lower": 0.5}, {"min": 0.8, "max": 0.2}, 0.5])
def test_invalid_continuous_rule_structures_are_rejected(tmp_path, value):
    document = _document(tmp_path)
    document["rules"]["entry_filter"][0]["gap"] = value
    with pytest.raises(ValueError):
        load_conditions(_write(tmp_path / "bad.yaml", document))


def test_unknown_references_and_wrong_phase_categories_are_rejected(tmp_path):
    document = _document(tmp_path)
    document["rules"]["entry_filter"][0]["missing"] = True
    with pytest.raises(ValueError, match="unknown feature"):
        load_conditions(_write(tmp_path / "bad.yaml", document))
    del document["rules"]["entry_filter"][0]["missing"]
    document["rules"]["entry_filter"][0]["ror"] = {"min": 0.5}
    with pytest.raises(ValueError, match="cannot be used"):
        load_conditions(_write(tmp_path / "bad.yaml", document))


def test_unused_phase_must_be_null(tmp_path):
    path = save_conditions(
        _saved(OptimizationMode.ENTRY_FILTER), tmp_path / "entry.yaml"
    )
    document = yaml.safe_load(path.read_text())
    document["rules"]["early_exit"] = []
    with pytest.raises(ValueError, match="must be null"):
        load_conditions(_write(path, document))


@pytest.mark.parametrize(
    "problem", ["dtype", "nan", "range", "order", "boolean", "active", "shape"]
)
def test_invalid_save_inputs_do_not_create_output_or_directories(tmp_path, problem):
    saved = _saved()
    bounds = saved.bounds.clone()
    if problem == "dtype":
        saved = replace(saved, bounds=bounds.double())
    elif problem == "active":
        saved = replace(saved, active=saved.active.float())
    elif problem == "shape":
        saved = replace(saved, bounds=bounds[None])
    else:
        if problem == "nan":
            bounds[0, 3, 0] = float("nan")
        elif problem == "range":
            bounds[0, 3, 0] = -0.01
        elif problem == "order":
            bounds[0, 3] = torch.tensor([0.8, 0.2])
        else:
            bounds[0, 0] = torch.tensor([0.2, 1.0])
        saved = replace(saved, bounds=bounds)
    path = tmp_path / "uncreated" / "rules.yaml"
    with pytest.raises(ValueError):
        save_conditions(saved, path)
    assert not path.parent.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exit_slippage": -1},
        {"lookback_days": 0},
        {"start_date": datetime(2024, 1, 1)},
        {"start_date": date(2025, 1, 1)},
        {"features": (FeatureSpec("bad", "continuous", "any", "unknown"),)},
    ],
)
def test_invalid_saved_metadata_is_rejected_before_writing(tmp_path, kwargs):
    path = tmp_path / "rules.yaml"
    with pytest.raises((ValueError, TypeError)):
        save_conditions(replace(_saved(), **kwargs), path)
    assert not path.exists()


@pytest.mark.parametrize("suffix", [".csv", ".yaml"])
def test_conditions_cannot_overwrite_protected_source_or_export(tmp_path, suffix):
    path = tmp_path / ("protected" + suffix)
    path.write_text("original bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="protected"):
        save_conditions(_saved(), path, protected_paths=(path,))
    assert path.read_text() == "original bytes"


def test_atomic_failure_preserves_existing_conditions_and_cleans_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "rules.yaml"
    path.write_text("existing artifact", encoding="utf-8")

    def fail(*_args):
        raise OSError("replacement failed")

    monkeypatch.setattr(_conditions.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failed"):
        save_conditions(_saved(), path)
    assert path.read_text() == "existing artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_duplicate_keys_and_python_tags_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\nversion: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_conditions(path)
    path.write_text("!!python/object/apply:builtins.eval ['1 + 2']", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid conditions YAML"):
        load_conditions(path)
