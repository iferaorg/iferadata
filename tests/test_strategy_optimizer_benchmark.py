"""Deterministic benchmark workloads and reproducible CPU measurement records."""

from dataclasses import fields
import json

import pytest
import torch

from benchmarks.benchmark_strategy_optimization import (
    BenchmarkConfig,
    append_record,
    main,
    make_candidates,
    make_fixture,
    run_benchmark,
    source_hashes,
    summarize_result,
)
from ifera.strategy_optimization._evaluation import StrategyEvaluator


def _config(**overrides):
    return BenchmarkConfig(
        **{
            "device": "cpu",
            "trades": 4,
            "market_days": 5,
            "exit_minutes": 3,
            "batch": 3,
            "conjunctions": 3,
            "population": 4,
            "generations": 0,
            "metric_batch_size": 4,
            "observation_batch_size": 2,
            "objective_batch_size": 2,
            "warmups": 0,
            "repeats": 2,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"trades": 0},
        {"trades": 6},
        {"market_days": False},
        {"population": 1},
        {"exit_minutes": -1},
        {"generations": -1},
        {"warmups": -1},
        {"iterations": 0},
        {"repeats": 1.5},
        {"stage": "unknown"},
        {"metric": "unknown"},
        {"mode": "unknown"},
    ],
)
def test_invalid_workloads_are_rejected_before_measurement(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_fixture_is_seeded_and_keeps_trade_identity_and_absent_leg_masks():
    config = _config()
    first, features = make_fixture(config)
    second, _ = make_fixture(config)
    for field in fields(first):
        actual, expected = getattr(first, field.name), getattr(second, field.name)
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, equal_nan=True)
        else:
            assert actual == expected
    assert first.entry_features.shape == (4, len(features))
    assert first.exit_features.shape == (12, len(features))
    assert first.exit_trades.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert first.trade_days.unique().numel() == config.trades
    for index, feature in enumerate(features):
        if feature.template == "leg_delta":
            ignored = first.entry_ignored[:, index]
            assert ignored.sum() == 2
            assert torch.isnan(first.entry_features[ignored, index]).all()
    changed, _ = make_fixture(_config(seed=config.seed + 1))
    assert not torch.equal(first.original_ror, changed.original_ror)


@pytest.mark.parametrize("mode", ["entry_filter", "early_exit", "combined"])
def test_candidates_are_valid_and_evaluation_workload_is_batch_independent(mode):
    config = _config(mode=mode)
    tensors, features = make_fixture(config)
    bounds, active = make_candidates(config, features)
    assert ((bounds >= 0) & (bounds <= 1)).all()
    assert (bounds[..., 0] <= bounds[..., 1]).all()
    boolean_count = sum(feature.kind == "bool" for feature in features)
    assert (
        (bounds[:, :, :boolean_count] == 0) | (bounds[:, :, :boolean_count] == 1)
    ).all()
    assert not active[1].any()
    batched = StrategyEvaluator(tensors, features, mode, 2).evaluate(bounds, active)
    for index in range(config.batch):
        single = StrategyEvaluator(tensors, features, mode, 20).evaluate(
            bounds[index : index + 1], active[index : index + 1]
        )
        torch.testing.assert_close(single.daily_ror[0], batched.daily_ror[index])
        torch.testing.assert_close(single.entered[0], batched.entered[index])
        torch.testing.assert_close(single.exit_indices[0], batched.exit_indices[index])


@pytest.mark.parametrize("stage", ["evaluator", "evolution", "metrics"])
def test_cpu_measurements_record_timing_and_output_semantics(stage):
    config = _config(stage=stage)
    record = run_benchmark(config)
    assert record["device_name"] == "CPU"
    assert record["config"]["stage"] == stage
    assert record["config"]["seed"] == config.seed
    assert record["cold_ms"] > 0
    assert len(record["samples_ms"]) == config.repeats
    assert 0 < record["min_ms"] <= record["median_ms"] <= record["max_ms"]
    if stage == "evaluator":
        assert record["result"]["entered"] >= config.trades
    elif stage == "evolution":
        assert record["result"]["population_size"] == config.population
        assert record["result"]["evaluations"] == config.population
    else:
        assert record["result"]["shape"] == [config.metric_batch_size]


def test_append_preserves_each_experiment_and_rejects_nonlist_output(tmp_path):
    path = tmp_path / "nested" / "results.json"
    append_record(path, {"label": "baseline", "median_ms": 2.0})
    append_record(path, {"label": "change_one", "median_ms": 1.0})
    assert json.loads(path.read_text()) == [
        {"label": "baseline", "median_ms": 2.0},
        {"label": "change_one", "median_ms": 1.0},
    ]
    path.write_text("{}")
    with pytest.raises(ValueError, match="JSON list"):
        append_record(path, {})


def test_cli_writes_labeled_results_without_cuda(tmp_path, capsys):
    output = tmp_path / "measurements.json"
    main(
        [
            "--stage",
            "metrics",
            "--device",
            "cpu",
            "--label",
            "cpu_smoke",
            "--trades",
            "2",
            "--market-days",
            "4",
            "--metric-batch-size",
            "3",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )
    stored = json.loads(output.read_text())
    assert stored[0]["config"]["label"] == "cpu_smoke"
    assert stored[0]["result"]["shape"] == [3]
    assert json.loads(capsys.readouterr().out) == stored[0]


def test_unexpected_benchmark_result_fails_explicitly():
    with pytest.raises(TypeError, match="Unexpected benchmark output"):
        summarize_result(None)


def test_manifest_hashes_each_measured_source_file():
    manifest = source_hashes()
    assert len(manifest) == 4
    assert "ifera/strategy_optimization/_evaluation.py" in manifest
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in manifest.values())
