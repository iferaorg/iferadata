"""Reproduce isolated strategy optimizer measurements without database I/O.

Run with ``python -m benchmarks.benchmark_strategy_optimization``. Each label
appends one record to the output JSON, allowing one-change-at-a-time comparisons.
Fixture construction, device transfers, output checksums, and profiler export are
outside the timed region. CUDA timing synchronizes before and after each sample.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import polars as pl
import torch
from torch import Tensor

from ifera.algorithms.evolutionary_dnf import DNFOptimizationResult
from ifera.strategy_optimization._data import StrategyData
from ifera.strategy_optimization._evaluation import (
    OptimizationMode,
    StrategyEvaluation,
    StrategyEvaluator,
    StrategyTensors,
)
from ifera.strategy_optimization._features import FeatureSpec, load_features
from ifera.strategy_optimization._preparation import PreparedStrategy
from ifera.strategy_optimization.metrics import (
    annualized_return,
    return_over_add,
    sortino,
    with_scarcity_penalty,
)
from ifera.strategy_optimization.optimizer import _evolve

_METRICS: dict[str, Callable[[Tensor], Tensor]] = {
    "annualized_return": annualized_return,
    "sortino": sortino,
    "return_over_add": return_over_add,
    "scarcity": with_scarcity_penalty(return_over_add, 0.1, 0.4),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Dimensions and iteration controls recorded with every measurement."""

    stage: str = "evaluator"
    label: str = "baseline"
    device: str = "cuda"
    mode: str = "combined"
    trades: int = 960
    market_days: int = 970
    exit_minutes: int = 179
    batch: int = 32
    conjunctions: int = 8
    population: int = 128
    generations: int = 3
    observation_batch_size: int = 512
    objective_batch_size: int = 32
    metric_batch_size: int = 1024
    metric: str = "annualized_return"
    seed: int = 1729
    warmups: int = 2
    repeats: int = 5
    iterations: int = 1

    def __post_init__(self) -> None:
        if self.stage not in ("evaluator", "evolution", "metrics"):
            raise ValueError("stage must be evaluator, evolution, or metrics")
        OptimizationMode(self.mode)
        if self.metric not in _METRICS:
            raise ValueError(f"Unknown metric: {self.metric}")
        for name in (
            "trades",
            "market_days",
            "batch",
            "conjunctions",
            "population",
            "observation_batch_size",
            "objective_batch_size",
            "metric_batch_size",
            "repeats",
            "iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("exit_minutes", "generations", "warmups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.trades > self.market_days:
            raise ValueError("trades must not exceed market_days")
        if self.population < 2:
            raise ValueError("population must be at least two")


def make_fixture(
    config: BenchmarkConfig,
) -> tuple[StrategyTensors, tuple[FeatureSpec, ...]]:
    """Build reproducible CPU observations with the application's feature schema."""
    generator = torch.Generator().manual_seed(config.seed)
    features = load_features()
    count = len(features)
    boolean_count = sum(feature.kind == "bool" for feature in features)
    entry = torch.rand((config.trades, count), generator=generator)
    if boolean_count:
        entry[:, :boolean_count] = 0
        entry[
            torch.arange(config.trades), torch.arange(config.trades) % boolean_count
        ] = 1
    ignored = torch.zeros_like(entry, dtype=torch.bool)
    for index, feature in enumerate(features):
        if feature.template == "leg_delta":
            absent = (torch.arange(config.trades) % 2 == 0) == (
                feature.parameters["leg"].endswith("call")
            )
            ignored[:, index] = absent
            entry[absent, index] = torch.nan
        elif feature.template in ("rsi", "stoch", "cci", "adx", "momentum", "macd"):
            entry[: min(14, config.trades), index] = torch.nan
    samples = config.trades * config.exit_minutes
    exits = entry.repeat_interleave(config.exit_minutes, dim=0)
    if samples:
        noise = torch.randn((samples, count), generator=generator) * 0.025
        exits = (exits + noise).clamp_(0.0, 1.0)
        exits[:, :boolean_count] = entry[:, :boolean_count].repeat_interleave(
            config.exit_minutes, dim=0
        )
    original = torch.randn(config.trades, generator=generator) * 30 + 3
    original[::15] = -100
    tensors = StrategyTensors(
        entry_features=entry,
        exit_features=exits,
        entry_ignored=ignored,
        exit_ignored=ignored.repeat_interleave(config.exit_minutes, dim=0),
        exit_trades=torch.arange(config.trades).repeat_interleave(config.exit_minutes),
        exit_ror=torch.randn(samples, generator=generator) * 20 + 3,
        original_ror=original,
        trade_days=torch.linspace(0, config.market_days - 1, config.trades).long(),
        market_days=config.market_days,
    )
    return tensors, features


def make_candidates(
    config: BenchmarkConfig, features: tuple[FeatureSpec, ...]
) -> tuple[Tensor, Tensor]:
    """Construct mixed Boolean/continuous clauses with sparse restrictions."""
    generator = torch.Generator().manual_seed(config.seed + 1)
    shape = (config.batch, config.conjunctions, len(features), 2)
    bounds = torch.rand(shape, generator=generator).sort(dim=-1).values
    boolean_count = sum(feature.kind == "bool" for feature in features)
    bounds[:, :, :boolean_count] = (
        torch.randint(
            0,
            2,
            (config.batch, config.conjunctions, boolean_count, 2),
            generator=generator,
        )
        .float()
        .sort(dim=-1)
        .values
    )
    restricted = torch.rand(shape[:-1], generator=generator) < 0.3
    bounds[..., 0] = torch.where(restricted, bounds[..., 0], 0.0)
    bounds[..., 1] = torch.where(restricted, bounds[..., 1], 1.0)
    active = torch.rand((config.batch, config.conjunctions), generator=generator) < 0.25
    bounds[0, 0, :, 0] = 0
    bounds[0, 0, :, 1] = 1
    active[0, 0] = True
    if config.batch > 1:
        active[1] = False
    return bounds, active


def _operation(config: BenchmarkConfig, device: torch.device) -> Callable[[], object]:
    """Create the measured operation and perform all input transfers in advance."""
    metric = _METRICS[config.metric]
    if config.stage == "metrics":
        generator = torch.Generator().manual_seed(config.seed)
        returns = (
            torch.randn(
                (config.metric_batch_size, config.market_days), generator=generator
            )
            * 3
            + 0.2
        )
        returns[:, ::4] = 0
        resident_returns = returns.to(device)
        return lambda: metric(resident_returns)
    tensors, features = make_fixture(config)
    resident = tensors.to(device)
    if config.stage == "evaluator":
        bounds, active = make_candidates(config, features)
        bounds, active = bounds.to(device), active.to(device)
        evaluator = StrategyEvaluator(
            resident, features, config.mode, config.observation_batch_size
        )
        return lambda: evaluator.evaluate(bounds, active)
    dates = tuple(
        date(2020, 1, 1) + timedelta(days=index) for index in range(config.market_days)
    )
    data = StrategyData(
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), dates, Path("synthetic.csv")
    )
    prepared = PreparedStrategy(data, features, resident, pl.DataFrame())
    parameters = {
        "device": device,
        "seed": config.seed,
        "max_population": config.population,
        "generations": config.generations,
        "max_conjunctions": config.conjunctions,
        "objective_batch_size": config.objective_batch_size,
    }
    return lambda: _evolve(
        prepared,
        OptimizationMode(config.mode),
        metric,
        parameters,
        config.observation_batch_size,
        False,
    )[0]


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA only when measuring a CUDA workload."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_operation(
    operation: Callable[[], object], device: torch.device, iterations: int
) -> tuple[float, object]:
    """Measure synchronized wall time, including Python launch and callback costs."""
    _synchronize(device)
    started = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = operation()
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations, result


def summarize_result(result: object) -> dict[str, Any]:
    """Record numerical checks after timing to detect changed workload semantics."""
    if isinstance(result, StrategyEvaluation):
        return {
            "daily_ror_sum": float(result.daily_ror.double().sum().cpu()),
            "entered": int(result.entered.sum().cpu()),
            "early_exits": int((result.exit_indices >= 0).sum().cpu()),
        }
    if isinstance(result, DNFOptimizationResult):
        return {
            "objective": result.objective_value,
            "fitness": result.fitness,
            "evaluations": result.evaluations,
            "population_size": result.population_size,
        }
    if isinstance(result, Tensor):
        return {
            "score_sum": float(result.double().sum().cpu()),
            "shape": list(result.shape),
        }
    raise TypeError(f"Unexpected benchmark output: {type(result).__name__}")


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append an experiment result while preserving earlier measurements."""
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(records, list):
        raise ValueError("The benchmark output must contain a JSON list")
    records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def source_hashes() -> dict[str, str]:
    """Identify the source revision for each measured implementation component."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "ifera/strategy_optimization/_evaluation.py",
        "ifera/strategy_optimization/metrics.py",
        "ifera/strategy_optimization/optimizer.py",
        "ifera/algorithms/evolutionary_dnf.py",
    )
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in paths
    }


def run_benchmark(
    config: BenchmarkConfig, profile_path: Path | None = None
) -> dict[str, Any]:
    """Measure cold and warmed execution; optional profiling is outside timings."""
    device = torch.device(config.device)
    implementation = source_hashes()
    operation = _operation(config, device)
    with torch.no_grad():
        cold_ms, result = _time_operation(operation, device, 1)
        for _ in range(config.warmups):
            operation()
        _synchronize(device)
        allocated_before = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
        samples = []
        for _ in range(config.repeats):
            elapsed_ms, result = _time_operation(operation, device, config.iterations)
            samples.append(elapsed_ms)
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "torch_version": torch.__version__,
            "source_sha256": implementation,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "cold_ms": cold_ms,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
            "result": summarize_result(result),
        }
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            record["peak_allocated_bytes"] = peak
            record["extra_peak_allocated_bytes"] = max(0, peak - allocated_before)
        if profile_path is not None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True
            ) as profile:
                operation()
                _synchronize(device)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile.export_chrome_trace(str(profile_path))
            record["profile_path"] = str(profile_path)
            record["profile_summary"] = profile.key_averages().table(
                sort_by=(
                    "self_device_time_total"
                    if device.type == "cuda"
                    else "self_cpu_time_total"
                ),
                row_limit=15,
            )
    return record


def main(argv: Sequence[str] | None = None) -> None:
    """Run one independently labeled experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = BenchmarkConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(value), default=value
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/strategy-optimization.json"),
    )
    parser.add_argument("--profile", type=Path)
    arguments = vars(parser.parse_args(argv))
    output = arguments.pop("output")
    profile = arguments.pop("profile")
    record = run_benchmark(BenchmarkConfig(**arguments), profile)
    append_record(output, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
