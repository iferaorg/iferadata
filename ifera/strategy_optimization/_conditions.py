"""Save self-contained, sparse winning DNFs and restore them without evolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor
import yaml

from ._data import validate_lookback_days
from ._evaluation import OptimizationMode
from ._features import FeatureSpec, _PARAMETERS, _feature_spec, _mapping
from ._output import format_conditions
from ._preparation import validate_slippage
from ._thresholds import round_bounds

_TOP_FIELDS = {
    "version",
    "threshold_units",
    "mode",
    "execution",
    "features",
    "rules",
    "human_readable",
}
_FEATURE_FIELDS = {
    "template",
    "type",
    "category",
    "min_value",
    "max_value",
    "granularity",
    "parameters",
}
_EXECUTION_FIELDS = {"exit_slippage", "lookback_days", "start_date", "end_date"}
_PHASES = ("entry_filter", "early_exit")


@dataclass(frozen=True)
class SavedConditions:
    """Resolved calculators, normalized float32 rules, and replay settings.

    Saved artifacts retain effective active clauses and referenced features.
    Tensor padding and inactive genome contents are reconstructed rather than
    serialized; phase behavior and exact float32 restriction values are retained.
    """

    mode: OptimizationMode
    features: tuple[FeatureSpec, ...]
    bounds: Tensor
    active: Tensor
    exit_slippage: float = 0.0
    lookback_days: int = 400
    start_date: date | None = None
    end_date: date | None = None


def _fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    """Require exactly the fields declared by this artifact version."""
    result = _mapping(value, label)
    if set(result) != expected:
        raise ValueError(f"{label} requires exactly fields {sorted(expected)}")
    return result


def _date(value: object, label: str) -> date | None:
    """Accept optional ISO dates while rejecting datetimes and numeric dates."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be an ISO date or null") from exc
    raise ValueError(f"{label} must be an ISO date or null")


def _execution(value: object) -> dict[str, Any]:
    """Validate replay settings, allowing default settings in manual artifacts."""
    fields = _mapping(value, "Execution settings")
    if set(fields) - _EXECUTION_FIELDS:
        raise ValueError(f"Execution settings accept only {sorted(_EXECUTION_FIELDS)}")
    slippage = validate_slippage(fields.get("exit_slippage", 0.0))
    lookback = fields.get("lookback_days", 400)
    validate_lookback_days(lookback)
    start = _date(fields.get("start_date"), "start_date")
    end = _date(fields.get("end_date"), "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be after end_date")
    return {
        "exit_slippage": slippage,
        "lookback_days": lookback,
        "start_date": start,
        "end_date": end,
    }


def _snapshot_feature(name: str, snapshot: object) -> FeatureSpec:
    """Validate embedded metadata without consulting packaged template YAML."""
    fields = _fields(snapshot, _FEATURE_FIELDS, f"Feature snapshot {name}")
    template = fields["template"]
    if not isinstance(template, str) or template not in _PARAMETERS:
        raise ValueError(f"Unknown saved feature template: {template!r}")
    parameters = _mapping(fields["parameters"], f"Feature {name} parameters")
    metadata = {
        key: fields[key] for key in _FEATURE_FIELDS - {"template", "parameters"}
    }
    metadata["parameters"] = sorted(_PARAMETERS[template])
    return _feature_spec(
        name, {"template": template, **parameters}, {template: metadata}
    )


def _feature_snapshot(feature: FeatureSpec) -> dict[str, Any]:
    """Build a fully resolved, human-readable calculator snapshot."""
    return {
        "template": feature.template,
        "type": feature.kind,
        "category": feature.category,
        "min_value": feature.min_value,
        "max_value": feature.max_value,
        "granularity": feature.granularity,
        "parameters": dict(feature.parameters),
    }


def _validated_features(features: tuple[FeatureSpec, ...]) -> tuple[FeatureSpec, ...]:
    """Validate manually constructed specs as strictly as loaded snapshots."""
    names = set()
    resolved = []
    for feature in features:
        if not isinstance(feature, FeatureSpec):
            raise ValueError("Saved features must contain FeatureSpec instances")
        if not isinstance(feature.name, str):
            raise ValueError("Saved feature names must be strings")
        if feature.name in names:
            raise ValueError(f"Duplicate saved feature name: {feature.name}")
        names.add(feature.name)
        resolved.append(_snapshot_feature(feature.name, _feature_snapshot(feature)))
    return tuple(resolved)


def _validated_tensors(saved: SavedConditions) -> tuple[Tensor, Tensor]:
    """Require the EA's finite normalized float32 representation before saving."""
    bounds, active = saved.bounds, saved.active
    if not isinstance(bounds, Tensor) or bounds.dtype != torch.float32:
        raise ValueError("Saved bounds must be a float32 tensor")
    if (
        bounds.ndim != 3
        or bounds.shape[0] < 1
        or bounds.shape[1:] != (len(saved.features), 2)
    ):
        raise ValueError("Saved bounds must have shape (clauses, features, 2)")
    if (
        not isinstance(active, Tensor)
        or active.dtype != torch.bool
        or active.shape != bounds.shape[:1]
    ):
        raise ValueError("Saved active clauses must be a Boolean (clauses,) tensor")
    bounds = bounds.detach().cpu()
    active = active.detach().cpu()
    if not bool(torch.isfinite(bounds).all()) or bool(
        ((bounds < 0) | (bounds > 1)).any()
    ):
        raise ValueError("Saved bounds must be finite and normalized to [0, 1]")
    if bool((bounds[..., 0] > bounds[..., 1]).any()):
        raise ValueError("Saved lower bounds must not exceed upper bounds")
    for index, feature in enumerate(saved.features):
        if feature.kind == "bool" and bool(
            ((bounds[:, index] != 0) & (bounds[:, index] != 1)).any()
        ):
            raise ValueError(
                f"Boolean feature {feature.name} requires 0/1 interval bounds"
            )
    return bounds, active


def _phase_rules(
    bounds: Tensor, active: Tensor, features: tuple[FeatureSpec, ...], category: str
) -> list[dict[str, Any]]:
    """Omit inactive clauses, unrestricted endpoints, and phase-irrelevant atoms."""
    clauses = []
    for intervals, enabled in zip(bounds.tolist(), active.tolist()):
        if not enabled:
            continue
        clause: dict[str, Any] = {}
        for (lower, upper), feature in zip(intervals, features):
            if feature.category not in (category, "any") or (lower == 0 and upper == 1):
                continue
            if feature.kind == "bool":
                clause[feature.name] = lower == 1
            else:
                limits = {}
                if lower > 0:
                    limits["min"] = lower
                if upper < 1:
                    limits["max"] = upper
                clause[feature.name] = limits
        clauses.append(clause)
    return clauses


def _rules(
    mode: OptimizationMode,
    bounds: Tensor,
    active: Tensor,
    features: tuple[FeatureSpec, ...],
) -> dict[str, list[dict[str, Any]] | None]:
    """Retain explicit phase separation so unequal clause counts remain safe."""
    if mode == OptimizationMode.COMBINED:
        split = (bounds.shape[0] + 1) // 2
        return {
            "entry_filter": _phase_rules(
                bounds[:split], active[:split], features, "entry"
            ),
            "early_exit": _phase_rules(
                bounds[split:], active[split:], features, "exit"
            ),
        }
    return {
        "entry_filter": (
            _phase_rules(bounds, active, features, "entry")
            if mode == OptimizationMode.ENTRY_FILTER
            else None
        ),
        "early_exit": (
            _phase_rules(bounds, active, features, "exit")
            if mode == OptimizationMode.EARLY_EXIT
            else None
        ),
    }


def _document(saved: SavedConditions) -> dict[str, Any]:
    """Validate and serialize only effective rules and the calculators they need."""
    mode = OptimizationMode(saved.mode)
    features = _validated_features(saved.features)
    execution = _execution({name: getattr(saved, name) for name in _EXECUTION_FIELDS})
    bounds, active = _validated_tensors(saved)
    bounds = round_bounds(bounds, features)
    rules = _rules(mode, bounds, active, features)
    names = {
        name
        for clauses in rules.values()
        if clauses is not None
        for clause in clauses
        for name in clause
    }
    snapshots = {
        feature.name: _feature_snapshot(feature)
        for feature in features
        if feature.name in names
    }
    return {
        "version": 1,
        "threshold_units": "normalized",
        "mode": mode.value,
        "execution": {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in execution.items()
        },
        "features": snapshots,
        "rules": rules,
        "human_readable": format_conditions(bounds, active, features, mode),
    }


def _endpoint(value: object) -> float:
    """Reject YAML booleans and nonfinite/out-of-range continuous endpoints."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Continuous rule bounds must be finite numbers in [0, 1]")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError("Continuous rule bounds must be finite numbers in [0, 1]")
    return numeric


def _interval(value: object, feature: FeatureSpec) -> tuple[float, float]:
    """Convert one literal or inclusive interval to normalized tensor endpoints."""
    if feature.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"Boolean rule {feature.name} must be true or false")
        return float(value), float(value)
    limits = _mapping(value, f"Continuous rule {feature.name}")
    if not limits or set(limits) - {"min", "max"}:
        raise ValueError(f"Continuous rule {feature.name} accepts min and/or max")
    lower, upper = _endpoint(limits.get("min", 0.0)), _endpoint(limits.get("max", 1.0))
    if lower > upper:
        raise ValueError(f"Continuous rule {feature.name} min must not exceed max")
    return lower, upper


def _load_rules(
    mode: OptimizationMode, rules: object, features: tuple[FeatureSpec, ...]
) -> tuple[Tensor, Tensor]:
    """Reconstruct padded phase slots with Boolean-first feature ordering."""
    phases = _fields(rules, set(_PHASES), "Rules")
    enabled = set(_PHASES) if mode == OptimizationMode.COMBINED else {mode.value}
    for phase, clauses in phases.items():
        if (phase in enabled and not isinstance(clauses, list)) or (
            phase not in enabled and clauses is not None
        ):
            raise ValueError(
                f"Rules for {phase} must be {'a list' if phase in enabled else 'null'}"
            )
    width = max(1, *(len(phases[phase]) for phase in enabled))
    count = width * 2 if mode == OptimizationMode.COMBINED else width
    bounds = torch.zeros(count, len(features), 2, dtype=torch.float32)
    bounds[..., 1] = 1
    active = torch.zeros(count, dtype=torch.bool)
    by_name = {feature.name: (index, feature) for index, feature in enumerate(features)}
    for phase in enabled:
        offset = (
            width if mode == OptimizationMode.COMBINED and phase == "early_exit" else 0
        )
        category = "entry" if phase == "entry_filter" else "exit"
        for clause_index, definition in enumerate(phases[phase]):
            clause = _mapping(definition, f"{phase} clause")
            active[offset + clause_index] = True
            for name, value in clause.items():
                if name not in by_name:
                    raise ValueError(f"Rule references an unknown feature: {name!r}")
                feature_index, feature = by_name[name]
                if feature.category not in (category, "any"):
                    raise ValueError(f"Feature {name} cannot be used in {phase}")
                bounds[offset + clause_index, feature_index] = torch.tensor(
                    _interval(value, feature), dtype=torch.float32
                )
    return bounds, active


def _unique_keys(node: yaml.Node | None, seen: set[int]) -> None:
    """Reject duplicate or non-string YAML mapping keys, including nested ones."""
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, yaml.MappingNode):
        keys = set()
        for key, value in node.value:
            if (
                not isinstance(key, yaml.ScalarNode)
                or key.tag != "tag:yaml.org,2002:str"
            ):
                raise ValueError("Conditions YAML requires string mapping keys")
            if key.value in keys:
                raise ValueError(f"Duplicate conditions YAML key: {key.value}")
            keys.add(key.value)
            _unique_keys(value, seen)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _unique_keys(child, seen)


def load_conditions(path: Path | str) -> SavedConditions:
    """Load a versioned snapshot using safe YAML and strict schema validation."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        _unique_keys(yaml.compose(text, Loader=yaml.SafeLoader), set())
        document = _fields(yaml.safe_load(text), _TOP_FIELDS, "Conditions document")
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid conditions YAML: {path}") from exc
    if (
        isinstance(document["version"], bool)
        or not isinstance(document["version"], int)
        or document["version"] != 1
    ):
        raise ValueError("Unsupported conditions version; expected version 1")
    if document["threshold_units"] != "normalized":
        raise ValueError("Conditions threshold_units must be normalized")
    if not isinstance(document["human_readable"], str):
        raise ValueError("Conditions human_readable must be a string")
    mode = OptimizationMode(document["mode"])
    snapshots = _mapping(document["features"], "Feature snapshots")
    features = tuple(
        sorted(
            (_snapshot_feature(name, value) for name, value in snapshots.items()),
            key=lambda feature: feature.kind != "bool",
        )
    )
    bounds, active = _load_rules(mode, document["rules"], features)
    execution = _execution(document["execution"])
    return SavedConditions(mode, features, bounds, active, **execution)


def save_conditions(
    saved: SavedConditions,
    path: Path | str,
    *,
    protected_paths: tuple[Path, ...] = (),
) -> Path:
    """Atomically save validated rules while protecting source/output CSV paths."""
    target = Path(path)
    if any(target.resolve() == protected.resolve() for protected in protected_paths):
        raise ValueError("Conditions output must differ from protected CSV/YAML paths")
    document = _document(saved)
    serialized = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
