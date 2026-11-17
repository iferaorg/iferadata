"""Validated YAML feature definitions and point-in-time feature calculation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml

from . import _indicators
from ._events import canonical_event_type, event_flags
from ._feature_indicators import (
    BOOLEAN_INDICATORS,
    INDICATOR_TEMPLATES,
    indicator_feature,
)

_PARAMETERS = {
    "is_day": {"day"},
    "time": set(),
    "rsi": {"days", "days_ago"},
    "stoch": {"days", "smooth_k", "smooth_d", "component", "days_ago"},
    "cci": {"days", "days_ago"},
    "adx": {"days", "days_ago"},
    "momentum": {"days", "days_ago"},
    "macd": {"fast", "slow", "signal", "component", "days_ago"},
    "change_since_open": set(),
    "change_since_prev_close": set(),
    "gap": set(),
    "leg_delta": {"leg"},
    "position_premium": set(),
    "position_risk": set(),
    "position_reward_risk": set(),
    "minutes_since_open": set(),
    "return_pct": {"component"},
    "ror_pct": {"component"},
    "underlying_above_ma": {"days", "ma_type"},
    "ma_comparison": {"days1", "days2", "ma_type"},
    "event_day": {"event_type", "days"},
    "stoch_rsi": {"days", "rsi_days", "smooth_k", "smooth_d", "component", "days_ago"},
    "williams_r": {"days", "days_ago"},
    "ultimate": {"short", "medium", "long", "days_ago"},
    "mfi": {"days", "days_ago"},
    "chande": {"days", "days_ago"},
    "macd_above_signal": {"fast", "slow", "signal", "days_ago"},
    "position_bid_ask_spread": set(),
    "position_itm_pct": set(),
}
_PARAMETER_DEFAULTS = {
    "return_pct": {"component": "last"},
    "ror_pct": {"component": "last"},
}
_BOOLEAN_TEMPLATES = {"is_day", "event_day"} | BOOLEAN_INDICATORS
_OBSERVATION_COLUMNS = {
    "position_premium": "premium",
    "position_risk": "risk",
    "position_reward_risk": "reward_risk",
    "minutes_since_open": "minutes_open",
    "return_pct": "return_pct",
    "ror_pct": "ror_pct",
    "position_bid_ask_spread": "position_bid_ask_spread",
    "position_itm_pct": "position_itm_pct",
}
_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")
_LEGS = ("short_call", "short_put", "long_call", "long_put")


@dataclass(frozen=True)
class FeatureSpec:
    """One normalized feature in optimizer order (Boolean features come first).

    NaN represents unavailable data. An absent option-leg delta is ignored when
    evaluating a condition. Other unavailable features cannot satisfy a
    restriction; unrestricted conditions do not require feature availability.
    """

    name: str
    kind: str
    category: str
    template: str
    min_value: float = 0.0
    max_value: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    granularity: float | None = None

    def __post_init__(self) -> None:
        """Require optional continuous-feature precision in original feature units."""
        if self.granularity is None:
            return
        if self.kind != "continuous":
            raise ValueError(
                f"Feature {self.name} granularity requires continuous type"
            )
        if (
            isinstance(self.granularity, bool)
            or not isinstance(self.granularity, Real)
            or not math.isfinite(self.granularity)
            or self.granularity <= 0
        ):
            raise ValueError(
                f"Feature {self.name} granularity must be a finite positive number"
            )

    @property
    def needs_daily_history(self) -> bool:
        """Identify features requiring completed sessions before observation."""
        return self.template in INDICATOR_TEMPLATES or self.template in (
            "change_since_prev_close",
            "gap",
        )

    def normalize(self, values: _indicators.FloatArray) -> _indicators.FloatArray:
        """Clip finite observations to the configured inclusive unit interval."""
        values = np.where(np.isfinite(values), values, np.nan)
        return np.clip(
            (values - self.min_value) / (self.max_value - self.min_value), 0.0, 1.0
        )


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Require a YAML mapping with string keys."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return value


def _validate_parameters(template: str, parameters: Mapping[str, Any]) -> None:
    """Validate calculator arguments before an optimization starts."""
    if template not in _PARAMETERS:
        raise ValueError(f"Unknown feature template: {template!r}")
    expected = _PARAMETERS[template]
    if set(parameters) != expected:
        raise ValueError(f"{template} requires parameters {sorted(expected)}")
    _validate_periods(template, parameters)
    _validate_components(template, parameters)
    if template in ("rsi", "stoch_rsi"):
        period = (
            parameters["rsi_days"] if template == "stoch_rsi" else parameters["days"]
        )
        if period < 2:
            raise ValueError(f"{template} RSI period must be at least 2 closing prices")
    if (
        template in ("macd", "macd_above_signal")
        and parameters["fast"] >= parameters["slow"]
    ):
        raise ValueError(f"{template}.fast must be smaller than {template}.slow")
    if template == "is_day" and str(parameters["day"]).lower() not in _DAYS:
        raise ValueError("is_day.day must name a weekday from Monday to Friday")
    if template == "leg_delta" and parameters["leg"] not in _LEGS:
        raise ValueError(f"leg_delta.leg must be one of {_LEGS}")
    if template == "event_day":
        _validate_event(parameters)


def _validate_periods(template: str, parameters: Mapping[str, Any]) -> None:
    """Validate indicator periods separately from signed market-session offsets."""
    for key in (
        "days",
        "days1",
        "days2",
        "rsi_days",
        "days_ago",
        "fast",
        "slow",
        "signal",
        "smooth_k",
        "smooth_d",
        "short",
        "medium",
        "long",
    ):
        if key not in parameters:
            continue
        if template == "event_day" and key == "days":
            continue
        value = parameters[key]
        minimum = 0 if key == "days_ago" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{template}.{key} must be an integer >= {minimum}")


def _validate_components(template: str, parameters: Mapping[str, Any]) -> None:
    """Validate finite choices for averages, oscillators, and position extrema."""
    if "ma_type" in parameters and parameters["ma_type"] not in (
        "SMA",
        "EMA",
        "TRIMA",
        "KAMA",
    ):
        raise ValueError("ma_type must be SMA, EMA, TRIMA, or KAMA")
    if template == "macd":
        if parameters["component"] not in ("line", "signal", "histogram"):
            raise ValueError("macd.component must be line, signal, or histogram")
    if template in ("stoch", "stoch_rsi") and parameters["component"] not in ("k", "d"):
        raise ValueError(f"{template}.component must be k or d")
    if template in ("return_pct", "ror_pct") and parameters.get(
        "component", "last"
    ) not in ("last", "high", "low"):
        raise ValueError(f"{template}.component must be last, high, or low")


def _validate_event(parameters: Mapping[str, Any]) -> None:
    """Require a supported event and a bounded signed market-session offset."""
    canonical_event_type(parameters["event_type"])
    days = parameters["days"]
    if isinstance(days, bool) or not isinstance(days, int) or not -10 <= days <= 10:
        raise ValueError("event_day.days must be an integer in [-10, 10]")


def _feature_spec(
    name: str, definition: object, templates: dict[str, Any]
) -> FeatureSpec:
    """Resolve and validate a configured instance against its template."""
    if not name.strip():
        raise ValueError("Feature names must not be empty")
    instance = _mapping(definition, f"Feature {name}").copy()
    template = instance.pop("template", None)
    if not isinstance(template, str) or template not in templates:
        raise ValueError(f"Feature {name} references an unknown template: {template!r}")
    metadata = _mapping(templates[template], f"Template {template}")
    unknown = set(metadata) - {
        "type",
        "category",
        "min_value",
        "max_value",
        "parameters",
        "granularity",
    }
    if unknown:
        raise ValueError(
            f"Template {template} contains unknown fields: {sorted(unknown)}"
        )
    declared_parameters = metadata.get("parameters", [])
    instance = {**_PARAMETER_DEFAULTS.get(template, {}), **instance}
    if not isinstance(declared_parameters, list) or set(declared_parameters) != set(
        instance
    ):
        raise ValueError(f"Feature {name} requires parameters {declared_parameters}")
    _validate_parameters(template, instance)
    kind, category = metadata.get("type"), metadata.get("category")
    if kind not in ("bool", "continuous"):
        raise ValueError(f"Feature {name} type must be bool or continuous")
    if category not in ("entry", "exit", "any"):
        raise ValueError(f"Feature {name} category must be entry, exit, or any")
    if template in _BOOLEAN_TEMPLATES and kind != "bool":
        raise ValueError(f"{template} requires Boolean type")
    if template not in _BOOLEAN_TEMPLATES and kind != "continuous":
        raise ValueError(f"{template} requires continuous type")
    if kind == "continuous" and not {"min_value", "max_value"} <= set(metadata):
        raise ValueError(f"Feature {name} requires min_value and max_value")
    minimum, maximum = metadata.get("min_value", 0.0), metadata.get("max_value", 1.0)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (minimum, maximum)
    ):
        raise ValueError(f"Feature {name} normalization bounds must be finite numbers")
    if minimum >= maximum or (kind == "bool" and (minimum, maximum) != (0, 1)):
        raise ValueError(f"Feature {name} requires increasing normalization bounds")
    required_category = (
        "entry"
        if template in ("position_premium", "position_risk", "position_reward_risk")
        else (
            "exit"
            if template
            in ("minutes_since_open", "return_pct", "ror_pct", "position_itm_pct")
            else None
        )
    )
    if required_category is not None and category != required_category:
        raise ValueError(f"{template} requires category {required_category}")
    return FeatureSpec(
        name,
        kind,
        category,
        template,
        float(minimum),
        float(maximum),
        instance,
        metadata.get("granularity"),
    )


def load_features(path: Path | str | None = None) -> tuple[FeatureSpec, ...]:
    """Read defaults or a user YAML file, returning Boolean-first feature order.

    Templates are package-owned metadata in ``feature_templates.yaml``. A user
    file contains only ``features`` referencing those templates; only explicitly
    listed feature instances are included.
    """
    template_path = Path(__file__).with_name("feature_templates.yaml")
    with template_path.open(encoding="utf-8") as stream:
        template_configuration = _mapping(
            yaml.safe_load(stream), "Packaged feature templates"
        )
    if set(template_configuration) != {"feature_templates"}:
        raise ValueError("Template configuration accepts only feature_templates")
    templates = _mapping(template_configuration["feature_templates"], "Templates")
    feature_path = (
        Path(__file__).with_name("default_features.yaml")
        if path is None
        else Path(path)
    )
    with feature_path.open(encoding="utf-8") as stream:
        configuration = _mapping(yaml.safe_load(stream), "Feature configuration")
    if set(configuration) - {"features"}:
        raise ValueError("Feature configuration accepts only features")
    definitions = _mapping(configuration.get("features"), "Features")
    if not definitions:
        raise ValueError("At least one feature must be configured")
    specs = [
        _feature_spec(name, value, templates) for name, value in definitions.items()
    ]
    return tuple(sorted(specs, key=lambda spec: spec.kind != "bool"))


def _column(observations: pl.DataFrame, name: str) -> _indicators.FloatArray:
    """Get a numeric observation column without converting nulls into zeros."""
    if name not in observations.columns:
        raise ValueError(f"Feature observations require column {name!r}")
    return observations[name].cast(pl.Float64).to_numpy()


def _percentage(
    numerator: _indicators.FloatArray, denominator: _indicators.FloatArray
) -> _indicators.FloatArray:
    """Return signed percentage changes, leaving invalid reference prices missing."""
    return np.divide(
        100.0 * (numerator - denominator),
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )


def _calendar_feature(
    observations: pl.DataFrame,
    spec: FeatureSpec,
    events: Mapping[str, tuple[date, ...]] | None,
) -> _indicators.FloatArray:
    """Evaluate date, clock, and preloaded economic-calendar conditions."""
    if spec.template == "event_day":
        cached = f"__event_{spec.name}"
        if cached in observations.columns:
            return _column(observations, cached)
        return event_flags(
            observations["datetime"].to_list(),
            spec.parameters["event_type"],
            spec.parameters["days"],
            events=events,
        )
    if spec.template == "is_day":
        weekday = _DAYS.index(str(spec.parameters["day"]).lower()) + 1
        return (observations["datetime"].dt.weekday().to_numpy() == weekday).astype(
            np.float64
        )
    timestamps = observations["datetime"]
    return (
        timestamps.dt.hour().cast(pl.Float64).to_numpy() * 3600
        + timestamps.dt.minute().cast(pl.Float64).to_numpy() * 60
        + timestamps.dt.second().cast(pl.Float64).to_numpy()
    )


def _raw_feature(
    observations: pl.DataFrame,
    history: pl.DataFrame,
    spec: FeatureSpec,
    events: Mapping[str, tuple[date, ...]] | None = None,
) -> _indicators.FloatArray:
    """Dispatch one registered calculator without evaluating arbitrary YAML code."""
    template = spec.template
    if template in INDICATOR_TEMPLATES:
        return indicator_feature(observations, history, spec)
    if template in ("is_day", "time", "event_day"):
        return _calendar_feature(observations, spec, events)
    if template == "leg_delta":
        name = f"{spec.parameters['leg']}_delta"
        return (
            _column(observations, name)
            if name in observations.columns
            else np.full(observations.height, np.nan)
        )
    if template in _OBSERVATION_COLUMNS:
        component = spec.parameters.get("component", "last")
        name = _OBSERVATION_COLUMNS[template]
        if template in ("return_pct", "ror_pct") and component != "last":
            name += f"_{component}"
        return _column(observations, name)
    if template in ("change_since_open", "change_since_prev_close", "gap"):
        numerator = _column(
            observations, "day_open" if template == "gap" else "underlying_price"
        )
        denominator = _column(
            observations,
            "day_open" if template == "change_since_open" else "prev_close",
        )
        return _percentage(numerator, denominator)
    raise ValueError(f"Unknown feature template: {template!r}")


def build_features(
    observations: pl.DataFrame,
    daily_history: pl.DataFrame,
    specs: tuple[FeatureSpec, ...],
    *,
    events: Mapping[str, tuple[date, ...]] | None = None,
) -> torch.Tensor:
    """Build CPU float32 observations without using later intraday or daily bars.

    Inputs describe a single underlying symbol. ``daily_history`` contains dated
    daily OHLC bars. Observations supply the current day's partial OHLC through
    each timestamp; the caller can move the resulting tensor to CUDA once.
    """
    if "datetime" not in observations.columns:
        raise ValueError("Feature observations require datetime")
    if any(
        spec.kind == "bool" for spec in specs[sum(s.kind == "bool" for s in specs) :]
    ):
        raise ValueError(
            "Boolean feature specifications must come before continuous features"
        )
    if not specs or observations.is_empty():
        return torch.empty((observations.height, len(specs)), dtype=torch.float32)
    columns = [
        spec.normalize(_raw_feature(observations, daily_history, spec, events))
        for spec in specs
    ]
    return torch.from_numpy(np.column_stack(columns).astype(np.float32))
