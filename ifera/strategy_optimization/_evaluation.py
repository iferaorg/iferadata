"""Evaluate strategy predicates in bounded, vectorized PyTorch work batches."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum

import torch
from torch import Tensor

from ._features import FeatureSpec
from ._thresholds import ThresholdRounder


class OptimizationMode(str, Enum):
    """How an evolved DNF changes the original strategy."""

    ENTRY_FILTER = "entry_filter"
    EARLY_EXIT = "early_exit"
    COMBINED = "combined"


@dataclass(frozen=True)
class StrategyTensors:
    """Precomputed observations; no database work occurs during evolution.

    Exit observations are ordered chronologically within each trade. An ignored
    feature denotes an absent option leg, rather than an unavailable indicator.
    Returns are percentages and ``trade_days`` maps trades into the full calendar.
    """

    entry_features: Tensor
    exit_features: Tensor
    entry_ignored: Tensor
    exit_ignored: Tensor
    exit_trades: Tensor
    exit_ror: Tensor
    original_ror: Tensor
    trade_days: Tensor
    market_days: int

    def to(self, device: torch.device) -> StrategyTensors:
        """Move observations to the optimizer's selected device once."""
        return replace(
            self,
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
                if isinstance(getattr(self, field.name), Tensor)
            },
        )


@dataclass(frozen=True)
class StrategyEvaluation:
    """Batched entry decisions, first exit indices, and daily percentage RoR."""

    entered: Tensor
    exit_indices: Tensor
    daily_ror: Tensor


class StrategyEvaluator:
    """Apply DNFs across candidates, trades, minutes, clauses, and features.

    Only observation chunks use a Python loop. CUDA fuses predicate operations;
    CPU uses eager evaluation. ``observation_batch_size`` bounds workspace, with
    defaults of 32,768 on CUDA and 512 on CPU.
    Missing indicator values fail restricted conditions. Unrestricted conditions
    and features for absent legs pass, as do features excluded from this phase.
    """

    def __init__(
        self,
        tensors: StrategyTensors,
        features: tuple[FeatureSpec, ...],
        mode: OptimizationMode | str,
        observation_batch_size: int | None = None,
    ) -> None:
        device = tensors.entry_features.device
        if observation_batch_size is None:
            observation_batch_size = 32768 if device.type == "cuda" else 512
        if (
            isinstance(observation_batch_size, bool)
            or not isinstance(observation_batch_size, int)
            or observation_batch_size < 1
        ):
            raise ValueError("observation_batch_size must be a positive integer")
        self.tensors = tensors
        self.features = features
        self.mode = OptimizationMode(mode)
        self.observation_batch_size = observation_batch_size
        self.round_bounds = ThresholdRounder(features, device)
        self.entry_categories = torch.tensor(
            [feature.category in ("any", "entry") for feature in features],
            dtype=torch.bool,
            device=device,
        )
        self.exit_categories = torch.tensor(
            [feature.category in ("any", "exit") for feature in features],
            dtype=torch.bool,
            device=device,
        )
        self._matcher = (
            torch.compile(self._matches, dynamic=True)
            if device.type == "cuda"
            else self._matches
        )

    @staticmethod
    def _matches(
        bounds: Tensor,
        active: Tensor,
        values: Tensor,
        ignored: Tensor,
        categories: Tensor,
    ) -> Tensor:
        """Reduce feature intervals with AND, then active clauses with OR."""
        lower = bounds[:, None, :, :, 0]
        upper = bounds[:, None, :, :, 1]
        unrestricted = (lower == 0) & (upper == 1)
        inside = (values[None, :, None] >= lower) & (values[None, :, None] <= upper)
        inside |= unrestricted | ignored[None, :, None] | ~categories
        return (inside.all(dim=-1) & active[:, None]).any(dim=-1)

    def _entry_matches(self, bounds: Tensor, active: Tensor) -> Tensor:
        """Evaluate entry observations without retaining clause workspaces."""
        values = self.tensors.entry_features
        result = torch.empty(
            (bounds.shape[0], values.shape[0]), dtype=torch.bool, device=bounds.device
        )
        for start in range(0, values.shape[0], self.observation_batch_size):
            stop = start + self.observation_batch_size
            result[:, start:stop] = self._matcher(
                bounds,
                active,
                values[start:stop],
                self.tensors.entry_ignored[start:stop],
                self.entry_categories,
            )
        return result

    def _first_exits(self, bounds: Tensor, active: Tensor) -> Tensor:
        """Reduce matching observation indices to each trade's earliest exit."""
        tensors = self.tensors
        count = tensors.exit_features.shape[0]
        first = torch.full(
            (bounds.shape[0], tensors.original_ror.numel()),
            count,
            dtype=torch.long,
            device=bounds.device,
        )
        if bounds.shape[1] == 0:
            return first
        for start in range(0, count, self.observation_batch_size):
            stop = min(count, start + self.observation_batch_size)
            matches = self._matcher(
                bounds,
                active,
                tensors.exit_features[start:stop],
                tensors.exit_ignored[start:stop],
                self.exit_categories,
            )
            indices = torch.arange(start, stop, device=bounds.device)
            matched = torch.where(matches, indices[None], count)
            trades = tensors.exit_trades[start:stop].expand(bounds.shape[0], -1)
            first.scatter_reduce_(1, trades, matched, reduce="amin", include_self=True)
        return first

    @torch.no_grad()
    def evaluate(self, bounds: Tensor, active: Tensor) -> StrategyEvaluation:
        """Apply one candidate batch and fill no-trade market days with zero."""
        bounds = self.round_bounds(bounds)
        tensors = self.tensors
        batch = bounds.shape[0]
        trade_count = tensors.original_ror.numel()
        observation_count = tensors.exit_features.shape[0]
        entered = torch.ones(
            (batch, trade_count), dtype=torch.bool, device=bounds.device
        )
        first = torch.full(
            (batch, trade_count),
            observation_count,
            dtype=torch.long,
            device=bounds.device,
        )
        if self.mode == OptimizationMode.COMBINED:
            split = (bounds.shape[1] + 1) // 2
            entered = self._entry_matches(bounds[:, :split], active[:, :split])
            first = self._first_exits(bounds[:, split:], active[:, split:])
        elif self.mode == OptimizationMode.ENTRY_FILTER:
            entered = self._entry_matches(bounds, active)
        else:
            first = self._first_exits(bounds, active)
        returns = tensors.original_ror.expand(batch, -1)
        if observation_count:
            padded_ror = torch.cat((tensors.exit_ror, tensors.exit_ror.new_zeros(1)))
            returns = torch.where(first < observation_count, padded_ror[first], returns)
        returns = torch.where(entered, returns, 0.0)
        daily = returns.new_zeros((batch, tensors.market_days))
        daily.scatter_(1, tensors.trade_days.expand(batch, -1), returns)
        first = torch.where(entered & (first < observation_count), first, -1)
        return StrategyEvaluation(entered, first, daily)
