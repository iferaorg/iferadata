"""Trading policy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn

from ..data_models import InstrumentData
from .open_position_policy import OpenPositionPolicy
from .stop_loss_policy import StopLossPolicy
from .position_maintenance_policy import PositionMaintenancePolicy
from .trading_done_policy import TradingDonePolicy


class BaseTradingPolicy(nn.Module, ABC):
    """Abstract base class for trading policies."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Reset the policy to its initial state."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, stop loss and done tensors."""
        raise NotImplementedError


class TradingPolicy(BaseTradingPolicy):
    """Composite trading policy coordinating multiple sub-policies."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        open_position_policy: OpenPositionPolicy,
        initial_stop_loss_policy: StopLossPolicy,
        position_maintenance_policy: PositionMaintenancePolicy,
        trading_done_policy: TradingDonePolicy,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.open_position_policy = open_position_policy
        self.initial_stop_loss_policy = initial_stop_loss_policy
        self.position_maintenance_policy = position_maintenance_policy
        self.trading_done_policy = trading_done_policy
        self._batch_size = batch_size
        self._last_date_idx = instrument_data.data.shape[0] - 1
        self._last_time_idx = instrument_data.data.shape[1] - 1

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Reset all sub-policies to their initial state."""
        self.open_position_policy.reset(state)
        self.initial_stop_loss_policy.reset(state)
        self.position_maintenance_policy.reset(state)
        self.trading_done_policy.reset(state)

    def forward(
        self,
        state: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        position = state["position"]

        no_position_mask = position == 0
        has_position_mask = position != 0

        action = self.open_position_policy(state, no_position_mask)

        opening_position_mask = action != 0

        stop_loss = self.initial_stop_loss_policy(state, action)
        self.position_maintenance_policy.masked_reset(state, opening_position_mask)

        maintenance_actions, maintenance_stops = self.position_maintenance_policy(state)
        action = torch.where(has_position_mask, maintenance_actions, action)
        stop_loss = torch.where(has_position_mask, maintenance_stops, stop_loss)

        done = self.trading_done_policy(state)
        last_bar_mask = (date_idx == self._last_date_idx) & (
            time_idx == self._last_time_idx
        )
        done = done | last_bar_mask

        return action, stop_loss, done
