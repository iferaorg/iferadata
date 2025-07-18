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

    @abstractmethod
    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
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
        self.reset()

    def reset(self) -> None:
        """Reset all sub-policies to their initial state."""
        if hasattr(self.open_position_policy, "reset"):
            self.open_position_policy.reset()
        if hasattr(self.initial_stop_loss_policy, "reset"):
            self.initial_stop_loss_policy.reset()
        if hasattr(self.position_maintenance_policy, "reset"):
            self.position_maintenance_policy.reset()
        if hasattr(self.trading_done_policy, "reset"):
            self.trading_done_policy.reset()

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        no_position_mask = position == 0
        has_position_mask = position != 0

        action = self.open_position_policy(date_idx, time_idx, no_position_mask)

        opening_position_mask = action != 0

        stop_loss = self.initial_stop_loss_policy(
            date_idx, time_idx, position, action, prev_stop
        )
        self.position_maintenance_policy.masked_reset(opening_position_mask)

        maintenance_actions, maintenance_stops = self.position_maintenance_policy(
            date_idx, time_idx, position, prev_stop, entry_price
        )
        action = torch.where(has_position_mask, maintenance_actions, action)
        stop_loss = torch.where(has_position_mask, maintenance_stops, stop_loss)

        done = self.trading_done_policy(
            date_idx, time_idx, position, prev_stop, entry_price
        )
        last_bar_mask = (date_idx == self.instrument_data.data.shape[0] - 1) & (
            time_idx == self.instrument_data.data.shape[1] - 1
        )
        done = done | last_bar_mask

        return action, stop_loss, done
