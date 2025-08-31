"""Trading policy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Optional, Tuple

import torch
from torch import nn

from ..data_models import InstrumentData
from ..torch_utils import get_devices, get_module_device
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
        self.open_position_policy = open_position_policy
        self.initial_stop_loss_policy = initial_stop_loss_policy
        self.position_maintenance_policy = position_maintenance_policy
        self.trading_done_policy = trading_done_policy
        _ = batch_size
        self._last_date_idx = instrument_data.data.shape[0] - 1
        self._last_time_idx = instrument_data.data.shape[1] - 1

    def clone(self, device: torch.device | str) -> "TradingPolicy":
        """Return a deep copy of the policy moved to ``device``.

        All sub-policies are duplicated and the new instance is transferred to the
        target device, leaving the original unmodified.
        """
        cloned_policy = copy.deepcopy(self)
        cloned_policy.to(device)
        return cloned_policy

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

        done = state["done"] | self.trading_done_policy(state)
        last_bar_mask = (date_idx == self._last_date_idx) & (
            time_idx == self._last_time_idx
        )
        done = done | last_bar_mask

        no_position_mask = (position == 0) & ~done
        has_position_mask = (position != 0) & ~done

        action = self.open_position_policy(state, no_position_mask)

        opening_position_mask = action != 0

        stop_loss = self.initial_stop_loss_policy(state, action)
        self.position_maintenance_policy.masked_reset(state, opening_position_mask)

        maintenance_actions, maintenance_stops = self.position_maintenance_policy(state)
        action = torch.where(has_position_mask, maintenance_actions, action)
        stop_loss = torch.where(has_position_mask, maintenance_stops, stop_loss)

        return action, stop_loss, done


def clone_trading_policy_for_devices(
    policy: TradingPolicy, devices: Optional[list[torch.device]] = None
) -> list[TradingPolicy]:
    """Clone ``policy`` for each device.

    The original ``policy`` is reused if its device appears in ``devices``.  All
    other devices receive a deep-copied clone moved to the respective device.
    When ``devices`` is ``None`` the same default logic as
    :class:`MultiGPUSingleMarketEnv` is applied, using all available CUDA
    devices or the CPU if CUDA is unavailable.
    """

    resolved_devices = get_devices(devices)
    policy_device = get_module_device(policy)
    cloned_policies: list[TradingPolicy] = []
    used_original = False

    for device in resolved_devices:
        if device == policy_device and not used_original:
            cloned_policies.append(policy)
            used_original = True
        else:
            cloned_policies.append(policy.clone(device))

    return cloned_policies
