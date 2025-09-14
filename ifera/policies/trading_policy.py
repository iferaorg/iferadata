"""Trading policy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Optional, Tuple

import torch
from torch import nn
import tensordict as td

from ..data_models import InstrumentData
from ..torch_utils import get_devices, get_module_device
from .policy_base import PolicyBase


class TradingPolicy(PolicyBase):
    """Composite trading policy coordinating multiple sub-policies."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        open_position_policy: PolicyBase,
        initial_stop_loss_policy: PolicyBase,
        position_maintenance_policy: PolicyBase,
        trading_done_policy: PolicyBase,
    ) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.open_position_policy = open_position_policy
        self.initial_stop_loss_policy = initial_stop_loss_policy
        self.position_maintenance_policy = position_maintenance_policy
        self.trading_done_policy = trading_done_policy
        self._last_date_idx = instrument_data.data.shape[0] - 1
        self._last_time_idx = instrument_data.data.shape[1] - 1

    def copy_to(self, device: torch.device) -> TradingPolicy:
        """Return a copy of the policy on the specified device."""
        return TradingPolicy(
            self.instrument_data.copy_to(device),
            self.open_position_policy.copy_to(device),
            self.initial_stop_loss_policy.copy_to(device),
            self.position_maintenance_policy.copy_to(device),
            self.trading_done_policy.copy_to(device),
        )

    def clone(self, device: torch.device | str) -> "TradingPolicy":
        """Return a deep copy of the policy moved to ``device``.

        All sub-policies are duplicated and the new instance is transferred to the
        target device, leaving the original unmodified.
        """
        cloned_policy = copy.deepcopy(self)
        cloned_policy.to(device)
        return cloned_policy

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """Reset all sub-policies to their initial state."""
        td_out = td.TensorDict({}, batch_size=state.batch_size, device=state.device)
        td_out.update(self.open_position_policy.reset(state))
        td_out.update(self.initial_stop_loss_policy.reset(state))   
        td_out.update(self.position_maintenance_policy.reset(state))
        td_out.update(self.trading_done_policy.reset(state))
        
        return td_out

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """Reset all sub-policies where mask is True."""
        td_out = td.TensorDict({}, batch_size=state.batch_size, device=state.device)
        td_out.update(self.open_position_policy.masked_reset(state, mask))
        td_out.update(self.initial_stop_loss_policy.masked_reset(state, mask))
        td_out.update(self.position_maintenance_policy.masked_reset(state, mask))
        td_out.update(self.trading_done_policy.masked_reset(state, mask))
        
        return td_out

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        position = state["position"]

        state_out = td.TensorDict({}, batch_size=state.batch_size, device=state.device)
        state_in = state.copy()

        result = self.trading_done_policy(state_in)
        last_bar_mask = (date_idx == self._last_date_idx) & (
            time_idx == self._last_time_idx
        )
        done = state_in["done"] | result["done"] | last_bar_mask
        result["done"] = done
        state_out.update(result)
        state_in.update(result)

        state_in["no_position_mask"] = (position == 0) & ~done
        state_in["has_position_mask"] = (position != 0) & ~done

        result = self.open_position_policy(state_in)
        state_out.update(result)
        state_in.update(result)

        opening_position_mask = result["action"] != 0

        result = self.initial_stop_loss_policy(state_in)
        state_out.update(result)
        state_in.update(result)

        result = self.position_maintenance_policy.masked_reset(state_in, opening_position_mask)
        state_out.update(result)
        state_in.update(result)

        result = (self.position_maintenance_policy(state_in))
        state_out.update(result)

        return state_out


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
