"""Policies that signal the end of a trading episode."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
import tensordict as td

from .policy_base import PolicyBase


class AlwaysFalseDonePolicy(PolicyBase):
    """Trading done policy that never signals completion."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        # Register _false as a buffer so it gets moved with .to(device)
        self._false: torch.Tensor
        self.register_buffer(
            "_false", torch.tensor((), dtype=torch.bool, device=device)
        )

    def copy_to(self, device: torch.device) -> AlwaysFalseDonePolicy:
        """Return a copy of the policy on the specified device."""
        return AlwaysFalseDonePolicy(device)

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """Initialize _false buffer based on state batch size."""
        # Update device if it has changed
        self._device = state.device
        self._false = torch.zeros(
            state.batch_size, dtype=torch.bool, device=state.device
        )

        return td.TensorDict({}, batch_size=state.batch_size, device=state.device)

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """No internal state to reset."""
        _ = mask
        return td.TensorDict({}, batch_size=state.batch_size, device=state.device)

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        """Always return False."""
        td_out = td.TensorDict(
            {"done": state["done"]},  # Preserve existing done flags
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out


class SingleTradeDonePolicy(PolicyBase):
    """Signals done once a non-zero position returns to zero."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device

    def copy_to(self, device: torch.device) -> SingleTradeDonePolicy:
        """Return a copy of the policy on the specified device."""
        return SingleTradeDonePolicy(device)

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """Reset ``had_position`` for all batches."""
        # Update device if it has changed
        self._device = state.device

        td_out = td.TensorDict(
            {
                "had_position": torch.zeros(
                    state.batch_size, dtype=torch.bool, device=state.device
                )
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """Reset ``had_position`` where mask is True."""
        td_out = td.TensorDict(
            {"had_position": torch.where(mask, False, state["had_position"])},
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        position = state["position"]
        had_position = state["had_position"]

        done = (position == 0) & had_position

        td_out = td.TensorDict(
            {
                "done": state["done"] | done,
                "had_position": torch.where(position != 0, True, had_position),
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out
