"""Policies for determining when to open new positions."""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from torch import nn
import tensordict as td

from .policy_base import PolicyBase


class AlwaysOpenPolicy(PolicyBase):
    """Policy that always opens a position."""

    def __init__(self, direction: int, device: torch.device) -> None:
        super().__init__()
        self.direction: torch.Tensor
        self.register_buffer(
            "direction", torch.tensor(direction, dtype=torch.int32, device=device)
        )

    def copy_to(self, device: torch.device) -> AlwaysOpenPolicy:
        """Return a copy of the policy on the specified device."""
        return AlwaysOpenPolicy(int(self.direction.item()), device)

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """AlwaysOpenPolicy holds no state so nothing to reset."""
        return td.TensorDict({}, batch_size=state.batch_size, device=state.device)

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """AlwaysOpenPolicy holds no state so nothing to reset."""
        return td.TensorDict({}, batch_size=state.batch_size, device=state.device)

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        """Return the direction for all batches that can open a position."""
        no_position_mask = state["no_position_mask"]

        td_out = td.TensorDict(
            {
                "action": no_position_mask * self.direction,
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out


class OpenOncePolicy(PolicyBase):
    """Policy that opens a position once and holds it."""

    def __init__(self, direction: int, device: torch.device) -> None:
        super().__init__()
        self.direction: torch.Tensor
        self.register_buffer(
            "direction", torch.tensor(direction, dtype=torch.int32, device=device)
        )

    def copy_to(self, device: torch.device) -> OpenOncePolicy:
        """Return a copy of the policy on the specified device."""
        return OpenOncePolicy(int(self.direction.item()), device)

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """Reset ``opened`` state to ``False`` for all batches."""
        td_out = td.TensorDict(
            {
                "opened": torch.zeros(
                    state.batch_size, dtype=torch.bool, device=state.device
                )
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """Reset ``opened`` state to ``False`` where mask is True."""
        td_out = td.TensorDict(
            {
                "opened": torch.where(mask, False, state["opened"]),
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def forward(self, state: td.TensorDict) -> td.TensorDict:
        """Return the direction for all batches that can open a position."""
        opened = state["opened"]
        no_position_mask = state["no_position_mask"]
        open_mask = no_position_mask & ~opened

        td_out = td.TensorDict(
            {"action": open_mask * self.direction, "opened": opened | open_mask},
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out
