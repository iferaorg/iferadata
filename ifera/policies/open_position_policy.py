"""Policies for determining when to open new positions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..state import State


class OpenPositionPolicy(nn.Module, ABC):
    """Abstract base class for open position policies."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self, state: State, batch_size: int, device: torch.device) -> None:
        """Reset any internal state held by the policy."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state: State,
        no_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return actions for opening new positions."""
        raise NotImplementedError


class AlwaysOpenPolicy(OpenPositionPolicy):
    """Policy that always opens a position."""

    def __init__(self, direction: int, device: torch.device) -> None:
        super().__init__()
        self.direction: torch.Tensor
        self.register_buffer(
            "direction", torch.tensor(direction, dtype=torch.int32, device=device)
        )

    def reset(self, state: State, batch_size: int, device: torch.device) -> None:
        """AlwaysOpenPolicy holds no state so nothing to reset."""
        return None

    def forward(
        self,
        state: State,
        no_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the direction for all batches that can open a position."""
        _ = state
        return no_position_mask * self.direction


class OpenOncePolicy(OpenPositionPolicy):
    """Policy that opens a position once and holds it."""

    def __init__(self, direction: int, device: torch.device) -> None:
        super().__init__()
        self.direction: torch.Tensor
        self.register_buffer(
            "direction", torch.tensor(direction, dtype=torch.int32, device=device)
        )

    def reset(self, state: State, batch_size: int, device: torch.device) -> None:
        """Reset ``opened`` state to ``False`` for all batches."""
        state.opened = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def forward(
        self,
        state: State,
        no_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the direction for all batches that can open a position."""
        opened = state.opened
        open_mask = no_position_mask & ~opened
        state.opened = opened | open_mask
        return open_mask * self.direction
