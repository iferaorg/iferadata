"""Policies that signal the end of a trading episode."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..state import State


class TradingDonePolicy(nn.Module, ABC):
    """Abstract base class for trading done policies."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def masked_reset(self, state: State, mask: torch.Tensor) -> None:
        """Reset the policy's state for the specified batch elements."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, state: State, batch_size: int, device: torch.device) -> None:
        """Fully reset the policy state."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state: State,
    ) -> torch.Tensor:
        """Return tensor indicating finished episodes."""
        raise NotImplementedError


class AlwaysFalseDonePolicy(TradingDonePolicy):
    """Trading done policy that never signals completion."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        # Register _false as a buffer so it gets moved with .to(device)
        self._false: torch.Tensor
        self.register_buffer(
            "_false", torch.tensor((), dtype=torch.bool, device=device)
        )

    def reset(self, state: State, batch_size: int, device: torch.device) -> None:
        """Initialize _false buffer based on state batch size."""
        # Update device if it has changed
        if hasattr(self, "_device"):
            self._device = device
        self._false = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def masked_reset(self, state: State, mask: torch.Tensor) -> None:
        """No internal state to reset."""
        _ = state
        _ = mask

    def forward(
        self,
        state: State,
    ) -> torch.Tensor:
        _ = state
        return self._false


class SingleTradeDonePolicy(TradingDonePolicy):
    """Signals done once a non-zero position returns to zero."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device

    def reset(self, state: State, batch_size: int, device: torch.device) -> None:
        """Reset ``had_position`` for all batches."""
        # Update device if it has changed
        if hasattr(self, "_device"):
            self._device = device
        state.had_position = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def masked_reset(self, state: State, mask: torch.Tensor) -> None:
        state.had_position = torch.where(mask, False, state.had_position)

    def forward(
        self,
        state: State,
    ) -> torch.Tensor:
        position = state.position
        had_position = state.had_position

        done = (position == 0) & had_position
        state.had_position = torch.where(position != 0, True, had_position)

        return done
