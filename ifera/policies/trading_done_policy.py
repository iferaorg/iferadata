"""Policies that signal the end of a trading episode."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class TradingDonePolicy(nn.Module, ABC):
    """Abstract base class for trading done policies."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        """Reset the policy's state for the specified batch elements."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Fully reset the policy state."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return tensor indicating finished episodes."""
        raise NotImplementedError


class AlwaysFalseDonePolicy(TradingDonePolicy):
    """Trading done policy that never signals completion."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self._false: torch.Tensor | None = None

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Initialize _false buffer based on state batch size."""
        batch_size = next(iter(state.values())).shape[0]
        if self._false is None or self._false.shape[0] != batch_size:
            self._false = torch.zeros(batch_size, dtype=torch.bool, device=self._device)

    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        """No internal state to reset."""
        _ = state
        _ = mask

    def forward(
        self,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _ = state
        # Ensure buffer is created if needed
        if self._false is None:
            self.reset(state)
        return self._false


class SingleTradeDonePolicy(TradingDonePolicy):
    """Signals done once a non-zero position returns to zero."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self.had_position: torch.Tensor | None = None

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Reset ``had_position`` for all batches."""
        batch_size = next(iter(state.values())).shape[0]
        if self.had_position is None or self.had_position.shape[0] != batch_size:
            self.had_position = torch.zeros(batch_size, dtype=torch.bool, device=self._device)
        state["had_position"] = torch.zeros_like(self.had_position)

    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        state["had_position"] = torch.where(mask, False, state["had_position"])

    def forward(
        self,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        position = state["position"]
        had_position = state["had_position"]
        
        # Ensure buffer is created if needed
        if self.had_position is None:
            self.reset(state)
            had_position = state["had_position"]
            
        done = (position == 0) & had_position
        state["had_position"] = torch.where(position != 0, True, had_position)

        return done
