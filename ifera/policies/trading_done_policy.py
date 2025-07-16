"""Policies that signal the end of a trading episode."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class TradingDonePolicy(nn.Module, ABC):
    """Abstract base class for trading done policies."""

    @abstractmethod
    def reset(self, mask: torch.Tensor) -> None:
        """Reset the policy's state for the specified batch elements."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ) -> torch.Tensor:
        """Return tensor indicating finished episodes."""
        raise NotImplementedError


class AlwaysFalseDonePolicy(TradingDonePolicy):
    """Trading done policy that never signals completion."""

    def __init__(self, batch_size: int, device: torch.device) -> None:
        super().__init__()
        self._false = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def reset(self, mask: torch.Tensor) -> None:
        return None

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ) -> torch.Tensor:
        _ = date_idx, time_idx, position, prev_stop, entry_price
        return self._false


class SingleTradeDonePolicy(TradingDonePolicy):
    """Signals done once a non-zero position returns to zero."""

    def __init__(self, batch_size: int, device: torch.device) -> None:
        super().__init__()
        self.had_position = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._indices = torch.arange(batch_size, device=device)

    def reset(self, mask: torch.Tensor) -> None:
        self.had_position = torch.where(mask, False, self.had_position)

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ) -> torch.Tensor:
        _ = date_idx, time_idx, prev_stop, entry_price
        done = (position == 0) & self.had_position
        self.had_position = torch.where(position != 0, True, self.had_position)
        return done
