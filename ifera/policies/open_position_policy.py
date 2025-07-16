"""Policies for determining when to open new positions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class OpenPositionPolicy(nn.Module, ABC):
    """Abstract base class for open position policies."""

    @abstractmethod
    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        no_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return actions for opening new positions."""
        raise NotImplementedError


class AlwaysOpenPolicy(OpenPositionPolicy):
    """Policy that always opens a position."""

    def __init__(self, direction: int, batch_size: int, device: torch.device) -> None:
        super().__init__()
        _ = batch_size
        self.direction = torch.tensor(direction, dtype=torch.int32, device=device)

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        no_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = date_idx, time_idx
        return no_position_mask * self.direction


class OpenOncePolicy(OpenPositionPolicy):
    """Policy that opens a position once and holds it."""

    def __init__(self, direction: int, batch_size: int, device: torch.device) -> None:
        super().__init__()
        self.direction = torch.tensor(direction, dtype=torch.int32, device=device)
        self.opened = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        no_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = date_idx, time_idx
        open_mask = no_position_mask & ~self.opened
        self.opened = self.opened | open_mask
        return open_mask * self.direction
