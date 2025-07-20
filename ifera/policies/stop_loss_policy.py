"""Policies for calculating stop-loss levels."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..data_models import InstrumentData


class StopLossPolicy(nn.Module, ABC):
    """Abstract base class for stop loss policies."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Reset policy state."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return stop loss levels."""
        raise NotImplementedError


class ArtrStopLossPolicy(StopLossPolicy):
    """ATR-based stop loss policy."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        atr_multiple: float,
        alpha: float = 1.0 / 14.0,
        acrossday: bool = True,
    ) -> None:
        super().__init__()
        self.idata = instrument_data
        self.atr_multiple = atr_multiple
        if len(instrument_data.artr) == 0:
            instrument_data.calculate_artr(alpha=alpha, acrossday=acrossday)
        self._data = instrument_data.data
        self._artr = instrument_data.artr

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """ArtrStopLossPolicy does not maintain state."""
        return None

    def forward(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        position = state["position"]
        prev_stop = state["prev_stop_loss"]

        direction = (position + action).sign()

        prev_stop = torch.where(
            torch.isnan(prev_stop) & (direction != 0),
            torch.inf * direction * -1,
            prev_stop,
        )

        artr = self._artr[date_idx, time_idx] * self.atr_multiple + 1.0

        reference_channel = torch.where(
            position == 0, 3, torch.where(direction > 0, 1, 2)
        )
        reference_price = self._data[date_idx, time_idx, reference_channel]

        stop_price = torch.where(
            direction > 0,
            torch.maximum(prev_stop, reference_price / artr),
            torch.minimum(prev_stop, reference_price * artr),
        )
        stop_price = torch.where(
            torch.isnan(stop_price) | (direction == 0), prev_stop, stop_price
        )

        return stop_price


class InitialArtrStopLossPolicy(StopLossPolicy):
    """Stop loss policy for setting initial stops using ATR."""

    def __init__(
        self, instrument_data: InstrumentData, atr_multiple: float, batch_size: int
    ) -> None:
        super().__init__()
        self.artr_policy = ArtrStopLossPolicy(instrument_data, atr_multiple)
        dtype = instrument_data.data.dtype
        device = instrument_data.device
        self._zero = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self._nan = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """InitialArtrStopLossPolicy holds no state to reset."""
        _ = state
        return None

    def forward(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.artr_policy(state, action)
