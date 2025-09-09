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
    def reset(
        self, state: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> None:
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
        self.atr_multiple = atr_multiple
        if len(instrument_data.artr) == 0:
            instrument_data.calculate_artr(alpha=alpha, acrossday=acrossday)
        self._data: torch.Tensor
        self.register_buffer("_data", instrument_data.data)
        self._artr: torch.Tensor
        self.register_buffer("_artr", instrument_data.artr)

        # Pre-calculate potential stops for all combinations
        self._cached_potential_stops: torch.Tensor
        self.register_buffer(
            "_cached_potential_stops", self._precompute_potential_stops()
        )

    def _precompute_potential_stops(self) -> torch.Tensor:
        """Pre-compute potential stop values for all combinations of position and direction."""
        dates, times = self._artr.shape

        # Calculate artr for all positions
        artr = self._artr * self.atr_multiple + 1.0  # Shape: [dates, times]

        # Create tensor with shape [dates, times, 2, 2]
        # Dimension 2: position == 0 (index 0 = False, index 1 = True)
        # Dimension 3: direction > 0 (index 0 = False, index 1 = True)
        potential_stops = torch.zeros(
            dates, times, 2, 2, dtype=self._data.dtype, device=self._data.device
        )

        # For each combination of (position==0, direction>0):
        for pos_is_zero in [0, 1]:  # 0 = position != 0, 1 = position == 0
            for dir_positive in [0, 1]:  # 0 = direction <= 0, 1 = direction > 0

                # Determine reference channel
                if pos_is_zero == 1:  # position == 0
                    reference_channel = 0  # Open price
                else:  # position != 0
                    if dir_positive == 1:  # direction > 0
                        reference_channel = 1  # High price
                    else:  # direction <= 0
                        reference_channel = 2  # Low price

                # Get reference prices for all dates/times
                reference_prices = self._data[
                    :, :, reference_channel
                ]  # Shape: [dates, times]

                # Calculate potential stop
                if dir_positive == 1:  # direction > 0
                    potential_stop = reference_prices / artr
                else:  # direction <= 0
                    potential_stop = reference_prices * artr

                potential_stops[:, :, pos_is_zero, dir_positive] = potential_stop

        return potential_stops

    def reset(
        self, state: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> None:
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

        # Use cached potential stops instead of recalculating
        pos_is_zero = (position == 0).long()  # Convert boolean to 0/1
        dir_positive = (direction > 0).long()  # Convert boolean to 0/1

        # Index into cached tensor
        potential_stop = self._cached_potential_stops[
            date_idx, time_idx, pos_is_zero, dir_positive
        ]

        stop_price = torch.where(
            direction > 0,
            torch.maximum(prev_stop, potential_stop),
            torch.minimum(prev_stop, potential_stop),
        )
        stop_price = torch.where(
            torch.isnan(stop_price) | (direction == 0), prev_stop, stop_price
        )

        return stop_price


class InitialArtrStopLossPolicy(StopLossPolicy):
    """Stop loss policy for setting initial stops using ATR."""

    def __init__(self, instrument_data: InstrumentData, atr_multiple: float) -> None:
        super().__init__()
        self.artr_policy = ArtrStopLossPolicy(instrument_data, atr_multiple)

    def reset(
        self, state: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> None:
        """InitialArtrStopLossPolicy holds no state to reset."""
        _ = state

    def forward(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.artr_policy(state, action)
