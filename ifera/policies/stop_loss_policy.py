"""Policies for calculating stop-loss levels."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
import tensordict as td

from ..data_models import InstrumentData
from .policy_base import PolicyBase


class ArtrStopLossPolicy(PolicyBase):
    """ATR-based stop loss policy."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        atr_multiple: float,
        alpha: float = 1.0 / 14.0,
        acrossday: bool = True,
    ) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.atr_multiple = atr_multiple
        self.alpha = alpha
        self.acrossday = acrossday

        self._data: torch.Tensor
        self.register_buffer("_data", instrument_data.data)
        self._artr: torch.Tensor
        self.register_buffer(
            "_artr", torch.tensor((), device=self._data.device, dtype=self._data.dtype)
        )

        # Pre-calculate potential stops for all combinations
        self._cached_potential_stops: torch.Tensor
        self.register_buffer(
            "_cached_potential_stops",
            torch.tensor((), device=self._data.device, dtype=self._data.dtype),
        )

    def copy_to(self, device: torch.device) -> ArtrStopLossPolicy:
        """Return a copy of the policy on the specified device."""
        return ArtrStopLossPolicy(
            self.instrument_data.copy_to(device),
            self.atr_multiple,
            self.alpha,
            self.acrossday,
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

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """ArtrStopLossPolicy does not maintain state."""
        instrument_data = self.instrument_data

        if (
            len(instrument_data.artr) == 0
            or instrument_data.artr_alpha != self.alpha
            or instrument_data.artr_acrossday != self.acrossday
        ):
            instrument_data.calculate_artr(alpha=self.alpha, acrossday=self.acrossday)

        self._artr = instrument_data.artr

        if (
            self._cached_potential_stops.numel() == 0
            or self._cached_potential_stops.shape[0] != self._artr.shape[0]
            or self._cached_potential_stops.shape[1] != self._artr.shape[1]
        ):
            self._cached_potential_stops = self._precompute_potential_stops()

        return td.TensorDict({}, batch_size=state.batch_size, device=state.device)

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """ArtrStopLossPolicy does not maintain state."""
        return td.TensorDict({}, batch_size=state.batch_size, device=state.device)

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        """Calculate stop loss levels based on ATR."""
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        position = state["position"]
        prev_stop = state["prev_stop_loss"]
        action = state["action"]

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

        td_out = td.TensorDict(
            {"stop_loss": stop_price},
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out


class InitialArtrStopLossPolicy(PolicyBase):
    """Stop loss policy for setting initial stops using ATR."""

    def __init__(self, instrument_data: InstrumentData, atr_multiple: float) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.atr_multiple = atr_multiple
        self.artr_policy = ArtrStopLossPolicy(instrument_data, atr_multiple)

    def copy_to(self, device: torch.device) -> InitialArtrStopLossPolicy:
        """Return a copy of the policy on the specified device."""
        return InitialArtrStopLossPolicy(
            self.instrument_data.copy_to(device),
            self.atr_multiple,
        )

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """InitialArtrStopLossPolicy holds no state to reset."""
        return self.artr_policy.reset(state)

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """InitialArtrStopLossPolicy holds no state to reset."""
        return self.artr_policy.masked_reset(state, mask)

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        """Calculate initial stop loss levels based on ATR."""
        no_position_mask = state["no_position_mask"]
        td_out = self.artr_policy(state)
        td_out["stop_loss"] = torch.where(
            no_position_mask, torch.nan, td_out["stop_loss"]
        )

        return td_out
