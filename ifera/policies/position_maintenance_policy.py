"""Policies for managing open positions and updating stops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn
from einops import repeat

from ..data_models import InstrumentData, DataManager
from ..config import ConfigManager
from ..file_manager import FileManager
from .stop_loss_policy import ArtrStopLossPolicy


class PositionMaintenancePolicy(nn.Module, ABC):
    """Abstract base class for position maintenance policies."""

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action and stop loss tensors."""
        raise NotImplementedError


class ScaledArtrMaintenancePolicy(PositionMaintenancePolicy):
    """A position maintenance policy that adjusts stop-loss using scaled ATR."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        stages: List[str],
        atr_multiple: float,
        wait_for_breakeven: bool,
        minimum_improvement: float,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.atr_multiple = atr_multiple
        self.wait_for_breakeven = wait_for_breakeven
        self.minimum_improvement = minimum_improvement

        cm = ConfigManager()
        dm = DataManager()
        fm = FileManager()

        if stages[0] != instrument_data.instrument.interval:
            raise ValueError("The first stage must match the instrument's interval.")

        self.derived_configs = [
            cm.get_base_instrument_config(
                symbol=instrument_data.instrument.symbol,
                interval=stage,
                contract_code=instrument_data.instrument.contract_code,
            )
            for stage in stages
        ]

        with fm.persistentContext():
            self.derived_data = [
                dm.get_instrument_data(
                    config,
                    dtype=instrument_data.dtype,
                    device=instrument_data.device,
                    backadjust=instrument_data.backadjust,
                )
                for config in self.derived_configs
            ]

        date_idx_base = torch.arange(
            self.derived_data[0].data.size(0), device=instrument_data.device
        )
        time_idx_base = torch.arange(
            self.derived_data[0].data.size(1), device=instrument_data.device
        )
        date_idx = repeat(date_idx_base, "d -> d t", t=time_idx_base.size(0))
        time_idx = repeat(time_idx_base, "t -> d t", d=date_idx_base.size(0))

        self.converted_indices = [
            self.derived_data[s].convert_indices(
                self.derived_data[0], date_idx, time_idx
            )
            for s in range(len(self.derived_data))
        ]

        self.artr_policies = [
            ArtrStopLossPolicy(data, self.atr_multiple) for data in self.derived_data
        ]
        self.stage_count = len(stages)

        device = instrument_data.device
        dtype = instrument_data.dtype
        self.stage = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.base_price = torch.full(
            (batch_size,), float("nan"), dtype=dtype, device=device
        )
        self._action = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self._stop = torch.empty(batch_size, dtype=dtype, device=device)
        self._zero = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self._nan = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)

    def reset(self, mask: torch.Tensor) -> None:
        self.stage = torch.where(mask, self._zero, self.stage)
        self.base_price = torch.where(mask, self._nan, self.base_price)

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self._action
        stop_loss = prev_stop.clone()

        has_position_mask = position != 0
        current_stage = self.stage
        current_base_price = self.base_price.clone()

        nan_base_mask = torch.isnan(self.base_price) & has_position_mask
        if self.wait_for_breakeven:
            current_base_price = torch.where(
                nan_base_mask, entry_price, current_base_price
            )
        else:
            finite_prev_stop = torch.isfinite(prev_stop)
            set_mask = nan_base_mask & finite_prev_stop
            current_base_price = torch.where(set_mask, prev_stop, current_base_price)

        stage0_mask = current_stage == 0
        if self.wait_for_breakeven:
            potential_stop = self.artr_policies[0](
                date_idx,
                time_idx,
                position * stage0_mask,
                self._zero,
                stop_loss,
            )
            improve_mask_subset = stage0_mask & (
                (position > 0) & (potential_stop > entry_price)
                | (position < 0) & (potential_stop < entry_price)
            )
            stop_loss = torch.where(improve_mask_subset, potential_stop, stop_loss)
            current_stage = torch.where(improve_mask_subset, 1, current_stage)
        else:
            current_stage = torch.where(
                stage0_mask & has_position_mask, 1, current_stage
            )

        for s in range(1, self.stage_count):
            stage_mask = current_stage == s
            conv_date_idx = self.converted_indices[s][0][date_idx, time_idx]
            conv_time_idx = self.converted_indices[s][1][date_idx, time_idx]
            potential_stop = self.artr_policies[s](
                conv_date_idx,
                conv_time_idx,
                position * stage_mask,
                self._zero,
                stop_loss,
            )
            improvement = torch.where(
                position > 0,
                potential_stop - stop_loss,
                stop_loss - potential_stop,
            )
            min_improvement = self.minimum_improvement * torch.abs(
                current_base_price - stop_loss
            )
            improve_mask_subset = (
                stage_mask & (improvement > min_improvement) & (conv_date_idx >= 0)
            )
            stop_loss = torch.where(improve_mask_subset, potential_stop, stop_loss)
            current_stage = torch.where(
                improve_mask_subset & (s < self.stage_count - 1),
                s + 1,
                current_stage,
            )

        self.stage = current_stage
        self.base_price = current_base_price

        return action, stop_loss


class PercentGainMaintenancePolicy(PositionMaintenancePolicy):
    """Maintain a percentage of unrealized gains using ATR for stage one."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        stage1_atr_multiple: float,
        trailing_stop: bool,
        skip_stage1: bool,
        keep_percent: float,
        anchor_type: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        if anchor_type not in [
            "entry",
            "initial_stop",
            "artificial",
            "last_stage1_stop",
            "artificial_stage2",
        ]:
            raise ValueError("Invalid anchor_type")
        if anchor_type in ["last_stage1_stop", "artificial_stage2"] and (
            skip_stage1 or not trailing_stop
        ):
            raise ValueError(
                "anchor_type 'last_stage1_stop' and 'artificial_stage2' require"
                "skip_stage1=False and trailing_stop=True"
            )
        if anchor_type in ["artificial", "artificial_stage2"] and not (
            0 < keep_percent < 1
        ):
            raise ValueError(
                "keep_percent must be between 0 and 1 for 'artificial' and "
                "'artificial_stage2' anchor_types"
            )

        self.instrument_data = instrument_data
        self.trailing_stop = trailing_stop
        self.skip_stage1 = skip_stage1
        self.keep_percent = keep_percent
        self.anchor_type = anchor_type
        self.artr_policy = ArtrStopLossPolicy(instrument_data, stage1_atr_multiple)

        device = instrument_data.device
        dtype = instrument_data.dtype
        initial_stage = 1 if self.skip_stage1 else 0
        self.stage = torch.full(
            (batch_size,), initial_stage, dtype=torch.long, device=device
        )
        self.anchor = torch.full(
            (batch_size,), float("nan"), dtype=dtype, device=device
        )
        self._action = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self._stop = torch.empty(batch_size, dtype=dtype, device=device)
        self._initial_stage = torch.full(
            (batch_size,), initial_stage, dtype=torch.long, device=device
        )
        self._nan = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)

    def reset(self, mask: torch.Tensor) -> None:
        self.stage = torch.where(mask, self._initial_stage, self.stage)
        self.anchor = torch.where(mask, self._nan, self.anchor)

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self._action
        stop_loss = prev_stop.clone()

        has_position_mask = position != 0
        current_stage = self.stage.clone()

        first_call_mask = torch.isnan(self.anchor) & has_position_mask
        if self.anchor_type == "entry":
            self.anchor = torch.where(first_call_mask, entry_price, self.anchor)
        elif self.anchor_type == "initial_stop":
            self.anchor = torch.where(first_call_mask, prev_stop, self.anchor)
        elif self.anchor_type == "artificial":
            reference_channel = torch.where(position > 0, 1, 2)
            reference_price = self.instrument_data.data[
                date_idx, time_idx, reference_channel
            ]
            self.anchor = torch.where(
                first_call_mask,
                (prev_stop - self.keep_percent * reference_price)
                / (1 - self.keep_percent),
                self.anchor,
            )

        stage1_mask = (current_stage == 0) & has_position_mask
        atr_stop = self.artr_policy(
            date_idx, time_idx, position * stage1_mask, action, prev_stop
        )
        move_to_stage2 = stage1_mask & (
            (position > 0) & (atr_stop > entry_price)
            | (position < 0) & (atr_stop < entry_price)
        )

        if self.anchor_type == "last_stage1_stop":
            self.anchor = torch.where(move_to_stage2, atr_stop, self.anchor)
        elif self.anchor_type == "artificial_stage2":
            reference_channel = torch.where(position > 0, 1, 2)
            reference_price = self.instrument_data.data[
                date_idx, time_idx, reference_channel
            ]
            self.anchor = torch.where(
                move_to_stage2,
                (atr_stop - self.keep_percent * reference_price)
                / (1 - self.keep_percent),
                self.anchor,
            )

        self.stage = torch.where(move_to_stage2, 1, current_stage)

        if self.trailing_stop:
            stop_loss = torch.where(stage1_mask, atr_stop, stop_loss)

        stage2_mask = current_stage == 1
        anchor = self.anchor
        reference_channel = torch.where(position > 0, 1, 2)
        reference_price = self.instrument_data.data[
            date_idx, time_idx, reference_channel
        ]
        candidate_stop_loss = torch.where(
            position > 0,
            torch.maximum(
                prev_stop, anchor + self.keep_percent * (reference_price - anchor)
            ),
            torch.minimum(
                prev_stop, anchor - self.keep_percent * (anchor - reference_price)
            ),
        )
        stop_loss = torch.where(stage2_mask, candidate_stop_loss, stop_loss)

        return action, stop_loss
