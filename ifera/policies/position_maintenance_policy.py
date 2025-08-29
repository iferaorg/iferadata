"""Policies for managing open positions and updating stops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch._dynamo  # pylint: disable=protected-access
from torch import nn
from einops import repeat
from ..data_models import InstrumentData, DataManager
from ..config import ConfigManager
from ..file_manager import FileManager
from .stop_loss_policy import ArtrStopLossPolicy

torch._dynamo.config.capture_scalar_outputs = True  # pylint: disable=protected-access


class PositionMaintenancePolicy(nn.Module, ABC):
    """Abstract base class for position maintenance policies."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        """Reset the policy's state for the specified batch elements."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Reset the entire policy state."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state: dict[str, torch.Tensor],
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
        self.atr_multiple = atr_multiple
        self.wait_for_breakeven = wait_for_breakeven
        self.minimum_improvement = minimum_improvement

        cm = ConfigManager()
        dm = DataManager()
        fm = FileManager()

        if stages[0] != instrument_data.instrument.interval:
            raise ValueError("The first stage must match the instrument's interval.")

        derived_configs = [
            cm.get_base_instrument_config(
                symbol=instrument_data.instrument.symbol,
                interval=stage,
                contract_code=instrument_data.instrument.contract_code,
            )
            for stage in stages
        ]

        with fm.persistentContext():
            derived_data = [
                dm.get_instrument_data(
                    config,
                    dtype=instrument_data.dtype,
                    device=instrument_data.device,
                    backadjust=instrument_data.backadjust,
                )
                for config in derived_configs
            ]

        date_idx_base = torch.arange(
            derived_data[0].data.size(0), device=instrument_data.device
        )
        time_idx_base = torch.arange(
            derived_data[0].data.size(1), device=instrument_data.device
        )
        date_idx = repeat(date_idx_base, "d -> d t", t=time_idx_base.size(0))
        time_idx = repeat(time_idx_base, "t -> d t", d=date_idx_base.size(0))

        converted = [
            derived_data[s].convert_indices(derived_data[0], date_idx, time_idx)
            for s in range(len(derived_data))
        ]

        conv_date_idx, conv_time_idx = zip(*converted)
        self.conv_date_idx: torch.Tensor
        self.register_buffer("conv_date_idx", torch.stack(conv_date_idx))
        self.conv_time_idx: torch.Tensor
        self.register_buffer("conv_time_idx", torch.stack(conv_time_idx))

        self.artr_policies = nn.ModuleList(
            ArtrStopLossPolicy(data, self.atr_multiple) for data in derived_data
        )
        self.stage_count = len(stages)

        device = instrument_data.device
        dtype = instrument_data.dtype
        self._action: torch.Tensor
        self.register_buffer(
            "_action", torch.zeros(batch_size, dtype=torch.int32, device=device)
        )
        self._zero: torch.Tensor
        self.register_buffer(
            "_zero", torch.zeros(batch_size, dtype=torch.int32, device=device)
        )
        self._nan: torch.Tensor
        self.register_buffer(
            "_nan", torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
        )

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Fully reset internal stage and base price."""
        state["maint_stage"] = self._zero.clone()
        state["base_price"] = self._nan.clone()
        state["entry_date_idx"] = torch.full_like(self._zero, -1)
        state["entry_time_idx"] = torch.full_like(self._zero, -1)

    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        state["maint_stage"] = torch.where(mask, self._zero, state["maint_stage"])
        state["base_price"] = torch.where(mask, self._nan, state["base_price"])
        state["entry_date_idx"] = torch.where(
            mask, state["date_idx"], state["entry_date_idx"]
        )
        state["entry_time_idx"] = torch.where(
            mask, state["time_idx"], state["entry_time_idx"]
        )

    def forward(
        self,
        state: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self._action.clone()
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        entry_price = state["entry_price"]
        stop_loss = state["prev_stop_loss"].clone()
        position = state["position"]
        base_price = state["base_price"]
        stage = state["maint_stage"]
        entry_date_idx = state["entry_date_idx"]
        entry_time_idx = state["entry_time_idx"]
        has_position_mask = position != 0

        nan_base_mask = torch.isnan(base_price) & has_position_mask
        if self.wait_for_breakeven:
            base_price = torch.where(nan_base_mask, entry_price, base_price)
        else:
            finite_prev_stop = torch.isfinite(stop_loss)
            set_mask = nan_base_mask & finite_prev_stop
            base_price = torch.where(set_mask, stop_loss, base_price)

        stage0_mask = (stage == 0) & has_position_mask

        # Check time condition for stage 0 -> 1 transition
        conv_entry_date_idx = self.conv_date_idx[:, entry_date_idx, entry_time_idx]
        conv_entry_time_idx = self.conv_time_idx[:, entry_date_idx, entry_time_idx]
        conv_date_idx = self.conv_date_idx[:, date_idx, time_idx]
        conv_time_idx = self.conv_time_idx[:, date_idx, time_idx]

        # Ensure a full period of the next stage has passed since entry
        time_condition = (
            (conv_entry_date_idx[1] >= 0)
            & (
                (conv_date_idx[1] > conv_entry_date_idx[1])
                | (
                    (conv_date_idx[1] == conv_entry_date_idx[1])
                    & (conv_time_idx[1] > conv_entry_time_idx[1])
                )
            )
        )

        if self.wait_for_breakeven:
            potential_stop = self.artr_policies[0](
                state,
                self._zero,
            )
            improve_mask_subset = stage0_mask & (
                (position > 0) & (potential_stop > entry_price)
                | (position < 0) & (potential_stop < entry_price)
            )
            improve_mask_subset = improve_mask_subset & time_condition
            stop_loss = torch.where(improve_mask_subset, potential_stop, stop_loss)
            stage = torch.where(improve_mask_subset, 1, stage)
        else:
            stage = torch.where(stage0_mask & time_condition, 1, stage)

        for s in range(1, self.stage_count):
            stage_mask = (stage == s) & has_position_mask
            conv_state = {
                "date_idx": conv_date_idx[s],
                "time_idx": conv_time_idx[s],
                "position": position * stage_mask,
                "prev_stop_loss": stop_loss,
            }
            potential_stop = self.artr_policies[s](conv_state, self._zero)
            improvement = torch.where(
                position > 0,
                potential_stop - stop_loss,
                stop_loss - potential_stop,
            )
            min_improvement = self.minimum_improvement * torch.abs(
                base_price - stop_loss
            )
            improve_mask_subset = (
                stage_mask & (improvement > min_improvement) & (conv_date_idx[s] >= 0)
            )

            # Check time condition for stage s -> s+1 transition (if not last stage)
            if s < self.stage_count - 1:
                time_condition = (
                    (conv_entry_date_idx[s + 1] >= 0)
                    & (
                        (conv_date_idx[s + 1] > conv_entry_date_idx[s + 1])
                        | (
                            (conv_date_idx[s + 1] == conv_entry_date_idx[s + 1])
                            & (conv_time_idx[s + 1] > conv_entry_time_idx[s + 1])
                        )
                    )
                )
                improve_mask_subset = improve_mask_subset & time_condition

            stop_loss = torch.where(improve_mask_subset, potential_stop, stop_loss)
            stage = torch.where(
                improve_mask_subset & (s < self.stage_count - 1),
                s + 1,
                stage,
            )

        state["maint_stage"] = stage
        state["base_price"] = base_price
        # Note: entry_date_idx and entry_time_idx are set in masked_reset and should not be modified here

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

        self._data: torch.Tensor
        self.register_buffer("_data", instrument_data.data)
        self.trailing_stop = trailing_stop
        self.skip_stage1 = skip_stage1
        self.keep_percent = keep_percent
        self.anchor_type = anchor_type
        self.artr_policy = ArtrStopLossPolicy(instrument_data, stage1_atr_multiple)

        device = instrument_data.device
        dtype = instrument_data.dtype
        initial_stage = 1 if self.skip_stage1 else 0
        self._action: torch.Tensor
        self.register_buffer(
            "_action", torch.zeros(batch_size, dtype=torch.int32, device=device)
        )
        self._initial_stage: torch.Tensor
        self.register_buffer(
            "_initial_stage",
            torch.full((batch_size,), initial_stage, dtype=torch.long, device=device),
        )
        self._nan: torch.Tensor
        self.register_buffer(
            "_nan", torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
        )

    def reset(self, state: dict[str, torch.Tensor]) -> None:
        """Fully reset stage and anchor state."""
        state["maint_stage"] = self._initial_stage.clone()
        state["maint_anchor"] = self._nan.clone()

    def masked_reset(self, state: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        state["maint_stage"] = torch.where(
            mask, self._initial_stage, state["maint_stage"]
        )
        state["maint_anchor"] = torch.where(mask, self._nan, state["maint_anchor"])

    def forward(
        self,
        state: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self._action
        stop_loss = state["prev_stop"].clone()
        position = state["position"]
        entry_price = state["entry_price"]
        prev_stop = state["prev_stop"]
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        anchor = state["maint_anchor"]
        stage = state["maint_stage"]

        has_position_mask = position != 0

        first_call_mask = torch.isnan(anchor) & has_position_mask
        if self.anchor_type == "entry":
            anchor = torch.where(first_call_mask, entry_price, anchor)
        elif self.anchor_type == "initial_stop":
            anchor = torch.where(first_call_mask, prev_stop, anchor)
        elif self.anchor_type == "artificial":
            reference_channel = torch.where(position > 0, 1, 2)
            reference_price = self._data[date_idx, time_idx, reference_channel]
            anchor = torch.where(
                first_call_mask,
                (prev_stop - self.keep_percent * reference_price)
                / (1 - self.keep_percent),
                anchor,
            )

        stage1_mask = (stage == 0) & has_position_mask
        atr_stop = self.artr_policy(
            date_idx, time_idx, position * stage1_mask, action, prev_stop
        )
        move_to_stage2 = stage1_mask & (
            (position > 0) & (atr_stop > entry_price)
            | (position < 0) & (atr_stop < entry_price)
        )

        if self.anchor_type == "last_stage1_stop":
            anchor = torch.where(move_to_stage2, atr_stop, anchor)
        elif self.anchor_type == "artificial_stage2":
            reference_channel = torch.where(position > 0, 1, 2)
            reference_price = self._data[date_idx, time_idx, reference_channel]
            anchor = torch.where(
                move_to_stage2,
                (atr_stop - self.keep_percent * reference_price)
                / (1 - self.keep_percent),
                anchor,
            )

        stage = torch.where(move_to_stage2, 1, stage)

        if self.trailing_stop:
            stop_loss = torch.where(stage1_mask, atr_stop, stop_loss)

        stage2_mask = (stage == 1) & has_position_mask
        reference_channel = torch.where(position > 0, 1, 2)
        reference_price = self._data[date_idx, time_idx, reference_channel]
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

        state["maint_stage"] = stage
        state["maint_anchor"] = anchor

        return action, stop_loss
