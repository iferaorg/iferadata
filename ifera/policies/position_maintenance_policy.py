"""Policies for managing open positions and updating stops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch._dynamo  # pylint: disable=protected-access
from torch import device, nn
import tensordict as td
from einops import repeat
from ..data_models import InstrumentData, DataManager
from ..config import ConfigManager
from ..file_manager import FileManager
from .stop_loss_policy import ArtrStopLossPolicy
from .policy_base import PolicyBase

torch._dynamo.config.capture_scalar_outputs = True  # pylint: disable=protected-access


class ScaledArtrMaintenancePolicy(PolicyBase):
    """A position maintenance policy that adjusts stop-loss using scaled ATR."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        stages: List[str],
        atr_multiple: float,
        wait_for_breakeven: bool,
        minimum_improvement: float,
    ) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.stages = stages
        self.atr_multiple = atr_multiple
        self.wait_for_breakeven = wait_for_breakeven
        self.minimum_improvement = minimum_improvement
        self.dtype = instrument_data.dtype

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

        # Register helper tensors as buffers so they get moved with .to(device)
        self._action: torch.Tensor
        self._zero: torch.Tensor
        self._nan: torch.Tensor
        self.register_buffer(
            "_action",
            torch.tensor((), dtype=torch.int32, device=instrument_data.device),
        )
        self.register_buffer(
            "_zero", torch.tensor((), dtype=torch.int32, device=instrument_data.device)
        )
        self.register_buffer(
            "_nan", torch.tensor((), dtype=self.dtype, device=instrument_data.device)
        )

    def copy_to(self, device: torch.device) -> ScaledArtrMaintenancePolicy:
        """Return a copy of the policy on the specified device."""
        return ScaledArtrMaintenancePolicy(
            self.instrument_data.copy_to(device),
            stages=self.stages,
            atr_multiple=self.atr_multiple,
            wait_for_breakeven=self.wait_for_breakeven,
            minimum_improvement=self.minimum_improvement,
        )

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """Fully reset internal stage and base price."""
        # Create or recreate helper buffers for the new batch size and device
        self._action = torch.zeros(
            state.batch_size, dtype=torch.int32, device=state.device
        )
        self._zero = torch.zeros(
            state.batch_size, dtype=torch.int32, device=state.device
        )
        self._nan = torch.full(
            state.batch_size, float("nan"), dtype=self.dtype, device=state.device
        )

        for s in range(self.stage_count):
            self.artr_policies[s].reset(state)  # type: ignore

        td_out = td.TensorDict(
            {
                "maint_stage": self._zero.clone(),
                "base_price": self._nan.clone(),
                "entry_date_idx": torch.full_like(self._zero, -1),
                "entry_time_idx": torch.full_like(self._zero, -1),
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        td_out = td.TensorDict(
            {
                "maint_stage": torch.where(mask, self._zero, state["maint_stage"]),
                "base_price": torch.where(mask, self._nan, state["base_price"]),
                "entry_date_idx": torch.where(mask, -1, state["entry_date_idx"]),
                "entry_time_idx": torch.where(mask, -1, state["entry_time_idx"]),
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        entry_price = state["entry_price"]
        stop_loss = state["prev_stop_loss"]
        position = state["position"]
        base_price = state["base_price"]
        stage = state["maint_stage"]
        entry_date_idx = state["entry_date_idx"]
        entry_time_idx = state["entry_time_idx"]
        has_position_mask = state["has_position_mask"]
        action = torch.where(has_position_mask, self._action, state["action"])

        nan_base_mask = torch.isnan(base_price) & has_position_mask
        if self.wait_for_breakeven:
            base_price = torch.where(nan_base_mask, entry_price, base_price)
        else:
            finite_prev_stop = torch.isfinite(stop_loss)
            set_mask = nan_base_mask & finite_prev_stop
            base_price = torch.where(set_mask, stop_loss, base_price)

        # Check time condition for stage 0 -> 1 transition
        conv_entry_date_idx = self.conv_date_idx[:, entry_date_idx, entry_time_idx]
        conv_entry_time_idx = self.conv_time_idx[:, entry_date_idx, entry_time_idx]
        conv_date_idx = self.conv_date_idx[:, date_idx, time_idx]
        conv_time_idx = self.conv_time_idx[:, date_idx, time_idx]

        # Ensure a full period of the next stage has passed since entry
        time_condition = torch.zeros_like(conv_date_idx, dtype=torch.bool)
        time_condition[:-1] = (conv_entry_date_idx[1:] >= 0) & (
            (conv_date_idx[1:] > conv_entry_date_idx[1:])
            | (
                (conv_date_idx[1:] == conv_entry_date_idx[1:])
                & (conv_time_idx[1:] > conv_entry_time_idx[1:])
            )
        )

        stage0_mask = (stage == 0) & has_position_mask

        if self.wait_for_breakeven:
            artr_state = td.TensorDict(
                {
                    "date_idx": date_idx,
                    "time_idx": time_idx,
                    "position": position * stage0_mask,
                    "prev_stop_loss": stop_loss,
                    "action": self._zero,
                },
                batch_size=state.batch_size,
                device=state.device,
            )
            potential_stop = self.artr_policies[0](artr_state)["stop_loss"]
            improve_mask_subset = stage0_mask & (
                (position > 0) & (potential_stop > entry_price)
                | (position < 0) & (potential_stop < entry_price)
            )
            improve_mask_subset = improve_mask_subset & time_condition[0]
            stop_loss = torch.where(improve_mask_subset, potential_stop, stop_loss)
            stage = torch.where(improve_mask_subset, 1, stage)
        else:
            stage = torch.where(stage0_mask & time_condition[0], 1, stage)

        for s in range(1, self.stage_count):
            stage_mask = (stage == s) & has_position_mask
            # Create a temporary state for the artr policy
            conv_state = td.TensorDict(
                {
                    "date_idx": conv_date_idx[s],
                    "time_idx": conv_time_idx[s],
                    "position": position * stage_mask,
                    "prev_stop_loss": stop_loss,
                    "action": self._zero,
                },
                batch_size=state.batch_size,
                device=state.device,
            )
            potential_stop = self.artr_policies[s](conv_state)["stop_loss"]
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
            improve_mask_subset = improve_mask_subset & time_condition[s]

            stop_loss = torch.where(improve_mask_subset, potential_stop, stop_loss)
            stage = torch.where(
                improve_mask_subset,
                s + 1,
                stage,
            )

        td_out = td.TensorDict(
            {
                "action": action,
                "stop_loss": stop_loss,
                "maint_stage": stage,
                "base_price": base_price,
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out


class PercentGainMaintenancePolicy(PolicyBase):
    """Maintain a percentage of unrealized gains using ATR for stage one."""

    def __init__(
        self,
        instrument_data: InstrumentData,
        stage1_atr_multiple: float,
        trailing_stop: bool,
        skip_stage1: bool,
        keep_percent: float,
        anchor_type: str,
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
        self.instrument_data = instrument_data
        self.stage1_atr_multiple = stage1_atr_multiple
        self.trailing_stop = trailing_stop
        self.skip_stage1 = skip_stage1
        self.keep_percent = keep_percent
        self.anchor_type = anchor_type
        self.artr_policy = ArtrStopLossPolicy(instrument_data, stage1_atr_multiple)

        # Store device, dtype and initial stage for lazy buffer creation
        self._device = instrument_data.device
        self._dtype = instrument_data.dtype
        self._initial_stage_value = 1 if self.skip_stage1 else 0
        # Register helper tensors as buffers so they get moved with .to(device)
        self._action: torch.Tensor
        self._initial_stage: torch.Tensor
        self._nan: torch.Tensor
        self.register_buffer(
            "_action",
            torch.tensor((), dtype=torch.int32, device=instrument_data.device),
        )
        self.register_buffer(
            "_initial_stage",
            torch.tensor((), dtype=torch.long, device=instrument_data.device),
        )
        self.register_buffer(
            "_nan", torch.tensor((), dtype=self._dtype, device=instrument_data.device)
        )

    def copy_to(self, device: torch.device) -> PercentGainMaintenancePolicy:
        """Return a copy of the policy on the specified device."""
        return PercentGainMaintenancePolicy(
            self.instrument_data.copy_to(device),
            stage1_atr_multiple=self.stage1_atr_multiple,
            trailing_stop=self.trailing_stop,
            skip_stage1=self.skip_stage1,
            keep_percent=self.keep_percent,
            anchor_type=self.anchor_type,
        )

    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """Fully reset stage and anchor state."""
        # Update device if it has changed
        if hasattr(self, "_device"):
            self._device = state.device

        # Create buffers for the new batch size and device
        self._action = torch.zeros(
            state.batch_size, dtype=torch.int32, device=state.device
        )
        self._nan = torch.full(
            state.batch_size, float("nan"), dtype=self._dtype, device=state.device
        )

        self._initial_stage = torch.full(
            state.batch_size,
            self._initial_stage_value,
            dtype=torch.long,
            device=state.device,
        )

        td_out = td.TensorDict(
            {
                "maint_stage": self._initial_stage.clone(),
                "maint_anchor": self._nan.clone(),
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        td_out = td.TensorDict(
            {
                "maint_stage": torch.where(
                    mask, self._initial_stage, state["maint_stage"]
                ),
                "maint_anchor": torch.where(mask, self._nan, state["maint_anchor"]),
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out

    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        action = self._action
        stop_loss = state["prev_stop_loss"].clone()
        position = state["position"]
        entry_price = state["entry_price"]
        prev_stop = state["prev_stop_loss"]
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        anchor = state["maint_anchor"]
        stage = state["maint_stage"]
        has_position_mask = state["has_position_mask"]

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
        atr_stop = self.artr_policy(state, action)
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

        td_out = td.TensorDict(
            {
                "action": action,
                "stop_loss": stop_loss,
                "maint_stage": stage,
                "maint_anchor": anchor,
            },
            batch_size=state.batch_size,
            device=state.device,
        )

        return td_out
