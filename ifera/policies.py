"""
Module containing trading policy implementations for making trading decisions.

Trading policies are responsible for determining actions based on market data
and current trading state. The module provides a composite policy structure
that delegates decisions to specialized sub-policies depending on the current
trading context.
"""

import torch
from torch import nn
from typing import List, Tuple, Union, Dict
from abc import ABC, abstractmethod
from .data_models import InstrumentData, DataManager
from .config import ConfigManager


class PositionMaintenancePolicy(nn.Module, ABC):
    """
    Abstract base class for position maintenance policies.

    Position maintenance policies are responsible for managing existing positions,
    including updating stop loss levels and determining when to close positions.

    All implementations must provide reset and forward methods with the specified
    signatures.
    """

    @abstractmethod
    def reset(self, mask: torch.Tensor) -> None:
        """
        Reset the policy's state for the specified batch elements.

        Args:
            mask (torch.Tensor): A boolean tensor where True indicates batches to reset.
        """
        pass

    @abstractmethod
    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actions and stop loss levels for the current state.

        Args:
            date_idx (torch.Tensor): Batch of date indices.
            time_idx (torch.Tensor): Batch of time indices.
            position (torch.Tensor): Current positions (non-zero for existing positions).
            prev_stop (torch.Tensor): Previous stop loss levels.
            entry_price (torch.Tensor): Entry prices for the positions.
            batch_indices (torch.Tensor): Indices of the batch elements.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action, stop_loss)
                - action: Actions to take (0 = no action, positive = buy, negative = sell)
                - stop_loss: Updated stop loss levels.
        """
        pass


class TradingPolicy(nn.Module):
    """
    A composite trading policy that delegates to specialized sub-policies.

    This policy orchestrates multiple sub-policies that handle different aspects
    of trading decision making:
    - Opening new positions
    - Setting initial stop loss levels
    - Managing existing positions
    - Determining when episodes should terminate (for backtesting/RL)

    All methods operate on batched inputs and outputs, supporting vectorized
    operations for efficiency.

    Parameters
    ----------
    instrument_data : InstrumentData
        Data for the financial instrument being traded
    open_position_policy : object
        Policy for determining actions when current position is 0
    initial_stop_loss_policy : object
        Policy for setting stop loss levels when opening new positions
    position_maintenance_policy : PositionMaintenancePolicy
        Policy for managing existing positions (actions and stop loss updates)

    Notes
    -----
    All sub-policies must implement a forward method with compatible signatures.
    """

    def __init__(
        self,
        instrument_data: InstrumentData,
        open_position_policy: torch.nn.Module,
        initial_stop_loss_policy: torch.nn.Module,
        position_maintenance_policy: PositionMaintenancePolicy,
    ) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.open_position_policy = open_position_policy
        self.initial_stop_loss_policy = initial_stop_loss_policy
        self.position_maintenance_policy = position_maintenance_policy

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determine trading actions and stop loss levels based on current state.

        This method delegates to the appropriate sub-policies based on the current
        position state and already-done episodes.

        Parameters
        ----------
        date_idx : torch.Tensor
            Batch of date indices
        time_idx : torch.Tensor
            Batch of time indices
        position : torch.Tensor
            Current positions for each batch element (0 = no position)
        prev_stop : torch.Tensor
            Previous stop loss levels
        entry_price : torch.Tensor
            Entry price for current positions

        Returns
        -------
        action : torch.Tensor
            Actions to take (0 = no action, positive = buy, negative = sell)
        stop_loss : torch.Tensor
            Stop loss price levels
        """
        # Initialize result tensors
        action = torch.zeros_like(position)
        stop_loss = torch.full_like(prev_stop, float("nan"))

        # Create masks for different position states
        no_position_mask = position == 0
        has_position_mask = position != 0

        # Handle batches with no position (position == 0)
        if no_position_mask.any():
            # Get actions for opening positions
            open_actions = self.open_position_policy(
                date_idx[no_position_mask],
                time_idx[no_position_mask],
                position[no_position_mask],
            )
            action[no_position_mask] = open_actions

            # For batches where we're opening a position, get initial stop loss
            opening_position_mask = no_position_mask & (action != 0)
            if opening_position_mask.any():
                initial_stops = self.initial_stop_loss_policy(
                    date_idx[opening_position_mask],
                    time_idx[opening_position_mask],
                    action[opening_position_mask],
                )
                stop_loss[opening_position_mask] = initial_stops
                self.position_maintenance_policy.reset(opening_position_mask)

        # Handle batches with existing positions
        if has_position_mask.any():
            maintenance_actions, maintenance_stops = self.position_maintenance_policy(
                date_idx[has_position_mask],
                time_idx[has_position_mask],
                position[has_position_mask],
                prev_stop[has_position_mask],
                entry_price[has_position_mask],
                torch.where(has_position_mask)[0],
            )
            action[has_position_mask] = maintenance_actions
            stop_loss[has_position_mask] = maintenance_stops

        return action, stop_loss


class AlwaysOpenPolicy(nn.Module):
    """
    Policy that always opens a position.

    This policy is used to test the trading policy structure with a simple
    sub-policy that always opens a position.

    Parameters
    ----------
    direction : int
        Direction of the position (1 = long, -1 = short
    """

    def __init__(self, direction) -> None:
        super().__init__()
        self.direction = direction

    def forward(
        self, date_idx: torch.Tensor, time_idx: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine actions for opening new positions.

        Parameters
        ----------
        date_idx : torch.Tensor
            Batch of date indices
        time_idx : torch.Tensor
            Batch of time indices
        position : torch.Tensor
            Current positions for each batch element (0 = no position)

        Returns
        -------
        action : torch.Tensor
            Actions to take (0 = no action, positive = buy, negative = sell)
        """
        _, _ = date_idx, time_idx
        return torch.ones_like(position) * self.direction


class ArtrStopLossPolicy(nn.Module):
    """
    Policy for setting stop loss levels based on ATR multiples.

    This policy sets stop loss levels based on the average true range (ATR)
    of the instrument data. The stop loss is set at a multiple of the ATR
    from the current price.

    Parameters
    ----------
    instrument_data : InstrumentData
        Data for the financial instrument being traded
    atr_multiple : float
        Multiple of the ATR to use for setting stop loss levels
    """

    def __init__(self, instrument_data: InstrumentData, atr_multiple: float) -> None:
        super().__init__()
        self.idata = instrument_data
        self.atr_multiple = atr_multiple

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        action: torch.Tensor,
        prev_stop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Determine stop loss levels for opening new positions.

        Parameters
        ----------
        date_idx : torch.Tensor
            Batch of date indices
        time_idx : torch.Tensor
            Batch of time indices
        position : torch.Tensor
            Current positions for each batch element (0 = no
            position, positive = long, negative = short)
        action : torch.Tensor
            Actions to take (0 = no action, positive = buy, negative = sell)
        prev_stop : torch.Tensor
            Previous stop loss levels

        Returns
        -------
        stop_loss : torch.Tensor
            Stop loss price levels
        """
        direction = (position + action).sign()

        # Replace NaN values in prev_stop with infinity (negative for long positions,
        # positive for short positions) to ensure proper stop loss initialization
        prev_stop = torch.where(
            torch.isnan(prev_stop), torch.inf * direction * -1, prev_stop
        )

        # Relative ATR multiplier for stop loss levels
        artr = self.idata.artr[date_idx, time_idx] * self.atr_multiple + 1.0

        # For new positions use the close, for trailing stops use the high/low
        reference_channel = torch.where(
            position == 0, 3, torch.where(direction > 0, 1, 2)
        )
        reference_price = self.idata.data[date_idx, time_idx, reference_channel]

        stop_price = torch.where(
            direction > 0,
            torch.maximum(prev_stop, reference_price / artr),
            torch.minimum(prev_stop, reference_price * artr),
        )
        stop_price = torch.where(torch.isnan(stop_price), prev_stop, stop_price)

        return stop_price


class InitialArtrStopLossPolicy(nn.Module):
    """
    Policy for setting initial stop loss levels based on ATR multiples.

    This policy sets stop loss levels based on the average true range (ATR)
    of the instrument data. The stop loss is set at a multiple of the ATR
    from the current price.

    Parameters
    ----------
    instrument_data : InstrumentData
        Data for the financial instrument being traded
    atr_multiple : float
        Multiple of the ATR to use for setting stop loss levels
    """

    def __init__(self, instrument_data: InstrumentData, atr_multiple: float) -> None:
        super().__init__()
        self.artr_policy = ArtrStopLossPolicy(instrument_data, atr_multiple)
        self.dtype = instrument_data.data.dtype

    def forward(
        self, date_idx: torch.Tensor, time_idx: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine stop loss levels for opening new positions.

        Parameters
        ----------
        date_idx : torch.Tensor
            Batch of date indices
        time_idx : torch.Tensor
            Batch of time indices
        action : torch.Tensor
            Actions to take (0 = no action, positive = buy, negative = sell)

        Returns
        -------
        stop_loss : torch.Tensor
            Stop loss price levels
        """
        return self.artr_policy(
            date_idx,
            time_idx,
            torch.zeros_like(action),
            action,
            torch.full_like(action, float("nan"), dtype=self.dtype),
        )


class ScaledArtrMaintenancePolicy(PositionMaintenancePolicy):
    """
    A position maintenance policy that adjusts stop loss levels using scaled ARTR.

    This policy uses multiple stages with increasing time intervals to calculate
    stop loss levels based on ARTR. It progresses through stages when certain
    conditions are met, such as achieving breakeven or sufficient improvement
    in the stop loss level.

    Attributes:
        instrument_data (InstrumentData): Data for the financial instrument.
        stages (List[str]): List of time intervals (e.g., ["1m", "5m", "1h"]).
        atr_multiple (float): Multiplier for ARTR in stop loss calculations.
        wait_for_breakeven (bool): Whether to wait for breakeven before advancing stages.
        minimum_improvement (float): Minimum improvement ratio for stage progression.
        stage (torch.Tensor): Current stage for each batch element.
        base_price (torch.Tensor): Base price for improvement calculations.
    """

    def __init__(
        self,
        instrument_data: InstrumentData,
        stages: Union[List[str], Dict[str, str]],
        atr_multiple: float,
        wait_for_breakeven: bool,
        minimum_improvement: float,
    ) -> None:
        super().__init__()
        self.instrument_data = instrument_data
        self.atr_multiple = atr_multiple
        self.wait_for_breakeven = wait_for_breakeven
        self.minimum_improvement = minimum_improvement

        config_manager = ConfigManager()
        data_manager = DataManager()

        # Handle stages input: list or dict
        if isinstance(stages, list):
            # For a list, assume each stage is derived from the previous one
            base_stage = stages[0]
            parent_dict = {stages[0]: ""}
            for i in range(1, len(stages)):
                parent_dict[stages[i]] = stages[i - 1]
            stage_list = stages
        elif isinstance(stages, dict):
            # For a dict, use key-value pairs where value is the parent interval
            base_stages = [k for k, v in stages.items() if v == "" or v is None]
            if len(base_stages) != 1:
                raise ValueError("Exactly one stage must have an empty or None parent.")
            base_stage = base_stages[0]
            parent_dict = stages
            stage_list = list(
                stages.keys()
            )  # Order of stages follows dict insertion order
        else:
            raise TypeError(
                "stages must be a list of strings or a dict of stage:parent."
            )

        # Validate that the base stage matches the instrument's interval
        if base_stage != instrument_data.instrument.interval:
            raise ValueError("The base stage must match the instrument's interval.")

        # Build configurations for each stage
        config_dict = {base_stage: instrument_data.instrument}
        processed = {base_stage}

        while len(processed) < len(stage_list):
            added = False
            for stage in stage_list:
                if stage not in processed and parent_dict[stage] in processed:
                    parent_config = config_dict[parent_dict[stage]]
                    derived_config = config_manager.create_derived_base_config(
                        parent_config, stage
                    )
                    config_dict[stage] = derived_config
                    processed.add(stage)
                    added = True
            if not added:
                raise ValueError(
                    "Cannot derive all stages: missing parent or cycle detected."
                )

        # Create derived data and policies in the order of stage_list
        self.derived_configs = [config_dict[stage] for stage in stage_list]
        self.derived_data = [
            data_manager.get_instrument_data(
                config,
                dtype=instrument_data.dtype,
                device=instrument_data.device,
            )
            for config in self.derived_configs
        ]
        self.artr_policies = [
            ArtrStopLossPolicy(data, self.atr_multiple) for data in self.derived_data
        ]
        self.stage_count = len(stage_list)

        # Initialize state tensors (to be set in reset)
        self.stage = torch.tensor(())
        self.base_price = torch.tensor(())

    def reset(self, mask: torch.Tensor) -> None:
        """
        Reset the policy's state tensors for a given batch size.

        Args:
            mask (torch.Tensor): A boolean tensor where True indicates batches to reset.
        """
        if self.stage.shape[0] == 0:
            device = self.instrument_data.device
            dtype = self.instrument_data.dtype
            batch_size = mask.shape[0]
            self.stage = torch.zeros(batch_size, dtype=torch.long, device=device)
            self.base_price = torch.full(
                (batch_size,), float("nan"), dtype=dtype, device=device
            )
        else:
            self.stage[mask] = 0
            self.base_price[mask] = float("nan")

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actions and stop loss levels for the current state.

        This method always returns an action of 0 (no new trades) and updates
        stop loss levels based on ARTR calculations across different stages.

        Args:
            date_idx (torch.Tensor): Batch of date indices.
            time_idx (torch.Tensor): Batch of time indices.
            position (torch.Tensor): Current positions (non-zero for existing positions).
            prev_stop (torch.Tensor): Previous stop loss levels.
            entry_price (torch.Tensor): Entry prices for the positions.
            batch_indices (torch.Tensor): Indices of the batch elements.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action, stop_loss)
                - action: Always zero tensor.
                - stop_loss: Updated stop loss levels.

        Raises:
            ValueError: If the policy state is not initialized or batch size mismatches.
        """
        # Check if state is initialized
        if self.stage.shape[0] == 0 or batch_indices.max() >= self.stage.shape[0]:
            raise ValueError(
                "Policy state not initialized or batch indices out of range. Call reset with appropriate batch size first."
            )

        # Initialize outputs
        action = torch.zeros_like(position)
        stop_loss = prev_stop.clone()

        # Extract current state for the subset
        current_stage = self.stage[batch_indices]
        current_base_price = self.base_price[batch_indices]

        # Set base_price
        nan_base_mask = torch.isnan(current_base_price)
        if self.wait_for_breakeven:
            self.base_price[batch_indices[nan_base_mask]] = entry_price[nan_base_mask]
            current_base_price[nan_base_mask] = entry_price[nan_base_mask]
        else:
            finite_prev_stop = ~torch.isnan(prev_stop) & ~torch.isinf(prev_stop)
            set_mask = nan_base_mask & finite_prev_stop
            self.base_price[batch_indices[set_mask]] = prev_stop[set_mask]
            current_base_price[set_mask] = prev_stop[set_mask]

        # Handle stage 0
        stage0_mask = current_stage == 0
        if stage0_mask.any():
            if self.wait_for_breakeven:
                subset_date_idx = date_idx[stage0_mask]
                subset_time_idx = time_idx[stage0_mask]
                subset_position = position[stage0_mask]
                subset_stop_loss = stop_loss[stage0_mask]
                subset_entry_price = entry_price[stage0_mask]
                potential_stop = self.artr_policies[0](
                    subset_date_idx,
                    subset_time_idx,
                    subset_position,
                    torch.zeros_like(subset_position),
                    subset_stop_loss,
                )
                improve_mask_subset = (subset_position > 0) & (
                    potential_stop > subset_entry_price
                ) | (subset_position < 0) & (potential_stop < subset_entry_price)
                stop_loss[stage0_mask] = torch.where(
                    improve_mask_subset, potential_stop, subset_stop_loss
                )
                current_stage[stage0_mask] = torch.where(
                    improve_mask_subset,
                    torch.tensor(1, dtype=torch.long, device=self.stage.device),
                    current_stage[stage0_mask],
                )
            else:
                current_stage[stage0_mask] = 1

        # Handle stages > 0
        for s in range(1, self.stage_count):
            stage_mask = current_stage == s
            if stage_mask.any():
                subset_date_idx = date_idx[stage_mask]
                subset_time_idx = time_idx[stage_mask]
                subset_date_idx, subset_time_idx = self.derived_data[s].convert_indices(
                    self.derived_data[0], subset_date_idx, subset_time_idx
                )
                subset_position = position[stage_mask]
                subset_stop_loss = stop_loss[stage_mask]
                subset_base_price = current_base_price[stage_mask]
                potential_stop = self.artr_policies[s](
                    subset_date_idx,
                    subset_time_idx,
                    subset_position,
                    torch.zeros_like(subset_position),
                    subset_stop_loss,
                )
                improvement = torch.where(
                    subset_position > 0,
                    potential_stop - subset_stop_loss,
                    subset_stop_loss - potential_stop,
                )
                min_improvement = self.minimum_improvement * torch.abs(
                    subset_base_price - subset_stop_loss
                )
                improve_mask_subset = improvement > min_improvement
                stop_loss[stage_mask] = torch.where(
                    improve_mask_subset, potential_stop, subset_stop_loss
                )
                if s < self.stage_count - 1:
                    current_stage[stage_mask] = torch.where(
                        improve_mask_subset,
                        torch.tensor(s + 1, dtype=torch.long, device=self.stage.device),
                        current_stage[stage_mask],
                    )

        # Update the full state with the modified subset
        self.stage[batch_indices] = current_stage
        self.base_price[batch_indices] = current_base_price

        return action, stop_loss


class PercentGainMaintenancePolicy(PositionMaintenancePolicy):
    """
    A position maintenance policy that adjusts stop-loss levels to maintain a fixed percentage of unrealized gains.

    This policy operates in two stages:
    - **Stage 1**: Waits until the price moves N ATR into profit, using ArtrStopLossPolicy.
            Transitions to Stage 2 when the ATR-based stop-loss exceeds the entry price (higher for long, lower for short).
    - **Stage 2**: Sets the stop-loss to retain a fixed percentage of unrealized gains relative to an anchor price, acting as a trailing stop.

    Attributes:
        instrument_data (InstrumentData): Financial instrument data.
        stage1_atr_multiple (float): ATR multiple for Stage 1 stop-loss calculation.
        trailing_stop (bool): Whether to adjust stop-loss as a trailing stop in Stage 1.
        skip_stage1 (bool): If True, skips Stage 1 and starts at Stage 2.
        keep_percent (float): Percentage of unrealized gains to retain (0 < keep_percent < 1).
        anchor_type (str): Anchor price type ("entry", "initial_stop", "artificial", "last_stage1_stop", "artificial_stage2").
        artr_policy (ArtrStopLossPolicy): Policy for ATR-based stop-loss in Stage 1.
        stage (torch.Tensor): Current stage per batch element (0 = Stage 1, 1 = Stage 2).
        anchor (torch.Tensor): Anchor price per batch element for Stage 2 calculations.
    """

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
        # Validate anchor_type and conditions
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
                "anchor_type 'last_stage1_stop' and 'artificial_stage2' require skip_stage1=False and trailing_stop=True"
            )
        if anchor_type in ["artificial", "artificial_stage2"] and not (
            0 < keep_percent < 1
        ):
            raise ValueError(
                "keep_percent must be between 0 and 1 for 'artificial' and 'artificial_stage2' anchor_types"
            )

        self.instrument_data = instrument_data
        self.trailing_stop = trailing_stop
        self.skip_stage1 = skip_stage1
        self.keep_percent = keep_percent
        self.anchor_type = anchor_type
        self.artr_policy = ArtrStopLossPolicy(instrument_data, stage1_atr_multiple)

        # Initialize state tensors (populated in reset)
        self.stage = torch.tensor(())
        self.anchor = torch.tensor(())

    def reset(self, mask: torch.Tensor) -> None:
        """
        Reset the policy's state for specified batch elements.

        Args:
            mask (torch.Tensor): Boolean tensor where True indicates batches to reset.
        """
        if self.stage.shape[0] == 0:
            device = self.instrument_data.device
            dtype = self.instrument_data.dtype
            batch_size = mask.shape[0]
            initial_stage = 1 if self.skip_stage1 else 0
            self.stage = torch.full(
                (batch_size,), initial_stage, dtype=torch.long, device=device
            )
            self.anchor = torch.full(
                (batch_size,), float("nan"), dtype=dtype, device=device
            )
        else:
            initial_stage = 1 if self.skip_stage1 else 0
            self.stage[mask] = initial_stage
            self.anchor[mask] = float("nan")

    def forward(
        self,
        date_idx: torch.Tensor,
        time_idx: torch.Tensor,
        position: torch.Tensor,
        prev_stop: torch.Tensor,
        entry_price: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actions and stop-loss levels based on the current state.

        Args:
            date_idx (torch.Tensor): Batch of date indices.
            time_idx (torch.Tensor): Batch of time indices.
            position (torch.Tensor): Current positions (non-zero).
            prev_stop (torch.Tensor): Previous stop-loss levels.
            entry_price (torch.Tensor): Entry prices for positions.
            batch_indices (torch.Tensor): Indices of batch elements to process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action, stop_loss)
                - action: Always zero (no new positions opened).
                - stop_loss: Updated stop-loss levels.
        """
        if self.stage.shape[0] == 0 or batch_indices.max() >= self.stage.shape[0]:
            raise ValueError(
                "Policy state not initialized or batch indices out of range. Call reset first."
            )

        # Initialize outputs: no new actions, start with previous stop-loss
        action = torch.zeros_like(position)
        stop_loss = prev_stop.clone()

        # Get current state for the batch
        current_stage = self.stage[batch_indices]

        # Set anchor for certain anchor_types on first call
        first_call_mask = torch.isnan(self.anchor[batch_indices])
        set_anchor_mask = first_call_mask & (
            self.anchor_type in ["entry", "initial_stop", "artificial"]
        )
        if set_anchor_mask.any():
            subset = set_anchor_mask
            indices = batch_indices[subset]
            if self.anchor_type == "entry":
                self.anchor[indices] = entry_price[subset]
            elif self.anchor_type == "initial_stop":
                self.anchor[indices] = prev_stop[subset]
            elif self.anchor_type == "artificial":
                reference_channel = torch.where(
                    position[subset] > 0, 1, 2
                )  # 1 for high, 2 for low
                reference_price = self.instrument_data.data[
                    date_idx[subset], time_idx[subset], reference_channel
                ]
                self.anchor[indices] = (
                    prev_stop[subset] - self.keep_percent * reference_price
                ) / (1 - self.keep_percent)

        # Stage 1: ATR-based waiting period
        stage1_mask = current_stage == 0
        if stage1_mask.any():
            subset = stage1_mask
            indices = batch_indices[subset]
            atr_stop = self.artr_policy(
                date_idx[subset],
                time_idx[subset],
                position[subset],
                torch.zeros_like(position[subset]),
                prev_stop[subset],
            )
            move_to_stage2 = (position[subset] > 0) & (
                atr_stop > entry_price[subset]
            ) | (position[subset] < 0) & (atr_stop < entry_price[subset])
            if move_to_stage2.any():
                move_indices = indices[move_to_stage2]
                if self.anchor_type == "last_stage1_stop":
                    self.anchor[move_indices] = atr_stop[move_to_stage2]
                elif self.anchor_type == "artificial_stage2":
                    reference_channel = torch.where(
                        position[subset][move_to_stage2] > 0, 1, 2
                    )
                    reference_price = self.instrument_data.data[
                        date_idx[subset][move_to_stage2],
                        time_idx[subset][move_to_stage2],
                        reference_channel,
                    ]
                    self.anchor[move_indices] = (
                        atr_stop[move_to_stage2] - self.keep_percent * reference_price
                    ) / (1 - self.keep_percent)
                self.stage[move_indices] = 1
            if self.trailing_stop:
                stop_loss[subset] = atr_stop

        # Stage 2: Percent-based stop-loss
        stage2_mask = current_stage == 1
        if stage2_mask.any():
            subset = stage2_mask
            indices = batch_indices[subset]
            anchor = self.anchor[indices]
            # Use high for long, low for short as reference price
            reference_channel = torch.where(position[subset] > 0, 1, 2)
            reference_price = self.instrument_data.data[
                date_idx[subset], time_idx[subset], reference_channel
            ]
            # Calculate candidate stop-loss
            candidate_stop_loss = torch.where(
                position[subset] > 0,
                anchor + self.keep_percent * (reference_price - anchor),
                anchor - self.keep_percent * (anchor - reference_price),
            )
            # Ensure stop-loss only tightens (trailing stop behavior)
            stop_loss[subset] = torch.where(
                position[subset] > 0,
                torch.maximum(prev_stop[subset], candidate_stop_loss),
                torch.minimum(prev_stop[subset], candidate_stop_loss),
            )

        return action, stop_loss
