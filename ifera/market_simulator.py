"""
Module containing market simulator implementations for backtesting trading strategies.
"""

from typing import Final

import torch
from einops import rearrange

from .data_models import InstrumentData


class MarketSimulatorIntraday:
    """
    Class representing a market simulator for intraday trading.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument data for the market simulator.

    Attributes
    ----------
    instrument : InstrumentData
        Instrument data for the market simulator.
    data : torch.tensor
        Tensor containing the market data.
    dtype : torch.dtype
        Data type of the market data.
    device : torch.device
        Device on which the market data is stored.
    channels : dict
        Dictionary mapping channel names to their indices in the data tensor.
    use_max_commission_mask : torch.tensor
        Boolean tensor indicating whether to use the maximum commission percentage.
    slippage_pct : torch.tensor
        Slippage percentage for the market simulator.

    Methods
    -------
    calculate_step(date_idx, time_idx, position, action, stop_loss=None)
        Calculate the result of a trading step given the current state and an action.
    calculate_commission(action_abs, price)
        Calculate the commission for an action.
    """

    CHANNELS: Final = {
        "open": 0,
        "high": 1,
        "low": 2,
        "close": 3,
        "volume": 4,
    }

    def __init__(self, instrument_data: InstrumentData):
        self.instrument: Final = instrument_data.instrument
        self.data: Final = instrument_data.data
        self.mask: Final = instrument_data.valid_mask.any(dim=-1)  # Shape: [date, time]

        self.data_flat: Final = rearrange(self.data, "d t c -> (d t) c")

        self._use_max_commission_mask: Final = torch.tensor(
            self.instrument.max_commission_pct > 0.0,
            dtype=torch.bool,
            device=instrument_data.device,
        )
        self._slippage_pct: Final = (
            self.instrument.slippage / self.instrument.reference_price
        )
        self._inf_tensor: Final = torch.tensor(
            float("inf"), device=self.data.device, dtype=self.data.dtype
        )
        self._neg_inf_tensor: Final = torch.tensor(
            -float("inf"), device=self.data.device, dtype=self.data.dtype
        )
        self._nan_tensor: Final = torch.tensor(
            float("nan"), device=self.data.device, dtype=self.data.dtype
        )
        self._zero_tensor: Final = torch.tensor(
            0, device=self.data.device, dtype=torch.int64
        )

    # pylint: disable=too-many-locals
    @torch.compile()
    def calculate_step(
        self, position_action: torch.Tensor, stop_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the result of a trading step given the current state and an action.

        Parameters
        ----------
        position_action : torch.IntTensor
            Tensor containing the position and action information for each batch.
            Has shape of (batch_size, 4) where the columns are:
            position_action[..., 0]:
                Indices of the dates in the data tensor.
            position_action[..., 1]:
                Indices of the time steps in the data tensor.
            position_action[..., 2]:
                Current position. Positive for long positions, negative for short positions,
                and zero for no position.
            position_action[..., 3]:
                Action to take. 0 = do nothing, positive = buy, negative = sell.
                The absolute value is the number of contracts to buy or sell.
        stop_loss : torch.Tensor, optional
            The stop loss price level. If provided, the function will calculate the effect of
            the stop loss order, i.e., if the price reaches this level, the position will be
            closed to prevent further losses. The default is None.

        Returns
        -------
        profit : torch.Tensor
            Profit from closing positions during this step.
        new_position : torch.Tensor
            New position after the action is taken.
        execution_price : torch.Tensor
            Execution price for the action.
        cashflow : torch.Tensor
            Cashflow from the action.

        Shapes
        ------
        - position_action: (batch_size, 4)
        - stop_loss: (batch_size,)
        - profit: (batch_size,)
        - new_position: (batch_size,)

        This function calculates the profit from closing positions, the new position, and
        the new average entry price after taking the action. The action can extend the current
        position, shrink the current position, or switch the direction of the current position.
        The function also takes into account the commission for the action and the slippage in
        the execution price. If a stop loss level is provided, the function will also calculate
        the effect of the stop loss order.
        """
        date_idx = position_action[..., 0]
        time_idx = position_action[..., 1]
        position = position_action[..., 2]
        action = position_action[..., 3]

        action = torch.where(self.mask[date_idx, time_idx], action, self._zero_tensor)
        stop_loss = torch.where(
            self.mask[date_idx, time_idx], stop_loss, self._nan_tensor
        )

        (
            action_sign,
            action_abs,
            new_position,
            close_position,
            open_position,
            kept_position,
        ) = _calculate_positions(position, action)

        new_position_sign = torch.sign(new_position)

        current_price = self.data[date_idx, time_idx, self.CHANNELS["open"]]
        slippage = (current_price * self._slippage_pct).clamp(
            min=self.instrument.min_slippage
        )
        execution_price = current_price + slippage * action_sign
        commission = self.calculate_commission(action_abs, execution_price)

        # Create default stop loss if none provided
        inf_value = torch.where(
            new_position_sign == 1, self._neg_inf_tensor, self._inf_tensor
        )
        stop_loss = torch.where(stop_loss.isnan(), inf_value, stop_loss)

        stop_check_price = torch.where(
            new_position_sign == 1,
            self.data[date_idx, time_idx, self.CHANNELS["low"]],
            self.data[date_idx, time_idx, self.CHANNELS["high"]],
        )
        stop_mask = (new_position_sign * (stop_loss - stop_check_price)) > 0
        stop_position = new_position * stop_mask
        new_position = new_position - stop_position
        stop_price = torch.where(
            new_position_sign >= 0,
            torch.min(stop_loss, current_price),
            torch.max(stop_loss, current_price),
        ) - slippage * torch.sign(stop_position)
        stop_commision = self.calculate_commission(torch.abs(stop_position), stop_price)

        close_price = self.data[date_idx, time_idx, self.CHANNELS["close"]]
        close_price = torch.where(stop_mask, stop_price, close_price)

        prev_close_price = self._get_prev_close_price(date_idx, time_idx)

        position_value_delta = (
            (execution_price - prev_close_price) * close_position
            + (close_price - execution_price) * open_position
            + (close_price - prev_close_price) * kept_position
        ) * self.instrument.contract_multiplier

        profit = position_value_delta - commission - stop_commision

        cashflow = (
            (
                execution_price * (close_position - open_position)
                + stop_price.nan_to_num(posinf=0.0, neginf=0.0) * stop_position
            )
            * self.instrument.contract_multiplier
            - commission
            - stop_commision
        )

        return profit, new_position, execution_price, cashflow

    def calculate_commission(
        self, action_abs: torch.Tensor, price: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the commission for an action.

        Parameters
        ----------
        action_abs : torch.Tensor
            Absolute value of the action.
        price : torch.Tensor
            Price of the instrument.

        Returns
        -------
        commission : torch.Tensor
            Commission for the action.
        """
        commission = (action_abs * self.instrument.commission).clamp(
            min=self.instrument.min_commission
        )
        max_commission = price * action_abs * self.instrument.max_commission_pct
        commission = torch.where(
            self._use_max_commission_mask,
            commission.clamp(max=max_commission),
            commission,
        )

        return commission

    def _get_prev_close_price(
        self, date_idx: torch.Tensor, time_idx: torch.Tensor
    ) -> torch.Tensor:
        flat_idx = date_idx * self.data.shape[1] + time_idx
        return torch.where(
            flat_idx > 0,
            self.data_flat[flat_idx - 1, self.CHANNELS["close"]],
            self.data_flat[flat_idx, self.CHANNELS["open"]],
        )


def _calculate_positions(position: torch.Tensor, action: torch.Tensor) -> tuple:
    action_sign = torch.sign(action)
    action_abs = torch.abs(action)
    position_sign = torch.sign(position)
    position_abs = torch.abs(position)

    close_mask = (action_sign != position_sign) & (action != 0) & (position != 0)
    switch_mask = close_mask & (action_abs > position_abs)
    not_switch_mask = ~switch_mask
    open_mask = (
        (action_sign == 1) & (position_sign >= 0)
        | (action_sign == -1) & (position_sign <= 0)
        | switch_mask
    )

    new_position = position + action
    close_position = (position - new_position * not_switch_mask) * close_mask
    open_position = (new_position - position * not_switch_mask) * open_mask
    kept_position = not_switch_mask * (
        torch.min(position_abs, torch.abs(new_position)) * position_sign
    )

    return (
        action_sign,
        action_abs,
        new_position,
        close_position,
        open_position,
        kept_position,
    )
