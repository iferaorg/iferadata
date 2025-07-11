"""Environment module for single instrument trading."""

from __future__ import annotations

from typing import Optional

import torch

from .config import BaseInstrumentConfig
from .data_models import DataManager, InstrumentData
from .market_simulator import MarketSimulatorIntraday
from .policies import TradingPolicy


class SingleMarketEnv:
    """Environment for simulating a single market using :class:`MarketSimulatorIntraday`."""

    def __init__(
        self,
        instrument_config: BaseInstrumentConfig,
        broker_name: str,
        *,
        backadjust: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        dm = DataManager()
        self.instrument_data: InstrumentData = dm.get_instrument_data(
            instrument_config, dtype=dtype, device=device, backadjust=backadjust
        )
        self.market_simulator = MarketSimulatorIntraday(
            instrument_data=self.instrument_data, broker_name=broker_name
        )
        self.device = self.instrument_data.device
        self.dtype = self.instrument_data.dtype

        # State tensors populated in reset
        self.date_idx = torch.tensor((), dtype=torch.int32, device=self.device)
        self.time_idx = torch.tensor((), dtype=torch.int32, device=self.device)
        self.position = torch.tensor((), dtype=torch.int32, device=self.device)
        self.prev_stop_loss = torch.tensor((), dtype=self.dtype, device=self.device)
        self.entry_price = torch.tensor((), dtype=self.dtype, device=self.device)
        self.total_profit = torch.tensor((), dtype=self.dtype, device=self.device)
        self.done = torch.tensor((), dtype=torch.bool, device=self.device)

    def reset(
        self,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        trading_policy: Optional[TradingPolicy] = None,
    ) -> None:
        """Reset the environment state for a new simulation."""

        batch_size = start_date_idx.shape[0]
        self.date_idx = start_date_idx.to(torch.int32).to(self.device)
        self.time_idx = start_time_idx.to(torch.int32).to(self.device)
        self.position = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.prev_stop_loss = torch.full(
            (batch_size,), float("nan"), dtype=self.dtype, device=self.device
        )
        self.entry_price = torch.full(
            (batch_size,), float("nan"), dtype=self.dtype, device=self.device
        )
        self.total_profit = torch.zeros(
            batch_size, dtype=self.dtype, device=self.device
        )
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if trading_policy is not None and hasattr(
            trading_policy.trading_done_policy, "reset"
        ):
            trading_policy.trading_done_policy.reset(
                torch.ones(batch_size, dtype=torch.bool, device=self.device)
            )

    @torch.compile(mode="max-autotune")
    def step(self, trading_policy: TradingPolicy) -> torch.Tensor:
        """Run one simulation step using ``trading_policy``."""

        action, stop_loss, done = trading_policy(
            self.date_idx,
            self.time_idx,
            self.position,
            self.prev_stop_loss,
            self.entry_price,
        )

        next_date_idx, next_time_idx = self.instrument_data.get_next_indices(
            self.date_idx, self.time_idx
        )
        next_date_idx = torch.where(done, self.date_idx, next_date_idx)
        next_time_idx = torch.where(done, self.time_idx, next_time_idx)
        profit, new_position, execution_price, _ = self.market_simulator.calculate_step(
            next_date_idx, next_time_idx, self.position, action, stop_loss
        )
        profit = torch.where(done, torch.zeros_like(profit), profit)
        new_position = torch.where(done, self.position, new_position)
        execution_price = torch.where(done, self.entry_price, execution_price)

        entry_mask = (self.position == 0) & (new_position != 0)
        self.entry_price = torch.where(entry_mask, execution_price, self.entry_price)

        self.date_idx = next_date_idx
        self.time_idx = next_time_idx
        self.position = new_position
        self.prev_stop_loss = stop_loss
        self.total_profit += profit
        self.done |= done

        return profit

    def rollout(
        self,
        trading_policy: TradingPolicy,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Execute a rollout until ``done`` for all batches or ``max_steps`` reached."""

        self.reset(start_date_idx, start_time_idx, trading_policy)
        steps = 0
        while True:
            torch.compiler.cudagraph_mark_step_begin()
            self.step(trading_policy)
            steps += 1
            if self.done.all() or (max_steps is not None and steps >= max_steps):
                break
        return self.total_profit.clone()
