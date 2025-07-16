"""Environment module for single instrument trading."""

from __future__ import annotations

from typing import Optional

import torch
import datetime as dt
from rich.live import Live
from rich.table import Table

from .config import BaseInstrumentConfig
from .data_models import DataManager, InstrumentData
from .market_simulator import MarketSimulatorIntraday
from .policies import TradingPolicy


def make_table(
    data_tensor, date_idx_t, time_idx_t, maintenance_policy, stop_loss_t, total_profit_t, entry_price, steps, contract_multiplier
):
    cost_basis_t = entry_price.nan_to_num(1) * contract_multiplier
    profit_perc_t = (total_profit_t / cost_basis_t) * 100
    idx = profit_perc_t.argmax()
    date_idx = date_idx_t[idx].item()
    time_idx = time_idx_t[idx].item()
    stop_loss = stop_loss_t[idx].nan_to_num().item()
    profit_perc = profit_perc_t[idx].item()
    ord_date = data_tensor[date_idx, time_idx, 2].to(torch.int64).item()
    time_seconds = data_tensor[date_idx, time_idx, 3].to(torch.int64).item()
    stage_str = (
        maintenance_policy.derived_configs[maintenance_policy.stage[idx]].interval
        if maintenance_policy.stage.shape[0] > 0
        else "N/A"
    )
    current_high = data_tensor[date_idx, time_idx, 5].item()
    current_low = data_tensor[date_idx, time_idx, 6].item()
    total_profit = total_profit_t[idx].item()
    date_str = dt.date.fromordinal(ord_date).strftime("%Y-%m-%d")
    time_str = f"{time_seconds // 3600:02}:{(time_seconds % 3600) // 60:02}:{time_seconds % 60:02}"
    ord_date = f"{date_str} {time_str}"
    table = Table(title=f"Simulation Status for batch ID {idx} on {ord_date}", show_lines=False)

    table.add_column("Steps", justify="right", style="white")
    table.add_column("Stage", justify="right", style="cyan")
    table.add_column("Current High", justify="right", style="green")
    table.add_column("Current Low", justify="right", style="yellow")
    table.add_column("Stop Loss", justify="right", style="yellow")
    table.add_column("Profit", justify="right", style="magenta")
    table.add_column("Profit %", justify="right", style="magenta")

    table.add_row(
        f"{steps}",
        stage_str,
        f"{current_high:.2f}",
        f"{current_low:.2f}",
        f"{stop_loss:.2f}",
        f"{total_profit:.2f}",
        f"{profit_perc:.2f}%",
    )

    return table




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

    #@torch.compile(mode="max-autotune")
    def step(self, trading_policy: TradingPolicy):
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
        entry_price = torch.where(entry_mask, execution_price, self.entry_price)

        return (
            profit,
            new_position,
            next_date_idx,
            next_time_idx,
            entry_price,
            stop_loss,
            done,
        )

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
            (
                profit,
                new_position,
                next_date_idx,
                next_time_idx,
                entry_price,
                stop_loss,
                done,
            ) = self.step(trading_policy)

            self.position = new_position.clone()
            self.date_idx = next_date_idx.clone()
            self.time_idx = next_time_idx.clone()
            self.entry_price = entry_price.clone()
            self.prev_stop_loss = stop_loss.clone()
            self.total_profit = self.total_profit + profit
            self.done = self.done | done
            steps += 1

            if self.done.all() or (max_steps is not None and steps >= max_steps):
                break
        return self.total_profit.clone()

    def rollout_with_display(
        self,
        trading_policy: TradingPolicy,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Execute a rollout until ``done`` for all batches or ``max_steps`` reached."""

        self.reset(start_date_idx, start_time_idx, trading_policy)
        contract_multiplier = self.instrument_data.instrument.contract_multiplier
        steps = 0
        with Live(
            make_table(
                self.instrument_data.data_full,
                self.date_idx,
                self.time_idx,
                trading_policy.position_maintenance_policy,
                self.prev_stop_loss,
                self.total_profit,
                self.entry_price,
                steps,
                contract_multiplier
            ),
            refresh_per_second=4,
        ) as live:
            while True:
                #torch.compiler.cudagraph_mark_step_begin()
                (
                    profit,
                    new_position,
                    next_date_idx,
                    next_time_idx,
                    entry_price,
                    stop_loss,
                    done,
                ) = self.step(trading_policy)

                self.position = new_position.clone()
                self.date_idx = next_date_idx.clone()
                self.time_idx = next_time_idx.clone()
                self.entry_price = entry_price.clone()
                self.prev_stop_loss = stop_loss.clone()
                self.total_profit = self.total_profit + profit
                self.done = self.done | done
                steps += 1

                live.update(
                    make_table(
                        self.instrument_data.data_full,
                        self.date_idx,
                        self.time_idx,
                        trading_policy.position_maintenance_policy,
                        self.prev_stop_loss,
                        self.total_profit,
                        self.entry_price,
                        steps,
                        contract_multiplier
                    )
                )

                if self.done.all() or (max_steps is not None and steps >= max_steps):
                    break
        return self.total_profit.clone()
