"""Environment module for single instrument trading."""

from __future__ import annotations

from typing import Optional

import datetime as dt
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
import torch
from torch.compiler import nested_compile_region
from torch.compiler import nested_compile_region
from rich.live import Live
from rich.table import Table

from .config import BaseInstrumentConfig
from .data_models import DataManager, InstrumentData
from .market_simulator import MarketSimulatorIntraday
from .policies import TradingPolicy
from .torch_utils import get_devices


def make_table(
    data_tensor,
    date_idx_t,
    time_idx_t,
    maintenance_policy,
    stop_loss_t,
    total_profit_t,
    entry_price,
    steps,
    contract_multiplier,
):
    """Create a ``rich`` table summarizing the current simulation state."""
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
    table = Table(
        title=f"Simulation Status for batch ID {idx} on {ord_date}", show_lines=False
    )

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
        self.max_date_idx = self.instrument_data.data.shape[0] - 1

        # State tensors populated in reset
        self.state = {
            "date_idx": torch.tensor((), dtype=torch.int32, device=self.device),
            "time_idx": torch.tensor((), dtype=torch.int32, device=self.device),
            "position": torch.tensor((), dtype=torch.int32, device=self.device),
            "prev_stop_loss": torch.tensor((), dtype=self.dtype, device=self.device),
            "entry_price": torch.tensor((), dtype=self.dtype, device=self.device),
            "total_profit": torch.tensor((), dtype=self.dtype, device=self.device),
            "done": torch.tensor((), dtype=torch.bool, device=self.device),
        }

    def reset(
        self,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        trading_policy: Optional[TradingPolicy] = None,
    ) -> None:
        """Reset the environment state for a new simulation."""

        batch_size = start_date_idx.shape[0]
        self.state = {
            "date_idx": start_date_idx.to(torch.int32).to(self.device),
            "time_idx": start_time_idx.to(torch.int32).to(self.device),
            "position": torch.zeros(batch_size, dtype=torch.int32, device=self.device),
            "prev_stop_loss": torch.full(
                (batch_size,), float("nan"), dtype=self.dtype, device=self.device
            ),
            "entry_price": torch.full(
                (batch_size,), float("nan"), dtype=self.dtype, device=self.device
            ),
            "total_profit": torch.zeros(
                batch_size, dtype=self.dtype, device=self.device
            ),
            "done": torch.zeros(batch_size, dtype=torch.bool, device=self.device),
        }
        if trading_policy is not None:
            trading_policy.reset(self.state)

    @nested_compile_region
    @nested_compile_region
    @torch.compile(mode="max-autotune", fullgraph=True)
    def step(self, trading_policy: TradingPolicy, state: dict[str, torch.Tensor]):
        """Run one simulation step using ``trading_policy``."""
        date_idx = state["date_idx"]
        time_idx = state["time_idx"]
        position = state["position"]
        entry_price = state["entry_price"]

        action, stop_loss, done = trading_policy(state)

        next_date_idx, next_time_idx = self.instrument_data.get_next_indices(
            date_idx, time_idx
        )
        done = done | (next_date_idx > self.max_date_idx)
        next_date_idx = torch.where(done, date_idx, next_date_idx)
        next_time_idx = torch.where(done, time_idx, next_time_idx)
        profit, new_position, execution_price, _ = self.market_simulator.calculate_step(
            next_date_idx, next_time_idx, position, action, stop_loss
        )
        profit = torch.where(done, torch.zeros_like(profit), profit)
        new_position = torch.where(done, position, new_position)
        execution_price = torch.where(done, entry_price, execution_price)

        entry_mask = (position == 0) & (action != 0)
        entry_price = torch.where(entry_mask, execution_price, entry_price)

        had_position = state.get("had_position")
        if had_position is not None:
            had_position = had_position | (action != 0)

        result = {
            "profit": profit,
            "position": new_position,
            "date_idx": next_date_idx,
            "time_idx": next_time_idx,
            "entry_price": entry_price,
            "prev_stop_loss": stop_loss,
            "done": done,
        }
        if had_position is not None:
            result["had_position"] = had_position

        return result

    def rollout(
        self,
        trading_policy: TradingPolicy,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Execute a rollout until ``done`` for all batches or ``max_steps`` reached.

        Returns a tuple containing the total profit, the ``date_idx`` and
        ``time_idx`` at which ``done`` first became ``True`` for each batch, and
        the number of steps taken. If an episode never reaches ``done`` these
        indices will be ``nan``.
        Returns a tuple containing the total profit, the ``date_idx`` and
        ``time_idx`` at which ``done`` first became ``True`` for each batch, and
        the number of steps taken. If an episode never reaches ``done`` these
        indices will be ``nan``.
        """

        self.reset(start_date_idx, start_time_idx, trading_policy)
        steps = 0
        done_date_idx = torch.full(
            self.state["date_idx"].shape,
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )
        done_time_idx = torch.full(
            self.state["time_idx"].shape,
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )

        while True:
            torch.compiler.cudagraph_mark_step_begin()
            step_state = self.step(trading_policy, self.state)

            newly_done = (~self.state["done"]) & step_state["done"]
            done_date_idx = torch.where(
                newly_done,
                self.state["date_idx"].to(self.dtype),
                done_date_idx,
            )
            done_time_idx = torch.where(
                newly_done,
                self.state["time_idx"].to(self.dtype),
                done_time_idx,
            )

            for key in step_state:
                if key != "done" and key in self.state:
                    self.state[key] = step_state[key].clone()

            for key in self.state:
                if key not in step_state:
                    self.state[key] = self.state[key].clone()

            self.state["total_profit"] = (
                self.state["total_profit"] + step_state["profit"]
            )
            self.state["done"] = self.state["done"] | step_state["done"]
            steps += 1

            if self.state["done"].all() or (
                max_steps is not None and steps >= max_steps
            ):
                break

        return (
            self.state["total_profit"].clone(),
            done_date_idx.clone(),
            done_time_idx.clone(),
            steps,
            steps,
        )

    def rollout_with_display(
        self,
        trading_policy: TradingPolicy,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute a rollout until ``done`` for all batches or ``max_steps`` reached.

        Returns a tuple containing the total profit, and the ``date_idx`` and
        ``time_idx`` at which ``done`` first became ``True`` for each batch. If an
        episode never reaches ``done`` these indices will be ``nan``.
        """

        self.reset(start_date_idx, start_time_idx, trading_policy)
        contract_multiplier = self.instrument_data.instrument.contract_multiplier
        steps = 0
        done_date_idx = torch.full(
            self.state["date_idx"].shape,
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )
        done_time_idx = torch.full(
            self.state["time_idx"].shape,
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )
        with Live(
            make_table(
                self.instrument_data.data_full,
                self.state["date_idx"],
                self.state["time_idx"],
                trading_policy.position_maintenance_policy,
                self.state["prev_stop_loss"],
                self.state["total_profit"],
                self.state["entry_price"],
                steps,
                contract_multiplier,
            ),
            refresh_per_second=4,
        ) as live:
            while True:
                # torch.compiler.cudagraph_mark_step_begin()
                step_state = self.step(trading_policy, self.state)

                newly_done = (~self.state["done"]) & step_state["done"]
                done_date_idx = torch.where(
                    newly_done,
                    self.state["date_idx"].to(self.dtype),
                    done_date_idx,
                )
                done_time_idx = torch.where(
                    newly_done,
                    self.state["time_idx"].to(self.dtype),
                    done_time_idx,
                )

                for key in step_state:
                    if key != "done":
                        self.state[key] = step_state[key].clone()

                self.state["total_profit"] = (
                    self.state["total_profit"] + step_state["profit"]
                )
                self.state["done"] = self.state["done"] | step_state["done"]
                steps += 1

                live.update(
                    make_table(
                        self.instrument_data.data_full,
                        self.state["date_idx"],
                        self.state["time_idx"],
                        trading_policy.position_maintenance_policy,
                        self.state["prev_stop_loss"],
                        self.state["total_profit"],
                        self.state["entry_price"],
                        steps,
                        contract_multiplier,
                    )
                )

                if self.state["done"].all() or (
                    max_steps is not None and steps >= max_steps
                ):
                    break
        return (
            self.state["total_profit"].clone(),
            done_date_idx.clone(),
            done_time_idx.clone(),
        )


def _run_rollout_worker(
    instrument_config: BaseInstrumentConfig,
    broker_name: str,
    backadjust: bool,
    device: torch.device,
    dtype: torch.dtype,
    start_date_idx_chunk: torch.Tensor,
    start_time_idx_chunk: torch.Tensor,
    trading_policy: TradingPolicy,
    max_steps: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Worker function to run a full rollout on a single device."""
    trading_policy.to(device)  # Move policy to the worker's device

    env = SingleMarketEnv(
        instrument_config,
        broker_name,
        backadjust=backadjust,
        device=device,
        dtype=dtype,
    )
    total_profit, done_date_idx, done_time_idx, steps = env.rollout(
        trading_policy, start_date_idx_chunk, start_time_idx_chunk, max_steps
    )
    return (
        total_profit.cpu(),
        done_date_idx.cpu(),
        done_time_idx.cpu(),
        steps,
    )  # Move to CPU for pickling


def _run_rollout_worker(
    instrument_config: BaseInstrumentConfig,
    broker_name: str,
    backadjust: bool,
    device: torch.device,
    dtype: torch.dtype,
    start_date_idx_chunk: torch.Tensor,
    start_time_idx_chunk: torch.Tensor,
    trading_policy: TradingPolicy,
    max_steps: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Worker function to run a full rollout on a single device."""
    trading_policy.to(device)  # Move policy to the worker's device

    env = SingleMarketEnv(
        instrument_config,
        broker_name,
        backadjust=backadjust,
        device=device,
        dtype=dtype,
    )
    total_profit, done_date_idx, done_time_idx, steps = env.rollout(
        trading_policy, start_date_idx_chunk, start_time_idx_chunk, max_steps
    )
    return (
        total_profit.cpu(),
        done_date_idx.cpu(),
        done_time_idx.cpu(),
        steps,
    )  # Move to CPU for pickling


class MultiGPUSingleMarketEnv:
    """Run a :class:`SingleMarketEnv` on multiple devices by splitting the batch."""

    def __init__(
        self,
        instrument_config: BaseInstrumentConfig,
        broker_name: str,
        *,
        backadjust: bool = False,
        devices: Optional[list[torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        devices = get_devices(devices)

        self.envs = [
            SingleMarketEnv(
                instrument_config,
                broker_name,
                backadjust=backadjust,
                device=device,
                dtype=dtype,
            )
            for device in devices
        ]
        self.devices = devices
        self.instrument_config = instrument_config
        self.broker_name = broker_name
        self.backadjust = backadjust
        self.dtype = dtype
        self.instrument_config = instrument_config
        self.broker_name = broker_name
        self.backadjust = backadjust
        self.dtype = dtype

    def _chunk_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        batch_size = tensor.shape[0]
        per_device = (batch_size + len(self.envs) - 1) // len(self.envs)
        return [
            tensor[i * per_device : min((i + 1) * per_device, batch_size)]
            for i in range(len(self.envs))
        ]

    def reset(
        self,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        trading_policies: list[TradingPolicy],
    ) -> None:
        """Reset all underlying environments."""
        if len(trading_policies) != len(self.envs):
            raise ValueError("Mismatch between policies and devices")

        d_chunks = self._chunk_tensor(start_date_idx)
        t_chunks = self._chunk_tensor(start_time_idx)
        for env, d_chunk, t_chunk, policy in zip(
            self.envs, d_chunks, t_chunks, trading_policies
        ):
            env.reset(d_chunk.to(env.device), t_chunk.to(env.device), policy)

    def step(
        self, trading_policies: list[TradingPolicy]
    ) -> list[dict[str, torch.Tensor]]:
        """Execute one step for each environment."""
        step_states = []
        for env, policy in zip(self.envs, trading_policies):
            step_states.append(env.step(policy, env.state))
        return step_states

    def _rollout_inner(
    def _rollout_inner(
        self,
        trading_policies: list[TradingPolicy],
        done_date_idx: list[torch.Tensor],
        done_time_idx: list[torch.Tensor],
        done_date_idx: list[torch.Tensor],
        done_time_idx: list[torch.Tensor],
        max_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        steps = 0
        while True:
            torch.compiler.cudagraph_mark_step_begin()
            step_states = self.step(trading_policies)

            for i, (env, step_state) in enumerate(zip(self.envs, step_states)):
                newly_done = (~env.state["done"]) & step_state["done"]
                done_date_idx[i] = torch.where(
                    newly_done,
                    env.state["date_idx"].to(env.dtype),
                    done_date_idx[i],
                )
                done_time_idx[i] = torch.where(
                    newly_done,
                    env.state["time_idx"].to(env.dtype),
                    done_time_idx[i],
                )

                for key in step_state:
                    if key != "done" and key in env.state:
                        env.state[key] = step_state[key].clone()

                for key in env.state:
                    if key not in step_state:
                        env.state[key] = env.state[key].clone()

                env.state["total_profit"] = (
                    env.state["total_profit"] + step_state["profit"]
                )
                env.state["done"] = env.state["done"] | step_state["done"]

            steps += 1
            if all(env.state["done"].all() for env in self.envs) or (
                max_steps is not None and steps >= max_steps
            ):
                break

        return (
            torch.cat([env.state["total_profit"].clone() for env in self.envs]),
            torch.cat([idx.clone() for idx in done_date_idx]),
            torch.cat([idx.clone() for idx in done_time_idx]),
            steps,
            steps,
        )

    def rollout(
        self,
        trading_policy: TradingPolicy,
        trading_policy: TradingPolicy,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Run rollouts on all devices in parallel.
        """Run rollouts on all devices in parallel.

        Returns a tuple containing the total profit, the ``date_idx`` and
        ``time_idx`` at which ``done`` first became ``True`` for each batch, and
        the maximum number of steps taken across all workers. If an episode never
        reaches ``done`` these indices will be ``nan``.
        Returns a tuple containing the total profit, the ``date_idx`` and
        ``time_idx`` at which ``done`` first became ``True`` for each batch, and
        the maximum number of steps taken across all workers. If an episode never
        reaches ``done`` these indices will be ``nan``.
        """
        # Set multiprocessing start method to 'spawn' if any device is CUDA
        # This prevents CUDA re-initialization errors in forked subprocesses
        has_cuda_device = any(device.type == "cuda" for device in self.devices)
        if (
            has_cuda_device
            and multiprocessing.get_start_method(allow_none=True) != "spawn"
        ):
            multiprocessing.set_start_method("spawn", force=True)

        # Move trading policy to CPU to ensure clean pickling
        trading_policy = trading_policy.to(torch.device("cpu"))

        # Chunk the inputs
        d_chunks = self._chunk_tensor(start_date_idx)
        t_chunks = self._chunk_tensor(start_time_idx)

        # Submit parallel rollouts
        with ProcessPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = [
                executor.submit(
                    _run_rollout_worker,
                    self.instrument_config,
                    self.broker_name,
                    self.backadjust,
                    device,
                    self.dtype,
                    d_chunk,
                    t_chunk,
                    copy.deepcopy(trading_policy),  # Deep copy for each worker
                    max_steps,
                )
                for device, d_chunk, t_chunk in zip(self.devices, d_chunks, t_chunks)
            ]

            # Collect results in order to maintain chunk ordering
            results = [future.result() for future in futures]

        # Sort results by original order (if needed) and concatenate
        total_profits = [r[0] for r in results]
        done_date_idxs = [r[1] for r in results]
        done_time_idxs = [r[2] for r in results]
        steps_list = [r[3] for r in results]

        # Concatenate tensors
        total_profit = torch.cat(total_profits)
        done_date_idx = torch.cat(done_date_idxs)
        done_time_idx = torch.cat(done_time_idxs)
        max_steps = max(steps_list)

        return total_profit, done_date_idx, done_time_idx, max_steps
