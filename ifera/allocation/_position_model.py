"""Daily equity ledgers with fixed, net-liquidation-based risk per new trade.

The linear valuation convention spreads each observed terminal return across
its holding sessions. These are synthetic marks, not observed market prices.
Equity and entry budgets are updated independently: marks change equity, while
the original risk reserved for a trade is released only when that trade closes.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
import math

import polars as pl
import torch


@dataclass(frozen=True, slots=True)
class PositionTrade:
    """One trade, with blockers identified by indices in its model's trades."""

    strategy_index: int
    opened: datetime
    closed: datetime
    return_on_risk: float
    blockers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PositionSimulation:
    """Continuous simulated daily equity returns and the underlying entry audit."""

    daily_returns: torch.Tensor
    active: torch.Tensor
    trade_results: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _TradeTiming:
    """Precomputed session indices and a trade's unscaled synthetic marks."""

    opening_day: int
    closing_day: int | None
    final_day: int
    holding_sessions: int
    daily_mark: float
    closing_adjustment: float
    active_blockers: tuple[int, ...]


@dataclass(slots=True)
class _Ledger:
    """Candidate-batched state; storage grows with trades plus sessions."""

    equity: torch.Tensor
    realized_equity: torch.Tensor
    reserved: torch.Tensor
    open_count: torch.Tensor
    risks: torch.Tensor
    alive: torch.Tensor
    daily_returns: torch.Tensor
    active: torch.Tensor
    entry_equities: torch.Tensor | None
    entry_allocations: torch.Tensor | None
    statuses: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class PositionModel:
    """Trade-level data and a reusable, allocation-dependent daily simulation.

    ``dates`` contains every market session in the modeled interval. Entries
    use start-of-day equity, including earlier synthetic unrealized gains and
    losses. The budget is checked at entry, without inventing forced exits when
    marks subsequently reduce equity. Each accepted position retains its entry
    risk until its recorded close. A rejected entry is skipped in full.

    Workspace is O(batch * (trade_count + session_count)), not their product.
    ``candidate_batch_size`` gives a conservative ledger-only memory bound.
    """

    dates: tuple[date, ...]
    strategy_names: tuple[str, ...]
    trades: tuple[PositionTrade, ...]
    valuation_method: str = "linear"
    _holding_sessions: tuple[int, ...] = field(default=(), repr=False)
    _timings: tuple[_TradeTiming, ...] = field(init=False, repr=False)
    _events: tuple[tuple[tuple[bool, int], ...], ...] = field(init=False, repr=False)
    _open_marks: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...] = field(
        init=False, repr=False
    )
    _cuda_evaluators: dict[
        tuple[torch.device, torch.dtype],
        Callable[[torch.Tensor, float], torch.Tensor] | None,
    ] = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Validate identities and prepare reusable day/event schedules."""
        if self.valuation_method not in ("linear", "close_only"):
            raise ValueError("valuation_method must be 'linear' or 'close_only'")
        if tuple(sorted(set(self.dates))) != self.dates:
            raise ValueError("PositionModel dates must be unique and increasing")
        if not self.strategy_names or len(set(self.strategy_names)) != len(
            self.strategy_names
        ):
            raise ValueError("PositionModel strategy names must be nonempty and unique")
        if self._holding_sessions and len(self._holding_sessions) != len(self.trades):
            raise ValueError("Holding session counts must match the number of trades")

        timings = tuple(self._trade_timing(index) for index in range(len(self.trades)))
        object.__setattr__(self, "_timings", timings)
        object.__setattr__(
            self, "_holding_sessions", tuple(item.holding_sessions for item in timings)
        )
        object.__setattr__(self, "_events", self._build_events())
        object.__setattr__(self, "_open_marks", self._build_open_marks())

    def _trade_timing(self, index: int) -> _TradeTiming:
        """Validate one trade and retain its original full holding duration."""
        trade = self.trades[index]
        _validate_trade(trade, index, len(self.strategy_names))
        opening_day = bisect_left(self.dates, trade.opened.date())
        closing_day = bisect_left(self.dates, trade.closed.date())
        if (
            opening_day == len(self.dates)
            or self.dates[opening_day] != trade.opened.date()
        ):
            raise ValueError(f"Trade {index} opens outside the model's market sessions")
        full_close = closing_day < len(self.dates)
        if full_close and self.dates[closing_day] != trade.closed.date():
            raise ValueError(
                f"Trade {index} closes outside the model's market sessions"
            )
        if not full_close and not self._holding_sessions:
            raise ValueError(f"Trade {index} closes after the model's final session")
        sessions = (
            self._holding_sessions[index]
            if self._holding_sessions
            else closing_day - opening_day + 1
        )
        visible_sessions = min(closing_day + 1, len(self.dates)) - opening_day
        if sessions < visible_sessions or sessions < 1:
            raise ValueError(
                "Original holding duration cannot be shorter than visible sessions"
            )
        daily_mark = (
            trade.return_on_risk / sessions
            if self.valuation_method == "linear"
            else 0.0
        )
        return _TradeTiming(
            opening_day=opening_day,
            closing_day=closing_day if full_close else None,
            final_day=min(closing_day, len(self.dates) - 1),
            holding_sessions=sessions,
            daily_mark=daily_mark,
            closing_adjustment=trade.return_on_risk - daily_mark * sessions,
            active_blockers=tuple(
                blocker
                for blocker in trade.blockers
                if self.trades[blocker].opened
                <= trade.opened
                < self.trades[blocker].closed
            ),
        )

    def _build_events(self) -> tuple[tuple[tuple[bool, int], ...], ...]:
        """Order closes before simultaneous entries; instant trades close on entry."""
        events: list[list[tuple[datetime, int, int, int]]] = [[] for _ in self.dates]
        for index, (trade, timing) in enumerate(zip(self.trades, self._timings)):
            events[timing.opening_day].append(
                (trade.opened, 1, trade.strategy_index, index)
            )
            if timing.closing_day is not None and trade.closed != trade.opened:
                events[timing.closing_day].append(
                    (trade.closed, 0, trade.strategy_index, index)
                )
        return tuple(
            tuple((bool(kind), index) for _, kind, _, index in sorted(day_events))
            for day_events in events
        )

    def _build_open_marks(
        self,
    ) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
        """Precompute cumulative marks for positions still open at each day-end."""
        if self.valuation_method == "close_only":
            return tuple(((), ()) for _ in self.dates)
        live: set[int] = set()
        schedule: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
        for day, events in enumerate(self._events):
            for is_opening, index in events:
                if (
                    is_opening
                    and self.trades[index].closed != self.trades[index].opened
                ):
                    live.add(index)
                else:
                    live.discard(index)
            indices = tuple(sorted(live))
            marks = tuple(
                self.trades[index].return_on_risk
                * (day - self._timings[index].opening_day + 1)
                / self._timings[index].holding_sessions
                for index in indices
            )
            schedule.append((indices, marks))
        return tuple(schedule)

    def returns_tensor(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return a unit-original-risk daily proxy for correlation and activity.

        Self-overlapping trades contribute additively. This allocation-independent
        proxy is not the portfolio return used to score allocation candidates.
        """
        if not dtype.is_floating_point:
            raise ValueError("Position returns dtype must be floating point")
        shape = (len(self.dates), len(self.strategy_names))
        returns = torch.zeros(shape, device=device, dtype=dtype)
        active = torch.zeros(shape, device=device, dtype=torch.bool)
        for trade, timing in zip(self.trades, self._timings):
            period = slice(timing.opening_day, timing.final_day + 1)
            active[period, trade.strategy_index] = True
            returns[period, trade.strategy_index] += timing.daily_mark
            if timing.closing_day is not None:
                returns[
                    timing.closing_day, trade.strategy_index
                ] += timing.closing_adjustment
        return returns.masked_fill(~active, torch.nan)

    def window(
        self, start: date, stop: date, closed_only: bool = True
    ) -> PositionModel:
        """Select entries in a half-open window, optionally purging unclosed trades.

        Training defaults to fully contained trades. Simulation windows retain
        original mark denominators when a recorded close is beyond ``stop``.
        Positions opened before ``start`` are not imported into a new account.
        """
        if stop <= start:
            raise ValueError("Position window stop must be after start")
        dates = self.dates[
            bisect_left(self.dates, start) : bisect_left(self.dates, stop)
        ]
        indices = tuple(
            index
            for index, trade in enumerate(self.trades)
            if start <= trade.opened.date() < stop
            and (not closed_only or trade.closed.date() < stop)
        )
        return self._selected_model(
            indices, dates, tuple(range(len(self.strategy_names)))
        )

    def subset(self, support: tuple[int, ...]) -> PositionModel:
        """Retain a strategy support and remap both strategy and trade identities."""
        if (
            not support
            or len(set(support)) != len(support)
            or any(index < 0 or index >= len(self.strategy_names) for index in support)
        ):
            raise ValueError(
                "Position subset must contain unique valid strategy indices"
            )
        selected = set(support)
        indices = tuple(
            index
            for index, trade in enumerate(self.trades)
            if trade.strategy_index in selected
        )
        return self._selected_model(indices, self.dates, support)

    def _selected_model(
        self,
        indices: tuple[int, ...],
        dates: tuple[date, ...],
        support: tuple[int, ...],
    ) -> PositionModel:
        """Rebuild a filtered model without shortening any trade's mark path."""
        trade_map = {old: new for new, old in enumerate(indices)}
        strategy_map = {old: new for new, old in enumerate(support)}
        trades = tuple(
            PositionTrade(
                strategy_index=strategy_map[self.trades[index].strategy_index],
                opened=self.trades[index].opened,
                closed=self.trades[index].closed,
                return_on_risk=self.trades[index].return_on_risk,
                blockers=tuple(
                    trade_map[blocker]
                    for blocker in self.trades[index].blockers
                    if blocker in trade_map
                ),
            )
            for index in indices
        )
        return PositionModel(
            dates=dates,
            strategy_names=tuple(self.strategy_names[index] for index in support),
            trades=trades,
            valuation_method=self.valuation_method,
            _holding_sessions=tuple(self._holding_sessions[index] for index in indices),
        )

    def candidate_batch_size(
        self,
        memory_budget_bytes: int = 64 * 1024 * 1024,
        dtype: torch.dtype = torch.float32,
    ) -> int:
        """Bound ledger workspace only; bootstrap/scoring buffers need separate space."""
        item_size = torch.empty((), dtype=dtype).element_size()
        max_open = max((len(indices) for indices, _ in self._open_marks), default=0)
        bytes_per_candidate = (
            (len(self.trades) + len(self.dates) + 2 * max_open + 32) * item_size
            + len(self.dates)
            + 16
        )
        return max(1, memory_budget_bytes // bytes_per_candidate)

    @torch.no_grad()
    def evaluate(
        self, allocations: torch.Tensor, max_total_allocation: float
    ) -> torch.Tensor:
        """Evaluate constant candidate allocations shaped ``[candidate, strategy]``."""
        self._validate_allocations(allocations, max_total_allocation)
        if allocations.is_cuda:
            # Prepare the static event schedule once, including across refinements.
            # Import lazily so CPU-only installations do not require Triton.
            from ._position_cuda import (  # pylint: disable=import-outside-toplevel
                prepare_position_evaluator,
            )

            key = (allocations.device, allocations.dtype)
            if key not in self._cuda_evaluators:
                self._cuda_evaluators[key] = prepare_position_evaluator(self, *key)
            evaluator = self._cuda_evaluators[key]
            if evaluator is not None:
                return evaluator(allocations, max_total_allocation)
        ledger = self._run_ledger(allocations.unsqueeze(1), max_total_allocation, False)
        return ledger.daily_returns

    @torch.no_grad()
    def simulate(
        self, allocations: torch.Tensor, max_total_allocation: float
    ) -> PositionSimulation:
        """Apply one ``[day, strategy]`` schedule, retaining old positions across changes."""
        self._validate_allocations(allocations, max_total_allocation)
        if allocations.shape[0] != len(self.dates):
            raise ValueError("Simulation allocations must have one row per model day")
        ledger = self._run_ledger(allocations.unsqueeze(0), max_total_allocation, True)
        if bool(
            torch.any(
                ~torch.isfinite(ledger.daily_returns) | (ledger.daily_returns <= -1.0)
            ).item()
        ):
            raise ValueError(
                "Position simulation reached nonpositive or nonfinite account equity"
            )
        return PositionSimulation(
            daily_returns=ledger.daily_returns[0],
            active=ledger.active[0],
            trade_results=self._trade_results(ledger),
        )

    def _validate_allocations(
        self, allocations: torch.Tensor, max_total_allocation: float
    ) -> None:
        """Validate external allocation and budget inputs before running the ledger."""
        if allocations.ndim != 2 or allocations.shape[1] != len(self.strategy_names):
            raise ValueError(
                "Position allocations must have shape [batch/day, strategy]"
            )
        if not allocations.is_floating_point():
            raise ValueError("Position allocations must have a floating-point dtype")
        if bool(torch.any(~torch.isfinite(allocations) | (allocations < 0)).item()):
            raise ValueError("Position allocations must be finite and nonnegative")
        if not math.isfinite(max_total_allocation) or max_total_allocation < 0:
            raise ValueError("max_total_allocation must be finite and nonnegative")

    def _run_ledger(
        self, allocations: torch.Tensor, cap: float, record_entries: bool
    ) -> _Ledger:
        """Advance through sessions and timestamp-ordered events for all candidates."""
        ledger = _new_ledger(
            allocations, len(self.dates), len(self.trades), record_entries
        )
        scheduled = allocations.shape[1] != 1
        for day, events in enumerate(self._events):
            ledger.active[:, day] = ledger.open_count > 0
            weights = allocations[:, day if scheduled else 0, :]
            for is_opening, index in events:
                if is_opening:
                    self._open_trade(ledger, weights, cap, day, index)
                    if self.trades[index].opened == self.trades[index].closed:
                        self._close_trade(ledger, index)
                else:
                    self._close_trade(ledger, index)
            indices, marks = self._open_marks[day]
            next_equity = ledger.realized_equity
            if indices:
                mark_values = allocations.new_tensor(marks)
                marked_pnl = (ledger.risks[:, list(indices)] * mark_values).sum(dim=1)
                next_equity = next_equity + marked_pnl
            _finish_day(ledger, next_equity, day)
        return ledger

    def _open_trade(
        self,
        ledger: _Ledger,
        weights: torch.Tensor,
        cap: float,
        day: int,
        index: int,
    ) -> None:
        """Admit a complete new trade, or preserve a specific rejection reason."""
        trade, timing = self.trades[index], self._timings[index]
        weight = weights[:, trade.strategy_index]
        requested_risk = weight * ledger.equity
        blocked = torch.zeros_like(ledger.alive)
        if timing.active_blockers:
            blocked = (ledger.risks[:, list(timing.active_blockers)] > 0).any(dim=1)
        budget = ledger.equity * cap
        tolerance = 8 * torch.finfo(weights.dtype).eps * budget.abs()
        fits = requested_risk + ledger.reserved <= budget + tolerance
        accepted = (weight > 0) & ledger.alive & ~blocked & fits
        risk = torch.where(accepted, requested_risk, 0.0)
        ledger.risks[:, index] = risk
        ledger.reserved += risk
        ledger.open_count += accepted.to(dtype=torch.int32)
        ledger.active[:, day] |= accepted
        if ledger.entry_equities is not None:
            ledger.entry_equities[:, index] = ledger.equity
        if ledger.entry_allocations is not None:
            ledger.entry_allocations[:, index] = weight
        if ledger.statuses is not None:
            ledger.statuses[:, index] = torch.where(
                weight <= 0,
                0,
                torch.where(blocked, 1, torch.where(accepted, 3, 2)),
            )

    def _close_trade(self, ledger: _Ledger, index: int) -> None:
        """Realize the final mark and release the original entry-risk reservation."""
        risk = ledger.risks[:, index]
        ledger.reserved.sub_(risk).clamp_(min=0)
        ledger.open_count -= (risk > 0).to(dtype=torch.int32)
        ledger.realized_equity += risk * self.trades[index].return_on_risk
        flat = ledger.open_count == 0
        ledger.reserved.masked_fill_(flat, 0)

    def _trade_results(self, ledger: _Ledger) -> pl.DataFrame:
        """Build one simulation's entry and marked/realized P/L audit table."""
        if (
            ledger.entry_allocations is None
            or ledger.entry_equities is None
            or ledger.statuses is None
        ):
            raise RuntimeError("Trade audit requires recorded entry details")
        risks = ledger.risks[0].cpu().tolist()
        statuses = ("zero_allocation", "blocked", "capital", "executed")
        marked_returns = [
            timing.daily_mark * (timing.final_day - timing.opening_day + 1)
            + (timing.closing_adjustment if timing.closing_day is not None else 0.0)
            for timing in self._timings
        ]
        return pl.DataFrame(
            {
                "trade_index": list(range(len(self.trades))),
                "strategy_index": [trade.strategy_index for trade in self.trades],
                "strategy": [
                    self.strategy_names[trade.strategy_index] for trade in self.trades
                ],
                "opened": [trade.opened for trade in self.trades],
                "closed": [trade.closed for trade in self.trades],
                "allocation": ledger.entry_allocations[0].cpu().tolist(),
                "entry_equity": ledger.entry_equities[0].cpu().tolist(),
                "risk": risks,
                "pnl": [risk * mark for risk, mark in zip(risks, marked_returns)],
                "realized_pnl": [
                    (
                        risk * trade.return_on_risk
                        if timing.closing_day is not None
                        else 0.0
                    )
                    for risk, trade, timing in zip(risks, self.trades, self._timings)
                ],
                "status": [
                    statuses[index] for index in ledger.statuses[0].cpu().tolist()
                ],
            },
            schema_overrides={
                "trade_index": pl.Int64,
                "strategy_index": pl.Int64,
                "strategy": pl.String,
                "opened": pl.Datetime("us"),
                "closed": pl.Datetime("us"),
                "allocation": pl.Float64,
                "entry_equity": pl.Float64,
                "risk": pl.Float64,
                "pnl": pl.Float64,
                "realized_pnl": pl.Float64,
                "status": pl.String,
            },
        )


def _validate_trade(trade: PositionTrade, index: int, strategy_count: int) -> None:
    """Reject malformed position identities before constructing event schedules."""
    if not 0 <= trade.strategy_index < strategy_count:
        raise ValueError(f"Trade {index} has an invalid strategy index")
    if trade.closed < trade.opened:
        raise ValueError(f"Trade {index} closes before it opens")
    if not math.isfinite(trade.return_on_risk):
        raise ValueError(f"Trade {index} return_on_risk must be finite")
    if any(blocker < 0 or blocker >= index for blocker in trade.blockers):
        raise ValueError(f"Trade {index} blockers must identify earlier trades")


def _new_ledger(
    allocations: torch.Tensor, days: int, trades: int, record_entries: bool
) -> _Ledger:
    """Allocate only candidate-by-trade and candidate-by-day buffers."""
    batch = allocations.shape[0]
    device, dtype = allocations.device, allocations.dtype
    return _Ledger(
        equity=torch.ones(batch, device=device, dtype=dtype),
        realized_equity=torch.ones(batch, device=device, dtype=dtype),
        reserved=torch.zeros(batch, device=device, dtype=dtype),
        open_count=torch.zeros(batch, device=device, dtype=torch.int32),
        risks=torch.zeros((batch, trades), device=device, dtype=dtype),
        alive=torch.ones(batch, device=device, dtype=torch.bool),
        daily_returns=torch.zeros((batch, days), device=device, dtype=dtype),
        active=torch.zeros((batch, days), device=device, dtype=torch.bool),
        entry_equities=(
            torch.zeros((batch, trades), device=device, dtype=dtype)
            if record_entries
            else None
        ),
        entry_allocations=(
            torch.zeros((batch, trades), device=device, dtype=dtype)
            if record_entries
            else None
        ),
        statuses=(
            torch.zeros((batch, trades), device=device, dtype=torch.int8)
            if record_entries
            else None
        ),
    )


def _finish_day(ledger: _Ledger, next_equity: torch.Tensor, day: int) -> None:
    """Record daily net-liquidation return, marking insolvent paths as invalid."""
    daily_pnl = next_equity - ledger.equity
    finite = torch.isfinite(next_equity) & torch.isfinite(daily_pnl)
    denominator = torch.where(ledger.alive, ledger.equity, 1.0)
    daily_return = daily_pnl / denominator
    finite &= torch.isfinite(daily_return)
    ledger.daily_returns[:, day] = torch.where(
        ledger.alive & finite, daily_return, -1.0
    )
    ledger.alive &= finite & (next_equity > 0)
    ledger.equity = torch.where(ledger.alive, next_equity, 0.0)
