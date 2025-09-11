"""State management for trading environment using dataclass instead of dict."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class State:
    """Trading environment state as a dataclass to avoid Torch Dynamo side effects.

    This replaces the previous dict[str, torch.Tensor] state to fix compilation
    issues with torch.compile() when policies mutate state attributes.
    """

    # Core environment state
    date_idx: torch.Tensor
    time_idx: torch.Tensor
    position: torch.Tensor
    entry_price: torch.Tensor
    entry_position: torch.Tensor
    done: torch.Tensor
    total_profit: torch.Tensor
    total_profit_percent: torch.Tensor
    entry_cost: torch.Tensor
    prev_stop_loss: torch.Tensor

    # Policy-specific state
    had_position: torch.Tensor  # Used by SingleTradeDonePolicy
    opened: torch.Tensor  # Used by OpenOncePolicy
    maint_stage: torch.Tensor  # Used by ScaledArtrMaintenancePolicy
    base_price: torch.Tensor  # Used by ScaledArtrMaintenancePolicy
    entry_date_idx: torch.Tensor  # Used by ScaledArtrMaintenancePolicy
    entry_time_idx: torch.Tensor  # Used by ScaledArtrMaintenancePolicy
    maint_anchor: torch.Tensor  # Used by PercentGainMaintenancePolicy
    prev_stop: torch.Tensor  # Used in some maintenance policies

    # Temporary state used in step calculations
    profit: torch.Tensor
    profit_percent: torch.Tensor

    @classmethod
    def create(
        cls,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        start_date_idx: torch.Tensor,
        start_time_idx: torch.Tensor,
    ) -> State:
        """Factory method to initialize State with default tensors."""
        return cls(
            # Core environment state
            date_idx=start_date_idx.to(torch.int32).to(device),
            time_idx=start_time_idx.to(torch.int32).to(device),
            position=torch.zeros(batch_size, dtype=torch.int32, device=device),
            entry_price=torch.full(
                (batch_size,), float("nan"), dtype=dtype, device=device
            ),
            entry_position=torch.zeros(batch_size, dtype=torch.int32, device=device),
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            total_profit=torch.zeros(batch_size, dtype=dtype, device=device),
            total_profit_percent=torch.zeros(batch_size, dtype=dtype, device=device),
            entry_cost=torch.zeros(batch_size, dtype=dtype, device=device),
            prev_stop_loss=torch.full(
                (batch_size,), float("nan"), dtype=dtype, device=device
            ),
            # Policy-specific state - initialized to default values
            had_position=torch.zeros(batch_size, dtype=torch.bool, device=device),
            opened=torch.zeros(batch_size, dtype=torch.bool, device=device),
            maint_stage=torch.zeros(batch_size, dtype=torch.int32, device=device),
            base_price=torch.full(
                (batch_size,), float("nan"), dtype=dtype, device=device
            ),
            entry_date_idx=torch.full(
                (batch_size,), -1, dtype=torch.int32, device=device
            ),
            entry_time_idx=torch.full(
                (batch_size,), -1, dtype=torch.int32, device=device
            ),
            maint_anchor=torch.full(
                (batch_size,), float("nan"), dtype=dtype, device=device
            ),
            prev_stop=torch.full(
                (batch_size,), float("nan"), dtype=dtype, device=device
            ),
            # Initialize profit fields to zeros
            profit=torch.zeros(batch_size, dtype=dtype, device=device),
            profit_percent=torch.zeros(batch_size, dtype=dtype, device=device),
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert State to dict format for backward compatibility.

        This is useful during the transition period and for any code
        that still expects dict format.
        """
        result = {
            "date_idx": self.date_idx,
            "time_idx": self.time_idx,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_position": self.entry_position,
            "done": self.done,
            "total_profit": self.total_profit,
            "total_profit_percent": self.total_profit_percent,
            "entry_cost": self.entry_cost,
            "prev_stop_loss": self.prev_stop_loss,
            "had_position": self.had_position,
            "opened": self.opened,
            "maint_stage": self.maint_stage,
            "base_price": self.base_price,
            "entry_date_idx": self.entry_date_idx,
            "entry_time_idx": self.entry_time_idx,
            "maint_anchor": self.maint_anchor,
            "prev_stop": self.prev_stop,
            "profit": self.profit,
            "profit_percent": self.profit_percent,
        }

        return result

    @classmethod
    def from_dict(cls, state_dict: dict[str, torch.Tensor]) -> State:
        """Create State from dict format for backward compatibility."""
        return cls(
            date_idx=state_dict["date_idx"],
            time_idx=state_dict["time_idx"],
            position=state_dict["position"],
            entry_price=state_dict["entry_price"],
            entry_position=state_dict["entry_position"],
            done=state_dict["done"],
            total_profit=state_dict["total_profit"],
            total_profit_percent=state_dict["total_profit_percent"],
            entry_cost=state_dict["entry_cost"],
            prev_stop_loss=state_dict["prev_stop_loss"],
            had_position=state_dict.get(
                "had_position", torch.zeros_like(state_dict["done"])
            ),
            opened=state_dict.get("opened", torch.zeros_like(state_dict["done"])),
            maint_stage=state_dict.get(
                "maint_stage", torch.zeros_like(state_dict["position"])
            ),
            base_price=state_dict.get(
                "base_price", torch.full_like(state_dict["entry_price"], float("nan"))
            ),
            entry_date_idx=state_dict.get(
                "entry_date_idx", torch.full_like(state_dict["position"], -1)
            ),
            entry_time_idx=state_dict.get(
                "entry_time_idx", torch.full_like(state_dict["position"], -1)
            ),
            maint_anchor=state_dict.get(
                "maint_anchor", torch.full_like(state_dict["entry_price"], float("nan"))
            ),
            prev_stop=state_dict.get(
                "prev_stop", torch.full_like(state_dict["entry_price"], float("nan"))
            ),
            profit=state_dict.get(
                "profit", torch.zeros_like(state_dict["entry_price"])
            ),
            profit_percent=state_dict.get(
                "profit_percent", torch.zeros_like(state_dict["entry_price"])
            ),
        )
