from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import nn
import tensordict as td


class PolicyBase(nn.Module, ABC):
    """Abstract base class for policies."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def copy_to(self, device: torch.device) -> PolicyBase:
        """Return a copy of the policy on the specified device."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, state: td.TensorDict) -> td.TensorDict:
        """Reset any internal state held by the policy."""
        raise NotImplementedError

    @abstractmethod
    def masked_reset(self, state: td.TensorDict, mask: torch.Tensor) -> td.TensorDict:
        """Reset any internal state held by the policy, using a mask."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state: td.TensorDict,
    ) -> td.TensorDict:
        """Forward pass of the policy. Returns updated state TensorDict."""
        raise NotImplementedError
