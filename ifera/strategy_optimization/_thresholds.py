"""Project normalized DNF thresholds onto feature grids in original units."""

from __future__ import annotations

import torch
from torch import Tensor

from ._features import FeatureSpec


class ThresholdRounder:
    """Cache device-local grid metadata for candidate batches of any size.

    Observations stay untouched. Float64 intermediates match feature preparation
    and avoid moving inclusive boundaries by a float32 rounding error. Exact zero
    and one endpoints retain their meaning as unrestricted interval tails.
    """

    def __init__(self, features: tuple[FeatureSpec, ...], device: torch.device) -> None:
        self.enabled = any(feature.granularity is not None for feature in features)
        self.minimum = torch.tensor(
            [[feature.min_value] for feature in features],
            dtype=torch.float64,
            device=device,
        )
        self.maximum = torch.tensor(
            [[feature.max_value] for feature in features],
            dtype=torch.float64,
            device=device,
        )
        self.scale = self.maximum - self.minimum
        self.granularity = torch.tensor(
            [[feature.granularity or 1.0] for feature in features],
            dtype=torch.float64,
            device=device,
        )
        self.rounded_features = torch.tensor(
            [[feature.granularity is not None] for feature in features],
            dtype=torch.bool,
            device=device,
        )

    def __call__(self, bounds: Tensor) -> Tensor:
        """Round to nearest zero-based grid points, using ties-to-even, then clip."""
        if not self.enabled:
            return bounds
        raw = self.minimum + bounds.to(dtype=torch.float64) * self.scale
        rounded = (raw / self.granularity).round() * self.granularity
        rounded = rounded.clamp(min=self.minimum, max=self.maximum)
        normalized = ((rounded - self.minimum) / self.scale).clamp(0.0, 1.0)
        preserve = (bounds == 0) | (bounds == 1) | ~self.rounded_features
        return torch.where(preserve, bounds, normalized.to(dtype=bounds.dtype))


def round_bounds(bounds: Tensor, features: tuple[FeatureSpec, ...]) -> Tensor:
    """Project batched or unbatched bounds without changing candidate storage.

    Repeated evaluation should construct one ``ThresholdRounder`` and reuse its
    metadata. This convenience wrapper also serves one-off condition rendering.
    """
    return ThresholdRounder(features, bounds.device)(bounds)
