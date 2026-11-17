"""Positive batched optimization metrics for daily return-on-risk percentages.

The input shape is ``(candidates, market_days)`` and a value of ``5`` means
five percent. All market days, including days without trades, belong in the
time dimension. Built-in metrics retain the input device and floating dtype;
wrappers preserve the child metric's output dtype.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from numbers import Real

import torch
from torch import Tensor

from ifera.allocation._capital_allocation_drawdown import negative_rms_drawdown

_MIN_OBJECTIVE = 1e-6


def _validate_returns_metadata(returns: Tensor) -> None:
    """Check shape and dtype without launching a tensor calculation."""
    if returns.ndim != 2 or returns.shape[1] == 0:
        raise ValueError("returns must have shape (batch, days) with at least one day")
    if not returns.is_floating_point():
        raise TypeError("returns must have a floating dtype")


def _validate_returns(returns: Tensor) -> None:
    """Require a nonempty day axis and finite floating percentage returns."""
    _validate_returns_metadata(returns)
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("returns must contain only finite values")


def _validate_periods(annualization_periods: int) -> None:
    """Require a positive integer annualization period count."""
    if (
        isinstance(annualization_periods, bool)
        or not isinstance(annualization_periods, int)
        or annualization_periods <= 0
    ):
        raise ValueError("annualization_periods must be a positive integer")


def _validate(returns: Tensor, annualization_periods: int) -> None:
    """Require valid returns and a positive annualization period count."""
    _validate_returns(returns)
    _validate_periods(annualization_periods)


def _positive(value: Tensor) -> Tensor:
    """Shift a metric by one and keep overflow and losses finite and positive."""
    return torch.nan_to_num(
        value + 1.0,
        nan=_MIN_OBJECTIVE,
        posinf=torch.finfo(value.dtype).max,
        neginf=_MIN_OBJECTIVE,
    ).clamp_min(_MIN_OBJECTIVE)


def _annualized_return(returns: Tensor, annualization_periods: int) -> Tensor:
    """Compute annualized return without repeating the public input validation."""
    return _positive(returns.mean(dim=1) * annualization_periods)


def _sortino(returns: Tensor, annualization_periods: int) -> Tensor:
    """Compute Sortino using only graph-compatible tensor arithmetic."""
    numerator = returns.mean(dim=1) * annualization_periods
    denominator = returns.clamp_max(0.0).square().mean(dim=1).sqrt() * math.sqrt(
        annualization_periods
    )
    return _positive(
        torch.where(
            numerator < 0.0,
            numerator,
            numerator / denominator.clamp_min(_MIN_OBJECTIVE),
        )
    )


def _return_over_add(returns: Tensor, annualization_periods: int) -> Tensor:
    """Compute ADD on new log-return storage, leaving the caller's input intact."""
    numerator = returns.mean(dim=1) * annualization_periods
    log_returns = (returns / 100.0).clamp_min(-1.0).log1p()
    denominator = negative_rms_drawdown(log_returns, dim=1).abs() * 100.0
    return _positive(
        torch.where(
            numerator < 0.0,
            numerator,
            numerator / denominator.clamp_min(_MIN_OBJECTIVE),
        )
    )


def _annualized_return_with_validity(
    returns: Tensor, annualization_periods: int
) -> tuple[Tensor, Tensor]:
    """Fuse the finite-input reduction with annualized-return arithmetic."""
    return (
        _annualized_return(returns, annualization_periods),
        torch.isfinite(returns).all(),
    )


def _sortino_with_validity(
    returns: Tensor, annualization_periods: int
) -> tuple[Tensor, Tensor]:
    """Fuse the finite-input reduction with Sortino arithmetic."""
    return _sortino(returns, annualization_periods), torch.isfinite(returns).all()


def _return_over_add_with_validity(
    returns: Tensor, annualization_periods: int
) -> tuple[Tensor, Tensor]:
    """Fuse the finite-input reduction with ADD arithmetic."""
    return (
        _return_over_add(returns, annualization_periods),
        torch.isfinite(returns).all(),
    )


# Report invalid inputs with recoverable Python errors after the fused reduction.
# The default graph-break policy permits eager fallback after many shape variants.
_compiled_annualized_return = torch.compile(
    _annualized_return_with_validity, dynamic=True
)
_compiled_sortino = torch.compile(_sortino_with_validity, dynamic=True)
_compiled_return_over_add = torch.compile(_return_over_add_with_validity, dynamic=True)


def _compute_metric(
    returns: Tensor,
    annualization_periods: int,
    eager: Callable[[Tensor, int], Tensor],
    compiled: Callable[[Tensor, int], tuple[Tensor, Tensor]],
) -> Tensor:
    """Keep CPU validation eager and fuse finite checks into CUDA metric work."""
    if returns.device.type != "cuda":
        _validate(returns, annualization_periods)
        return eager(returns, annualization_periods)
    _validate_returns_metadata(returns)
    try:
        _validate_periods(annualization_periods)
    except ValueError:
        # Retain finite-input error precedence when both arguments are invalid.
        _validate_returns(returns)
        raise
    values, valid = compiled(returns, annualization_periods)
    if not bool(valid):
        raise ValueError("returns must contain only finite values")
    return values


def annualized_return(returns: Tensor, *, annualization_periods: int = 252) -> Tensor:
    """Return annualized sum of RoR percentages plus one, clamped positive."""
    return _compute_metric(
        returns, annualization_periods, _annualized_return, _compiled_annualized_return
    )


def sortino(returns: Tensor, *, annualization_periods: int = 252) -> Tensor:
    """Return annualized Sortino plus one, using zero as the target return.

    Downside deviation is the RMS of negative returns over *all* market days,
    annualized by ``sqrt(annualization_periods)``. A negative annualized numerator
    is used directly: dividing it would reward candidates with greater downside.
    A zero downside deviation is floored at ``1e-6`` percentage points.
    """
    return _compute_metric(returns, annualization_periods, _sortino, _compiled_sortino)


def return_over_add(returns: Tensor, *, annualization_periods: int = 252) -> Tensor:
    """Return annualized RoR divided by RMS drawdown percent, plus one.

    ADD uses the allocation package's compounded-equity implementation. Returns
    of -100% or less represent absorbing bankruptcy: equity becomes zero and
    stays zero. Only the equity calculation clips losses to -100%; the numerator
    retains the original returns, including losses beyond the initial risk.
    Negative numerators are not divided. Zero ADD is floored at ``1e-6`` percent.
    """
    return _compute_metric(
        returns, annualization_periods, _return_over_add, _compiled_return_over_add
    )


def with_scarcity_penalty(
    metric: Callable[[Tensor], Tensor],
    lower_threshold: float,
    upper_threshold: float,
) -> Callable[[Tensor], Tensor]:
    """Wrap a batched metric with a square-root trading-frequency penalty.

    Thresholds must satisfy ``0 <= lower_threshold < upper_threshold <= 1``.
    Frequency is the fraction of nonzero daily returns, counting gains and
    losses alike. A zero-return trade cannot be distinguished from a no-trade
    day by this metric's returns-only input contract.

    The child is evaluated once for the entire batch. Below the lower threshold,
    return 1e-6. Inside the interval, multiply the child by
    ``sqrt(1e-6 + (1 - 1e-6) * (frequency - lower) / (upper - lower))``.
    At or above the upper threshold, return the child unchanged. The wrapper
    does not add another objective floor; the optimizer applies its usual floor.
    Result dtype and device follow the child metric.
    """
    if not callable(metric):
        raise TypeError("metric must be callable")
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in (lower_threshold, upper_threshold)
    ):
        raise TypeError("scarcity thresholds must be real numbers")
    lower = float(lower_threshold)
    upper = float(upper_threshold)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= lower_threshold < upper_threshold <= 1"
        )

    def penalized(returns: Tensor) -> Tensor:
        """Evaluate the child before applying each candidate's scarcity penalty."""
        _validate_returns(returns)
        values = metric(returns)
        if (
            not isinstance(values, Tensor)
            or values.shape != (returns.shape[0],)
            or not values.is_floating_point()
            or values.device != returns.device
        ):
            raise ValueError(
                "child metric must return floating (batch,) scores on the input device"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("child metric must return finite scores")
        frequency = (
            returns.count_nonzero(dim=1).to(dtype=torch.float64) / returns.shape[1]
        )
        progress = ((frequency - lower) / (upper - lower)).clamp(0.0, 1.0)
        factor = (_MIN_OBJECTIVE + (1.0 - _MIN_OBJECTIVE) * progress).sqrt()
        scaled = values * factor.to(dtype=values.dtype)
        return torch.where(
            frequency < lower,
            torch.full_like(values, _MIN_OBJECTIVE),
            torch.where(frequency >= upper, values, scaled),
        )

    return penalized
