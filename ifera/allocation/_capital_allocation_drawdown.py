"""Drawdown calculations used by the capital-allocation search."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch._higher_order_ops.map import map as torch_higher_order_map

_BOOTSTRAP_RUN_CHUNK_SIZE = 16
_higher_order_map = cast(Callable[..., torch.Tensor], torch_higher_order_map)


def negative_rms_drawdown(
    portfolio_log_returns: torch.Tensor, dim: int
) -> torch.Tensor:
    """Return negative RMS fractional drawdown along a time dimension."""
    log_equity = portfolio_log_returns.cumsum_(dim=dim)
    running_peak = torch.cummax(log_equity, dim=dim).values
    running_peak.clamp_min_(0.0)
    log_equity.sub_(running_peak).expm1_()
    average_drawdown = log_equity.square_().mean(dim=dim).sqrt_().neg_()
    average_drawdown.clamp_min_(-1.0)
    return average_drawdown


def _bootstrap_chunk_drawdowns(
    bootstrap_indices: torch.Tensor, portfolio_returns: torch.Tensor
) -> torch.Tensor:
    """Calculate ADD for one bootstrap-index chunk."""
    sampled_log_returns = portfolio_returns[:, bootstrap_indices]
    sampled_log_returns.log1p_()
    return negative_rms_drawdown(sampled_log_returns, dim=2)


def _mapped_bootstrap_drawdowns(
    portfolio_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor,
) -> torch.Tensor:
    """Map the non-pointwise path calculation over equally sized run chunks."""
    run_count, bootstrap_length = bootstrap_indices.shape
    padding = (-run_count) % _BOOTSTRAP_RUN_CHUNK_SIZE
    padded_indices = torch.nn.functional.pad(
        bootstrap_indices,
        (0, 0, 0, padding),
    )
    index_chunks = padded_indices.reshape(
        -1,
        _BOOTSTRAP_RUN_CHUNK_SIZE,
        bootstrap_length,
    )
    chunk_drawdowns = _higher_order_map(
        _bootstrap_chunk_drawdowns,
        index_chunks,
        portfolio_returns,
    )
    return chunk_drawdowns.permute(1, 0, 2).reshape(portfolio_returns.shape[0], -1)[
        :, :run_count
    ]


def bootstrap_average_drawdown(
    portfolio_returns: torch.Tensor,
    bootstrap_indices: torch.Tensor,
    percentile: float,
) -> torch.Tensor:
    """Return the lower-percentile ADD across shared resampled paths."""
    if torch.compiler.is_compiling():
        # General map supports the scans and reduction in the mapped body while
        # preserving the bounded 16-run path workspace under torch.compile.
        run_drawdowns = _mapped_bootstrap_drawdowns(
            portfolio_returns,
            bootstrap_indices,
        )
    else:
        # The higher-order map has substantial eager tracing overhead. Retain the
        # same bounded chunks while avoiding that cost outside torch.compile.
        run_count = bootstrap_indices.shape[0]
        run_drawdowns = torch.empty(
            (portfolio_returns.shape[0], run_count),
            device=portfolio_returns.device,
            dtype=portfolio_returns.dtype,
        )
        for start in range(0, run_count, _BOOTSTRAP_RUN_CHUNK_SIZE):
            stop = min(start + _BOOTSTRAP_RUN_CHUNK_SIZE, run_count)
            sampled_log_returns = portfolio_returns[:, bootstrap_indices[start:stop]]
            sampled_log_returns.log1p_()
            run_drawdowns[:, start:stop] = negative_rms_drawdown(
                sampled_log_returns,
                dim=2,
            )
    return torch.quantile(
        run_drawdowns,
        percentile / 100.0,
        dim=1,
        interpolation="linear",
    )
