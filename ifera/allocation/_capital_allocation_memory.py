"""Memory planning for vectorized capital-allocation candidate batches."""

from __future__ import annotations

from collections.abc import Callable

import torch

_BOOTSTRAP_MEMORY_SAFETY_FACTOR = 2
_BOOTSTRAP_RUN_CHUNK_SIZE = 16
_BATCH_MEMORY_BUDGET_BYTES = 12 * 1024 * 1024 * 1024
_MAX_CANDIDATES_PER_BATCH = 4 * 1024 * 1024
_PREFIX_STACK_MEMORY_SAFETY_FACTOR = 4
_CUDA_FREE_MEMORY_SAFETY_FACTOR = 2


def candidate_batch_size(
    time_count: int,
    strategy_count: int,
    clique_count: int,
    dtype: torch.dtype,
    device: torch.device,
    uses_prefix_expansion: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
    bootstrap_paths: bool = True,
    block_group_count: int = 0,
) -> int:
    """Choose a conservative candidate batch size for path calculations."""
    element_size = torch.empty((), dtype=dtype).element_size()
    path_bytes = time_count * (3 * element_size + 8)
    bootstrap_index_bytes = 0
    if bootstrap_runs > 0:
        if bootstrap_paths:
            padded_runs = (
                (bootstrap_runs + _BOOTSTRAP_RUN_CHUNK_SIZE - 1)
                // _BOOTSTRAP_RUN_CHUNK_SIZE
                * _BOOTSTRAP_RUN_CHUNK_SIZE
            )
            bootstrap_path_bytes = (
                time_count * element_size
                + _BOOTSTRAP_RUN_CHUNK_SIZE * bootstrap_length * (2 * element_size + 8)
                + padded_runs * (2 * element_size + 8)
            )
            path_bytes = _BOOTSTRAP_MEMORY_SAFETY_FACTOR * bootstrap_path_bytes
            # Compiled map retains the input and its padded equal-chunk copy.
            bootstrap_index_bytes = (
                (bootstrap_runs + padded_runs) * bootstrap_length * 8
            )
        else:
            bootstrap_index_bytes = bootstrap_runs * bootstrap_length * 8

    prefix_stack_bytes = 0
    if uses_prefix_expansion:
        prefix_stack_bytes = _PREFIX_STACK_MEMORY_SAFETY_FACTOR * (
            4 * strategy_count * (strategy_count - 1)
            + 8 * strategy_count * clique_count
            + 24 * strategy_count
        )
    block_tensor_bytes = block_group_count * (strategy_count * element_size + 2 * 8)
    block_workspace_bytes = (
        strategy_count + block_group_count * (3 * element_size + 1)
        if block_group_count
        else 0
    )
    candidate_bytes = max(
        path_bytes
        + (strategy_count + clique_count) * 32
        + prefix_stack_bytes
        + block_workspace_bytes,
        1,
    )

    memory_budget = _available_memory_budget(device)
    fixed_input_bytes = bootstrap_index_bytes + block_tensor_bytes
    if bootstrap_index_bytes >= memory_budget:
        raise ValueError("bootstrap index tensor exceeds the search memory budget")
    if block_tensor_bytes >= memory_budget - bootstrap_index_bytes:
        raise ValueError("trade-block tensor exceeds the search memory budget")
    memory_budget -= fixed_input_bytes
    if candidate_bytes > memory_budget:
        raise ValueError("one candidate exceeds the search memory budget")
    return max(1, min(_MAX_CANDIDATES_PER_BATCH, memory_budget // candidate_bytes))


def validate_bootstrap_diagnostic_memory(
    time_count: int,
    strategy_count: int,
    clique_count: int,
    dtype: torch.dtype,
    device: torch.device,
    bootstrap_runs: int,
    bootstrap_length: int,
) -> None:
    """Validate the eager one-winner bootstrap workspace and input indices."""
    element_size = torch.empty((), dtype=dtype).element_size()
    run_chunk_size = min(_BOOTSTRAP_RUN_CHUNK_SIZE, bootstrap_runs)
    path_bytes = _BOOTSTRAP_MEMORY_SAFETY_FACTOR * (
        time_count * element_size
        + run_chunk_size * bootstrap_length * (2 * element_size + 8)
        + bootstrap_runs * (2 * element_size + 8)
    )
    candidate_bytes = path_bytes + (strategy_count + clique_count) * 32
    index_bytes = bootstrap_runs * bootstrap_length * 8
    memory_budget = _available_memory_budget(device)
    if index_bytes >= memory_budget:
        raise ValueError("bootstrap index tensor exceeds the search memory budget")
    if candidate_bytes > memory_budget - index_bytes:
        raise ValueError("one candidate exceeds the search memory budget")


def _available_memory_budget(device: torch.device) -> int:
    """Return the configured budget capped by conservative free CUDA memory."""
    memory_budget = _BATCH_MEMORY_BUDGET_BYTES
    if device.type == "cuda":
        free_memory, _ = torch.cuda.mem_get_info(device)
        memory_budget = min(
            memory_budget, free_memory // _CUDA_FREE_MEMORY_SAFETY_FACTOR
        )
    return memory_budget


def search_candidate_batch_size(
    time_count: int,
    strategy_count: int,
    clique_count: int,
    dtype: torch.dtype,
    device: torch.device,
    uses_prefix_expansion: bool,
    bootstrap_needed: bool,
    drawdown_required: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
    maximum_candidates: int,
    planner: Callable[..., int] = candidate_batch_size,
    block_group_count: int = 0,
) -> int:
    """Plan grid scoring separately from final-winner bootstrap diagnostics."""
    bootstrap_scoring_needed = bootstrap_needed and drawdown_required
    if bootstrap_needed and not bootstrap_scoring_needed:
        validate_bootstrap_diagnostic_memory(
            time_count,
            strategy_count,
            clique_count,
            dtype,
            device,
            bootstrap_runs,
            bootstrap_length,
        )
    planner_arguments = (
        time_count,
        strategy_count,
        clique_count,
        dtype,
        device,
        uses_prefix_expansion,
        bootstrap_runs if bootstrap_needed else 0,
        bootstrap_length if bootstrap_needed else 0,
        bootstrap_scoring_needed,
    )
    if block_group_count:
        planned_size = planner(*planner_arguments, block_group_count=block_group_count)
    else:
        planned_size = planner(*planner_arguments)
    return min(maximum_candidates, planned_size)
