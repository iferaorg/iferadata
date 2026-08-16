"""Memory planning for vectorized capital-allocation candidate batches."""

from __future__ import annotations

import torch

_BOOTSTRAP_MEMORY_SAFETY_FACTOR = 2
_BOOTSTRAP_RUN_CHUNK_SIZE = 16
_BATCH_MEMORY_BUDGET_BYTES = 4 * 1024 * 1024 * 1024
_MAX_CANDIDATES_PER_BATCH = 4 * 1024 * 1024
_PREFIX_STACK_MEMORY_SAFETY_FACTOR = 4


def candidate_batch_size(
    time_count: int,
    strategy_count: int,
    clique_count: int,
    dtype: torch.dtype,
    device: torch.device,
    uses_prefix_expansion: bool,
    bootstrap_runs: int,
    bootstrap_length: int,
) -> int:
    """Choose a conservative candidate batch size for path calculations."""
    element_size = torch.empty((), dtype=dtype).element_size()
    path_bytes = time_count * (3 * element_size + 8)
    bootstrap_index_bytes = 0
    if bootstrap_runs > 0:
        run_chunk_size = min(_BOOTSTRAP_RUN_CHUNK_SIZE, bootstrap_runs)
        bootstrap_path_bytes = (
            time_count * element_size
            + run_chunk_size * bootstrap_length * (2 * element_size + 8)
            + bootstrap_runs * (2 * element_size + 8)
        )
        path_bytes = _BOOTSTRAP_MEMORY_SAFETY_FACTOR * bootstrap_path_bytes
        # The compiled higher-order map pads/copies indices into equal chunks;
        # reserve both that input and the original shared index matrix.
        bootstrap_index_bytes = 2 * bootstrap_runs * bootstrap_length * 8

    prefix_stack_bytes = 0
    if uses_prefix_expansion:
        prefix_stack_bytes = _PREFIX_STACK_MEMORY_SAFETY_FACTOR * (
            4 * strategy_count * (strategy_count - 1)
            + 8 * strategy_count * clique_count
            + 16 * strategy_count
        )
    candidate_bytes = max(
        path_bytes + (strategy_count + clique_count) * 32 + prefix_stack_bytes,
        1,
    )

    memory_budget = _BATCH_MEMORY_BUDGET_BYTES
    if device.type == "cuda":
        free_memory, _ = torch.cuda.mem_get_info(device)
        memory_budget = min(memory_budget, free_memory // 4)
    if bootstrap_index_bytes >= memory_budget:
        raise ValueError("bootstrap index tensor exceeds the search memory budget")
    memory_budget -= bootstrap_index_bytes
    if candidate_bytes > memory_budget:
        raise ValueError("one candidate exceeds the search memory budget")
    return max(1, min(_MAX_CANDIDATES_PER_BATCH, memory_budget // candidate_bytes))
