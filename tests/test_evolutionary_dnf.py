"""Behavioral tests for tensor-based evolutionary DNF optimization."""

from itertools import product
from pathlib import Path

import pytest
import torch

from ifera.algorithms import evolutionary_dnf
from ifera.algorithms.evolutionary_dnf import EvolutionaryDNFOptimizer


def _matches(bounds, active_conjunctions, features):
    """Interpret the documented inclusive DNF encoding against feature rows."""
    atoms = (features[None, None, :, :] >= bounds[:, :, None, :, 0]) & (
        features[None, None, :, :] <= bounds[:, :, None, :, 1]
    )
    return (atoms.all(dim=-1) & active_conjunctions[:, :, None]).any(dim=1)


def _constant_objective(bounds, _active_conjunctions):
    return torch.ones(bounds.shape[0], device=bounds.device)


@pytest.mark.parametrize(
    "boolean_count,continuous_count", [(3, 2), (3, 0), (0, 2), (0, 0)]
)
def test_evolution_preserves_the_objective_tensor_contract(
    boolean_count, continuous_count
):
    """Initialization and all evolutionary operators preserve valid intervals."""
    calls = []

    def objective(bounds, active_conjunctions):
        assert bounds.shape[1:] == (3, boolean_count + continuous_count, 2)
        assert bounds.dtype == torch.float32
        assert bounds.device.type == "cpu"
        assert active_conjunctions.shape == bounds.shape[:2]
        assert active_conjunctions.dtype == torch.bool
        assert active_conjunctions.device == bounds.device
        assert not bounds.requires_grad
        assert not torch.is_grad_enabled()
        assert torch.isfinite(bounds).all()
        assert ((bounds >= 0) & (bounds <= 1)).all()
        assert (bounds[..., 0] <= bounds[..., 1]).all()
        boolean_bounds = bounds[:, :, :boolean_count]
        assert ((boolean_bounds == 0) | (boolean_bounds == 1)).all()
        calls.append(bounds.shape[0])
        return 1.0 + active_conjunctions.float().sum(dim=1)

    optimizer = EvolutionaryDNFOptimizer(
        boolean_count,
        continuous_count,
        max_conjunctions=3,
        device="cpu",
        seed=13,
        objective_batch_size=7,
        mutation_rate=1.0,
        conjunction_mutation_rate=0.5,
        crossover_rate=1.0,
    )
    result = optimizer.optimize(objective, max_population=24, generations=4)

    assert result.bounds.shape == (3, boolean_count + continuous_count, 2)
    assert result.active_conjunctions.shape == (3,)
    assert result.generations == 4
    assert 2 <= result.population_size <= 24
    assert max(calls) <= 7
    assert result.evaluations == sum(calls)
    assert result.objective_value > 0


def test_documented_dnf_semantics_and_seeded_constants():
    """Inactive padding is false and unrestricted active clauses are true."""
    features = torch.tensor([[0.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    bounds = torch.tensor(
        [
            [[[1.0, 1.0], [0.5, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
        ]
    )
    active = torch.tensor([[True, False], [True, False], [False, False]])
    assert _matches(bounds, active, features).tolist() == [
        [False, True, True],
        [True, True, True],
        [False, False, False],
    ]

    seen = []

    def objective(candidate_bounds, candidate_active):
        seen.append(_matches(candidate_bounds, candidate_active, features))
        return torch.ones(candidate_bounds.shape[0])

    EvolutionaryDNFOptimizer(1, 1, device="cpu", seed=4).optimize(
        objective, max_population=16, generations=0
    )
    predictions = torch.cat(seen)
    assert predictions.all(dim=1).any()
    assert (~predictions.any(dim=1)).any()


def test_penalties_count_active_boolean_literals_and_continuous_comparisons():
    """Two-sided ranges cost two atoms and inactive clauses incur no cost."""
    bounds = torch.tensor(
        [
            [
                [[1.0, 1.0], [0.0, 1.0], [0.2, 0.8], [0.0, 0.6]],
                [[0.0, 1.0], [0.0, 0.0], [0.5, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.0, 0.0], [0.2, 0.8], [0.2, 0.8]],
            ],
            [
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.0, 0.0], [0.2, 0.8], [0.2, 0.8]],
            ],
        ]
    )
    active = torch.tensor([[True, True, False], [True, False, False]])
    optimizer = EvolutionaryDNFOptimizer(
        2,
        2,
        max_conjunctions=3,
        boolean_complexity_penalty=0.3,
        continuous_complexity_penalty=0.05,
        device="cpu",
    )

    boolean_conditions, continuous_conditions = optimizer.complexity(bounds, active)
    assert boolean_conditions.tolist() == [2, 0]
    assert continuous_conditions.tolist() == [4, 0]
    fitness = optimizer.fitness(torch.tensor([2.0, 2.0]), bounds, active)
    assert fitness.tolist() == pytest.approx([2.0 / 1.8, 2.0])


def test_seeded_search_is_reproducible_without_changing_global_random_state():
    """Independent local generators produce the same result on a fixed device."""
    random_state = torch.random.get_rng_state().clone()
    first = EvolutionaryDNFOptimizer(2, 1, device="cpu", seed=17).optimize(
        _constant_objective, max_population=24, generations=5
    )
    second = EvolutionaryDNFOptimizer(2, 1, device="cpu", seed=17).optimize(
        _constant_objective, max_population=24, generations=5
    )

    assert torch.equal(first.bounds, second.bounds)
    assert torch.equal(first.active_conjunctions, second.active_conjunctions)
    assert first.objective_value == second.objective_value
    assert first.fitness == second.fitness
    assert torch.equal(torch.random.get_rng_state(), random_state)


def test_evolves_a_boolean_dnf_using_only_black_box_feedback():
    """A synthetic environment supplies all knowledge of its hidden truth table."""
    features = torch.tensor(list(product([0.0, 1.0], repeat=3)))
    target = (features[:, 0].bool() & ~features[:, 1].bool()) | (
        ~features[:, 0].bool() & features[:, 2].bool()
    )

    def objective(bounds, active):
        return 1.0 + (_matches(bounds, active, features) == target).float().mean(dim=1)

    options = dict(
        max_conjunctions=2,
        device="cpu",
        seed=45,
        boolean_complexity_penalty=0.001,
        continuous_complexity_penalty=0.0,
    )
    initial = EvolutionaryDNFOptimizer(3, 0, **options).optimize(
        objective, max_population=96, generations=0
    )
    evolved = EvolutionaryDNFOptimizer(3, 0, **options).optimize(
        objective, max_population=96, generations=40
    )

    assert evolved.fitness >= initial.fitness
    assert evolved.objective_value == 2.0
    assert torch.equal(
        _matches(evolved.bounds[None], evolved.active_conjunctions[None], features)[0],
        target,
    )


def test_evolves_continuous_bounds_toward_a_hidden_optimum():
    """Continuous perturbations improve threshold precision beyond initialization."""
    target = torch.tensor([0.312, 0.714])

    def objective(bounds, active):
        quality = 1.0 - (bounds[:, 0, 0] - target).abs().mean(dim=1)
        return 1.0 + quality * active[:, 0]

    options = dict(
        max_conjunctions=1,
        device="cpu",
        seed=22,
        continuous_complexity_penalty=0.0,
    )
    initial = EvolutionaryDNFOptimizer(0, 1, **options).optimize(
        objective, max_population=48, generations=0
    )
    evolved = EvolutionaryDNFOptimizer(0, 1, **options).optimize(
        objective, max_population=48, generations=40
    )

    assert evolved.objective_value >= initial.objective_value
    assert evolved.objective_value > 1.98


def test_result_reports_the_scores_of_the_returned_genome():
    """The final result retains its raw score, penalized fitness, and atom counts."""
    optimizer = EvolutionaryDNFOptimizer(
        1,
        1,
        max_conjunctions=2,
        boolean_complexity_penalty=0.2,
        continuous_complexity_penalty=0.3,
        device="cpu",
        seed=8,
    )

    def objective(bounds, active):
        boolean_conditions, continuous_conditions = optimizer.complexity(bounds, active)
        return 1.0 + boolean_conditions.float() + continuous_conditions.float()

    result = optimizer.optimize(objective, max_population=24, generations=8)
    bounds = result.bounds.unsqueeze(0)
    active = result.active_conjunctions.unsqueeze(0)
    bool_count, cont_count = optimizer.complexity(bounds, active)
    assert result.boolean_conditions == bool_count.item()
    assert result.continuous_conditions == cont_count.item()
    assert result.objective_value == objective(bounds, active).item()
    assert result.fitness == pytest.approx(
        optimizer.fitness(objective(bounds, active), bounds, active).item()
    )


@pytest.mark.parametrize(
    "invalid_output",
    [
        lambda count: torch.ones(count, 1),
        lambda count: torch.ones(count + 1),
        lambda count: torch.tensor(1.0),
        lambda count: torch.full((count,), float("nan")),
        lambda count: torch.full((count,), float("inf")),
        lambda count: torch.zeros(count),
        lambda count: -torch.ones(count),
        lambda count: torch.ones(count, dtype=torch.long),
        lambda count: torch.ones(count, dtype=torch.bool),
        lambda count: torch.ones(count, dtype=torch.complex64),
        lambda count: [1.0] * count,
    ],
)
def test_rejects_invalid_black_box_objective_outputs(invalid_output):
    """Malformed feedback fails at the objective boundary."""
    optimizer = EvolutionaryDNFOptimizer(1, 0, device="cpu")
    with pytest.raises((TypeError, ValueError)):
        optimizer.optimize(
            lambda bounds, _active: invalid_output(bounds.shape[0]),
            max_population=4,
            generations=0,
        )


@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, MemoryError])
def test_objective_oom_retries_smaller_batches(error_type):
    """A black box can consume more memory than the configured estimate."""
    attempted_sizes = []
    successful_sizes = []

    def objective(bounds, _active):
        count = bounds.shape[0]
        attempted_sizes.append(count)
        if count > 3:
            raise error_type("simulated objective workspace exhaustion")
        successful_sizes.append(count)
        return torch.ones(count)

    result = EvolutionaryDNFOptimizer(
        1, 0, device="cpu", objective_batch_size=16, seed=1
    ).optimize(objective, max_population=16, generations=2)

    assert attempted_sizes[0] > 3
    assert successful_sizes
    assert max(successful_sizes) <= 3
    assert result.evaluations == sum(successful_sizes)
    first_success = next(
        index for index, size in enumerate(attempted_sizes) if size <= 3
    )
    assert max(attempted_sizes[first_success:]) <= 3


@pytest.mark.parametrize(
    "message",
    [
        "DefaultCPUAllocator: can't allocate memory: you tried to allocate 8192 bytes.",
        "DefaultCPUAllocator: not enough memory: you tried to allocate 8192 bytes.",
    ],
)
def test_cpu_allocator_runtime_errors_retry_smaller_objective_batches(message):
    """PyTorch reports CPU allocation exhaustion as an ordinary RuntimeError."""
    attempted_sizes = []
    successful_sizes = []

    def objective(bounds, _active):
        count = bounds.shape[0]
        attempted_sizes.append(count)
        if count > 3:
            raise RuntimeError(message)
        successful_sizes.append(count)
        return torch.ones(count)

    result = EvolutionaryDNFOptimizer(
        1, 0, device="cpu", objective_batch_size=16, seed=4
    ).optimize(objective, max_population=16, generations=0)

    assert attempted_sizes[0] > 3
    assert successful_sizes
    assert max(successful_sizes) <= 3
    assert result.evaluations == sum(successful_sizes) == 16


def test_cpu_allocator_runtime_error_at_batch_size_one_is_reported():
    """The original allocator message survives when smaller batches cannot help."""
    attempted_sizes = []
    message = (
        "DefaultCPUAllocator: can't allocate memory: you tried to allocate 8192 bytes."
    )

    def objective(bounds, _active):
        attempted_sizes.append(bounds.shape[0])
        raise RuntimeError(message)

    optimizer = EvolutionaryDNFOptimizer(1, 0, device="cpu", objective_batch_size=4)
    with pytest.raises(RuntimeError, match="DefaultCPUAllocator"):
        optimizer.optimize(objective, max_population=4, generations=0)
    assert attempted_sizes[-1] == 1


@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, MemoryError])
def test_objective_oom_at_batch_size_one_is_reported(error_type):
    """Batching cannot recover when a single candidate exhausts memory."""
    attempted_sizes = []

    def objective(bounds, _active):
        attempted_sizes.append(bounds.shape[0])
        raise error_type("simulated irreducible workspace exhaustion")

    optimizer = EvolutionaryDNFOptimizer(1, 0, device="cpu", objective_batch_size=4)
    with pytest.raises(error_type):
        optimizer.optimize(objective, max_population=4, generations=0)
    assert attempted_sizes[-1] == 1


def test_generation_population_allocation_oom_shrinks_and_preserves_the_elite(
    monkeypatch,
):
    """A failed next-population allocation can recover using fewer candidates."""
    optimizer = EvolutionaryDNFOptimizer(
        1,
        0,
        max_conjunctions=1,
        boolean_complexity_penalty=0,
        mutation_rate=0,
        conjunction_mutation_rate=0,
        device="cpu",
        seed=6,
    )
    allocate = optimizer._empty_population
    allocation_sizes = []

    def allocate_with_pressure(size):
        allocation_sizes.append(size)
        if len(allocation_sizes) > 1 and size > 8:
            raise torch.OutOfMemoryError("simulated next-population allocation failure")
        return allocate(size)

    def objective(bounds, active):
        unrestricted = (bounds[:, 0, 0, 0] == 0) & (bounds[:, 0, 0, 1] == 1)
        return 1.0 + 9.0 * (unrestricted & active[:, 0]).float()

    monkeypatch.setattr(optimizer, "_empty_population", allocate_with_pressure)
    result = optimizer.optimize(objective, max_population=16, generations=1)

    assert allocation_sizes[:2] == [16, 16]
    assert allocation_sizes[-1] <= 8
    assert 2 <= result.population_size <= 8
    assert result.objective_value == result.fitness == 10.0


def test_offspring_workspace_oom_retries_smaller_batches(monkeypatch):
    """Crossover and mutation can recover without losing the population."""
    optimizer = EvolutionaryDNFOptimizer(2, 1, device="cpu", seed=10)
    make_children = optimizer._children
    attempted_sizes = []
    successful_sizes = []

    def children_with_pressure(parents, size, generator, parent_indices):
        attempted_sizes.append(size)
        if size > 3:
            raise torch.OutOfMemoryError("simulated mutation workspace exhaustion")
        successful_sizes.append(size)
        return make_children(parents, size, generator, parent_indices)

    monkeypatch.setattr(optimizer, "_children", children_with_pressure)
    result = optimizer.optimize(_constant_objective, max_population=16, generations=1)

    assert attempted_sizes[0] > 3
    assert successful_sizes
    assert max(successful_sizes) <= 3
    assert result.population_size == 16
    assert result.objective_value == 1.0


def test_initialization_workspace_oom_recovers_before_objective_evaluation(monkeypatch):
    """Initialization retries allocator failures before exposing any candidates."""
    optimizer = EvolutionaryDNFOptimizer(1, 1, device="cpu", seed=2)
    random_bounds = optimizer._random_bounds
    attempted_sizes = []
    scored_sizes = []

    def bounds_with_pressure(size, generator):
        attempted_sizes.append(size)
        if size > 7:
            raise torch.OutOfMemoryError(
                "simulated initialization workspace exhaustion"
            )
        return random_bounds(size, generator)

    def objective(bounds, _active):
        scored_sizes.append(bounds.shape[0])
        return torch.ones(bounds.shape[0])

    monkeypatch.setattr(optimizer, "_random_bounds", bounds_with_pressure)
    result = optimizer.optimize(objective, max_population=16, generations=0)

    assert attempted_sizes[0] > 7
    assert attempted_sizes[-1] <= 7
    assert 2 <= result.population_size <= 16
    assert result.evaluations == sum(scored_sizes) == result.population_size


def test_unrelated_objective_errors_are_not_treated_as_memory_failures():
    """A callback programming error propagates without repeated evaluation."""
    calls = []

    def objective(bounds, _active):
        calls.append(bounds.shape[0])
        raise RuntimeError("environment calculation failed")

    with pytest.raises(RuntimeError, match="environment calculation failed"):
        EvolutionaryDNFOptimizer(1, 0, device="cpu").optimize(
            objective, max_population=8, generations=0
        )
    assert len(calls) == 1


def test_explicit_memory_budget_limits_the_live_population():
    """A requested maximum never forces a population beyond its memory budget."""
    result = EvolutionaryDNFOptimizer(
        32, 32, device="cpu", max_memory_bytes=2_000_000, seed=3
    ).optimize(_constant_objective, max_population=10_000, generations=1)
    assert 2 <= result.population_size < 10_000


def test_population_shrinks_when_available_memory_falls(monkeypatch):
    """The next generation accounts for memory pressure from the environment."""
    calls = []
    optimizer = EvolutionaryDNFOptimizer(2, 1, max_conjunctions=4, device="cpu", seed=9)
    monkeypatch.setattr(
        optimizer, "_available_memory", lambda: 50_000 if calls else 10_000_000
    )

    def objective(bounds, active):
        calls.append(bounds.shape[0])
        return 1.0 + active.float().sum(dim=1)

    result = optimizer.optimize(objective, max_population=48, generations=1)
    assert calls[0] == 48
    assert 2 <= result.population_size < 48
    assert result.evaluations == sum(calls)


def test_objective_batches_recheck_available_memory(monkeypatch):
    """Work batches shrink immediately when the black box reduces free memory."""
    calls = []
    optimizer = EvolutionaryDNFOptimizer(
        1,
        1,
        max_conjunctions=2,
        objective_batch_size=8,
        objective_bytes_per_candidate=10_000,
        device="cpu",
        seed=7,
    )
    monkeypatch.setattr(
        optimizer, "_available_memory", lambda: 40_000 if calls else 10_000_000
    )

    def objective(bounds, _active):
        calls.append(bounds.shape[0])
        return torch.ones(bounds.shape[0])

    result = optimizer.optimize(objective, max_population=24, generations=0)
    assert calls[0] == 8
    assert max(calls[1:]) < 8
    assert result.evaluations == sum(calls) == 24


def test_insufficient_memory_fails_before_calling_the_objective(monkeypatch):
    optimizer = EvolutionaryDNFOptimizer(1, 1, device="cpu")
    monkeypatch.setattr(optimizer, "_available_memory", lambda: 1)
    calls = []

    def objective(bounds, _active):
        calls.append(bounds.shape[0])
        return torch.ones(bounds.shape[0])

    with pytest.raises(MemoryError):
        optimizer.optimize(objective, max_population=24, generations=0)
    assert not calls


@pytest.mark.parametrize("legacy", [False, True])
def test_cpu_memory_respects_parent_cgroup_limits(monkeypatch, legacy):
    """Ancestor limits constrain memory even when a leaf has a looser limit."""
    if legacy:
        root = "/sys/fs/cgroup/memory"
        membership = "5:cpu,memory:/jobs/worker\n"
        limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
    else:
        root = "/sys/fs/cgroup"
        membership = "0::/jobs/worker\n"
        limit_name, usage_name = "memory.max", "memory.current"
    files = {
        "/proc/meminfo": "MemTotal: 16000 kB\nMemAvailable: 8000 kB\n",
        "/proc/self/cgroup": membership,
        f"{root}/{limit_name}": "max",
        f"{root}/{usage_name}": "0",
        f"{root}/jobs/{limit_name}": "6000",
        f"{root}/jobs/{usage_name}": "3000",
        f"{root}/jobs/worker/{limit_name}": "5000",
        f"{root}/jobs/worker/{usage_name}": "1000",
    }

    def read_text(path, **_kwargs):
        if str(path) not in files:
            raise FileNotFoundError(str(path))
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read_text)
    assert evolutionary_dnf._cpu_available_memory() == 3000


def test_cpu_memory_falls_back_to_system_pages(monkeypatch):
    """CPU memory detection still works when procfs and cgroups are absent."""

    def read_text(path, **_kwargs):
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(
        evolutionary_dnf.os,
        "sysconf",
        lambda name: 4096 if name == "SC_PAGE_SIZE" else 1000,
    )
    assert evolutionary_dnf._cpu_available_memory() == 4_096_000


def test_cuda_available_memory_includes_reusable_allocator_cache(monkeypatch):
    """Cached, unallocated CUDA storage contributes to usable device memory."""
    optimizer = EvolutionaryDNFOptimizer(1, 1, device="cuda:0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (100, 1000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 400)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 200)
    assert optimizer._available_memory() == 300


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_boolean_features": -1},
        {"num_continuous_features": -1},
        {"max_conjunctions": 0},
        {"survival_rate": 0.0},
        {"survival_rate": 1.0},
        {"mutation_rate": -0.1},
        {"crossover_rate": 1.1},
        {"conjunction_mutation_rate": float("nan")},
        {"atom_removal_probability": -0.1},
        {"bound_reset_probability": 1.1},
        {"initial_atom_probability": 1.1},
        {"initial_conjunction_probability": -0.1},
        {"crossover_swap_probability": 1.1},
        {"fine_grained_crossover_probability": -0.1},
        {"fine_grained_crossover_probability": 1.1},
        {"selection_threshold": -0.1},
        {"selection_threshold": 1.0},
        {"selection_threshold": float("nan")},
        {"tournament_size": 0},
        {"bound_mutation_scale": -0.1},
        {"boolean_complexity_penalty": -0.1},
        {"continuous_complexity_penalty": -0.1},
        {"memory_fraction": 0.0},
        {"memory_fraction": 1.1},
        {"objective_batch_size": 0},
        {"objective_bytes_per_candidate": -1},
        {"max_memory_bytes": 0},
    ],
)
def test_rejects_invalid_hyperparameters(kwargs):
    """Invalid settings fail before any population allocation."""
    options = {"num_boolean_features": 1, "num_continuous_features": 1, "device": "cpu"}
    options.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        EvolutionaryDNFOptimizer(**options)


@pytest.mark.parametrize("kwargs", [{"max_population": 1}, {"generations": -1}])
def test_rejects_invalid_run_parameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        EvolutionaryDNFOptimizer(1, 0, device="cpu").optimize(
            _constant_objective, **kwargs
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_objective_and_result_stay_on_the_requested_device():
    """CUDA operators pass device-local batches directly to the objective."""
    seen_devices = []

    def objective(bounds, active):
        seen_devices.append(bounds.device)
        assert active.is_cuda
        return 1.0 + active.float().sum(dim=1)

    result = EvolutionaryDNFOptimizer(2, 1, device="cuda", seed=5).optimize(
        objective, max_population=24, generations=3
    )
    assert seen_devices
    assert all(device.type == "cuda" for device in seen_devices)
    assert result.bounds.is_cuda
    assert result.active_conjunctions.is_cuda
