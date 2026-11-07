"""Behavioral tests for selection, crossover, and endpoint-level DNF mutation."""

import pytest
import torch

from algorithms.evolutionary_dnf import EvolutionaryDNFOptimizer, _Population


def _population(scores):
    """Give each candidate a distinct lower bound and intentionally opposite raw rank."""
    size = len(scores)
    bounds = torch.ones(size, 1, 1, 2)
    bounds[:, 0, 0, 0] = torch.arange(size) / (size + 1)
    return _Population(
        bounds,
        torch.ones(size, 1, dtype=torch.bool),
        torch.arange(size, 0, -1, dtype=torch.float64),
        torch.tensor(scores, dtype=torch.float64),
    )


def _generator(seed=42):
    return torch.Generator(device="cpu").manual_seed(seed)


def test_divisive_fitness_preserves_positive_scores_and_objective_scaling():
    """Penalty weights are dimensionless and do not depend on objective units."""
    optimizer = EvolutionaryDNFOptimizer(
        1,
        1,
        max_conjunctions=1,
        boolean_complexity_penalty=0.5,
        continuous_complexity_penalty=0.25,
        device="cpu",
    )
    bounds = torch.tensor(
        [
            [[[1.0, 1.0], [0.2, 0.8]]],
            [[[0.0, 1.0], [0.0, 1.0]]],
        ]
    )
    active = torch.ones(2, 1, dtype=torch.bool)
    objective = torch.tensor([1.0e-8, 1.0e-8], dtype=torch.float64)
    scores = optimizer.fitness(objective, bounds, active)
    assert scores.tolist() == pytest.approx([5.0e-9, 1.0e-8], abs=1.0e-15)
    assert (scores > 0).all()
    assert torch.allclose(
        optimizer.fitness(objective * 1.0e12, bounds, active), scores * 1.0e12
    )
    assert optimizer.fitness(objective, bounds, torch.zeros_like(active)).tolist() == (
        objective.tolist()
    )


@pytest.mark.parametrize(
    "scores,threshold,expected",
    [
        ([1, 2, 3, 4, 5], 0.2, {1, 2, 3, 4}),
        ([5, 1, 4, 2, 3], 0.4, {0, 2, 4}),
        ([1, 2, 3, 4], 0.2, {0, 1, 2, 3}),
        ([1, 2, 3, 4, 5], 0.0, {0, 1, 2, 3, 4}),
        ([1, 2, 3, 4, 5], 0.99, {4}),
    ],
)
def test_parent_pool_excludes_the_requested_worst_fraction(scores, threshold, expected):
    """Rank by penalized fitness and round the excluded count down."""
    optimizer = EvolutionaryDNFOptimizer(
        0, 1, max_conjunctions=1, selection_threshold=threshold, device="cpu"
    )
    parents = _population(scores)
    indices = optimizer._parent_indices(parents)
    assert set(indices.tolist()) == expected
    assert torch.equal(parents.scores, torch.tensor(scores, dtype=torch.float64))


def test_tied_parent_scores_keep_the_requested_number_of_candidates():
    """A score tie must not accidentally exclude the entire population."""
    optimizer = EvolutionaryDNFOptimizer(
        0, 1, max_conjunctions=1, selection_threshold=0.4, device="cpu"
    )
    eligible = optimizer._parent_indices(_population([1, 1, 1, 1, 1]))
    assert len(eligible) == len(set(eligible.tolist())) == 3


def test_tournaments_only_sample_the_filtered_parent_pool():
    """Even single-contestant tournaments never admit an excluded candidate."""
    optimizer = EvolutionaryDNFOptimizer(
        0,
        1,
        max_conjunctions=1,
        selection_threshold=0.4,
        tournament_size=1,
        crossover_rate=0,
        mutation_rate=0,
        conjunction_mutation_rate=0,
        device="cpu",
    )
    parents = _population([5, 1, 4, 2, 3])
    eligible = optimizer._parent_indices(parents)
    children, active = optimizer._children(parents, 4096, _generator(), eligible)
    observed = set(children[:, 0, 0, 0].tolist())
    allowed = set(parents.bounds[eligible, 0, 0, 0].tolist())
    assert observed == allowed
    assert active.all()


def test_parent_pool_is_computed_once_per_generation_despite_work_chunking(monkeypatch):
    """Filtering the whole population is not repeated for each offspring batch."""
    optimizer = EvolutionaryDNFOptimizer(1, 1, device="cpu", seed=3)
    filter_parents = optimizer._parent_indices
    batch_limit = optimizer._batch_limit
    pool_sizes = []

    def record_parent_pool(parents):
        pool_sizes.append(parents.size)
        return filter_parents(parents)

    monkeypatch.setattr(optimizer, "_parent_indices", record_parent_pool)
    monkeypatch.setattr(
        optimizer,
        "_batch_limit",
        lambda requested, resident, workspace: min(
            2, batch_limit(requested, resident, workspace)
        ),
    )
    optimizer.optimize(
        lambda bounds, _active: torch.ones(bounds.shape[0]),
        max_population=16,
        generations=3,
    )
    assert pool_sizes == [16, 16, 16]


@pytest.mark.parametrize("copy_genes", [False, True])
def test_crossover_uses_three_exclusive_slot_modes_and_independent_flags(
    monkeypatch, copy_genes
):
    """One roll selects the mode; fine mode can copy genes and flags separately."""
    optimizer = EvolutionaryDNFOptimizer(
        0,
        1,
        max_conjunctions=3,
        crossover_rate=0.8,
        fine_grained_crossover_probability=0.5,
        crossover_swap_probability=0.5,
        device="cpu",
    )
    bounds = torch.tensor([[[[0.2, 0.8]], [[0.2, 0.8]], [[0.2, 0.8]]]])
    other_bounds = torch.tensor([[[[0.4, 0.6]], [[0.4, 0.6]], [[0.4, 0.6]]]])
    active = torch.zeros(1, 3, dtype=torch.bool)
    other_active = torch.ones_like(active)
    first_bounds = bounds.clone()
    draws = []

    def deterministic_rand(*sizes, **kwargs):
        shape = tuple(sizes[0]) if len(sizes) == 1 else tuple(sizes)
        draws.append(shape)
        if len(draws) == 1:
            assert shape == (1, 3)
            return torch.tensor([[0.1, 0.5, 0.9]], device=kwargs.get("device"))
        value = (
            (0.0 if copy_genes else 1.0)
            if len(shape) == 4
            else (1.0 if copy_genes else 0.0)
        )
        return torch.full(shape, value, device=kwargs.get("device"))

    monkeypatch.setattr(torch, "rand", deterministic_rand)
    optimizer._crossover(bounds, active, other_bounds, other_active, _generator())

    assert torch.equal(bounds[:, 0], other_bounds[:, 0])
    assert active[0, 0]
    assert torch.equal(
        bounds[:, 1], (other_bounds if copy_genes else first_bounds)[:, 1]
    )
    assert bool(active[0, 1]) is not copy_genes
    assert torch.equal(bounds[:, 2], first_bounds[:, 2])
    assert not active[0, 2]
    assert draws == [(1, 3), (1, 3, 1, 2), (1, 3)]


def test_fine_crossover_mixes_individual_bounds_and_rejects_crossing_pairs():
    """Incoming endpoints retain their meaning and never form invalid intervals."""
    optimizer = EvolutionaryDNFOptimizer(
        0,
        1,
        max_conjunctions=1,
        crossover_rate=1,
        fine_grained_crossover_probability=1,
        crossover_swap_probability=0.5,
        device="cpu",
    )
    bounds = torch.tensor([0.2, 0.4]).expand(4096, 1, 1, 2).clone()
    other_bounds = torch.tensor([0.6, 0.8]).expand_as(bounds).clone()
    active = torch.zeros(4096, 1, dtype=torch.bool)
    other_active = torch.ones_like(active)
    optimizer._crossover(bounds, active, other_bounds, other_active, _generator(7))

    first = (bounds[..., 0] == 0.2) & (bounds[..., 1] == 0.4)
    mixed = (bounds[..., 0] == 0.2) & (bounds[..., 1] == 0.8)
    second = (bounds[..., 0] == 0.6) & (bounds[..., 1] == 0.8)
    assert (first | mixed | second).all()
    assert first.any() and mixed.any() and second.any()
    for kind in (first, mixed, second):
        flags = active[kind[..., 0]]
        assert flags.any() and (~flags).any()


@pytest.mark.parametrize("operation", ["remove", "reset", "perturb"])
def test_mutation_selects_each_endpoint_independently(operation):
    """Lower-only, upper-only, both, and neither all occur at independent rates."""
    optimizer = EvolutionaryDNFOptimizer(
        0,
        1,
        max_conjunctions=1,
        mutation_rate=0.5,
        conjunction_mutation_rate=0,
        atom_removal_probability=1.0 if operation == "remove" else 0.0,
        bound_reset_probability=1.0 if operation == "reset" else 0.0,
        bound_mutation_scale=0.05,
        device="cpu",
    )
    original = torch.tensor([0.3, 0.7]).expand(4096, 1, 1, 2).clone()
    bounds = original.clone()
    active = torch.ones(4096, 1, dtype=torch.bool)
    optimizer._mutate(bounds, active, _generator(19))

    changed = bounds != original
    lower, upper = changed[..., 0], changed[..., 1]
    combinations = torch.stack(
        (~lower & ~upper, lower & ~upper, ~lower & upper, lower & upper)
    )
    assert combinations.float().mean(dim=(1, 2, 3)).tolist() == pytest.approx(
        [0.25] * 4, abs=0.04
    )
    assert (bounds[..., 0] <= bounds[..., 1]).all()
    assert ((bounds >= 0) & (bounds <= 1)).all()
    assert active.all()
    if operation == "remove":
        assert ((bounds[..., 0] == 0.0) | (bounds[..., 0] == 0.3)).all()
        assert ((bounds[..., 1] == 1.0) | (bounds[..., 1] == 0.7)).all()


def test_independent_boolean_endpoint_mutations_preserve_the_three_valid_states():
    """Conflicting proposals resolve without a systematic true/false preference."""
    optimizer = EvolutionaryDNFOptimizer(
        1,
        0,
        max_conjunctions=1,
        mutation_rate=1,
        conjunction_mutation_rate=0,
        atom_removal_probability=0,
        device="cpu",
    )
    bounds = torch.tensor([0.0, 1.0]).expand(8192, 1, 1, 2).clone()
    active = torch.ones(8192, 1, dtype=torch.bool)
    optimizer._mutate(bounds, active, _generator(83))
    low, high = bounds[..., 0], bounds[..., 1]
    false = (low == 0) & (high == 0)
    unrestricted = (low == 0) & (high == 1)
    true = (low == 1) & (high == 1)
    assert (false | unrestricted | true).all()
    assert unrestricted.float().mean().item() > 0.15
    assert false.float().mean().item() == pytest.approx(
        true.float().mean().item(), abs=0.04
    )


def test_mutation_repairs_crossings_without_changing_an_unselected_endpoint(
    monkeypatch,
):
    """A large one-sided proposal must preserve the opposite, unselected bound."""
    optimizer = EvolutionaryDNFOptimizer(
        0,
        1,
        max_conjunctions=1,
        mutation_rate=0.5,
        conjunction_mutation_rate=0,
        atom_removal_probability=0,
        bound_reset_probability=0,
        bound_mutation_scale=1,
        device="cpu",
    )
    bounds = torch.tensor([0.3, 0.7]).expand(3, 1, 1, 2).clone()
    active = torch.ones(3, 1, dtype=torch.bool)
    draw_count = 0

    def deterministic_rand(*sizes, **kwargs):
        nonlocal draw_count
        draw_count += 1
        shape = tuple(sizes[0]) if len(sizes) == 1 else tuple(sizes)
        if draw_count == 4:
            return torch.tensor(
                [[[[0.0, 1.0]]], [[[1.0, 0.0]]], [[[0.0, 0.0]]]],
                device=kwargs.get("device"),
            )
        return torch.ones(shape, device=kwargs.get("device"))

    def deterministic_noise(shape, **kwargs):
        return torch.tensor([1.0, -1.0], device=kwargs.get("device")).expand(shape)

    monkeypatch.setattr(torch, "rand", deterministic_rand)
    monkeypatch.setattr(torch, "randn", deterministic_noise)
    optimizer._mutate(bounds, active, _generator())

    torch.testing.assert_close(
        bounds[:, 0, 0], torch.tensor([[0.7, 0.7], [0.3, 0.3], [0.0, 1.0]])
    )
    assert active.all()
