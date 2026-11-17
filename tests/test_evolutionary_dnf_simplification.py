"""Optional post-evolution pruning accepts the best strict fitness improvement."""

import pytest
import torch

from ifera.algorithms.evolutionary_dnf import EvolutionaryDNFOptimizer


class _FrozenOptimizer(EvolutionaryDNFOptimizer):
    """Start each population row at an explicit genome without seeded constants."""

    def __init__(self, bounds, active, boolean_count=0, **parameters):
        self.initial_bounds = torch.as_tensor(bounds, dtype=torch.float32).clone()
        self.initial_active = torch.as_tensor(active, dtype=torch.bool).clone()
        parameters.setdefault("device", "cpu")
        parameters.setdefault("seed", 17)
        super().__init__(
            boolean_count,
            self.initial_bounds.shape[1] - boolean_count,
            max_conjunctions=self.initial_bounds.shape[0],
            **parameters,
        )
        self.initial_population = None

    def _initialize(self, population, _generator):
        population.bounds.copy_(self.initial_bounds.to(self.device))
        population.active.copy_(self.initial_active.to(self.device))
        self.initial_population = population


def _optimizer(bounds, active=None, boolean_count=0, **parameters):
    if active is None:
        active = [True] * len(bounds)
    parameters.setdefault("simplify", True)
    return _FrozenOptimizer(bounds, active, boolean_count, **parameters)


def _constant(bounds, _active):
    return torch.ones(bounds.shape[0], device=bounds.device)


def _assert_result_matches_objective(optimizer, result, objective=_constant):
    bounds = result.bounds[None]
    active = result.active_conjunctions[None]
    values = objective(bounds, active)
    boolean, continuous = optimizer.complexity(bounds, active)
    assert result.objective_value == float(values[0])
    assert result.fitness == float(optimizer.fitness(values, bounds, active)[0])
    assert result.boolean_conditions == int(boolean[0])
    assert result.continuous_conditions == int(continuous[0])


def test_generic_optimizer_keeps_simplification_disabled_by_default():
    optimizer = _FrozenOptimizer([[[1.0, 1.0]]], [True], boolean_count=1)
    result = optimizer.optimize(_constant, max_population=4, generations=0)

    torch.testing.assert_close(result.bounds, optimizer.initial_bounds)
    assert result.evaluations == 4
    assert result.simplification_steps == result.simplification_evaluations == 0


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_boolean_literal_is_removed_as_a_whole_interval(value):
    source = torch.tensor([[[value, value]]])
    original = source.clone()
    optimizer = _optimizer(source, boolean_count=1)
    result = optimizer.optimize(_constant, max_population=4, generations=0)

    assert result.bounds.tolist() == [[[0.0, 1.0]]]
    assert result.active_conjunctions.tolist() == [True]
    assert result.boolean_conditions == 0
    assert result.fitness == 1.0
    assert result.simplification_steps == 1
    assert result.simplification_evaluations == 1
    assert result.evaluations == 5
    torch.testing.assert_close(source, original)
    assert optimizer.initial_population is not None
    torch.testing.assert_close(
        optimizer.initial_population.bounds, original.expand(4, -1, -1, -1)
    )
    _assert_result_matches_objective(optimizer, result)


def test_continuous_restrictions_are_removed_one_endpoint_at_a_time():
    optimizer = _optimizer([[[0.25, 0.75]]], continuous_complexity_penalty=1.0)
    observed = []

    def objective(bounds, active):
        observed.extend(bounds.detach().cpu().tolist())
        return _constant(bounds, active)

    result = optimizer.optimize(objective, max_population=4, generations=0)

    assert result.bounds.tolist() == [[[0.0, 1.0]]]
    neighbors = observed[4:]
    assert len(neighbors) == 3
    assert sorted(neighbors[:2]) == sorted([[[[0.0, 0.75]]], [[[0.25, 1.0]]]])
    assert neighbors[-1] == [[[0.0, 1.0]]]
    assert result.simplification_steps == 2
    assert result.simplification_evaluations == 3
    assert result.evaluations == 7
    assert result.continuous_conditions == 0
    _assert_result_matches_objective(optimizer, result)


def test_each_pass_scores_all_neighbors_and_accepts_global_best_then_repeats():
    optimizer = _optimizer(
        [[[0.2, 1.0], [0.3, 1.0], [0.4, 1.0]]],
        continuous_complexity_penalty=0.0,
    )
    observed_masks = []
    # Removing A first traps a first-improvement search at 15. Best removal B
    # reaches 20, then removing C reaches 30, where the final removal would lose.
    scores = [10.0, 15.0, 20.0, 14.0, 11.0, 14.0, 30.0, 29.0]

    def objective(bounds, _active):
        weights = torch.tensor([1, 2, 4], device=bounds.device)
        removed = bounds[:, 0, :, 0] == 0
        masks = (removed.long() * weights).sum(dim=1)
        observed_masks.extend(masks.cpu().tolist())
        return bounds.new_tensor(scores)[masks]

    result = optimizer.optimize(objective, max_population=4, generations=0)

    assert sorted(observed_masks[4:7]) == [1, 2, 4]
    assert sorted(observed_masks[7:9]) == [3, 6]
    assert observed_masks[9:] == [7]
    torch.testing.assert_close(
        result.bounds, torch.tensor([[[0.2, 1.0], [0.0, 1.0], [0.0, 1.0]]])
    )
    assert result.objective_value == result.fitness == 30.0
    assert result.simplification_steps == 2
    assert result.simplification_evaluations == 6
    assert result.evaluations == 10
    assert result.continuous_conditions == 1
    _assert_result_matches_objective(optimizer, result, objective)


def test_lower_raw_objective_is_accepted_when_penalized_fitness_improves():
    optimizer = _optimizer([[[0.4, 1.0]]], continuous_complexity_penalty=1.0)

    def objective(bounds, _active):
        return torch.where(bounds[:, 0, 0, 0] == 0, 9.0, 10.0)

    result = optimizer.optimize(objective, max_population=4, generations=0)

    assert result.objective_value == result.fitness == 9.0
    assert result.bounds.tolist() == [[[0.0, 1.0]]]
    assert result.simplification_steps == 1
    _assert_result_matches_objective(optimizer, result, objective)


@pytest.mark.parametrize("penalty,restricted_value", [(0.0, 1.0), (0.5, 1.5)])
def test_equal_fitness_is_not_an_improvement(penalty, restricted_value):
    optimizer = _optimizer([[[0.4, 1.0]]], continuous_complexity_penalty=penalty)

    def objective(bounds, _active):
        return torch.where(bounds[:, 0, 0, 0] == 0, 1.0, restricted_value)

    result = optimizer.optimize(objective, max_population=4, generations=0)

    torch.testing.assert_close(result.bounds, optimizer.initial_bounds)
    assert result.fitness == 1.0
    assert result.simplification_steps == 0
    assert result.simplification_evaluations == 1
    assert result.evaluations == 5
    _assert_result_matches_objective(optimizer, result, objective)


@pytest.mark.parametrize("active", [[True, False], [False, False]])
def test_inactive_clauses_and_unrestricted_features_do_not_generate_neighbors(active):
    optimizer = _optimizer(
        [[[0.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [0.2, 0.8]]],
        active,
        boolean_count=1,
    )
    result = optimizer.optimize(_constant, max_population=4, generations=0)

    torch.testing.assert_close(result.bounds, optimizer.initial_bounds)
    assert result.active_conjunctions.tolist() == active
    assert result.simplification_steps == result.simplification_evaluations == 0
    assert result.evaluations == 4


def test_no_features_has_no_simplification_neighbors():
    optimizer = _optimizer(torch.empty(1, 0, 2))
    result = optimizer.optimize(_constant, max_population=4, generations=0)
    assert result.bounds.shape == (1, 0, 2)
    assert result.evaluations == 4
    assert result.simplification_steps == result.simplification_evaluations == 0


def test_active_flags_and_inactive_restrictions_are_preserved_across_multiple_clauses():
    initial = torch.tensor(
        [
            [[1.0, 1.0], [0.2, 0.8]],
            [[0.0, 0.0], [0.3, 0.7]],
            [[0.0, 0.0], [0.4, 1.0]],
        ]
    )
    optimizer = _optimizer(
        initial, [True, False, True], boolean_count=1, objective_batch_size=3
    )
    result = optimizer.optimize(_constant, max_population=4, generations=0)

    assert result.active_conjunctions.tolist() == [True, False, True]
    torch.testing.assert_close(result.bounds[1], initial[1])
    assert result.bounds[[0, 2], :, 0].eq(0).all()
    assert result.bounds[[0, 2], :, 1].eq(1).all()
    assert result.simplification_steps == 5
    assert result.simplification_evaluations == 15
    assert result.evaluations == 19
    assert result.boolean_conditions == result.continuous_conditions == 0
    _assert_result_matches_objective(optimizer, result)


def test_required_boolean_literal_is_preserved_while_another_condition_is_removed():
    optimizer = _optimizer([[[0.0, 0.0], [0.4, 1.0]]], boolean_count=1)

    def objective(bounds, _active):
        unchanged_false = (bounds[:, 0, 0, 0] == 0) & (bounds[:, 0, 0, 1] == 0)
        return torch.where(unchanged_false, 10.0, 1.0)

    result = optimizer.optimize(objective, max_population=4, generations=0)

    assert result.bounds.tolist() == [[[0.0, 0.0], [0.0, 1.0]]]
    assert result.boolean_conditions == 1
    assert result.continuous_conditions == 0
    assert result.simplification_steps == 1
    _assert_result_matches_objective(optimizer, result, objective)


def test_callbacks_only_observe_ga_generations_and_retained_storage_stays_unchanged():
    optimizer = _optimizer(
        [[[1.0, 1.0], [0.25, 0.75]]],
        boolean_count=1,
        mutation_rate=0,
        conjunction_mutation_rate=0,
        crossover_rate=0,
    )
    callbacks = []
    successful_sizes = []

    def objective(bounds, active):
        successful_sizes.append(bounds.shape[0])
        return _constant(bounds, active)

    def observe(generation, bounds, active, _objectives, _fitness):
        callbacks.append((generation, bounds, bounds.clone(), active, active.clone()))

    optimizer.set_evaluation_callback(observe)
    result = optimizer.optimize(objective, max_population=6, generations=2)

    assert [record[0] for record in callbacks] == [0, 1, 2]
    for _generation, bounds, original_bounds, active, original_active in callbacks:
        torch.testing.assert_close(bounds, original_bounds)
        torch.testing.assert_close(active, original_active)
        assert bounds.shape[0] == 6
    assert result.generations == 2
    assert result.population_size == 6
    assert result.simplification_steps == 3
    assert result.simplification_evaluations == 6
    assert result.evaluations == sum(successful_sizes)
    assert result.evaluations == 22
    _assert_result_matches_objective(optimizer, result)


@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, MemoryError])
def test_simplification_oom_retries_and_reuses_reduced_batch_cap(error_type):
    initial = torch.tensor([[[0.2, 1.0]] * 6])
    optimizer = _optimizer(initial, objective_batch_size=8)
    attempts, successful_sizes = [], []

    def objective(bounds, active):
        is_trial = bool((bounds != initial).any())
        if is_trial:
            attempts.append(bounds.shape[0])
            if bounds.shape[0] > 2:
                raise error_type("simulated trial objective workspace exhaustion")
        successful_sizes.append(bounds.shape[0])
        return _constant(bounds, active)

    result = optimizer.optimize(objective, max_population=4, generations=0)

    assert attempts[0] > 2
    first_success = next(index for index, size in enumerate(attempts) if size <= 2)
    assert max(attempts[first_success:]) <= 2
    assert result.simplification_steps == 6
    assert result.simplification_evaluations == 21
    assert result.evaluations == sum(successful_sizes) == 25
    assert result.population_size == 4
    assert result.generations == 0
    _assert_result_matches_objective(optimizer, result)


@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, MemoryError])
def test_irreducible_single_trial_oom_propagates(error_type):
    optimizer = _optimizer([[[0.2, 1.0]]])
    initial = optimizer.initial_bounds
    attempts = []

    def objective(bounds, active):
        if bool((bounds != initial).any()):
            attempts.append(bounds.shape[0])
            raise error_type("irreducible simplification trial failure")
        return _constant(bounds, active)

    with pytest.raises(error_type, match="irreducible simplification"):
        optimizer.optimize(objective, max_population=4, generations=0)
    assert attempts[-1] == 1


@pytest.mark.parametrize(
    "invalid_output",
    [
        lambda bounds: torch.ones(bounds.shape[0], 1),
        lambda bounds: torch.ones(bounds.shape[0], dtype=torch.int64),
        lambda bounds: torch.zeros(bounds.shape[0]),
        lambda bounds: torch.full((bounds.shape[0],), float("nan")),
        lambda bounds: [1.0] * bounds.shape[0],
    ],
)
def test_invalid_trial_objective_outputs_propagate(invalid_output):
    optimizer = _optimizer([[[0.2, 1.0]]])
    initial = optimizer.initial_bounds

    def objective(bounds, active):
        if bool((bounds != initial).any()):
            return invalid_output(bounds)
        return _constant(bounds, active)

    with pytest.raises((TypeError, ValueError)):
        optimizer.optimize(objective, max_population=4, generations=0)


def test_unrelated_trial_runtime_error_is_not_retried():
    optimizer = _optimizer([[[0.2, 1.0]]])
    initial = optimizer.initial_bounds
    trial_calls = []

    def objective(bounds, active):
        if bool((bounds != initial).any()):
            trial_calls.append(bounds.shape[0])
            raise RuntimeError("trial environment calculation failed")
        return _constant(bounds, active)

    with pytest.raises(RuntimeError, match="trial environment calculation failed"):
        optimizer.optimize(objective, max_population=4, generations=0)
    assert trial_calls == [1]


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_simplification_option_requires_boolean(value):
    with pytest.raises(ValueError, match="simplify"):
        EvolutionaryDNFOptimizer(1, 0, simplify=value, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_simplification_matches_cpu():
    initial = [[[1.0, 1.0], [0.25, 0.75]]]
    cpu = _optimizer(initial, boolean_count=1).optimize(
        _constant, max_population=4, generations=0
    )
    cuda_optimizer = _optimizer(initial, boolean_count=1, device="cuda")
    cuda = cuda_optimizer.optimize(_constant, max_population=4, generations=0)
    torch.testing.assert_close(cuda.bounds.cpu(), cpu.bounds)
    torch.testing.assert_close(cuda.active_conjunctions.cpu(), cpu.active_conjunctions)
    assert cuda.objective_value == cpu.objective_value
    assert cuda.fitness == cpu.fitness
    assert cuda.simplification_steps == cpu.simplification_steps
    assert cuda.simplification_evaluations == cpu.simplification_evaluations
    assert cuda.evaluations == cpu.evaluations
