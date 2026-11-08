"""Tests for observing evaluated DNF populations without copying their tensors."""

import pytest
import torch

from ifera.algorithms.evolutionary_dnf import EvolutionaryDNFOptimizer


def _quality(bounds, active):
    """Give every genome a deterministic, positive device-local objective."""
    return 1.0 + active.float().sum(1) + 0.001 * bounds.sum((1, 2, 3))


def test_callback_observes_each_complete_population_without_tensor_copies(monkeypatch):
    """All callback tensors are the actual fully scored population buffers."""
    optimizer = EvolutionaryDNFOptimizer(
        2, 1, max_conjunctions=3, objective_batch_size=3, device="cpu", seed=9
    )
    allocate = optimizer._empty_population
    current = {}
    calls = []
    objective_batches = []

    def record_population(size):
        population = allocate(size)
        current["population"] = population
        return population

    def objective(bounds, active):
        objective_batches.append(bounds.shape[0])
        return _quality(bounds, active)

    def callback(generation, bounds, active, objective_values, scores):
        population = current["population"]
        assert bounds is population.bounds
        assert active is population.active
        assert objective_values is population.objectives
        assert scores is population.scores
        assert bounds.shape == (population.size, 3, 3, 2)
        assert active.shape == (population.size, 3)
        assert objective_values.shape == scores.shape == (population.size,)
        assert bounds.dtype == torch.float32
        assert active.dtype == torch.bool
        assert objective_values.dtype == scores.dtype == torch.float64
        assert all(
            tensor.device.type == "cpu"
            for tensor in (bounds, active, objective_values, scores)
        )
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(objective_values, _quality(bounds, active).double())
        torch.testing.assert_close(
            scores, optimizer.fitness(objective_values, bounds, active)
        )
        calls.append((generation, population.size, scores.max().item()))

    monkeypatch.setattr(optimizer, "_empty_population", record_population)
    assert optimizer.set_evaluation_callback(callback) is None
    result = optimizer.optimize(objective, max_population=12, generations=3)

    assert [generation for generation, _size, _score in calls] == [0, 1, 2, 3]
    assert [size for _generation, size, _score in calls] == [12] * 4
    assert len(objective_batches) > len(calls)
    assert result.evaluations == sum(objective_batches)
    assert [score for _generation, _size, score in calls] == sorted(
        score for _generation, _size, score in calls
    )
    assert result.fitness == calls[-1][2]


def test_callback_setter_replaces_disables_and_preserves_registration_across_runs():
    """Setter changes take effect and each run restarts generation numbering."""
    optimizer = EvolutionaryDNFOptimizer(1, 0, device="cpu", seed=5)
    first_calls = []
    second_calls = []

    def first_callback(generation, *_tensors):
        first_calls.append(generation)

    def second_callback(generation, *_tensors):
        second_calls.append(generation)

    optimizer.set_evaluation_callback(first_callback)
    optimizer.set_evaluation_callback(second_callback)
    optimizer.optimize(_quality, max_population=8, generations=1)
    optimizer.optimize(_quality, max_population=8, generations=0)
    assert first_calls == []
    assert second_calls == [0, 1, 0]

    optimizer.set_evaluation_callback(None)
    optimizer.optimize(_quality, max_population=8, generations=2)
    assert second_calls == [0, 1, 0]

    optimizer.set_evaluation_callback(first_callback)
    optimizer.optimize(_quality, max_population=8, generations=0)
    assert first_calls == [0]


def test_callback_can_disable_itself_during_an_optimization_run():
    optimizer = EvolutionaryDNFOptimizer(1, 0, device="cpu", seed=5)
    generations = []

    def callback(generation, *_tensors):
        generations.append(generation)
        optimizer.set_evaluation_callback(None)

    optimizer.set_evaluation_callback(callback)
    optimizer.optimize(_quality, max_population=8, generations=3)
    assert generations == [0]


@pytest.mark.parametrize("invalid_callback", [0, False, "callback", object()])
def test_callback_setter_rejects_noncallables_without_replacing_valid_callback(
    invalid_callback,
):
    optimizer = EvolutionaryDNFOptimizer(1, 0, device="cpu")
    generations = []
    optimizer.set_evaluation_callback(
        lambda generation, *_tensors: generations.append(generation)
    )
    with pytest.raises(TypeError):
        optimizer.set_evaluation_callback(invalid_callback)
    optimizer.optimize(_quality, max_population=4, generations=0)
    assert generations == [0]


@pytest.mark.parametrize(
    "error_type", [RuntimeError, MemoryError, torch.OutOfMemoryError]
)
def test_callback_errors_propagate_without_objective_or_observer_retries(error_type):
    """Observer failures do not trigger the objective's allocation recovery."""
    optimizer = EvolutionaryDNFOptimizer(1, 0, objective_batch_size=3, device="cpu")
    error = error_type("observer failed")
    generations = []
    evaluated = []

    def objective(bounds, active):
        evaluated.append(bounds.shape[0])
        return _quality(bounds, active)

    def callback(generation, *_tensors):
        generations.append(generation)
        raise error

    optimizer.set_evaluation_callback(callback)
    with pytest.raises(error_type, match="observer failed") as exception:
        optimizer.optimize(objective, max_population=8, generations=3)
    assert exception.value is error
    assert generations == [0]
    assert sum(evaluated) == 8


@pytest.mark.parametrize("successful_candidates", [3, 8])
def test_callback_does_not_observe_a_failed_or_partially_evaluated_population(
    successful_candidates,
):
    optimizer = EvolutionaryDNFOptimizer(1, 0, objective_batch_size=3, device="cpu")
    evaluated = 0
    generations = []

    def objective(bounds, active):
        nonlocal evaluated
        if evaluated >= successful_candidates:
            raise RuntimeError("environment failed")
        evaluated += bounds.shape[0]
        return _quality(bounds, active)

    optimizer.set_evaluation_callback(
        lambda generation, *_tensors: generations.append(generation)
    )
    with pytest.raises(RuntimeError, match="environment failed"):
        optimizer.optimize(objective, max_population=8, generations=2)
    assert generations == ([] if successful_candidates == 3 else [0])


def test_objective_oom_retries_notify_only_after_the_complete_population_is_scored():
    optimizer = EvolutionaryDNFOptimizer(1, 0, device="cpu", seed=3)
    attempted = []
    evaluated = []
    generations = []

    def objective(bounds, active):
        size = bounds.shape[0]
        attempted.append(size)
        if size > 3:
            raise torch.OutOfMemoryError("environment workspace exhausted")
        evaluated.append(size)
        return _quality(bounds, active)

    def callback(generation, bounds, active, objective_values, scores):
        generations.append(generation)
        assert bounds.shape[0] == active.shape[0] == objective_values.numel() == 12
        torch.testing.assert_close(objective_values, _quality(bounds, active).double())
        torch.testing.assert_close(
            scores, optimizer.fitness(objective_values, bounds, active)
        )

    optimizer.set_evaluation_callback(callback)
    result = optimizer.optimize(objective, max_population=12, generations=1)
    assert max(attempted) > 3
    assert max(evaluated) <= 3
    assert result.evaluations == sum(evaluated)
    assert generations == [0, 1]


def test_callback_observes_the_effective_population_after_memory_shrinking(monkeypatch):
    optimizer = EvolutionaryDNFOptimizer(1, 1, device="cpu", seed=4)
    population_limit = optimizer._population_limit
    generations = []

    def shrinking_limit(requested, resident_bytes=0, selection_bytes=0):
        size = population_limit(requested, resident_bytes, selection_bytes)
        return min(size, 4 if resident_bytes else 12)

    def callback(generation, bounds, active, objective_values, scores):
        size = bounds.shape[0]
        assert active.shape[0] == objective_values.shape[0] == scores.shape[0] == size
        generations.append((generation, size))

    monkeypatch.setattr(optimizer, "_population_limit", shrinking_limit)
    optimizer.set_evaluation_callback(callback)
    result = optimizer.optimize(_quality, max_population=12, generations=2)
    assert generations == [(0, 12), (1, 4), (2, 4)]
    assert result.population_size == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_callback_receives_device_local_cuda_population_tensors():
    optimizer = EvolutionaryDNFOptimizer(1, 1, device="cuda", seed=3)
    generations = []

    def callback(generation, bounds, active, objective_values, scores):
        assert all(
            tensor.is_cuda for tensor in (bounds, active, objective_values, scores)
        )
        generations.append(generation)

    optimizer.set_evaluation_callback(callback)
    optimizer.optimize(_quality, max_population=8, generations=1)
    assert generations == [0, 1]
