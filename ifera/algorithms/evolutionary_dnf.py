"""Evolve disjunctive normal form (DNF) predicates using a black-box environment.

The objective has signature ``objective(bounds, active_conjunctions) -> values``:

* ``bounds`` is float32, shaped ``(batch, max_conjunctions, features, 2)``.
  Boolean features come first, followed by normalized continuous features.
  The final axis holds inclusive lower/upper bounds. Boolean bounds are exactly
  ``[0, 0]`` (false), ``[1, 1]`` (true), or ``[0, 1]`` (unrestricted).
* ``active_conjunctions`` is Boolean, shaped ``(batch, max_conjunctions)``.
  Features within a conjunction are ANDed; active conjunctions are ORed.
  Inactive conjunctions are ignored. No active conjunctions means FALSE; an
  active conjunction with no restricted features means TRUE.
* ``values`` must be a floating tensor shaped ``(batch,)`` on the same device,
  containing finite, strictly positive objective values to maximize.

The objective must score candidates independently of batch size and order, and
must neither mutate nor retain its input tensors. Deterministic scores preserve
the meaning of elitism. It alone accesses observations or feature distributions.
Optimization runs without autograd. Results and callback inputs use the selected
device; no feature data is copied by the optimizer.

For example, an environment with ``features`` shaped ``(samples, features)``
can compute membership (chunk its own samples if necessary)::

    inside = ((features[None, None] >= bounds[:, :, None, :, 0])
              & (features[None, None] <= bounds[:, :, None, :, 1])).all(-1)
    matches = (inside & active_conjunctions[:, :, None]).any(1)
    return 1.0 + matches.float().mean(1)

Fitness is ``objective / (1 + boolean_penalty * boolean_conditions
                          + continuous_penalty * continuous_conditions)``.
A Boolean literal costs one condition. Each nontrivial continuous bound costs
one inequality, so an interior interval costs two. Inactive clauses cost nothing.
This counts syntactic conditions, without logical minimization. Penalties are
dimensionless: rescaling all objective values preserves the fitness ranking.

An optional observer can be installed with ``set_evaluation_callback(callback)``.
After each complete population evaluation it receives
``callback(generation, bounds, active, objective, score)``. Generation zero is the
initial population; subsequent generations are numbered one through the requested
count. The tensors are the population's actual storage, without copies or device
transfers. Treat them as read-only; retaining them keeps that storage alive.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch import Tensor

Objective = Callable[[Tensor, Tensor], Tensor]
EvaluationCallback = Callable[[int, Tensor, Tensor, Tensor, Tensor], None]


def _is_memory_error(error: Exception) -> bool:
    """Recognize allocator failures, including PyTorch's CPU RuntimeError form."""
    return isinstance(error, (MemoryError, torch.OutOfMemoryError)) or any(
        marker in str(error)
        for marker in (
            "DefaultCPUAllocator: can't allocate memory:",
            "DefaultCPUAllocator: not enough memory:",
        )
    )


def _read_memory_integer(path: Path) -> Optional[int]:
    """Read a cgroup byte counter, treating missing/unlimited counters as absent."""
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _cpu_available_memory() -> int:
    """Read available Linux RAM, including cgroup v1/v2 ancestor limits."""
    available = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except OSError:
        pass

    roots = {Path("/sys/fs/cgroup"): ("memory.max", "memory.current")}
    legacy_root = Path("/sys/fs/cgroup/memory")
    roots[legacy_root] = ("memory.limit_in_bytes", "memory.usage_in_bytes")
    try:
        memberships = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        memberships = []
    for root, (limit_name, usage_name) in roots.items():
        directories = {root}
        for membership in memberships:
            _, controllers, relative = membership.split(":", 2)
            if (root == legacy_root and "memory" in controllers.split(",")) or (
                root != legacy_root and not controllers
            ):
                directory = root / relative.lstrip("/")
                while directory != root and root in directory.parents:
                    directories.add(directory)
                    directory = directory.parent
        for directory in directories:
            limit = _read_memory_integer(directory / limit_name)
            usage = _read_memory_integer(directory / usage_name)
            if limit is not None and usage is not None:
                available = min(available, max(0, limit - usage))
    return max(0, available)


@dataclass(frozen=True)
class DNFOptimizationResult:
    """Best penalized candidate, with unbatched tensors owned by the result.

    ``generations`` counts evolutionary updates after the initial evaluation;
    ``evaluations`` counts successfully scored candidates, excluding OOM retries.
    ``population_size`` is the final effective population, which may be smaller
    than the requested maximum. Memory use does not grow with generation count.
    Optional final simplification adds its trials to ``evaluations`` and reports
    accepted removals and trial counts separately; it does not add generations.
    """

    bounds: Tensor
    active_conjunctions: Tensor
    objective_value: float
    fitness: float
    boolean_conditions: int
    continuous_conditions: int
    population_size: int
    generations: int
    evaluations: int
    simplification_steps: int = 0
    simplification_evaluations: int = 0


@dataclass
class _Population:
    """Contiguous candidate storage and cached objective/fitness values."""

    bounds: Tensor
    active: Tensor
    objectives: Tensor
    scores: Tensor

    @property
    def size(self) -> int:
        """Return the number of candidates."""
        return self.bounds.shape[0]

    @property
    def nbytes(self) -> int:
        """Return live tensor storage in bytes."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.bounds, self.active, self.objectives, self.scores)
        )


class EvolutionaryDNFOptimizer:
    """Vectorized filtered tournaments, mixed crossover, mutation, and elitism.

    Only feature counts are required. Before tournaments, ``selection_threshold``
    excludes the worst ``floor(population_size * selection_threshold)`` candidates
    by penalized fitness. At least one parent remains; ties are resolved by top-k.

    Each conjunction slot uses one roll to choose unchanged (``1-crossover_rate``),
    whole-clause copy (``crossover_rate * (1-fine_grained_crossover_probability)``),
    or fine crossover (``crossover_rate * fine_grained_crossover_probability``).
    During fine crossover, each endpoint and the activation flag independently
    copy from the second parent with ``crossover_swap_probability``. Conflicting
    mixed bounds restore that feature's first-parent pair. ``crossover_rate``
    applies per slot, and ``crossover_swap_probability`` applies to fine genes.

    ``mutation_rate`` selects lower and upper bounds independently. Given a bound
    mutation, ``atom_removal_probability`` removes only that restriction (lower
    becomes zero; upper becomes one). Otherwise Boolean endpoints are resampled
    as zero/one, while continuous endpoints receive Gaussian perturbations or a
    uniform reset (``bound_reset_probability``). An unrestricted continuous bound
    is reset when added. Crossing proposals clamp a sole mutated endpoint against
    its unchanged counterpart; two mutated continuous endpoints are sorted. Two
    conflicting Boolean mutations resolve randomly to true or false. Independent
    clause toggles use ``conjunction_mutation_rate``.

    Initial conjunctions are enabled with ``initial_conjunction_probability``;
    atoms with ``initial_atom_probability``. With features present, each random
    conjunction gets at least one atom. FALSE and TRUE are also seeded explicitly.
    At least one elite and one offspring survive each update. ``survival_rate``
    controls the fraction of cached elites; selection uses ``tournament_size``.

    ``device=None`` selects CUDA when available, otherwise CPU. A local ``seed``
    restarts the same random stream on each optimize call without changing global
    RNG state. Reproducibility is for the same device and deterministic objective.

    Free memory is checked before each generation and work batch. Population and
    workspace estimates use ``memory_fraction`` of available memory, optionally
    capped by ``max_memory_bytes``. CPU detection supports Linux and cgroups;
    CUDA includes reusable allocator cache. ``objective_batch_size`` caps callback
    batches; ``objective_bytes_per_candidate`` estimates its extra workspace.
    Allocator OOMs reduce batches (or the initial population); a single-candidate
    failure propagates. A discovered objective batch cap is reused for subsequent
    generations. Unknown callback workspace, other processes, and OS OOM
    kills prevent an absolute memory guarantee. Set a workspace estimate and a
    conservative batch cap for objectives with large or nonlinear workspaces.

    ``simplify=True`` greedily removes one active restriction at a time from the
    winning DNF after evolution. Every round scores all single-removal neighbors
    and accepts the best only when penalized fitness strictly improves. Boolean
    literals become unrestricted; continuous lower/upper sides are independent.
    Conjunction slots and their activation flags are preserved.
    """

    def __init__(
        self,
        num_boolean_features: int,
        num_continuous_features: int,
        *,
        max_conjunctions: int = 8,
        survival_rate: float = 0.1,
        tournament_size: int = 3,
        selection_threshold: float = 0.2,
        crossover_rate: float = 0.8,
        crossover_swap_probability: float = 0.5,
        fine_grained_crossover_probability: float = 0.5,
        mutation_rate: float = 0.1,
        conjunction_mutation_rate: float = 0.05,
        atom_removal_probability: float = 0.2,
        bound_reset_probability: float = 0.1,
        bound_mutation_scale: float = 0.1,
        initial_atom_probability: float = 0.3,
        initial_conjunction_probability: float = 0.25,
        boolean_complexity_penalty: float = 0.01,
        continuous_complexity_penalty: float = 0.02,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
        memory_fraction: float = 0.5,
        objective_batch_size: Optional[int] = None,
        objective_bytes_per_candidate: int = 0,
        max_memory_bytes: Optional[int] = None,
        simplify: bool = False,
    ) -> None:
        parameters = locals().copy()
        parameters.pop("self")
        self._validate_parameters(parameters)
        self.num_boolean_features = num_boolean_features
        self.num_continuous_features = num_continuous_features
        self.max_conjunctions = max_conjunctions
        self.survival_rate = survival_rate
        self.tournament_size = tournament_size
        self.selection_threshold = selection_threshold
        self.crossover_rate = crossover_rate
        self.crossover_swap_probability = crossover_swap_probability
        self.fine_grained_crossover_probability = fine_grained_crossover_probability
        self.mutation_rate = mutation_rate
        self.conjunction_mutation_rate = conjunction_mutation_rate
        self.atom_removal_probability = atom_removal_probability
        self.bound_reset_probability = bound_reset_probability
        self.bound_mutation_scale = bound_mutation_scale
        self.initial_atom_probability = initial_atom_probability
        self.initial_conjunction_probability = initial_conjunction_probability
        self.boolean_complexity_penalty = boolean_complexity_penalty
        self.continuous_complexity_penalty = continuous_complexity_penalty
        self.seed = seed
        self.memory_fraction = memory_fraction
        self.objective_batch_size = objective_batch_size
        self.objective_bytes_per_candidate = objective_bytes_per_candidate
        self.max_memory_bytes = max_memory_bytes
        self.simplify = simplify
        self.device = torch.device(
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        self.num_features = num_boolean_features + num_continuous_features
        self._genome_bytes = max_conjunctions * (self.num_features * 8 + 1)
        self._workspace_bytes = max(256, self._genome_bytes * 32 + tournament_size * 48)
        self._evaluation_callback: Optional[EvaluationCallback] = None

    def set_evaluation_callback(self, callback: Optional[EvaluationCallback]) -> None:
        """Install or replace the evaluation observer; pass ``None`` to disable it.

        The signature is ``callback(generation, bounds, active, objective, score)``.
        It runs once after all evaluation batches finish, including generation
        zero, with the entire current population and its cached elites. Bounds
        have shape ``(population, max_conjunctions, features, 2)``; active flags
        have shape ``(population, max_conjunctions)``; objective and score each
        have shape ``(population,)`` and contain raw values and penalized fitness.

        All tensors are passed directly by reference on their existing device.
        The observer must not modify them. Retaining references prolongs their
        storage lifetime. It runs without autograd, its return value is ignored,
        and its exceptions propagate without evaluation retries. Registration
        persists across optimize calls; generation numbering restarts each run.
        Final simplification is separate and does not send population callbacks.
        """
        if callback is not None and not callable(callback):
            raise TypeError("evaluation callback must be callable or None")
        self._evaluation_callback = callback

    def _notify_evaluation(self, generation: int, population: _Population) -> None:
        """Pass completed population tensors to the current observer without copies."""
        if self._evaluation_callback is not None:
            self._evaluation_callback(
                generation,
                population.bounds,
                population.active,
                population.objectives,
                population.scores,
            )

    @staticmethod
    def _validate_parameters(parameters: dict[str, object]) -> None:
        """Reject invalid counts, probabilities, scales, and memory controls."""
        if not isinstance(parameters["simplify"], bool):
            raise ValueError("simplify must be a bool")
        minimums = {
            "num_boolean_features": 0,
            "num_continuous_features": 0,
            "max_conjunctions": 1,
            "tournament_size": 1,
            "objective_bytes_per_candidate": 0,
            "objective_batch_size": 1,
            "max_memory_bytes": 1,
            "seed": 0,
        }
        for name, minimum in minimums.items():
            value = parameters[name]
            if value is None and name in (
                "objective_batch_size",
                "max_memory_bytes",
                "seed",
            ):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        probabilities = (
            "survival_rate",
            "selection_threshold",
            "crossover_rate",
            "crossover_swap_probability",
            "fine_grained_crossover_probability",
            "mutation_rate",
            "conjunction_mutation_rate",
            "atom_removal_probability",
            "bound_reset_probability",
            "initial_atom_probability",
            "initial_conjunction_probability",
            "memory_fraction",
        )
        for name in probabilities:
            value = parameters[name]
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
            if name in ("survival_rate", "memory_fraction") and not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
            if name == "selection_threshold" and value == 1:
                raise ValueError("selection_threshold must be less than 1")
        for name in (
            "bound_mutation_scale",
            "boolean_complexity_penalty",
            "continuous_complexity_penalty",
        ):
            value = parameters[name]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")

    def _available_memory(self) -> int:
        """Return currently usable device bytes without allocating a tensor."""
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            reusable = torch.cuda.memory_reserved(
                self.device
            ) - torch.cuda.memory_allocated(self.device)
            return free + reusable
        return _cpu_available_memory()

    def _population_limit(
        self, requested: int, resident_bytes: int = 0, selection_bytes: int = 0
    ) -> int:
        """Plan two populations, selection/mutation scratch, and objective work."""
        budget = int((self._available_memory() + resident_bytes) * self.memory_fraction)
        if self.max_memory_bytes is not None:
            budget = min(budget, self.max_memory_bytes)
        # Existing parents remain at their original size when offspring shrink.
        budget -= resident_bytes + selection_bytes
        population_copies = 1 if resident_bytes else 2
        per_candidate = (
            population_copies * (self._genome_bytes + 16)
            + self._workspace_bytes
            + self.objective_bytes_per_candidate
            + 24
        )
        size = min(requested, budget // per_candidate)
        if size < 2:
            raise MemoryError("Memory budget cannot accommodate a population of two")
        return size

    def _batch_limit(self, requested: int, resident_bytes: int, workspace: int) -> int:
        """Recheck free memory before a bounded objective or reproduction batch."""
        budget = int(self._available_memory() * self.memory_fraction)
        if self.max_memory_bytes is not None:
            budget = min(budget, max(0, self.max_memory_bytes - resident_bytes))
        size = min(requested, budget // max(1, workspace))
        if size < 1:
            raise MemoryError("Memory budget cannot accommodate one work item")
        return size

    def _empty_population(self, size: int) -> _Population:
        """Allocate the only generation-sized storage needed by the search."""
        return _Population(
            torch.empty(
                size,
                self.max_conjunctions,
                self.num_features,
                2,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.empty(
                size, self.max_conjunctions, dtype=torch.bool, device=self.device
            ),
            torch.empty(size, dtype=torch.float64, device=self.device),
            torch.empty(size, dtype=torch.float64, device=self.device),
        )

    def _random_bounds(self, size: int, generator: torch.Generator) -> Tensor:
        """Sample restricted literals and one/two-sided continuous intervals."""
        shape = (size, self.max_conjunctions, self.num_features)
        endpoints = torch.rand(*shape, 2, device=self.device, generator=generator)
        lower = endpoints.amin(-1)
        upper = endpoints.amax(-1)
        mode = torch.randint(3, shape, device=self.device, generator=generator)
        lower.masked_fill_(mode == 0, 0)
        upper.masked_fill_(mode == 1, 1)
        boolean = torch.randint(
            2,
            (*shape[:2], self.num_boolean_features),
            device=self.device,
            generator=generator,
        ).float()
        lower[..., : self.num_boolean_features] = boolean
        upper[..., : self.num_boolean_features] = boolean
        return torch.stack((lower, upper), -1)

    def _initialize(self, population: _Population, generator: torch.Generator) -> None:
        """Fill candidates in bounded batches, then seed constant FALSE and TRUE."""
        offset = 0
        while offset < population.size:
            size = self._batch_limit(
                population.size - offset, population.nbytes, self._workspace_bytes
            )
            bounds = self._random_bounds(size, generator)
            restricted = (
                torch.rand(bounds.shape[:-1], device=self.device, generator=generator)
                < self.initial_atom_probability
            )
            if self.num_features:
                forced = torch.randint(
                    self.num_features,
                    (size, self.max_conjunctions, 1),
                    device=self.device,
                    generator=generator,
                )
                restricted.scatter_(-1, forced, True)
            bounds[..., 0].masked_fill_(~restricted, 0)
            bounds[..., 1].masked_fill_(~restricted, 1)
            population.bounds[offset : offset + size] = bounds
            population.active[offset : offset + size] = (
                torch.rand(
                    size, self.max_conjunctions, device=self.device, generator=generator
                )
                < self.initial_conjunction_probability
            )
            offset += size
        population.active[0] = False
        population.active[1] = False
        population.active[1, 0] = True
        population.bounds[1, ..., 0] = 0
        population.bounds[1, ..., 1] = 1

    def _initial_population(self, size: int, generator: torch.Generator) -> _Population:
        """Keep partially initialized buffers out of allocator retry tracebacks."""
        population = self._empty_population(size)
        self._initialize(population, generator)
        return population

    def complexity(
        self, bounds: Tensor, active_conjunctions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Count Boolean literals and continuous inequalities per batched DNF.

        Inputs use the objective's representation. Inactive clauses are ignored;
        a continuous range with both bounds restricted counts as two conditions.
        """
        split = self.num_boolean_features
        boolean = (
            ((bounds[..., :split, 0] > 0) | (bounds[..., :split, 1] < 1))
            & active_conjunctions[..., None]
        ).sum((-1, -2))
        continuous = (
            ((bounds[..., split:, 0] > 0).long() + (bounds[..., split:, 1] < 1))
            * active_conjunctions[..., None]
        ).sum((-1, -2))
        return boolean, continuous

    def fitness(
        self, objective_values: Tensor, bounds: Tensor, active_conjunctions: Tensor
    ) -> Tensor:
        """Divide raw objective values by one plus weighted syntactic complexity."""
        boolean, continuous = self.complexity(bounds, active_conjunctions)
        denominator = (
            1.0
            + self.boolean_complexity_penalty * boolean.to(torch.float64)
            + self.continuous_complexity_penalty * continuous.to(torch.float64)
        )
        return objective_values.to(torch.float64) / denominator

    def _clear_cache(self) -> None:
        """Release reusable CUDA cache after a recoverable allocator failure."""
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def _score_batch(
        self, objective: Objective, bounds: Tensor, active: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Validate environment feedback and calculate fitness within the OOM retry."""
        values = objective(bounds, active)
        if not isinstance(values, Tensor) or not values.is_floating_point():
            raise TypeError("objective must return a floating point tensor")
        if values.shape != (bounds.shape[0],) or values.device != bounds.device:
            raise ValueError("objective must return shape (batch,) on the input device")
        if not bool((torch.isfinite(values) & (values > 0)).all()):
            raise ValueError("objective values must be finite and strictly positive")
        return values, self.fitness(values, bounds, active)

    def _evaluate(
        self,
        objective: Objective,
        population: _Population,
        limit: int,
        start: int = 0,
    ) -> tuple[int, int]:
        """Score only new candidates, halving failed objective batches on OOM."""
        offset = start
        while offset < population.size:
            size = self._batch_limit(
                min(limit, population.size - offset),
                population.nbytes,
                self._workspace_bytes + self.objective_bytes_per_candidate,
            )
            bounds = population.bounds[offset : offset + size]
            active = population.active[offset : offset + size]
            try:
                values, scores = self._score_batch(objective, bounds, active)
            except (RuntimeError, MemoryError) as error:
                if not _is_memory_error(error) or size == 1:
                    raise
                limit = max(1, size // 2)
            else:
                population.objectives[offset : offset + size] = values
                population.scores[offset : offset + size] = scores
                offset += size
                del values, scores
                continue
            self._clear_cache()
        return population.size - start, limit

    def _simplification_bounds(self, bounds: Tensor) -> Tensor:
        """Canonicalize a winner without changing its effective objective/fitness."""
        return bounds

    @staticmethod
    def _selected_candidate(population: _Population, index: int) -> _Population:
        """Own a single candidate without retaining the source population."""
        return _Population(
            population.bounds[index : index + 1].clone(),
            population.active[index : index + 1].clone(),
            population.objectives[index : index + 1].clone(),
            population.scores[index : index + 1].clone(),
        )

    def _best_removal(
        self, objective: Objective, incumbent: _Population, restrictions: Tensor
    ) -> _Population:
        """Construct, score, and retain the best neighbor inside the OOM retry."""
        size = restrictions.shape[0]
        neighbors = self._empty_population(size)
        neighbors.bounds.copy_(incumbent.bounds.expand(size, -1, -1, -1))
        neighbors.active.copy_(incumbent.active.expand(size, -1))
        clause, feature, endpoint = restrictions.unbind(dim=1)
        rows = torch.arange(size, device=self.device)
        # In valid Boolean intervals exactly one endpoint restricts the literal,
        # so resetting it also removes the complete Boolean condition.
        neighbors.bounds[rows, clause, feature, endpoint] = endpoint.to(
            dtype=neighbors.bounds.dtype
        )
        values, scores = self._score_batch(
            objective, neighbors.bounds, neighbors.active
        )
        neighbors.objectives.copy_(values)
        neighbors.scores.copy_(scores)
        return self._selected_candidate(neighbors, int(scores.argmax().item()))

    def _simplify(
        self, objective: Objective, incumbent: _Population, limit: int
    ) -> tuple[_Population, int, int]:
        """Greedily widen the best DNF, streaming bounded single-removal batches."""
        incumbent.bounds = self._simplification_bounds(incumbent.bounds).clone()
        steps, evaluations = 0, 0
        workspace = (
            self._workspace_bytes
            + self.objective_bytes_per_candidate
            + self._genome_bytes
            + 16
        )
        while True:
            bounds = incumbent.bounds[0]
            restricted = torch.stack((bounds[..., 0] > 0, bounds[..., 1] < 1), dim=-1)
            restrictions = torch.nonzero(
                restricted & incumbent.active[0, :, None, None]
            )
            count = restrictions.shape[0]
            if not count:
                break
            winner = incumbent
            offset = 0
            while offset < count:
                resident = (
                    incumbent.nbytes
                    + winner.nbytes
                    + restrictions.numel() * restrictions.element_size()
                )
                size = self._batch_limit(
                    min(limit, count - offset), resident, workspace
                )
                try:
                    candidate = self._best_removal(
                        objective, incumbent, restrictions[offset : offset + size]
                    )
                except (RuntimeError, MemoryError) as error:
                    if not _is_memory_error(error) or size == 1:
                        raise
                    limit = max(1, size // 2)
                else:
                    evaluations += size
                    offset += size
                    if bool(candidate.scores[0] > winner.scores[0]):
                        winner = candidate
                    del candidate
                    continue
                self._clear_cache()
            if not bool(winner.scores[0] > incumbent.scores[0]):
                break
            incumbent = winner
            steps += 1
        return incumbent, steps, evaluations

    def _mutate(
        self, bounds: Tensor, active: Tensor, generator: torch.Generator
    ) -> None:
        """Mutate endpoints independently, preserving any unselected endpoint."""
        fresh = torch.rand(bounds.shape, device=self.device, generator=generator)
        split = self.num_boolean_features
        fresh[..., :split, :].round_()
        continuous = bounds[..., split:, :]
        noise = (
            torch.randn(continuous.shape, device=self.device, generator=generator)
            * self.bound_mutation_scale
        )
        perturbed = (continuous + noise).clamp_(0, 1)
        unrestricted = torch.stack(
            (continuous[..., 0] == 0, continuous[..., 1] == 1), -1
        )
        reset = (
            torch.rand(continuous.shape, device=self.device, generator=generator)
            < self.bound_reset_probability
        ) | unrestricted
        fresh[..., split:, :] = torch.where(reset, fresh[..., split:, :], perturbed)
        remove = (
            torch.rand(bounds.shape, device=self.device, generator=generator)
            < self.atom_removal_probability
        )
        fresh[..., 0].masked_fill_(remove[..., 0], 0)
        fresh[..., 1].masked_fill_(remove[..., 1], 1)
        mutate = (
            torch.rand(bounds.shape, device=self.device, generator=generator)
            < self.mutation_rate
        )
        proposed = torch.where(mutate, fresh, bounds)
        lower = torch.where(mutate[..., 0], proposed.amin(-1), proposed[..., 0])
        upper = torch.where(mutate[..., 1], proposed.amax(-1), proposed[..., 1])
        boolean_conflict = (
            proposed[..., :split, 0] > proposed[..., :split, 1]
        ) & mutate[..., :split, :].all(-1)
        resolution = torch.randint(
            2, boolean_conflict.shape, device=self.device, generator=generator
        ).float()
        lower[..., :split] = torch.where(
            boolean_conflict, resolution, lower[..., :split]
        )
        upper[..., :split] = torch.where(
            boolean_conflict, resolution, upper[..., :split]
        )
        bounds[..., 0].copy_(lower)
        bounds[..., 1].copy_(upper)
        active.logical_xor_(
            torch.rand(active.shape, device=self.device, generator=generator)
            < self.conjunction_mutation_rate
        )

    def _parent_indices(self, parents: _Population) -> Tensor:
        """Build the ranked eligible parent pool before any tournaments."""
        count = max(1, parents.size - int(parents.size * self.selection_threshold))
        return parents.scores.topk(count, sorted=False).indices

    def _crossover(
        self,
        bounds: Tensor,
        active: Tensor,
        other_bounds: Tensor,
        other_active: Tensor,
        generator: torch.Generator,
    ) -> None:
        """Choose exactly one crossover mode per slot and repair mixed conflicts."""
        roll = torch.rand(active.shape, device=self.device, generator=generator)
        whole_cutoff = self.crossover_rate * (
            1 - self.fine_grained_crossover_probability
        )
        whole = roll < whole_cutoff
        fine = (roll >= whole_cutoff) & (roll < self.crossover_rate)
        swap_bounds = whole[..., None, None] | (
            fine[..., None, None]
            & (
                torch.rand(bounds.shape, device=self.device, generator=generator)
                < self.crossover_swap_probability
            )
        )
        mixed = torch.where(swap_bounds, other_bounds, bounds)
        conflict = mixed[..., 0] > mixed[..., 1]
        bounds.copy_(torch.where(conflict[..., None], bounds, mixed))
        swap_active = whole | (
            fine
            & (
                torch.rand(active.shape, device=self.device, generator=generator)
                < self.crossover_swap_probability
            )
        )
        active.copy_(torch.where(swap_active, other_active, active))

    def _children(
        self,
        parents: _Population,
        size: int,
        generator: torch.Generator,
        parent_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run tournaments within the filtered pool, then cross and mutate."""
        contestants = torch.randint(
            parent_indices.numel(),
            (size, 2, self.tournament_size),
            device=self.device,
            generator=generator,
        )
        contestants = parent_indices[contestants]
        winners = parents.scores[contestants].argmax(-1, keepdim=True)
        indices = contestants.gather(-1, winners).squeeze(-1)
        bounds = parents.bounds[indices[:, 0]]
        active = parents.active[indices[:, 0]]
        self._crossover(
            bounds,
            active,
            parents.bounds[indices[:, 1]],
            parents.active[indices[:, 1]],
            generator,
        )
        self._mutate(bounds, active, generator)
        return bounds, active

    def _elite_population(
        self, parents: _Population, size: int
    ) -> tuple[_Population, int, Tensor]:
        """Allocate a generation, copy elites, and cache eligible parent indices."""
        elites = min(size - 1, max(1, int(size * self.survival_rate)))
        indices = parents.scores.topk(elites, sorted=False).indices
        children = self._empty_population(size)
        children.bounds[:elites] = parents.bounds[indices]
        children.active[:elites] = parents.active[indices]
        children.objectives[:elites] = parents.objectives[indices]
        children.scores[:elites] = parents.scores[indices]
        return children, elites, self._parent_indices(parents)

    def _generation(
        self, parents: _Population, generator: torch.Generator
    ) -> tuple[_Population, int]:
        """Copy elites and generate offspring in memory-bounded vectorized chunks."""
        size = self._population_limit(
            parents.size, parents.nbytes, selection_bytes=parents.size * 24
        )
        while True:
            try:
                children, elites, parent_indices = self._elite_population(parents, size)
            except (RuntimeError, MemoryError) as error:
                if not _is_memory_error(error) or size <= 2:
                    raise
                size = max(2, size // 2)
            else:
                break
            self._clear_cache()
        offset = elites
        limit = size
        while offset < size:
            batch = self._batch_limit(
                min(limit, size - offset),
                parents.nbytes + children.nbytes + parent_indices.numel() * 8,
                self._workspace_bytes,
            )
            try:
                bounds, active = self._children(
                    parents, batch, generator, parent_indices
                )
            except (RuntimeError, MemoryError) as error:
                if not _is_memory_error(error) or batch == 1:
                    raise
                limit = max(1, batch // 2)
            else:
                children.bounds[offset : offset + batch] = bounds
                children.active[offset : offset + batch] = active
                offset += batch
                del bounds, active
                continue
            self._clear_cache()
        return children, elites

    @torch.no_grad()
    def optimize(
        self, objective: Objective, max_population: int = 1024, generations: int = 100
    ) -> DNFOptimizationResult:
        """Maximize penalized objective using at most ``max_population`` candidates.

        The initial population is evaluated even for ``generations=0``. Subsequent
        generations retain cached elites, so the best fitness cannot decrease for
        deterministic objectives. Requested population is an upper bound and may
        shrink as free memory changes. There are no per-individual/feature loops;
        Python loops cover generations, greedy rounds, memory chunks, and OOM retries.
        An installed evaluation callback receives the complete population after
        each evaluation, starting at generation zero, without cloning its tensors.
        Optional simplification then improves the final winner separately, without
        population callbacks; its trials are included in the evaluation count.
        """
        if not callable(objective):
            raise TypeError("objective must be callable")
        for name, value, minimum in (
            ("max_population", max_population, 2),
            ("generations", generations, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        generator = torch.Generator(device=self.device)
        if self.seed is None:
            generator.seed()
        else:
            generator.manual_seed(self.seed)
        size = self._population_limit(max_population)
        while True:
            try:
                population = self._initial_population(size, generator)
            except (RuntimeError, MemoryError) as error:
                if not _is_memory_error(error) or size <= 2:
                    raise
                size = max(2, size // 2)
            else:
                break
            self._clear_cache()
        evaluations, batch_limit = self._evaluate(
            objective, population, self.objective_batch_size or population.size
        )
        self._notify_evaluation(0, population)
        for generation in range(1, generations + 1):
            population, elites = self._generation(population, generator)
            new_evaluations, batch_limit = self._evaluate(
                objective, population, batch_limit, elites
            )
            evaluations += new_evaluations
            self._notify_evaluation(generation, population)
        best = int(population.scores.argmax().item())
        winner = self._selected_candidate(population, best)
        population_size = population.size
        del population
        simplification_steps, simplification_evaluations = 0, 0
        if self.simplify:
            winner, simplification_steps, simplification_evaluations = self._simplify(
                objective, winner, batch_limit
            )
        boolean, continuous = self.complexity(winner.bounds, winner.active)
        return DNFOptimizationResult(
            bounds=winner.bounds[0].clone(),
            active_conjunctions=winner.active[0].clone(),
            objective_value=float(winner.objectives[0].item()),
            fitness=float(winner.scores[0].item()),
            boolean_conditions=int(boolean.item()),
            continuous_conditions=int(continuous.item()),
            population_size=population_size,
            generations=generations,
            evaluations=evaluations + simplification_evaluations,
            simplification_steps=simplification_steps,
            simplification_evaluations=simplification_evaluations,
        )
