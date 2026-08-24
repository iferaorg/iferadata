"""Tests for exact support selection around the continuous Kelly solver."""

from types import SimpleNamespace

import torch

from ifera.allocation import _kelly_selection
from ifera.allocation._strategy_selection import StrategySelectionConstraints


def _constraints(
    strategy_count: int,
    maximum_active: int,
    conflicts: tuple[tuple[int, int], ...] = (),
) -> StrategySelectionConstraints:
    """Build the production prior-conflict representation for a test graph."""
    prior: list[list[int]] = [[] for _ in range(strategy_count)]
    for first, second in conflicts:
        lower, upper = sorted((first, second))
        prior[upper].append(lower)
    return StrategySelectionConstraints(
        maximum_active,
        tuple(torch.tensor(values, dtype=torch.int64) for values in prior),
    )


def test_ordinary_simplex_preserves_the_solver_result():
    """A single overlap component uses the ordinary Kelly simplex."""
    transposed_returns = torch.tensor(
        [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], dtype=torch.float64
    )

    result = _kelly_selection.maximize_kelly_with_selection(
        transposed_returns,
        0.8,
        [(0, 1)],
        None,
    )

    assert result is not None
    torch.testing.assert_close(
        result, torch.tensor([0.0, 0.8], dtype=torch.float64), atol=1e-9, rtol=0.0
    )


def test_all_disjoint_cardinality_search_selects_the_best_strategy():
    """Best-bound branching enforces a global active-strategy limit."""
    transposed_returns = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.7],
        ],
        dtype=torch.float32,
    )

    result = _kelly_selection.maximize_kelly_with_selection(
        transposed_returns,
        1.0,
        [],
        _constraints(3, maximum_active=1),
    )

    assert result is not None
    assert result.dtype == torch.float32
    assert result.device == transposed_returns.device
    assert torch.equal(result, torch.tensor([0.0, 0.0, 1.0]))


def test_conflict_branching_returns_the_best_compatible_support():
    """Mutually exclusive strategies cannot both remain positive."""
    transposed_returns = torch.tensor(
        [
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.6],
            [0.0, 0.5, 0.0],
        ],
        dtype=torch.float64,
    )

    result = _kelly_selection.maximize_kelly_with_selection(
        transposed_returns,
        1.0,
        [],
        _constraints(3, maximum_active=3, conflicts=((1, 2),)),
    )

    assert result is not None
    # Strategy 1 is better than its conflicting strategy 2. Strategy 0 is
    # compatible with both and remains independently fully allocated.
    torch.testing.assert_close(
        result,
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
        atol=1e-9,
        rtol=0.0,
    )
    assert result[2].item() == 0.0


def test_disjoint_clique_components_are_optimized_independently():
    """Disjoint simplex components may each reuse the capital budget."""
    transposed_returns = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=torch.float64,
    )

    result = _kelly_selection.maximize_kelly_with_selection(
        transposed_returns,
        1.0,
        [(0, 1), (2, 3)],
        _constraints(5, maximum_active=2),
    )

    assert result is not None
    # The unconstrained component winners are 1, 3, and 4. The cardinality
    # branch drops the weakest of those while retaining each component budget.
    torch.testing.assert_close(
        result,
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float64),
        atol=1e-9,
        rtol=0.0,
    )


def test_overlapping_maximal_cliques_report_unsupported_geometry():
    """A strategy shared by two clique constraints needs the grid fallback."""
    transposed_returns = torch.full((3, 4), 0.1)

    result = _kelly_selection.maximize_kelly_with_selection(
        transposed_returns,
        1.0,
        [(0, 1), (1, 2)],
        _constraints(3, maximum_active=2),
    )

    assert result is None


def test_unconverged_relaxation_reports_continuous_search_failure(monkeypatch):
    """An unfinished FW relaxation must fall back instead of pruning unsafely."""

    def unfinished_solver(*_args, **_kwargs):
        return SimpleNamespace(
            allocation=torch.zeros(2, dtype=torch.float64),
            mean_log_growth=0.0,
            upper_bound=1.0,
            converged=False,
        )

    monkeypatch.setattr(_kelly_selection, "maximize_kelly_growth", unfinished_solver)

    result = _kelly_selection.maximize_kelly_with_selection(
        torch.tensor([[0.1, 0.0], [0.0, 0.2]], dtype=torch.float64),
        1.0,
        [],
        _constraints(2, maximum_active=1),
    )

    assert result is None
