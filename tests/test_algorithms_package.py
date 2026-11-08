"""Public API and package discovery tests for algorithms."""

from importlib.util import find_spec

from setuptools import find_packages

from ifera import algorithms
from ifera.algorithms.evolutionary_dnf import (
    DNFOptimizationResult,
    EvolutionaryDNFOptimizer,
)
from ifera.algorithms.regression_decision_tree import RegressionDecisionTree


def test_algorithms_package_exports_public_api():
    assert set(algorithms.__all__) == {
        "RegressionDecisionTree",
        "DNFOptimizationResult",
        "EvolutionaryDNFOptimizer",
    }
    assert algorithms.RegressionDecisionTree is RegressionDecisionTree
    assert algorithms.DNFOptimizationResult is DNFOptimizationResult
    assert algorithms.EvolutionaryDNFOptimizer is EvolutionaryDNFOptimizer


def test_algorithms_is_discovered_as_an_ifera_subpackage():
    packages = find_packages()
    assert "ifera.algorithms" in packages
    assert "algorithms" not in packages
    assert find_spec("algorithms") is None
