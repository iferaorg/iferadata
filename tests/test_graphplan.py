"""Tests for the GraphPlan algorithm."""

from pathlib import Path

from algorithms.graphplan import GraphPlan


def test_graphplan_sample():
    """Ensure that a plan is found for the sample YAML problem."""
    yaml_path = Path(__file__).with_name("test_graphplan_sample.yml")
    planner = GraphPlan(str(yaml_path))
    plan = planner.run()
    assert plan is not None
    flattened = [act for layer in plan for act in layer]
    for action in ["paint_green", "increment_to_1", "set_to_3", "make_safe"]:
        assert action in flattened
