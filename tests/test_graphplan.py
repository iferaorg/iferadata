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
    noop_names = {a["name"] for a in planner.all_actions if a.get("noop")}
    assert all(act not in noop_names for act in flattened)
    for action in ["paint_green", "increment_to_1", "set_to_3", "make_safe"]:
        assert action in flattened


def test_graphplan_backtrack():
    """GraphPlan should backtrack across levels to find valid plan."""
    yaml_path = Path(__file__).with_name("test_graphplan_backtrack.yml")
    planner = GraphPlan(str(yaml_path))
    plan = planner.run()
    assert plan is not None
    flattened = sorted(act for layer in plan for act in layer)
    assert flattened == ["a2", "b2"]
