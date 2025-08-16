"""Tests for the GraphPlan algorithm."""

from pathlib import Path

from algorithms.graphplan import main


def test_graphplan_sample(capsys):
    """Ensure that a plan is found for the sample YAML problem."""
    yaml_path = Path(__file__).with_name("test_graphplan_sample.yml")
    main(str(yaml_path))
    captured = capsys.readouterr().out
    assert "Plan found" in captured
    for action in ["paint_green", "increment_to_1", "set_to_3", "make_safe"]:
        assert action in captured
