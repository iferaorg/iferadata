"""Tests for merging duplicate filter terms in Split.__str__() output."""

import torch

from ifera.optionalpha import Split, FilterInfo


def test_merge_left_filters_same_filter():
    """Test merging multiple left filters on the same filter - should keep lower threshold."""
    # Create depth 1 splits with left filters on the same column
    split_a = Split(
        mask=torch.tensor([True, True, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 1.5, "left")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 0.5, "left")],
        parents=[],
    )

    # Create child split: (filter_x < 1.5) AND (filter_x < 0.5)
    # Should simplify to: (filter_x < 0.5)
    child = Split(
        mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b)],
    )

    result = str(child)

    # Should only have filter_x once with the lower threshold
    assert "(filter_x < 0.5)" in result
    # Should NOT have the higher threshold
    assert "1.5" not in result
    # Should only appear once in the conjunction
    assert result.count("filter_x") == 1


def test_merge_right_filters_same_filter():
    """Test merging multiple right filters on the same filter - should keep higher threshold."""
    # Create depth 1 splits with right filters on the same column
    split_a = Split(
        mask=torch.tensor([False, False, True, True], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_y", 2.0, "right")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([False, False, False, True], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_y", 3.0, "right")],
        parents=[],
    )

    # Create child split: (filter_y > 2.0) AND (filter_y > 3.0)
    # Should simplify to: (filter_y > 3.0)
    child = Split(
        mask=torch.tensor([False, False, False, True], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b)],
    )

    result = str(child)

    # Should only have filter_y once with the higher threshold
    assert "(filter_y > 3.0)" in result
    # Should NOT have the lower threshold
    assert "2.0" not in result
    # Should only appear once in the conjunction
    assert result.count("filter_y") == 1


def test_no_merge_different_directions():
    """Test that filters with different directions are NOT merged."""
    # Create splits with opposite directions on the same filter
    split_a = Split(
        mask=torch.tensor([False, True, True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_z", 1.5, "left")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([False, True, True, True], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_z", 2.5, "right")],
        parents=[],
    )

    # Create child split: (filter_z < 1.5) AND (filter_z > 2.5)
    # Should NOT be merged (different directions)
    child = Split(
        mask=torch.tensor([False, True, True, False], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b)],
    )

    result = str(child)

    # Should have both filters
    assert "(filter_z < 1.5)" in result
    assert "(filter_z > 2.5)" in result
    # Should appear twice in the conjunction
    assert result.count("filter_z") == 2


def test_merge_multiple_filters_in_conjunction():
    """Test merging when there are multiple different filters in a conjunction."""
    # Create splits with different filters
    split_a = Split(
        mask=torch.tensor([True, True, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 1.5, "left")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 0.5, "left")],
        parents=[],
    )

    split_c = Split(
        mask=torch.tensor([True, True, True, False], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_y", 2.0, "right")],
        parents=[],
    )

    # Create depth 2 split: A AND B
    depth_2 = Split(
        mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b)],
    )

    # Create depth 3 split: (A AND B) AND C
    # This is: (filter_x < 1.5) AND (filter_x < 0.5) AND (filter_y > 2.0)
    # Should simplify to: (filter_x < 0.5) AND (filter_y > 2.0)
    depth_3 = Split(
        mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        filters=[],
        parents=[(depth_2, split_c)],
    )

    result = str(depth_3)

    # Should have both filters in the conjunction
    assert "(filter_x < 0.5)" in result
    assert "(filter_y > 2.0)" in result
    # filter_x should appear only once
    assert result.count("filter_x") == 1
    # filter_y should appear only once
    assert result.count("filter_y") == 1
    # Should NOT have the higher threshold for filter_x
    assert "1.5" not in result


def test_merge_with_three_same_filters():
    """Test merging when there are three or more of the same filter."""
    # Create three splits with left filters on the same column
    split_a = Split(
        mask=torch.tensor([True, True, True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 3.0, "left")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([True, True, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 2.0, "left")],
        parents=[],
    )

    split_c = Split(
        mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 1.0, "left")],
        parents=[],
    )

    # Create depth 2: A AND B
    depth_2 = Split(
        mask=torch.tensor([True, True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b)],
    )

    # Create depth 3: (A AND B) AND C
    # This is: (filter_x < 3.0) AND (filter_x < 2.0) AND (filter_x < 1.0)
    # Should simplify to: (filter_x < 1.0)
    depth_3 = Split(
        mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        filters=[],
        parents=[(depth_2, split_c)],
    )

    result = str(depth_3)

    # Should only have filter_x once with the lowest threshold
    assert "(filter_x < 1.0)" in result
    # Should NOT have the other thresholds
    assert "2.0" not in result
    assert "3.0" not in result
    # Should only appear once
    assert result.count("filter_x") == 1


def test_no_merge_different_filters():
    """Test that different filters are not merged."""
    # Create splits with different filters
    split_a = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_a", 1.5, "left")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_b", 2.5, "left")],
        parents=[],
    )

    # Create child split
    child = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b)],
    )

    result = str(child)

    # Should have both filters
    assert "(filter_a < 1.5)" in result
    assert "(filter_b < 2.5)" in result


def test_merge_in_or_relationship():
    """Test merging when there are OR relationships (multiple parent pairs)."""
    # Create parent splits
    split_a = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 2.0, "left")],
        parents=[],
    )

    split_b = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 1.0, "left")],
        parents=[],
    )

    split_c = Split(
        mask=torch.tensor([False, True, True], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_y", 3.0, "right")],
        parents=[],
    )

    # Create child with two parent pairs (OR relationship)
    # (A AND B) OR (B AND C)
    # Which is: (filter_x < 2.0 AND filter_x < 1.0) OR (filter_x < 1.0 AND filter_y > 3.0)
    # Should simplify to: (filter_x < 1.0) OR (filter_x < 1.0 AND filter_y > 3.0)
    child = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split_a, split_b), (split_b, split_c)],
    )

    result = str(child)

    # First conjunction should be simplified
    lines = result.strip().split("\n")
    assert len(lines) == 3  # header + 2 conjunctions

    # Each conjunction should have filter_x only once
    for line in lines[1:]:  # Skip header
        # Count occurrences in this line
        assert line.count("filter_x") == 1


def test_merge_empty_conjunction():
    """Test that empty conjunctions are handled correctly."""
    split = Split(
        mask=torch.tensor([False, False], dtype=torch.bool),
        filters=[],
        parents=[],
    )

    result = str(split)
    assert "(empty)" in result


def test_merge_single_filter_no_change():
    """Test that single filters are not affected by merging."""
    split = Split(
        mask=torch.tensor([True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_x", 1.5, "left")],
        parents=[],
    )

    result = str(split)
    assert "(filter_x < 1.5)" in result
    assert result.count("filter_x") == 1
