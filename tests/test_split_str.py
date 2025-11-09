"""Tests for Split.__str__ method improvements."""

import torch
import pytest
from ifera.optionalpha import Split, FilterInfo


def test_split_str_deduplicates_repeated_conjunctions():
    """Test that repeated conjunctions differing only in order are deduplicated."""
    # Create two depth-1 splits
    split1 = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_a", 1.5, "right")],
        parents=[],
    )

    split2 = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_b", 10.0, "left")],
        parents=[],
    )

    # Create a child split with duplicate parent pairs in different order
    child_split = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[],
        parents=[(split1, split2), (split2, split1)],
    )

    str_repr = str(child_split)
    lines = str_repr.split("\n")

    # Should have header + 1 unique conjunction (not 2)
    assert len(lines) == 2, f"Expected 2 lines (header + 1 conjunction), got {len(lines)}"

    # The conjunction line should contain both filters
    assert "(filter_a >= 1.5)" in lines[1]
    assert "(filter_b <= 10.0)" in lines[1]


def test_split_str_sorts_by_filter_idx():
    """Test that conjunction terms are sorted by filter_idx."""
    # Create splits with different filter_idx values
    split1 = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(5, "filter_c", 2.5, "left")],
        parents=[],
    )

    split2 = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[FilterInfo(2, "filter_d", 15.0, "right")],
        parents=[],
    )

    # Create a child split
    child_split = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[],
        parents=[(split1, split2)],
    )

    str_repr = str(child_split)

    # filter_d (idx=2) should come before filter_c (idx=5)
    idx_d = str_repr.find("filter_d")
    idx_c = str_repr.find("filter_c")
    assert idx_d < idx_c, "filter_d (idx=2) should appear before filter_c (idx=5)"


def test_split_str_uses_ampersand():
    """Test that conjunction uses '&' instead of 'and'."""
    # Create two depth-1 splits
    split1 = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_a", 1.5, "right")],
        parents=[],
    )

    split2 = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_b", 10.0, "left")],
        parents=[],
    )

    # Create a child split
    child_split = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[],
        parents=[(split1, split2)],
    )

    str_repr = str(child_split)

    # Should use '&' not 'and'
    assert " & " in str_repr, "Expected '&' separator in conjunction"
    assert " and " not in str_repr, "Should not use 'and' separator"


def test_split_str_depth_1_single_filter():
    """Test that depth 1 splits with single filter work correctly."""
    split = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_a", 1.5, "left")],
        parents=[],
    )

    str_repr = str(split)

    # Should have a single line with the filter
    lines = str_repr.split("\n")
    assert len(lines) == 2, "Expected header + 1 filter line"
    assert "(filter_a <= 1.5)" in lines[1]
    assert " & " not in lines[1], "Single filter should not have '&'"


def test_split_str_multiple_parent_pairs():
    """Test splits with multiple parent pairs."""
    # Create three depth-1 splits
    split1 = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(0, "filter_a", 1.5, "right")],
        parents=[],
    )

    split2 = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[FilterInfo(1, "filter_b", 10.0, "left")],
        parents=[],
    )

    split3 = Split(
        mask=torch.tensor([False, True, True], dtype=torch.bool),
        filters=[FilterInfo(2, "filter_c", 5.0, "right")],
        parents=[],
    )

    # Create a child split with multiple parent pairs
    child_split = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split1, split2), (split1, split3)],
    )

    str_repr = str(child_split)
    lines = str_repr.split("\n")

    # Should have 2 different conjunctions (OR relationship)
    assert len(lines) == 3, f"Expected 3 lines (header + 2 conjunctions), got {len(lines)}"


def test_split_str_complex_sorting():
    """Test sorting with multiple filters in different orders."""
    # Create splits with filter_idx values: 7, 2, 5
    split1 = Split(
        mask=torch.tensor([True, True, False], dtype=torch.bool),
        filters=[FilterInfo(7, "filter_g", 1.0, "left")],
        parents=[],
    )

    split2 = Split(
        mask=torch.tensor([True, False, True], dtype=torch.bool),
        filters=[FilterInfo(2, "filter_b", 2.0, "right")],
        parents=[],
    )

    split3 = Split(
        mask=torch.tensor([True, True, True], dtype=torch.bool),
        filters=[FilterInfo(5, "filter_e", 3.0, "left")],
        parents=[],
    )

    # Create intermediate child
    child1 = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(split1, split2)],
    )

    # Create final child combining all three
    child2 = Split(
        mask=torch.tensor([True, False, False], dtype=torch.bool),
        filters=[],
        parents=[(child1, split3)],
    )

    str_repr = str(child2)

    # Order should be: filter_b (2), filter_e (5), filter_g (7)
    idx_b = str_repr.find("filter_b")
    idx_e = str_repr.find("filter_e")
    idx_g = str_repr.find("filter_g")

    assert idx_b < idx_e < idx_g, "Filters should appear in order: filter_b, filter_e, filter_g"


def test_split_str_empty_split():
    """Test that empty splits are handled correctly."""
    split = Split(
        mask=torch.tensor([False, False, False], dtype=torch.bool),
        filters=[],
        parents=[],
    )

    str_repr = str(split)

    assert "(empty)" in str_repr, "Empty split should show '(empty)'"
