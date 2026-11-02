"""Regression tests for bounded boolean conjunction preprocessing."""

from datetime import date, timedelta
from itertools import product

import polars as pl
import pytest
import torch

from ifera import optionalpha


def _tables(boolean_count):
    """Keep computed columns constant while varying independent boolean filters."""
    rows = list(product([0, 1], repeat=boolean_count))
    dates = [date(2022, 1, 3) + timedelta(weeks=index) for index in range(len(rows))]
    trades = pl.DataFrame(
        {"date": dates, "risk": [100.0] * len(rows), "profit": [10.0] * len(rows)}
    )
    filters = pl.DataFrame(
        {
            "date": dates,
            **{
                f"flag_{index}": [row[index] for row in rows]
                for index in range(boolean_count)
            },
        }
    )
    return trades, filters


def _prefer_fewer_samples(profits, returns, masks, date_ordinals):
    del profits, returns, date_ordinals
    return -masks.float().sum(dim=1)


@pytest.mark.parametrize(
    "boolean_count,left_only,keep_best_n", [(7, True, 2), (6, False, 3)]
)
def test_boolean_frontier_is_scored_and_pruned_before_each_expansion(
    monkeypatch, boolean_count, left_only, keep_best_n
):
    """A boolean beam must bound even the first round, in either filter direction."""
    trades, filters = _tables(boolean_count)
    original_generate = optionalpha._generate_child_splits
    score_batch_sizes = []
    frontier_sizes = []

    def record_score(profits, returns, masks, date_ordinals):
        score_batch_sizes.append(masks.shape[0])
        return _prefer_fewer_samples(profits, returns, masks, date_ordinals)

    def bounded_children(parent_list_a, parent_list_b, exclusion_mask):
        assert score_batch_sizes, "Boolean atoms must be scored before expansion"
        frontier_sizes.append(len(parent_list_a))
        assert len(parent_list_a) <= keep_best_n
        return original_generate(parent_list_a, parent_list_b, exclusion_mask)

    monkeypatch.setattr(optionalpha, "_generate_child_splits", bounded_children)
    generator = optionalpha.SplitGenerator(
        spread_width=20,
        left_only_filters=filters.columns[1:] if left_only else [],
        max_depth=1,
        score_func=record_score,
        keep_best_n=keep_best_n,
        pregenerate_boolean_conjunctions=True,
    )

    splits = generator.generate(trades, filters)[2]

    assert frontier_sizes
    assert len(frontier_sizes) > 1
    assert len(splits) <= keep_best_n
    assert any(split.mask.sum().item() == 1 for split in splits)


def test_boolean_preprocessing_retains_atoms_and_only_best_conjunctions(monkeypatch):
    """The extension pool preserves atoms without retaining every prior conjunction."""
    trades, filters = _tables(7)
    original_generate = optionalpha._generate_boolean_splits
    pool_sizes = []

    def bounded_pool(atoms, *args, **kwargs):
        original_masks = {tuple(split.mask.tolist()) for split in atoms}
        splits = original_generate(atoms, *args, **kwargs)
        returned_masks = {tuple(split.mask.tolist()) for split in splits}
        assert original_masks <= returned_masks
        assert len(returned_masks - original_masks) <= 2
        assert len(returned_masks) == len(splits)
        pool_sizes.append(len(splits))
        return splits

    monkeypatch.setattr(optionalpha, "_generate_boolean_splits", bounded_pool)
    generator = optionalpha.SplitGenerator(
        spread_width=20,
        left_only_filters=filters.columns[1:],
        max_depth=1,
        score_func=_prefer_fewer_samples,
        keep_best_n=2,
        pregenerate_boolean_conjunctions=True,
    )

    splits = generator.generate(trades, filters)[2]

    assert pool_sizes
    assert len(splits) == 2
    assert splits[0].score == -1.0


def test_boolean_deduplication_avoids_dense_new_by_existing_matrix(monkeypatch):
    """Exact boolean deduplication must not allocate a quadratic comparison tensor."""
    trades, filters = _tables(3)

    def reject_dense_comparison(*args, **kwargs):
        del args, kwargs
        pytest.fail("Boolean preprocessing must not build a dense pairwise mask matrix")

    monkeypatch.setattr(
        optionalpha, "_find_redundant_mask_indices", reject_dense_comparison
    )
    generator = optionalpha.SplitGenerator(
        spread_width=20,
        left_only_filters=filters.columns[1:],
        max_depth=1,
        pregenerate_boolean_conjunctions=True,
    )

    splits = generator.generate(trades, filters)[2]

    rows = list(product([0, 1], repeat=3))
    expected_masks = {
        tuple(
            all(value == 0 for value, used in zip(row, subset) if used) for row in rows
        )
        for subset in product([False, True], repeat=3)
        if any(subset)
    }
    assert len(splits) == 7
    assert {tuple(split.mask.tolist()) for split in splits} == expected_masks


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_split_mask_keys_are_exact_with_partial_final_bytes(noncontiguous):
    """Keys distinguish tail bits, preserve duplicates, and support strided masks."""
    masks = torch.tensor(
        [
            [False] * 9,
            [True] * 9,
            [False] * 8 + [True],
            [True] + [False] * 8,
            [False] * 9,
            [True] * 9,
        ]
    )
    if noncontiguous:
        storage = torch.zeros((len(masks), 18), dtype=torch.bool)
        storage[:, ::2] = masks
        masks = storage[:, ::2]
        assert not masks.is_contiguous()
    splits = [optionalpha.Split(mask, [], []) for mask in masks]

    keys = optionalpha._split_mask_keys(splits)

    assert len(keys) == 6
    assert all(isinstance(key, bytes) for key in keys)
    assert keys[0] == keys[4]
    assert keys[1] == keys[5]
    assert len(set(keys)) == 4


def test_split_mask_keys_handles_empty_input():
    """No tensor stack or device transfer is needed for an empty collection."""
    assert optionalpha._split_mask_keys([]) == []


def test_split_mask_keys_distinguishes_mask_lengths_across_calls():
    """Unused packed bits cannot make masks of different sample lengths identical."""
    masks = [[True, False], [True, False, False], [True] + [False] * 7]
    keys = [
        optionalpha._split_mask_keys([optionalpha.Split(torch.tensor(mask), [], [])])[0]
        for mask in masks
    ]

    assert len(set(keys)) == len(masks)


def test_split_mask_keys_preserves_order_across_batches():
    """Packing multiple batches preserves position and detects cross-batch duplicates."""
    masks = torch.zeros((2051, 9), dtype=torch.bool)
    masks[[1023, 1024, 2050], -1] = True
    splits = [optionalpha.Split(mask, [], []) for mask in masks]

    keys = optionalpha._split_mask_keys(splits)

    assert len(keys) == len(splits)
    assert keys[1023] == keys[1024] == keys[2050]
    assert keys[0] == keys[1022] == keys[1025] == keys[2049]
    assert keys[0] != keys[-1]
    assert len(set(keys)) == 2


def test_exact_mask_merge_preserves_filters_parents_and_first_score():
    """Mask grouping combines explanations and retains the established score policy."""
    mask = torch.tensor([True] + [False] * 8)
    parent_a = optionalpha.Split(torch.tensor([True] * 3 + [False] * 6), [], [])
    parent_b = optionalpha.Split(torch.tensor([True] * 2 + [False] * 7), [], [])
    first_filter = optionalpha.FilterInfo(0, "first", 0.5, "left")
    alias_filter = optionalpha.FilterInfo(1, "alias", 0.5, "left")
    first = optionalpha.Split(mask, [first_filter], [{parent_a, parent_b}])
    second = optionalpha.Split(
        mask.clone(), [alias_filter], [{parent_b, parent_a}, {parent_a}]
    )
    first.score = 3.0
    second.score = -4.0

    merged = optionalpha._merge_identical_splits([first, second])

    assert len(merged) == 1
    assert merged[0].filters == [first_filter, alias_filter]
    assert merged[0].score == 3.0
    assert {frozenset(parents) for parents in merged[0].parents} == {
        frozenset([parent_a, parent_b]),
        frozenset([parent_a]),
    }
    assert len(merged[0].parents) == 2
    assert first.parents == [{parent_a, parent_b}]
    assert second.parents == [{parent_a, parent_b}, {parent_a}]


def test_boolean_pruning_without_beam_leaves_scores_unmodified():
    """Exhaustive preprocessing does not require or invoke a score function."""
    splits = [
        optionalpha.Split(torch.tensor([True, False]), [], []),
        optionalpha.Split(torch.tensor([False, True]), [], []),
    ]
    splits[1].score = 3.0

    def reject_scoring(*args):
        del args
        pytest.fail("A missing beam must leave boolean scores unmodified")

    retained = optionalpha._prune_boolean_splits(
        splits, None, reject_scoring, None, None, None
    )

    assert retained == splits
    assert [split.score for split in retained] == [None, 3.0]


def test_boolean_pruning_scores_in_bounded_batches_and_selects_global_best():
    """The beam compares all batches while preserving previously assigned scores."""
    indices = torch.arange(2051)
    powers = torch.arange(12)
    masks = (indices.unsqueeze(1).bitwise_right_shift(powers) & 1).bool()
    splits = [optionalpha.Split(mask, [], []) for mask in masks]
    splits[0].score = 99999.0
    profits = (2.0**powers).float()
    returns = torch.zeros(12)
    date_ordinals = torch.arange(12)
    batch_sizes = []

    def score_binary_value(profits, returns, masks, date_ordinals):
        del returns, date_ordinals
        batch_sizes.append(masks.shape[0])
        return (masks.float() * profits).sum(dim=1)

    retained = optionalpha._prune_boolean_splits(
        splits, 3, score_binary_value, profits, returns, date_ordinals
    )

    assert retained == [splits[0], splits[2050], splits[2049]]
    assert [split.score for split in retained] == [99999.0, 2050.0, 2049.0]
    assert sum(batch_sizes) == 2050
    assert max(batch_sizes) <= 1024
    previous_batches = batch_sizes.copy()

    rescored = optionalpha._prune_boolean_splits(
        splits, 3, score_binary_value, profits, returns, date_ordinals
    )

    assert rescored == retained
    assert batch_sizes == previous_batches


def test_boolean_pruning_retained_masks_do_not_pin_rejected_storage():
    """A surviving row must release the full child tensor containing rejected masks."""
    masks = torch.ones((10, 9), dtype=torch.bool)
    parent = optionalpha.Split(torch.ones(9, dtype=torch.bool), [], [])
    splits = [optionalpha.Split(mask, [], [{parent}]) for mask in masks]
    for index, split in enumerate(splits):
        split.score = float(index)

    retained = optionalpha._prune_boolean_splits(
        splits,
        2,
        _prefer_fewer_samples,
        torch.zeros(9),
        torch.zeros(9),
        torch.arange(9),
    )

    assert retained == [splits[9], splits[8]]
    for split in retained:
        assert split.parents == [{parent}]
        assert split.mask.untyped_storage().nbytes() == (
            split.mask.numel() * split.mask.element_size()
        )


@pytest.mark.parametrize(
    "missing_input", ["score_func", "profits", "returns", "date_ordinals"]
)
def test_boolean_pruning_requires_complete_score_inputs(missing_input):
    """A requested beam cannot silently skip scoring when its inputs are missing."""
    splits = [optionalpha.Split(torch.tensor([True, False]), [], [])]
    options = {
        "score_func": _prefer_fewer_samples,
        "profits": torch.zeros(2),
        "returns": torch.zeros(2),
        "date_ordinals": torch.arange(2),
    }
    options[missing_input] = None

    with pytest.raises(ValueError, match="Boolean pruning requires"):
        optionalpha._prune_boolean_splits(splits, 1, **options)
