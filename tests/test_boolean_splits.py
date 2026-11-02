"""Regression tests for boolean filter evaluation and effective split depth."""

from datetime import date, timedelta
from itertools import product
import re

import polars as pl
import pytest
import torch

from ifera.optionalpha import (
    FilterInfo,
    Split,
    SplitGenerator,
    _compute_child_parent_sets,
    _merge_identical_splits,
    prepare_splits,
)


def _tables(columns):
    """Use one weekday and constant risk to keep computed filters constant."""
    count = len(next(iter(columns.values())))
    dates = [date(2022, 1, 3) + timedelta(weeks=index) for index in range(count)]
    trades = pl.DataFrame(
        {"date": dates, "risk": [100.0] * count, "profit": [10.0] * count}
    )
    return trades, pl.DataFrame({"date": dates, **columns})


def _zero_score(profits, returns, masks, date_ordinals):
    del profits, returns, date_ordinals
    return torch.zeros(masks.shape[0], dtype=torch.float32, device=masks.device)


def _prefer_fewer_samples(profits, returns, masks, date_ordinals):
    del profits, returns, date_ordinals
    return -masks.float().sum(dim=1)


def _generate(columns, **kwargs):
    trades, filters = _tables(columns)
    kwargs.setdefault("pregenerate_boolean_conjunctions", True)
    generator = SplitGenerator(
        spread_width=20, left_only_filters=list(columns), **kwargs
    )
    return generator.generate(trades, filters)[2]


def _mask_keys(splits):
    return {tuple(split.mask.tolist()) for split in splits}


def _find_mask(splits, expected):
    matches = [split for split in splits if split.mask.tolist() == expected]
    assert len(matches) == 1
    return matches[0]


def _term_names(split):
    return {
        frozenset(condition[1] for condition in term) for term in split._get_dnf_terms()
    }


@pytest.mark.parametrize("values", [[False, True], [0, 1], [-2, 7]])
@pytest.mark.parametrize("pregenerate_boolean_conjunctions", [False, True])
def test_two_value_filters_always_pass_evaluation(
    values, pregenerate_boolean_conjunctions
):
    """Exactly two raw values bypass score improvement in either direction."""
    columns = {"flag": values * 30, "continuous": list(range(60))}
    trades, filters = _tables(columns)
    generator = SplitGenerator(
        spread_width=20,
        score_func=_zero_score,
        min_score_improvement=1.0,
        filter_eval_folds=5,
        filter_eval_repeats=1,
        pregenerate_boolean_conjunctions=pregenerate_boolean_conjunctions,
    )

    _, _, splits = generator.generate(trades, filters)

    assert len(splits) == 2
    assert {info.filter_name for split in splits for info in split.filters} == {"flag"}
    assert {info.direction for split in splits for info in split.filters} == {
        "left",
        "right",
    }


@pytest.mark.parametrize("pregenerate_boolean_conjunctions", [None, False])
@pytest.mark.parametrize("max_depth, expected_count", [(1, 3), (2, 6), (3, 7)])
def test_disabled_boolean_pregeneration_uses_ordinary_depth(
    pregenerate_boolean_conjunctions, max_depth, expected_count
):
    """Default and explicit opt-out consume one depth per boolean predicate."""
    rows = list(product([0, 1], repeat=3))
    columns = {name: [row[index] for row in rows] for index, name in enumerate("abc")}
    trades, filters = _tables(columns)
    options = {}
    if pregenerate_boolean_conjunctions is not None:
        options["pregenerate_boolean_conjunctions"] = pregenerate_boolean_conjunctions
    generator = SplitGenerator(
        spread_width=20,
        left_only_filters=list(columns),
        max_depth=max_depth,
        **options,
    )

    splits = generator.generate(trades, filters)[2]

    assert len(splits) == expected_count
    assert (
        max(len(term) for split in splits for term in _term_names(split)) == max_depth
    )
    assert ((True,) + (False,) * 7 in _mask_keys(splits)) == (max_depth == 3)


def test_disabled_pregeneration_does_not_call_boolean_closure(monkeypatch):
    """The opt-out skips preprocessing while keeping boolean evaluation bypasses."""
    rows = list(product([0, 1], repeat=3))
    columns = {name: [row[index] for row in rows] for index, name in enumerate("abc")}
    trades, filters = _tables(columns)

    def reject_boolean_closure(*args, **kwargs):
        del args, kwargs
        pytest.fail("Boolean closure must not run when pregeneration is disabled")

    monkeypatch.setattr(
        "ifera.optionalpha._generate_boolean_splits", reject_boolean_closure
    )
    generator = SplitGenerator(
        spread_width=20,
        left_only_filters=list(columns),
        max_depth=3,
        score_func=_zero_score,
        min_score_improvement=1.0,
    )

    splits = generator.generate(trades, filters)[2]

    assert len(splits) == 7
    assert (True,) + (False,) * 7 in _mask_keys(splits)


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_disabled_pregeneration_counts_numeric_and_boolean_conditions(max_depth):
    """Two boolean predicates plus one numeric predicate require three depths."""
    rows = list(product([0, 1], [0, 1], [0, 1, 2]))
    columns = {
        name: [row[index] for row in rows]
        for index, name in enumerate(["a", "b", "numeric"])
    }

    splits = _generate(
        columns, max_depth=max_depth, pregenerate_boolean_conjunctions=False
    )

    expected = tuple(all(value == 0 for value in row) for row in rows)
    assert (expected in _mask_keys(splits)) == (max_depth == 3)


@pytest.mark.parametrize("pregenerate_boolean_conjunctions", [None, False, True])
def test_prepare_splits_forwards_boolean_pregeneration(
    pregenerate_boolean_conjunctions,
):
    """The public wrapper defaults to ordinary depth and forwards an explicit choice."""
    columns = {"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}
    trades, filters = _tables(columns)
    options = {}
    if pregenerate_boolean_conjunctions is not None:
        options["pregenerate_boolean_conjunctions"] = pregenerate_boolean_conjunctions

    splits = prepare_splits(
        trades,
        filters,
        20,
        list(columns),
        [],
        torch.device("cpu"),
        torch.float32,
        max_depth=1,
        **options,
    )[2]

    enabled = pregenerate_boolean_conjunctions is True
    assert len(splits) == (3 if enabled else 2)
    assert ((True, False, False, False) in _mask_keys(splits)) == enabled


def test_enabled_pregeneration_prints_total_and_original_counts(capsys):
    """The depth-one diagnostic accounts separately for numeric and boolean atoms."""
    rows = list(product([0, 1], [0, 1], [0, 1, 2], [0, 1, 2]))
    columns = {
        name: [row[index] for row in rows]
        for index, name in enumerate(["a", "b", "numeric_a", "numeric_b"])
    }

    splits = _generate(columns, max_depth=1)

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "depth 1 splits" in line
    )
    assert len(splits) == 7
    assert line.startswith("Generated 7 depth 1 splits")
    assert [int(number) for number in re.findall(r"\d+", line)] == [7, 1, 4, 2, 1]
    assert "numeric" in line and "boolean" in line and "conjunction" in line


@pytest.mark.parametrize(
    "columns, expected_counts",
    [
        (
            {
                "a": [0, 0, 1, 1],
                "a_alias": [0, 0, 1, 1],
                "b": [0, 1, 0, 1],
            },
            [4, 1, 0, 3, 1],
        ),
        (
            {
                "a": [0, 0, 1, 1],
                "b": [0, 1, 0, 1],
                "conjunction_alias": [0, 1, 1, 1],
            },
            [3, 1, 0, 3, 0],
        ),
        ({"numeric": [0, 1, 2]}, [2, 1, 2, 0, 0]),
    ],
)
def test_enabled_print_counts_distinguish_original_masks_from_conjunctions(
    columns, expected_counts, capsys
):
    """Original aliases count as atoms, and only new boolean masks count as extras."""
    _generate(columns, max_depth=1)

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "depth 1 splits" in line
    )
    assert [int(number) for number in re.findall(r"\d+", line)] == expected_counts


def test_enabled_print_counts_numeric_candidates_before_evaluation(capsys):
    """The generation total includes numeric predicates later rejected by evaluation."""
    splits = _generate(
        {
            "a": [0, 0, 1, 1] * 15,
            "b": [0, 1, 0, 1] * 15,
            "numeric": [0, 1, 2] * 20,
        },
        max_depth=1,
        score_func=_zero_score,
        min_score_improvement=1.0,
        filter_eval_folds=5,
        filter_eval_repeats=1,
    )

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "depth 1 splits" in line
    )
    assert len(splits) == 3
    assert [int(number) for number in re.findall(r"\d+", line)] == [5, 1, 2, 2, 1]


def test_disabled_pregeneration_keeps_original_depth_one_printout(capsys):
    """The default diagnostic keeps its prior format without conjunction details."""
    trades, filters = _tables({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})

    SplitGenerator(spread_width=20, left_only_filters=["a", "b"]).generate(
        trades, filters
    )

    assert "Generated 2 depth 1 splits.\n" in capsys.readouterr().out


def test_raw_cardinality_not_rounded_buckets_defines_boolean():
    """Rounding a continuous column to two buckets must not bypass evaluation."""
    splits = _generate(
        {"continuous": [0.0, 0.001, 1.0] * 20},
        filter_granularities={"continuous": 1.0},
        score_func=_zero_score,
        min_score_improvement=1.0,
        filter_eval_folds=5,
        filter_eval_repeats=1,
    )

    assert splits == []


@pytest.mark.parametrize("boolean_count", [2, 3, 4])
def test_all_boolean_conjunctions_fit_depth_one(boolean_count):
    """The entire boolean closure is available at effective depth one."""
    rows = list(product([0, 1], repeat=boolean_count))
    columns = {
        f"flag_{index}": [row[index] for row in rows] for index in range(boolean_count)
    }

    splits = _generate(columns, max_depth=1)

    expected = [all(value == 0 for value in row) for row in rows]
    conjunction = _find_mask(splits, expected)
    assert frozenset(columns) in _term_names(conjunction)
    assert len(splits) == (2**boolean_count) - 1
    assert len(_mask_keys(splits)) == len(splits)
    for split in splits:
        for parents in split.parents:
            assert all(parent.parents == [] for parent in parents)
            assert torch.equal(
                split.mask, torch.stack([parent.mask for parent in parents]).all(dim=0)
            )


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_boolean_group_consumes_one_depth_alongside_numeric_filters(max_depth):
    """Each numeric predicate consumes another depth after the boolean group."""
    rows = list(product([0, 1], [0, 1], [0, 1, 2], [0, 1, 2]))
    columns = {
        name: [row[index] for row in rows]
        for index, name in enumerate(["flag_a", "flag_b", "numeric_a", "numeric_b"])
    }

    splits = _generate(columns, max_depth=max_depth)

    boolean_only = [a == b == 0 for a, b, _, _ in rows]
    one_numeric = [a == b == numeric_a == 0 for a, b, numeric_a, _ in rows]
    two_numeric = [all(value == 0 for value in row) for row in rows]
    keys = _mask_keys(splits)
    assert tuple(boolean_only) in keys
    assert (tuple(one_numeric) in keys) == (max_depth >= 2)
    assert (tuple(two_numeric) in keys) == (max_depth >= 3)


def test_boolean_closure_respects_min_samples():
    """Atoms can pass min_samples while their conjunction is excluded."""
    splits = _generate(
        {"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]},
        max_depth=1,
        min_samples=2,
    )

    assert len(splits) == 2
    assert all(split.sample_count == 2 for split in splits)
    assert all(split.parents == [] for split in splits)


def test_boolean_closure_respects_subsets():
    """A conjunction equal to a parent does not acquire redundant parents."""
    splits = _generate({"a": [0, 0, 0, 1], "b": [0, 0, 1, 1]}, max_depth=1)

    assert len(splits) == 2
    assert all(split.parents == [] for split in splits)


@pytest.mark.parametrize("similarity_threshold, expected_count", [(0.6, 2), (0.61, 3)])
def test_boolean_closure_respects_similarity(similarity_threshold, expected_count):
    """The same similarity cutoff applies during boolean preprocessing."""
    splits = _generate(
        {"a": [0, 0, 0, 1, 1], "b": [0, 0, 1, 0, 1]},
        max_depth=1,
        similarity_threshold=similarity_threshold,
    )

    assert len(splits) == expected_count


def test_identical_boolean_children_merge_and_preserve_parent_alternatives():
    """Distinct conjunctions with identical rows share a split and keep both terms."""
    splits = _generate(
        {"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "c": [0, 0, 1, 0]},
        max_depth=1,
    )

    merged = _find_mask(splits, [True, False, False, False])
    assert {frozenset(["a", "b"]), frozenset(["b", "c"])} <= _term_names(merged)
    assert len(_mask_keys(splits)) == len(splits)
    parent_sets = [frozenset(parents) for parents in merged.parents]
    assert len(parent_sets) == len(set(parent_sets))


def test_keep_best_n_can_reach_deeper_boolean_conjunctions():
    """A retained boolean frontier can keep improving through three-filter conjunctions."""
    rows = list(product([0, 1], repeat=3))
    columns = {name: [row[index] for row in rows] for index, name in enumerate("abc")}

    splits = _generate(
        columns,
        max_depth=1,
        score_func=_prefer_fewer_samples,
        keep_best_n=1,
        filter_eval_folds=5,
        filter_eval_repeats=1,
    )

    assert len(splits) == 1
    assert splits[0].mask.tolist() == [True] + [False] * 7
    assert splits[0].score == -1.0


def test_reusing_generator_recomputes_boolean_columns():
    """A previously boolean column must be evaluated when its cardinality changes."""
    generator = SplitGenerator(
        spread_width=20,
        left_only_filters=["flag"],
        score_func=_zero_score,
        min_score_improvement=1.0,
        filter_eval_folds=5,
        filter_eval_repeats=1,
    )
    first = _tables({"flag": [0, 1] * 30})
    second = _tables({"flag": [0, 1, 2] * 20})

    first_splits = generator.generate(*first)[2]
    second_splits = generator.generate(*second)[2]

    assert len(first_splits) == 1
    assert second_splits == []


def test_without_boolean_columns_numeric_depth_stays_unchanged():
    """Continuous-only inputs still add one numeric predicate per depth."""
    rows = list(product([0, 1, 2], repeat=2))
    columns = {"a": [a for a, _ in rows], "b": [b for _, b in rows]}

    depth_one = _generate(columns, max_depth=1)
    depth_two = _generate(columns, max_depth=2)

    assert len(depth_one) == 4
    assert len(depth_two) == 8
    assert (True,) + (False,) * 8 not in _mask_keys(depth_one)
    assert (True,) + (False,) * 8 in _mask_keys(depth_two)


def test_merged_atom_and_boolean_child_keep_both_dnf_representations():
    """A mask matching a numeric atom still renders its boolean conjunction alias."""
    a = Split(
        torch.tensor([True, True, True, True, False, False, False, False]),
        [FilterInfo(0, "a", 0.5, "left")],
        [],
    )
    b = Split(
        torch.tensor([True, True, False, False, True, True, False, False]),
        [FilterInfo(1, "b", 0.5, "left")],
        [],
    )
    numeric = Split(a.mask & b.mask, [FilterInfo(2, "numeric", 0.5, "left")], [])
    conjunction = Split(a.mask & b.mask, [], [{a, b}])

    merged = _merge_identical_splits([numeric, conjunction])[0]

    assert _term_names(merged) == {
        frozenset(["numeric"]),
        frozenset(["a", "b"]),
    }
    extra = Split(
        torch.tensor([True, False, True, False, True, False, True, False]),
        [FilterInfo(3, "extra", 0.5, "left")],
        [],
    )
    child = Split(
        merged.mask & extra.mask,
        [],
        _compute_child_parent_sets(merged, extra),
    )
    assert _term_names(child) == {
        frozenset(["numeric", "extra"]),
        frozenset(["a", "b", "extra"]),
    }
    for parents in child.parents:
        assert child not in parents
        assert torch.equal(
            child.mask, torch.stack([parent.mask for parent in parents]).all(dim=0)
        )


@pytest.mark.parametrize("pregenerate_boolean_conjunctions", [False, True])
def test_boolean_columns_are_not_offered_to_cross_validation(
    monkeypatch, pregenerate_boolean_conjunctions
):
    """Only continuous columns need the score improvement evaluation."""
    trades, filters = _tables({"flag": [-2, 7] * 30, "continuous": list(range(60))})
    generator = SplitGenerator(
        spread_width=20,
        score_func=_zero_score,
        filter_eval_folds=5,
        filter_eval_repeats=1,
        pregenerate_boolean_conjunctions=pregenerate_boolean_conjunctions,
    )
    original_evaluator = generator._evaluate_filters
    offered_names = set()

    def record_evaluation(splits, *args, **kwargs):
        offered_names.update(
            info.filter_name for split in splits for info in split.filters
        )
        return original_evaluator(splits, *args, **kwargs)

    monkeypatch.setattr(generator, "_evaluate_filters", record_evaluation)

    generator.generate(trades, filters)

    assert offered_names == {"continuous"}


def test_boolean_closure_handles_both_filter_directions():
    """Every nonempty partial assignment of three boolean filters is generated."""
    rows = list(product([0, 1], repeat=3))
    columns = {name: [row[index] for row in rows] for index, name in enumerate("abc")}
    trades, filters = _tables(columns)
    generator = SplitGenerator(
        spread_width=20, max_depth=1, pregenerate_boolean_conjunctions=True
    )

    splits = generator.generate(trades, filters)[2]

    expected = {
        tuple(
            all(
                value is None or row[index] == value
                for index, value in enumerate(terms)
            )
            for row in rows
        )
        for terms in product([None, 0, 1], repeat=3)
        if any(value is not None for value in terms)
    }
    assert len(splits) == 26
    assert _mask_keys(splits) == expected


def test_weekday_and_skip_filters_pass_evaluation_and_share_one_depth():
    """The motivating weekday exclusions combine with another boolean at depth one."""
    dates = [
        date(2022, 1, 3) + timedelta(weeks=week, days=weekday)
        for week in range(12)
        for weekday in range(5)
    ]
    skip_values = [index % 2 for index in range(len(dates))]
    trades = pl.DataFrame(
        {"date": dates, "risk": [100.0] * len(dates), "profit": [10.0] * len(dates)}
    )
    filters = pl.DataFrame({"date": dates, "skip_FOMC": skip_values})
    generator = SplitGenerator(
        spread_width=20,
        left_only_filters=["skip_FOMC"],
        max_depth=1,
        score_func=_zero_score,
        min_score_improvement=1.0,
        filter_eval_folds=5,
        filter_eval_repeats=1,
        pregenerate_boolean_conjunctions=True,
    )

    splits = generator.generate(trades, filters)[2]

    expected = [
        day.weekday() not in [1, 2] and skip == 0
        for day, skip in zip(dates, skip_values)
    ]
    combined = _find_mask(splits, expected)
    assert frozenset(["is_tuesday", "is_wednesday", "skip_FOMC"]) in _term_names(
        combined
    )


def test_empty_evaluated_splits_do_not_stack_empty_masks():
    """Removing every continuous filter also works when deeper splits are requested."""
    splits = _generate(
        {"continuous": list(range(60))},
        max_depth=3,
        score_func=_zero_score,
        min_score_improvement=1.0,
        filter_eval_folds=5,
        filter_eval_repeats=1,
    )

    assert splits == []


@pytest.mark.parametrize(
    "atom_count, min_samples, expected_count",
    [(0, 1, 0), (1, 1, 1), (2, 1, 3), (2, 2, 2)],
)
def test_boolean_generation_helper_handles_empty_atoms_and_exclusions(
    atom_count, min_samples, expected_count
):
    """The closure helper supports empty inputs and honors child sample limits."""
    from ifera.optionalpha import _generate_boolean_splits

    atoms = [
        Split(
            torch.tensor(mask),
            [FilterInfo(index, f"flag_{index}", 0.5, "left")],
            [],
        )
        for index, mask in enumerate(
            [[True, True, False, False], [True, False, True, False]][:atom_count]
        )
    ]

    splits = _generate_boolean_splits(atoms, torch.device("cpu"), min_samples, 1.0)

    assert len(splits) == expected_count
    assert len(_mask_keys(splits)) == expected_count


def test_retained_numeric_seed_can_use_pruned_boolean_block_at_depth_two():
    """A numeric beam winner can still use a full boolean block omitted by pruning."""
    rows = list(product([0, 1], [0, 1], [0, 1, 2]))
    columns = {
        name: [row[index] for row in rows]
        for index, name in enumerate(["a", "b", "numeric"])
    }
    trades, filters = _tables(columns)
    profits = [
        100.0 if a == b == numeric == 0 else 10.0 if numeric == 0 else -100.0
        for a, b, numeric in rows
    ]
    trades = trades.with_columns(pl.Series("profit", profits))

    def mean_profit_score(profits, returns, masks, date_ordinals):
        del returns, date_ordinals
        totals = (masks.to(profits.dtype) * profits.unsqueeze(0)).sum(dim=1)
        return totals / masks.sum(dim=1).clamp_min(1)

    common_options = {
        "spread_width": 20,
        "left_only_filters": list(columns),
        "score_func": mean_profit_score,
        "keep_best_n": 1,
        "filter_eval_folds": 5,
        "filter_eval_repeats": 1,
        "pregenerate_boolean_conjunctions": True,
    }
    depth_one = SplitGenerator(max_depth=1, **common_options).generate(trades, filters)[
        2
    ]
    depth_two = SplitGenerator(max_depth=2, **common_options).generate(trades, filters)[
        2
    ]

    assert len(depth_one) == len(depth_two) == 1
    assert depth_one[0].mask.tolist() == [numeric == 0 for _, _, numeric in rows]
    assert _term_names(depth_one[0]) == {frozenset(["numeric"])}
    assert depth_one[0].score == 32.5
    assert depth_two[0].mask.tolist() == [
        all(value == 0 for value in row) for row in rows
    ]
    assert _term_names(depth_two[0]) == {frozenset(["a", "b", "numeric"])}
    assert depth_two[0].score == 100.0


def test_generated_numeric_boolean_aliases_survive_next_depth_without_parent_cycles():
    """Merging an atomic mask with a boolean conjunction preserves both child forms."""
    columns = {
        "a": [0, 0, 0, 0, 1, 1, 1, 1],
        "b": [0, 0, 1, 1, 0, 0, 1, 1],
        "numeric": [0, 0, 1, 1, 1, 1, 2, 2],
        "extra": [0, 1, 0, 1, 0, 1, 2, 2],
    }
    depth_one = _generate(columns, max_depth=1)
    depth_two = _generate(columns, max_depth=2)

    def check_parent_graph(split, ancestors):
        assert split not in ancestors
        for parents in split.parents:
            assert torch.equal(
                split.mask, torch.stack([parent.mask for parent in parents]).all(dim=0)
            )
            for parent in parents:
                check_parent_graph(parent, ancestors | {split})

    for split in depth_one + depth_two:
        check_parent_graph(split, set())
    base_alias = _find_mask(
        depth_one, [True, True, False, False, False, False, False, False]
    )
    assert _term_names(base_alias) == {
        frozenset(["numeric"]),
        frozenset(["a", "b"]),
    }
    child_alias = _find_mask(
        depth_two, [True, False, False, False, False, False, False, False]
    )
    assert _term_names(child_alias) == {
        frozenset(["numeric", "extra"]),
        frozenset(["a", "b", "extra"]),
    }
    assert len(_mask_keys(depth_two)) == len(depth_two)


def test_numeric_beam_can_add_boolean_group_after_two_numeric_conditions():
    """A boolean group may first enter the beam after two numeric conditions."""
    rows = list(product([0, 1], [0, 1, 2], [0, 1, 2]))
    columns = {
        name: [row[index] for row in rows]
        for index, name in enumerate(["flag", "numeric_a", "numeric_b"])
    }
    trades, filters = _tables(columns)
    profits = [
        (
            (100.0 if flag == 0 else 80.0)
            if numeric_a == numeric_b == 0
            else (-10.0 if flag == 0 else 70.0) if numeric_a == 0 else -100.0
        )
        for flag, numeric_a, numeric_b in rows
    ]
    trades = trades.with_columns(pl.Series("profit", profits))

    def mean_profit_score(profits, returns, masks, date_ordinals):
        del returns, date_ordinals
        totals = (masks.to(profits.dtype) * profits.unsqueeze(0)).sum(dim=1)
        return totals / masks.sum(dim=1).clamp_min(1)

    best_at_depth = {}
    for max_depth in [1, 2, 3]:
        generator = SplitGenerator(
            spread_width=20,
            left_only_filters=list(columns),
            max_depth=max_depth,
            score_func=mean_profit_score,
            keep_best_n=1,
            filter_eval_folds=5,
            filter_eval_repeats=1,
            pregenerate_boolean_conjunctions=True,
        )
        best_at_depth[max_depth] = generator.generate(trades, filters)[2][0]

    assert _term_names(best_at_depth[1]) == {frozenset(["numeric_a"])}
    assert best_at_depth[1].score == 50.0
    assert _term_names(best_at_depth[2]) == {frozenset(["numeric_a", "numeric_b"])}
    assert best_at_depth[2].score == 90.0
    assert best_at_depth[3].mask.tolist() == [
        all(value == 0 for value in row) for row in rows
    ]
    assert best_at_depth[3].score == 100.0


def test_boolean_atom_keeps_identical_conjunction_alias_without_parent_cycles():
    """A conjunction matching an existing boolean atom retains both explanations."""
    splits = _generate(
        {"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "c": [0, 1, 1, 1]},
        max_depth=1,
    )

    def check_parent_graph(split, ancestors):
        assert split not in ancestors
        for parents in split.parents:
            for parent in parents:
                check_parent_graph(parent, ancestors | {split})

    for split in splits:
        check_parent_graph(split, set())
    alias = _find_mask(splits, [True, False, False, False])
    assert _term_names(alias) == {frozenset(["c"]), frozenset(["a", "b"])}
    assert len(_mask_keys(splits)) == len(splits)


@pytest.mark.parametrize(
    "representation, expected",
    [
        ("numeric_atom", True),
        ("boolean_atom", False),
        ("numeric_child", True),
        ("mixed_child", False),
        ("numeric_alternative", True),
        ("nested_numeric_child", True),
        ("mixed_alias", True),
    ],
)
def test_numeric_only_representation_uses_any_valid_lineage(representation, expected):
    """Numeric aliases and numeric-only parent alternatives permit a later boolean block."""
    from ifera.optionalpha import _has_numeric_only_representation

    numeric_a = Split(
        torch.tensor([True, True, False, False]),
        [FilterInfo(0, "numeric_a", 0.5, "left")],
        [],
    )
    numeric_b = Split(
        torch.tensor([True, False, True, False]),
        [FilterInfo(1, "numeric_b", 0.5, "left")],
        [],
    )
    boolean = Split(
        torch.tensor([True, False, False, True]),
        [FilterInfo(2, "flag", 0.5, "left")],
        [],
    )
    numeric_child = Split(numeric_a.mask & numeric_b.mask, [], [{numeric_a, numeric_b}])
    mixed_child = Split(numeric_a.mask & boolean.mask, [], [{numeric_a, boolean}])
    numeric_alternative = Split(
        numeric_child.mask, [], [{numeric_a, boolean}, {numeric_a, numeric_b}]
    )
    nested_numeric_child = Split(numeric_child.mask, [], [{numeric_child, numeric_a}])
    mixed_alias = Split(
        numeric_child.mask,
        [FilterInfo(3, "numeric_alias", 0.5, "left")],
        [{numeric_a, boolean}],
    )
    target = {
        "numeric_atom": numeric_a,
        "boolean_atom": boolean,
        "numeric_child": numeric_child,
        "mixed_child": mixed_child,
        "numeric_alternative": numeric_alternative,
        "nested_numeric_child": nested_numeric_child,
        "mixed_alias": mixed_alias,
    }[representation]
    numeric_bases = {numeric_a, numeric_b, mixed_alias}
    cache = {}

    assert _has_numeric_only_representation(target, numeric_bases, cache) is expected
    cached_results = cache.copy()
    assert _has_numeric_only_representation(target, numeric_bases, cache) is expected
    assert cache == cached_results


def test_child_candidate_batches_merge_duplicate_masks_before_beam_limit():
    """Duplicate children across batches cannot consume the beam's distinct slots."""
    children = []
    for index in range(10):
        left = Split(
            torch.tensor([True, True, False, False]),
            [FilterInfo(index * 2, f"left_{index}", 0.5, "left")],
            [],
        )
        right = Split(
            torch.tensor([True, False, True, False]),
            [FilterInfo(index * 2 + 1, f"right_{index}", 0.5, "left")],
            [],
        )
        children.append(Split(left.mask & right.mask, [], [{left, right}]))
    other_left = Split(
        torch.tensor([True, True, True, False]),
        [FilterInfo(20, "other_left", 0.5, "left")],
        [],
    )
    other_right = Split(
        torch.tensor([True, True, False, True]),
        [FilterInfo(21, "other_right", 0.5, "left")],
        [],
    )
    children.append(
        Split(other_left.mask & other_right.mask, [], [{other_left, other_right}])
    )
    generator = SplitGenerator(
        spread_width=20, score_func=_prefer_fewer_samples, keep_best_n=2
    )
    profits = torch.ones(4)

    candidates = generator._select_child_candidates(
        children, [], profits, profits, torch.arange(4, dtype=torch.long)
    )

    assert len(_mask_keys(candidates)) == len(candidates)
    assert _mask_keys(candidates) == {
        (True, False, False, False),
        (True, True, False, False),
    }
    singleton = _find_mask(candidates, [True, False, False, False])
    assert len(singleton.parents) == 10
