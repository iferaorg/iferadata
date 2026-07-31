import math

import polars as pl
import pytest

from ifera.prop_firm_meta_strategy import (
    BATCH_OUTPUT_COLUMNS,
    CSV_INPUT_COLUMNS,
    EvaluationPhase,
    FundedInitialPhase,
    FundedPayoutLoopPhase,
    PropFirmMetaStrategy,
    evaluate_prop_firm_meta_strategy,
    evaluate_prop_firm_meta_strategies_from_csv,
)


def test_deterministic_finite_payout_loop():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=100.0,
            reset_fee=50.0,
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=1_000.0,
            max_payout_count=3,
            pass_rate_percent=100.0,
            average_days_to_pass=10.0,
            average_days_to_fail=0.0,
        ),
    )

    stats = evaluate_prop_firm_meta_strategy(strategy)

    assert stats.expected_payout_count == 3.0
    assert stats.expected_total_payouts == 3_000.0
    assert stats.expected_total_fees == 100.0
    assert stats.expected_net_profit == 2_900.0
    assert stats.total_ev_percent == 2_900.0
    assert stats.expected_duration_days == 30.0
    assert stats.annualized_net_profit == pytest.approx(35_283.333333333336)
    assert stats.annualized_ev_percent == pytest.approx(35_283.333333333336)
    assert stats.probability_of_profit == 1.0
    assert stats.profit_probability_method == "exact"
    assert stats.terminates_almost_surely


def test_evaluation_reset_distribution_drives_profit_probability():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=100.0,
            reset_fee=20.0,
            pass_rate_percent=50.0,
            average_days_to_pass=2.0,
            average_days_to_fail=1.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=100.0,
            average_days_to_pass=3.0,
            average_days_to_fail=0.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=200.0,
            max_payout_count=1,
            pass_rate_percent=100.0,
            average_days_to_pass=5.0,
            average_days_to_fail=0.0,
        ),
    )

    stats = evaluate_prop_firm_meta_strategy(strategy)

    assert stats.phase1.expected_reset_count == 1.0
    assert stats.phase1.expected_reset_fees == 20.0
    assert stats.phase1.expected_days == 3.0
    assert stats.expected_reset_count == 1.0
    assert stats.expected_total_fees == 120.0
    assert stats.expected_net_profit == 80.0
    assert stats.expected_duration_days == 11.0
    assert stats.probability_of_profit == pytest.approx(0.96875)
    assert stats.profit_probability_method == "exact"


def test_funded_initial_failures_reset_to_evaluation():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=100.0,
            reset_fee=50.0,
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=50.0,
            average_days_to_pass=1.0,
            average_days_to_fail=2.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=200.0,
            max_payout_count=1,
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
    )

    stats = evaluate_prop_firm_meta_strategy(strategy)

    assert stats.phase2.expected_reset_count == 1.0
    assert stats.phase2.expected_reset_fees == 50.0
    assert stats.phase2.expected_days == 3.0
    assert stats.expected_total_fees == 150.0
    assert stats.expected_net_profit == 50.0
    assert stats.expected_duration_days == 3.0
    assert stats.probability_of_profit == pytest.approx(0.75)


def test_fractional_pass_rates_are_treated_as_probabilities():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=98.0,
            reset_fee=95.0,
            pass_rate_percent=0.5,
            average_days_to_pass=60.0,
            average_days_to_fail=30.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=0.5,
            average_days_to_pass=60.0,
            average_days_to_fail=40.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=3_500.0,
            max_payout_count=5,
            pass_rate_percent=0.6,
            average_days_to_pass=60.0,
            average_days_to_fail=40.0,
        ),
    )

    terminal_probability = 0.6**5
    expected_failed_runs = (1.0 - terminal_probability) / terminal_probability
    failed_streak_mean = sum(
        streak_length * (0.6**streak_length) * 0.4 for streak_length in range(5)
    ) / (1.0 - terminal_probability)
    expected_payouts = 5.0 + expected_failed_runs * failed_streak_mean
    expected_resets = 3.0 + expected_failed_runs * 4.0
    expected_duration = 280.0 + expected_payouts * 60.0 + expected_failed_runs * 320.0
    expected_net_profit = expected_payouts * 3_500.0 - 98.0 - expected_resets * 95.0

    stats = evaluate_prop_firm_meta_strategy(strategy)

    assert stats.phase1.expected_reset_count == 1.0
    assert stats.phase2.expected_reset_count == 2.0
    assert stats.phase3.expected_reset_count_per_payout == pytest.approx(8.0 / 3.0)
    assert stats.expected_payout_count == pytest.approx(expected_payouts)
    assert stats.expected_reset_count == pytest.approx(expected_resets)
    assert stats.expected_duration_days == pytest.approx(expected_duration)
    assert stats.expected_net_profit == pytest.approx(expected_net_profit)
    assert stats.annual.expected_net_profit == pytest.approx(
        expected_net_profit * 365.0 / expected_duration
    )


def test_finite_payout_cap_counts_consecutive_payouts_after_resets():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=100.0,
            reset_fee=10.0,
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=100.0,
            max_payout_count=2,
            pass_rate_percent=50.0,
            average_days_to_pass=2.0,
            average_days_to_fail=1.0,
        ),
    )

    stats = evaluate_prop_firm_meta_strategy(strategy)

    assert stats.expected_payout_count == 3.0
    assert stats.expected_total_payouts == 300.0
    assert stats.expected_reset_count == 3.0
    assert stats.expected_total_fees == 130.0
    assert stats.expected_net_profit == 170.0
    assert stats.expected_duration_days == 9.0
    assert stats.profit_probability_method == "normal_approximation"
    assert "annual=AnnualStats" in repr(stats)
    assert "expected_duration_days" not in repr(stats)


def test_uncapped_perfect_payout_loop_is_infinite_but_annualized():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=100.0,
            reset_fee=50.0,
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=100.0,
            average_days_to_pass=0.0,
            average_days_to_fail=0.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=1_000.0,
            max_payout_count=None,
            pass_rate_percent=100.0,
            average_days_to_pass=10.0,
            average_days_to_fail=0.0,
        ),
    )

    stats = evaluate_prop_firm_meta_strategy(strategy)

    assert math.isinf(stats.expected_payout_count)
    assert math.isinf(stats.expected_total_payouts)
    assert stats.expected_total_fees == 100.0
    assert math.isinf(stats.expected_net_profit)
    assert math.isinf(stats.expected_duration_days)
    assert stats.annualized_net_profit == 36_500.0
    assert stats.annualized_ev_percent == 36_500.0
    assert stats.probability_of_profit == 1.0
    assert stats.profit_probability_method == "long_run_drift"
    assert not stats.terminates_almost_surely


def test_zero_evaluation_pass_rate_never_reaches_funded_account():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=100.0,
            reset_fee=10.0,
            pass_rate_percent=0.0,
            average_days_to_pass=5.0,
            average_days_to_fail=2.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=100.0,
            average_days_to_pass=1.0,
            average_days_to_fail=1.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=1_000.0,
            max_payout_count=1,
            pass_rate_percent=100.0,
            average_days_to_pass=10.0,
            average_days_to_fail=1.0,
        ),
    )

    stats = evaluate_prop_firm_meta_strategy(strategy)

    assert stats.expected_payout_count == 0.0
    assert stats.expected_total_payouts == 0.0
    assert math.isinf(stats.expected_total_fees)
    assert stats.expected_net_profit == -math.inf
    assert stats.expected_fee_return_percent == -100.0
    assert math.isinf(stats.expected_duration_days)
    assert stats.probability_of_profit == 0.0
    assert stats.profit_probability_method == "unreachable_evaluation"
    assert not stats.terminates_almost_surely


def test_invalid_percent_raises_value_error():
    strategy = PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=100.0,
            reset_fee=10.0,
            pass_rate_percent=101.0,
            average_days_to_pass=5.0,
            average_days_to_fail=2.0,
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=100.0,
            average_days_to_pass=1.0,
            average_days_to_fail=1.0,
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=1_000.0,
            max_payout_count=1,
            pass_rate_percent=100.0,
            average_days_to_pass=10.0,
            average_days_to_fail=1.0,
        ),
    )

    with pytest.raises(ValueError, match="evaluation.pass_rate_percent"):
        evaluate_prop_firm_meta_strategy(strategy)


def test_batch_csv_evaluates_rows_in_input_order_and_writes_output(tmp_path):
    input_path = tmp_path / "prop_firm_inputs.csv"
    output_path = tmp_path / "prop_firm_outputs.csv"
    rows = [
        [
            "Firm A",
            "Eval 50k",
            "Mean reversion",
            98,
            95,
            0.5,
            60,
            30,
            0.5,
            60,
            40,
            3500,
            5,
            0.6,
            60,
            40,
        ],
        [
            "Firm B",
            "Direct",
            "Momentum",
            100,
            25,
            100,
            0,
            0,
            100,
            0,
            0,
            1000,
            2,
            50,
            2,
            1,
        ],
    ]
    csv_text = ",".join(CSV_INPUT_COLUMNS) + "\n"
    csv_text += "\n".join(",".join(str(value) for value in row) for row in rows)
    input_path.write_text(csv_text, encoding="utf-8")

    output_df = evaluate_prop_firm_meta_strategies_from_csv(input_path, output_path)

    assert output_df.columns == list(BATCH_OUTPUT_COLUMNS)
    assert output_df.select(list(CSV_INPUT_COLUMNS[:3])).rows() == [
        ("Firm A", "Eval 50k", "Mean reversion"),
        ("Firm B", "Direct", "Momentum"),
    ]
    assert output_path.exists()
    assert output_df["first_payout_probability_without_reset"].to_list() == [
        pytest.approx(0.15),
        pytest.approx(0.5),
    ]
    assert output_df["first_payout_expected_days"].to_list()[0] == pytest.approx(
        83.0 / 0.15
    )
    assert output_df["first_payout_expected_net_profit"].to_list()[0] == pytest.approx(
        3500.0 - 98.0 - ((1.0 - 0.15) / 0.15) * 95.0
    )
    assert output_df["first_payout_probability_of_profit"].to_list()[
        0
    ] == pytest.approx(1.0 - 0.85**36)

    written_df = pl.read_csv(output_path)
    assert written_df.columns == output_df.columns
    assert written_df.select(list(CSV_INPUT_COLUMNS[:3])).rows() == [
        ("Firm A", "Eval 50k", "Mean reversion"),
        ("Firm B", "Direct", "Momentum"),
    ]


def test_batch_csv_allows_blank_max_payout_count(tmp_path):
    input_path = tmp_path / "uncapped.csv"
    row = [
        "Firm C",
        "Uncapped",
        "Carry",
        100,
        50,
        100,
        0,
        0,
        100,
        0,
        0,
        1000,
        "",
        100,
        10,
        0,
    ]
    input_path.write_text(
        ",".join(CSV_INPUT_COLUMNS) + "\n" + ",".join(str(value) for value in row),
        encoding="utf-8",
    )

    output_df = evaluate_prop_firm_meta_strategies_from_csv(input_path)

    assert output_df["terminates_almost_surely"].to_list() == [False]
    assert output_df["annual_expected_net_profit"].to_list() == [36500.0]
