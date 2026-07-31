"""Evaluate prop-firm account meta-strategies from phase-level statistics.

The model intentionally does not know anything about a concrete trading strategy.
Instead, each phase is represented by pass/fail probabilities and average
durations. Reset fees are counted uniformly whether the reset was caused by an
evaluation failure, a funded-initial failure, or a funded payout-loop failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import polars as pl

# The result objects are intentionally data-heavy: their job is to make the
# intermediate assumptions and final account-level metrics visible to callers.
# pylint: disable=too-few-public-methods,too-many-instance-attributes,too-many-lines
# pylint: disable=too-many-locals

TRADING_DAYS_PER_YEAR = 365.0
PathLike = Union[str, Path]

IDENTIFIER_COLUMNS = ("firm_name", "account_name", "strategy_name")
STRATEGY_INPUT_COLUMNS = (
    "new_account_fee",
    "reset_fee",
    "evaluation_pass_rate",
    "evaluation_average_days_to_pass",
    "evaluation_average_days_to_fail",
    "funded_initial_pass_rate",
    "funded_initial_average_days_to_pass",
    "funded_initial_average_days_to_fail",
    "max_payout",
    "max_payout_count",
    "payout_loop_pass_rate",
    "payout_loop_average_days_to_pass",
    "payout_loop_average_days_to_fail",
)
CSV_INPUT_COLUMNS = IDENTIFIER_COLUMNS + STRATEGY_INPUT_COLUMNS
BATCH_OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + (
    "annual_expected_payout_count",
    "annual_expected_total_payouts",
    "annual_expected_total_fees",
    "annual_expected_net_profit",
    "annual_ev_percent",
    "annual_expected_reset_count",
    "first_payout_expected_days",
    "first_payout_expected_net_profit",
    "first_payout_expected_total_fees",
    "first_payout_expected_reset_count",
    "first_payout_probability_without_reset",
    "first_payout_probability_of_profit",
    "first_payout_annualized_net_profit",
    "first_payout_annualized_ev_percent",
    "lifetime_expected_payout_count",
    "lifetime_expected_total_payouts",
    "lifetime_expected_total_fees",
    "lifetime_expected_net_profit",
    "lifetime_total_ev_percent",
    "lifetime_expected_fee_return_percent",
    "lifetime_expected_duration_days",
    "annualized_net_profit",
    "annualized_ev_percent",
    "probability_of_profit",
    "profit_probability_method",
    "expected_reset_count",
    "terminates_almost_surely",
    "phase1_expected_reset_count",
    "phase1_expected_reset_fees",
    "phase1_expected_days",
    "phase2_expected_reset_count",
    "phase2_expected_reset_fees",
    "phase2_expected_days",
    "phase3_expected_reset_count_per_payout",
    "phase3_expected_reset_fees_per_payout",
    "phase3_expected_days_per_payout",
    "phase3_expected_net_profit_per_payout",
)


@dataclass(frozen=True)
class EvaluationPhase:
    """Inputs for the evaluation phase before the account is funded.

    Pass rates may be provided as fractions from 0 to 1 or percentages from 1
    to 100. For example, both ``0.5`` and ``50.0`` represent a 50% pass rate.
    """

    new_account_fee: float
    reset_fee: float
    pass_rate_percent: float
    average_days_to_pass: float
    average_days_to_fail: float


@dataclass(frozen=True)
class FundedInitialPhase:
    """Inputs for the funded phase that tries to reach one max-payout unit.

    Pass rates may be provided as fractions from 0 to 1 or percentages from 1
    to 100. For example, both ``0.5`` and ``50.0`` represent a 50% pass rate.
    """

    pass_rate_percent: float
    average_days_to_pass: float
    average_days_to_fail: float


@dataclass(frozen=True)
class FundedPayoutLoopPhase:
    """Inputs for the funded loop that repeatedly earns one max-payout unit.

    Pass rates may be provided as fractions from 0 to 1 or percentages from 1
    to 100. For example, both ``0.6`` and ``60.0`` represent a 60% pass rate.
    """

    max_payout: float
    max_payout_count: Optional[int]
    pass_rate_percent: float
    average_days_to_pass: float
    average_days_to_fail: float


@dataclass(frozen=True)
class PropFirmMetaStrategy:
    """Complete input set for a prop-firm account meta-strategy."""

    evaluation: EvaluationPhase
    funded_initial: FundedInitialPhase
    payout_loop: FundedPayoutLoopPhase


@dataclass(frozen=True)
class AnnualStats:
    """Annualized expected account statistics."""

    period_days: float
    expected_payout_count: float
    expected_total_payouts: float
    expected_total_fees: float
    expected_net_profit: float
    ev_percent: float
    expected_reset_count: float


@dataclass(frozen=True)
class FirstPayoutStats:
    """Statistics for reaching the first payout from a fresh account."""

    expected_days: float
    expected_net_profit: float
    expected_total_fees: float
    expected_reset_count: float
    probability_without_reset: float
    probability_of_profit: float
    annualized_net_profit: float
    annualized_ev_percent: float


@dataclass(frozen=True)
class PhaseProgressStats:
    """Expected resets, reset fees, and days needed for one phase objective."""

    expected_reset_count: float
    expected_reset_fees: float
    expected_days: float


@dataclass(frozen=True)
class PayoutLoopStats:
    """Expected reset burden and duration for one successful payout cycle."""

    expected_reset_count_per_payout: float
    expected_reset_fees_per_payout: float
    expected_days_per_payout: float
    expected_net_profit_per_payout: float


@dataclass(frozen=True)
class PropFirmMetaStrategyStats:
    """Account-level and phase-level statistics for a meta-strategy."""

    annual: AnnualStats
    first_payout: FirstPayoutStats
    expected_payout_count: float = field(repr=False)
    expected_total_payouts: float = field(repr=False)
    expected_total_fees: float = field(repr=False)
    expected_net_profit: float = field(repr=False)
    total_ev_percent: float = field(repr=False)
    expected_fee_return_percent: float = field(repr=False)
    expected_duration_days: float = field(repr=False)
    annualized_net_profit: float
    annualized_ev_percent: float
    probability_of_profit: float
    profit_probability_method: str
    expected_reset_count: float = field(repr=False)
    terminates_almost_surely: bool
    phase1: PhaseProgressStats
    phase2: PhaseProgressStats
    phase3: PayoutLoopStats


@dataclass(frozen=True)
class _ResetMoments:
    """Mean and variance for reset counts in setup and payout cycles."""

    setup_mean: float
    setup_variance: float
    payout_cycle_mean: float
    payout_cycle_variance: float


@dataclass(frozen=True)
class _ConsecutivePayoutMoments:
    """Moments for Phase 3 payouts and failures before the account terminates."""

    expected_success_count: float
    success_count_variance: float
    expected_failure_count: float
    failure_count_variance: float
    success_failure_covariance: float


def evaluate_prop_firm_meta_strategy(
    strategy: PropFirmMetaStrategy,
    *,
    exact_probability_reset_limit: int = 2_000,
) -> PropFirmMetaStrategyStats:
    """Evaluate account-level EV, duration, and profit probability.

    ``total_ev_percent`` and ``annualized_ev_percent`` use the new account fee as
    their denominator. ``expected_fee_return_percent`` uses all expected fees,
    including resets, as its denominator.

    Finite ``max_payout_count`` values are treated as consecutive Phase 3
    payouts on the current funded account. A reset moves the payout streak back
    to zero, but payouts already received still count toward total account EV.

    Profit probability is exact for a one-payout cap and deterministic finite
    cases. Larger consecutive payout caps use a normal approximation because
    both payout count and reset count are random.
    """

    _validate_strategy(strategy, exact_probability_reset_limit)

    evaluation_probability = _percent_to_probability(
        strategy.evaluation.pass_rate_percent
    )
    initial_probability = _percent_to_probability(
        strategy.funded_initial.pass_rate_percent
    )
    payout_probability = _percent_to_probability(strategy.payout_loop.pass_rate_percent)

    phase1 = _calculate_phase1(strategy.evaluation, evaluation_probability)
    phase2 = _calculate_phase2(
        strategy.evaluation.reset_fee,
        strategy.funded_initial,
        initial_probability,
        phase1,
    )
    phase3 = _calculate_phase3(
        strategy.evaluation.reset_fee,
        strategy.payout_loop,
        payout_probability,
        phase1,
        phase2,
    )
    first_payout = _calculate_first_payout_stats(
        strategy,
        evaluation_probability,
        initial_probability,
        payout_probability,
    )

    if evaluation_probability == 0.0:
        return _unreachable_evaluation_stats(
            strategy,
            phase1,
            phase2,
            phase3,
            first_payout,
        )

    if initial_probability == 0.0:
        return _unreachable_funded_initial_stats(
            strategy,
            phase1,
            phase2,
            phase3,
            first_payout,
        )

    max_payout_count = strategy.payout_loop.max_payout_count
    if max_payout_count == 0:
        return _finite_payout_stats(
            strategy=strategy,
            payout_count=0,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            first_payout=_no_first_payout_stats(),
            profit_probability=0.0,
            profit_probability_method="deterministic",
        )

    if payout_probability == 0.0:
        return _unreachable_payout_loop_stats(
            strategy,
            phase1,
            phase2,
            phase3,
            first_payout,
        )

    if max_payout_count is None:
        return _unbounded_payout_loop_stats(
            strategy,
            phase1,
            phase2,
            phase3,
            first_payout,
        )

    return _finite_consecutive_payout_stats(
        strategy=strategy,
        consecutive_payout_count=max_payout_count,
        evaluation_probability=evaluation_probability,
        initial_probability=initial_probability,
        payout_probability=payout_probability,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        first_payout=first_payout,
        exact_probability_reset_limit=exact_probability_reset_limit,
    )


def evaluate_prop_firm_meta_strategies_from_csv(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    exact_probability_reset_limit: int = 2_000,
) -> pl.DataFrame:
    """Evaluate a CSV batch of prop-firm meta-strategies.

    The input CSV must contain columns in this order:

    ``firm_name``, ``account_name``, ``strategy_name``,
    ``new_account_fee``, ``reset_fee``, ``evaluation_pass_rate``,
    ``evaluation_average_days_to_pass``, ``evaluation_average_days_to_fail``,
    ``funded_initial_pass_rate``, ``funded_initial_average_days_to_pass``,
    ``funded_initial_average_days_to_fail``, ``max_payout``,
    ``max_payout_count``, ``payout_loop_pass_rate``,
    ``payout_loop_average_days_to_pass``, ``payout_loop_average_days_to_fail``.

    Pass rates may be fractions from 0 to 1 or percentage values above 1. A blank
    ``max_payout_count`` is treated as no payout cap. Rows are evaluated and
    returned in the same order as the input rows. When ``output_path`` is given,
    the same result frame is also written to that CSV path.
    """

    input_df = pl.read_csv(
        input_path,
        null_values=["", "null", "NULL", "none", "None", "nan", "NaN"],
    )
    _validate_batch_input_columns(input_df.columns)

    output_rows: list[dict[str, object]] = []
    for row in input_df.iter_rows(named=True):
        strategy = _strategy_from_csv_row(row)
        stats = evaluate_prop_firm_meta_strategy(
            strategy,
            exact_probability_reset_limit=exact_probability_reset_limit,
        )
        output_rows.append(_stats_to_output_row(row, stats))

    output_df = (
        pl.DataFrame(output_rows)
        if output_rows
        else pl.DataFrame(schema=_batch_output_schema())
    )
    output_df = output_df.select(list(BATCH_OUTPUT_COLUMNS))
    if output_path is not None:
        output_df.write_csv(output_path)
    return output_df


def _validate_batch_input_columns(columns: list[str]) -> None:
    """Validate that the batch input columns match the documented schema."""

    expected_columns = list(CSV_INPUT_COLUMNS)
    if columns != expected_columns:
        raise ValueError(
            "Input CSV columns must be exactly: " + ", ".join(expected_columns)
        )


def _strategy_from_csv_row(row: Mapping[str, object]) -> PropFirmMetaStrategy:
    """Build a strategy input object from a Polars CSV row."""

    return PropFirmMetaStrategy(
        evaluation=EvaluationPhase(
            new_account_fee=_float_from_row(row, "new_account_fee"),
            reset_fee=_float_from_row(row, "reset_fee"),
            pass_rate_percent=_float_from_row(row, "evaluation_pass_rate"),
            average_days_to_pass=_float_from_row(
                row,
                "evaluation_average_days_to_pass",
            ),
            average_days_to_fail=_float_from_row(
                row,
                "evaluation_average_days_to_fail",
            ),
        ),
        funded_initial=FundedInitialPhase(
            pass_rate_percent=_float_from_row(row, "funded_initial_pass_rate"),
            average_days_to_pass=_float_from_row(
                row,
                "funded_initial_average_days_to_pass",
            ),
            average_days_to_fail=_float_from_row(
                row,
                "funded_initial_average_days_to_fail",
            ),
        ),
        payout_loop=FundedPayoutLoopPhase(
            max_payout=_float_from_row(row, "max_payout"),
            max_payout_count=_optional_int_from_row(row, "max_payout_count"),
            pass_rate_percent=_float_from_row(row, "payout_loop_pass_rate"),
            average_days_to_pass=_float_from_row(
                row,
                "payout_loop_average_days_to_pass",
            ),
            average_days_to_fail=_float_from_row(
                row,
                "payout_loop_average_days_to_fail",
            ),
        ),
    )


def _stats_to_output_row(
    input_row: Mapping[str, object],
    stats: PropFirmMetaStrategyStats,
) -> dict[str, object]:
    """Flatten result dataclasses into a CSV-friendly row."""

    return {
        "firm_name": input_row["firm_name"],
        "account_name": input_row["account_name"],
        "strategy_name": input_row["strategy_name"],
        "annual_expected_payout_count": stats.annual.expected_payout_count,
        "annual_expected_total_payouts": stats.annual.expected_total_payouts,
        "annual_expected_total_fees": stats.annual.expected_total_fees,
        "annual_expected_net_profit": stats.annual.expected_net_profit,
        "annual_ev_percent": stats.annual.ev_percent,
        "annual_expected_reset_count": stats.annual.expected_reset_count,
        "first_payout_expected_days": stats.first_payout.expected_days,
        "first_payout_expected_net_profit": stats.first_payout.expected_net_profit,
        "first_payout_expected_total_fees": stats.first_payout.expected_total_fees,
        "first_payout_expected_reset_count": stats.first_payout.expected_reset_count,
        "first_payout_probability_without_reset": (
            stats.first_payout.probability_without_reset
        ),
        "first_payout_probability_of_profit": stats.first_payout.probability_of_profit,
        "first_payout_annualized_net_profit": (
            stats.first_payout.annualized_net_profit
        ),
        "first_payout_annualized_ev_percent": (
            stats.first_payout.annualized_ev_percent
        ),
        "lifetime_expected_payout_count": stats.expected_payout_count,
        "lifetime_expected_total_payouts": stats.expected_total_payouts,
        "lifetime_expected_total_fees": stats.expected_total_fees,
        "lifetime_expected_net_profit": stats.expected_net_profit,
        "lifetime_total_ev_percent": stats.total_ev_percent,
        "lifetime_expected_fee_return_percent": stats.expected_fee_return_percent,
        "lifetime_expected_duration_days": stats.expected_duration_days,
        "annualized_net_profit": stats.annualized_net_profit,
        "annualized_ev_percent": stats.annualized_ev_percent,
        "probability_of_profit": stats.probability_of_profit,
        "profit_probability_method": stats.profit_probability_method,
        "expected_reset_count": stats.expected_reset_count,
        "terminates_almost_surely": stats.terminates_almost_surely,
        "phase1_expected_reset_count": stats.phase1.expected_reset_count,
        "phase1_expected_reset_fees": stats.phase1.expected_reset_fees,
        "phase1_expected_days": stats.phase1.expected_days,
        "phase2_expected_reset_count": stats.phase2.expected_reset_count,
        "phase2_expected_reset_fees": stats.phase2.expected_reset_fees,
        "phase2_expected_days": stats.phase2.expected_days,
        "phase3_expected_reset_count_per_payout": (
            stats.phase3.expected_reset_count_per_payout
        ),
        "phase3_expected_reset_fees_per_payout": (
            stats.phase3.expected_reset_fees_per_payout
        ),
        "phase3_expected_days_per_payout": stats.phase3.expected_days_per_payout,
        "phase3_expected_net_profit_per_payout": (
            stats.phase3.expected_net_profit_per_payout
        ),
    }


def _batch_output_schema() -> dict[str, Any]:
    """Return the output schema used for empty batch results."""

    schema: dict[str, Any] = {column: pl.Float64 for column in BATCH_OUTPUT_COLUMNS}
    for column in IDENTIFIER_COLUMNS:
        schema[column] = pl.Utf8
    schema["profit_probability_method"] = pl.Utf8
    schema["terminates_almost_surely"] = pl.Boolean
    return schema


def _float_from_row(row: Mapping[str, object], column: str) -> float:
    """Read a required finite float-like value from a batch row."""

    value = _required_row_value(row, column)
    return _coerce_float(value, column)


def _coerce_float(value: object, column: str) -> float:
    """Coerce a CSV scalar to float with a column-specific error."""

    if isinstance(value, bool):
        raise ValueError(f"{column} must be numeric")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{column} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc


def _optional_int_from_row(row: Mapping[str, object], column: str) -> Optional[int]:
    """Read an optional integer from a batch row."""

    value = row.get(column)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{column} must be an integer or blank")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{column} must be an integer or blank")

    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be an integer or blank") from exc

    int_value = int(float_value)
    if int_value != float_value:
        raise ValueError(f"{column} must be an integer or blank")
    return int_value


def _required_row_value(row: Mapping[str, object], column: str) -> object:
    """Read a required non-blank value from a batch row."""

    value = row.get(column)
    if _is_blank_value(value):
        raise ValueError(f"{column} is required")
    return value


def _is_blank_value(value: object) -> bool:
    """Return whether a CSV value should be treated as blank."""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _calculate_phase1(
    evaluation: EvaluationPhase,
    pass_probability: float,
) -> PhaseProgressStats:
    """Calculate expected progress from evaluation start to funded initial."""

    if pass_probability == 0.0:
        reset_fees = math.inf if evaluation.reset_fee > 0.0 else 0.0
        return PhaseProgressStats(
            expected_reset_count=math.inf,
            expected_reset_fees=reset_fees,
            expected_days=math.inf,
        )

    failure_probability = 1.0 - pass_probability
    expected_failures = failure_probability / pass_probability
    return PhaseProgressStats(
        expected_reset_count=expected_failures,
        expected_reset_fees=expected_failures * evaluation.reset_fee,
        expected_days=(
            evaluation.average_days_to_pass
            + expected_failures * evaluation.average_days_to_fail
        ),
    )


def _calculate_phase2(
    reset_fee: float,
    funded_initial: FundedInitialPhase,
    pass_probability: float,
    phase1: PhaseProgressStats,
) -> PhaseProgressStats:
    """Calculate expected progress from funded initial to payout-loop ready."""

    if pass_probability == 0.0:
        reset_fees = math.inf if reset_fee > 0.0 else 0.0
        return PhaseProgressStats(
            expected_reset_count=math.inf,
            expected_reset_fees=reset_fees,
            expected_days=math.inf,
        )

    failure_probability = 1.0 - pass_probability
    expected_failures = failure_probability / pass_probability
    reset_count_per_failure = 1.0 + phase1.expected_reset_count
    return PhaseProgressStats(
        expected_reset_count=expected_failures * reset_count_per_failure,
        expected_reset_fees=expected_failures * reset_count_per_failure * reset_fee,
        expected_days=(
            funded_initial.average_days_to_pass
            + expected_failures
            * (funded_initial.average_days_to_fail + phase1.expected_days)
        ),
    )


def _calculate_phase3(
    reset_fee: float,
    payout_loop: FundedPayoutLoopPhase,
    pass_probability: float,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
) -> PayoutLoopStats:
    """Calculate expected burden for one successful Phase 3 payout."""

    if pass_probability == 0.0:
        reset_fees = math.inf if reset_fee > 0.0 else 0.0
        return PayoutLoopStats(
            expected_reset_count_per_payout=math.inf,
            expected_reset_fees_per_payout=reset_fees,
            expected_days_per_payout=math.inf,
            expected_net_profit_per_payout=-math.inf,
        )

    failure_probability = 1.0 - pass_probability
    expected_failures = failure_probability / pass_probability
    reset_count_per_failure = (
        1.0 + phase1.expected_reset_count + phase2.expected_reset_count
    )
    reset_fees_per_payout = expected_failures * reset_count_per_failure * reset_fee
    days_per_payout = payout_loop.average_days_to_pass + expected_failures * (
        payout_loop.average_days_to_fail + phase1.expected_days + phase2.expected_days
    )
    return PayoutLoopStats(
        expected_reset_count_per_payout=expected_failures * reset_count_per_failure,
        expected_reset_fees_per_payout=reset_fees_per_payout,
        expected_days_per_payout=days_per_payout,
        expected_net_profit_per_payout=payout_loop.max_payout - reset_fees_per_payout,
    )


def _calculate_first_payout_stats(
    strategy: PropFirmMetaStrategy,
    evaluation_probability: float,
    initial_probability: float,
    payout_probability: float,
) -> FirstPayoutStats:
    """Calculate statistics for the first payout from the original run start."""

    first_payout_probability = (
        evaluation_probability * initial_probability * payout_probability
    )
    if first_payout_probability == 0.0:
        return _no_first_payout_stats()

    evaluation_failure_probability = 1.0 - evaluation_probability
    initial_failure_probability = 1.0 - initial_probability
    payout_failure_probability = 1.0 - payout_probability
    expected_reset_count = (1.0 - first_payout_probability) / first_payout_probability
    expected_days = (
        evaluation_failure_probability * strategy.evaluation.average_days_to_fail
        + evaluation_probability * strategy.evaluation.average_days_to_pass
        + evaluation_probability
        * initial_failure_probability
        * strategy.funded_initial.average_days_to_fail
        + evaluation_probability
        * initial_probability
        * strategy.funded_initial.average_days_to_pass
        + evaluation_probability
        * initial_probability
        * payout_failure_probability
        * strategy.payout_loop.average_days_to_fail
        + first_payout_probability * strategy.payout_loop.average_days_to_pass
    ) / first_payout_probability
    expected_total_fees = (
        strategy.evaluation.new_account_fee
        + expected_reset_count * strategy.evaluation.reset_fee
    )
    expected_net_profit = strategy.payout_loop.max_payout - expected_total_fees
    annualized_net_profit = _annualize_profit(expected_net_profit, expected_days)

    return FirstPayoutStats(
        expected_days=expected_days,
        expected_net_profit=expected_net_profit,
        expected_total_fees=expected_total_fees,
        expected_reset_count=expected_reset_count,
        probability_without_reset=first_payout_probability,
        probability_of_profit=_first_payout_profit_probability(
            strategy,
            first_payout_probability,
        ),
        annualized_net_profit=annualized_net_profit,
        annualized_ev_percent=_percent_of(
            annualized_net_profit,
            strategy.evaluation.new_account_fee,
        ),
    )


def _no_first_payout_stats() -> FirstPayoutStats:
    """Return first-payout stats for an unreachable first payout."""

    return FirstPayoutStats(
        expected_days=math.inf,
        expected_net_profit=math.nan,
        expected_total_fees=math.nan,
        expected_reset_count=math.inf,
        probability_without_reset=0.0,
        probability_of_profit=0.0,
        annualized_net_profit=math.nan,
        annualized_ev_percent=math.nan,
    )


def _first_payout_profit_probability(
    strategy: PropFirmMetaStrategy,
    first_payout_probability: float,
) -> float:
    """Calculate probability that the account is profitable at first payout."""

    if strategy.evaluation.reset_fee == 0.0:
        return (
            1.0
            if strategy.payout_loop.max_payout > strategy.evaluation.new_account_fee
            else 0.0
        )

    max_profitable_resets = _max_profitable_reset_count(
        strategy.payout_loop.max_payout,
        strategy.evaluation.new_account_fee,
        strategy.evaluation.reset_fee,
    )
    if max_profitable_resets < 0:
        return 0.0

    failure_probability = 1.0 - first_payout_probability
    return 1.0 - failure_probability ** (max_profitable_resets + 1)


def _finite_payout_stats(
    *,
    strategy: PropFirmMetaStrategy,
    payout_count: int,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
    phase3: PayoutLoopStats,
    first_payout: FirstPayoutStats,
    profit_probability: float,
    profit_probability_method: str,
) -> PropFirmMetaStrategyStats:
    """Build stats for a strategy that reaches a finite payout cap."""

    setup_reset_count = phase1.expected_reset_count + phase2.expected_reset_count
    expected_reset_count = (
        setup_reset_count + payout_count * phase3.expected_reset_count_per_payout
    )
    expected_total_payouts = payout_count * strategy.payout_loop.max_payout
    expected_total_fees = (
        strategy.evaluation.new_account_fee
        + expected_reset_count * strategy.evaluation.reset_fee
    )
    expected_net_profit = expected_total_payouts - expected_total_fees
    expected_duration_days = (
        phase1.expected_days
        + phase2.expected_days
        + payout_count * phase3.expected_days_per_payout
    )
    annualized_net_profit = _annualize_profit(
        expected_net_profit,
        expected_duration_days,
    )
    annual = _annualize_expected_stats(
        period_days=TRADING_DAYS_PER_YEAR,
        expected_payout_count=float(payout_count),
        expected_total_payouts=expected_total_payouts,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        expected_duration_days=expected_duration_days,
        expected_reset_count=expected_reset_count,
        new_account_fee=strategy.evaluation.new_account_fee,
    )

    return PropFirmMetaStrategyStats(
        annual=annual,
        first_payout=first_payout,
        expected_payout_count=float(payout_count),
        expected_total_payouts=expected_total_payouts,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        total_ev_percent=_percent_of(
            expected_net_profit,
            strategy.evaluation.new_account_fee,
        ),
        expected_fee_return_percent=_percent_of(
            expected_net_profit,
            expected_total_fees,
        ),
        expected_duration_days=expected_duration_days,
        annualized_net_profit=annualized_net_profit,
        annualized_ev_percent=_percent_of(
            annualized_net_profit,
            strategy.evaluation.new_account_fee,
        ),
        probability_of_profit=profit_probability,
        profit_probability_method=profit_probability_method,
        expected_reset_count=expected_reset_count,
        terminates_almost_surely=True,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
    )


def _finite_consecutive_payout_stats(
    *,
    strategy: PropFirmMetaStrategy,
    consecutive_payout_count: int,
    evaluation_probability: float,
    initial_probability: float,
    payout_probability: float,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
    phase3: PayoutLoopStats,
    first_payout: FirstPayoutStats,
    exact_probability_reset_limit: int,
) -> PropFirmMetaStrategyStats:
    """Build stats for a finite consecutive-payout termination rule."""

    payout_moments = _consecutive_payout_moments(
        consecutive_payout_count,
        payout_probability,
    )
    setup_reset_count = phase1.expected_reset_count + phase2.expected_reset_count
    setup_days = phase1.expected_days + phase2.expected_days
    expected_reset_count = setup_reset_count + payout_moments.expected_failure_count * (
        1.0 + setup_reset_count
    )
    expected_total_payouts = (
        payout_moments.expected_success_count * strategy.payout_loop.max_payout
    )
    expected_total_fees = (
        strategy.evaluation.new_account_fee
        + expected_reset_count * strategy.evaluation.reset_fee
    )
    expected_net_profit = expected_total_payouts - expected_total_fees
    expected_duration_days = (
        setup_days
        + payout_moments.expected_success_count
        * strategy.payout_loop.average_days_to_pass
        + payout_moments.expected_failure_count
        * (strategy.payout_loop.average_days_to_fail + setup_days)
    )
    annualized_net_profit = _annualize_profit(
        expected_net_profit,
        expected_duration_days,
    )
    annual = _annualize_expected_stats(
        period_days=TRADING_DAYS_PER_YEAR,
        expected_payout_count=payout_moments.expected_success_count,
        expected_total_payouts=expected_total_payouts,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        expected_duration_days=expected_duration_days,
        expected_reset_count=expected_reset_count,
        new_account_fee=strategy.evaluation.new_account_fee,
    )
    profit_probability, probability_method = _calculate_consecutive_profit_probability(
        strategy=strategy,
        consecutive_payout_count=consecutive_payout_count,
        evaluation_probability=evaluation_probability,
        initial_probability=initial_probability,
        payout_probability=payout_probability,
        payout_moments=payout_moments,
        exact_probability_reset_limit=exact_probability_reset_limit,
    )

    return PropFirmMetaStrategyStats(
        annual=annual,
        first_payout=first_payout,
        expected_payout_count=payout_moments.expected_success_count,
        expected_total_payouts=expected_total_payouts,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        total_ev_percent=_percent_of(
            expected_net_profit,
            strategy.evaluation.new_account_fee,
        ),
        expected_fee_return_percent=_percent_of(
            expected_net_profit,
            expected_total_fees,
        ),
        expected_duration_days=expected_duration_days,
        annualized_net_profit=annualized_net_profit,
        annualized_ev_percent=_percent_of(
            annualized_net_profit,
            strategy.evaluation.new_account_fee,
        ),
        probability_of_profit=profit_probability,
        profit_probability_method=probability_method,
        expected_reset_count=expected_reset_count,
        terminates_almost_surely=True,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
    )


def _unbounded_payout_loop_stats(
    strategy: PropFirmMetaStrategy,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
    phase3: PayoutLoopStats,
    first_payout: FirstPayoutStats,
) -> PropFirmMetaStrategyStats:
    """Build stats for an uncapped payout loop."""

    setup_reset_count = phase1.expected_reset_count + phase2.expected_reset_count
    setup_fees = (
        strategy.evaluation.new_account_fee
        + setup_reset_count * strategy.evaluation.reset_fee
    )
    cycle_net_profit = phase3.expected_net_profit_per_payout
    expected_total_fees = (
        math.inf if phase3.expected_reset_fees_per_payout > 0.0 else setup_fees
    )
    expected_net_profit = _infinite_horizon_net_profit(cycle_net_profit)
    annualized_net_profit = _annualize_profit(
        cycle_net_profit,
        phase3.expected_days_per_payout,
    )
    annual = _annualize_expected_stats(
        period_days=TRADING_DAYS_PER_YEAR,
        expected_payout_count=1.0,
        expected_total_payouts=strategy.payout_loop.max_payout,
        expected_total_fees=phase3.expected_reset_fees_per_payout,
        expected_net_profit=cycle_net_profit,
        expected_duration_days=phase3.expected_days_per_payout,
        expected_reset_count=phase3.expected_reset_count_per_payout,
        new_account_fee=strategy.evaluation.new_account_fee,
    )

    return PropFirmMetaStrategyStats(
        annual=annual,
        first_payout=first_payout,
        expected_payout_count=math.inf,
        expected_total_payouts=math.inf,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        total_ev_percent=_percent_of(
            expected_net_profit,
            strategy.evaluation.new_account_fee,
        ),
        expected_fee_return_percent=_percent_of(
            expected_net_profit,
            expected_total_fees,
        ),
        expected_duration_days=math.inf,
        annualized_net_profit=annualized_net_profit,
        annualized_ev_percent=_percent_of(
            annualized_net_profit,
            strategy.evaluation.new_account_fee,
        ),
        probability_of_profit=_unbounded_probability_of_profit(cycle_net_profit),
        profit_probability_method="long_run_drift",
        expected_reset_count=(
            math.inf
            if phase3.expected_reset_count_per_payout > 0.0
            else setup_reset_count
        ),
        terminates_almost_surely=False,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
    )


def _unreachable_evaluation_stats(
    strategy: PropFirmMetaStrategy,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
    phase3: PayoutLoopStats,
    first_payout: FirstPayoutStats,
) -> PropFirmMetaStrategyStats:
    """Build stats for an account that can never pass evaluation."""

    expected_total_fees = (
        math.inf
        if strategy.evaluation.reset_fee > 0.0
        else strategy.evaluation.new_account_fee
    )
    expected_net_profit = -expected_total_fees
    return _nonterminating_no_payout_stats(
        strategy=strategy,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        expected_reset_count=math.inf,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        first_payout=first_payout,
        probability_method="unreachable_evaluation",
    )


def _unreachable_funded_initial_stats(
    strategy: PropFirmMetaStrategy,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
    phase3: PayoutLoopStats,
    first_payout: FirstPayoutStats,
) -> PropFirmMetaStrategyStats:
    """Build stats for an account that can never reach the payout loop."""

    expected_total_fees = (
        math.inf
        if strategy.evaluation.reset_fee > 0.0
        else strategy.evaluation.new_account_fee
    )
    expected_net_profit = -expected_total_fees
    return _nonterminating_no_payout_stats(
        strategy=strategy,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        expected_reset_count=math.inf,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        first_payout=first_payout,
        probability_method="unreachable_funded_initial",
    )


def _unreachable_payout_loop_stats(
    strategy: PropFirmMetaStrategy,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
    phase3: PayoutLoopStats,
    first_payout: FirstPayoutStats,
) -> PropFirmMetaStrategyStats:
    """Build stats for an account that can reach Phase 3 but never get paid."""

    expected_total_fees = (
        math.inf
        if strategy.evaluation.reset_fee > 0.0
        else strategy.evaluation.new_account_fee
    )
    expected_net_profit = -expected_total_fees
    return _nonterminating_no_payout_stats(
        strategy=strategy,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        expected_reset_count=math.inf,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        first_payout=first_payout,
        probability_method="unreachable_payout_loop",
    )


def _nonterminating_no_payout_stats(
    *,
    strategy: PropFirmMetaStrategy,
    expected_total_fees: float,
    expected_net_profit: float,
    expected_reset_count: float,
    phase1: PhaseProgressStats,
    phase2: PhaseProgressStats,
    phase3: PayoutLoopStats,
    first_payout: FirstPayoutStats,
    probability_method: str,
) -> PropFirmMetaStrategyStats:
    """Build stats for a nonterminating account path with zero payouts."""

    annual = AnnualStats(
        period_days=TRADING_DAYS_PER_YEAR,
        expected_payout_count=0.0,
        expected_total_payouts=0.0,
        expected_total_fees=math.nan,
        expected_net_profit=math.nan,
        ev_percent=math.nan,
        expected_reset_count=math.nan,
    )

    return PropFirmMetaStrategyStats(
        annual=annual,
        first_payout=first_payout,
        expected_payout_count=0.0,
        expected_total_payouts=0.0,
        expected_total_fees=expected_total_fees,
        expected_net_profit=expected_net_profit,
        total_ev_percent=_percent_of(
            expected_net_profit,
            strategy.evaluation.new_account_fee,
        ),
        expected_fee_return_percent=_percent_of(
            expected_net_profit,
            expected_total_fees,
        ),
        expected_duration_days=math.inf,
        annualized_net_profit=math.nan,
        annualized_ev_percent=math.nan,
        probability_of_profit=0.0,
        profit_probability_method=probability_method,
        expected_reset_count=expected_reset_count,
        terminates_almost_surely=False,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
    )


def _calculate_profit_probability(
    strategy: PropFirmMetaStrategy,
    payout_count: int,
    evaluation_probability: float,
    initial_probability: float,
    payout_probability: float,
    exact_probability_reset_limit: int,
) -> tuple[float, str]:
    """Calculate probability that finite payouts exceed total account fees."""

    total_payouts = payout_count * strategy.payout_loop.max_payout
    if strategy.evaluation.reset_fee == 0.0:
        return (
            1.0 if total_payouts > strategy.evaluation.new_account_fee else 0.0,
            "deterministic",
        )

    max_profitable_resets = _max_profitable_reset_count(
        total_payouts,
        strategy.evaluation.new_account_fee,
        strategy.evaluation.reset_fee,
    )
    if max_profitable_resets < 0:
        return 0.0, "deterministic"

    if max_profitable_resets <= exact_probability_reset_limit:
        probability = _exact_reset_count_cdf(
            max_profitable_resets,
            payout_count,
            evaluation_probability,
            initial_probability,
            payout_probability,
        )
        return probability, "exact"

    moments = _reset_moments(
        evaluation_probability,
        initial_probability,
        payout_probability,
    )
    probability = _normal_reset_count_cdf(
        max_profitable_resets,
        moments.setup_mean + payout_count * moments.payout_cycle_mean,
        moments.setup_variance + payout_count * moments.payout_cycle_variance,
    )
    return probability, "normal_approximation"


def _calculate_consecutive_profit_probability(
    *,
    strategy: PropFirmMetaStrategy,
    consecutive_payout_count: int,
    evaluation_probability: float,
    initial_probability: float,
    payout_probability: float,
    payout_moments: _ConsecutivePayoutMoments,
    exact_probability_reset_limit: int,
) -> tuple[float, str]:
    """Calculate profit probability for a finite consecutive-payout cap."""

    if consecutive_payout_count == 1 or payout_probability == 1.0:
        return _calculate_profit_probability(
            strategy,
            consecutive_payout_count,
            evaluation_probability,
            initial_probability,
            payout_probability,
            exact_probability_reset_limit,
        )

    reset_moments = _reset_moments(
        evaluation_probability,
        initial_probability,
        payout_probability,
    )
    setup_reset_mean = reset_moments.setup_mean
    setup_reset_variance = reset_moments.setup_variance
    reset_count_variance = setup_reset_variance * (
        1.0 + payout_moments.expected_failure_count
    ) + payout_moments.failure_count_variance * (1.0 + setup_reset_mean) * (
        1.0 + setup_reset_mean
    )
    success_reset_covariance = payout_moments.success_failure_covariance * (
        1.0 + setup_reset_mean
    )
    net_profit_variance = (
        strategy.payout_loop.max_payout
        * strategy.payout_loop.max_payout
        * payout_moments.success_count_variance
        + strategy.evaluation.reset_fee
        * strategy.evaluation.reset_fee
        * reset_count_variance
        - 2.0
        * strategy.payout_loop.max_payout
        * strategy.evaluation.reset_fee
        * success_reset_covariance
    )
    expected_net_profit = (
        strategy.payout_loop.max_payout * payout_moments.expected_success_count
        - strategy.evaluation.new_account_fee
        - strategy.evaluation.reset_fee
        * (
            setup_reset_mean
            + payout_moments.expected_failure_count * (1.0 + setup_reset_mean)
        )
    )
    probability = _normal_positive_probability(
        expected_net_profit,
        net_profit_variance,
    )
    return probability, "normal_approximation"


def _consecutive_payout_moments(
    consecutive_payout_count: int,
    pass_probability: float,
) -> _ConsecutivePayoutMoments:
    """Return payout and failure moments until N consecutive Phase 3 passes."""

    if consecutive_payout_count == 0:
        return _ConsecutivePayoutMoments(
            expected_success_count=0.0,
            success_count_variance=0.0,
            expected_failure_count=0.0,
            failure_count_variance=0.0,
            success_failure_covariance=0.0,
        )
    if pass_probability == 1.0:
        return _ConsecutivePayoutMoments(
            expected_success_count=float(consecutive_payout_count),
            success_count_variance=0.0,
            expected_failure_count=0.0,
            failure_count_variance=0.0,
            success_failure_covariance=0.0,
        )

    terminal_probability = pass_probability**consecutive_payout_count
    if terminal_probability == 0.0:
        return _ConsecutivePayoutMoments(
            expected_success_count=math.inf,
            success_count_variance=math.inf,
            expected_failure_count=math.inf,
            failure_count_variance=math.inf,
            success_failure_covariance=math.inf,
        )

    failed_run_probability = 1.0 - terminal_probability
    failure_mean = failed_run_probability / terminal_probability
    failure_variance = failed_run_probability / (
        terminal_probability * terminal_probability
    )
    failed_streak_mean, failed_streak_variance = _failed_streak_moments(
        consecutive_payout_count,
        pass_probability,
        failed_run_probability,
    )
    success_mean = consecutive_payout_count + failure_mean * failed_streak_mean
    success_variance = (
        failure_mean * failed_streak_variance
        + failure_variance * failed_streak_mean * failed_streak_mean
    )
    success_failure_covariance = failure_variance * failed_streak_mean

    return _ConsecutivePayoutMoments(
        expected_success_count=success_mean,
        success_count_variance=success_variance,
        expected_failure_count=failure_mean,
        failure_count_variance=failure_variance,
        success_failure_covariance=success_failure_covariance,
    )


def _failed_streak_moments(
    consecutive_payout_count: int,
    pass_probability: float,
    failed_run_probability: float,
) -> tuple[float, float]:
    """Return moments of payouts earned in a failed Phase 3 streak."""

    failure_probability = 1.0 - pass_probability
    mean = 0.0
    second_moment = 0.0
    probability = failure_probability / failed_run_probability
    probability_step = pass_probability

    for streak_length in range(consecutive_payout_count):
        mean += streak_length * probability
        second_moment += streak_length * streak_length * probability
        probability *= probability_step

    variance = second_moment - mean * mean
    return mean, max(0.0, variance)


def _exact_reset_count_cdf(
    max_resets: int,
    payout_count: int,
    evaluation_probability: float,
    initial_probability: float,
    payout_probability: float,
) -> float:
    """Calculate an exact CDF for total reset count up to ``max_resets``."""

    phase1_distribution = _geometric_failures_pmf(
        evaluation_probability,
        max_resets,
    )
    phase2_failure_distribution = _shift_distribution(
        phase1_distribution,
        1,
        max_resets,
    )
    phase2_distribution = _compound_geometric_pmf(
        initial_probability,
        phase2_failure_distribution,
        max_resets,
    )
    setup_distribution = _convolve_truncated(
        phase1_distribution,
        phase2_distribution,
        max_resets,
    )
    phase3_failure_distribution = _shift_distribution(
        setup_distribution,
        1,
        max_resets,
    )
    phase3_distribution = _compound_geometric_pmf(
        payout_probability,
        phase3_failure_distribution,
        max_resets,
    )
    payout_distribution = _distribution_power(
        phase3_distribution,
        payout_count,
        max_resets,
    )
    total_distribution = _convolve_truncated(
        setup_distribution,
        payout_distribution,
        max_resets,
    )
    return min(1.0, max(0.0, sum(total_distribution)))


def _reset_moments(
    evaluation_probability: float,
    initial_probability: float,
    payout_probability: float,
) -> _ResetMoments:
    """Return mean and variance for reset counts."""

    phase1_mean, phase1_variance = _geometric_failure_moments(evaluation_probability)
    phase2_failure_mean = 1.0 + phase1_mean
    phase2_failure_variance = phase1_variance
    phase2_attempt_mean, phase2_attempt_variance = _compound_geometric_moments(
        initial_probability,
        phase2_failure_mean,
        phase2_failure_variance,
    )
    setup_mean = phase1_mean + phase2_attempt_mean
    setup_variance = phase1_variance + phase2_attempt_variance
    phase3_failure_mean = 1.0 + setup_mean
    phase3_failure_variance = setup_variance
    payout_cycle_mean, payout_cycle_variance = _compound_geometric_moments(
        payout_probability,
        phase3_failure_mean,
        phase3_failure_variance,
    )
    return _ResetMoments(
        setup_mean=setup_mean,
        setup_variance=setup_variance,
        payout_cycle_mean=payout_cycle_mean,
        payout_cycle_variance=payout_cycle_variance,
    )


def _geometric_failure_moments(pass_probability: float) -> tuple[float, float]:
    """Return mean and variance of failures before first success."""

    failure_probability = 1.0 - pass_probability
    return (
        failure_probability / pass_probability,
        failure_probability / (pass_probability * pass_probability),
    )


def _compound_geometric_moments(
    pass_probability: float,
    component_mean: float,
    component_variance: float,
) -> tuple[float, float]:
    """Return moments for a sum of geometric-failure-count components."""

    failure_probability = 1.0 - pass_probability
    failure_count_mean = failure_probability / pass_probability
    failure_count_variance = failure_probability / (pass_probability * pass_probability)
    return (
        failure_count_mean * component_mean,
        failure_count_mean * component_variance
        + failure_count_variance * component_mean * component_mean,
    )


def _geometric_failures_pmf(pass_probability: float, max_failures: int) -> list[float]:
    """Return truncated PMF of failures before first success."""

    failure_probability = 1.0 - pass_probability
    result = []
    probability = pass_probability
    for _ in range(max_failures + 1):
        result.append(probability)
        probability *= failure_probability
    return result


def _compound_geometric_pmf(
    pass_probability: float,
    component_distribution: list[float],
    max_resets: int,
) -> list[float]:
    """Return truncated PMF for a geometric number of component sums."""

    result = [0.0] * (max_resets + 1)
    term = [0.0] * (max_resets + 1)
    term[0] = 1.0
    failure_probability = 1.0 - pass_probability
    failure_power = 1.0

    for _ in range(max_resets + 1):
        weight = pass_probability * failure_power
        _add_weighted_distribution(result, term, weight)
        failure_power *= failure_probability
        term = _convolve_truncated(term, component_distribution, max_resets)
        if not any(term):
            break

    return result


def _distribution_power(
    distribution: list[float],
    exponent: int,
    max_resets: int,
) -> list[float]:
    """Return truncated PMF of a sum of ``exponent`` iid distributions."""

    result = [0.0] * (max_resets + 1)
    result[0] = 1.0
    base = distribution
    remaining_exponent = exponent

    while remaining_exponent > 0:
        if remaining_exponent % 2 == 1:
            result = _convolve_truncated(result, base, max_resets)
        remaining_exponent //= 2
        if remaining_exponent:
            base = _convolve_truncated(base, base, max_resets)

    return result


def _convolve_truncated(
    left: list[float],
    right: list[float],
    max_resets: int,
) -> list[float]:
    """Convolve two reset-count distributions and truncate to ``max_resets``."""

    result = [0.0] * (max_resets + 1)
    for left_index, left_probability in enumerate(left[: max_resets + 1]):
        if left_probability == 0.0:
            continue
        remaining_resets = max_resets - left_index
        for right_index, right_probability in enumerate(right[: remaining_resets + 1]):
            if right_probability != 0.0:
                result[left_index + right_index] += left_probability * right_probability
    return result


def _shift_distribution(
    distribution: list[float],
    shift: int,
    max_resets: int,
) -> list[float]:
    """Shift a reset-count distribution by a non-negative number of resets."""

    result = [0.0] * (max_resets + 1)
    if shift > max_resets:
        return result

    for index, probability in enumerate(distribution[: max_resets - shift + 1]):
        result[index + shift] = probability
    return result


def _add_weighted_distribution(
    target: list[float],
    source: list[float],
    weight: float,
) -> None:
    """Add a weighted distribution into ``target`` in place."""

    if weight == 0.0:
        return
    for index, probability in enumerate(source):
        if probability != 0.0:
            target[index] += weight * probability


def _normal_reset_count_cdf(
    max_resets: int,
    mean: float,
    variance: float,
) -> float:
    """Approximate the reset-count CDF with a continuity-corrected normal."""

    if variance == 0.0:
        return 1.0 if mean <= max_resets else 0.0

    standard_deviation = math.sqrt(variance)
    z_score = (max_resets + 0.5 - mean) / standard_deviation
    probability = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    return min(1.0, max(0.0, probability))


def _normal_positive_probability(mean: float, variance: float) -> float:
    """Approximate the probability that a normal variable is positive."""

    if variance <= 0.0:
        if mean > 0.0:
            return 1.0
        if mean < 0.0:
            return 0.0
        return 0.0
    standard_deviation = math.sqrt(variance)
    z_score = mean / standard_deviation
    probability = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    return min(1.0, max(0.0, probability))


def _max_profitable_reset_count(
    total_payouts: float,
    new_account_fee: float,
    reset_fee: float,
) -> int:
    """Return the largest reset count where payouts are strictly profitable."""

    profit_margin_before_resets = total_payouts - new_account_fee
    if profit_margin_before_resets <= 0.0:
        return -1
    reset_limit = profit_margin_before_resets / reset_fee
    return math.floor(math.nextafter(reset_limit, -math.inf))


def _infinite_horizon_net_profit(cycle_net_profit: float) -> float:
    """Return the expected net-profit direction for an uncapped payout loop."""

    if cycle_net_profit > 0.0:
        return math.inf
    if cycle_net_profit < 0.0:
        return -math.inf
    return math.nan


def _unbounded_probability_of_profit(cycle_net_profit: float) -> float:
    """Return long-run profit probability for an uncapped payout loop."""

    if cycle_net_profit > 0.0:
        return 1.0
    if cycle_net_profit < 0.0:
        return 0.0
    return math.nan


def _annualize_expected_stats(
    *,
    period_days: float,
    expected_payout_count: float,
    expected_total_payouts: float,
    expected_total_fees: float,
    expected_net_profit: float,
    expected_duration_days: float,
    expected_reset_count: float,
    new_account_fee: float,
) -> AnnualStats:
    """Scale expected totals to a yearly active-account rate."""

    if expected_duration_days == 0.0:
        scale = math.inf
    elif not math.isfinite(expected_duration_days):
        scale = 0.0
    else:
        scale = period_days / expected_duration_days

    annual_net_profit = expected_net_profit * scale
    return AnnualStats(
        period_days=period_days,
        expected_payout_count=expected_payout_count * scale,
        expected_total_payouts=expected_total_payouts * scale,
        expected_total_fees=expected_total_fees * scale,
        expected_net_profit=annual_net_profit,
        ev_percent=_percent_of(annual_net_profit, new_account_fee),
        expected_reset_count=expected_reset_count * scale,
    )


def _annualize_profit(net_profit: float, duration_days: float) -> float:
    """Convert a net profit over ``duration_days`` into yearly profit."""

    if duration_days == 0.0:
        if net_profit > 0.0:
            return math.inf
        if net_profit < 0.0:
            return -math.inf
        return math.nan
    if not math.isfinite(net_profit) or not math.isfinite(duration_days):
        return math.nan
    return net_profit * TRADING_DAYS_PER_YEAR / duration_days


def _percent_of(numerator: float, denominator: float) -> float:
    """Return ``numerator`` as a percentage of ``denominator``."""

    if denominator == 0.0:
        if numerator > 0.0:
            return math.inf
        if numerator < 0.0:
            return -math.inf
        return math.nan
    if numerator == -math.inf and denominator == math.inf:
        return -100.0
    if math.isinf(numerator) and math.isinf(denominator):
        return math.nan
    return 100.0 * numerator / denominator


def _percent_to_probability(pass_rate_percent: float) -> float:
    """Convert a percent or fractional rate value to a probability."""

    if pass_rate_percent <= 1.0:
        return pass_rate_percent
    return pass_rate_percent / 100.0


def _validate_strategy(
    strategy: PropFirmMetaStrategy,
    exact_probability_reset_limit: int,
) -> None:
    """Validate inputs before running the evaluator."""

    if (
        not isinstance(exact_probability_reset_limit, int)
        or isinstance(exact_probability_reset_limit, bool)
        or exact_probability_reset_limit < 0
    ):
        raise ValueError("exact_probability_reset_limit must be a non-negative int")

    _validate_non_negative_finite(
        strategy.evaluation.new_account_fee,
        "evaluation.new_account_fee",
    )
    _validate_non_negative_finite(
        strategy.evaluation.reset_fee,
        "evaluation.reset_fee",
    )
    _validate_percent(
        strategy.evaluation.pass_rate_percent,
        "evaluation.pass_rate_percent",
    )
    _validate_non_negative_finite(
        strategy.evaluation.average_days_to_pass,
        "evaluation.average_days_to_pass",
    )
    _validate_non_negative_finite(
        strategy.evaluation.average_days_to_fail,
        "evaluation.average_days_to_fail",
    )

    _validate_percent(
        strategy.funded_initial.pass_rate_percent,
        "funded_initial.pass_rate_percent",
    )
    _validate_non_negative_finite(
        strategy.funded_initial.average_days_to_pass,
        "funded_initial.average_days_to_pass",
    )
    _validate_non_negative_finite(
        strategy.funded_initial.average_days_to_fail,
        "funded_initial.average_days_to_fail",
    )

    _validate_non_negative_finite(
        strategy.payout_loop.max_payout,
        "payout_loop.max_payout",
    )
    _validate_max_payout_count(strategy.payout_loop.max_payout_count)
    _validate_percent(
        strategy.payout_loop.pass_rate_percent,
        "payout_loop.pass_rate_percent",
    )
    _validate_non_negative_finite(
        strategy.payout_loop.average_days_to_pass,
        "payout_loop.average_days_to_pass",
    )
    _validate_non_negative_finite(
        strategy.payout_loop.average_days_to_fail,
        "payout_loop.average_days_to_fail",
    )


def _validate_non_negative_finite(value: float, name: str) -> None:
    """Validate that a numeric input is finite and non-negative."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_percent(value: float, name: str) -> None:
    """Validate a percentage input."""

    _validate_non_negative_finite(value, name)
    if value > 100.0:
        raise ValueError(f"{name} must be between 0 and 100")


def _validate_max_payout_count(max_payout_count: Optional[int]) -> None:
    """Validate an optional maximum payout count."""

    if max_payout_count is None:
        return
    if not isinstance(max_payout_count, int) or isinstance(max_payout_count, bool):
        raise ValueError("payout_loop.max_payout_count must be an int or None")
    if max_payout_count < 0:
        raise ValueError("payout_loop.max_payout_count must be non-negative")
