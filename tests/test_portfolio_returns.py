"""Fees reduce trade P/L before conversion to fractional return on risk."""

from pathlib import Path

import polars as pl
import pytest

from ifera.allocation._portfolio_returns import (
    strategy_return_expression,
    validate_fees_per_leg,
)


def _frame(**overrides):
    values = {
        "P/L": [100.0],
        "Risk": [400.0],
        "Quantity": [3.0],
        "Legs": ["long call 100 / short call 105"],
        "Status": ["expired"],
        "ROR": [999.0],
    }
    values.update(overrides)
    return pl.DataFrame(values)


def _returns(frame, fee=0.0):
    return frame.select(
        strategy_return_expression(frame, Path("strategy.csv"), fee).alias("return")
    )["return"].to_list()


@pytest.mark.parametrize(
    "status,sides",
    [
        ("expired", 1),
        (" EXPIRED ", 1),
        ("closed", 2),
        ("preexp", 2),
        ("", 2),
        (None, 2),
    ],
)
def test_expired_status_only_charges_entry_fees(status, sides):
    result = _returns(_frame(Status=[status]), 1.5)
    assert result == pytest.approx([(100.0 - 1.5 * 2 * sides * 3) / 400.0])


@pytest.mark.parametrize("leg_count", [1, 2, 4, 7])
def test_each_slash_separated_leg_is_charged(leg_count):
    legs = " / ".join(f"long call {100 + index}" for index in range(leg_count))
    result = _returns(_frame(Legs=[legs], Status=["closed"]), 0.75)
    assert result == pytest.approx([(100 - 0.75 * leg_count * 2 * 3) / 400])


def test_case_insensitive_columns_and_fractional_quantity():
    frame = _frame(Quantity=[1.5]).rename(
        {"P/L": " p/l ", "Risk": "RISK", "Legs": "legs", "Status": "STATUS"}
    )
    assert _returns(frame, 2.0) == pytest.approx([(100 - 2 * 2 * 1.5) / 400])


def test_row_specific_leg_counts_quantities_and_statuses_are_vectorized():
    frame = pl.DataFrame(
        {
            "P/L": [100.0, -25.0, 0.0],
            "Risk": [400.0, 250.0, 100.0],
            "Quantity": [3, 2, 1],
            "Legs": [
                "long call 100 / short call 105",
                "long call 100 / short call 105 / long put 95 / short put 90",
                "long put 95",
            ],
            "Status": ["expired", "closed", "expired"],
        }
    )
    assert _returns(frame, 1.25) == pytest.approx([0.23125, -0.18, -0.0125])


def test_zero_fee_uses_precise_accounting_and_does_not_require_fee_metadata():
    frame = pl.DataFrame({"P/L": [1.0], "Risk": [3.0], "ROR": [33.33]})
    assert _returns(frame) == pytest.approx([1 / 3])


def test_zero_fee_keeps_legacy_ror_only_inputs():
    assert _returns(pl.DataFrame({"ror": [12.5]})) == [0.125]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_legacy_invalid_ror_retains_loader_error_message(value):
    with pytest.raises(ValueError, match="non-finite ROR"):
        _returns(pl.DataFrame({"ROR": [value]}))


def test_negative_pnl_and_fee_losses_beyond_full_risk_are_preserved():
    assert _returns(_frame(**{"P/L": [-400.0]}), 1.0) == pytest.approx([-1.015])


@pytest.mark.parametrize("column", ["P/L", "Risk", "Quantity", "Legs", "Status"])
def test_nonzero_fee_requires_accounting_and_fee_columns(column):
    with pytest.raises(ValueError, match=column):
        _returns(_frame().drop(column), 1.0)


@pytest.mark.parametrize("column", ["P/L", "Risk"])
def test_partial_accounting_does_not_silently_fall_back_to_ror(column):
    with pytest.raises(ValueError, match="P/L|Risk"):
        _returns(pl.DataFrame({column: [1.0], "ROR": [10.0]}))


@pytest.mark.parametrize("column", ["Risk", "Quantity"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), None, "bad"])
def test_invalid_risk_or_quantity_is_rejected(column, value):
    with pytest.raises(ValueError, match=column):
        _returns(_frame(**{column: [value]}), 1.0)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), None, "bad"])
def test_invalid_profit_is_rejected(value):
    with pytest.raises(ValueError, match="P/L"):
        _returns(_frame(**{"P/L": [value]}), 1.0)


@pytest.mark.parametrize(
    "legs", [None, "", " ", "long call 100 /", "/ long call 100", "x // y"]
)
def test_missing_or_empty_leg_components_are_rejected(legs):
    with pytest.raises(ValueError, match="Legs"):
        _returns(_frame(Legs=[legs]), 1.0)


def test_duplicate_case_insensitive_columns_are_rejected_with_filename():
    with pytest.raises(ValueError, match="strategy.csv.*multiple columns"):
        _returns(_frame().with_columns(pl.lit(1).alias("risk")))


@pytest.mark.parametrize("fee", [0, 0.0, 1.25])
def test_valid_fee_parameter(fee):
    assert validate_fees_per_leg(fee) == float(fee)


@pytest.mark.parametrize("fee", [True, False, None, "1.0", []])
def test_invalid_fee_parameter_type(fee):
    with pytest.raises(TypeError, match="fees_per_leg"):
        validate_fees_per_leg(fee)


@pytest.mark.parametrize("fee", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_fee_parameter_value(fee):
    with pytest.raises(ValueError, match="fees_per_leg"):
        validate_fees_per_leg(fee)
