"""Trade-ledger input parsing and exact cross-strategy blocking tests."""

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from ifera.allocation._position_loading import load_position_model
from ifera.allocation import portfolio_allocation
from ifera.allocation.portfolio_allocation import _nyse_market_dates
from ifera.settings import settings


def _write(path: Path, rows: dict[str, list[object]]) -> Path:
    pl.DataFrame(rows).write_csv(path)
    return path


def _load(*paths: Path):
    return load_position_model(
        paths, tuple(path.stem for path in paths), "linear", _nyse_market_dates
    )


def test_multiday_minimal_schema_extends_calendar_until_last_close(tmp_path):
    path = _write(
        tmp_path / "overnight.csv",
        {
            "Opened": ["2024-01-05 14:00:00"],
            "Closed": ["2024-01-09 10:00:00"],
            "ROR": [25.0],
        },
    )
    model = _load(path)
    assert model is not None
    assert model.dates == (date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9))
    assert model.strategy_names == ("overnight",)
    assert model.trades[0].return_on_risk == 0.25
    assert model.trades[0].closed == datetime(2024, 1, 9, 10)


def test_ordinary_same_day_portfolio_keeps_legacy_path(tmp_path):
    path = _write(
        tmp_path / "intraday.csv",
        {
            "Opened": ["2024-01-08 09:30:00", "2024-01-09 09:30:00"],
            "Closed": ["2024-01-08 16:00:00", "2024-01-09 16:00:00"],
            "ROR": [10.0, -10.0],
        },
    )
    assert _load(path) is None


def test_without_any_closed_columns_duplicates_keep_legacy_path(tmp_path):
    path = _write(
        tmp_path / "legacy.csv",
        {"Opened": ["2024-01-08", "2024-01-08"], "ROR": [10.0, 5.0]},
    )
    assert _load(path) is None


def test_multiple_entries_same_day_activate_position_model(tmp_path):
    path = _write(
        tmp_path / "repeated.csv",
        {
            "Opened": ["2024-01-08 10:00:00", "2024-01-08 11:00:00"],
            "Closed": ["2024-01-08 14:00:00", "2024-01-08 15:00:00"],
            "ROR": [10.0, -5.0],
        },
    )
    model = _load(path)
    assert model is not None
    assert len(model.trades) == 2
    assert [trade.return_on_risk for trade in model.trades] == [0.1, -0.05]
    assert all(not trade.blockers for trade in model.trades)


def test_mixed_legacy_strategy_gets_inferred_same_day_close(tmp_path):
    overnight = _write(
        tmp_path / "overnight.csv",
        {
            "Opened": ["2024-01-08 10:00:00"],
            "Closed": ["2024-01-09 15:00:00"],
            "ROR": [10.0],
        },
    )
    intraday = _write(
        tmp_path / "intraday.csv",
        {"Opened": ["2024-01-08", "2024-01-09 17:00:00"], "ROR": [5.0, 10.0]},
    )
    model = _load(overnight, intraday)
    assert model is not None
    same_day = [trade for trade in model.trades if trade.strategy_index == 1]
    assert same_day[0].closed == datetime(2024, 1, 8, 16)
    assert same_day[1].closed == datetime(2024, 1, 9, 17)


def test_exact_trade_blockers_follow_chronology_and_same_time_tiebreak(tmp_path):
    first = _write(
        tmp_path / "A.csv",
        {
            "Opened": ["2024-01-09 10:00:00", "2024-01-08 10:00:00"],
            "Closed": ["2024-01-10 15:00:00", "2024-01-09 11:00:00"],
            "ROR": [1.0, 2.0],
            "Symbol": ["SPX", "SPX"],
            "Exp": ["2024-01-12", "2024-01-12"],
            "Legs": ["long put 4500", "long put 4500"],
        },
    )
    second = _write(
        tmp_path / "B.csv",
        {
            "Opened": ["2024-01-09 10:00:00", "2024-01-10 15:00:00"],
            "Closed": ["2024-01-10 14:00:00", "2024-01-10 16:00:00"],
            "ROR": [3.0, 4.0],
            "Symbol": ["SPX", "SPX"],
            "Exp": ["2024-01-12", "2024-01-12"],
            "Legs": ["short put 4500 / long put 4510", "short put 4500"],
        },
    )
    model = _load(first, second)
    assert model is not None
    assert [trade.strategy_index for trade in model.trades] == [0, 0, 1, 1]
    assert [trade.return_on_risk for trade in model.trades] == [0.02, 0.01, 0.03, 0.04]
    assert [trade.blockers for trade in model.trades] == [(), (), (0, 1), ()]


@pytest.mark.parametrize(
    ("opened", "closed", "ror", "message"),
    [
        ("invalid", "2024-01-09", 10.0, "Opened"),
        ("2024-01-08", "invalid", 10.0, "Closed"),
        ("2024-01-09", "2024-01-08", 10.0, "before Opened"),
        ("2024-01-08", "2024-01-09", float("inf"), "non-finite ROR"),
        ("2024-01-08", "2024-01-09", None, "non-finite ROR"),
        ("2024-01-06", "2024-01-09", 10.0, "non-NYSE"),
        ("2024-01-08", "2024-01-13", 10.0, "non-NYSE"),
    ],
)
def test_invalid_position_rows_are_rejected(tmp_path, opened, closed, ror, message):
    path = _write(
        tmp_path / "invalid.csv",
        {"Opened": [opened], "Closed": [closed], "ROR": [ror]},
    )
    with pytest.raises(ValueError, match=message):
        _load(path)


def test_partial_option_metadata_is_rejected(tmp_path):
    path = _write(
        tmp_path / "partial.csv",
        {
            "Opened": ["2024-01-08"],
            "Closed": ["2024-01-09"],
            "ROR": [10.0],
            "Symbol": ["SPX"],
        },
    )
    with pytest.raises(ValueError, match="incomplete trade-block metadata"):
        _load(path)


@pytest.mark.parametrize(
    ("symbol", "expiration", "leg"),
    [
        ("NDX", "2024-01-12", "short put 4500"),
        ("SPX", "2024-01-19", "short put 4500"),
        ("SPX", "2024-01-12", "long put 4500"),
        ("SPX", "2024-01-12", "short call 4500"),
        ("SPX", "2024-01-12", "short put 4510"),
    ],
)
def test_unrelated_option_legs_do_not_block(tmp_path, symbol, expiration, leg):
    common = {
        "Opened": ["2024-01-08 10:00:00"],
        "Closed": ["2024-01-09 16:00:00"],
        "ROR": [10.0],
    }
    first = _write(
        tmp_path / "A.csv",
        {
            **common,
            "Symbol": ["SPX"],
            "Exp": ["2024-01-12"],
            "Legs": ["long put 4500"],
        },
    )
    second = _write(
        tmp_path / "B.csv",
        {**common, "Symbol": [symbol], "Exp": [expiration], "Legs": [leg]},
    )
    model = _load(first, second)
    assert model is not None
    assert [trade.blockers for trade in model.trades] == [(), ()]


def test_case_insensitive_columns_and_example_timestamp_format(tmp_path):
    path = _write(
        tmp_path / "case.csv",
        {
            " opened ": ["Jan 8, 2024 12:20pm"],
            " CLOSED ": ["Jan 9, 2024 3:45pm"],
            " rOr ": [25.0],
        },
    )
    model = _load(path)
    assert model is not None
    assert model.trades[0].opened == datetime(2024, 1, 8, 12, 20)
    assert model.trades[0].closed == datetime(2024, 1, 9, 15, 45)


def test_closed_only_intraday_csv_can_mix_with_legacy_daily_csv(tmp_path, monkeypatch):
    directory = tmp_path / "results" / "portfolios" / "intraday"
    directory.mkdir(parents=True)
    _write(
        directory / "A.csv",
        {
            "Opened": ["2024-01-08 10:00:00"],
            "Closed": ["2024-01-08 15:00:00"],
            "ROR": [10.0],
        },
    )
    _write(directory / "B.csv", {"Opened": ["2024-01-08"], "ROR": [5.0]})
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    names, frame, returns, blocks, model = portfolio_allocation._load_portfolio_inputs(
        "intraday", pl.Float32, "linear"
    )

    assert names == ("A", "B")
    assert frame.height == 1
    assert returns.tolist()[0] == pytest.approx([0.1, 0.05])
    assert blocks == ()
    assert model is None


def test_legacy_loader_still_rejects_partial_option_metadata(tmp_path):
    path = _write(
        tmp_path / "partial.csv",
        {
            "Opened": ["2024-01-08 10:00:00"],
            "Closed": ["2024-01-08 15:00:00"],
            "ROR": [10.0],
            "Symbol": ["SPX"],
        },
    )
    with pytest.raises(ValueError, match="incomplete trade-block metadata"):
        portfolio_allocation._load_strategy_csv_data(path, pl.Float32, 0)
