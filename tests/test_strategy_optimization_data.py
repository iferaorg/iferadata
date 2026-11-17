"""Single-strategy accounting and Oracle minute-source contracts."""

from datetime import date, datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
from unittest.mock import MagicMock

import oracledb
import polars as pl
import pytest


from ifera.settings import settings
from ifera.strategy_optimization import _data, _oracle_queries


@pytest.fixture(autouse=True)
def isolated_daily_history(monkeypatch):
    """Keep Oracle query contract tests independent of external daily history."""
    completion = MagicMock(return_value=None)
    monkeypatch.setattr(_data, "complete_daily_history", completion)
    return completion


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))
    folder = tmp_path / "results" / "portfolios" / "sample"
    folder.mkdir(parents=True)
    path = folder / "strategy.csv"
    rows = {
        "Opened": ["Jan 8, 2024 1:00pm", "Jan 10, 2024 1:00pm"],
        "Closed": ["Jan 8, 2024 4:00pm", "Jan 10, 2024 4:00pm"],
        "Symbol": ["SPX", "SPX"],
        "Exp": ["Jan 8, 2024", "Jan 10, 2024"],
        "Legs": ["long put 4,970 / short put 4,980"] * 2,
        "Quantity": [2, 1],
        "Risk": [1600.0, 800.0],
        "P/L": [400.0, -200.0],
        "ROR": [24.99, -25.0],
        "Premium": [400.0, 200.0],
        "Reward/Risk": [25.0, 25.0],
        "Open Price": [2.0, 2.0],
        "Status": ["expired", "expired"],
        "Custom": ["keep", "unchanged"],
    }
    pl.DataFrame(rows).write_csv(path)
    return path


def test_load_preserves_raw_and_recomputes_accounting_calendar(ledger):
    data = _data.load_strategy("sample", "strategy")
    assert data.raw.equals(pl.read_csv(ledger))
    assert data.path == ledger
    assert data.trades["opening_price"].to_list() == [-2.0, -2.0]
    assert data.trades["ror"].to_list() == [25.0, -25.0]
    assert data.trades["reward_risk"].to_list() == [25.0, 25.0]
    assert data.dates == (date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10))
    assert data.legs.height == 4


@pytest.mark.parametrize("as_string", [False, True])
def test_load_strategy_csv_accepts_arbitrary_path_without_writing_source(
    ledger, tmp_path, as_string
):
    directory = tmp_path / "external ledgers"
    directory.mkdir()
    path = directory / "original backtest.csv"
    path.write_bytes(ledger.read_bytes())
    original = path.read_bytes()
    modified = path.stat().st_mtime_ns
    files = set(tmp_path.rglob("*"))

    result = _data.load_strategy_csv(str(path) if as_string else path)

    assert result.path == path
    assert result.raw.equals(pl.read_csv(ledger))
    assert result.trades["opening_price"].to_list() == [-2.0, -2.0]
    assert result.trades["ror"].to_list() == [25.0, -25.0]
    assert result.dates == (date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10))
    assert result.legs.height == 4
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == modified
    assert set(tmp_path.rglob("*")) == files


def test_load_strategy_csv_matches_named_portfolio_loader(ledger):
    named = _data.load_strategy("sample", "strategy")
    direct = _data.load_strategy_csv(ledger)
    assert direct.raw.equals(named.raw)
    assert direct.trades.equals(named.trades)
    assert direct.legs.equals(named.legs)
    assert direct.dates == named.dates
    assert direct.path == named.path


@pytest.mark.parametrize(
    "expression",
    [
        pl.lit("Jan 9, 2024").alias("Exp"),
        pl.lit(0.0).alias("Risk"),
        pl.lit(1.5).alias("Quantity"),
        pl.lit(float("nan")).alias("Open Price"),
        pl.lit("unsupported leg").alias("Legs"),
    ],
)
def test_load_strategy_csv_preserves_named_loader_validation(ledger, expression):
    pl.read_csv(ledger).with_columns(expression).write_csv(ledger)
    with pytest.raises(ValueError) as named:
        _data.load_strategy("sample", "strategy")
    with pytest.raises(ValueError) as direct:
        _data.load_strategy_csv(ledger)
    assert str(direct.value) == str(named.value)


def test_load_strategy_csv_preserves_missing_column_errors(ledger):
    pl.read_csv(ledger).drop("Quantity").write_csv(ledger)
    with pytest.raises(ValueError, match="strategy.csv.*Quantity"):
        _data.load_strategy_csv(ledger)


def test_load_strategy_csv_reports_header_only_empty_file(ledger):
    pl.read_csv(ledger).clear().write_csv(ledger)
    with pytest.raises(ValueError, match="Strategy CSV is empty: strategy.csv"):
        _data.load_strategy_csv(ledger)


def test_load_strategy_csv_rejects_empty_path():
    with pytest.raises(ValueError, match="nonempty CSV file path"):
        _data.load_strategy_csv("")


def test_load_strategy_csv_preserves_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        _data.load_strategy_csv(tmp_path / "missing.csv")


def test_debit_spread_sign_and_multicontract_max_profit(ledger):
    raw = (
        pl.read_csv(ledger)
        .with_columns(
            pl.lit("long call 4,970 / short call 4,980").alias("Legs"),
            pl.Series("Risk", [400.0, 200.0]),
            pl.lit(800.0).alias("Max Profit"),
        )
        .drop("Reward/Risk")
    )
    raw.write_csv(ledger)
    data = _data.load_strategy("sample", "strategy.csv")
    assert data.trades["opening_price"].to_list() == [2.0, 2.0]
    assert data.trades["reward_risk"].to_list() == [400.0, 400.0]


@pytest.mark.parametrize("name", ["", "..", "../strategy", "a/b", "a\\b"])
def test_rejects_unsafe_strategy_names(ledger, name):
    with pytest.raises(ValueError, match="strategy_name"):
        _data.load_strategy("sample", name)


@pytest.mark.parametrize(
    "expression, message",
    [
        (pl.lit("Jan 9, 2024").alias("Exp"), "0DTE"),
        (pl.lit("Jan 8, 2024 1:00pm").alias("Opened"), "0DTE"),
        (pl.lit(0.0).alias("Risk"), "positive"),
        (pl.lit(1.5).alias("Quantity"), "whole"),
        (pl.lit(float("nan")).alias("Open Price"), "finite"),
    ],
)
def test_rejects_invalid_trade_inputs(ledger, expression, message):
    pl.read_csv(ledger).with_columns(expression).write_csv(ledger)
    with pytest.raises(ValueError, match=message):
        _data.load_strategy("sample", "strategy")


def test_rejects_repeated_trade_day(ledger):
    raw = pl.read_csv(ledger)
    pl.concat([raw.head(1), raw.head(1)]).write_csv(ledger)
    with pytest.raises(ValueError, match="one trade"):
        _data.load_strategy("sample", "strategy")


def test_settings_require_credentials_and_do_not_dispose_external_engine(monkeypatch):
    monkeypatch.setattr(settings, "IFERA_DB_USER", "")
    with pytest.raises(ValueError, match="Configure"):
        _data.OracleMarketDataSource()
    engine = MagicMock()
    with _data.OracleMarketDataSource(engine) as source:
        assert source.engine is engine
    engine.dispose.assert_not_called()


def test_owned_engine_uses_safe_dsn_url_and_is_disposed(monkeypatch):
    monkeypatch.setattr(settings, "IFERA_DB_USER", "user")
    monkeypatch.setattr(settings, "IFERA_DB_PASSWORD", "p@ss/with:punctuation")
    monkeypatch.setattr(settings, "IFERA_DSN", "IFERA1")
    factory = MagicMock()
    monkeypatch.setattr(_data, "create_engine", factory)
    with _data.OracleMarketDataSource():
        url = factory.call_args.args[0]
        assert url.password == "p@ss/with:punctuation"
        assert url.query["dsn"] == "IFERA1"
    factory.return_value.dispose.assert_called_once()


def test_naked_short_call_has_credit_opening_price(ledger):
    pl.read_csv(ledger).with_columns(
        pl.lit("shortcall").alias("Type"),
        pl.lit("short call 4,980").alias("Legs"),
    ).write_csv(ledger)
    data = _data.load_strategy("sample", "strategy")
    assert data.trades["opening_price"].to_list() == [-2.0, -2.0]


class _TemporaryClob:
    """Store a bound request document without emulating Oracle's SQL execution."""

    def __init__(self):
        self.value = ""

    def write(self, value):
        assert isinstance(value, str)
        self.value = value

    def read(self):
        return self.value

    def close(self):
        raise AssertionError("An unopened temporary Oracle LOB must not be closed")


class _MarketCursor:
    """Record fixed SQL calls and stream independently supplied result rows."""

    def __init__(self, underlying, options, fetch_limit=None, failure=None):
        self.rows = {
            _oracle_queries.UNDERLYING_SQL: underlying,
            _oracle_queries.OPTION_SQL: options,
        }
        self.fetch_limit = fetch_limit
        self.failure = failure
        self.arraysize = None
        self.prefetchrows = None
        self.inputsizes = {}
        self.executions = []
        self.fetch_calls = 0
        self.remaining = []
        self.closed = False

    def setinputsizes(self, **sizes):
        self.inputsizes = sizes

    def execute(self, statement, parameters):
        assert statement in self.rows, "Unexpected SQL or runtime schema discovery"
        assert set(parameters) == {"requests"}
        document = parameters["requests"]
        assert isinstance(document, _TemporaryClob)
        self.executions.append(
            {
                "statement": statement,
                "requests": json.loads(document.read()),
                "payload": document.read(),
                "arraysize": self.arraysize,
                "prefetchrows": self.prefetchrows,
                "inputsizes": dict(self.inputsizes),
            }
        )
        if self.failure is not None and statement == _oracle_queries.OPTION_SQL:
            raise self.failure
        self.remaining = list(self.rows[statement])

    def fetchmany(self, size=None):
        self.fetch_calls += 1
        limit = self.fetch_limit if self.fetch_limit is not None else size
        limit = len(self.remaining) if limit is None else limit
        chunk, self.remaining = self.remaining[:limit], self.remaining[limit:]
        return chunk

    def close(self):
        self.closed = True


def _engine(underlying, options, fetch_limit=None, failure=None):
    cursor = _MarketCursor(underlying, options, fetch_limit, failure)
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    dbapi = connection.connection.dbapi_connection
    dbapi.cursor.return_value = cursor
    dbapi.createlob.side_effect = lambda _kind: _TemporaryClob()
    return engine, cursor, dbapi


def _strategy(ledger, count=2, symbols=("SPX",)):
    """Construct valid independent trade days, retaining normalized leg metadata."""
    original = _data.load_strategy("sample", "strategy")
    days = (
        _data._nyse_market_dates(date(2024, 1, 8), date(2025, 12, 31))["date"]
        .head(count)
        .to_list()
    )
    trades, legs, raw = [], [], []
    original_legs = original.legs.filter(pl.col("trade_index") == 0).to_dicts()
    for index, day in enumerate(days):
        symbol = symbols[index % len(symbols)]
        trade = original.trades.row(0, named=True)
        trade.update(
            trade_index=index,
            symbol=symbol,
            expiration=day,
            opened=datetime.combine(day, datetime.min.time()).replace(hour=13),
            closed=datetime.combine(day, datetime.min.time()).replace(hour=16),
        )
        trades.append(trade)
        for leg in original_legs:
            legs.append(
                {
                    **leg,
                    "trade_index": index,
                    "symbol": symbol,
                    "expiration": day,
                    "date": day,
                    "opened": trade["opened"],
                    "closed": trade["closed"],
                }
            )
        row = original.raw.row(0, named=True)
        row.update(
            Symbol=symbol,
            Opened=trade["opened"].strftime("%Y-%m-%d %H:%M:%S"),
            Closed=trade["closed"].strftime("%Y-%m-%d %H:%M:%S"),
            Exp=str(day),
        )
        raw.append(row)
    calendar = tuple(_data._nyse_market_dates(days[0], days[-1])["date"].to_list())
    return _data.StrategyData(
        pl.from_dicts(raw),
        pl.from_dicts(trades),
        pl.from_dicts(legs),
        calendar,
        original.path,
    )


def _market_rows(data):
    underlying, options = [], []
    for trade in data.trades.iter_rows(named=True):
        opened = trade["opened"].replace(second=0, microsecond=0)
        underlying.extend(
            [
                (trade["symbol"], opened, Decimal("5000.23")),
                (trade["symbol"], opened + timedelta(minutes=1), Decimal("5001.25")),
            ]
        )
        legs = data.legs.filter(pl.col("trade_index") == trade["trade_index"])
        for leg in legs.iter_rows(named=True):
            options.append(
                (
                    trade["trade_index"],
                    opened,
                    leg["direction"],
                    leg["right"],
                    Decimal(str(leg["strike"])),
                    Decimal("1.10"),
                    Decimal("1.30"),
                    Decimal("1.20"),
                    Decimal("-0.25"),
                )
            )
    return underlying, options


@pytest.mark.parametrize("trade_count", [1, 2, 100])
def test_static_loader_uses_two_bulk_statements_for_any_trade_count(
    ledger, trade_count
):
    data = _strategy(ledger, trade_count)
    engine, cursor, dbapi = _engine(*_market_rows(data))
    result = _data.OracleMarketDataSource(engine).load(data, lookback_days=7)
    assert len(cursor.executions) == 2
    assert [call["statement"] for call in cursor.executions] == [
        _oracle_queries.UNDERLYING_SQL,
        _oracle_queries.OPTION_SQL,
    ]
    assert result.options.height == trade_count * 2
    assert result.underlying.height == trade_count * 2
    assert cursor.closed
    engine.connect.assert_called_once()
    engine.dispose.assert_not_called()
    assert dbapi.createlob.call_count == 2
    for call in dbapi.createlob.call_args_list:
        assert call.args == (oracledb.DB_TYPE_CLOB,)
    for call in cursor.executions:
        assert call["arraysize"] == 8192
        assert call["prefetchrows"] == 8193
        assert call["inputsizes"] == {"requests": oracledb.DB_TYPE_CLOB}
        assert "USER_SYNONYMS" not in call["statement"].upper()
        assert "ALL_TAB_COLUMNS" not in call["statement"].upper()


def test_bulk_requests_preserve_contract_identity_and_exact_time_bounds(ledger):
    data = _strategy(ledger)
    data.trades[0, "opened"] = datetime(2024, 1, 8, 13, 0, 45)
    data.trades[0, "closed"] = datetime(2024, 1, 8, 13, 0, 45)
    data.trades[1, "closed"] = data.trades[1, "opened"]
    engine, cursor, _ = _engine(*_market_rows(data))
    _data.OracleMarketDataSource(engine).load(data, lookback_days=7)
    history = cursor.executions[0]["requests"]
    assert history == [
        {
            "symbol": "SPX",
            "start": (
                datetime.combine(data.dates[0], datetime.min.time()) - timedelta(days=7)
            ).isoformat(timespec="microseconds"),
            "end": datetime.combine(
                data.dates[-1] + timedelta(days=1), datetime.min.time()
            ).isoformat(timespec="microseconds"),
        }
    ]
    requests = cursor.executions[1]["requests"]
    assert len(requests) == 4
    first = [item for item in requests if item["trade_index"] == 0]
    assert {item["right"] for item in first} == {"P"}
    assert {item["strike"] for item in first} == {4970.0, 4980.0}
    assert {item["direction"] for item in first} == {"long", "short"}
    assert {item["symbol"] for item in first} == {"SPX"}
    assert {item["expiration"] for item in first} == {"2024-01-08"}
    assert {item["start"] for item in first} == {"2024-01-08T13:00:00.000000"}
    assert {item["end"] for item in first} == {"2024-01-08T13:00:45.000000"}
    second = [item for item in requests if item["trade_index"] == 1]
    assert {item["end"] for item in second} == {"2024-01-09T13:00:00.000001"}


def test_large_request_documents_are_bound_as_temporary_clobs(ledger):
    data = _strategy(ledger, 100)
    engine, cursor, _ = _engine(*_market_rows(data))
    _data.OracleMarketDataSource(engine).load(data)
    options = cursor.executions[1]
    assert len(options["payload"].encode()) > 32767
    assert len(options["requests"]) == 200
    assert "2024-01-08" not in options["statement"]
    assert options["payload"] not in options["statement"]
    assert ":requests" in options["statement"]


def test_daily_completion_runs_after_oracle_connection_is_released(
    ledger, isolated_daily_history
):
    data = _strategy(ledger)
    engine, cursor, _ = _engine(*_market_rows(data))
    expected = pl.DataFrame(
        {
            "symbol": ["SPX"],
            "date": [date(2024, 1, 5)],
            "open": [4990.0],
            "high": [5020.0],
            "low": [4980.0],
            "close": [5000.0],
        }
    )

    def fill(loaded, underlying, lookback):
        assert loaded is data
        assert underlying.height == 4
        assert lookback == 7
        assert cursor.closed
        engine.connect.return_value.__exit__.assert_called_once()
        return expected

    isolated_daily_history.side_effect = fill
    market = _data.OracleMarketDataSource(engine).load(data, lookback_days=7)
    assert market.daily_history is expected
    isolated_daily_history.assert_called_once()
    assert len(cursor.executions) == 2


def test_daily_completion_can_be_disabled_for_minute_only_features(
    ledger, isolated_daily_history
):
    data = _strategy(ledger)
    engine, cursor, _ = _engine(*_market_rows(data))
    market = _data.OracleMarketDataSource(engine, fill_daily_history=False).load(data)
    assert market.daily_history is None
    isolated_daily_history.assert_not_called()
    assert len(cursor.executions) == 2


def test_static_loader_normalizes_oracle_numbers_and_streams_all_chunks(ledger):
    data = _strategy(ledger, 3)
    engine, cursor, _ = _engine(*_market_rows(data), fetch_limit=2)
    result = _data.OracleMarketDataSource(engine).load(data)
    assert result.underlying.schema == {
        "symbol": pl.String,
        "datetime": pl.Datetime("us"),
        "price": pl.Float64,
    }
    assert result.options.schema == {
        "trade_index": pl.Int64,
        "datetime": pl.Datetime("us"),
        "direction": pl.String,
        "right": pl.String,
        "strike": pl.Float64,
        "bid": pl.Float64,
        "ask": pl.Float64,
        "mid": pl.Float64,
        "delta": pl.Float64,
    }
    assert result.underlying["price"].to_list() == [5000.23, 5001.25] * 3
    assert result.options["mid"].to_list() == [1.2] * 6
    assert result.options["delta"].to_list() == [-0.25] * 6
    assert cursor.fetch_calls >= 8


def test_unknown_delta_stays_missing_without_an_extra_query(ledger):
    data = _strategy(ledger)
    underlying, options = _market_rows(data)
    options = [row[:-1] + (None,) for row in options]
    engine, cursor, _ = _engine(underlying, options)
    result = _data.OracleMarketDataSource(engine).load(data)
    assert result.options.schema["delta"] == pl.Float64
    assert result.options["delta"].null_count() == result.options.height
    assert len(cursor.executions) == 2


def test_multiple_underlyings_are_requested_together_without_cross_linking(ledger):
    data = _strategy(ledger, 4, symbols=("SPX", "VIX"))
    engine, cursor, _ = _engine(*_market_rows(data))
    result = _data.OracleMarketDataSource(engine).load(data)
    assert {row["symbol"] for row in cursor.executions[0]["requests"]} == {"SPX", "VIX"}
    requests = cursor.executions[1]["requests"]
    assert {row["symbol"] for row in requests if row["trade_index"] == 1} == {"VIX"}
    assert {row["symbol"] for row in requests if row["trade_index"] == 2} == {"SPX"}
    assert result.options["trade_index"].unique().sort().to_list() == [0, 1, 2, 3]
    assert result.underlying["symbol"].unique().sort().to_list() == ["SPX", "VIX"]


def test_sparse_underlying_grid_drops_unavailable_snapshot_rows(ledger):
    data = _strategy(ledger)
    underlying, options = _market_rows(data)
    underlying.insert(0, ("SPX", datetime(2024, 1, 8, 9, 30), None))
    engine, _, _ = _engine(underlying, options)
    result = _data.OracleMarketDataSource(engine).load(data)
    assert result.underlying.height == 4
    assert result.underlying["price"].null_count() == 0


@pytest.mark.parametrize("all_null", [False, True])
def test_missing_underlying_or_root_raises_clear_error(ledger, all_null):
    data = _strategy(ledger)
    rows = [("SPX", datetime(2024, 1, 8, 13), None)] if all_null else []
    engine, cursor, _ = _engine(rows, _market_rows(data)[1])
    with pytest.raises(
        ValueError, match="underlying minute history or option root.*SPX"
    ):
        _data.OracleMarketDataSource(engine).load(data)
    assert cursor.closed


def test_missing_one_underlying_is_not_hidden_by_another_symbols_data(ledger):
    data = _strategy(ledger, symbols=("SPX", "VIX"))
    underlying, options = _market_rows(data)
    underlying = [row for row in underlying if row[0] == "SPX"]
    engine, cursor, _ = _engine(underlying, options)
    with pytest.raises(ValueError, match="VIX"):
        _data.OracleMarketDataSource(engine).load(data)
    assert cursor.closed


@pytest.mark.parametrize("missing_trade", [None, 1])
def test_missing_quotes_are_reported_for_each_original_trade(
    ledger, missing_trade, isolated_daily_history
):
    data = _strategy(ledger)
    underlying, options = _market_rows(data)
    options = [
        row for row in options if missing_trade is not None and row[0] != missing_trade
    ]
    engine, cursor, _ = _engine(underlying, options)
    with pytest.raises(ValueError, match="option quotes") as error:
        _data.OracleMarketDataSource(engine).load(data)
    assert "1" in str(error.value) if missing_trade == 1 else "0" in str(error.value)
    assert cursor.closed
    isolated_daily_history.assert_not_called()


@pytest.mark.parametrize("missing_indices", [(1,), (0, 1)])
def test_missing_quote_error_identifies_each_trade_date_and_symbol(
    ledger, missing_indices, isolated_daily_history
):
    data = _strategy(ledger, symbols=("SPX", "VIX"))
    underlying, options = _market_rows(data)
    options = [row for row in options if row[0] not in missing_indices]
    engine, cursor, _ = _engine(underlying, options)

    with pytest.raises(ValueError, match="No option quotes") as error:
        _data.OracleMarketDataSource(engine).load(data)

    message = str(error.value)
    for index in missing_indices:
        assert f"trade {index} (" in message
        assert f"date=2024-01-{8 + index:02d}" in message
        assert f"symbol={('SPX', 'VIX')[index]}" in message
        assert f"opened=2024-01-{8 + index:02d} 13:00:00" in message
        assert f"closed=2024-01-{8 + index:02d} 16:00:00" in message
        assert f"expiration=2024-01-{8 + index:02d}" in message
    assert f"source={data.path.name}" in message
    if len(missing_indices) > 1:
        assert message.index("trade 0 (") < message.index("trade 1 (")
    assert len(cursor.executions) == 2
    assert cursor.closed
    isolated_daily_history.assert_not_called()


def test_database_failure_closes_the_native_cursor(ledger):
    data = _strategy(ledger)
    engine, cursor, _ = _engine(
        *_market_rows(data), failure=RuntimeError("database failure")
    )
    with pytest.raises(RuntimeError, match="database failure"):
        _data.OracleMarketDataSource(engine).load(data)
    assert cursor.closed
    engine.connect.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("lookback", [False, 0, -1, 1.5, "7"])
def test_invalid_lookback_is_rejected_before_connecting(ledger, lookback):
    data = _strategy(ledger)
    engine, cursor, _ = _engine(*_market_rows(data))
    with pytest.raises(ValueError, match="positive integer"):
        _data.OracleMarketDataSource(engine).load(data, lookback_days=lookback)
    engine.connect.assert_not_called()
    assert cursor.executions == []
