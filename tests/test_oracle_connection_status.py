"""The Oracle script must establish a connection before reporting success."""

from unittest.mock import MagicMock

import pytest

import oracle_0dte_wfa as oracle


def test_failed_connection_does_not_print_connection_success(monkeypatch, capsys):
    """Creating an engine alone must not produce a connected status."""
    failure = RuntimeError("connection failed")
    engine = MagicMock()
    engine.connect.side_effect = failure
    monkeypatch.setattr(oracle, "create_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(
        oracle, "load_static_backtest_data", MagicMock(side_effect=failure)
    )

    with pytest.raises(RuntimeError, match="connection failed"):
        oracle.main(run_analysis=False)

    assert "Connected to Oracle successfully" not in capsys.readouterr().out
    engine.connect.assert_called_once()
    engine.dispose.assert_called_once()


def test_connection_is_verified_before_success_and_data_loading(monkeypatch, capsys):
    """A real query must succeed before success is announced or data is loaded."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    monkeypatch.setattr(oracle, "create_engine", MagicMock(return_value=engine))
    stop_after_connection = RuntimeError("stop after connection verification")

    def load_static(_engine):
        engine.connect.assert_called_once()
        connection.execute.assert_called_once()
        query = connection.execute.call_args.args[0]
        assert str(query).strip().upper() == "SELECT 1 FROM DUAL"
        assert "Connected to Oracle successfully" in capsys.readouterr().out
        raise stop_after_connection

    monkeypatch.setattr(oracle, "load_static_backtest_data", load_static)

    with pytest.raises(RuntimeError, match="stop after connection verification"):
        oracle.main(run_analysis=False)

    engine.dispose.assert_called_once()
