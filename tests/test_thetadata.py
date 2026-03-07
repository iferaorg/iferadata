"""
Tests for ThetaData REST API client and endpoints.

These tests use mocking to avoid requiring an actual ThetaData server.
"""

import io
from typing import get_type_hints
from unittest.mock import Mock, patch
import polars as pl
import pytest
import httpx

from thetadata import ThetaDataClient
from thetadata import stock_list_symbols, stock_list_dates, stock_snapshot_ohlc
from thetadata import option_list_symbols, option_list_expirations
from thetadata import index_list_symbols, index_snapshot_price


class TestThetaDataClient:
    """Tests for the ThetaDataClient class."""

    def test_client_initialization(self):
        """Test that the client initializes correctly."""
        client = ThetaDataClient()
        assert client.base_url == "http://localhost:25503/v3"
        assert client.timeout == 60.0
        client.close()

    def test_client_custom_base_url(self):
        """Test that custom base URL is set correctly."""
        client = ThetaDataClient(base_url="http://example.com:8080/v3")
        assert client.base_url == "http://example.com:8080/v3"
        client.close()

    def test_client_context_manager(self):
        """Test that the client can be used as a context manager."""
        with ThetaDataClient() as client:
            assert client is not None
            assert isinstance(client.client, httpx.Client)

    def test_parse_csv_response(self):
        """Test CSV response parsing."""
        client = ThetaDataClient()
        mock_response = Mock()
        mock_response.text = "symbol\nAAPL\nGOOG\nMSFT\n"

        df = client._parse_csv_response(mock_response)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3
        assert "symbol" in df.columns
        assert df["symbol"].to_list() == ["AAPL", "GOOG", "MSFT"]
        client.close()

    def test_parse_csv_response_empty(self):
        """Test CSV response parsing with empty data."""
        client = ThetaDataClient()
        mock_response = Mock()
        mock_response.text = ""

        df = client._parse_csv_response(mock_response)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 0
        client.close()

    def test_parse_json_response(self):
        """Test JSON response parsing."""
        client = ThetaDataClient()
        mock_response = Mock()
        mock_response.json.return_value = {"symbol": ["AAPL", "GOOG", "MSFT"]}

        df = client._parse_json_response(mock_response)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3
        assert "symbol" in df.columns
        client.close()

    def test_get_dataframe_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        client = ThetaDataClient()
        with pytest.raises(ValueError, match="Unsupported output format"):
            with patch.object(client, "_make_request"):
                client.get_dataframe("/test", output_format="xml")
        client.close()


class TestStockEndpoints:
    """Tests for stock endpoints."""

    @patch("thetadata.stock.ThetaDataClient")
    def test_stock_list_symbols(self, mock_client_class):
        """Test stock_list_symbols function."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_df = pl.DataFrame({"symbol": ["AAPL", "GOOG"]})
        mock_client.get_dataframe.return_value = mock_df

        result = stock_list_symbols()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2
        mock_client.get_dataframe.assert_called_once_with(
            "/stock/list/symbols", output_format="csv"
        )

    @patch("thetadata.stock.ThetaDataClient")
    def test_stock_list_dates(self, mock_client_class):
        """Test stock_list_dates function."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_df = pl.DataFrame({"symbol": ["AAPL"], "date": ["20240101"]})
        mock_client.get_dataframe.return_value = mock_df

        result = stock_list_dates("trade", "AAPL")

        assert isinstance(result, pl.DataFrame)
        mock_client.get_dataframe.assert_called_once()
        call_args = mock_client.get_dataframe.call_args
        assert call_args[0][0] == "/stock/list/dates/trade"
        assert call_args[0][1] == {"symbol": "AAPL"}

    @patch("thetadata.stock.ThetaDataClient")
    def test_stock_snapshot_ohlc(self, mock_client_class):
        """Test stock_snapshot_ohlc function."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_df = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "open": [150.0],
                "high": [155.0],
                "low": [149.0],
                "close": [152.0],
            }
        )
        mock_client.get_dataframe.return_value = mock_df

        result = stock_snapshot_ohlc("AAPL")

        assert isinstance(result, pl.DataFrame)
        mock_client.get_dataframe.assert_called_once()
        call_args = mock_client.get_dataframe.call_args
        assert call_args[0][0] == "/stock/snapshot/ohlc"
        assert call_args[0][1] == {"symbol": "AAPL", "venue": "nqb"}


class TestOptionEndpoints:
    """Tests for option endpoints."""

    @patch("thetadata.option.ThetaDataClient")
    def test_option_list_symbols(self, mock_client_class):
        """Test option_list_symbols function."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_df = pl.DataFrame({"symbol": ["AAPL", "SPY"]})
        mock_client.get_dataframe.return_value = mock_df

        result = option_list_symbols()

        assert isinstance(result, pl.DataFrame)
        mock_client.get_dataframe.assert_called_once_with(
            "/option/list/symbols", output_format="csv"
        )

    @patch("thetadata.option.ThetaDataClient")
    def test_option_list_expirations(self, mock_client_class):
        """Test option_list_expirations function."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_df = pl.DataFrame({"expiration": ["20240119", "20240216"]})
        mock_client.get_dataframe.return_value = mock_df

        result = option_list_expirations("AAPL")

        assert isinstance(result, pl.DataFrame)
        mock_client.get_dataframe.assert_called_once()
        call_args = mock_client.get_dataframe.call_args
        assert call_args[0][0] == "/option/list/expirations"
        assert call_args[0][1] == {"symbol": "AAPL"}


class TestIndexEndpoints:
    """Tests for index endpoints."""

    @patch("thetadata.index.ThetaDataClient")
    def test_index_list_symbols(self, mock_client_class):
        """Test index_list_symbols function."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_df = pl.DataFrame({"symbol": ["SPX", "NDX"]})
        mock_client.get_dataframe.return_value = mock_df

        result = index_list_symbols()

        assert isinstance(result, pl.DataFrame)
        mock_client.get_dataframe.assert_called_once_with(
            "/index/list/symbols", output_format="csv"
        )

    @patch("thetadata.index.ThetaDataClient")
    def test_index_snapshot_price(self, mock_client_class):
        """Test index_snapshot_price function."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_df = pl.DataFrame({"symbol": ["SPX"], "price": [4500.0]})
        mock_client.get_dataframe.return_value = mock_df

        result = index_snapshot_price("SPX")

        assert isinstance(result, pl.DataFrame)
        mock_client.get_dataframe.assert_called_once()
        call_args = mock_client.get_dataframe.call_args
        assert call_args[0][0] == "/index/snapshot/price"
        assert call_args[0][1] == {"symbol": "SPX"}


class TestClientWithRealClient:
    """Tests using a real client instance passed to functions."""

    @patch.object(httpx.Client, "get")
    def test_stock_with_client_parameter(self, mock_get):
        """Test that functions work when passed a client instance."""
        mock_response = Mock()
        mock_response.text = "symbol\nAAPL\n"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with ThetaDataClient() as client:
            result = stock_list_symbols(client=client)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert result["symbol"].to_list() == ["AAPL"]


def test_public_thetadata_api_uses_polars_return_types():
    """Ensure public ThetaData APIs annotate table outputs as Polars DataFrames."""
    functions = [
        ThetaDataClient.get_dataframe,
        stock_list_symbols,
        stock_list_dates,
        stock_snapshot_ohlc,
        option_list_symbols,
        option_list_expirations,
        index_list_symbols,
        index_snapshot_price,
    ]

    for function in functions:
        hints = get_type_hints(function)
        assert hints.get("return") is pl.DataFrame
