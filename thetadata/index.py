"""
ThetaData Index API endpoints.

This module provides functions for accessing index data from the ThetaData REST API.
"""

from typing import Optional
import pandas as pd
from .client import ThetaDataClient


def index_list_symbols(
    client: Optional[ThetaDataClient] = None, output_format: str = "csv"
) -> pd.DataFrame:
    """
    List all traded index symbols.

    Parameters
    ----------
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: symbol
    """
    endpoint = "/index/list/symbols"

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, output_format=output_format)
    return client.get_dataframe(endpoint, output_format=output_format)


def index_list_dates(
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    List all available dates for an index symbol.

    Parameters
    ----------
    symbol : str
        The index symbol. Use '*' for all symbols.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with available dates
    """
    endpoint = "/index/list/dates"
    params = {"symbol": symbol}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def index_snapshot_ohlc(
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time OHLC snapshot for indices.

    Parameters
    ----------
    symbol : str
        The index symbol. Use '*' for all symbols, or comma-separated for multiple.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLC data including timestamp, symbol, volume, high, low, count, close, open
    """
    endpoint = "/index/snapshot/ohlc"
    params = {"symbol": symbol}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def index_snapshot_price(
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time price snapshot for indices.

    Parameters
    ----------
    symbol : str
        The index symbol. Use '*' for all symbols, or comma-separated for multiple.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with price data including timestamp, symbol, price
    """
    endpoint = "/index/snapshot/price"
    params = {"symbol": symbol}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def index_history_eod(
    symbol: str,
    start_date: str,
    end_date: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get end-of-day historical data for indices.

    Parameters
    ----------
    symbol : str
        The index symbol.
    start_date : str
        The start date in YYYYMMDD format.
    end_date : str
        The end date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with EOD data
    """
    endpoint = "/index/history/eod"
    params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def index_history_ohlc(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical OHLC data for indices.

    Parameters
    ----------
    symbol : str
        The index symbol.
    start_date : str
        The start date in YYYYMMDD format.
    end_date : str
        The end date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLC data
    """
    endpoint = "/index/history/ohlc"
    params = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def index_history_price(
    symbol: str,
    start_date: str,
    end_date: str,
    client: Optional[ThetaDataClient] = None,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical price data for indices.

    Parameters
    ----------
    symbol : str
        The index symbol.
    start_date : str
        The start date in YYYYMMDD format.
    end_date : str
        The end date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with price data
    """
    endpoint = "/index/history/price"
    params = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def index_at_time_price(
    symbol: str,
    date: str,
    time: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get price data at a specific time for indices.

    Parameters
    ----------
    symbol : str
        The index symbol.
    date : str
        The date in YYYYMMDD format.
    time : str
        The time in HH:MM:SS format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with price data at the specified time
    """
    endpoint = "/index/at_time/price"
    params = {"symbol": symbol, "date": date, "time": time}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)
