"""
ThetaData Stock API endpoints.

This module provides functions for accessing stock data from the ThetaData REST API.
"""

from typing import Optional
import pandas as pd
from .client import ThetaDataClient


def stock_list_symbols(
    client: Optional[ThetaDataClient] = None, output_format: str = "csv"
) -> pd.DataFrame:
    """
    List all traded stock symbols.

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
    endpoint = "/stock/list/symbols"

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, output_format=output_format)
    return client.get_dataframe(endpoint, output_format=output_format)


def stock_list_dates(
    request_type: str,
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    List all available dates for a stock with a given request type and symbol.

    Parameters
    ----------
    request_type : str
        The request type ('trade' or 'quote').
    symbol : str
        The stock symbol(s). Use '*' for all symbols, or comma-separated for multiple.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: symbol, date
    """
    endpoint = f"/stock/list/dates/{request_type}"
    params = {"symbol": symbol}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_snapshot_ohlc(
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time OHLC snapshot for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol(s). Use '*' for all symbols, or comma-separated for multiple.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLC data including timestamp, symbol, volume, high, low, count, close, open
    """
    endpoint = "/stock/snapshot/ohlc"
    params = {"symbol": symbol, "venue": venue}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_snapshot_trade(
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time last trade for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol(s). Use '*' for all symbols, or comma-separated for multiple.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade data including timestamp, symbol, sequence, size, condition, price
    """
    endpoint = "/stock/snapshot/trade"
    params = {"symbol": symbol, "venue": venue}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_snapshot_quote(
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time last quote for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol(s). Use '*' for all symbols, or comma-separated for multiple.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with quote data including timestamp, symbol, bid/ask prices and sizes
    """
    endpoint = "/stock/snapshot/quote"
    params = {"symbol": symbol, "venue": venue}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_history_eod(
    symbol: str,
    start_date: str,
    end_date: str,
    client: Optional[ThetaDataClient] = None,
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get end-of-day historical data for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol.
    start_date : str
        The start date in YYYYMMDD format.
    end_date : str
        The end date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with EOD data
    """
    endpoint = "/stock/history/eod"
    params = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "venue": venue,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_history_ohlc(
    symbol: str,
    date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical OHLC data for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol.
    date : str
        The date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLC data
    """
    endpoint = "/stock/history/ohlc"
    params = {
        "symbol": symbol,
        "date": date,
        "interval": interval,
        "start_time": start_time,
        "end_time": end_time,
        "venue": venue,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_history_trade(
    symbol: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade data for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade data
    """
    endpoint = "/stock/history/trade"
    params = {
        "symbol": symbol,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "venue": venue,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_history_quote(
    symbol: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical quote data for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with quote data
    """
    endpoint = "/stock/history/quote"
    params = {
        "symbol": symbol,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "venue": venue,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_history_trade_quote(
    symbol: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    venue: str = "nqb",
    exclusive: bool = True,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade and quote data for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    exclusive : bool, optional
        If True, match quotes with timestamps < trade timestamp. Default is True.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade and quote data
    """
    endpoint = "/stock/history/trade_quote"
    params = {
        "symbol": symbol,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "venue": venue,
        "exclusive": str(exclusive).lower(),
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_at_time_trade(
    symbol: str,
    date: str,
    time: str,
    client: Optional[ThetaDataClient] = None,
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get trade data at a specific time for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol.
    date : str
        The date in YYYYMMDD format.
    time : str
        The time in HH:MM:SS format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade data at the specified time
    """
    endpoint = "/stock/at_time/trade"
    params = {"symbol": symbol, "date": date, "time": time, "venue": venue}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def stock_at_time_quote(
    symbol: str,
    date: str,
    time: str,
    client: Optional[ThetaDataClient] = None,
    venue: str = "nqb",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get quote data at a specific time for stocks.

    Parameters
    ----------
    symbol : str
        The stock symbol.
    date : str
        The date in YYYYMMDD format.
    time : str
        The time in HH:MM:SS format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    venue : str, optional
        The venue ('nqb' or 'utp_cta'). Default is 'nqb'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with quote data at the specified time
    """
    endpoint = "/stock/at_time/quote"
    params = {"symbol": symbol, "date": date, "time": time, "venue": venue}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)
