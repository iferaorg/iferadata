"""
ThetaData Option History API endpoints.

This module provides functions for accessing historical options data from the ThetaData REST API.
"""

from typing import Optional
import pandas as pd
from .client import ThetaDataClient


def option_history_eod(
    symbol: str,
    expiration: str,
    start_date: str,
    end_date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get end-of-day historical data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    start_date : str
        The start date in YYYYMMDD format.
    end_date : str
        The end date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with EOD data
    """
    endpoint = "/option/history/eod"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "start_date": start_date,
        "end_date": end_date,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_ohlc(
    symbol: str,
    expiration: str,
    date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical OHLC data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
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
    endpoint = "/option/history/ohlc"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "interval": interval,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_trade(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade data
    """
    endpoint = "/option/history/trade"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_quote(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical quote data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with quote data
    """
    endpoint = "/option/history/quote"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_trade_quote(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    exclusive: bool = True,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade and quote data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    exclusive : bool, optional
        If True, match quotes with timestamps < trade timestamp. Default is True.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade and quote data
    """
    endpoint = "/option/history/trade_quote"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
        "exclusive": str(exclusive).lower(),
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_open_interest(
    symbol: str,
    expiration: str,
    start_date: str,
    end_date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical open interest data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    start_date : str
        The start date in YYYYMMDD format.
    end_date : str
        The end date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with open interest data
    """
    endpoint = "/option/history/open_interest"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "start_date": start_date,
        "end_date": end_date,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


# Greeks historical endpoints
def option_history_greeks_eod(
    symbol: str,
    expiration: str,
    start_date: str,
    end_date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get end-of-day historical Greeks data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    start_date : str
        The start date in YYYYMMDD format.
    end_date : str
        The end date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with EOD Greeks data
    """
    endpoint = "/option/history/greeks/eod"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "start_date": start_date,
        "end_date": end_date,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_greeks_all(
    symbol: str,
    expiration: str,
    date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical all Greeks data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with all Greeks data
    """
    endpoint = "/option/history/greeks/all"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "interval": interval,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_trade_greeks_all(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade Greeks (all) data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade Greeks data
    """
    endpoint = "/option/history/trade_greeks/all"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_greeks_first_order(
    symbol: str,
    expiration: str,
    date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical first-order Greeks data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with first-order Greeks data
    """
    endpoint = "/option/history/greeks/first_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "interval": interval,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_trade_greeks_first_order(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade Greeks (first-order) data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade first-order Greeks data
    """
    endpoint = "/option/history/trade_greeks/first_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_greeks_second_order(
    symbol: str,
    expiration: str,
    date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical second-order Greeks data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with second-order Greeks data
    """
    endpoint = "/option/history/greeks/second_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "interval": interval,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_trade_greeks_second_order(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade Greeks (second-order) data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade second-order Greeks data
    """
    endpoint = "/option/history/trade_greeks/second_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_greeks_third_order(
    symbol: str,
    expiration: str,
    date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical third-order Greeks data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with third-order Greeks data
    """
    endpoint = "/option/history/greeks/third_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "interval": interval,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_trade_greeks_third_order(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade Greeks (third-order) data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade third-order Greeks data
    """
    endpoint = "/option/history/trade_greeks/third_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_greeks_implied_volatility(
    symbol: str,
    expiration: str,
    date: str,
    interval: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical implied volatility data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    interval : str
        The time interval (e.g., '1m', '5m', '1h').
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with implied volatility data
    """
    endpoint = "/option/history/greeks/implied_volatility"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "interval": interval,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_history_trade_greeks_implied_volatility(
    symbol: str,
    expiration: str,
    date: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get historical trade Greeks (implied volatility) data for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    start_time : str, optional
        The start time in HH:MM:SS format. Default is '09:30:00'.
    end_time : str, optional
        The end time in HH:MM:SS format. Default is '16:00:00'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade implied volatility data
    """
    endpoint = "/option/history/trade_greeks/implied_volatility"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "right": right,
        "strike": strike,
        "start_time": start_time,
        "end_time": end_time,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


# At-time endpoints
def option_at_time_trade(
    symbol: str,
    expiration: str,
    date: str,
    time: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get trade data at a specific time for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    time : str
        The time in HH:MM:SS format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with trade data at the specified time
    """
    endpoint = "/option/at_time/trade"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "time": time,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_at_time_quote(
    symbol: str,
    expiration: str,
    date: str,
    time: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get quote data at a specific time for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    date : str
        The date in YYYYMMDD format.
    time : str
        The time in HH:MM:SS format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    strike : str, optional
        The strike price or '*' for all strikes. Default is '*'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with quote data at the specified time
    """
    endpoint = "/option/at_time/quote"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "date": date,
        "time": time,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)
