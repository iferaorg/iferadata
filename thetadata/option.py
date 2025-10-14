"""
ThetaData Option API endpoints.

This module provides functions for accessing options data from the ThetaData REST API.
"""

from typing import Optional
import pandas as pd
from .client import ThetaDataClient


# List endpoints
def option_list_symbols(
    client: Optional[ThetaDataClient] = None, output_format: str = "csv"
) -> pd.DataFrame:
    """
    List all traded option symbols (underlying assets).

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
    endpoint = "/option/list/symbols"

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, output_format=output_format)
    return client.get_dataframe(endpoint, output_format=output_format)


def option_list_dates(
    request_type: str,
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    List all available dates for options with a given request type and symbol.

    Parameters
    ----------
    request_type : str
        The request type ('trade' or 'quote').
    symbol : str
        The underlying symbol. Use '*' for all symbols.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with available dates
    """
    endpoint = f"/option/list/dates/{request_type}"
    params = {"symbol": symbol}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_list_expirations(
    symbol: str,
    client: Optional[ThetaDataClient] = None,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    List all expiration dates for a given underlying symbol.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with expiration dates
    """
    endpoint = "/option/list/expirations"
    params = {"symbol": symbol}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_list_strikes(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    List all strike prices for a given underlying and expiration.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
    client : ThetaDataClient, optional
        The ThetaData client instance. If None, a new client is created.
    right : str, optional
        The option right ('call', 'put', or 'both'). Default is 'both'.
    output_format : str, optional
        The desired output format ('csv' or 'json'). Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with strike prices
    """
    endpoint = "/option/list/strikes"
    params = {"symbol": symbol, "expiration": expiration, "right": right}

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_list_contracts(
    request_type: str,
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    List all option contracts for a given underlying, expiration, and request type.

    Parameters
    ----------
    request_type : str
        The request type ('trade' or 'quote').
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with option contracts
    """
    endpoint = f"/option/list/contracts/{request_type}"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


# Snapshot endpoints
def option_snapshot_ohlc(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time OHLC snapshot for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with OHLC data
    """
    endpoint = "/option/snapshot/ohlc"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_snapshot_trade(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time last trade for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with trade data
    """
    endpoint = "/option/snapshot/trade"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_snapshot_quote(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time last quote for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with quote data
    """
    endpoint = "/option/snapshot/quote"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_snapshot_open_interest(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time open interest for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
    endpoint = "/option/snapshot/open_interest"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


# Snapshot Greeks endpoints
def option_snapshot_greeks_implied_volatility(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time implied volatility for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with implied volatility data
    """
    endpoint = "/option/snapshot/greeks/implied_volatility"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_snapshot_greeks_all(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time all Greeks for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with all Greeks data
    """
    endpoint = "/option/snapshot/greeks/all"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_snapshot_greeks_first_order(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time first-order Greeks (delta, vega, rho) for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with first-order Greeks data
    """
    endpoint = "/option/snapshot/greeks/first_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_snapshot_greeks_second_order(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time second-order Greeks (gamma, vanna, vomma) for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with second-order Greeks data
    """
    endpoint = "/option/snapshot/greeks/second_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)


def option_snapshot_greeks_third_order(
    symbol: str,
    expiration: str,
    client: Optional[ThetaDataClient] = None,
    right: str = "both",
    strike: str = "*",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Get real-time third-order Greeks for options.

    Parameters
    ----------
    symbol : str
        The underlying symbol.
    expiration : str
        The expiration date in YYYYMMDD format.
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
        DataFrame with third-order Greeks data
    """
    endpoint = "/option/snapshot/greeks/third_order"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "right": right,
        "strike": strike,
    }

    if client is None:
        with ThetaDataClient() as c:
            return c.get_dataframe(endpoint, params, output_format)
    return client.get_dataframe(endpoint, params, output_format)
