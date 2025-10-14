"""
ThetaData REST API Package

This package provides a Python interface to the ThetaData REST API v3.
All functions that return lists/tables return them as pandas DataFrames.

Example usage:
    from thetadata import stock_list_symbols, ThetaDataClient

    # Use default client (creates new client for each call)
    symbols = stock_list_symbols()

    # Or use a persistent client
    with ThetaDataClient() as client:
        symbols = stock_list_symbols(client=client)
        dates = stock_list_dates("trade", "AAPL", client=client)
"""

from .client import ThetaDataClient

# Stock endpoints
from .stock import (
    stock_list_symbols,
    stock_list_dates,
    stock_snapshot_ohlc,
    stock_snapshot_trade,
    stock_snapshot_quote,
    stock_history_eod,
    stock_history_ohlc,
    stock_history_trade,
    stock_history_quote,
    stock_history_trade_quote,
    stock_at_time_trade,
    stock_at_time_quote,
)

# Option endpoints
from .option import (
    option_list_symbols,
    option_list_dates,
    option_list_expirations,
    option_list_strikes,
    option_list_contracts,
    option_snapshot_ohlc,
    option_snapshot_trade,
    option_snapshot_quote,
    option_snapshot_open_interest,
    option_snapshot_greeks_implied_volatility,
    option_snapshot_greeks_all,
    option_snapshot_greeks_first_order,
    option_snapshot_greeks_second_order,
    option_snapshot_greeks_third_order,
)

# Option history endpoints
from .option_history import (
    option_history_eod,
    option_history_ohlc,
    option_history_trade,
    option_history_quote,
    option_history_trade_quote,
    option_history_open_interest,
    option_history_greeks_eod,
    option_history_greeks_all,
    option_history_trade_greeks_all,
    option_history_greeks_first_order,
    option_history_trade_greeks_first_order,
    option_history_greeks_second_order,
    option_history_trade_greeks_second_order,
    option_history_greeks_third_order,
    option_history_trade_greeks_third_order,
    option_history_greeks_implied_volatility,
    option_history_trade_greeks_implied_volatility,
    option_at_time_trade,
    option_at_time_quote,
)

# Index endpoints
from .index import (
    index_list_symbols,
    index_list_dates,
    index_snapshot_ohlc,
    index_snapshot_price,
    index_history_eod,
    index_history_ohlc,
    index_history_price,
    index_at_time_price,
)

__all__ = [
    # Client
    "ThetaDataClient",
    # Stock
    "stock_list_symbols",
    "stock_list_dates",
    "stock_snapshot_ohlc",
    "stock_snapshot_trade",
    "stock_snapshot_quote",
    "stock_history_eod",
    "stock_history_ohlc",
    "stock_history_trade",
    "stock_history_quote",
    "stock_history_trade_quote",
    "stock_at_time_trade",
    "stock_at_time_quote",
    # Option
    "option_list_symbols",
    "option_list_dates",
    "option_list_expirations",
    "option_list_strikes",
    "option_list_contracts",
    "option_snapshot_ohlc",
    "option_snapshot_trade",
    "option_snapshot_quote",
    "option_snapshot_open_interest",
    "option_snapshot_greeks_implied_volatility",
    "option_snapshot_greeks_all",
    "option_snapshot_greeks_first_order",
    "option_snapshot_greeks_second_order",
    "option_snapshot_greeks_third_order",
    "option_history_eod",
    "option_history_ohlc",
    "option_history_trade",
    "option_history_quote",
    "option_history_trade_quote",
    "option_history_open_interest",
    "option_history_greeks_eod",
    "option_history_greeks_all",
    "option_history_trade_greeks_all",
    "option_history_greeks_first_order",
    "option_history_trade_greeks_first_order",
    "option_history_greeks_second_order",
    "option_history_trade_greeks_second_order",
    "option_history_greeks_third_order",
    "option_history_trade_greeks_third_order",
    "option_history_greeks_implied_volatility",
    "option_history_trade_greeks_implied_volatility",
    "option_at_time_trade",
    "option_at_time_quote",
    # Index
    "index_list_symbols",
    "index_list_dates",
    "index_snapshot_ohlc",
    "index_snapshot_price",
    "index_history_eod",
    "index_history_ohlc",
    "index_history_price",
    "index_at_time_price",
]
