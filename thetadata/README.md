# ThetaData REST API Usage Examples

This document provides examples of how to use the ThetaData REST API Python package.

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Using Default Client

The simplest way to use the API is to let each function create its own client:

```python
from thetadata import stock_list_symbols, stock_history_ohlc

# List all stock symbols
symbols = stock_list_symbols()
print(symbols.head())

# Get OHLC data for a stock
ohlc_data = stock_history_ohlc(
    symbol="AAPL",
    date="20240102",
    interval="1m"
)
print(ohlc_data.head())
```

### Using a Persistent Client

For better performance when making multiple requests, use a persistent client:

```python
from thetadata import ThetaDataClient, stock_list_symbols, stock_list_dates

with ThetaDataClient() as client:
    # List all stock symbols
    symbols = stock_list_symbols(client=client)
    
    # Get available dates for AAPL trade data
    dates = stock_list_dates("trade", "AAPL", client=client)
    
    print(f"Found {len(symbols)} symbols")
    print(f"Found {len(dates)} dates for AAPL")
```

### Custom Base URL

If you need to connect to a different server:

```python
from thetadata import ThetaDataClient, stock_list_symbols

client = ThetaDataClient(base_url="http://example.com:8080/v3")
symbols = stock_list_symbols(client=client)
client.close()
```

## Stock Endpoints

### List Symbols

```python
from thetadata import stock_list_symbols

# Get all stock symbols
symbols = stock_list_symbols()
print(symbols)
```

### Snapshot Data

```python
from thetadata import stock_snapshot_ohlc, stock_snapshot_trade, stock_snapshot_quote

# Get OHLC snapshot
ohlc = stock_snapshot_ohlc("AAPL")

# Get last trade
trade = stock_snapshot_trade("AAPL")

# Get last quote
quote = stock_snapshot_quote("AAPL")
```

### Historical Data

```python
from thetadata import stock_history_eod, stock_history_ohlc, stock_history_trade

# Get end-of-day data
eod_data = stock_history_eod(
    symbol="AAPL",
    start_date="20240101",
    end_date="20240131"
)

# Get intraday OHLC data
ohlc_data = stock_history_ohlc(
    symbol="AAPL",
    date="20240102",
    interval="5m",
    start_time="09:30:00",
    end_time="16:00:00"
)

# Get tick data
trade_data = stock_history_trade(
    symbol="AAPL",
    date="20240102"
)
```

## Option Endpoints

### List Options

```python
from thetadata import (
    option_list_symbols,
    option_list_expirations,
    option_list_strikes
)

# Get all option symbols (underlying assets)
symbols = option_list_symbols()

# Get all expiration dates for AAPL options
expirations = option_list_expirations("AAPL")

# Get all strikes for a specific expiration
strikes = option_list_strikes("AAPL", "20240119")
```

### Snapshot Data

```python
from thetadata import (
    option_snapshot_ohlc,
    option_snapshot_quote,
    option_snapshot_greeks_all
)

# Get OHLC for all strikes and rights
ohlc = option_snapshot_ohlc("AAPL", "20240119")

# Get OHLC for calls only at strike 150
ohlc_calls = option_snapshot_ohlc(
    "AAPL",
    "20240119",
    right="call",
    strike="150"
)

# Get Greeks
greeks = option_snapshot_greeks_all("AAPL", "20240119")
```

### Historical Data

```python
from thetadata import (
    option_history_eod,
    option_history_ohlc,
    option_history_greeks_all
)

# Get end-of-day option data
eod = option_history_eod(
    symbol="AAPL",
    expiration="20240119",
    start_date="20240101",
    end_date="20240110"
)

# Get intraday OHLC
ohlc = option_history_ohlc(
    symbol="AAPL",
    expiration="20240119",
    date="20240102",
    interval="5m",
    right="call",
    strike="150"
)

# Get historical Greeks
greeks = option_history_greeks_all(
    symbol="AAPL",
    expiration="20240119",
    date="20240102",
    interval="1m"
)
```

## Index Endpoints

### List and Snapshot

```python
from thetadata import (
    index_list_symbols,
    index_snapshot_ohlc,
    index_snapshot_price
)

# Get all index symbols
symbols = index_list_symbols()

# Get OHLC snapshot for SPX
ohlc = index_snapshot_ohlc("SPX")

# Get current price
price = index_snapshot_price("SPX")
```

### Historical Data

```python
from thetadata import (
    index_history_eod,
    index_history_ohlc,
    index_history_price
)

# Get end-of-day data
eod = index_history_eod(
    symbol="SPX",
    start_date="20240101",
    end_date="20240131"
)

# Get intraday OHLC
ohlc = index_history_ohlc(
    symbol="SPX",
    start_date="20240102",
    end_date="20240102",
    interval="1m"
)

# Get tick prices
prices = index_history_price(
    symbol="SPX",
    start_date="20240102",
    end_date="20240102"
)
```

## Output Formats

All functions support both CSV (default) and JSON output formats:

```python
from thetadata import stock_list_symbols

# Get data as CSV (default)
symbols_csv = stock_list_symbols(output_format="csv")

# Get data as JSON
symbols_json = stock_list_symbols(output_format="json")
```

Both formats are automatically converted to pandas DataFrames for easy manipulation.

## Error Handling

The client raises `httpx.HTTPStatusError` for failed requests:

```python
import httpx
from thetadata import stock_snapshot_ohlc

try:
    data = stock_snapshot_ohlc("INVALID_SYMBOL")
except httpx.HTTPStatusError as e:
    print(f"Request failed: {e.response.status_code}")
    print(f"Error: {e}")
```

## Notes

- All functions that return lists/tables return pandas DataFrames
- The default base URL is `http://localhost:25503/v3` (ThetaData Terminal)
- Date parameters should be in YYYYMMDD format (e.g., "20240102")
- Time parameters should be in HH:MM:SS format (e.g., "09:30:00")
- Multiple symbols can be passed comma-separated or use "*" for all symbols
