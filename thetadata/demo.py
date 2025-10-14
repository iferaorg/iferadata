#!/usr/bin/env python
"""
ThetaData REST API Demo

This script demonstrates how to use the ThetaData REST API client.
Note: Requires a running ThetaData Terminal at http://localhost:25503
"""

from thetadata import (
    ThetaDataClient,
    stock_list_symbols,
    stock_snapshot_ohlc,
    option_list_symbols,
    option_list_expirations,
    index_list_symbols,
    index_snapshot_price,
)


def demo_basic_usage():
    """Demonstrate basic usage with default client."""
    print("\n=== Basic Usage Demo ===\n")

    try:
        # Get all stock symbols (will create a new client internally)
        print("Fetching stock symbols...")
        symbols = stock_list_symbols()
        print(f"Found {len(symbols)} stock symbols")
        print(f"First 5 symbols:\n{symbols.head()}\n")
    except Exception as e:
        print(f"Error fetching stock symbols: {e}\n")


def demo_persistent_client():
    """Demonstrate usage with a persistent client."""
    print("\n=== Persistent Client Demo ===\n")

    try:
        # Use a persistent client for multiple requests
        with ThetaDataClient() as client:
            print("Fetching data with persistent client...")

            # Get stock snapshot
            print("\nFetching AAPL OHLC snapshot...")
            ohlc = stock_snapshot_ohlc("AAPL", client=client)
            print(f"AAPL OHLC:\n{ohlc}\n")

            # Get option symbols
            print("Fetching option symbols...")
            opt_symbols = option_list_symbols(client=client)
            print(f"Found {len(opt_symbols)} option symbols")

            # Get option expirations for AAPL
            print("\nFetching AAPL option expirations...")
            expirations = option_list_expirations("AAPL", client=client)
            print(f"Found {len(expirations)} expiration dates")
            if len(expirations) > 0:
                print(f"First 5 expirations:\n{expirations.head()}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def demo_index_data():
    """Demonstrate fetching index data."""
    print("\n=== Index Data Demo ===\n")

    try:
        # Get index symbols
        print("Fetching index symbols...")
        symbols = index_list_symbols()
        print(f"Found {len(symbols)} index symbols")
        print(f"Index symbols:\n{symbols}\n")

        # Get SPX price
        print("Fetching SPX price...")
        price = index_snapshot_price("SPX")
        print(f"SPX Price:\n{price}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def demo_custom_client():
    """Demonstrate custom client configuration."""
    print("\n=== Custom Client Configuration Demo ===\n")

    # Create a client with custom settings
    client = ThetaDataClient(base_url="http://localhost:25503/v3", timeout=30.0)
    print(f"Client configured with base_url: {client.base_url}")
    print(f"Client timeout: {client.timeout}s\n")

    # Clean up
    client.close()


def main():
    """Run all demos."""
    print("=" * 60)
    print("ThetaData REST API Demo")
    print("=" * 60)

    print("\nNote: This demo requires a running ThetaData Terminal")
    print("at http://localhost:25503 to fetch real data.\n")

    # Run demos
    demo_basic_usage()
    demo_persistent_client()
    demo_index_data()
    demo_custom_client()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
