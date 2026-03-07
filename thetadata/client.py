"""
ThetaData REST API Client

This module provides the base HTTP client for interacting with the ThetaData REST API.
"""

import io
from typing import Optional, Dict, Any
import httpx
import polars as pl


class ThetaDataClient:
    """
    Base client for ThetaData REST API.

    Handles HTTP requests and response parsing.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:25503/v3",
        timeout: float = 60.0,
    ):
        """
        Initialize the ThetaData client.

        Parameters
        ----------
        base_url : str, optional
            The base URL for the ThetaData API. Default is "http://localhost:25503/v3"
        timeout : float, optional
            Request timeout in seconds. Default is 60.0
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=self.timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "csv",
    ) -> httpx.Response:
        """
        Make an HTTP GET request to the API.

        Parameters
        ----------
        endpoint : str
            The API endpoint path (e.g., "/stock/list/symbols")
        params : dict, optional
            Query parameters for the request
        output_format : str, optional
            The desired output format. Default is "csv"

        Returns
        -------
        httpx.Response
            The HTTP response object

        Raises
        ------
        httpx.HTTPStatusError
            If the request fails
        """
        url = f"{self.base_url}{endpoint}"

        # Add format parameter if not already present
        if params is None:
            params = {}
        if "format" not in params:
            params["format"] = output_format

        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response

    def _parse_csv_response(self, response: httpx.Response) -> pl.DataFrame:
        """
        Parse CSV response into a Polars DataFrame.

        Parameters
        ----------
        response : httpx.Response
            The HTTP response containing CSV data

        Returns
        -------
        pl.DataFrame
            The parsed data as a DataFrame
        """
        text = response.text.strip()
        if not text:
            return pl.DataFrame()

        return pl.read_csv(io.StringIO(text))

    def _parse_json_response(self, response: httpx.Response) -> pl.DataFrame:
        """
        Parse JSON response into a Polars DataFrame.

        Parameters
        ----------
        response : httpx.Response
            The HTTP response containing JSON data

        Returns
        -------
        pl.DataFrame
            The parsed data as a DataFrame
        """
        data = response.json()
        if not data:
            return pl.DataFrame()

        return pl.DataFrame(data)

    def get_dataframe(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "csv",
    ) -> pl.DataFrame:
        """
        Make a request and return the result as a Polars DataFrame.

        Parameters
        ----------
        endpoint : str
            The API endpoint path
        params : dict, optional
            Query parameters for the request
        output_format : str, optional
            The desired output format ('csv' or 'json'). Default is 'csv'

        Returns
        -------
        pl.DataFrame
            The API response data as a DataFrame

        Raises
        ------
        httpx.HTTPStatusError
            If the request fails
        ValueError
            If the output format is not supported
        """
        response = self._make_request(endpoint, params, output_format)

        if output_format == "csv":
            return self._parse_csv_response(response)
        if output_format == "json":
            return self._parse_json_response(response)
        raise ValueError(f"Unsupported output format: {output_format}")
