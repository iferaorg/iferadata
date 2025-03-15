from typing import cast

import pytest
import torch
from ifera.masked_series import (
    compress_tensor,
    decompress_tensor,
    masked_artr,
    masked_ema,
    masked_rtr,
    masked_sma,
)
from ifera.series import artr, ema, rtr, sma  # used for expected values


# Fixtures
@pytest.fixture
def sample_vector():
    # A simple 1D tensor for testing SMA and EMA functions.
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_vector_masked_all_true(sample_vector):
    # Create a tensor with a mask where all entries are valid.
    mask = torch.ones_like(sample_vector, dtype=torch.bool)
    return sample_vector, mask


@pytest.fixture
def sample_vector_masked_partial():
    # Create a tensor with a mask where some entries are masked out.
    # Here, positions 1 and 3 are invalid.
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, False, True, False, True])
    return data, mask


@pytest.fixture
def ohlcv_two_points():
    # OHLCV data for testing functions that work on financial data.
    # Shape: [time, channels] where channels = [open, high, low, close, volume]
    return torch.tensor(
        [[10.0, 12.0, 8.0, 10.0, 100.0], [11.0, 13.0, 9.0, 12.0, 150.0]]
    )


@pytest.fixture
def ohlcv_two_points_masked(ohlcv_two_points):
    # Wrap the OHLCV tensor as a masked tensor with all entries valid.
    # Here we add a batch dimension so that the input shape is [1, time, channels]
    data = ohlcv_two_points.unsqueeze(0)
    mask = torch.ones(data.shape[:-1], dtype=torch.bool)
    return data, mask


@pytest.fixture
def ohlcv_single_date():
    # OHLCV data for a single date with multiple time points.
    # Shape: [date, time, channels]
    return torch.tensor(
        [
            [
                [10.0, 12.0, 8.0, 10.0, 100.0],
                [11.0, 13.0, 9.0, 12.0, 150.0],
                [12.0, 14.0, 10.0, 13.0, 120.0],
            ]
        ]
    )


@pytest.fixture
def ohlcv_single_date_masked(ohlcv_single_date):
    mask = torch.ones(ohlcv_single_date.shape[:-1], dtype=torch.bool)
    return ohlcv_single_date, mask


# Tests for helper functions: compress_tensor and decompress_tensor
def test_compress_decompress():
    # Test that decompressing a compressed tensor recovers the original
    # valid data in the right positions.
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, False, True, False, True])
    compressed = compress_tensor(data, mask)
    decompressed = decompress_tensor(compressed, mask)
    # In positions where mask is True, we expect original values;
    # other positions are filled with NaN.
    expected = torch.tensor([1.0, float("nan"), 3.0, float("nan"), 5.0])
    torch.testing.assert_close(
        decompressed, expected, rtol=1e-4, atol=1e-4, equal_nan=True
    )


# Tests for masked SMA
def test_masked_sma(sample_vector_masked_all_true, sample_vector):
    data, mask = sample_vector_masked_all_true
    window = 3
    # For an unmasked series, masked_sma should yield the same result as sma.
    expected = sma(sample_vector, window)
    result_data = masked_sma(data, mask, window)
    torch.testing.assert_close(result_data, expected, rtol=1e-4, atol=1e-4)


# Tests for masked EMA
def test_masked_ema(sample_vector_masked_all_true, sample_vector):
    data, mask = sample_vector_masked_all_true
    alpha = 0.5
    expected = ema(sample_vector, alpha)
    result_data = masked_ema(data, mask, alpha)
    torch.testing.assert_close(result_data, expected, rtol=1e-4, atol=1e-4)


# Tests for masked RTR
def test_masked_rtr(ohlcv_two_points_masked, ohlcv_two_points):
    # For a fully valid masked tensor, masked_rtr should match rtr from series.py.
    data, mask = ohlcv_two_points_masked
    # rtr expects shape (..., time, channels); we provided data with batch dimension.
    t = ohlcv_two_points.unsqueeze(0)
    expected = rtr(t)
    result_data = masked_rtr(data, mask)

    # Remove the batch dimension for comparison.
    torch.testing.assert_close(
        result_data.squeeze(0),
        expected.squeeze(0),  # type: ignore
        rtol=1e-4,
        atol=1e-4,
    )


# Tests for masked ARTR
def test_masked_artr(ohlcv_single_date_masked, ohlcv_single_date):
    data, mask = ohlcv_single_date_masked
    alpha = 0.5
    # Compute expected ARTR using series.artr on unmasked data.
    expected = artr(ohlcv_single_date, alpha, acrossday=False)
    result_data = masked_artr(data, mask, alpha, acrossday=False)
    torch.testing.assert_close(result_data, expected, rtol=1e-4, atol=1e-4)

    # Also test acrossday=True. For a single date, the result should be identical.
    result_data_flat = masked_artr(data, mask, alpha, acrossday=True)
    torch.testing.assert_close(result_data_flat, expected, rtol=1e-4, atol=1e-4)
