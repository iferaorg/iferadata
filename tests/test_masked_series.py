import torch
import pytest
from ifera.masked_series import (
    ohlcv_to_masked,
    compress_tensor,
    decompress_tensor,
    masked_sma,
    masked_ema,
    masked_rtr,
    masked_artr,
)
from ifera.series import sma, ema, rtr, artr  # used for expected values
from torch.masked import masked_tensor
from typing import cast

# Fixtures


@pytest.fixture
def sample_vector():
    # A simple 1D tensor for testing SMA and EMA functions.
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_vector_masked_all_true(sample_vector):
    # Create a MaskedTensor where all entries are valid.
    mask = torch.ones_like(sample_vector, dtype=torch.bool)
    return masked_tensor(sample_vector, mask)


@pytest.fixture
def sample_vector_masked_partial():
    # Create a MaskedTensor with some entries masked out.
    # Here, positions 1 and 3 are invalid.
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, False, True, False, True])
    return masked_tensor(data, mask)


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
    mask = torch.ones_like(data, dtype=torch.bool)
    return masked_tensor(data, mask)


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
    mask = torch.ones_like(ohlcv_single_date, dtype=torch.bool)
    return masked_tensor(ohlcv_single_date, mask)


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


# Tests for ohlcv_to_masked


def test_ohlcv_to_masked():
    # Create OHLCV data where the volume (last channel) has a zero indicating missing data.
    # Using shape [1, 2, 5] to match the documented expected shape ([..., date, time, 5]).
    data = torch.tensor(
        [[[10.0, 12.0, 8.0, 10.0, 0.0], [11.0, 13.0, 9.0, 12.0, 150.0]]]
    )
    masked = ohlcv_to_masked(data)
    # Expected mask: derived from volume: first time step (volume=0) becomes False,
    # second time step becomes True.
    # The mask should be broadcast to the shape of data, i.e. [1, 2, 5]
    expected_mask = torch.tensor(
        [[[False, False, False, False, False], [True, True, True, True, True]]]
    )
    torch.testing.assert_close(
        masked.get_mask().float(), expected_mask.float(), rtol=1e-4, atol=1e-4
    )
    # The underlying data should match the original.
    torch.testing.assert_close(masked.get_data(), data, rtol=1e-4, atol=1e-4)


# Tests for masked SMA


def test_masked_sma(sample_vector_masked_all_true, sample_vector):
    window = 3
    # For an unmasked series, masked_sma should yield the same result as sma.
    expected = sma(sample_vector, window)
    result_masked = masked_sma(sample_vector_masked_all_true, window)
    torch.testing.assert_close(result_masked.get_data(), expected, rtol=1e-4, atol=1e-4)


# Tests for masked EMA


def test_masked_ema(sample_vector_masked_all_true, sample_vector):
    alpha = 0.5
    expected = ema(sample_vector, alpha)
    result_masked = masked_ema(sample_vector_masked_all_true, alpha)
    torch.testing.assert_close(result_masked.get_data(), expected, rtol=1e-4, atol=1e-4)


# Tests for masked RTR


def test_masked_rtr(ohlcv_two_points_masked, ohlcv_two_points):
    # For a fully valid masked tensor, masked_rtr should match rtr from series.py.
    # rtr expects shape (..., time, channels); we provided data with batch dimension.
    t = ohlcv_two_points.unsqueeze(0)
    expected = rtr(t)
    result_masked = masked_rtr(ohlcv_two_points_masked)
    result = cast(torch.Tensor, result_masked.get_data())
    # Remove the batch dimension for comparison.
    torch.testing.assert_close(
        result.squeeze(0),
        expected.squeeze(0),  # type: ignore
        rtol=1e-4,
        atol=1e-4,
    )


# Tests for masked ARTR


def test_masked_artr(ohlcv_single_date_masked, ohlcv_single_date):
    alpha = 0.5
    # Compute expected ARTR using series.artr on unmasked data.
    expected = artr(ohlcv_single_date, alpha, acrossday=False)
    result_masked = masked_artr(ohlcv_single_date_masked, alpha, acrossday=False)
    torch.testing.assert_close(result_masked.get_data(), expected, rtol=1e-4, atol=1e-4)

    # Also test acrossday=True. For a single date, the result should be identical.
    result_masked_flat = masked_artr(ohlcv_single_date_masked, alpha, acrossday=True)
    torch.testing.assert_close(
        result_masked_flat.get_data(), expected, rtol=1e-4, atol=1e-4
    )
