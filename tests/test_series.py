import pytest
import torch
from ifera.series import artr, ema, ema_slow, ffill, rtr, sma

# Fixtures for reusable test data


@pytest.fixture
def sample_vector():
    # A simple 1D tensor for testing sma and ema functions.
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def ohlcv_two_points():
    # Two time-point OHLCV data (shape: [time, channels]) for testing rtr.
    # Channels: [open, high, low, close, volume]
    return torch.tensor(
        [[10.0, 12.0, 8.0, 10.0, 100.0], [11.0, 13.0, 9.0, 12.0, 150.0]]
    )


@pytest.fixture
def ohlcv_single_date():
    # OHLCV data for a single date with 3 time points (shape: [date, time, channels])
    return torch.tensor(
        [
            [
                [10.0, 12.0, 8.0, 10.0, 100.0],
                [11.0, 13.0, 9.0, 12.0, 150.0],
                [12.0, 14.0, 10.0, 13.0, 120.0],
            ]
        ]
    )


# Tests for sma
def test_sma(sample_vector):
    # For window = 3:
    # Expected: [1.0, (1+2)/2=1.5, (1+2+3)/3=2.0, (2+3+4)/3=3.0, (3+4+5)/3=4.0]
    result = sma(sample_vector, 3)
    expected = torch.tensor([1.0, 1.5, 2.0, 3.0, 4.0])
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


# Tests for ema (comparing to the iterative slow version)
def test_ema(sample_vector):
    alpha = 0.5
    result = ema(sample_vector, alpha)
    expected = ema_slow(sample_vector, alpha)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_ema_chunked(sample_vector):
    # Test EMA using a chunked computation to reduce memory usage.
    alpha = 0.3
    result = ema(sample_vector, alpha, chunk_size=2)
    expected = ema_slow(sample_vector, alpha)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


# Test for forward-fill (ffill)
def test_ffill():
    # Create a tensor with NaNs.
    # Expected behavior:
    # - The first element remains NaN.
    # - The second element is 2.0.
    # - The third element fills forward with 2.0.
    # - The fourth element is 4.0.
    # - The fifth element fills forward with 4.0.
    t = torch.tensor([float("nan"), 2.0, float("nan"), 4.0, float("nan")])
    result = ffill(t)
    expected = torch.tensor([float("nan"), 2.0, 2.0, 4.0, 4.0])
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4, equal_nan=True)


# Test for rtr
def test_rtr(ohlcv_two_points):
    # rtr expects input shape (..., time, channels); add a batch dimension.
    t = ohlcv_two_points.unsqueeze(0)  # shape: (1, 2, 5)
    result = rtr(t)
    # Expected:
    # First time step: rtr = (high / low - 1) = 12/8 - 1 = 0.5
    # Second time step: rtr = (max(high, prev_close) / min(low, prev_close) - 1)
    #   prev_close from first step is 10, so max(13, 10) = 13 and min(9, 10) = 9, yielding 13/9 - 1.
    expected = torch.tensor([[0.5, 13 / 9 - 1]])
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


# Test for artr (average relative true range)
def test_artr(ohlcv_single_date):
    # Test for a single date (acrossday=False) and acrossday=True should yield the same result
    alpha = 0.5
    # Compute ARTR for a single date with 3 time points.
    result = artr(ohlcv_single_date, alpha, acrossday=False)

    # Manually compute expected rtr values:
    # Time 1: 12/8 - 1 = 0.5
    # Time 2: max(13, 10)/min(9,10) - 1 = 13/9 - 1 ≈ 0.4444444
    # Time 3: max(14, 12)/min(10,12) - 1 = 14/10 - 1 = 0.4
    expected_rtr = torch.tensor([[0.5, 13 / 9 - 1, 0.4]])
    # Now compute the EMA of these rtr values with alpha = 0.5 (using the recursive definition):
    ema_expected = torch.zeros_like(expected_rtr)
    ema_expected[0, 0] = expected_rtr[0, 0]
    ema_expected[0, 1] = (1 - alpha) * ema_expected[0, 0] + alpha * expected_rtr[0, 1]
    ema_expected[0, 2] = (1 - alpha) * ema_expected[0, 1] + alpha * expected_rtr[0, 2]

    torch.testing.assert_close(result, ema_expected, rtol=1e-4, atol=1e-4)

    # Also test with acrossday=True (flattening date and time)
    result_flat = artr(ohlcv_single_date, alpha, acrossday=True)
    torch.testing.assert_close(result_flat, ema_expected, rtol=1e-4, atol=1e-4)
