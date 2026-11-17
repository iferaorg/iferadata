"""CUDA metric kernels preserve the validated eager metric contracts."""

import pytest
import torch

from ifera.strategy_optimization import metrics

_METRICS = (
    ("annualized_return", "_annualized_return"),
    ("sortino", "_sortino"),
    ("return_over_add", "_return_over_add"),
)


@pytest.mark.parametrize("public_name,helper_name", _METRICS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eager_metric_helper_matches_public_cpu(public_name, helper_name, dtype):
    """Pure helpers support empty batches and do not alter input returns."""
    public = getattr(metrics, public_name)
    helper = getattr(metrics, helper_name)
    generator = torch.Generator().manual_seed(1729)
    for shape in ((0, 17), (1, 1), (3, 17), (7, 29)):
        returns = torch.randn(shape, dtype=dtype, generator=generator)
        original = returns.clone()
        actual = helper(returns, 12)
        torch.testing.assert_close(actual, public(returns, annualization_periods=12))
        assert actual.dtype == dtype
        assert actual.device == returns.device
        assert actual.shape == (shape[0],)
        torch.testing.assert_close(returns, original)


@pytest.mark.parametrize("public_name,helper_name", _METRICS)
def test_cpu_metrics_do_not_compile(monkeypatch, public_name, helper_name):
    """Ordinary CPU metrics keep the eager implementation."""
    del helper_name

    def unexpected_compiled_call(*_args, **_kwargs):
        raise AssertionError("CPU metric unexpectedly dispatched to CUDA compilation")

    monkeypatch.setattr(metrics, f"_compiled_{public_name}", unexpected_compiled_call)
    actual = getattr(metrics, public_name)(torch.zeros(2, 3))
    torch.testing.assert_close(actual, torch.ones(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("public_name,helper_name", _METRICS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_cuda_metric_matches_eager(public_name, helper_name, dtype):
    """Dynamic kernels preserve dtype, shapes, bankruptcy, and input storage."""
    public = getattr(metrics, public_name)
    helper = getattr(metrics, helper_name)
    generator = torch.Generator().manual_seed(1729)
    for shape in ((3, 17), (7, 29)):
        returns = torch.randn(shape, dtype=dtype, generator=generator) * 3 + 0.2
        returns[0] = 0
        returns[1, :2] = torch.tensor([-125.0, 200.0], dtype=dtype)
        resident = returns.cuda()
        original = resident.clone()
        expected = helper(resident, 252)
        actual = public(resident)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (shape[0],)
        assert actual.dtype == dtype
        assert actual.device == resident.device
        torch.testing.assert_close(resident, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("public_name,helper_name", _METRICS)
def test_cuda_metrics_reject_invalid_values_without_losing_context(
    public_name, helper_name
):
    """Invalid input raises recoverable Python errors without poisoning CUDA."""
    del helper_name
    public = getattr(metrics, public_name)
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            public(torch.tensor([[value]], device="cuda"))
    torch.testing.assert_close(
        public(torch.zeros(2, 3, device="cuda")), torch.ones(2, device="cuda")
    )


@pytest.mark.parametrize("public_name,helper_name", _METRICS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_metric_helpers_report_input_validity(public_name, helper_name, dtype):
    """Finite reduction flags reflect the input independently of the metric floor."""
    del public_name
    helper = getattr(metrics, helper_name)
    checked = getattr(metrics, f"{helper_name}_with_validity")
    for values in ([1.0, -2.0], [float("nan"), 1.0], [float("inf"), 0.0]):
        returns = torch.tensor([values], dtype=dtype)
        original = returns.clone()
        scores, valid = checked(returns, 252)
        torch.testing.assert_close(scores, helper(returns, 252))
        assert valid.shape == ()
        assert valid.dtype == torch.bool
        assert valid.device == returns.device
        assert bool(valid) == bool(torch.isfinite(returns).all())
        torch.testing.assert_close(returns, original, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("public_name,helper_name", _METRICS)
def test_cuda_metric_validation_error_order(public_name, helper_name):
    """Fused validation keeps invalid-input errors ahead of invalid-period errors."""
    del helper_name
    public = getattr(metrics, public_name)
    with pytest.raises(ValueError, match="finite"):
        public(torch.tensor([[float("nan")]], device="cuda"), annualization_periods=0)
    with pytest.raises(ValueError, match="positive integer"):
        public(torch.zeros(2, 3, device="cuda"), annualization_periods=0)
    torch.testing.assert_close(
        public(torch.zeros(2, 3, device="cuda")), torch.ones(2, device="cuda")
    )
