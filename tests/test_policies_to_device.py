"""Tests for moving policy modules between devices."""

import torch
import pytest

from ifera.policies import (
    AlwaysFalseDonePolicy,
    AlwaysOpenPolicy,
    ArtrStopLossPolicy,
    InitialArtrStopLossPolicy,
    OpenOncePolicy,
    PercentGainMaintenancePolicy,
    ScaledArtrMaintenancePolicy,
    SingleTradeDonePolicy,
    TradingPolicy,
)
from ifera.data_models import DataManager


class DummyData:
    """Minimal instrument data for policy tests."""

    def __init__(self, instrument):
        self.instrument = instrument
        self.data = torch.zeros((2, 2, 4), dtype=torch.float32)
        self.artr = torch.zeros((2, 2), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backadjust = False

    def convert_indices(self, _base, date_idx, time_idx):
        return date_idx, time_idx

    def calculate_artr(self, alpha, acrossday):
        _ = alpha
        _ = acrossday
        return None


@pytest.fixture
def dummy_instrument_data(base_instrument_config):
    """Provide dummy instrument data based on the base config."""

    return DummyData(base_instrument_config)


@pytest.mark.parametrize(
    "policy_fn",
    [
        lambda d: AlwaysOpenPolicy(1, batch_size=2, device=d.device),
        lambda d: OpenOncePolicy(1, batch_size=2, device=d.device),
        lambda d: ArtrStopLossPolicy(d, 1.0),
        lambda d: InitialArtrStopLossPolicy(d, 1.0, batch_size=2),
        lambda d: PercentGainMaintenancePolicy(
            d,
            stage1_atr_multiple=1.0,
            trailing_stop=True,
            skip_stage1=False,
            keep_percent=0.5,
            anchor_type="entry",
            batch_size=2,
        ),
        lambda d: AlwaysFalseDonePolicy(batch_size=2, device=d.device),
        lambda d: SingleTradeDonePolicy(batch_size=2, device=d.device),
    ],
)
def test_policy_to_device(policy_fn, dummy_instrument_data):
    """Policies should move buffers when ``to`` is called."""

    policy = policy_fn(dummy_instrument_data)
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(target)
    assert all(buf.device == target for buf in policy.buffers())


def test_scaled_artr_policy_to_device(monkeypatch, dummy_instrument_data):
    """ScaledArtrMaintenancePolicy should move internal modules and buffers."""

    def dummy_get(self, instrument_config, **_):
        return DummyData(instrument_config)

    monkeypatch.setattr(DataManager, "get_instrument_data", dummy_get)

    policy = ScaledArtrMaintenancePolicy(
        dummy_instrument_data,
        [dummy_instrument_data.instrument.interval, "1h"],
        atr_multiple=1.0,
        wait_for_breakeven=False,
        minimum_improvement=0.1,
        batch_size=2,
    )
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(target)
    assert all(buf.device == target for buf in policy.buffers())


def test_trading_policy_to_device(dummy_instrument_data):
    """Composite TradingPolicy should move all sub-policy buffers."""

    batch_size = 2
    open_policy = AlwaysOpenPolicy(
        1, batch_size=batch_size, device=dummy_instrument_data.device
    )
    stop_policy = InitialArtrStopLossPolicy(dummy_instrument_data, 1.0, batch_size)
    maint_policy = PercentGainMaintenancePolicy(
        dummy_instrument_data,
        stage1_atr_multiple=1.0,
        trailing_stop=True,
        skip_stage1=False,
        keep_percent=0.5,
        anchor_type="entry",
        batch_size=batch_size,
    )
    done_policy = AlwaysFalseDonePolicy(batch_size, device=dummy_instrument_data.device)
    policy = TradingPolicy(
        dummy_instrument_data,
        open_policy,
        stop_policy,
        maint_policy,
        done_policy,
        batch_size=batch_size,
    )
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(target)
    assert all(buf.device == target for buf in policy.buffers())
