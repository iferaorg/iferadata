import torch
import pytest

from ifera.policies import (
    AlwaysOpenPolicy,
    OpenOncePolicy,
    ArtrStopLossPolicy,
    InitialArtrStopLossPolicy,
    PercentGainMaintenancePolicy,
    ScaledArtrMaintenancePolicy,
    AlwaysFalseDonePolicy,
    SingleTradeDonePolicy,
    TradingPolicy,
)
from ifera.data_models import DataManager


class DummyData:
    def __init__(self, instrument):
        self.instrument = instrument
        self.data = torch.zeros((2, 2, 4), dtype=torch.float32)
        self.artr = torch.ones((2, 2), dtype=torch.float32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backadjust = False

    def convert_indices(self, _base, date_idx, time_idx):
        return date_idx, time_idx


def _target_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture
def dummy_instrument_data(base_instrument_config):
    return DummyData(base_instrument_config)


def test_simple_policies_to_device(dummy_instrument_data):
    device = _target_device()
    policies = [
        AlwaysOpenPolicy(direction=1, batch_size=1, device=torch.device("cpu")),
        OpenOncePolicy(direction=1, batch_size=1, device=torch.device("cpu")),
        ArtrStopLossPolicy(dummy_instrument_data, atr_multiple=1.0),
        InitialArtrStopLossPolicy(
            dummy_instrument_data, atr_multiple=1.0, batch_size=1
        ),
        PercentGainMaintenancePolicy(
            dummy_instrument_data,
            stage1_atr_multiple=1.0,
            trailing_stop=True,
            skip_stage1=False,
            keep_percent=0.5,
            anchor_type="entry",
            batch_size=1,
        ),
        AlwaysFalseDonePolicy(batch_size=1, device=torch.device("cpu")),
        SingleTradeDonePolicy(batch_size=1, device=torch.device("cpu")),
    ]
    for policy in policies:
        policy.to(device)
        for tensor in policy.state_dict().values():
            assert tensor.device == device


def test_scaled_artr_policy_to_device(monkeypatch, dummy_instrument_data):
    def dummy_get(self, instrument_config, **_):
        return DummyData(instrument_config)

    monkeypatch.setattr(DataManager, "get_instrument_data", dummy_get)

    policy = ScaledArtrMaintenancePolicy(
        dummy_instrument_data,
        [dummy_instrument_data.instrument.interval, "1h"],
        atr_multiple=1.0,
        wait_for_breakeven=False,
        minimum_improvement=0.1,
        batch_size=1,
    )
    device = _target_device()
    policy.to(device)
    for tensor in policy.state_dict().values():
        assert tensor.device == device


def test_trading_policy_to_device(dummy_instrument_data):
    device = _target_device()
    open_policy = AlwaysOpenPolicy(
        direction=1, batch_size=1, device=torch.device("cpu")
    )
    initial_stop = InitialArtrStopLossPolicy(
        dummy_instrument_data, atr_multiple=1.0, batch_size=1
    )
    maintenance = PercentGainMaintenancePolicy(
        dummy_instrument_data,
        stage1_atr_multiple=1.0,
        trailing_stop=True,
        skip_stage1=False,
        keep_percent=0.5,
        anchor_type="entry",
        batch_size=1,
    )
    done_policy = AlwaysFalseDonePolicy(batch_size=1, device=torch.device("cpu"))
    trading_policy = TradingPolicy(
        dummy_instrument_data,
        open_policy,
        initial_stop,
        maintenance,
        done_policy,
        batch_size=1,
    )
    trading_policy.to(device)
    for tensor in trading_policy.state_dict().values():
        assert tensor.device == device


def test_trading_policy_clone_to_device(dummy_instrument_data):
    device = _target_device()
    open_policy = AlwaysOpenPolicy(
        direction=1, batch_size=1, device=torch.device("cpu")
    )
    initial_stop = InitialArtrStopLossPolicy(
        dummy_instrument_data, atr_multiple=1.0, batch_size=1
    )
    maintenance = PercentGainMaintenancePolicy(
        dummy_instrument_data,
        stage1_atr_multiple=1.0,
        trailing_stop=True,
        skip_stage1=False,
        keep_percent=0.5,
        anchor_type="entry",
        batch_size=1,
    )
    done_policy = AlwaysFalseDonePolicy(batch_size=1, device=torch.device("cpu"))
    trading_policy = TradingPolicy(
        dummy_instrument_data,
        open_policy,
        initial_stop,
        maintenance,
        done_policy,
        batch_size=1,
    )
    cloned = trading_policy.clone(device)
    assert cloned is not trading_policy
    for key, tensor in trading_policy.state_dict().items():
        cloned_tensor = cloned.state_dict()[key]
        assert tensor.data_ptr() != cloned_tensor.data_ptr()
        assert torch.allclose(tensor.cpu(), cloned_tensor.cpu(), equal_nan=True)
        assert tensor.device == torch.device("cpu")
        assert cloned_tensor.device == device
