import torch
import pytest

from ifera.data_models import DataManager
from ifera.config import BaseInstrumentConfig
from ifera.environments import MultiGPUSingleMarketEnv
from ifera.policies import (
    TradingPolicy,
    AlwaysOpenPolicy,
    SingleTradeDonePolicy,
    clone_trading_policy_for_devices,
)

from tests.test_single_market_env import (
    DummyData,
    DummyInitialStopLoss,
    CloseAfterOneStep,
)


@pytest.fixture
def dummy_data_three_steps_multi(base_instrument_config: BaseInstrumentConfig):
    return DummyData(base_instrument_config, steps=3)


def test_multi_gpu_env_rollout(monkeypatch, dummy_data_three_steps_multi):
    monkeypatch.setattr(
        DataManager,
        "get_instrument_data",
        lambda self, config, **_: dummy_data_three_steps_multi,
    )

    devices = [torch.device("cpu"), torch.device("cpu")]
    env = MultiGPUSingleMarketEnv(
        dummy_data_three_steps_multi.instrument, "IBKR", devices=devices
    )

    base_env = env.envs[0]
    base_policy = TradingPolicy(
        instrument_data=base_env.instrument_data,
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=1, device=base_env.instrument_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLoss(),
        position_maintenance_policy=CloseAfterOneStep(),
        trading_done_policy=SingleTradeDonePolicy(
            batch_size=1, device=base_env.instrument_data.device
        ),
        batch_size=1,
    )
    policies = clone_trading_policy_for_devices(base_policy, env.devices)

    start_d = torch.tensor([0, 0], dtype=torch.int32)
    start_t = torch.tensor([0, 0], dtype=torch.int32)

    result = env.rollout(policies, start_d, start_t, max_steps=5)
    assert result.shape == (2,)
    for sub_env in env.envs:
        assert sub_env.state["done"].all().item() is True
        assert sub_env.state["position"].sum().item() == 0
