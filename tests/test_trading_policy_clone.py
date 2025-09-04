import torch

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
from ifera.config import BaseInstrumentConfig


def test_clone_trading_policy_for_devices(base_instrument_config: BaseInstrumentConfig):
    dummy_data = DummyData(base_instrument_config, steps=3)
    base_policy = TradingPolicy(
        instrument_data=dummy_data,  # type: ignore[arg-type]
        open_position_policy=AlwaysOpenPolicy(
            1, batch_size=1, device=dummy_data.device
        ),
        initial_stop_loss_policy=DummyInitialStopLoss(),
        position_maintenance_policy=CloseAfterOneStep(),
        trading_done_policy=SingleTradeDonePolicy(
            batch_size=1, device=dummy_data.device
        ),
        batch_size=1,
    )

    devices = [torch.device("cpu"), torch.device("cpu")]
    policies = clone_trading_policy_for_devices(base_policy, devices)

    assert len(policies) == 2
    assert policies[0] is base_policy
    assert policies[1] is not base_policy
    for policy in policies:
        assert next(policy.buffers()).device == torch.device("cpu")
