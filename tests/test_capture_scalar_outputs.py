import torch

import ifera.policies.position_maintenance_policy  # noqa: F401


def test_capture_scalar_outputs_enabled():
    assert torch._dynamo.config.capture_scalar_outputs is True
