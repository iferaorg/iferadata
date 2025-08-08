"""Utility helpers for working with PyTorch devices."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def get_devices(devices: Optional[list[torch.device]] = None) -> list[torch.device]:
    """Return a list of devices using CUDA devices by default.

    If ``devices`` is ``None``, all available CUDA devices are returned. When
    no CUDA devices are available a single CPU device is used instead.
    """
    if devices is not None:
        return devices

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{idx}") for idx in range(device_count)]


def get_module_device(module: nn.Module) -> torch.device:
    """Return the device on which ``module`` is allocated."""
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")
