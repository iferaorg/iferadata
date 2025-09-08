#!/usr/bin/env python3
"""
Reproduce the broadcasting issue that occurs on multi-GPU systems.
"""
import torch
import pytest
from unittest.mock import patch, MagicMock
from ifera.data_models import DataManager
from ifera.config import BaseInstrumentConfig
from ifera.environments import MultiGPUSingleMarketEnv
from ifera.policies import (
    TradingPolicy,
    AlwaysOpenPolicy,
    SingleTradeDonePolicy,
)
from tests.test_single_market_env import (
    DummyData,
    DummyInitialStopLoss,
    CloseAfterOneStep,
)

def create_dummy_config():
    """Create a minimal base instrument config for testing."""
    from ifera.config import BaseInstrumentConfig
    return BaseInstrumentConfig(
        symbol="TEST",
        interval="1m",
        broker_symbol="TEST",
        type="test",
        total_steps=96,
        contract_multiplier=1.0,
        commission=1.0,
        min_commission=1.0,
        max_commission_pct=0.01,
        slippage=0.01,
        reference_price=100.0,
        min_slippage=0.01,
    )

def test_multi_gpu_broadcasting_issue():
    """Reproduce the broadcasting issue that occurs with multiple devices."""
    # Create dummy config and data - use the same approach as the tests
    from tests.conftest import base_instrument_config
    import pytest
    
    # We can't easily get the fixture, so let's use a simpler approach
    # Just test the chunking logic directly
    from ifera.environments import MultiGPUSingleMarketEnv
    
    
    # Test the chunking directly without creating the full environment
    devices = [torch.device("cpu") for _ in range(8)]
    
    # Create a mock environment just to test the chunking method
    class MockMultiGPUEnv:
        def __init__(self, devices):
            self.envs = devices  # Just use devices as envs for length
            
        def _chunk_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
            batch_size = tensor.shape[0]
            per_device = (batch_size + len(self.envs) - 1) // len(self.envs)
            return [
                tensor[i * per_device : min((i + 1) * per_device, batch_size)]
                for i in range(len(self.envs))
            ]
    
    env = MockMultiGPUEnv(devices)
    
    # Create a large batch size that doesn't divide evenly across 8 devices
    # This should show the issue with uneven chunk sizes
    batch_size = 5192940  # From the error message
    
    # Create start indices with the problematic batch size
    start_d = torch.randint(0, 2, (batch_size,), dtype=torch.int32)
    start_t = torch.randint(0, 3, (batch_size,), dtype=torch.int32)
    
    # Test the chunking function directly to see the issue
    d_chunks = env._chunk_tensor(start_d)
    t_chunks = env._chunk_tensor(start_t)
    
    print(f"Original tensor size: {start_d.shape[0]}")
    print(f"Number of devices: {len(devices)}")
    print(f"Chunk sizes: {[chunk.shape[0] for chunk in d_chunks]}")
    
    # This should show uneven chunk sizes which can cause broadcasting issues
    chunk_sizes = [chunk.shape[0] for chunk in d_chunks]
    print(f"Smallest chunk: {min(chunk_sizes)}")
    print(f"Largest chunk: {max(chunk_sizes)}")
    
    # The issue is likely when these uneven chunks are used in mask operations
    # that later get broadcast together
    
    return chunk_sizes

if __name__ == "__main__":
    chunk_sizes = test_multi_gpu_broadcasting_issue()
    print(f"Chunk size variation: {max(chunk_sizes) - min(chunk_sizes)}")