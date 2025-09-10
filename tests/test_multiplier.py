import datetime
import torch
import pytest
from unittest.mock import patch, mock_open
from ifera.data_models import InstrumentData, DataManager
from ifera.config import BaseInstrumentConfig


def test_multiplier_backadjust_false(monkeypatch, base_instrument_config):
    """Test that multiplier is constant 1.0 when backadjust=False."""
    dummy_data = torch.zeros((5, 10, 9), dtype=torch.float32)

    def dummy_load(self):
        self._data = dummy_data

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)

    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
        backadjust=False,
    )

    multiplier = data.multiplier
    assert multiplier.shape == (5, 10)
    assert torch.all(multiplier == 1.0)


def test_multiplier_backadjust_true_no_rollover_file(
    monkeypatch, base_instrument_config
):
    """Test that multiplier defaults to 1.0 when rollover file doesn't exist."""
    dummy_data = torch.zeros((5, 10, 9), dtype=torch.float32)

    def dummy_load(self):
        self._data = dummy_data

    def dummy_load_multiplier(self):
        # Simulate file not found
        self._multiplier = torch.ones((1, 1), dtype=self.dtype, device=self.device)

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)
    monkeypatch.setattr(InstrumentData, "_load_multiplier", dummy_load_multiplier)

    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
        backadjust=True,
    )

    multiplier = data.multiplier
    assert multiplier.shape == (5, 10)
    assert torch.all(multiplier == 1.0)


def test_multiplier_backadjust_true_with_rollover_data(
    monkeypatch, base_instrument_config
):
    """Test multiplier generation with rollover data."""
    # Create dummy data with specific trade_date and offset_time
    dummy_data = torch.zeros((3, 2, 9), dtype=torch.float32)

    # Set trade_date (channel 2) and offset_time (channel 3)
    base_date = datetime.date(2020, 1, 1).toordinal()
    dummy_data[:, :, 2] = torch.tensor(
        [
            [base_date, base_date],
            [base_date + 1, base_date + 1],
            [base_date + 2, base_date + 2],
        ],
        dtype=torch.float32,
    )

    dummy_data[:, :, 3] = torch.tensor(
        [[0, 3600], [0, 3600], [0, 3600]],  # 0 and 1 hour in seconds
        dtype=torch.float32,
    )

    def dummy_load(self):
        self._data = dummy_data

    # Mock rollover data
    rollover_data = {
        "rollovers": [
            {"date": "2020-01-02", "multiplier": 1.5},
            {"date": "2020-01-03", "multiplier": 0.8},
        ]
    }

    def dummy_load_multiplier(self):
        self._multiplier = self._generate_multiplier_tensor(rollover_data)

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)
    monkeypatch.setattr(InstrumentData, "_load_multiplier", dummy_load_multiplier)

    # Set rollover_offset to 1800 (30 minutes)
    base_instrument_config.rollover_offset = 1800

    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
        backadjust=True,
    )

    multiplier = data.multiplier
    assert multiplier.shape == (3, 2)

    # Check expected values:
    # Day 0 (2020-01-01): all 1.0 (before any rollover)
    assert torch.all(multiplier[0, :] == 1.0)

    # Day 1 (2020-01-02): 1.5 for all times (first rollover applies to entire day)
    assert (
        multiplier[1, 0] == 1.5
    )  # offset_time = 0, first rollover applies to entire day
    assert (
        multiplier[1, 1] == 1.5
    )  # offset_time = 3600, first rollover applies to entire day

    # Day 2 (2020-01-03): 1.5 before rollover_offset, 0.8 after
    assert multiplier[2, 0] == 1.5  # offset_time = 0 < 1800, inherits previous rollover
    assert (
        multiplier[2, 1] == 0.8
    )  # offset_time = 3600 >= 1800, new rollover takes effect


def test_multiplier_property_readonly(monkeypatch, base_instrument_config):
    """Test that multiplier property is read-only."""
    dummy_data = torch.zeros((5, 10, 9), dtype=torch.float32)

    def dummy_load(self):
        self._data = dummy_data

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)

    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
        backadjust=False,
    )

    # Should be able to read the property
    multiplier = data.multiplier
    assert multiplier is not None

    # Should not be able to assign to the property (would raise AttributeError)
    with pytest.raises(AttributeError):
        data.multiplier = torch.ones((5, 10))  # type: ignore
