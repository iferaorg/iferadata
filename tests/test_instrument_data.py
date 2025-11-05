import torch
import pytest
from ifera.data_models import InstrumentData, DataManager
from ifera.masked_series import masked_artr
from ifera.file_utils import make_path
from ifera.file_manager import FileManager
from ifera.enums import Source
from ifera import settings
import yaml
import datetime as dt


def test_calculate_artr(monkeypatch, base_instrument_config, ohlcv_single_date):
    dummy_data = torch.cat(
        [torch.zeros(ohlcv_single_date.shape[:-1] + (4,)), ohlcv_single_date],
        dim=-1,
    )

    def dummy_load(self):
        self._data = dummy_data

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)
    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
    )

    assert data.artr.numel() == 0

    result = data.calculate_artr(alpha=0.5, acrossday=False)

    mask = torch.ones(dummy_data.shape[:-1], dtype=torch.bool)
    expected = masked_artr(dummy_data[..., 4:], mask, alpha=0.5, acrossday=False)

    torch.testing.assert_close(result, expected.to(result.device))
    torch.testing.assert_close(data.artr, expected.to(result.device))


def test_multiplier_no_backadjust(
    monkeypatch, base_instrument_config, ohlcv_single_date
):
    dummy_data = torch.cat(
        [torch.zeros(ohlcv_single_date.shape[:-1] + (4,)), ohlcv_single_date],
        dim=-1,
    )

    def dummy_load(self):
        self._data = dummy_data

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)
    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
    )

    expected = torch.ones((1, dummy_data.shape[1]))
    torch.testing.assert_close(data.multiplier, expected.to(data.multiplier.device))


def test_multiplier_backadjust(monkeypatch, tmp_path, base_instrument_config):
    monkeypatch.setattr(base_instrument_config, "rollover_offset", 60)
    dummy = torch.zeros((2, 3, 9))
    day1 = dt.date(2020, 1, 1).toordinal()
    day2 = dt.date(2020, 1, 2).toordinal()
    offsets = torch.tensor([0, 60, 120], dtype=torch.float32)
    dummy[0, :, 2] = day1
    dummy[1, :, 2] = day2
    dummy[0, :, 3] = offsets
    dummy[1, :, 3] = offsets

    def dummy_load(self):
        self._data = dummy

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)
    monkeypatch.setattr(FileManager, "refresh_file", lambda self, url: None)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    path = make_path(Source.META, "futures", "rollover", base_instrument_config.symbol)
    content = [
        {"start_date": "2020-01-01", "contract_code": "A", "multiplier": 1.0},
        {"start_date": "2020-01-02", "contract_code": "B", "multiplier": 2.0},
    ]
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh)

    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
        backadjust=True,
    )

    expected = torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]])
    torch.testing.assert_close(data.multiplier, expected.to(data.multiplier.device))


def test_multiplier_backadjust_unquoted_start_date(
    monkeypatch, tmp_path, base_instrument_config
):
    monkeypatch.setattr(FileManager, "refresh_file", lambda self, url: None)
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    dummy = torch.zeros((1, 1, 9))
    dummy[0, 0, 2] = dt.date(2009, 10, 1).toordinal()
    dummy[0, 0, 3] = 0

    def dummy_load(self):
        self._data = dummy

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)

    path = make_path(Source.META, "futures", "rollover", base_instrument_config.symbol)
    content = [
        {
            "start_date": dt.date(2009, 10, 1),
            "contract_code": "X09",
            "multiplier": 2.406085968017578,
        }
    ]
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh)

    data = InstrumentData(
        base_instrument_config,
        sentinel=DataManager()._sentinel,
        backadjust=True,
    )

    expected = torch.full((1, 1), 2.406085968017578)
    torch.testing.assert_close(data.multiplier, expected.to(data.multiplier.device))


def test_copy_to(monkeypatch, base_instrument_config, ohlcv_single_date):
    dummy_data = torch.cat(
        [torch.zeros(ohlcv_single_date.shape[:-1] + (4,)), ohlcv_single_date],
        dim=-1,
    )

    def dummy_load(self):
        self._data = dummy_data

    monkeypatch.setattr(InstrumentData, "_load_data", dummy_load)

    # Create original data on CPU
    original_data = InstrumentData(
        base_instrument_config,
        dtype=torch.float32,
        device=torch.device("cpu"),
        sentinel=DataManager()._sentinel,
    )

    # Test copy_to same device (should get cached instance but still work)
    copied_same = original_data.copy_to(torch.device("cpu"))
    assert copied_same.device == torch.device("cpu")
    assert copied_same.dtype == torch.float32
    assert copied_same.instrument == original_data.instrument
    assert copied_same.backadjust == original_data.backadjust

    # Verify no ARTR data initially
    assert original_data.artr.numel() == 0
    assert copied_same.artr.numel() == 0

    # Calculate ARTR on original
    original_data.calculate_artr(alpha=0.3, acrossday=True)
    assert original_data.artr.numel() > 0
    assert original_data.artr_alpha == 0.3
    assert original_data.artr_acrossday is True

    # Copy to another device after ARTR calculation
    if torch.cuda.is_available():
        target_device = torch.device("cuda")
    else:
        # If CUDA not available, use CPU again to test the logic
        target_device = torch.device("cpu")

    copied_with_artr = original_data.copy_to(target_device)

    # Verify the copy has the same configuration
    assert copied_with_artr.device == target_device
    assert copied_with_artr.dtype == torch.float32
    assert copied_with_artr.instrument == original_data.instrument
    assert copied_with_artr.backadjust == original_data.backadjust

    # Verify ARTR was also calculated on the copy
    assert copied_with_artr.artr.numel() > 0
    assert copied_with_artr.artr_alpha == 0.3
    assert copied_with_artr.artr_acrossday is True

    # Verify ARTR data is equivalent (accounting for device differences)
    torch.testing.assert_close(copied_with_artr.artr.cpu(), original_data.artr.cpu())
