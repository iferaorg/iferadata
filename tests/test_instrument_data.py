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
    torch.testing.assert_close(data.multiplier, expected)


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
    torch.testing.assert_close(data.multiplier, expected)
