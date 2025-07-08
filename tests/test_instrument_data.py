import torch
import pytest
from ifera.data_models import InstrumentData, DataManager
from ifera.masked_series import masked_artr


def test_calculate_artr(monkeypatch, base_instrument_config, ohlcv_single_date):
    dummy_data = ohlcv_single_date

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
    expected = masked_artr(dummy_data, mask, alpha=0.5, acrossday=False)

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(data.artr, expected)
