from pathlib import Path

import torch
import pytest

from ifera import file_utils
from ifera.enums import Source


def test_make_path_creates_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils.settings, "DATA_FOLDER", str(tmp_path))
    path = file_utils.make_path(Source.TENSOR, "test", "1m", "AAPL")
    expected = Path(tmp_path, Source.TENSOR.value, "test", "1m", "AAPL").with_suffix(
        ".pt.gz"
    )
    assert path == expected
    assert path.parent.is_dir()


def test_make_path_tensor_backadjusted(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils.settings, "DATA_FOLDER", str(tmp_path))
    path = file_utils.make_path(Source.TENSOR_BACKADJUSTED, "futures", "1m", "AAPL")
    expected = Path(
        tmp_path,
        Source.TENSOR_BACKADJUSTED.value,
        "futures",
        "1m",
        "AAPL",
    ).with_suffix(".pt.gz")
    assert path == expected


def test_make_path_remove_file(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils.settings, "DATA_FOLDER", str(tmp_path))
    existing = Path(tmp_path, Source.RAW.value, "foo", "1h", "bar").with_suffix(".zip")
    existing.parent.mkdir(parents=True)
    existing.write_text("data")
    assert existing.exists()

    path = file_utils.make_path(Source.RAW, "foo", "1h", "bar", remove_file=True)
    assert path == existing
    assert not path.exists()


def test_make_instrument_path(tmp_path, monkeypatch, base_instrument_config):
    monkeypatch.setattr(file_utils.settings, "DATA_FOLDER", str(tmp_path))
    path = file_utils.make_instrument_path(Source.PROCESSED, base_instrument_config)
    expected = Path(
        tmp_path, Source.PROCESSED.value, "futures", "30m", "CL"
    ).with_suffix(".zip")
    assert path == expected


def test_write_and_read_tensor(tmp_path):
    file_name = tmp_path / "tensor.pt.gz"
    tensor = torch.arange(6).reshape(2, 3)
    file_utils.write_tensor_to_gzip(file_name.as_posix(), tensor)
    loaded = file_utils.read_tensor_from_gzip(file_name.as_posix())
    torch.testing.assert_close(loaded, tensor)
