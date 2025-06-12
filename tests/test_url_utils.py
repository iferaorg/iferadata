from pathlib import Path

from ifera.enums import Scheme, Source
from ifera import url_utils, file_utils


def test_make_url_file_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils.settings, "DATA_FOLDER", str(tmp_path))
    url = url_utils.make_url(Scheme.FILE, Source.TENSOR, "fut", "1m", "SYM")
    expected_path = Path(
        tmp_path, Source.TENSOR.value, "fut", "1m", "SYM"
    ).with_suffix(".pt.gz")
    assert url == f"file:{expected_path}"


def test_make_url_other_scheme():
    url = url_utils.make_url(Scheme.S3, Source.RAW, "data", "1d", "ABC")
    assert url == "s3:raw/data/1d/ABC.zip"
