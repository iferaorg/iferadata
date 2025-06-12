import datetime
from unittest.mock import MagicMock

import pytest

from ifera import s3_utils
from ifera.enums import Source


def test_make_s3_key(base_instrument_config):
    key = s3_utils.make_s3_key(Source.RAW, base_instrument_config, zipfile=True)  # type: ignore[arg-type]
    assert key == "raw/futures/30m/ES.zip"


def test_download_s3_file(tmp_path, mock_s3, dummy_progress, monkeypatch):
    key = "foo/bar.csv"
    target = tmp_path / "bar.csv"
    mock_s3.client.head_object.return_value = {"ContentLength": 4}
    monkeypatch.setattr(s3_utils.os, "makedirs", MagicMock())
    s3_utils.download_s3_file(key, str(target))
    mock_s3.client.head_object.assert_called_once_with(
        Bucket=s3_utils.settings.S3_BUCKET, Key=key
    )
    mock_s3.client.download_file.assert_called_once()


def test_upload_s3_file(tmp_path, mock_s3, dummy_progress):
    key = "upload/file.csv"
    local_path = tmp_path / "local.csv"
    local_path.write_text("data")
    result = s3_utils.upload_s3_file(key, str(local_path))
    mock_s3.client.upload_file.assert_called_once()
    assert key in mock_s3.last_modified
    assert "upload" in mock_s3.cached_prefixes
    assert result == key


def test_check_s3_file_exists_cache_hit(mock_s3):
    key = "exists.csv"
    mock_s3.last_modified[key] = datetime.datetime.now(tz=datetime.timezone.utc)
    assert s3_utils.check_s3_file_exists(key) is True
    mock_s3.client.list_objects_v2.assert_not_called()


def test_check_s3_file_exists_list(mock_s3):
    key = "other.csv"
    mock_s3.client.list_objects_v2.return_value = {"Contents": [{"Key": key}]}
    assert s3_utils.check_s3_file_exists(key) is True
    mock_s3.client.list_objects_v2.assert_called_once_with(
        Bucket=s3_utils.settings.S3_BUCKET, Prefix=""
    )
    assert "" in mock_s3.cached_prefixes


def test_check_s3_file_exists_cached_prefix(mock_s3):
    key = "prefix/file.csv"
    mock_s3.client.list_objects_v2.return_value = {"Contents": [{"Key": key}]}
    assert s3_utils.check_s3_file_exists(key) is True
    mock_s3.client.list_objects_v2.assert_called_once_with(
        Bucket=s3_utils.settings.S3_BUCKET, Prefix="prefix"
    )
    mock_s3.client.list_objects_v2.reset_mock()
    assert s3_utils.check_s3_file_exists(key) is True
    mock_s3.client.list_objects_v2.assert_not_called()


def test_get_s3_last_modified_cache(mock_s3):
    key = "time.csv"
    ts = datetime.datetime.now(tz=datetime.timezone.utc)
    mock_s3.last_modified[key] = ts
    assert s3_utils.get_s3_last_modified(key) == ts
    mock_s3.client.head_object.assert_not_called()


def test_get_s3_last_modified_head(mock_s3):
    key = "time2.csv"
    mock_s3.cache = False
    ts = datetime.datetime.now(tz=datetime.timezone.utc)
    mock_s3.client.head_object.return_value = {"LastModified": ts}
    assert s3_utils.get_s3_last_modified(key) == ts
    mock_s3.client.head_object.assert_called_once_with(
        Bucket=s3_utils.settings.S3_BUCKET, Key=key
    )


def test_list_s3_objects_cache(mock_s3):
    mock_s3.last_modified.update({"pre/a": 1, "pre/b": 1, "other": 1})
    keys = s3_utils.list_s3_objects("pre")
    assert set(keys) == {"pre/a", "pre/b"}


def test_delete_s3_file(mock_s3):
    key = "del.csv"
    mock_s3.last_modified[key] = 1
    s3_utils.delete_s3_file(key)
    mock_s3.client.delete_object.assert_called_once_with(
        Bucket=s3_utils.settings.S3_BUCKET, Key=key
    )
    assert key not in mock_s3.last_modified


def test_rename_s3_file(mock_s3):
    old_key = "old.csv"
    new_key = "new.csv"
    mock_s3.last_modified[old_key] = 1
    s3_utils.rename_s3_file(old_key, new_key)
    mock_s3.client.copy_object.assert_called_once_with(
        Bucket=s3_utils.settings.S3_BUCKET,
        CopySource={"Bucket": s3_utils.settings.S3_BUCKET, "Key": old_key},
        Key=new_key,
    )
    mock_s3.client.delete_object.assert_called_once_with(
        Bucket=s3_utils.settings.S3_BUCKET, Key=old_key
    )
    assert new_key in mock_s3.last_modified and old_key not in mock_s3.last_modified
