"""Tests for the S3 Batch parquet migration helpers."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import Mock

from ifera.data_loading import load_data
from ifera.s3_batch_parquet_migration import (
    convert_legacy_zip_to_parquet_dataset,
    lambda_handler,
    parquet_dataset_key_for_legacy_key,
    process_batch_task,
)


def _write_zip(path: Path, csv_name: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(csv_name, content)


def test_convert_legacy_raw_zip_to_parquet_dataset(tmp_path, monkeypatch):
    zip_path = tmp_path / "raw.zip"
    _write_zip(
        zip_path,
        "raw.csv",
        "\n".join(
            [
                "2022-01-01,09:30:00,1.0,2.0,0.5,1.5,100",
                "2022-01-02,09:30:00,1.5,2.5,1.0,2.0,200",
            ]
        )
        + "\n",
    )
    dataset_root = tmp_path / "raw.parquet"

    manifest = convert_legacy_zip_to_parquet_dataset(
        "raw/futures/30m/CL.zip",
        zip_path,
        dataset_root,
    )
    monkeypatch.setattr(
        "ifera.data_loading.make_instrument_path",
        lambda **_: dataset_root,
    )

    loaded = load_data(raw=True, instrument=Mock(), zipfile=False)

    assert manifest["partition_columns"] == ["date"]
    assert sorted(entry["path"] for entry in manifest["files"]) == [
        "date=2022-01-01/00000000.parquet",
        "date=2022-01-02/00000000.parquet",
    ]
    assert loaded.height == 2


def test_convert_legacy_processed_zip_to_parquet_dataset(tmp_path, monkeypatch):
    trade_date_ord = dt.date(2022, 1, 3).toordinal()
    zip_path = tmp_path / "processed.zip"
    _write_zip(
        zip_path,
        "processed.csv",
        "\n".join(
            [
                f"{trade_date_ord},0,{trade_date_ord},0,1.0,2.0,0.5,1.5,100",
                f"{trade_date_ord},1800,{trade_date_ord},1800,1.5,2.5,1.0,2.0,200",
            ]
        )
        + "\n",
    )
    dataset_root = tmp_path / "processed.parquet"

    manifest = convert_legacy_zip_to_parquet_dataset(
        "processed/futures/30m/CL.zip",
        zip_path,
        dataset_root,
    )
    monkeypatch.setattr(
        "ifera.data_loading.make_instrument_path",
        lambda **_: dataset_root,
    )

    loaded = load_data(raw=False, instrument=Mock(), zipfile=False)

    assert manifest["partition_columns"] == ["trade_date"]
    assert [entry["path"] for entry in manifest["files"]] == [
        "trade_date=2022-01-03/00000000.parquet"
    ]
    assert loaded.height == 2
    assert loaded["trade_date"].to_list() == [trade_date_ord, trade_date_ord]


def test_lambda_handler_migrates_batch_task(tmp_path, monkeypatch):
    source_zip = tmp_path / "source.zip"
    _write_zip(
        source_zip,
        "raw.csv",
        "2022-01-01,09:30:00,1.0,2.0,0.5,1.5,100\n",
    )
    uploads: list[str] = []
    manifests: list[tuple[str, dict]] = []

    class FakeS3Client:
        def download_file(
            self, _bucket, _key, target_path, ExtraArgs=None
        ):  # noqa: N803
            _ = ExtraArgs
            shutil.copyfile(source_zip, target_path)

        def upload_file(self, local_path, bucket, key, ExtraArgs=None):  # noqa: N803
            _ = local_path, bucket, ExtraArgs
            uploads.append(key)

        def put_object(self, Bucket, Key, Body, **kwargs):  # noqa: N803
            _ = Bucket, kwargs
            manifests.append((Key, json.loads(Body.decode("utf-8"))))

    fake_client = FakeS3Client()
    monkeypatch.setattr(
        "ifera.s3_batch_parquet_migration.boto3.client",
        lambda service_name: fake_client,
    )

    event = {
        "invocationSchemaVersion": "1.0",
        "invocationId": "invocation-1",
        "job": {"userArguments": {}},
        "tasks": [
            {
                "taskId": "task-1",
                "s3BucketArn": "arn:aws:s3:::test-bucket",
                "s3Key": "raw/futures/30m/CL.zip",
            }
        ],
    }

    result = lambda_handler(event, None)

    assert parquet_dataset_key_for_legacy_key("raw/futures/30m/CL.zip") in {
        key for key, _manifest in manifests
    }
    assert uploads == ["raw/futures/30m/CL.parquet/date=2022-01-01/00000000.parquet"]
    assert result["results"] == [
        {
            "taskId": "task-1",
            "resultCode": "Succeeded",
            "resultString": "Migrated raw/futures/30m/CL.zip to raw/futures/30m/CL.parquet",
        }
    ]


def test_process_batch_task_handles_long_task_id(tmp_path):
    source_zip = tmp_path / "source.zip"
    _write_zip(
        source_zip,
        "raw.csv",
        "2022-01-01,09:30:00,1.0,2.0,0.5,1.5,100\n",
    )
    uploads: list[str] = []
    manifests: list[tuple[str, dict]] = []

    class FakeS3Client:
        def download_file(
            self, _bucket, _key, target_path, ExtraArgs=None
        ):  # noqa: N803
            _ = ExtraArgs
            shutil.copyfile(source_zip, target_path)

        def upload_file(self, local_path, bucket, key, ExtraArgs=None):  # noqa: N803
            _ = local_path, bucket, ExtraArgs
            uploads.append(key)

        def put_object(self, Bucket, Key, Body, **kwargs):  # noqa: N803
            _ = Bucket, kwargs
            manifests.append((Key, json.loads(Body.decode("utf-8"))))

    task = {
        "taskId": "A" * 400,
        "s3BucketArn": "arn:aws:s3:::test-bucket",
        "s3Key": "raw/futures/30m/CL.zip",
    }

    result = process_batch_task(
        task,
        s3_client=FakeS3Client(),
        workdir_base=tmp_path / "ifera-parquet-migration",
    )

    assert result == "Migrated raw/futures/30m/CL.zip to raw/futures/30m/CL.parquet"
    assert parquet_dataset_key_for_legacy_key("raw/futures/30m/CL.zip") in {
        key for key, _manifest in manifests
    }
    assert uploads == ["raw/futures/30m/CL.parquet/date=2022-01-01/00000000.parquet"]
