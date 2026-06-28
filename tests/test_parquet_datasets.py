"""Tests for parquet dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ifera.parquet_datasets import (
    DATASET_MANIFEST_FILENAME,
    local_dataset_exists,
    read_local_dataset_manifest,
    sync_local_dataset_to_s3,
    sync_s3_dataset_to_local,
    write_parquet_dataset,
)


def _write_partition_file(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": [value]}).write_parquet(path)


def test_write_parquet_dataset_writes_manifest(tmp_path):
    dataset_root = tmp_path / "raw" / "futures" / "30m" / "CL.parquet"
    df = pl.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02"],
            "time": ["09:30:00", "09:30:00"],
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100, 200],
        }
    ).with_columns(pl.col("date").str.to_date())

    write_parquet_dataset(dataset_root, df, partition_columns=["date"])

    manifest_path = dataset_root / DATASET_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert local_dataset_exists(dataset_root) is True
    assert manifest["partition_columns"] == ["date"]
    assert len(manifest["files"]) == 2


def test_sync_s3_dataset_to_local_downloads_only_changed_partitions(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "raw.parquet"
    stale_path = dataset_root / "date=2022-01-01" / "00000000.parquet"
    obsolete_path = dataset_root / "date=2022-01-02" / "00000000.parquet"
    _write_partition_file(stale_path, 1)
    _write_partition_file(obsolete_path, 2)
    (dataset_root / DATASET_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "format": "parquet_hive_dataset",
                "format_version": 1,
                "compression": "zstd",
                "partition_columns": ["date"],
                "files": [
                    {
                        "path": "date=2022-01-01/00000000.parquet",
                        "size": 1,
                        "checksum_md5": "old",
                    },
                    {
                        "path": "date=2022-01-02/00000000.parquet",
                        "size": 1,
                        "checksum_md5": "obsolete",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    remote_manifest = {
        "format": "parquet_hive_dataset",
        "format_version": 1,
        "compression": "zstd",
        "partition_columns": ["date"],
        "files": [
            {
                "path": "date=2022-01-01/00000000.parquet",
                "size": 10,
                "checksum_md5": "new",
            },
            {
                "path": "date=2022-01-03/00000000.parquet",
                "size": 10,
                "checksum_md5": "newer",
            },
        ],
    }
    downloads: list[str] = []

    def _download(key: str, target: str) -> None:
        downloads.append(key)
        _write_partition_file(Path(target), 99)

    monkeypatch.setattr(
        "ifera.parquet_datasets.read_s3_dataset_manifest",
        lambda key: remote_manifest,
    )
    monkeypatch.setattr("ifera.parquet_datasets.download_s3_file", _download)

    sync_s3_dataset_to_local("raw/futures/30m/CL.parquet", dataset_root)

    assert sorted(downloads) == [
        "raw/futures/30m/CL.parquet/date=2022-01-01/00000000.parquet",
        "raw/futures/30m/CL.parquet/date=2022-01-03/00000000.parquet",
    ]
    assert obsolete_path.exists() is False
    assert (dataset_root / "date=2022-01-03" / "00000000.parquet").exists() is True
    assert read_local_dataset_manifest(dataset_root) == remote_manifest


def test_sync_local_dataset_to_s3_uploads_changed_partitions_and_deletes_removed(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "processed.parquet"
    current_path = dataset_root / "trade_date=2022-01-01" / "00000000.parquet"
    new_path = dataset_root / "trade_date=2022-01-02" / "00000000.parquet"
    _write_partition_file(current_path, 1)
    _write_partition_file(new_path, 2)
    manifest = {
        "format": "parquet_hive_dataset",
        "format_version": 1,
        "compression": "zstd",
        "partition_columns": ["trade_date"],
        "files": [
            {
                "path": "trade_date=2022-01-01/00000000.parquet",
                "size": current_path.stat().st_size,
                "checksum_md5": "same",
            },
            {
                "path": "trade_date=2022-01-02/00000000.parquet",
                "size": new_path.stat().st_size,
                "checksum_md5": "new",
            },
        ],
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / DATASET_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    remote_manifest = {
        "format": "parquet_hive_dataset",
        "format_version": 1,
        "compression": "zstd",
        "partition_columns": ["trade_date"],
        "files": [
            {
                "path": "trade_date=2022-01-01/00000000.parquet",
                "size": current_path.stat().st_size,
                "checksum_md5": "same",
            },
            {
                "path": "trade_date=2021-12-31/00000000.parquet",
                "size": 10,
                "checksum_md5": "obsolete",
            },
        ],
    }
    uploads: list[str] = []
    deletes: list[str] = []
    manifests: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        "ifera.parquet_datasets.read_s3_dataset_manifest",
        lambda key: remote_manifest,
    )
    monkeypatch.setattr(
        "ifera.parquet_datasets.upload_s3_file",
        lambda key, local_path: uploads.append(f"{key}:{Path(local_path).name}"),
    )
    monkeypatch.setattr(
        "ifera.parquet_datasets.delete_s3_file",
        lambda key: deletes.append(key),
    )
    monkeypatch.setattr(
        "ifera.parquet_datasets.put_s3_json_object",
        lambda key, payload: manifests.append((key, payload)),
    )

    sync_local_dataset_to_s3(dataset_root, "processed/futures/30m/CL.parquet")

    assert uploads == [
        "processed/futures/30m/CL.parquet/trade_date=2022-01-02/00000000.parquet:00000000.parquet"
    ]
    assert deletes == [
        "processed/futures/30m/CL.parquet/trade_date=2021-12-31/00000000.parquet"
    ]
    assert manifests == [("processed/futures/30m/CL.parquet", manifest)]
