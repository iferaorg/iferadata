"""Helpers for local and S3-backed parquet datasets."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Literal

import polars as pl

from .enums import Source
from .s3_utils import (
    check_s3_file_exists,
    delete_s3_file,
    download_s3_file,
    get_s3_json_object,
    get_s3_last_modified,
    list_s3_objects,
    put_s3_json_object,
    upload_s3_file,
)

DATASET_FORMAT_VERSION = 1
DATASET_MANIFEST_FILENAME = "_manifest.json"
DATASET_COMPRESSION: Literal["zstd"] = "zstd"
PARQUET_DATASET_SOURCES = frozenset({Source.RAW, Source.PROCESSED})


def is_parquet_dataset_source(source: Source | str) -> bool:
    """Return whether the source uses a parquet dataset root."""

    if isinstance(source, str):
        source = Source(source)
    return source in PARQUET_DATASET_SOURCES


def partition_columns_for_source(source: Source | str) -> list[str]:
    """Return the hive partition columns for a dataset source."""

    if isinstance(source, str):
        source = Source(source)

    if source == Source.RAW:
        return ["date"]
    if source == Source.PROCESSED:
        return ["trade_date"]
    raise ValueError(f"Source '{source.value}' is not stored as a parquet dataset")


def dataset_manifest_path(dataset_root: Path) -> Path:
    """Return the local manifest path for a dataset root."""

    return dataset_root / DATASET_MANIFEST_FILENAME


def _file_checksum(path: Path) -> str:
    """Return an md5 checksum for a file."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_file_entries(dataset_root: Path) -> list[dict[str, Any]]:
    """Collect manifest entries for local parquet partition files."""

    if not dataset_root.exists():
        return []

    entries: list[dict[str, Any]] = []
    for path in sorted(dataset_root.rglob("*.parquet")):
        if path.name == DATASET_MANIFEST_FILENAME:
            continue
        entries.append(
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "size": path.stat().st_size,
                "checksum_md5": _file_checksum(path),
            }
        )
    return entries


def build_dataset_manifest(
    partition_columns: Iterable[str],
    files: list[dict[str, Any]],
    compression: Literal["zstd"] = DATASET_COMPRESSION,
) -> dict[str, Any]:
    """Build a dataset manifest structure."""

    return {
        "format": "parquet_hive_dataset",
        "format_version": DATASET_FORMAT_VERSION,
        "compression": compression,
        "partition_columns": list(partition_columns),
        "files": files,
    }


def build_local_dataset_manifest(
    dataset_root: Path,
    partition_columns: Iterable[str],
    compression: Literal["zstd"] = DATASET_COMPRESSION,
) -> dict[str, Any]:
    """Build a manifest from files present under a local dataset root."""

    return build_dataset_manifest(
        partition_columns=partition_columns,
        files=_dataset_file_entries(dataset_root),
        compression=compression,
    )


def write_local_dataset_manifest(dataset_root: Path, manifest: dict[str, Any]) -> None:
    """Write a local dataset manifest to disk."""

    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_manifest_path(dataset_root)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_local_dataset_manifest(dataset_root: Path) -> dict[str, Any] | None:
    """Read a local dataset manifest."""

    manifest_path = dataset_manifest_path(dataset_root)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def local_dataset_exists(dataset_root: Path) -> bool:
    """Return whether a local dataset root contains a manifest."""

    return dataset_manifest_path(dataset_root).exists()


def local_dataset_mtime(dataset_root: Path) -> dt.datetime | None:
    """Return the manifest mtime for a local dataset."""

    manifest_path = dataset_manifest_path(dataset_root)
    if not manifest_path.exists():
        return None
    return dt.datetime.fromtimestamp(
        manifest_path.stat().st_mtime,
        tz=dt.timezone.utc,
    )


def touch_local_dataset_manifest(dataset_root: Path) -> None:
    """Touch the local dataset manifest to refresh the dataset timestamp."""

    manifest_path = dataset_manifest_path(dataset_root)
    if manifest_path.exists():
        manifest_path.touch()


def remove_local_dataset(dataset_root: Path) -> None:
    """Remove a local dataset root recursively."""

    if dataset_root.is_dir():
        shutil.rmtree(dataset_root)
    elif dataset_root.exists():
        dataset_root.unlink()


def _cleanup_empty_directories(dataset_root: Path) -> None:
    """Remove empty directories under a dataset root after file deletion."""

    if not dataset_root.exists():
        return

    for path in sorted(dataset_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def write_parquet_dataset(
    dataset_root: Path,
    df: pl.DataFrame,
    partition_columns: Iterable[str],
    compression: Literal["zstd"] = DATASET_COMPRESSION,
) -> None:
    """Write a hive-partitioned parquet dataset and refresh its manifest."""

    partition_columns = list(partition_columns)
    remove_local_dataset(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    if df.height > 0:
        df.write_parquet(
            str(dataset_root),
            compression=compression,
            partition_by=partition_columns,
            mkdir=True,
        )
    else:
        dataset_root.mkdir(parents=True, exist_ok=True)

    manifest = build_local_dataset_manifest(
        dataset_root, partition_columns, compression
    )
    write_local_dataset_manifest(dataset_root, manifest)


def read_s3_dataset_manifest(dataset_key: str) -> dict[str, Any] | None:
    """Read the root manifest object for an S3-backed dataset."""

    if not check_s3_file_exists(dataset_key):
        return None
    payload = get_s3_json_object(dataset_key)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset manifest stored at '{dataset_key}'")
    return payload


def _manifest_files_map(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Index manifest file entries by relative path."""

    if not manifest:
        return {}
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Dataset manifest 'files' entry must be a list")
    result: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("Malformed dataset manifest entry")
        result[str(entry["path"])] = entry
    return result


def sync_s3_dataset_to_local(dataset_key: str, dataset_root: Path) -> None:
    """Sync only changed parquet partitions from S3 to a local dataset root."""

    remote_manifest = read_s3_dataset_manifest(dataset_key)
    if remote_manifest is None:
        raise FileNotFoundError(f"S3 dataset manifest not found for '{dataset_key}'")

    local_manifest = read_local_dataset_manifest(dataset_root)
    if local_manifest is None and dataset_root.exists():
        remove_local_dataset(dataset_root)

    dataset_root.mkdir(parents=True, exist_ok=True)

    remote_files = _manifest_files_map(remote_manifest)
    local_files = _manifest_files_map(local_manifest)

    for relative_path in sorted(set(local_files) - set(remote_files)):
        target = dataset_root / relative_path
        if target.exists():
            target.unlink()

    for relative_path, remote_entry in remote_files.items():
        local_entry = local_files.get(relative_path)
        if (
            local_entry is not None
            and local_entry.get("size") == remote_entry.get("size")
            and local_entry.get("checksum_md5") == remote_entry.get("checksum_md5")
        ):
            continue

        target = dataset_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        download_s3_file(f"{dataset_key}/{relative_path}", str(target))

    _cleanup_empty_directories(dataset_root)
    write_local_dataset_manifest(dataset_root, remote_manifest)


def sync_local_dataset_to_s3(dataset_root: Path, dataset_key: str) -> None:
    """Sync only changed parquet partitions from a local dataset root to S3."""

    local_manifest = read_local_dataset_manifest(dataset_root)
    if local_manifest is None:
        raise FileNotFoundError(
            f"Local dataset manifest not found for '{dataset_root.as_posix()}'"
        )

    remote_manifest = read_s3_dataset_manifest(dataset_key)
    remote_files = _manifest_files_map(remote_manifest)

    if remote_manifest is None:
        existing_keys = list_s3_objects(f"{dataset_key}/")
        remote_files = {
            key.removeprefix(f"{dataset_key}/"): {
                "path": key.removeprefix(f"{dataset_key}/")
            }
            for key in existing_keys
        }

    local_files = _manifest_files_map(local_manifest)

    for relative_path, local_entry in local_files.items():
        remote_entry = remote_files.get(relative_path)
        if (
            remote_entry is not None
            and remote_entry.get("size") == local_entry.get("size")
            and remote_entry.get("checksum_md5") == local_entry.get("checksum_md5")
        ):
            continue

        upload_s3_file(
            f"{dataset_key}/{relative_path}", str(dataset_root / relative_path)
        )

    for relative_path in sorted(set(remote_files) - set(local_files)):
        delete_s3_file(f"{dataset_key}/{relative_path}")

    put_s3_json_object(dataset_key, local_manifest)


def s3_dataset_mtime(dataset_key: str) -> dt.datetime | None:
    """Return the root manifest mtime for an S3-backed dataset."""

    return get_s3_last_modified(dataset_key)
