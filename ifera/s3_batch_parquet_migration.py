"""Legacy zip-to-parquet migration helpers for S3 Batch Operations."""

# pylint: disable=duplicate-code

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import boto3  # type: ignore
import polars as pl

from .enums import Source
from .parquet_datasets import (
    DATASET_COMPRESSION,
    partition_columns_for_source,
    read_local_dataset_manifest,
    write_parquet_dataset,
)


def parquet_dataset_key_for_legacy_key(legacy_key: str) -> str:
    """Map a legacy raw/processed zip key to its parquet dataset manifest key."""

    if legacy_key.startswith("raw/") or legacy_key.startswith("processed/"):
        if not legacy_key.endswith(".zip"):
            raise ValueError(f"Legacy key '{legacy_key}' must end with '.zip'")
        return legacy_key.removesuffix(".zip") + ".parquet"
    raise ValueError(
        "Only legacy raw/processed zip keys can be migrated " f"(got '{legacy_key}')"
    )


def _source_from_legacy_key(legacy_key: str) -> Source:
    """Infer the dataset source from a legacy key."""

    if legacy_key.startswith("raw/"):
        return Source.RAW
    if legacy_key.startswith("processed/"):
        return Source.PROCESSED
    raise ValueError(f"Unsupported legacy key '{legacy_key}'")


def _read_legacy_zip_csv(
    zip_path: str | Path,
    columns: list[str],
    schema_overrides: dict[str, Any],
) -> pl.DataFrame:
    """Read the first CSV member from a legacy zip file."""

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if not names:
            raise ValueError(f"Zip file '{zip_path}' does not contain any members")
        with archive.open(names[0]) as csv_file:
            data = csv_file.read()

    return pl.read_csv(
        data,
        has_header=False,
        new_columns=columns,
        schema_overrides=schema_overrides,
    )


def _read_legacy_raw_zip(zip_path: str | Path) -> pl.DataFrame:
    """Read a legacy raw zip file into the parquet dataset schema."""

    df = _read_legacy_zip_csv(
        zip_path,
        columns=["date", "time", "open", "high", "low", "close", "volume"],
        schema_overrides={
            "date": pl.String,
            "time": pl.String,
            "open": pl.Float32,
            "high": pl.Float32,
            "low": pl.Float32,
            "close": pl.Float32,
            "volume": pl.Int32,
        },
    )
    return df.with_columns(pl.col("date").str.to_date(strict=False))


def _ordinal_to_date(value: int) -> dt.date:
    """Convert an ordinal integer into a date."""

    return dt.date.fromordinal(int(value))


def _read_legacy_processed_zip(zip_path: str | Path) -> pl.DataFrame:
    """Read a legacy processed zip file into the parquet dataset schema."""

    df = _read_legacy_zip_csv(
        zip_path,
        columns=[
            "ord_date",
            "time_seconds",
            "ord_trade_date",
            "offset_time_seconds",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
        schema_overrides={
            "ord_date": pl.Int32,
            "time_seconds": pl.Float64,
            "ord_trade_date": pl.Int32,
            "offset_time_seconds": pl.Float64,
            "open": pl.Float32,
            "high": pl.Float32,
            "low": pl.Float32,
            "close": pl.Float32,
            "volume": pl.Int32,
        },
    )
    return df.with_columns(
        pl.col("ord_trade_date")
        .map_elements(
            _ordinal_to_date,
            return_dtype=pl.Date,
        )
        .alias("trade_date")
    )


def convert_legacy_zip_to_parquet_dataset(
    legacy_key: str,
    zip_path: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Convert one legacy raw/processed zip file into a parquet dataset root."""

    source = _source_from_legacy_key(legacy_key)
    dataset_root = Path(dataset_root)

    if source == Source.RAW:
        df = _read_legacy_raw_zip(zip_path)
    else:
        df = _read_legacy_processed_zip(zip_path)

    write_parquet_dataset(
        dataset_root,
        df,
        partition_columns=partition_columns_for_source(source),
        compression=DATASET_COMPRESSION,
    )

    manifest = read_local_dataset_manifest(dataset_root)
    if manifest is None:
        raise RuntimeError(f"Dataset manifest was not written for '{dataset_root}'")
    return manifest


def _bucket_name_from_arn(bucket_arn: str) -> str:
    """Extract a bucket name from an S3 ARN."""

    if "arn:aws:s3:::" not in bucket_arn:
        raise ValueError(f"Invalid S3 bucket ARN '{bucket_arn}'")
    return bucket_arn.split("arn:aws:s3:::", maxsplit=1)[1]


def _download_task_object(
    s3_client: Any,
    bucket: str,
    key: str,
    target_path: Path,
    version_id: str | None = None,
) -> None:
    """Download the source object for a batch task."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    extra_args = {"VersionId": version_id} if version_id else None
    if extra_args is None:
        s3_client.download_file(bucket, key, str(target_path))
    else:
        s3_client.download_file(bucket, key, str(target_path), ExtraArgs=extra_args)


def _workdir_name_for_task(task_id: str) -> str:
    """Return a stable short directory name for a Batch Operations task."""

    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]
    return f"task-{digest}"


def _upload_dataset_to_s3(
    s3_client: Any,
    bucket: str,
    dataset_key: str,
    dataset_root: Path,
    manifest: dict[str, Any],
) -> None:
    """Upload partition files first, then publish the dataset-root manifest."""

    desired_keys = {
        f"{dataset_key}/{entry['path']}" for entry in manifest.get("files", [])
    }

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{dataset_key}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key not in desired_keys:
                    s3_client.delete_object(Bucket=bucket, Key=key)
    except AttributeError:
        pass

    for entry in manifest.get("files", []):
        relative_path = entry["path"]
        s3_client.upload_file(
            str(dataset_root / relative_path),
            bucket,
            f"{dataset_key}/{relative_path}",
            ExtraArgs={"StorageClass": "INTELLIGENT_TIERING"},
        )

    s3_client.put_object(
        Bucket=bucket,
        Key=dataset_key,
        Body=json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
        StorageClass="INTELLIGENT_TIERING",
    )


def process_batch_task(
    task: dict[str, Any],
    user_arguments: dict[str, str] | None = None,
    s3_client: Any | None = None,
    workdir_base: str | Path | None = None,
) -> str:
    """Convert a single S3 Batch Operations task."""

    _ = user_arguments
    s3_client = s3_client or boto3.client("s3")
    task_id = str(task["taskId"])
    bucket = _bucket_name_from_arn(str(task["s3BucketArn"]))
    legacy_key = str(task["s3Key"])
    version_id = task.get("s3VersionId")
    if version_id in {"", "null"}:
        version_id = None

    dataset_key = parquet_dataset_key_for_legacy_key(legacy_key)
    if workdir_base is None:
        workdir_base = Path(tempfile.gettempdir()) / "ifera-parquet-migration"
    workdir = Path(workdir_base) / _workdir_name_for_task(task_id)
    source_zip = workdir / "source.zip"
    dataset_root = workdir / Path(dataset_key).name

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        _download_task_object(
            s3_client,
            bucket,
            legacy_key,
            source_zip,
            version_id=str(version_id) if version_id is not None else None,
        )
        manifest = convert_legacy_zip_to_parquet_dataset(
            legacy_key,
            source_zip,
            dataset_root,
        )
        _upload_dataset_to_s3(s3_client, bucket, dataset_key, dataset_root, manifest)
    finally:
        if workdir.exists():
            shutil.rmtree(workdir)

    return f"Migrated {legacy_key} to {dataset_key}"


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Handle an S3 Batch Operations Lambda invocation."""

    invocation_schema_version = str(event.get("invocationSchemaVersion", "1.0"))
    invocation_id = str(event.get("invocationId", ""))
    user_arguments = event.get("job", {}).get("userArguments", {})
    s3_client = boto3.client("s3")

    results = []
    for task in event.get("tasks", []):
        task_id = str(task.get("taskId", "unknown-task"))
        try:
            result_string = process_batch_task(
                task,
                user_arguments=user_arguments,
                s3_client=s3_client,
            )
            result_code = "Succeeded"
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            result_code = "PermanentFailure"
            result_string = str(exc)

        results.append(
            {
                "taskId": task_id,
                "resultCode": result_code,
                "resultString": result_string[:1000],
            }
        )

    return {
        "invocationSchemaVersion": invocation_schema_version,
        "invocationId": invocation_id,
        "treatMissingKeysAs": "PermanentFailure",
        "results": results,
    }
