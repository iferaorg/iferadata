"""
Utilities for interacting with AWS S3.
"""

# pylint: disable=too-many-return-statements

import datetime
import json
import os
from typing import Any, List

# pylint: disable=protected-access

import boto3  # type: ignore
from tqdm import tqdm
from .config import BaseInstrumentConfig
from .enums import Source, extension_map, legacy_extension_map
from .decorators import singleton
from .settings import settings


@singleton
class S3ClientSingleton:
    """Singleton wrapper around boto3 S3 client with optional caching."""

    def __init__(self, cache: bool = True) -> None:
        self.client = boto3.client("s3")
        self.cache = cache

        if cache:
            self.last_modified: dict[str, datetime.datetime] = {}
            self.cached_prefixes: set[str] = set()
        else:
            self.last_modified = {}
            self.cached_prefixes = set()

    def _populate_cache(self, prefix: str) -> None:
        if prefix in self.cached_prefixes:
            return

        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                self.last_modified[obj["Key"]] = obj["LastModified"]

        self.cached_prefixes.add(prefix)


def make_s3_key(source: Source, instrument: BaseInstrumentConfig, zipfile: bool) -> str:
    """Build an S3 key for the instrument data file."""

    extension = legacy_extension_map[source] if zipfile else extension_map[source]
    return (
        f"{source.value}/"
        f"{instrument.type}/"
        f"{instrument.interval}/"
        f"{instrument.file_symbol}{extension}"
    )


def _key_prefix(key: str) -> str:
    """
    Extract the prefix from an S3 key.
    The prefix is everything up to the last '/' in the key.
    """
    if "/" in key:
        return key.rsplit("/", 1)[0]
    return ""


def _list_exact_s3_object(
    wrapper: S3ClientSingleton,
    key: str,
) -> dict[str, Any] | None:
    """Return a listed S3 object only when the key matches exactly."""

    try:
        response = wrapper.client.list_objects_v2(
            Bucket=settings.S3_BUCKET,
            Prefix=key,
            MaxKeys=1,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error listing objects in S3 bucket '{settings.S3_BUCKET}' with prefix '{key}'"
        ) from e

    for obj in response.get("Contents", []):
        if obj["Key"] != key:
            continue
        if wrapper.cache and "LastModified" in obj:
            wrapper.last_modified[key] = obj["LastModified"]
        return obj

    return None


def download_s3_file(
    key: str,
    target_path: str,
    version_id: str | None = None,
) -> None:
    """
    Download a file from S3 to the specified local target path with a progress bar.
    """
    s3_client = S3ClientSingleton().client

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    except Exception as e:
        raise OSError(
            f"Error creating directories for {os.path.dirname(target_path)}: {e}"
        ) from e

    try:
        # Get file size for progress bar
        head_args: dict[str, Any] = {"Bucket": settings.S3_BUCKET, "Key": key}
        if version_id is not None:
            head_args["VersionId"] = version_id
        response = s3_client.head_object(**head_args)
        file_size = response["ContentLength"]

        # Set up progress bar
        progress = tqdm(
            total=file_size, unit="B", unit_scale=True, desc=f"Downloading {key}"
        )

        def callback(bytes_transferred):
            progress.update(bytes_transferred)

        # Download with progress tracking
        extra_args = {"VersionId": version_id} if version_id is not None else None
        download_kwargs: dict[str, Any] = {"Callback": callback}
        if extra_args is not None:
            download_kwargs["ExtraArgs"] = extra_args
        s3_client.download_file(
            settings.S3_BUCKET,
            key,
            target_path,
            **download_kwargs,
        )
        progress.close()

    except Exception as e:
        raise RuntimeError(
            f"Error downloading file from S3 (bucket='{settings.S3_BUCKET}', key='{key}')"
        ) from e


def upload_s3_file(key: str, local_path: str) -> str:
    """
    Upload a file from the local directory to S3 with a progress bar.
    """
    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    try:
        # Get local file size for progress bar
        file_size = os.path.getsize(local_path)

        # Set up progress bar
        progress = tqdm(
            total=file_size, unit="B", unit_scale=True, desc=f"Uploading {key}"
        )

        def callback(bytes_transferred):
            progress.update(bytes_transferred)

        # Upload with progress tracking
        s3_client.upload_file(
            local_path,
            settings.S3_BUCKET,
            key,
            Callback=callback,
            ExtraArgs={"StorageClass": "INTELLIGENT_TIERING"},
        )
        progress.close()

        if wrapper.cache:
            wrapper._populate_cache(_key_prefix(key))
            wrapper.last_modified[key] = datetime.datetime.now(tz=datetime.timezone.utc)

    except Exception as e:
        raise RuntimeError(
            f"Error uploading file to S3 (bucket='{settings.S3_BUCKET}', key='{key}', "
            f"local_path='{local_path}')"
        ) from e

    return key


def put_s3_object_bytes(
    key: str,
    payload: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Write an in-memory object to S3."""

    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    try:
        s3_client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=payload,
            ContentType=content_type,
            StorageClass="INTELLIGENT_TIERING",
        )
        if wrapper.cache:
            wrapper._populate_cache(_key_prefix(key))
            wrapper.last_modified[key] = datetime.datetime.now(tz=datetime.timezone.utc)
    except Exception as e:
        raise RuntimeError(
            f"Error uploading object to S3 (bucket='{settings.S3_BUCKET}', key='{key}')"
        ) from e

    return key


def get_s3_object_bytes(key: str) -> bytes:
    """Read an object body from S3."""

    s3_client = S3ClientSingleton().client

    try:
        response = s3_client.get_object(Bucket=settings.S3_BUCKET, Key=key)
        body = response["Body"].read()
    except Exception as e:
        raise RuntimeError(
            f"Error reading object from S3 (bucket='{settings.S3_BUCKET}', key='{key}')"
        ) from e

    if not isinstance(body, bytes):
        raise TypeError(f"Expected bytes body for S3 key '{key}'")
    return body


def put_s3_json_object(key: str, payload: dict[str, Any]) -> str:
    """Serialize and upload a JSON object to S3."""

    return put_s3_object_bytes(
        key,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        content_type="application/json",
    )


def get_s3_json_object(key: str) -> dict[str, Any]:
    """Read and deserialize a JSON object from S3."""

    payload = json.loads(get_s3_object_bytes(key).decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object for S3 key '{key}'")
    return payload


def check_s3_file_exists(
    key: str,
) -> bool:  # pylint: disable=too-many-return-statements
    """
    Check if a file exists in the specified S3 bucket.
    """
    wrapper = S3ClientSingleton()

    if wrapper.cache:
        if key in wrapper.last_modified:
            return True

        if not key.endswith(".parquet"):
            wrapper._populate_cache(_key_prefix(key))
            return key in wrapper.last_modified

    return _list_exact_s3_object(wrapper, key) is not None


def get_s3_last_modified(  # pylint: disable=too-many-return-statements
    key: str,
) -> datetime.datetime | None:
    """
    Retrieve the last modified timestamp for an S3 object.
    Returns None if the object does not exist.
    """
    wrapper = S3ClientSingleton()

    if wrapper.cache:
        if key in wrapper.last_modified:
            return wrapper.last_modified[key]

        if not key.endswith(".parquet"):
            wrapper._populate_cache(_key_prefix(key))
            return wrapper.last_modified.get(key)

    response = _list_exact_s3_object(wrapper, key)
    if response is None:
        return None
    if "LastModified" not in response:
        raise RuntimeError(
            f"Missing LastModified metadata for s3://{settings.S3_BUCKET}/{key}"
        )
    return response["LastModified"]


def list_s3_objects(prefix: str, recursive: bool = True) -> List[str]:
    """
    List S3 object keys under the given prefix.
    """
    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    if not recursive:
        keys: list[str] = []
        paginator = s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=settings.S3_BUCKET,
            Prefix=prefix,
            Delimiter="/",
        ):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))

        return keys

    if wrapper.cache:
        wrapper._populate_cache(prefix)
        keys = [
            obj_key for obj_key in wrapper.last_modified if obj_key.startswith(prefix)
        ]
    else:
        keys = []
        paginator = s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            if "Contents" in page:
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])

    return keys


def delete_s3_file(key: str) -> None:
    """
    Delete a file from S3.
    """
    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    try:
        s3_client.delete_object(Bucket=settings.S3_BUCKET, Key=key)
        if wrapper.cache:
            wrapper.last_modified.pop(key, None)
    except Exception as e:
        raise RuntimeError(
            f"Error deleting file from S3 (bucket='{settings.S3_BUCKET}', key='{key}')"
        ) from e


def rename_s3_file(old_key: str, new_key: str) -> None:
    """
    Rename a file in S3 by copying it to a new key and deleting the old one.
    """
    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    try:
        s3_client.copy_object(
            Bucket=settings.S3_BUCKET,
            CopySource={"Bucket": settings.S3_BUCKET, "Key": old_key},
            Key=new_key,
        )
        s3_client.delete_object(Bucket=settings.S3_BUCKET, Key=old_key)

        if wrapper.cache:
            wrapper._populate_cache(_key_prefix(new_key))
            wrapper.last_modified[new_key] = datetime.datetime.now(
                tz=datetime.timezone.utc
            )
            wrapper.last_modified.pop(old_key, None)

    except Exception as e:
        raise RuntimeError(
            f"Error renaming file in S3 (bucket='{settings.S3_BUCKET}', "
            f"old_key='{old_key}', new_key='{new_key}')"
        ) from e
