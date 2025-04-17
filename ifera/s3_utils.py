"""
Utilities for interacting with AWS S3.
"""

import os
import datetime
from typing import List

import boto3  # type: ignore
import botocore.exceptions
from tqdm import tqdm
from .config import BaseInstrumentConfig
from .enums import Source
from .decorators import singleton
from .settings import settings


@singleton
class S3ClientSingleton:
    def __init__(self, cache: bool = True):
        self.client = boto3.client("s3")
        self.cache = cache

        if cache:
            self.last_modified = self.get_all_modified_dates()

    def get_all_modified_dates(self) -> dict[str, datetime.datetime]:
        paginator = self.client.get_paginator("list_objects_v2")
        modified_dates: dict[str, datetime.datetime] = {}

        for page in paginator.paginate(Bucket=settings.S3_BUCKET):
            for obj in page.get("Contents", []):
                modified_dates[obj["Key"]] = obj["LastModified"]

        return modified_dates


def make_s3_key(source: Source, instrument: BaseInstrumentConfig, zipfile: bool) -> str:
    """Build an S3 key for the instrument data file."""
    extension = ".zip" if zipfile else ".csv"
    return f"{source.value}/{instrument.type}/{instrument.interval}/{instrument.file_symbol}{extension}"


def download_s3_file(key: str, target_path: str) -> None:
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
        response = s3_client.head_object(Bucket=settings.S3_BUCKET, Key=key)
        file_size = response["ContentLength"]

        # Set up progress bar
        progress = tqdm(
            total=file_size, unit="B", unit_scale=True, desc=f"Downloading {key}"
        )

        def callback(bytes_transferred):
            progress.update(bytes_transferred)

        # Download with progress tracking
        s3_client.download_file(settings.S3_BUCKET, key, target_path, Callback=callback)
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
            wrapper.last_modified[key] = datetime.datetime.now(tz=datetime.timezone.utc)

    except Exception as e:
        raise RuntimeError(
            f"Error uploading file to S3 (bucket='{settings.S3_BUCKET}', key='{key}', "
            f"local_path='{local_path}')"
        ) from e

    return key


def check_s3_file_exists(key: str) -> bool:
    """
    Check if a file exists in the specified S3 bucket.
    """
    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    if wrapper.cache:
        if key in wrapper.last_modified:
            return True

    try:
        response = s3_client.list_objects_v2(
            Bucket=settings.S3_BUCKET, Prefix=key, MaxKeys=1
        )
    except Exception as e:
        raise RuntimeError(
            f"Error listing objects in S3 bucket '{settings.S3_BUCKET}' with prefix '{key}'"
        ) from e

    if "Contents" in response:
        for obj in response["Contents"]:
            if obj["Key"] == key:
                return True
    return False


def get_s3_last_modified(key: str) -> datetime.datetime | None:
    """
    Retrieve the last modified timestamp for an S3 object.
    Returns None if the object does not exist.
    """
    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    if wrapper.cache:
        if key in wrapper.last_modified:
            return wrapper.last_modified[key]
        else:
            return None

    try:
        response = s3_client.head_object(Bucket=settings.S3_BUCKET, Key=key)
        return response["LastModified"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # The object does not exist
            return None
        else:
            raise RuntimeError(
                f"Error retrieving S3 metadata for s3://{settings.S3_BUCKET}/{key}"
            ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error retrieving S3 metadata for s3://{settings.S3_BUCKET}/{key}"
        ) from e


def list_s3_objects(prefix: str) -> List[str]:
    """
    List S3 object keys under the given prefix.
    """
    wrapper = S3ClientSingleton()
    s3_client = wrapper.client

    if wrapper.cache:
        keys = [key for key in wrapper.last_modified.keys() if key.startswith(prefix)]
    else:
        keys = []
        paginator = s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            if "Contents" in page:
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])

    return keys
