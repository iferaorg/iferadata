"""
Utilities for interacting with AWS S3.
"""
import os
import boto3
from tqdm import tqdm

def download_s3_file(bucket: str, key: str, target_path: str) -> None:
    """
    Download a file from S3 to the specified local target path with a progress bar.
    """
    try:
        s3 = boto3.client('s3')
    except Exception as e:
        raise RuntimeError("Error initializing S3 client") from e

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    except Exception as e:
        raise OSError(
            f"Error creating directories for {os.path.dirname(target_path)}: {e}"
        ) from e

    try:
        # Get file size for progress bar
        response = s3.head_object(Bucket=bucket, Key=key)
        file_size = response['ContentLength']

        # Set up progress bar
        progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Downloading {key}')

        def callback(bytes_transferred):
            progress.update(bytes_transferred)

        # Download with progress tracking
        s3.download_file(bucket, key, target_path, 
                        Callback=callback)
        progress.close()

    except Exception as e:
        raise RuntimeError(
            f"Error downloading file from S3 (bucket='{bucket}', key='{key}')"
        ) from e

def upload_s3_file(bucket: str, key: str, local_path: str) -> str:
    """
    Upload a file from the local directory to S3 with a progress bar.
    """
    try:
        s3 = boto3.client('s3')
    except Exception as e:
        raise RuntimeError("Error initializing S3 client") from e

    try:
        # Get local file size for progress bar
        file_size = os.path.getsize(local_path)
        
        # Set up progress bar
        progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Uploading {key}')

        def callback(bytes_transferred):
            progress.update(bytes_transferred)

        # Upload with progress tracking
        s3.upload_file(local_path, bucket, key,
                      Callback=callback)
        progress.close()

    except Exception as e:
        raise RuntimeError(
            f"Error uploading file to S3 (bucket='{bucket}', key='{key}', "
            f"local_path='{local_path}')"
        ) from e

    return key

def check_s3_file_exists(bucket_name: str, key: str) -> bool:
    """
    Check if a file exists in the specified S3 bucket.
    """
    try:
        s3_client = boto3.client("s3")
    except Exception as e:
        raise RuntimeError("Error initializing S3 client") from e

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=key,
            MaxKeys=1
        )
    except Exception as e:
        raise RuntimeError(
            f"Error listing objects in S3 bucket '{bucket_name}' with prefix '{key}'"
        ) from e

    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'] == key:
                return True
    return False

def get_s3_last_modified(bucket: str, key: str) -> float:
    """
    Retrieve the last modified timestamp for an S3 object.
    """
    try:
        s3 = boto3.client("s3")
        response = s3.head_object(Bucket=bucket, Key=key)
        return response["LastModified"].timestamp()
    except Exception as e:
        raise RuntimeError(
            f"Error retrieving S3 metadata for s3://{bucket}/{key}"
        ) from e
