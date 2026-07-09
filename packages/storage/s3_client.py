"""Boto3 S3/MinIO client — the primary (local-model) lane's OWN store.

This is the only store the local diarization lane reads from or writes to;
it never touches Azure Blob (see `packages/storage/azure_blob.py` for the
Azure lane). All configuration comes from `packages/config/settings.py`
(S3_* in `.env`); nothing is hardcoded here.
"""

from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from packages.config.settings import get_settings


@lru_cache(maxsize=1)
def _client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,  # None => real AWS
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def ensure_bucket() -> None:
    """Create the bucket if it doesn't exist yet (MinIO starts out empty)."""
    client = _client()
    bucket = get_settings().s3_bucket
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def put_stream(fileobj: BinaryIO, key: str) -> str:
    """Stream an audio file into the bucket and return its object key."""
    ensure_bucket()
    _client().upload_fileobj(fileobj, get_settings().s3_bucket, key)
    return key


def open_stream(key: str) -> BinaryIO:
    """Open a readable stream for an object — used by the API audio proxy."""
    return _client().get_object(Bucket=get_settings().s3_bucket, Key=key)["Body"]


def download_to(key: str, path: Path) -> None:
    """Download an object to a local path — a worker scratch file for engines
    that need a real path on disk (deleted by the caller when done)."""
    _client().download_file(get_settings().s3_bucket, key, str(path))
