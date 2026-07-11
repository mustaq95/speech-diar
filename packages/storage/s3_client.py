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
from botocore.config import Config
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
        # Generous timeouts + retries: large files can stall mid-transfer on a
        # slow link, and the default (60s read timeout, no retries) trips on
        # them well before the caller (the audio proxy) gets a chance to resume.
        config=Config(connect_timeout=30, read_timeout=120, retries={"max_attempts": 3, "mode": "standard"}),
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


def open_stream(key: str, start: int = 0) -> BinaryIO:
    """Open a readable stream for an object — used by the API audio proxy.

    `start` resumes the stream from that byte offset (via an HTTP Range
    request) instead of from the beginning — used to recover from a
    mid-transfer read failure without re-sending already-delivered bytes.
    """
    kwargs = {"Bucket": get_settings().s3_bucket, "Key": key}
    if start:
        kwargs["Range"] = f"bytes={start}-"
    return _client().get_object(**kwargs)["Body"]


def head_object(key: str) -> int:
    """Total byte size of an object — lets the API audio proxy answer a
    browser's HTTP Range request with a correct Content-Range/Content-Length."""
    return _client().head_object(Bucket=get_settings().s3_bucket, Key=key)["ContentLength"]


def download_to(key: str, path: Path) -> None:
    """Download an object to a local path — a worker scratch file for engines
    that need a real path on disk (deleted by the caller when done)."""
    _client().download_file(get_settings().s3_bucket, key, str(path))
