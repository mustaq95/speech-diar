"""MinIO access for the local recording-API replica.

Deliberately separate from `packages/storage/s3_client.py` (which is hard-wired
to the eval platform's own bucket): the replica stands in for ADEO's *separate*
S3, so it uses its own bucket. Endpoint and credentials are still read from the
shared settings, so pointing MinIO somewhere else stays a `.env`-only change.
"""

from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from packages.config.settings import get_settings

RECORDINGS_BUCKET = "recordings-mock"


@lru_cache(maxsize=1)
def _client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,  # None => real AWS
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def key_for(session_id: str, agenda_item_id: str) -> str:
    return f"{session_id}/{agenda_item_id}"


def ensure_bucket() -> None:
    client = _client()
    try:
        client.head_bucket(Bucket=RECORDINGS_BUCKET)
    except ClientError:
        client.create_bucket(Bucket=RECORDINGS_BUCKET)


def put(key: str, data: bytes) -> None:
    ensure_bucket()
    _client().put_object(Bucket=RECORDINGS_BUCKET, Key=key, Body=data)


def get(key: str) -> bytes | None:
    """Return the object's bytes, or None if it doesn't exist."""
    try:
        resp = _client().get_object(Bucket=RECORDINGS_BUCKET, Key=key)
    except ClientError:
        return None
    return resp["Body"].read()
