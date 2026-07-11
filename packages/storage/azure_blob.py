"""Azure Blob Storage — the Azure lane's OWN store.

Batch transcription fetches audio over http(s). When the Azure model is
toggled on, the upload is streamed directly into this Blob container (never
into MinIO) and Azure batch fetches it via a read-only SAS URL generated
here. This module never reads from or writes to MinIO — see
`packages/storage/s3_client.py` for the primary (local-model) lane.

All configuration comes from `packages/config/settings.py` (AZURE_STORAGE_*
in `.env`); nothing is hardcoded here.
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import BinaryIO

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

from packages.config.settings import get_settings


def storage_configured() -> bool:
    settings = get_settings()
    return bool(settings.azure_storage_account_name and settings.azure_storage_account_key)


@lru_cache(maxsize=1)
def _service() -> BlobServiceClient:
    settings = get_settings()
    if not settings.azure_storage_account_name or not settings.azure_storage_account_key:
        raise RuntimeError("Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY in .env")
    return BlobServiceClient(
        f"https://{settings.azure_storage_account_name}.blob.core.windows.net",
        credential=settings.azure_storage_account_key,
    )


def _container():
    client = _service()
    container = client.get_container_client(get_settings().azure_storage_container_name)
    try:
        container.create_container()
    except Exception as exc:
        if getattr(exc, "error_code", None) != "ContainerAlreadyExists":
            raise
    return container


def blob_exists(key: str) -> bool:
    """True if this key is already staged — lets the caller skip a redundant upload."""
    return _container().get_blob_client(key).exists()


def put_stream(fileobj: BinaryIO, key: str) -> str:
    """Stream an audio file into the Azure lane's container and return its blob key."""
    blob = _container().get_blob_client(key)
    blob.upload_blob(fileobj, overwrite=True)
    return key


def blob_url(key: str) -> str:
    """Stable (non-SAS) reference URL for a blob — for display/debugging only;
    fetching audio always goes through `read_sas_url`."""
    return _service().get_blob_client(get_settings().azure_storage_container_name, key).url


def read_sas_url(blob_key: str) -> str:
    """Time-limited read URL — what Azure batch fetches the audio from."""
    settings = get_settings()
    client = _service()
    blob = client.get_blob_client(settings.azure_storage_container_name, blob_key)
    sas = generate_blob_sas(
        account_name=client.account_name,
        container_name=settings.azure_storage_container_name,
        blob_name=blob_key,
        account_key=settings.azure_storage_account_key,
        permission=BlobSasPermissions(read=True),
        start=datetime.now(timezone.utc) - timedelta(minutes=5),
        expiry=datetime.now(timezone.utc) + timedelta(hours=settings.azure_storage_sas_read_expiry_hours),
    )
    return f"{blob.url}?{sas}"


def open_stream(blob_key: str, start: int = 0):
    """Open a readable stream for a blob — used by the API audio proxy.

    Returns a `StorageStreamDownloader`; iterate `.chunks()` to stream it.
    `start` resumes the download from that byte offset instead of from the
    beginning — used to recover from a mid-transfer read failure without
    re-sending already-delivered bytes.
    """
    settings = get_settings()
    client = _service()
    blob = client.get_blob_client(settings.azure_storage_container_name, blob_key)
    return blob.download_blob(offset=start)


def blob_size(blob_key: str) -> int:
    """Total byte size of a blob — lets the API audio proxy answer a
    browser's HTTP Range request with a correct Content-Range/Content-Length."""
    settings = get_settings()
    client = _service()
    blob = client.get_blob_client(settings.azure_storage_container_name, blob_key)
    return blob.get_blob_properties().size
