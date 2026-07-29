"""Local stand-in for the production recording API (ADEO).

Speaks the same contract on :8215 so the eval platform's "Load Blob" path can
be exercised without ADEO: a Bearer-guarded stream endpoint backed by the
`recordings-mock` MinIO bucket the seed script populates. Switching to
production is then just a URL change on the eval side.

Run: uv run uvicorn tools.recording_api.server:app --port 8215
"""

from fastapi import FastAPI, Header, HTTPException, Response

from packages.config.settings import get_settings
from tools.recording_api import store

app = FastAPI(title="Recording API replica")


def _check_auth(authorization: str | None) -> None:
    expected = get_settings().recording_api_token
    if not expected:
        # Never silently open: an unset token is a misconfiguration, not "allow all".
        raise HTTPException(status_code=401, detail="RECORDING_API_TOKEN is not configured")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/v1/recording/{session_id}/{agenda_item_id}/stream")
def stream(session_id: str, agenda_item_id: str, authorization: str | None = Header(default=None)) -> Response:
    _check_auth(authorization)
    data = store.get(store.key_for(session_id, agenda_item_id))
    if data is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    # Raw bytes, as production streams them; the eval backend transcodes regardless.
    return Response(content=data, media_type="application/octet-stream")
