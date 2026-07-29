"""Seed the local recording-API replica from tests/samples/.

Uploads every audio file in tests/samples/ into the `recordings-mock` MinIO
bucket, keyed by a deterministic (sessionId, agendaItemId) pair, and prints a
ready-to-paste curl for each so you can drop it straight into "Load Blob".

Run: uv run python -m tools.recording_api.seed
"""

from pathlib import Path

from packages.config.settings import get_settings
from tools.recording_api import store
from tools.recording_api.ids import ids_for

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "tests" / "samples"


def main() -> None:
    token = get_settings().recording_api_token
    store.ensure_bucket()

    files = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        print(f"No audio files found in {SAMPLES_DIR}")
        return

    auth = token or "<RECORDING_API_TOKEN unset — set it in .env>"
    print(f"Seeding {len(files)} recording(s) into bucket '{store.RECORDINGS_BUCKET}':\n")
    for path in files:
        session_id, agenda_item_id = ids_for(path.name)
        store.put(store.key_for(session_id, agenda_item_id), path.read_bytes())
        url = f"http://localhost:8215/v1/recording/{session_id}/{agenda_item_id}/stream"
        print(f"# {path.name}")
        print(f'curl -H "Authorization: Bearer {auth}" "{url}"\n')


if __name__ == "__main__":
    main()
