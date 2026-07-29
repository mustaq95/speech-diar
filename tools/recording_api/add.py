"""Upload your own audio into the local recording-API replica.

Pushes any audio file (from any path, not just tests/samples/) into the
`recordings-mock` bucket and prints a ready-to-paste curl / Load Blob URL.

Run:
    uv run python -m tools.recording_api.add path/to/audio.wav
    uv run python -m tools.recording_api.add a.wav b.mp3           # several at once
    uv run python -m tools.recording_api.add a.wav --session my-sess --agenda my-agenda
"""

import argparse
import sys
from pathlib import Path

from packages.config.settings import get_settings
from tools.recording_api import store
from tools.recording_api.ids import ids_for


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload audio into the recording-API replica blob.")
    parser.add_argument("paths", nargs="+", type=Path, help="Audio file(s) to upload")
    parser.add_argument("--session", help="Custom sessionId (single file only; default: derived from filename)")
    parser.add_argument("--agenda", help="Custom agendaItemId (single file only; default: derived from filename)")
    args = parser.parse_args()

    if (args.session or args.agenda) and len(args.paths) != 1:
        parser.error("--session/--agenda can only be used with a single file")

    for path in args.paths:
        if not path.is_file():
            parser.error(f"not a file: {path}")

    token = get_settings().recording_api_token
    auth = token or "<RECORDING_API_TOKEN unset — set it in .env>"
    store.ensure_bucket()

    for path in args.paths:
        if args.session and args.agenda:
            session_id, agenda_item_id = args.session, args.agenda
        else:
            session_id, agenda_item_id = ids_for(path.name)
        store.put(store.key_for(session_id, agenda_item_id), path.read_bytes())
        url = f"http://localhost:8215/v1/recording/{session_id}/{agenda_item_id}/stream"
        print(f"# {path.name} -> {store.RECORDINGS_BUCKET}/{store.key_for(session_id, agenda_item_id)}")
        print(f'curl -H "Authorization: Bearer {auth}" "{url}"\n')


if __name__ == "__main__":
    sys.exit(main())
