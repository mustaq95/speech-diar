"""Live check against the real TryHamsa STT deployment.

Skipped unless RUN_LIVE_TESTS=1 (see conftest's `pytest_collection_modifyitems`),
exactly like `test_live_azure_batch.py`: it needs real credentials, sends real
audio to a real endpoint, and costs real time.

Run with:
    RUN_LIVE_TESTS=1 uv run pytest -m live -k hamsa -v

Worth running whenever the handshake options or the streaming loop change. The
protocol is the part no unit test can cover: everything else about this engine
is stubbed in `test_transcript_pipeline.py`.
"""

import time
from pathlib import Path

import pytest

from apps.background_worker.transcription.hamsa import adapter as hamsa_adapter
from apps.background_worker.transcription.hamsa import runner as hamsa_runner
from packages.config.settings import get_settings

SAMPLE = Path(__file__).parent / "samples" / "3-two-speakers-en.wav"


@pytest.mark.live
def test_hamsa_transcribes_a_real_clip() -> None:
    settings = get_settings()
    if not (settings.hamsa_ws_endpoint and settings.hamsa_stt_key):
        pytest.skip("HAMSA_STT_WS_URL / HAMSA_STT_KEY not configured in .env")

    started = time.monotonic()
    raw = hamsa_runner.run(str(SAMPLE))
    elapsed = time.monotonic() - started
    text = hamsa_adapter.adapt(raw)

    print(f"\n[hamsa] {elapsed:.1f}s for a ~55s clip -> {len(text.split())} words")
    print(f"[hamsa] {text[:300]}")

    assert raw, "the server sent no messages at all — handshake or streaming is broken"
    assert text.strip(), "connected and streamed, but no transcription came back"
    # The pacing contract: this engine streams audio in real time, so a clip
    # cannot come back faster than roughly half its duration. A result that is
    # dramatically faster means the audio was not actually streamed through.
    assert elapsed > 5, f"suspiciously fast ({elapsed:.1f}s) — was the whole clip sent?"
