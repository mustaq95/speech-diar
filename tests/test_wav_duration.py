"""Regression tests for `_wav_duration_file` — some encoders (streamed/live
recordings in particular) write a placeholder or otherwise inaccurate
`data` chunk size in the WAV header. Trusting it blindly can report a
duration many times longer than the real audio (observed: a real ~5s
recording reported as ~37 hours)."""

import struct
import tempfile

from apps.backend_api.routers.upload import _wav_duration_file
from tests.conftest import make_wav_bytes


def _wav_duration_sec(wav_bytes: bytes) -> float | None:
    """Write bytes to a temp WAV and read its duration from disk, the way the
    upload handler does."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(wav_bytes)
        f.flush()
        return _wav_duration_file(f.name)


def _corrupt_data_chunk_size(wav_bytes: bytes, bogus_size: int) -> bytes:
    """Overwrite the `data` sub-chunk's declared size field in place,
    leaving the actual PCM bytes (and every other header field) untouched —
    simulates an encoder that never patched the header with the real size."""
    marker = wav_bytes.find(b"data")
    assert marker != -1
    size_offset = marker + 4
    patched = bytearray(wav_bytes)
    patched[size_offset : size_offset + 4] = struct.pack("<I", bogus_size)
    return bytes(patched)


def test_duration_correct_for_well_formed_wav() -> None:
    wav_bytes = make_wav_bytes(duration_sec=5.0, framerate=16000)
    assert _wav_duration_sec(wav_bytes) == 5.0


def test_duration_capped_when_header_declares_absurdly_large_data_size() -> None:
    """The exact bug pattern: a 5-second recording whose header claims a
    ~37-hour-long data chunk. Duration must be derived from the actual
    uploaded bytes, not the bogus header field."""
    wav_bytes = make_wav_bytes(duration_sec=5.0, framerate=16000)
    corrupted = _corrupt_data_chunk_size(wav_bytes, bogus_size=1_000_000_000)

    duration = _wav_duration_sec(corrupted)

    assert duration is not None
    assert duration < 10.0  # nowhere near the bogus ~37-hour implied duration
    assert duration == 5.0  # exactly recoverable here: real bytes are all still present


def test_duration_still_correct_when_header_size_matches_reality() -> None:
    wav_bytes = make_wav_bytes(duration_sec=2.5, framerate=8000)
    assert _wav_duration_sec(wav_bytes) == 2.5


def test_duration_none_for_garbage_input() -> None:
    assert _wav_duration_sec(b"not a wav file at all") is None
