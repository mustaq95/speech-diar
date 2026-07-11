"""Unit tests for `_mono_audio_path` in apps/background_worker/models/_nim_shared.py.

Riva's ASR service rejects multi-channel audio outright ("Audio channel count
greater than 1 is currently not supported"), which is exactly what Parakeet
streaming/offline hit on a real stereo recording. These tests exercise the
downmix helper directly -- no gRPC/riva calls needed.
"""

import os
import tempfile
import wave

from apps.background_worker.models._nim_shared import _mono_audio_path


def _write_wav(path: str, *, n_channels: int, framerate: int = 16000, duration_sec: float = 1.0) -> None:
    n_frames = int(duration_sec * framerate)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(n_channels)
        wav.setsampwidth(2)
        wav.setframerate(framerate)
        wav.writeframes(b"\x00\x00" * n_frames * n_channels)


def test_mono_audio_path_passes_through_unchanged_when_already_mono() -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        _write_wav(f.name, n_channels=1)
        with _mono_audio_path(f.name) as mono_path:
            assert mono_path == f.name


def test_mono_audio_path_downmixes_stereo_to_mono() -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        _write_wav(f.name, n_channels=2, framerate=44100, duration_sec=2.0)
        with _mono_audio_path(f.name) as mono_path:
            assert mono_path != f.name
            with wave.open(mono_path, "rb") as mono_wav:
                assert mono_wav.getnchannels() == 1
                assert mono_wav.getframerate() == 44100  # unchanged -- only channel count is fixed
                assert mono_wav.getnframes() == 44100 * 2


def test_mono_audio_path_removes_temp_file_after_use() -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        _write_wav(f.name, n_channels=2)
        with _mono_audio_path(f.name) as mono_path:
            captured = mono_path
        assert not os.path.exists(captured)
