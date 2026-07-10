"""Shared execution + turn-grouping logic for the two Parakeet-Sortformer NIM
engines (streaming on :50051, offline batch on :50052 by default).

Both containers run the same NIM image (`diarizer=sortformer, vad=silero`);
only the profile (`mode=str` vs `mode=ofl`) and the RPC used (streaming vs
unary) differ, so that plumbing lives here once. Diarization speaker tags are
only exposed on Riva's gRPC ASR API via word-level `speaker_tag` — the NIM's
plain HTTP `/v1/audio/transcriptions` route returns flat text with no speaker
information (confirmed by probing both endpoints against a live sample).

Native shape produced by both runners (returned untouched, per the
runner/adapter split every model in this platform follows):
  {"audio_duration_sec": float | None,
   "words": [{"speaker_tag": int, "start_ms": int, "end_ms": int, "word": str}]}

`speaker_tag` is a proto3 field defaulting to 0; a word with no diarization
label is genuinely speaker 0, not "unknown" — there is no missing case to
handle.
"""

from contextlib import suppress
from typing import Any
from wave import open as wave_open

from packages.config.settings import get_settings

NimRawOutput = dict[str, Any]

# ~300ms per chunk at 16kHz — matches NVIDIA's own streaming client examples.
_STREAM_CHUNK_N_FRAMES = 4800


def wav_duration_sec(audio_path: str) -> float | None:
    with suppress(Exception):
        with wave_open(audio_path, "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


def _build_config(sample_rate_hertz: int, num_channels: int) -> Any:
    import riva.client

    settings = get_settings()
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate_hertz,
        audio_channel_count=num_channels,
        language_code=settings.nim_language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
    )
    riva.client.add_speaker_diarization_to_config(
        config,
        diarization_max_speakers=settings.nim_max_speakers,
        diarization_enable=True,
    )
    return config


def _words_from_alternative_results(results: Any) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for result in results:
        if not result.alternatives:
            continue
        for word in result.alternatives[0].words:
            words.append(
                {
                    "speaker_tag": word.speaker_tag,
                    "start_ms": word.start_time,
                    "end_ms": word.end_time,
                    "word": word.word,
                }
            )
    return words


def recognize_offline(audio_path: str, grpc_endpoint: str) -> NimRawOutput:
    """Unary `Recognize` against an offline-batch NIM container."""
    import riva.client

    settings = get_settings()
    wav_params = riva.client.get_wav_file_parameters(audio_path)
    config = _build_config(wav_params["framerate"], wav_params["nchannels"])

    auth = riva.client.Auth(uri=grpc_endpoint, use_ssl=False)
    asr_service = riva.client.ASRService(auth)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    future = asr_service.offline_recognize(audio_bytes, config, future=True)
    response = future.result(timeout=settings.nim_grpc_timeout_sec)

    return {
        "audio_duration_sec": wav_duration_sec(audio_path),
        "words": _words_from_alternative_results(response.results),
    }


def recognize_streaming(audio_path: str, grpc_endpoint: str) -> NimRawOutput:
    """`StreamingRecognize` against a streaming NIM container, fed the whole
    file chunk-by-chunk (this platform diarizes a completed upload, not a
    live mic — streaming here means the RPC shape, not real-time playback)."""
    import time

    import riva.client

    settings = get_settings()
    wav_params = riva.client.get_wav_file_parameters(audio_path)
    config = _build_config(wav_params["framerate"], wav_params["nchannels"])
    streaming_config = riva.client.StreamingRecognitionConfig(config=config, interim_results=False)

    auth = riva.client.Auth(uri=grpc_endpoint, use_ssl=False)
    asr_service = riva.client.ASRService(auth)

    words: list[dict[str, Any]] = []
    deadline = time.monotonic() + settings.nim_grpc_timeout_sec
    with riva.client.AudioChunkFileIterator(audio_path, chunk_n_frames=_STREAM_CHUNK_N_FRAMES) as audio_chunk_iterator:
        for response in asr_service.streaming_response_generator(audio_chunk_iterator, streaming_config):
            if time.monotonic() > deadline:
                raise RuntimeError(f"NIM streaming recognize timed out after {settings.nim_grpc_timeout_sec}s")
            words.extend(_words_from_alternative_results(r for r in response.results if r.is_final))

    return {
        "audio_duration_sec": wav_duration_sec(audio_path),
        "words": words,
    }


def group_words_into_turns(words: list[dict[str, Any]]) -> list[tuple[int, float, float]]:
    """Group consecutive same-`speaker_tag` words into one turn each.

    Never merges across a speaker change, and never re-merges two runs of the
    same speaker separated by a different speaker in between — this mirrors
    exactly what the model itself reported, per the project's "never
    coalesce" rule. Returns (speaker_tag, start_sec, end_sec) triples, sorted
    by start time.
    """
    ordered = sorted(words, key=lambda w: w["start_ms"])
    turns: list[tuple[int, float, float]] = []
    run_tag: int | None = None
    run_start_ms = 0
    run_end_ms = 0

    for word in ordered:
        if word["speaker_tag"] != run_tag:
            if run_tag is not None:
                turns.append((run_tag, run_start_ms / 1000, run_end_ms / 1000))
            run_tag = word["speaker_tag"]
            run_start_ms = word["start_ms"]
            run_end_ms = word["end_ms"]
        else:
            run_end_ms = max(run_end_ms, word["end_ms"])

    if run_tag is not None:
        turns.append((run_tag, run_start_ms / 1000, run_end_ms / 1000))

    return turns
