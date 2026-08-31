"""Live check against the real cohere-transcribe container.

Skipped unless RUN_LIVE_TESTS=1 (see conftest's `pytest_collection_modifyitems`),
like the other live tests: it needs the container up and sends real audio.

Run with:
    RUN_LIVE_TESTS=1 uv run pytest -m live -k cohere -v

`test_cohere_transcribe_runner.py` stubs the HTTP boundary and so pins the shape
of the REQUEST. Nothing there can notice the container answering differently,
which is the whole point of this file. Two things here would otherwise ship green:

  * the response JSON losing `text` (the adapter would quietly return "" and
    every recording would score as total error, which reads as a bad engine);
  * the model gaining, or losing, the ability to detect language on short audio.

That second one is load-bearing, and it is about the `language` field rather than
about chunk size. This endpoint has NO auto-detect, and omitting the field emits
English -- transcribing English speech and translating Arabic into English. That
is why `cohere_transcribe_language` defaults to "ar", and the tests below pin
each half of it so nobody resets that setting to "" believing it means "detect".
"""

import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest

from apps.background_worker.transcription.cohere import adapter as cohere_adapter
from apps.background_worker.transcription.cohere import runner as cohere_runner
from packages.config.settings import get_settings

SAMPLES = Path(__file__).parent / "samples"
EN_SAMPLE = SAMPLES / "3-two-speakers-en.wav"
AR_SAMPLE = SAMPLES / "youtube_ar_32min_8spk.16k.wav"

#: A window of the Arabic sample that is known to be Arabic speech. Not an
#: arbitrary offset: some windows of this file transcribe poorly for reasons
#: unrelated to language, and these tests are about the `language` field only.
AR_WINDOW_START_SEC = 60
AR_WINDOW_LEN_SEC = 54


def _arabic_chars(text: str) -> int:
    return sum(1 for char in text if "؀" <= char <= "ۿ")


def _require_container() -> None:
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.cohere_transcribe_url}/health", timeout=3.0)
    except httpx.HTTPError:
        pytest.skip(f"cohere-transcribe is not up at {settings.cohere_transcribe_url}")
    if response.status_code != 200:
        pytest.skip("cohere-transcribe /health is not 200")


def _cut(source: Path, out: Path, start: float, length: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ss", str(start),
         "-t", str(length), "-ar", "16000", "-ac", "1", str(out)],
        check=True,
    )
    return out


@pytest.mark.live
def test_the_response_still_has_the_shape_the_adapter_reads() -> None:
    """`{"text": ..., "usage": ...}`, and `usage` counts audio seconds.

    Asserted on the RAW output rather than through `adapt`, because `adapt` is
    written to return "" for a missing key — it would pass a shape change."""
    _require_container()

    raw = cohere_runner.run(str(EN_SAMPLE))

    assert isinstance(raw, dict), f"expected a JSON object, got {type(raw).__name__}"
    assert "text" in raw, f"response lost its 'text' key: {sorted(raw)}"
    assert isinstance(raw["text"], str)
    # Not a hard requirement of the adapter, which drops it — but it is the only
    # per-call figure the engine reports about itself, so a change is worth
    # knowing about before something starts depending on it.
    assert raw.get("usage", {}).get("type") == "duration", raw.get("usage")

    text = cohere_adapter.adapt(raw)
    assert len(text.split()) > 50, f"a ~55s English clip returned {len(text.split())} words: {text!r}"


@pytest.mark.live
def test_it_is_deterministic() -> None:
    """`temperature=0` in the runner is what makes a re-run comparable. If the
    server starts sampling anyway, every timing comparison silently loses its
    baseline."""
    _require_container()
    with tempfile.TemporaryDirectory() as tmp:
        clip = _cut(EN_SAMPLE, Path(tmp) / "clip.wav", 0, 10)
        first = cohere_adapter.adapt(cohere_runner.run(str(clip)))
        second = cohere_adapter.adapt(cohere_runner.run(str(clip)))
    assert first == second, f"same audio, two transcripts:\n{first!r}\n{second!r}"


def _transcribe(path: Path, language: str | None) -> httpx.Response:
    """One raw call with explicit control of the `language` field.

    Deliberately not `runner.run()`: that reads `.env`, so every test here would
    inherit whatever this host has configured and none of them could pin what the
    field itself does. The request shape is otherwise the runner's.
    """
    settings = get_settings()
    data = {"model": "cohere-transcribe", "response_format": "json", "temperature": "0"}
    if language is not None:
        data["language"] = language
    with open(path, "rb") as fh:
        return httpx.post(
            f"{settings.cohere_transcribe_url}/v1/audio/transcriptions",
            files={"file": (path.name, fh, "audio/wav")},
            data=data,
            timeout=settings.cohere_transcribe_timeout_sec,
        )


@pytest.mark.live
def test_the_endpoint_offers_no_auto_detect() -> None:
    """The single most load-bearing fact about this engine.

    `cohere_transcribe_language` defaults to "ar" ONLY because there is no way to
    ask the model to detect the language itself. If that ever changes, this 400
    becomes a 200 and the default should be revisited -- so the assertion is on
    the rejection, not on some property of a transcript.
    """
    _require_container()
    with tempfile.TemporaryDirectory() as tmp:
        clip = _cut(EN_SAMPLE, Path(tmp) / "clip.wav", 0, 5)
        response = _transcribe(clip, "auto")

    assert response.status_code == 400, (
        "language=auto was accepted; this endpoint may have gained auto-detect, "
        f"which would change why cohere_transcribe_language defaults to 'ar' -- {response.text[:300]}"
    )
    body = response.text
    assert "Unsupported language" in body, body[:300]
    # The supported set is what makes "no auto-detect" a fact rather than a guess.
    assert "'ar'" in body and "'en'" in body, body[:300]


@pytest.mark.live
def test_forcing_arabic_is_never_worse_than_omitting_the_language() -> None:
    """Omitting `language` makes the model pick ONE language from the content,
    and what it picks depends on the audio:

      * mostly-Arabic conversation  -> Arabic (391 chars on this sample)
      * heavily code-switched speech -> ENGLISH, translating the Arabic away
        (0 chars on a real Mixed-50/50 read-aloud, at every duration to 45 s)

    The second case is this platform's normal output, which is why the default is
    "ar" and not "". It is not asserted directly: no committed sample reproduces
    it (both the Arabic sample and an English+Arabic concatenation come back
    Arabic), and the one recording that does is a person's voice, not a fixture.

    What IS stable, and is asserted, is the ordering: "ar" never returns less
    Arabic than omitting. If that inverts, the default is no longer justified.
    """
    _require_container()
    with tempfile.TemporaryDirectory() as tmp:
        window = _cut(AR_SAMPLE, Path(tmp) / "ar.wav", AR_WINDOW_START_SEC, AR_WINDOW_LEN_SEC)
        omitted = _transcribe(window, None)
        forced = _transcribe(window, "ar")
        assert omitted.status_code == 200, omitted.text[:300]
        assert forced.status_code == 200, forced.text[:300]
        omitted_arabic = _arabic_chars(omitted.json()["text"])
        forced_arabic = _arabic_chars(forced.json()["text"])

    print(f"\n[cohere] Arabic chars: omitted={omitted_arabic}, forced ar={forced_arabic}")
    assert forced_arabic >= omitted_arabic, (
        "omitting the language produced MORE Arabic than forcing it, which would "
        f"undercut the 'ar' default -- omitted={omitted_arabic}, forced={forced_arabic}"
    )


@pytest.mark.live
def test_forcing_arabic_transcribes_arabic_and_leaves_english_alone() -> None:
    """"ar" does not force everything into Arabic -- it code-switches.

    Both halves matter. Arabic must come back as Arabic (otherwise the default is
    not doing its job), and English must survive unconverted (otherwise "ar"
    would be trading one wrong language for another).
    """
    _require_container()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        window = _cut(AR_SAMPLE, tmpdir / "ar.wav", AR_WINDOW_START_SEC, AR_WINDOW_LEN_SEC)
        forced = _transcribe(window, "ar")
        assert forced.status_code == 200, forced.text[:300]
        arabic_text = forced.json()["text"].strip()

        english = _cut(EN_SAMPLE, tmpdir / "en.wav", 0, 30)
        english_forced = _transcribe(english, "ar")
        assert english_forced.status_code == 200, english_forced.text[:300]
        english_text = english_forced.json()["text"].strip()

    arabic = _arabic_chars(arabic_text)
    leaked = _arabic_chars(english_text)
    print(f"\n[cohere] forced ar: {arabic} Arabic chars on Arabic audio, "
          f"{leaked} leaked into 30s of English")
    assert arabic > 100, f"forcing 'ar' did not produce Arabic: {arabic_text[:200]!r}"
    # Whole-file English is clean under "ar" (the leak is a short-chunk effect,
    # measured at 6 of 18 3s chunks and deliberately not asserted here).
    assert leaked == 0, (
        f"forcing 'ar' leaked {leaked} Arabic characters into whole-file English "
        f"audio, which was clean when measured: {english_text[:200]!r}"
    )
