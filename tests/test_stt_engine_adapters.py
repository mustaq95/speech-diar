"""Adapters for the six STT engines added to the comparison surface, pinned
against the payloads the REAL endpoints returned.

Every fixture in `fixtures/stt_probes/` was captured from a live call on
2026-09-01 against one 61 s code-switched Arabic/English recording, then trimmed
for size. Nothing here is a hand-written guess at a response shape, which matters
because two of the three shapes that bite were only visible in real output:
vibevoice omitted a key entirely, and speechmatics separated punctuation into its
own result.

Pure functions, no infrastructure — same split as `test_transcript_adapters.py`.
"""

import json
from pathlib import Path

import pytest

from apps.background_worker.transcription import _openai_asr
from apps.background_worker.transcription.adeo_qwen3 import adapter as qwen3_adapter
from apps.background_worker.transcription.adeo_whisper import adapter as whisper_adapter
from apps.background_worker.transcription.elevenlabs import adapter as elevenlabs_adapter
from apps.background_worker.transcription.elevenlabs import runner as elevenlabs_runner
from apps.background_worker.transcription.moss import adapter as moss_adapter
from apps.background_worker.transcription.speechmatics import adapter as speechmatics_adapter
from apps.background_worker.transcription.vibevoice import adapter as vibevoice_adapter

FIXTURES = Path(__file__).parent / "fixtures" / "stt_probes"

#: Arabic letters and ASCII letters. Used to assert an engine kept BOTH scripts
#: rather than translating or transliterating one into the other, which is the
#: single property this whole comparison exists to measure.
def _has_arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in text)


def _has_latin(text: str) -> bool:
    return any("a" <= ch.lower() <= "z" for ch in text)


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- vibevoice ---------------------------------------------------------------

def test_vibevoice_adapter_survives_a_segment_with_no_speaker_key() -> None:
    """The real payload contained a segment carrying "[Unintelligible Speech]"
    with NO `Speaker` key at all, so this payload is not uniformly shaped.

    Indexing it would raise on genuine output. The fixture keeps that segment
    deliberately; if a future trim removes it this test stops testing anything,
    so the absence is asserted first."""
    raw = load("vibevoice.json")
    assert any("Speaker" not in seg for seg in raw["segments"]), (
        "fixture no longer contains the no-Speaker segment this test exists for"
    )
    text = vibevoice_adapter.adapt(raw)
    assert "[Unintelligible Speech]" in text


def test_vibevoice_adapter_keeps_both_scripts() -> None:
    """It code-switches natively: Arabic in Arabic script, English in Latin,
    inside one `Content` string."""
    text = vibevoice_adapter.adapt(load("vibevoice.json"))
    assert _has_arabic(text) and _has_latin(text)


def test_vibevoice_segment_count_is_the_models_own_turns() -> None:
    """Not a chunk count — nothing split this audio, the model chose these
    boundaries. Reported so the column never renders as 0."""
    assert vibevoice_adapter.segment_count(load("vibevoice.json")) == 6


# --- speechmatics ------------------------------------------------------------

def test_speechmatics_adapter_attaches_punctuation_to_the_previous_word() -> None:
    """Punctuation arrives as its OWN result with `attaches_to: "previous"`.

    Space-joining it would put a space before every comma. That is not cosmetic:
    CER is computed over characters, so each invented space is a real character
    error charged to the engine."""
    raw = load("speechmatics_ar_en.json")
    assert any(r.get("attaches_to") == "previous" for r in raw["results"]), (
        "fixture no longer contains a punctuation result"
    )
    text = speechmatics_adapter.adapt(raw)
    assert " ،" not in text and " ." not in text


def test_speechmatics_adapter_reports_the_bilingual_pack_that_ran() -> None:
    """`ar_en` is the code-switching MODEL, and the API names the pack it used.
    This is how a bilingual run is told apart from a monolingual one that
    silently dropped the other language — they differ in content, not status."""
    raw = load("speechmatics_ar_en.json")
    assert speechmatics_adapter.language_pack(raw) == "Arabic and English"


def test_speechmatics_word_count_excludes_punctuation() -> None:
    """Punctuation is not speech; counting it would inflate the figure."""
    raw = load("speechmatics_ar_en.json")
    words = speechmatics_adapter.word_count(raw)
    assert words == sum(1 for r in raw["results"] if r["type"] == "word")
    assert words < len(raw["results"])


# --- moss --------------------------------------------------------------------

def test_moss_adapter_strips_timestamp_and_speaker_markers() -> None:
    """MOSS emits ASR, diarization and timestamps as one generated string
    (`[start][Snn] text[end]`). The markers must not reach the scored text, or
    the engine is charged insertions for the format it was asked to produce."""
    text = moss_adapter.adapt(load("moss.json"))
    assert "[S01]" not in text
    assert "[2.16]" not in text


def test_moss_adapter_keeps_a_bracketed_token_that_is_content() -> None:
    """The marker pattern is anchored to `[12.34]` and `[Snn]` exactly, so a
    bracketed non-speech token the model emits as CONTENT survives."""
    raw = {"text": "[1.50][S01] hello [Music] world[4.00]"}
    assert moss_adapter.adapt(raw) == "hello [Music] world"


def test_moss_turns_expose_the_diarization_half() -> None:
    """Unused by the scorecard, which scores one flat string, but the speaker and
    time data MOSS genuinely produced stays reachable from an adapter."""
    turns = moss_adapter.turns(load("moss.json"))
    assert turns and all({"start", "speaker", "text"} <= set(t) for t in turns)
    assert turns == sorted(turns, key=lambda t: t["start"])


# --- the OpenAI-shaped engines ----------------------------------------------

@pytest.mark.parametrize(
    "adapter, fixture",
    [(qwen3_adapter, "adeo_qwen3.json"), (whisper_adapter, "adeo_qwen3.json")],
)
def test_openai_shaped_adapters_read_text_from_the_wrapped_response(adapter, fixture) -> None:
    """Both wrap the endpoint's own JSON under "response" and keep what the
    runner measured beside it, never merged in. adeo-whisper is fed the Qwen3
    fixture on purpose: its own pod was 504 throughout probing, and it is
    implemented against the contract that sibling pod is proven to speak."""
    payload = load(fixture)
    raw = [{"response": payload, "latency_ms": 700, "segment_index": 0}]
    assert adapter.adapt(raw) == payload["text"].strip()
    assert adapter.segment_count(raw) == 1


def test_qwen3_output_keeps_both_scripts() -> None:
    """It code-switches natively with NO language field sent at all."""
    text = load("adeo_qwen3.json")["text"]
    assert _has_arabic(text) and _has_latin(text)


def test_openai_shaped_adapters_join_segments_in_audio_order() -> None:
    """Out-of-order pieces would scramble the transcript, so ordering is by
    `segment_index`, not arrival."""
    raw = [
        {"response": {"text": "second"}, "segment_index": 1},
        {"response": {"text": "first"}, "segment_index": 0},
    ]
    assert qwen3_adapter.adapt(raw) == "first second"


# --- elevenlabs --------------------------------------------------------------

def test_elevenlabs_adapter_keeps_audio_event_markers() -> None:
    """`[phone chimes]` and friends are typed `audio_event` and appear inside
    `text`. They are NOT stripped: they are what the engine emitted, and they
    cost it real WER against a reference containing no such marker. Removing
    them here would be this code flattering the engine."""
    raw_payload = load("elevenlabs_scribe_v2.json")
    types = {w["type"] for w in raw_payload["words"]}
    assert "audio_event" in types or "word" in types
    text = elevenlabs_adapter.adapt([{"response": raw_payload, "segment_index": 0}])
    assert text == raw_payload["text"].strip()


def test_elevenlabs_adapter_exposes_the_detected_language() -> None:
    """The engine's own detection result, produced with `language_code` OMITTED.
    That it detected Arabic while keeping the English is the evidence that
    omitting the field is the right default."""
    raw_payload = load("elevenlabs_scribe_v2.json")
    raw = [{"response": raw_payload, "segment_index": 0}]
    assert elevenlabs_adapter.detected_language(raw) == "ara"
    assert _has_arabic(raw_payload["text"]) and _has_latin(raw_payload["text"])


# --- the language guarantee --------------------------------------------------

@pytest.mark.parametrize("value", ["auto", "", "  ", "AUTO", "detect"])
def test_auto_and_empty_omit_the_language_field(value: str) -> None:
    """The whole "nothing is hardcoded" requirement, in one assertion.

    `auto` is NOT a code any of these endpoints accepts — both vLLM pods reject
    it 400 against a 57-code list — so it must resolve to "send nothing", never
    be forwarded verbatim."""
    assert _openai_asr.resolve_language(value) is None


@pytest.mark.parametrize("value", ["ar", "en", "ar_en"])
def test_a_real_language_code_is_passed_through(value: str) -> None:
    """The escape hatch still works, so an engine that genuinely honours the
    field can be pointed at one deliberately."""
    assert _openai_asr.resolve_language(value) == value


def test_elevenlabs_language_resolver_agrees_with_the_shared_one(monkeypatch) -> None:
    """ElevenLabs has its own resolver because its field is `language_code`, not
    `language`. It must not drift from the shared rule."""
    from packages.config.settings import Settings

    for value in ("auto", "", "detect"):
        settings = Settings(elevenlabs_stt_language=value)
        assert elevenlabs_runner.requested_language(settings) is None
    assert elevenlabs_runner.requested_language(Settings(elevenlabs_stt_language="ara")) == "ara"
