"""Consistency checks across the hand-maintained TTS tables.

The same shape as `test_model_registration.py`: adding an engine touches
`TTS_ENGINES`, `TTS_DELIVERY` and `COMPARISON_TTS_IDS`, and each one forgotten
fails quietly at runtime (an engine that never appears in the UI, or a figure
rendered with no transport label). These turn every mismatch into a failure.
"""

import pytest

from apps.background_worker.tts import (
    COMPARISON_TTS_IDS,
    TTS_DELIVERY,
    TTS_ENGINES,
    delivery_for,
    engine_for,
)
from packages.config.settings import get_settings

#: The wire contract's TtsDelivery literal. Kept here as a plain set so a new
#: value has to be added deliberately in both places.
VALID_DELIVERIES = {"stream", "single"}


def test_registry_and_delivery_map_list_the_same_engines() -> None:
    missing_delivery = set(TTS_ENGINES) - set(TTS_DELIVERY)
    missing_registry = set(TTS_DELIVERY) - set(TTS_ENGINES)
    assert not missing_delivery, f"In TTS_ENGINES but not TTS_DELIVERY: {sorted(missing_delivery)}"
    assert not missing_registry, f"In TTS_DELIVERY but not TTS_ENGINES: {sorted(missing_registry)}"


def test_every_engine_declares_a_valid_delivery() -> None:
    """Every figure the scorecard shows is labelled with its delivery, so an
    engine without one would render a latency with no way to read it."""
    for tts_id, engine in TTS_ENGINES.items():
        assert engine.delivery in VALID_DELIVERIES, f"{tts_id!r} has delivery {engine.delivery!r}"
        assert TTS_DELIVERY[tts_id] == engine.delivery, (
            f"{tts_id!r}: TTS_DELIVERY says {TTS_DELIVERY[tts_id]!r} but the engine says {engine.delivery!r}"
        )


def test_compared_engines_are_all_registered() -> None:
    for tts_id in COMPARISON_TTS_IDS:
        assert tts_id in TTS_ENGINES, f"COMPARISON_TTS_IDS names unregistered engine {tts_id!r}"


def test_comparison_ids_have_no_duplicates() -> None:
    """A duplicate would render the same engine twice and double-count it."""
    assert len(COMPARISON_TTS_IDS) == len(set(COMPARISON_TTS_IDS))


def test_lookups_return_none_for_an_unregistered_id() -> None:
    """The routes branch on None to answer 404 rather than raising."""
    assert engine_for("not-an-engine") is None
    assert delivery_for("not-an-engine") is None


@pytest.mark.parametrize("tts_id", sorted(TTS_ENGINES))
def test_every_engine_exposes_its_voices_and_a_default(tts_id: str) -> None:
    """Neither gateway has a list-voices endpoint, so the dropdown is whatever
    .env holds. An engine offering no voice at all could never be run."""
    engine = TTS_ENGINES[tts_id]
    settings = get_settings()
    voices = engine.voices(settings)
    assert isinstance(voices, list) and voices, f"{tts_id!r} exposes no voices"
    assert engine.default_voice(settings), f"{tts_id!r} has no default voice"


@pytest.mark.parametrize("tts_id", sorted(TTS_ENGINES))
def test_configured_is_a_predicate_over_settings(tts_id: str) -> None:
    """`configured` decides whether the UI offers the engine at all, so it must
    answer for any Settings without raising -- including a bare one."""
    assert isinstance(TTS_ENGINES[tts_id].configured(get_settings()), bool)


def test_tts_ids_do_not_collide_with_asr_ids() -> None:
    """Logs and reports mix the two, and `inception-stt` vs `inception-tts` is
    exactly the pair that would be confused if the ids were shared."""
    from apps.background_worker.transcription import ASR_ENGINES

    assert not set(TTS_ENGINES) & set(ASR_ENGINES)


# --- voice lists come from one comma-separated .env value --------------------
#
# `HAMSA_TTS_SPEAKER=Ruba,Sandra` was once set on a field that meant "the single
# default voice", beside a separate list field nobody updated. The default then
# became the literal string "Ruba,Sandra", which this vendor accepts with a 200
# and then drops the stream on -- indistinguishable from a network fault. One
# field now holds the list and position 0 is the default, so the two can no
# longer disagree.


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Ruba", ["Ruba"]),
        ("Ruba,Sandra", ["Ruba", "Sandra"]),
        ("Ruba, Sandra", ["Ruba", "Sandra"]),      # spaces after the comma
        ("Ruba,Sandra,", ["Ruba", "Sandra"]),      # the trailing-comma bug
        (" Ruba , Sandra ", ["Ruba", "Sandra"]),   # padding on both sides
        ("Ruba,,Sandra", ["Ruba", "Sandra"]),      # an empty entry
    ],
)
def test_speaker_list_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "hamsa_tts_speaker", raw)
    assert settings.hamsa_tts_speaker_options == expected
    assert settings.hamsa_tts_default_speaker == expected[0]


@pytest.mark.parametrize("raw", ["Ruba,Sandra", "Ruba,", " Ruba , Sandra "])
def test_default_voice_is_never_a_list(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """The regression guard for the whole bug class: whatever is in .env, the
    value actually POSTED as the speaker is a single clean name."""
    settings = get_settings()
    monkeypatch.setattr(settings, "hamsa_tts_speaker", raw)
    default = settings.hamsa_tts_default_speaker
    assert "," not in default
    assert default == default.strip()


def test_voice_options_never_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty list would make the dropdown unopenable and the default raise
    IndexError at request time, so a degenerate value falls back to itself."""
    settings = get_settings()
    for raw in (",", "   ", ""):
        monkeypatch.setattr(settings, "hamsa_tts_speaker", raw)
        assert settings.hamsa_tts_speaker_options
        assert settings.hamsa_tts_default_speaker is not None


def test_every_engine_declares_synthesis_params() -> None:
    """The voice picker renders these under the voice name. They must be
    engine-level settings (true of every voice), never per-voice metadata --
    neither gateway exposes any, so inventing some is the failure mode."""
    settings = get_settings()
    for tts_id, engine in TTS_ENGINES.items():
        params = engine.synthesis_params(settings)
        assert isinstance(params, dict), f"{tts_id!r} synthesis_params is not a dict"
        assert params, f"{tts_id!r} declares no synthesis params"
        for key, value in params.items():
            assert isinstance(key, str) and isinstance(value, str), f"{tts_id!r}: {key!r}={value!r}"
