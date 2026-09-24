"""Unit coverage for the voice-cloning adapter — the only code allowed to read
what the pod returns.

Every case here is a real shape observed against the live pod on 2026-09-09 or
a shape its own /openapi.json permits, not an invented one. The two that matter
most are the ones the vendor's written guide gets wrong: registration answers a
bare `null` on success, and a pod without the cloning model loaded answers a
`text/plain` 500 rather than the documented "error describing the cause".
"""

import pytest

from apps.background_worker.voice_clone.adapter import (
    VoiceCloneShapeError,
    adapt_extraction,
    adapt_registration,
    token_counts,
)


def _extract_raw(body, elapsed_ms: int = 1400) -> dict:
    return {
        "body": body,
        "headers": {"content-type": "application/json"},
        "status_code": 200,
        "elapsed_ms": elapsed_ms,
    }


def test_adapt_extraction_keeps_the_token_arrays_verbatim() -> None:
    """The arrays are re-posted at registration and are the only way to
    re-register after a pod restart, so anything done to them here is
    destructive."""
    global_ids = [8213, 1184, 4427, 331, 9042]
    semantic_ids = [412, 7761, 2038, 5590, 118]
    tokens = adapt_extraction(
        _extract_raw(
            {
                "global_token_ids": global_ids,
                "semantic_token_ids": semantic_ids,
                "prompt_text": "مرحبا بكم في هيئة أبوظبي الرقمية",
            }
        )
    )
    assert tokens.global_token_ids == global_ids
    assert tokens.semantic_token_ids == semantic_ids
    # Identity, not just equality: no copy, no coercion, no re-typing.
    assert tokens.global_token_ids is global_ids
    assert tokens.prompt_text == "مرحبا بكم في هيئة أبوظبي الرقمية"
    assert tokens.extract_ms == 1400


def test_adapt_extraction_counts_tokens_for_display() -> None:
    tokens = adapt_extraction(
        _extract_raw({"global_token_ids": [1] * 1024, "semantic_token_ids": [2] * 3872})
    )
    assert tokens.raw_meta["global_token_count"] == 1024
    assert tokens.raw_meta["semantic_token_count"] == 3872


def test_raw_meta_never_carries_the_token_arrays() -> None:
    """They live in their own columns; duplicating thousands of integers into a
    metadata blob would make every row unreadable for no gain."""
    tokens = adapt_extraction(
        _extract_raw({"global_token_ids": [1] * 50, "semantic_token_ids": [2] * 60})
    )
    assert "global_token_ids" not in tokens.raw_meta
    assert "semantic_token_ids" not in tokens.raw_meta


def test_missing_prompt_text_is_empty_not_an_error() -> None:
    """The tokens are what matter. The UI compares this against what was SENT
    rather than trusting it, so an absent echo must not fail the extraction."""
    tokens = adapt_extraction(_extract_raw({"global_token_ids": [1], "semantic_token_ids": [2]}))
    assert tokens.prompt_text == ""


@pytest.mark.parametrize(
    "body",
    [
        # The real failure this pod produces: a bare text/plain body. It arrives
        # with a 5xx so the runner raises first, but a 200 carrying it must not
        # be mistaken for success either.
        "Internal Server Error",
        None,
        [],
        42,
    ],
)
def test_a_non_object_body_is_a_shape_error(body) -> None:
    with pytest.raises(VoiceCloneShapeError):
        adapt_extraction(_extract_raw(body))


@pytest.mark.parametrize(
    "body",
    [
        {"semantic_token_ids": [1]},
        {"global_token_ids": [1]},
        {"prompt_text": "hi"},
    ],
)
def test_a_missing_token_field_is_a_shape_error(body) -> None:
    """A 200 with half the tokens is a contract change, not an outage — and
    registering half a voice would 'succeed' and then synthesize garbage."""
    with pytest.raises(VoiceCloneShapeError) as excinfo:
        adapt_extraction(_extract_raw(body))
    assert "missing" in str(excinfo.value)


def test_adapt_registration_treats_a_null_body_as_success() -> None:
    """`POST /tts/load_voice_cloning` really does answer `null` on success.

    A body carrying no information is easy to mistake for a body that failed to
    arrive; this is the one place that distinction is made.
    """
    outcome = adapt_registration(
        {"body": None, "headers": {"content-type": "application/json"}, "status_code": 200, "elapsed_ms": 310}
    )
    assert outcome.register_ms == 310
    assert outcome.raw_meta["status_code"] == 200
    assert outcome.raw_meta["body"] is None


def test_adapt_registration_does_not_reject_a_future_object_body() -> None:
    """A pod that starts returning an object instead of null must not begin
    failing here: the 2xx is the success signal, the body is recorded."""
    outcome = adapt_registration(
        {"body": {"ok": True}, "headers": {}, "status_code": 200, "elapsed_ms": 5}
    )
    assert outcome.raw_meta["body"] == {"ok": True}


def test_token_counts_never_guesses() -> None:
    """A count that cannot be taken is 0, not an estimate — and the stored
    arrays are unaffected either way."""
    assert token_counts([1, 2, 3], [4, 5]) == (3, 2)
    assert token_counts(None, "not-an-array") == (0, 0)
