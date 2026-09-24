"""The only code allowed to understand what the cloning endpoints return.

Two shapes, both small:

    /tts/voice_clone       -> {"global_token_ids": [...],
                               "semantic_token_ids": [...],
                               "prompt_text": "..."}
    /tts/load_voice_cloning -> null   (yes, really -- success is a bare null)

`null` as a success signal is the reason `adapt_registration` exists at all
rather than the route just checking a status code: a body that carries no
information is easy to mistake for a body that failed to arrive, and the one
place that distinction is made should be here.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VoiceTokens:
    """One extraction, reduced to what this repo stores and shows.

    `global_token_ids` / `semantic_token_ids` are kept as the pod returned
    them, unparsed and unconverted. They are re-posted verbatim at registration
    and are the ONLY way to re-register a voice after a pod restart drops it,
    so anything clever done to them here would be destructive.
    """

    global_token_ids: Any
    semantic_token_ids: Any
    #: What the pod echoed back, which is not guaranteed to equal what was sent
    #: -- the UI shows both so a silent rewrite is visible rather than assumed
    #: away.
    prompt_text: str
    extract_ms: int
    raw_meta: dict[str, Any]


class VoiceCloneShapeError(ValueError):
    """The pod answered 2xx with a body this adapter cannot read.

    Kept separate from the runner's upstream/connection errors: a 200 carrying
    the wrong shape is a contract change, not an outage, and the two need
    different responses from whoever is on call.
    """


def _token_count(value: Any) -> int:
    """How many tokens an array holds, for display only.

    The pod's schema types both fields as `array` three ways over (a nested
    shape, per its own OpenAPI `anyOf`), so this counts the outermost sequence
    and does not attempt to flatten. A count that cannot be taken is 0, never a
    guess -- and the stored arrays are unaffected either way.
    """
    if isinstance(value, list):
        return len(value)
    return 0


def adapt_extraction(raw: dict[str, Any]) -> VoiceTokens:
    """`runner.extract` output -> `VoiceTokens`."""
    body = raw.get("body")
    if not isinstance(body, dict):
        raise VoiceCloneShapeError(
            f"expected a JSON object from /tts/voice_clone, got {type(body).__name__}"
        )

    missing = [key for key in ("global_token_ids", "semantic_token_ids") if key not in body]
    if missing:
        raise VoiceCloneShapeError(
            f"/tts/voice_clone response is missing {', '.join(missing)}"
        )

    return VoiceTokens(
        global_token_ids=body["global_token_ids"],
        semantic_token_ids=body["semantic_token_ids"],
        # Absent is "" rather than an error: the tokens are what matter, and the
        # UI compares this against what was sent instead of trusting it.
        prompt_text=body.get("prompt_text") or "",
        extract_ms=int(raw.get("elapsed_ms") or 0),
        raw_meta={
            "status_code": raw.get("status_code"),
            "headers": raw.get("headers"),
            # NOT the token arrays: they are stored in their own columns and
            # duplicating thousands of integers into a metadata blob would make
            # every row unreadable for no gain.
            "global_token_count": _token_count(body["global_token_ids"]),
            "semantic_token_count": _token_count(body["semantic_token_ids"]),
        },
    )


@dataclass(frozen=True)
class Registration:
    """One registration call's outcome."""

    register_ms: int
    raw_meta: dict[str, Any]


def adapt_registration(raw: dict[str, Any]) -> Registration:
    """`runner.register` output -> `Registration`.

    A 2xx is the success signal. The body is `null` on success, so it is
    recorded rather than tested -- a future pod that returns an object instead
    must not start failing here, and one that returns a 2xx with an error
    payload would be an upstream contract break this cannot detect anyway.
    """
    return Registration(
        register_ms=int(raw.get("elapsed_ms") or 0),
        raw_meta={
            "status_code": raw.get("status_code"),
            "headers": raw.get("headers"),
            "body": raw.get("body"),
        },
    )


def token_counts(global_token_ids: Any, semantic_token_ids: Any) -> tuple[int, int]:
    """Both display counts, for the route that stores them."""
    return _token_count(global_token_ids), _token_count(semantic_token_ids)
