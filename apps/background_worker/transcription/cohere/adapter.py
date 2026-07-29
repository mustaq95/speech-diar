"""Translates the Cohere container's vLLM response into a transcript string.

Native shape: vLLM's OpenAI-compatible transcription JSON,

    {"text": "...", "usage": {...}}

The `usage` block is token accounting, not speech, and is dropped. The text is
passed through exactly as generated — no normalization happens here, because
the aligner (`../ctc_aligner/`) does its own alignment-facing normalization and
the panel displays the original words.
"""

from .runner import CohereRawOutput


def adapt(raw: CohereRawOutput) -> str:
    return (raw.get("text") or "").strip()
