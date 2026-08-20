"""Generates the read-aloud reference script.

The transcript comparison needs a ground truth. The read-aloud flow gets one by
generating a script FIRST and having someone read it: the words are known before
the audio exists, which is what makes the resulting WER a measurement rather
than an estimate.

Reached over an OpenAI-compatible chat-completions gateway (`LLM_*`), kept
separate from the STT gateway (`LITELLM_*` / `STT_*`) with no cross-default
between them — a silent fallback between two gateways is how the wrong model
gets called. They may well be the same host; they still get their own settings.

Two things this module refuses to do:

  * **Report a word count it did not measure.** `words_per_minute` only sizes the
    prompt. The count that comes back is `len(text.split())` on the generated
    text, because that is the number the WER denominator uses and a requested
    length is not evidence of a produced length.
  * **Repair the model's output.** Markdown fences and a stray preamble are
    stripped (they are formatting the prompt asked it to omit, not content), but
    the words themselves are passed through. A script that came back shorter than
    asked is reported at its real length rather than padded.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from packages.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

#: What each language-mix option asks the model for. Keys match
#: SCRIPT_LANGUAGE_MIXES in .env, which GET /config serves to the browser, so the
#: UI's buttons and these instructions cannot drift apart.
LANGUAGE_MIX_PROMPTS: dict[str, str] = {
    "ar": "Write entirely in Gulf/Emirati Arabic. Do not use any English words.",
    # Measured, not assumed: asking for "half and half" produced 76% Arabic / 21%
    # English, because the model reaches for Arabic sentences with English nouns
    # dropped in. An explicit per-language word budget plus "complete clauses"
    # is what actually moves it.
    "mixed-50-50": (
        "CODE-SWITCH IN EVEN HALVES. Of the total word count, about half the words "
        "must be Arabic and about half must be English — count them as you write. "
        "Do NOT write Arabic sentences with a few English nouns dropped in; that is "
        "the failure mode. Instead alternate COMPLETE CLAUSES: an Arabic clause, then "
        "a full English clause of comparable length, then back to Arabic. Switch "
        "language roughly every eight to twelve words, and never run more than about "
        "fifteen words in one language before switching. Both languages must carry "
        "real content — verbs, subjects and whole thoughts, not just borrowed "
        "terminology. Use Gulf/Emirati Arabic for the Arabic half."
    ),
    "en": "Write entirely in English. Do not use any Arabic.",
}

#: What each hard-case chip adds. Keys match SCRIPT_HARD_CASES in .env.
HARD_CASE_PROMPTS: dict[str, str] = {
    "code-switch-en": (
        "Switch between Arabic and English inside single sentences, not only between them."
    ),
    "proper-nouns": (
        "Include several Emirati personal names and Abu Dhabi place names, alongside a "
        "few Western names, of the kind a speech recognizer could plausibly "
        "mistranscribe. Every name must appear in the THIRD PERSON as part of what "
        "the speaker is describing — never as someone being spoken to."
    ),
    "numbers-dates": (
        "Include specific numbers, percentages, times, and dates, written as words or digits "
        "the way someone would actually say them aloud."
    ),
    "gulf-dialect": (
        "Use Gulf/Emirati dialect vocabulary and sentence particles rather than Modern "
        "Standard Arabic."
    ),
    "fast-speech": (
        "Include a few long run-on sentences with little punctuation, of the kind that get "
        "delivered quickly with no pause."
    ),
}

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")

#: Arabic script, for measuring what the model actually produced.
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN = re.compile(r"[A-Za-z]")


def measure_language_split(text: str) -> dict[str, float]:
    """Share of words that are Arabic vs English, measured from the text.

    A requested mix is an instruction, not an outcome. Asking for an even split
    produced anywhere from 38% to 73% Arabic across three runs at the same
    settings, so the card reports what came back rather than repeating what was
    asked for. Words containing neither script (bare numerals, punctuation) are
    counted in the total but in neither share, which is why the two do not sum
    to 1.
    """
    words = text.split()
    if not words:
        return {"arabic": 0.0, "english": 0.0, "words": 0}
    arabic = sum(1 for word in words if _ARABIC.search(word))
    english = sum(1 for word in words if _LATIN.search(word) and not _ARABIC.search(word))
    return {
        "arabic": round(arabic / len(words), 3),
        "english": round(english / len(words), 3),
        "words": len(words),
    }


class ScriptGenerationError(RuntimeError):
    """Base for failures generating a script."""


class ScriptGatewayUnconfigured(ScriptGenerationError):
    """No chat-completions gateway is configured on this host."""


class ScriptGatewayError(ScriptGenerationError):
    """The gateway was unreachable, timed out, or answered with an error."""


@dataclass(frozen=True)
class GeneratedScript:
    """A script plus the request that produced it.

    `word_count` is measured from `text`. `params` is what gets stored on the
    reference row, so a score can later be traced back to what was asked for.
    """

    text: str
    word_count: int
    generator_model: str
    params: dict[str, Any]


def build_prompt(
    minutes: float, language_mix: str, hard_cases: list[str], words_per_minute: int
) -> str:
    """The instruction sent to the model, assembled from the chosen options.

    Separate from the request so a test can assert the prompt carries every
    selected hard case without spending a call on it.
    """
    target_words = max(1, int(round(minutes * words_per_minute)))
    lines = [
        f"Write a script to be read aloud in about {minutes:g} minute(s) — "
        f"approximately {target_words} words.",
        "",
        LANGUAGE_MIX_PROMPTS.get(language_mix, LANGUAGE_MIX_PROMPTS["mixed-50-50"]),
        "",
        # The domain is not decoration. This script is the reference a speech
        # recognizer is scored against, so it has to contain the vocabulary these
        # speakers actually use — entity names, programme names, the shape of a
        # government KPI review. Generic filler would test the engines on words
        # nobody here says.
        # "addressing peers" was the original wording and it produced dialogue:
        # the model wrote vocatives and direct questions ("Ahmed, you got any
        # updates?"), which reads as one half of a conversation. A read-aloud
        # reference has to be deliverable start to finish by one person with
        # nobody answering.
        "SETTING: a senior executive briefing inside an Abu Dhabi government entity "
        "or GRE. One person delivering an uninterrupted spoken update.",
        "",
        "IT MUST BE A MONOLOGUE. Specifically:",
        "- No speaker labels, names, initials or dashes introducing lines.",
        "- Never address anyone by name and never use 'you' to mean a specific "
        "person in the room. No vocatives at all.",
        "- No questions put to another person, and no place where the text waits for "
        "an answer or reacts to one. A rhetorical question the speaker then answers "
        "themselves is fine.",
        "- No quoted or reported speech from other people.",
        "- Refer to colleagues only in the third person, as people whose work is "
        "being described.",
        "",
        "Draw the subject matter from what these executives actually discuss: digital "
        "transformation and service delivery, quarterly KPI and performance reviews, "
        "budget cycles and capital allocation, Emiratisation and workforce targets, "
        "AI and data platform rollouts, entity restructuring, regulatory approvals, "
        "and inter-entity coordination. Use the real institutional vocabulary of Abu "
        "Dhabi (entities, authorities, departments, programme and initiative names, "
        "committee and board structures, government fiscal years and quarters).",
        "",
        "It should sound like one person talking naturally: continuous prose, no "
        "speaker labels, no stage directions, no headings, no bullet points.",
    ]
    selected = [HARD_CASE_PROMPTS[case] for case in hard_cases if case in HARD_CASE_PROMPTS]
    if selected:
        lines += ["", "Additionally:"] + [f"- {line}" for line in selected]
    # A closing "rewrite if unbalanced" reminder was tried and removed: it swung
    # the output the other way (29-51% Arabic across three runs, from 38-73%
    # without it). The per-language budget above lands better centred on its own,
    # and the card reports the split that was actually produced.
    lines += [
        "",
        "Output ONLY the script text itself. No preamble, no explanation, no markdown, "
        "no quotation marks around it, no word count.",
    ]
    return "\n".join(lines)


def _clean(content: str) -> str:
    """Strip formatting the prompt asked the model to omit.

    Fences and a wrapping quote pair only. This does not touch the words: it is
    removing packaging, not editing a reference the engines will be scored
    against.
    """
    text = _FENCE.sub("", content.strip())
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text


#: The mix whose balance can be checked. `ar` and `en` ask for one language and
#: get it; only an even split has a measurable target to miss.
CHECKED_MIX = "mixed-50-50"


def generate(
    minutes: float,
    language_mix: str = CHECKED_MIX,
    hard_cases: list[str] | None = None,
    settings: Settings | None = None,
) -> GeneratedScript:
    """Generate a script, retrying an even-mix request that came back lopsided.

    Adherence to the even split is bimodal, not noisy: measured over six runs at
    identical settings the Arabic share was 42/92/94% at one minute and 78/78/46%
    at three. The model either alternates clauses properly or reverts to Arabic
    sentences with English nouns dropped in, and prompt wording alone did not
    move that. So the split is measured and a lopsided script is regenerated.

    The last attempt is returned even when it misses the band: a slightly
    unbalanced script still beats no script, and `params["mixAttempts"]` records
    how many tries it took so a host that consistently misses is visible rather
    than silently expensive.
    """
    settings = settings or get_settings()
    attempts = max(1, settings.script_mix_max_attempts) if language_mix == CHECKED_MIX else 1
    result = _generate_once(minutes, language_mix, hard_cases, settings)
    for attempt in range(2, attempts + 1):
        arabic = result.params["languageSplit"]["arabic"]
        if abs(arabic - 0.5) <= settings.script_mix_tolerance:
            break
        logger.info(
            "script mix %.0f%% Arabic is outside the band; regenerating (attempt %d of %d)",
            arabic * 100, attempt, attempts,
        )
        result = _generate_once(minutes, language_mix, hard_cases, settings)
        result.params["mixAttempts"] = attempt
    return result


def _generate_once(
    minutes: float,
    language_mix: str,
    hard_cases: list[str] | None,
    settings: Settings,
) -> GeneratedScript:
    """One generation call. Raises rather than returning a placeholder."""
    if not settings.llm_chat_url or not settings.llm_api_key or not settings.llm_model:
        raise ScriptGatewayUnconfigured(
            "script generation is not configured — set LLM_BASE_URL, LLM_API_KEY and "
            "LLM_MODEL in .env"
        )

    hard_cases = hard_cases or []
    prompt = build_prompt(minutes, language_mix, hard_cases, settings.script_words_per_minute)

    started = time.perf_counter()
    try:
        response = httpx.post(
            settings.llm_chat_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            verify=settings.litellm_httpx_verify,
            timeout=settings.llm_timeout_sec,
        )
    except httpx.TimeoutException as exc:
        raise ScriptGatewayError(
            f"script gateway did not respond within {settings.llm_timeout_sec}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise ScriptGatewayError(f"script gateway is not reachable: {exc}") from exc

    if response.status_code >= 400:
        # The body carries the actionable reason (a wrong model id, a rejected
        # key); httpx's own message would reduce it to the status line.
        raise ScriptGatewayError(f"script gateway {response.status_code}: {response.text[:400]}")

    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        finish_reason = payload["choices"][0].get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ScriptGatewayError(
            f"unexpected response shape from the script gateway: {response.text[:400]}"
        ) from exc

    text = _clean(content or "")
    if not text:
        raise ScriptGatewayError("the script gateway returned an empty script")

    word_count = len(text.split())
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "generated a %d-word script (%s, %s mix) in %d ms",
        word_count, settings.llm_model, language_mix, elapsed_ms,
    )
    return GeneratedScript(
        text=text,
        word_count=word_count,
        generator_model=settings.llm_model,
        params={
            "minutes": minutes,
            "languageMix": language_mix,
            "hardCases": hard_cases,
            "generatorModel": settings.llm_model,
            "targetWords": max(1, int(round(minutes * settings.script_words_per_minute))),
            # Recorded because it explains a script that came back short: the
            # model hit its token ceiling rather than choosing to stop.
            "finishReason": finish_reason,
            "generationMs": elapsed_ms,
            # What the model ACTUALLY produced, not what was asked for.
            "languageSplit": measure_language_split(text),
            "mixAttempts": 1,
        },
    )
