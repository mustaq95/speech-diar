"""Live-speech transcription: ASR + forced alignment.

A separate subsystem from the diarization models in `../models/`, deliberately.
The diarization contract (`DiarizationModelRun`) has no field for text, the
aligner DEPENDS on the ASR's output (every registered diarization model is
independent and fanned out in parallel), and `GET /models` drives the timeline
bands — an ASR engine listed there would render as a band with no segments.

The runner/adapter split still applies, per this platform's core rule: each
engine's `runner.py` executes it and returns its NATIVE output, and its
`adapter.py` is the only code allowed to understand that shape. No abstract
base class: three single-use engines do not need one.

The ASR engines' native output is also persisted verbatim
(`TranscriptResult.raw_output`) and served by one inspection route, since
`adapt` reduces it to a single string — hamsa's whole frame log becomes one
`" ".join(...)`. It travels as an opaque blob; nothing outside each engine's
own adapter parses it. The ALIGNER's native output is not kept: `words`
already carries the per-word timings it produces.

The mode, chosen per run from the Live Speech panel, selects one ASR engine:

    online  -> hamsa/             streams to a remote endpoint; audio leaves this host
    offline -> cohere/            local vLLM container; audio stays on-host

Both then run the SAME aligner (`ctc_aligner/`), so a transcript produced in
either mode carries timings derived the same way and the two are comparable.
"""

from dataclasses import dataclass
from typing import Any, Callable

from packages.config.settings import Settings, get_settings
from packages.shared_contracts.schemas import TranscriptionMode

from .cohere import adapter as cohere_adapter
from .cohere import runner as cohere_runner
from .hamsa import adapter as hamsa_adapter
from .hamsa import runner as hamsa_runner
from .inception import adapter as inception_adapter
from .inception import runner as inception_runner

#: Display name for the alignment stage, shown on the panel next to align_ms.
ALIGNER_NAME = "CTC Forced Aligner · MMS-300m"


@dataclass(frozen=True)
class AsrEngine:
    """One ASR engine: what it is called, where it runs, and how to run it."""

    asr_id: str
    mode: TranscriptionMode
    name: str
    #: audio path -> the engine's NATIVE output
    run: Callable[[str], Any]
    #: native output -> transcript text
    adapt: Callable[[Any], str]
    #: Is this engine actually configured on this host? Used to decide whether
    #: to enqueue a transcript at all, so an unconfigured host shows "not run"
    #: rather than a guaranteed failure on every upload.
    configured: Callable[[Settings], bool]

    #: (wav bytes, filename) -> one native entry, for the LIVE chunk route, plus
    #: the reader that finds the text inside that entry. None for an engine that
    #: has no chunk mode, which is not a gap to fill: `cohere-transcribe` is
    #: `transport="file"` precisely because chunking it changes what it sees.
    #:
    #: This pair exists so `POST /transcript/chunk` can DISPATCH on asr_id. It
    #: used to call the inception runner unconditionally while filing the result
    #: under whatever asr_id the request carried, so a second chunked engine
    #: would have had Inception's text and latency stored under its own name, at
    #: 200 OK, with nothing to notice but two suspiciously identical columns.
    run_chunk: Callable[[bytes, str], Any] | None = None
    chunk_text: Callable[[Any], str] | None = None
    #: native output -> how many pieces a BATCH run cut the recording into.
    #: Supplied by the adapter, because only it may read the native shape. None
    #: for an engine whose batch run is a single call (cohere: transport "file",
    #: so the chunk columns stay null rather than being filled with a 1).
    batch_segments: Callable[[Any], int | None] | None = None


def cohere_container_healthy(settings: Settings) -> bool:
    """Whether the offline engine's container is actually up.

    Offline is "configured" only when its container answers, not merely when a
    URL is set (the URL has a default). This is what disables the panel's
    offline toggle on a host where the container is down, instead of offering a
    run that would fail at connect. A short timeout keeps the boot-time
    `GET /config` probe from stalling when the container is absent.
    """
    url = settings.cohere_transcribe_url
    if not url:
        return False
    # Lazy import: the supervisor package pulls in heavier deps, and this module
    # is imported very early by the API. Reuses the same probe the GPU
    # supervisor uses for its own containers.
    from apps.background_worker.supervisor.containers import is_healthy

    return is_healthy(f"{url}/health", timeout=2.0)


ASR_ENGINES: dict[str, AsrEngine] = {
    engine.asr_id: engine
    for engine in (
        AsrEngine(
            asr_id="hamsa",
            mode="online",
            name="TryHamsa",
            run=hamsa_runner.run,
            adapt=hamsa_adapter.adapt,
            configured=lambda s: bool(s.hamsa_ws_endpoint and s.hamsa_stt_key),
        ),
        AsrEngine(
            asr_id="inception-stt",
            # Online: this gateway is remote, so the audio leaves the host —
            # same as hamsa. `mode` is no longer what picks an engine (asr_id
            # is); it stays as an honest description of where audio goes.
            mode="online",
            name="Inception-STT",
            run=inception_runner.run,
            adapt=inception_adapter.adapt,
            run_chunk=inception_runner.transcribe_bytes,
            chunk_text=inception_runner.text_of,
            batch_segments=inception_adapter.segment_count,
            configured=lambda s: bool(s.litellm_base_url and s.litellm_api_key),
        ),
        AsrEngine(
            asr_id="cohere-transcribe",
            mode="offline",
            name="Cohere",
            run=cohere_runner.run,
            adapt=cohere_adapter.adapt,
            run_chunk=cohere_runner.transcribe_bytes,
            chunk_text=cohere_runner.text_of,
            # Indirect through the module function so a test can monkeypatch it;
            # the lambda looks the name up at call time.
            configured=lambda s: cohere_container_healthy(s),
        ),
    )
}

#: Which engine each mode selects. Kept as an explicit map rather than a scan
#: of ASR_ENGINES so adding a second engine in one mode is a deliberate change,
#: not an accidental change of default.
MODE_TO_ASR_ID: dict[TranscriptionMode, str] = {
    "online": "hamsa",
    "offline": "cohere-transcribe",
}

#: The mode a run defaults to when the caller does not choose one: the upload
#: auto-transcribe path, which has no UI to pick from, and the mode the Live
#: Speech toggle starts on (the panel then always sends an explicit mode).
#:
#: Offline, because this is what runs unattended. It stays on this host and
#: costs seconds; online streams the audio to a remote service and is bound by
#: the recording's own length (~989s of wall clock for a 32-minute file). An
#: automatic run should never be the one that ships audio off the machine.
#: A code constant because it is not host-specific.
DEFAULT_TRANSCRIPTION_MODE: TranscriptionMode = "offline"


def asr_id_for_mode(mode: TranscriptionMode) -> str:
    """The engine id a given mode selects.

    Called at ENQUEUE time with the mode chosen for THIS run; the resulting id
    is stored on the job and the row, so a later run in the other mode never
    changes what an already-queued job runs or how an existing transcript is
    labelled.
    """
    return MODE_TO_ASR_ID[mode]


def default_asr_id(settings: Settings) -> str | None:
    """The engine to auto-transcribe an upload with, or None to skip.

    DEFAULT_TRANSCRIPTION_MODE's engine and no other -- deliberately no fallback
    to the opposite mode. Online sends the audio off this host and runs for
    roughly half the recording's duration, so it is never started on the
    operator's behalf; it runs only when someone asks for it from the panel.

    None when that one engine is not configured (e.g. the offline container is
    down): the panel then shows a neutral "no transcript" state with an offer to
    run one, rather than a guaranteed failure on every upload -- or a silent
    online run nobody chose. Only the upload path uses this; the panel picks a
    mode itself.
    """
    asr_id = MODE_TO_ASR_ID[DEFAULT_TRANSCRIPTION_MODE]
    engine = ASR_ENGINES.get(asr_id)
    return asr_id if engine and engine.configured(settings) else None


def engine_for(asr_id: str) -> AsrEngine | None:
    return ASR_ENGINES.get(asr_id)


#: Engines the transcript-evaluation surface compares, in display order.
#:
#: An explicit tuple, not "every configured engine": a column appearing because
#: someone started a container would change what the scorecard means. Adding one
#: is a deliberate edit here.
#:
#: cohere-transcribe is third and is NOT comparable on latency to the other two:
#: it is `transport="file"`, so it sees the whole recording once, after the fact,
#: while the others were fed live. Its error rates are measured against the same
#: reference and ARE comparable; its timing is a batch figure and every surface
#: labels it with the transport that produced it.
COMPARISON_ASR_IDS: tuple[str, ...] = ("hamsa", "inception-stt", "cohere-transcribe")

#: Which transport each engine's audio actually travels over. Not derivable from
#: `mode` -- hamsa and inception-stt are BOTH online, yet one streams
#: continuously with server-side VAD and the other only accepts short
#: request/response chunks. Every reported figure is labelled with this, because
#: a chunked engine carries its boundary cost inside its own error rate.
ASR_TRANSPORTS: dict[str, str] = {
    "hamsa": "stream",
    # Overridden at read time by `transport_for` when INCEPTION_BATCH_WHOLE_FILE
    # is set, which it is by default -- this entry is the split regime's value and
    # must never be trusted directly. Go through `transport_for`.
    "inception-stt": "chunks",
    # "file" because a stored-audio run of this engine genuinely is one whole-file
    # POST -- that is its native mode, and `run()` does no splitting. It is not a
    # correctness cliff: with COHERE_TRANSCRIBE_LANGUAGE at its "ar" default the
    # live chunked path returns proper Arabic too (463 Arabic characters over
    # 22x3s, measured), so the two modes are two honest pipelines rather than one
    # working and one broken.
    #
    # An earlier version of this comment claimed the model self-detects language
    # and that 3s was too short for it. That was wrong and is worth not
    # reintroducing: the endpoint has NO auto-detect (it rejects language=auto),
    # and an absent language field emits English at every duration tested, up to
    # 45s. The language field is what decides, not the chunk size. See
    # `cohere_transcribe_language` in packages/config/settings.py.
    "cohere-transcribe": "file",
}


#: How each engine can be FED, in menu order; entry 0 is the default.
#:
#:   live  -- fed while the operator reads, over the engine's own live transport
#:   batch -- run once over the stored recording after finalize, through `run_asr`
#:
#: A mode is not a transport. Inception-STT is "chunks" in BOTH modes: its batch
#: path still cuts the stored file at BATCH_SEGMENT_SECONDS, it just cuts from
#: storage and posts the pieces concurrently instead of one at a time behind a
#: speaker. What differs there is the timing (a live latency includes waiting for
#: someone to finish talking; a batch one is inference only), which is exactly
#: what `source` already records on the row. Cohere is the engine where the mode
#: also changes the transport, because its batch path is a single whole-file call.
#:
#: hamsa is deliberately live-only here. Its `run()` does work over stored audio,
#: so ("live", "batch") would function -- but a streaming engine replayed from a
#: file is measuring the socket, not the read-aloud, and nobody asked for it.
#: Adding it later is this one line.
ASR_FEED_MODES: dict[str, tuple[str, ...]] = {
    "hamsa": ("live",),
    "inception-stt": ("live", "batch"),
    "cohere-transcribe": ("live", "batch"),
}

#: The transport each engine uses when fed LIVE. Its batch counterpart is
#: `ASR_TRANSPORTS` above, which is what `run_asr` genuinely does -- the two
#: differ only for cohere-transcribe (chunks live, file batch).
LIVE_TRANSPORTS: dict[str, str] = {
    "hamsa": "stream",
    "inception-stt": "chunks",
    "cohere-transcribe": "chunks",
}


def feed_modes(asr_id: str) -> tuple[str, ...]:
    """The feed modes this engine offers; empty for an unregistered id."""
    return ASR_FEED_MODES.get(asr_id, ())


def default_feed_mode(asr_id: str) -> str | None:
    """The mode used when the caller does not choose one."""
    modes = ASR_FEED_MODES.get(asr_id)
    return modes[0] if modes else None


def resolve_feed_mode(asr_id: str, chosen: str | None = None) -> str:
    """Validate and resolve one engine's feed mode for a run.

    Raises KeyError for an unregistered engine and ValueError for a mode that
    engine does not offer, so a bad pick fails at the request rather than
    becoming a session that feeds the wrong route and mislabels its own output.
    """
    modes = ASR_FEED_MODES.get(asr_id)
    if not modes:
        raise KeyError(asr_id)
    if chosen is None:
        return modes[0]
    if chosen not in modes:
        raise ValueError(f"{asr_id!r} cannot run in {chosen!r} mode; it offers {list(modes)}")
    return chosen


def transport_for_mode(asr_id: str, mode: str) -> str | None:
    """The transport an engine actually uses in a given mode.

    Derived, never stored on the row as a third thing: `TranscriptResult` already
    carries `source` (live|batch) and `transport`, and those two together say
    everything the mode did.
    """
    if mode == "live":
        return LIVE_TRANSPORTS.get(asr_id)
    # Through `transport_for`, not the raw dict: inception-stt's batch transport
    # is resolved from a setting and the dict holds only one of its two values.
    return transport_for(asr_id)


def transport_for(asr_id: str) -> str | None:
    """The transport a STORED-AUDIO run of this engine uses, or None if
    unregistered. This is the `batch` half of `transport_for_mode`.

    Resolved rather than looked up, because inception-stt's batch behaviour is a
    setting: with INCEPTION_BATCH_WHOLE_FILE (the default) `run()` makes one
    whole-file call, and with it off it splits at BATCH_SEGMENT_SECONDS. A static
    entry beside a flag that changes the behaviour is the label-disagrees-with-
    reality bug this codebase has already paid for twice -- the row would be
    stamped "chunks" while the engine had swallowed the file whole, and every
    surface prints that label beside the score.
    """
    if asr_id == "inception-stt":
        return "file" if get_settings().inception_batch_whole_file else "chunks"
    return ASR_TRANSPORTS.get(asr_id)


def resolve_asr_ids(
    asr_ids: list[str] | None = None, mode: TranscriptionMode | None = None
) -> list[str]:
    """The engines a transcript request means, from either way of asking.

    `asr_ids` is the real selector and wins when given. `mode` is kept as an
    alias for the Live Speech panel, which predates multi-engine runs and asks
    by mode -- but a mode can no longer identify an engine on its own (hamsa and
    inception-stt are both "online"), so it resolves through MODE_TO_ASR_ID's
    one explicit choice per mode rather than by scanning for a match.

    Duplicates are collapsed and order is preserved: a caller asking for the
    same engine twice must not get two jobs writing to one row.

    Raises KeyError for an unregistered id, so a typo fails at the request
    instead of becoming a queued job that dies in a worker.
    """
    if asr_ids:
        seen: dict[str, None] = {}
        for asr_id in asr_ids:
            if asr_id not in ASR_ENGINES:
                raise KeyError(asr_id)
            seen[asr_id] = None
        return list(seen)
    return [MODE_TO_ASR_ID[mode or DEFAULT_TRANSCRIPTION_MODE]]
