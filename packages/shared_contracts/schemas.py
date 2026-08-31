"""Source of truth for every data structure that crosses a service boundary.

`schemas.py` (Pydantic, used by backend_api and background_worker) and
`types.ts` (TypeScript, consumed by apps/frontend) describe the SAME wire
format and must be kept in sync. The wire format is camelCase JSON; the
Python models use snake_case attributes with camelCase aliases.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

ModelStatus = Literal["queued", "running", "done", "failed"]

ModelLifecycleState = Literal["unloaded", "starting", "ready", "in_use", "stopping", "unhealthy"]

#: Which half of the transcript pipeline a run is in. The two stages are two
#: separate RQ jobs (ASR, then alignment), each timed on its own.
TranscriptStage = Literal["asr", "align"]

#: Where the ASR ran. `online` streamed the audio to a remote endpoint;
#: `offline` kept it on this host. On a produced transcript this is derived from
#: the engine that actually ran, not from the mode a new run would use.
#:
#: No longer what SELECTS an engine -- `asr_id` is. Two engines are `online`
#: (hamsa and inception-stt), so a mode cannot identify one. It remains an
#: honest statement about where the audio goes.
TranscriptionMode = Literal["online", "offline"]

#: How a transcript was produced. `live` was captured chunk by chunk while
#: someone read a script aloud; `batch` was run over stored audio. Not
#: interchangeable: they are different measurements of the same engine.
TranscriptSource = Literal["live", "batch"]

#: What the audio travelled over for a given engine. `stream` is continuous with
#: engine-side VAD; `chunks` is fixed-interval cuts; `file` is the whole recording
#: in one call, after the fact. A chunked engine carries its boundary cost inside
#: its own error rate, so every figure is labelled with this and the transports'
#: chunk statistics are NOT comparable to each other.
#:
#: `file` is not a live transport and its engine is fed nothing while the operator
#: reads: it runs once over the stored audio at finalize. It exists because an
#: engine whose native mode is a whole-file POST should be measured in that mode
#: -- cohere-transcribe's `run()` posts the recording in one call and does no
#: splitting, so "chunks" would misdescribe it. It is a second honest pipeline,
#: not a workaround: that engine's live chunked path works too.
TranscriptTransport = Literal["stream", "chunks", "file"]

#: Where a reference transcript came from. `script` was generated and read aloud,
#: so the words were known before the audio existed; `pasted` was supplied by
#: hand afterwards. A WER means something different against each.
ReferenceSource = Literal["script", "pasted"]

#: One edit operation aligning hypothesis to reference. `equal` words matched;
#: the rest are the three error classes WER counts.
AlignmentOp = Literal["equal", "sub", "del", "ins"]

#: How a TTS engine delivers its audio. `stream` is a single POST whose BODY
#: streams (hamsa-tts); `single` is a request/response call that still arrives
#: chunked but with no meaningful front-loading (inception-tts). Deliberately
#: not the STT side's `transport` vocabulary -- neither describes the other
#: direction correctly.
TtsDelivery = Literal["stream", "single"]


class ContractModel(BaseModel):
    """Base for all shared contracts: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class DiarizationSegment(ContractModel):
    """One contiguous stretch of speech attributed to a single speaker."""

    spk: int = Field(ge=0, description="Zero-based speaker index, stable within one model run")
    s: float = Field(ge=0, description="Start time in seconds from the beginning of the audio")
    e: float = Field(ge=0, description="End time in seconds (exclusive), always >= s")


class DiarizationModelRun(ContractModel):
    """The complete diarization output of one model over one audio file."""

    id: str = Field(description="Stable model identifier, e.g. 'pyannote'")
    name: str = Field(description="Full display name, e.g. 'PyAnnote Audio 3.1'")
    short: str = Field(description="Short label used in pills and compact rows")
    description: str = Field(description="One-line description shown in settings/model cards")
    segs: list[DiarizationSegment] = Field(default_factory=list, description="Speech segments, sorted by start time")
    num_spk: int = Field(default=0, ge=0, description="Distinct speakers detected (max spk + 1)")
    status: ModelStatus | None = Field(default=None, description="queued|running|done|failed; the real, true state")
    error: str | None = Field(default=None, description="Failure reason when status == 'failed'")
    loading_started_at: datetime | None = Field(
        default=None,
        description=(
            "When the GPU supervisor granted a slot and began waiting for the model's "
            "container to become healthy. None for models that never needed a cold start. "
            "The interval [loadingStartedAt, startedAt) is cold-start wait, not inference."
        ),
    )
    started_at: datetime | None = Field(
        default=None, description="When inference actually began (container confirmed healthy)"
    )
    finished_at: datetime | None = Field(default=None, description="When the worker finished running this model")
    processing_ms: int | None = Field(
        default=None,
        description=(
            "Wall-clock run time in ms, started_at->finished_at (excludes queue wait). "
            "For azure-batch, Azure's own reported duration when available."
        ),
    )


class DiarizationEvaluation(ContractModel):
    """Everything the frontend needs to render one evaluation session."""

    audio_file_id: int = Field(description="Primary key of the AudioFile this evaluation belongs to")
    duration_sec: float = Field(ge=0, description="Audio duration in seconds")
    upload_ms: int | None = Field(default=None, description="Client-perceived upload time (browser start -> API response)")
    models: list[DiarizationModelRun]


class ModelRawOutput(ContractModel):
    """One model's run on one recording, both representations side by side.

    An inspection surface, served only by `GET
    /evaluations/{id}/models/{model_id}/raw` and deliberately absent from the
    polled evaluation response: raw output is the one thing here that can run to
    megabytes. `raw_output` is left untyped because its shape is whatever the
    engine emits -- nothing outside that engine's own adapter may parse it.
    """

    model_id: str = Field(description="The model this run belongs to")
    status: ModelStatus | None = Field(default=None, description="queued|running|done|failed")
    raw_output: dict | list | None = Field(
        default=None,
        description=(
            "The engine's native output, verbatim. Null when the run predates raw-output "
            "persistence or did not reach completion -- only a done run stores one."
        ),
    )
    run: DiarizationModelRun | None = Field(
        default=None, description="The adapted contract built from that raw output"
    )


class TranscriptWord(ContractModel):
    """One word of the transcript, with the timing the aligner gave it.

    `s`/`e` are None when the aligner could not place the word (its characters
    are outside the CTC vocabulary, e.g. a Latin token in an Arabic-script
    model). Such a word is still carried, unplaced, rather than dropped or
    given a guessed time.
    """

    w: str = Field(description="The word exactly as the ASR emitted it, for display")
    s: float | None = Field(default=None, ge=0, description="Start time in seconds, or None if unaligned")
    e: float | None = Field(default=None, ge=0, description="End time in seconds, or None if unaligned")
    score: float | None = Field(
        default=None,
        description=(
            "The aligner's score for this word: a mean log-probability, so <= 0 and "
            "higher is better. NOT a 0..1 confidence -- do not render it as a percentage."
        ),
    )


class TranscriptAlignmentOp(ContractModel):
    """One step of the reference-to-hypothesis alignment, for the word-level
    error highlight.

    This comes from the WER edit-distance backtrace, NOT from any aligner or
    per-word timing: the boxed words in the UI are substitutions against the
    reference, which is a text comparison and needs no audio.
    """

    op: AlignmentOp
    ref: str | None = Field(default=None, description="Reference word; None for an insertion")
    hyp: str | None = Field(default=None, description="Hypothesis word; None for a deletion")
    #: Index into the hypothesis word list, so the UI can mark the rendered word.
    hyp_index: int | None = Field(default=None, ge=0)


class TranscriptMetrics(ContractModel):
    """How one engine's transcript scored against the recording's reference.

    Both normalized and raw rates are carried so the UI's normalization toggle
    is a read rather than a recompute, and so normalization's own effect is
    visible instead of being invisible preprocessing.

    `rtf` is None when the engine's transport cannot produce one: a real-time
    streaming protocol consumes audio at 1x by definition, so a figure for it
    would be invented. The UI shows "real-time bound" for a null, never a number.
    """

    wer: float = Field(ge=0, description="Word error rate, normalized text, 0..1+ (S+D+I over reference words)")
    cer: float = Field(ge=0, description="Character error rate, normalized text")
    wer_raw: float = Field(ge=0, description="Word error rate WITHOUT normalization")
    cer_raw: float = Field(ge=0, description="Character error rate WITHOUT normalization")
    ref_word_count: int = Field(ge=0)
    hyp_word_count: int = Field(ge=0)
    sub_count: int = Field(ge=0)
    del_count: int = Field(ge=0)
    ins_count: int = Field(ge=0)
    rtf: float | None = Field(
        default=None,
        ge=0,
        description="Processing time / audio duration; None when the transport is real-time bound",
    )


class TranscriptReference(ContractModel):
    """The ground truth a recording's engines are scored against.

    One per recording, never one per engine: comparability depends on every
    engine being scored against the same text.
    """

    audio_file_id: int
    source: ReferenceSource
    text: str
    word_count: int = Field(ge=0, description="Measured from `text`, never taken from a request")
    params: dict | None = Field(
        default=None,
        description="For a generated script: the request that produced it, plus the generating model",
    )


#: Which evaluation surface produced a recording. Diarization recordings come from
#: an upload; transcript recordings are read-aloud captures. They are listed
#: separately because they are different artifacts scored in different ways.
RecordingSurface = Literal["diarization", "transcript"]


class RecordingSummary(ContractModel):
    """One row of the recordings list.

    Carries what a list row needs and nothing more. The counts are computed by
    aggregate query on the server rather than derived client-side: the list used to
    be assembled by fetching every recording's full evaluation one request at a
    time, which is N requests to render one page.

    Deliberately NOT here: `payload` and `raw_output`. Both are deferred columns
    holding a whole model's output, and a list has no use for either.
    """

    audio_file_id: int
    surface: RecordingSurface
    filename: str = Field(description="Display name; not unique")
    duration_sec: float = Field(ge=0)
    created_at: str = Field(description="ISO-8601; the list is ordered by this, newest first")
    has_audio: bool = Field(
        default=True,
        description=(
            "Whether a recording exists yet. False for a transcript row created when its "
            "script was generated but never read aloud: `duration_sec` is 0 because nothing "
            "was measured, not because the audio is zero-length."
        ),
    )

    # --- diarization recordings ---
    model_count: int = Field(default=0, ge=0, description="Models that have a result row")
    speaker_count: int = Field(default=0, ge=0, description="Highest speaker count any model found")
    done_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)

    # --- transcript recordings ---
    engine_count: int = Field(default=0, ge=0, description="ASR engines compared")
    scored: bool = Field(default=False, description="Whether a reference exists and runs were scored")
    best_wer: float | None = Field(
        default=None, ge=0, description="Lowest WER across engines; None when unscored"
    )
    tts_count: int = Field(default=0, ge=0, description="TTS engines that have synthesized this recording's reference")


class ScriptRequest(ContractModel):
    """What to generate a read-aloud script for.

    The options are served by `GET /config` from `.env`, so the UI's slider stops
    and chips are not literals in the frontend and this request cannot ask for a
    combination the host was never configured to offer.
    """

    minutes: float = Field(gt=0, le=60, description="Target read-aloud length")
    language_mix: str = Field(
        default="mixed-70-30", description="One of GET /config transcript.scriptLanguageMixes"
    )
    hard_cases: list[str] = Field(
        default_factory=list, description="Subset of GET /config transcript.scriptHardCases"
    )


class GeneratedScript(ContractModel):
    """A generated script, with the request that produced it.

    `wordCount` is MEASURED from `text`, never echoed from the request: it becomes
    the WER denominator, so a requested length must never be mistaken for a
    produced one.
    """

    text: str
    word_count: int = Field(ge=0, description="Measured from `text`")
    generator_model: str = Field(description="The model that actually generated it")
    params: dict = Field(description="The request, plus finish reason and generation time")
    audio_file_id: int = Field(
        description=(
            "The recording row this script was saved to. Created with the script so it "
            "appears in Projects before anything has been recorded; finalize attaches the "
            "audio to this same row."
        ),
    )


class TranscriptRun(ContractModel):
    """One engine's live-speech transcript of one audio file: ASR text plus
    word-level timings, with each stage's real measured cost.

    Separate from `DiarizationModelRun` on purpose — this is transcription, not
    diarization, and the diarization contract has no place for text.

    One transcript per (audio file, engine): a recording holds an online AND an
    offline run at once, which is the comparison this platform exists to make.
    `GET /evaluations/{id}/transcript` therefore returns a list of these.
    """

    audio_file_id: int
    status: ModelStatus = Field(description="queued|running|done|failed across both stages")
    stage: TranscriptStage | None = Field(
        default=None, description="Stage currently running, or the stage that failed; None once done"
    )
    asr_id: str = Field(description="Engine that actually ran: 'hamsa' | 'cohere-transcribe'")
    mode: TranscriptionMode = Field(
        description="Where the ASR ran, derived from asr_id -- NOT from the current setting"
    )
    asr_name: str = Field(description="Display name of the ASR engine, e.g. 'TryHamsa STT'")
    aligner_name: str = Field(description="Display name of the forced aligner")
    text: str = Field(default="", description="Full ASR transcript; present as soon as the ASR stage finishes")
    words: list[TranscriptWord] = Field(default_factory=list, description="Empty until the alignment stage finishes")
    asr_ms: int | None = Field(default=None, description="Measured wall-clock time of the ASR stage in ms")
    align_ms: int | None = Field(default=None, description="Measured wall-clock time of the alignment stage in ms")
    error: str | None = Field(default=None, description="Failure reason when status == 'failed'")

    # --- transcript evaluation: how this run was produced, and how it scored ---
    source: TranscriptSource = Field(
        default="batch", description="live (read-aloud capture) or batch (stored audio)"
    )
    transport: TranscriptTransport | None = Field(
        default=None, description="stream or chunks — label every figure with it; the two are not comparable"
    )
    chunk_interval_sec: float | None = Field(
        default=None, gt=0, description="Cut interval for a chunked transport; None for a stream"
    )
    chunk_count: int | None = Field(default=None, ge=0)
    first_latency_ms: int | None = Field(
        default=None, ge=0, description="Measured latency of the first chunk/frame: the panel's 'lag'"
    )
    avg_latency_ms: int | None = Field(
        default=None, ge=0, description="Mean measured chunk latency — the comparable headline figure"
    )
    metrics: TranscriptMetrics | None = Field(
        default=None, description="None until this recording has a reference and the run has finished"
    )


class TranscriptRawOutput(ContractModel):
    """`ModelRawOutput`'s counterpart for one ASR engine's transcript.

    `raw_output` is the ASR engine's native output only -- hamsa's WebSocket
    frame log or cohere's response JSON. The aligner's native output is not
    persisted, because `run.words` already carries its per-word timings.
    """

    asr_id: str = Field(description="Engine that actually ran: 'hamsa' | 'cohere-transcribe'")
    status: ModelStatus | None = Field(default=None, description="queued|running|done|failed")
    raw_output: dict | list | None = Field(
        default=None,
        description=(
            "The ASR engine's native output, verbatim. Null when the transcript predates "
            "raw-output persistence or its ASR stage has not finished."
        ),
    )
    run: TranscriptRun | None = Field(
        default=None, description="The adapted transcript built from that raw output"
    )


class TtsRun(ContractModel):
    """One TTS engine's synthesis of one recording's reference text.

    Mirrors `TranscriptRun`'s per-engine shape, on the other axis: a recording
    holds one run per TTS engine at once (hamsa-tts AND inception-tts), which is
    the comparison this surface makes. `GET /evaluations/{id}/tts` returns a
    list of these.
    """

    audio_file_id: int
    tts_id: str = Field(description="Engine that ran: 'hamsa-tts' | 'inception-tts'")
    tts_name: str = Field(description="Display name of the TTS engine")
    status: str = Field(description="done|failed")
    error: str | None = Field(default=None, description="Failure reason when status == 'failed'")
    voice: str | None = Field(default=None, description="The voice used for this synthesis")
    delivery: TtsDelivery
    text_chars: int | None = Field(default=None, ge=0, description="Length of the text synthesized")
    audio_format: str | None = Field(default=None, description="wav|mp3; None when status == 'failed'")
    size_bytes: int | None = Field(default=None, ge=0, description="Exact size of the rendered audio")
    native_sample_rate: int | None = Field(default=None, ge=0, description="Measured from the rendered audio")
    audio_sec: float | None = Field(default=None, ge=0, description="Measured duration of the rendered audio")
    channels: int | None = Field(default=None, ge=0, description="Read from the rendered container")
    bit_depth: int | None = Field(
        default=None, ge=0, description="Read from the rendered container; None when it carries none (MP3)"
    )
    first_audio_ms: int | None = Field(default=None, ge=0, description="Measured time to first audio")
    synth_ms: int | None = Field(default=None, ge=0, description="Measured total synthesis wall-clock time")
    rtf: float | None = Field(
        default=None, ge=0, description="synth_ms / audio_sec; None when audio_sec is unknown"
    )


class TtsRawOutput(ContractModel):
    """`ModelRawOutput`'s counterpart for one TTS engine's synthesis.

    `raw_output` here is response metadata (status code, headers), never the
    audio bytes -- the audio itself is served by `GET
    /evaluations/{id}/tts/{ttsId}/audio`.
    """

    tts_id: str = Field(description="Engine that ran: 'hamsa-tts' | 'inception-tts'")
    status: str = Field(description="done|failed")
    raw_output: dict | None = Field(default=None, description="The engine's response metadata, verbatim")
    run: TtsRun | None = Field(default=None, description="The adapted contract built from that run")


class QueuedModel(ContractModel):
    """One model's initial state right after being enqueued."""

    id: str
    status: ModelStatus


class UploadAck(ContractModel):
    """Immediate response to `POST /upload` — the client then polls
    `GET /evaluations/{audioFileId}` for the real result."""

    audio_file_id: int
    models: list[QueuedModel]


class UploadTimingUpdate(ContractModel):
    """Body of `PATCH /evaluations/{audioFileId}` — the client's one-time
    measured upload time, persisted so it survives reopening the project."""

    upload_ms: int = Field(ge=0)


class ModelMetadata(ContractModel):
    """One entry in the `GET /models` registry — what a model IS, not a run's output."""

    id: str
    name: str
    short: str
    description: str
    available: bool = Field(description="False for engines that are pluggable stubs (report failed, never fake segments)")


class ModelContainerStatus(ContractModel):
    """One entry in `GET /models/status` — a model's real-time GPU-residency
    lifecycle state, platform-wide (shared across every evaluation, not
    scoped to one upload), derived fresh from Docker + RQ at request time
    rather than read from a stored mirror. Models with no container to
    manage (pyannote runs in-process) have no entry — never fabricate a
    state for them."""

    model_id: str
    state: ModelLifecycleState
    active_job_count: int = Field(ge=0, description="In-flight jobs on this model; >0 means untouchable by eviction/idle-unload")
    queued_job_count: int = Field(ge=0, description="Jobs waiting on this model because the residency cap is full")
    last_error: str | None = Field(default=None, description="Most recent failure reason, if any (e.g. a failed cold start)")


def normalize_model_run(run: DiarizationModelRun) -> DiarizationModelRun:
    """Enforce the contract's invariants (sorted segments, correct num_spk).

    Every worker-side adapter should pass its output through this before the
    result is persisted or served, mirroring `normalizeModelRun()` on the
    frontend side. This may only sort and recount — never merge or drop a
    segment a model actually produced.
    """
    segs = sorted(run.segs, key=lambda seg: seg.s)
    num_spk = max((seg.spk for seg in segs), default=-1) + 1
    return run.model_copy(update={"segs": segs, "num_spk": num_spk})
