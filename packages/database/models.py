"""SQLAlchemy ORM tables.

EvaluationResult.payload stores the unified contract
(packages/shared_contracts DiarizationModelRun) as JSON. Alongside it,
`raw_output` keeps the engine's own native output verbatim — stored as an
opaque blob and served only by the dedicated raw routes, never parsed here.
TranscriptResult carries the same pair for the live-speech ASR engines.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    audio_files: Mapped[list["AudioFile"]] = relationship(back_populates="owner")


class AudioFile(Base):
    __tablename__ = "audio_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    # Each lane owns its own location; a row only has the key(s) for the
    # lane(s) actually used — never both copied from one store to the other.
    s3_key: Mapped[str | None] = mapped_column(String(1024), unique=True, nullable=True)
    blob_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    blob_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float)
    # Which evaluation surface produced this recording: a diarization upload, or a
    # read-aloud capture from the transcript surface. Indexed because every listing
    # query filters on it.
    #
    # A stored column rather than a derived signal. The alternatives were all
    # accidents of implementation rather than statements of intent: guessing from
    # `filename`, or from the absence of EvaluationResult rows, or from the presence
    # of a TranscriptReference. Those coincide with the truth today and would drift
    # the first time someone diarizes a read-aloud recording.
    surface: Mapped[str] = mapped_column(
        String(16), default="diarization", index=True
    )  # diarization|transcript
    # Client-perceived upload time (browser upload start -> API response),
    # persisted via PATCH once the client has measured it.
    upload_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owner: Mapped[User] = relationship(back_populates="audio_files")
    results: Mapped[list["EvaluationResult"]] = relationship(back_populates="audio_file")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audio_file_id: Mapped[int] = mapped_column(ForeignKey("audio_files.id"), index=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # DiarizationModelRun JSON
    # The model's NATIVE output, exactly as its runner returned it -- never
    # parsed or reshaped here. Carries everything the adapter drops (transcript
    # text, per-word timings, confidences, the engine's own speaker labels).
    # dict | list because the pyannote and sherpa runners return lists.
    #
    # deferred: the only multi-MB column on this table, read by exactly one
    # route, while GET /evaluations/{id} re-SELECTs every row of it every
    # FRONTEND_POLL_INTERVAL_MS per in-flight recording. Undeferred, the ORM
    # would pull each blob out of TOAST on every one of those polls to build a
    # response that never contains it.
    raw_output: Mapped[dict | list | None] = mapped_column(JSON, nullable=True, deferred=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Worker-measured timing: started/finished bracket the run itself
    # (excludes queue wait); processing_ms = finished - started.
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Set when the supervisor grants a GPU slot and begins waiting for the
    # model's container to report healthy (see apps/background_worker/
    # supervisor/). started_at is stamped only once the container is
    # confirmed ready and inference actually begins, so
    # processing_ms = finished_at - started_at never includes cold-start
    # wait -- that interval is [loading_started_at, started_at) instead.
    loading_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    audio_file: Mapped[AudioFile] = relationship(back_populates="results")


class TranscriptResult(Base):
    """One live-speech transcript per (audio file, ASR engine): ASR text plus
    the forced aligner's word timings.

    Separate from `EvaluationResult` rather than another row there, because
    this is transcription, not diarization: it carries text (which the
    diarization contract has no field for), it has no speaker segments, and
    its two stages are timed independently instead of sharing one
    `processing_ms`.

    Keyed per engine, the same shape `EvaluationResult` uses per model: this is
    a comparison tool, so an online and an offline transcript of the same
    recording must coexist. Re-running one engine resets its own row and leaves
    the other engine's result untouched.
    """

    __tablename__ = "transcript_results"
    __table_args__ = (
        # Keyed on the FEED MODE as well as the engine, so one recording can hold
        # both an engine's live measurement and its batch one. It could not
        # before: a batch re-run reset the single row and the read-aloud numbers
        # were gone, which made "compare this engine's two modes" impossible on
        # the surface built to compare things.
        #
        # `source` is NOT NULL (it has a default), and that is load-bearing the
        # same way `tts_results.voice` is: Postgres treats NULLs as distinct
        # inside a UNIQUE, so a nullable column here would let duplicates through.
        UniqueConstraint(
            "audio_file_id", "asr_id", "source",
            name="uq_transcript_results_audio_file_asr_source",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audio_file_id: Mapped[int] = mapped_column(ForeignKey("audio_files.id"), index=True)
    # The engine that ACTUALLY produced this transcript ("hamsa" |
    # "cohere-transcribe"), stamped from the mode chosen for this run when the
    # job is enqueued. Half the row's identity, not a label: a run in the other
    # mode writes its OWN row and can never relabel this one. The UI's mode
    # indicator reads this column.
    asr_id: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    # Which of the two stages is live, or the one that failed; cleared on
    # success. Not a status in its own right -- `status` stays the single
    # queued/running/done/failed vocabulary the rest of the platform uses.
    stage: Mapped[str | None] = mapped_column(String(16), nullable=True)  # asr|align
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    # Written when the ASR stage finishes, so the UI can show the transcript
    # while alignment is still running. Text, not String: a 2-hour recording's
    # transcript has no useful length bound.
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # list[TranscriptWord] JSON; written when the alignment stage finishes.
    words: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # The ASR engine's NATIVE output, verbatim -- hamsa's whole WebSocket frame
    # log (a list) or cohere's bare response JSON (a dict), both of which its
    # adapter reduces to the single `text` string above. The ALIGNER's native
    # output is not kept: `words` already carries its per-word timings.
    #
    # deferred for the same reason as EvaluationResult.raw_output: the
    # transcript list is polled on its own timer while a run is in flight.
    raw_output: Mapped[dict | list | None] = mapped_column(JSON, nullable=True, deferred=True)
    asr_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    align_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- How this transcript was produced (the transcript-evaluation surface) ---
    # "live"  — captured chunk by chunk while someone read a script aloud.
    # "batch" — run over audio already in storage.
    # Stored, not derived: a live and a batch number are different measurements
    # of the same engine, and a scorecard that mixed them silently would be
    # comparing two things while claiming to compare one.
    source: Mapped[str] = mapped_column(String(8), default="batch")  # live|batch
    # A "live" row can be produced two ways, and they are not the same
    # measurement: someone actually read a script aloud (False), or stored audio
    # was cut and replayed through the live chunk route (True). The transcript is
    # comparable either way; the LATENCIES are not, because a replay is paced by
    # the loop rather than by speech. Stored rather than derived, and labelled
    # everywhere the timing figures appear, so the two never merge silently.
    # Always False for a batch row, which is not replayed of anything.
    replayed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    # What the audio actually travelled over, for the engine that produced this
    # row: "stream" (continuous, engine-side VAD) or "chunks" (fixed cuts).
    # The two are NOT interchangeable and the UI labels each figure with it —
    # a chunked engine carries its boundary cost in its own error rate.
    transport: Mapped[str | None] = mapped_column(String(16), nullable=True)  # stream|chunks
    chunk_interval_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Latency of the FIRST chunk and the mean across all of them, measured
    # send -> text-returned. Scalars rather than only the list below, because
    # these two are what the panel shows and the transcript list is polled
    # while a run is in flight.
    first_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Every chunk's measured latency, in order. Deferred: a 30-minute recording
    # at a 3s interval is 600 entries that no polled response needs.
    chunk_latencies_ms: Mapped[list | None] = mapped_column(JSON, nullable=True, deferred=True)
    # Processing time / audio duration. NULL means "not measurable for this
    # transport" rather than zero: a real-time streaming protocol consumes audio
    # at 1x by definition, so an RTF for it would be an invention. The UI shows
    # "real-time bound" for a null, never a number.
    rtf: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Scoring against the reference (see TranscriptReference) ---
    # Both normalized and raw are kept so the UI's "normalization on" toggle is
    # a read, not a recompute — and so the effect of normalization is itself
    # visible instead of being an invisible preprocessing step.
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)
    wer_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    cer_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    ref_word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hyp_word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sub_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    del_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ins_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The per-word edit operations the UI highlights. Deferred for the same
    # reason as chunk_latencies_ms: one entry per reference word.
    alignment: Mapped[list | None] = mapped_column(JSON, nullable=True, deferred=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    asr_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    align_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    audio_file: Mapped[AudioFile] = relationship()


class TranscriptReference(Base):
    """The canonical text of this recording: the ground truth one recording is
    scored against, whether read by a person or synthesized. Read-aloud and
    stt-comparison flows treat the text as ground truth to be matched; the TTS
    comparison flow treats it as input, the words meant to be spoken. At most
    one per recording either way.

    A separate table rather than a column on AudioFile, because a reference is
    not a property of the audio — it is a claim about what was said, with its own
    provenance. `source` records that provenance and is load-bearing: a WER
    against a script someone actually read aloud means something different from a
    WER against a transcript typed after the fact, and the scorecard says which.

    One per recording, not one per engine: every engine is scored against the
    same text, which is the only way their numbers are comparable.
    """

    __tablename__ = "transcript_references"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audio_file_id: Mapped[int] = mapped_column(
        ForeignKey("audio_files.id"), index=True, unique=True
    )
    # "script" — generated by the script generator and read aloud, so the words
    #            were known before the audio existed.
    # "pasted" — supplied by hand for a recording that already existed.
    source: Mapped[str] = mapped_column(String(8))  # script|pasted
    text: Mapped[str] = mapped_column(Text)
    # For a generated script: the request that produced it (minutes, language
    # mix, hard cases) plus the generating model. Kept so a score can be traced
    # back to what was asked for, and so a script can be regenerated like-for-like.
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    audio_file: Mapped[AudioFile] = relationship()


class TtsResult(Base):
    """One TTS engine's synthesis of one recording's reference text into audio.

    Keyed per (audio_file, tts_id), the same shape `TranscriptResult` uses per
    ASR engine: a comparison tool, so hamsa-tts and inception-tts synthesizing
    the same script must coexist, and re-running one engine resets only its own
    row. Hangs off the SAME `AudioFile`/`TranscriptReference` a read-aloud run
    would use -- there is no separate TTS "recording"; the reference text is
    shared between "read this aloud" and "synthesize this".

    `status` ("done" | "failed") exists for the same reason every other result
    table has one: without it, a failed engine leaves no row at all, and "never
    tried" becomes indistinguishable from "failed" -- so a failed synthesis
    still writes a row, with `error` set and the audio-shaped columns null.

    `raw_output` here is metadata only (status code, response headers) -- never
    the audio bytes. Base64ing minutes of audio into a JSON column would be a
    multi-MB unreadable duplicate of the object store, so the audio itself lives
    at `s3_key` and this column stays small enough to leave un-deferred (unlike
    `TranscriptResult.raw_output`, which holds a whole transcript/frame log and
    is polled while a run is in flight -- nothing polls this table).

    This is a brand new table, so `create_all` builds its current shape with no
    entry in `_ADDED_COLUMNS` (session.py) needed. That is only true at
    creation: a column added to this table LATER would need one, exactly like
    every other table here -- `test_added_columns_shim_matches_the_models`
    checks the direction that would otherwise fail silently, but cannot catch a
    forgotten shim entry for a column that already exists on every fresh DB.
    """

    __tablename__ = "tts_results"
    # One clip per (recording, engine, VOICE): each voice keeps its own stored
    # take, so switching voice in the UI reveals that voice's clip instead of
    # overwriting the last one. Re-running the SAME voice still replaces in
    # place -- that is the get-or-reset path in routers/tts.py.
    __table_args__ = (
        UniqueConstraint(
            "audio_file_id", "tts_id", "voice", name="uq_tts_results_audio_file_tts_voice"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audio_file_id: Mapped[int] = mapped_column(ForeignKey("audio_files.id"), index=True)
    # "hamsa-tts" | "inception-tts" -- half the row's identity, the other half
    # being audio_file_id; see the unique constraint above.
    tts_id: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # done|failed
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # --- what was asked for ---
    # NOT NULL is load-bearing, not tidiness: Postgres treats NULLs as distinct
    # inside a UNIQUE, so a null voice here would let unlimited duplicate rows
    # through uq_tts_results_audio_file_tts_voice.
    voice: Mapped[str] = mapped_column(String(64), nullable=False)
    # "stream" | "single", copied from TTS_DELIVERY (apps/background_worker/tts)
    # at write time by the route -- not imported here, since packages/database
    # importing apps/background_worker would be a layering violation.
    delivery: Mapped[str] = mapped_column(String(16))
    text_chars: Mapped[int] = mapped_column(Integer)

    # --- what came back, only present when status == "done" ---
    s3_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    audio_format: Mapped[str | None] = mapped_column(String(16), nullable=True)  # wav|mp3
    # Exact payload size in bytes. Named size_bytes, not bytes: `bytes` shadows
    # the builtin and this repo's other tables spell out what a count measures
    # (e.g. TranscriptResult.chunk_count) rather than leaving it to the type.
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    native_sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Read out of the returned container alongside the rate, not assumed from
    # the platform's canonical shape: these are what the ENGINE chose to emit,
    # and the two engines do not agree. Null when the container does not say
    # (MP3 carries no fixed sample width), never defaulted to 1/16.
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bit_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_audio_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    synth_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # (synth_ms/1000) / audio_sec; None when audio_sec is unknown (e.g. a
    # failed run), never a fabricated ratio.
    rtf: Mapped[float | None] = mapped_column(Float, nullable=True)
    # The engine's response metadata, verbatim (status code, headers) -- never
    # the audio itself. See the class docstring for why this is not deferred.
    raw_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    audio_file: Mapped[AudioFile] = relationship()


class ModelContainerState(Base):
    """One row per GPU container-managed local model (see
    apps/background_worker/supervisor/registry.py MANAGED_CONTAINERS).
    pyannote has no row: it runs in-process in the worker, not in a
    container the supervisor can start/stop.

    Deliberately stores ONLY what no other system can know — job accounting
    and in-flight claims. Everything else (container running/healthy, queue
    depth, the UI's lifecycle state) is derived at read time from its owner
    (Docker, RQ) by apps/background_worker/supervisor/state.py. An earlier
    version mirrored a full lifecycle_state machine here; every live bug it
    produced was mirror-vs-reality drift, repaired by an ever-growing set of
    daemon sweeps. Claims carry timestamps and their validity is evaluated
    when read (expired or worker-dead => treated as absent), so they cannot
    orphan and need no repair sweep.
    """

    __tablename__ = "model_container_state"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    active_job_count: Mapped[int] = mapped_column(Integer, default=0)  # >0 => untouchable by eviction/idle-unload
    # Cold-start claim: set when admission grants a slot, cleared atomically
    # by mark_job_started (claim converts to active_job_count) or
    # mark_unhealthy. Valid only while young enough AND a live busy worker
    # is actually processing a job for this model — see state.py.
    starting_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Eviction claim: set by the evicting process just before `docker stop`,
    # cleared right after. TTL-expires by interpretation if the evictor dies
    # mid-stop (docker stop is idempotent, so a successor can just re-stop).
    evicting_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_job_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_unhealthy_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
