"""Consistency checks across the three hand-maintained per-model tables.

Adding a model touches REGISTRY (models/__init__.py), LANE_MAP (lanes.py)
and, for containerized models, the supervisor registry. Each table forgotten
fails silently at runtime (dropped at upload, or run in-process without a
container); these tests turn every mismatch into a loud pytest failure.
"""

from types import SimpleNamespace

from apps.background_worker.lanes import LANE_MAP
from apps.background_worker.models import REGISTRY
from apps.background_worker.supervisor.registry import _registry
from packages.config.settings import get_settings


def test_registry_and_lane_map_list_the_same_models() -> None:
    missing_lane = set(REGISTRY) - set(LANE_MAP)
    missing_registry = set(LANE_MAP) - set(REGISTRY)
    assert not missing_lane, f"In REGISTRY but not LANE_MAP (upload will reject them): {sorted(missing_lane)}"
    assert not missing_registry, f"In LANE_MAP but not REGISTRY (jobs will fail 'Unknown model id'): {sorted(missing_registry)}"


def test_every_managed_container_model_is_a_registered_local_model() -> None:
    for model_id in _registry():
        assert model_id in REGISTRY, f"Supervisor manages {model_id!r} but it is not in REGISTRY"
        assert LANE_MAP.get(model_id) == "local", f"Supervisor-managed {model_id!r} must be in the local lane"


def test_queue_job_timeout_exceeds_every_cold_start_timeout() -> None:
    """Necessary condition only: the full invariant is queue_job_timeout_sec >=
    max(cold-start + inference) across models, but inference timeouts have no
    uniform settings shape — checking that part stays a manual step (see the
    CLAUDE.md recipe)."""
    settings = get_settings()
    for container in _registry().values():
        assert settings.queue_job_timeout_sec > container.cold_start_timeout_sec, (
            f"QUEUE_JOB_TIMEOUT_SEC ({settings.queue_job_timeout_sec}) must exceed "
            f"{container.model_id!r}'s cold-start timeout ({container.cold_start_timeout_sec}) "
            "or RQ kills the job before inference even starts"
        )


# --- ASR engines (the transcription subsystem) ------------------------------
#
# `ASR_ENGINES` is a second registry with its own parallel maps, and the same
# class of mistake is available: an engine registered but given no transport, or
# a transport declared for an engine that does not exist. The diarization
# registry has been cross-checked since it grew a third map; this does the same
# for the transcription one.

def test_every_asr_engine_declares_a_transport() -> None:
    """Transport is not derivable from `mode` — hamsa and inception-stt are BOTH
    online, yet one streams continuously and the other only takes short chunks.
    Every reported figure is labelled with it, so a missing entry means a run
    rendered with no transport at all."""
    from apps.background_worker.transcription import ASR_ENGINES, ASR_TRANSPORTS

    for asr_id in ASR_ENGINES:
        assert asr_id in ASR_TRANSPORTS, f"{asr_id!r} is registered but declares no transport"
        assert ASR_TRANSPORTS[asr_id] in ("stream", "chunks", "file")


def test_an_engine_can_be_chunked_exactly_when_its_live_transport_says_so() -> None:
    """The chunk route dispatches on `run_chunk`, and the transport label is what
    every surface prints beside the figure. If the two disagree, one of them is
    lying about how the audio reached the engine.

    Fails against the version of `POST /transcript/chunk` that called the
    inception runner unconditionally: there, an engine could be labelled "chunks"
    with no chunk entry point of its own and still return 200 OK, storing
    Inception's text and latency under its own name."""
    from apps.background_worker.transcription import ASR_ENGINES, LIVE_TRANSPORTS

    for asr_id, engine in ASR_ENGINES.items():
        chunkable = engine.run_chunk is not None and engine.chunk_text is not None
        assert chunkable == (LIVE_TRANSPORTS[asr_id] == "chunks"), (
            f"{asr_id!r} has a chunk entry point={chunkable} but its live transport is "
            f"{LIVE_TRANSPORTS[asr_id]!r}; the live chunk route dispatches on the entry "
            "point and every surface prints the transport, so the two must agree"
        )


def test_every_engine_declares_a_feed_mode_and_a_transport_for_each() -> None:
    """An engine with no modes cannot be put in a session at all, and one whose
    mode maps to no transport would render a run with a blank transport label."""
    from apps.background_worker.transcription import (
        ASR_ENGINES,
        ASR_FEED_MODES,
        ASR_TRANSPORTS,
        LIVE_TRANSPORTS,
        transport_for,
        transport_for_mode,
    )

    for asr_id in ASR_ENGINES:
        modes = ASR_FEED_MODES.get(asr_id)
        assert modes, f"{asr_id!r} declares no feed mode"
        assert modes[0] == "live", f"{asr_id!r} must default to live, not {modes[0]!r}"
        for mode in modes:
            assert mode in ("live", "batch")
            assert transport_for_mode(asr_id, mode) in ("stream", "chunks", "file"), (
                f"{asr_id!r} in {mode!r} maps to no known transport"
            )
        # The batch half must agree with what `run_asr` genuinely does, which is
        # what `transport_for` resolves -- NOT the raw ASR_TRANSPORTS entry.
        # inception-stt's batch transport comes from INCEPTION_BATCH_WHOLE_FILE,
        # so the dict holds only one of its two values and reading it directly is
        # how a label starts describing a run that did not happen.
        assert transport_for_mode(asr_id, "batch") == transport_for(asr_id)
        assert transport_for_mode(asr_id, "live") == LIVE_TRANSPORTS[asr_id]
        assert ASR_TRANSPORTS[asr_id] in ("stream", "chunks", "file")


def test_inception_batch_transport_follows_its_setting() -> None:
    """The label and the behaviour are driven by one switch, in both positions.

    `run()` sends the whole file when INCEPTION_BATCH_WHOLE_FILE is on and splits
    when it is off; `transport_for` must say "file" and "chunks" to match. A
    static entry beside the flag is the bug this asserts against -- the row would
    be stamped "chunks" while the engine swallowed the recording whole.
    """
    import apps.background_worker.transcription as transcription
    from packages.config.settings import get_settings

    real = get_settings()
    for whole_file, expected in ((True, "file"), (False, "chunks")):
        fake = SimpleNamespace(inception_batch_whole_file=whole_file)
        original = transcription.get_settings
        transcription.get_settings = lambda: fake
        try:
            assert transcription.transport_for("inception-stt") == expected
            assert transcription.transport_for_mode("inception-stt", "batch") == expected
            # Live is unaffected either way: both engines chunk at 3s there.
            assert transcription.transport_for_mode("inception-stt", "live") == "chunks"
        finally:
            transcription.get_settings = original
    assert transcription.transport_for("inception-stt") == (
        "file" if real.inception_batch_whole_file else "chunks"
    )


def test_a_batch_mode_engine_can_actually_run_over_stored_audio() -> None:
    """Offering batch means `run_asr` will call `engine.run(path)` on the whole
    recording. An engine listed for batch with no working `run` would queue a job
    that dies in a worker instead of failing at the request."""
    from apps.background_worker.transcription import ASR_ENGINES, ASR_FEED_MODES

    for asr_id, engine in ASR_ENGINES.items():
        if "batch" in ASR_FEED_MODES[asr_id]:
            assert callable(engine.run), f"{asr_id!r} offers batch but has no run()"


def test_no_feed_mode_is_declared_for_an_unregistered_engine() -> None:
    from apps.background_worker.transcription import (
        ASR_ENGINES,
        ASR_FEED_MODES,
        LIVE_TRANSPORTS,
    )

    for asr_id in ASR_FEED_MODES:
        assert asr_id in ASR_ENGINES, f"feed modes declared for unregistered {asr_id!r}"
    for asr_id in LIVE_TRANSPORTS:
        assert asr_id in ASR_ENGINES, f"live transport declared for unregistered {asr_id!r}"


def test_no_transport_is_declared_for_an_unregistered_engine() -> None:
    from apps.background_worker.transcription import ASR_ENGINES, ASR_TRANSPORTS

    for asr_id in ASR_TRANSPORTS:
        assert asr_id in ASR_ENGINES, f"transport declared for unregistered engine {asr_id!r}"


def test_compared_engines_are_registered() -> None:
    from apps.background_worker.transcription import ASR_ENGINES, COMPARISON_ASR_IDS

    assert COMPARISON_ASR_IDS, "the comparison surface would have no columns"
    for asr_id in COMPARISON_ASR_IDS:
        assert asr_id in ASR_ENGINES, f"compared engine {asr_id!r} is not registered"


def test_every_mode_resolves_to_a_registered_engine() -> None:
    """`mode` no longer selects an engine, but the Live Speech panel still asks by
    it, so each mode must still land somewhere real."""
    from apps.background_worker.transcription import ASR_ENGINES, MODE_TO_ASR_ID

    for mode, asr_id in MODE_TO_ASR_ID.items():
        assert asr_id in ASR_ENGINES, f"mode {mode!r} resolves to unregistered {asr_id!r}"


def test_added_columns_shim_matches_the_models() -> None:
    """There is no migration tool: a column added to a deployed table only exists
    if it is listed in `_ADDED_COLUMNS`. A name there that no model has is a typo
    that would ALTER a column nothing reads — and the reverse (a model column
    missing from the list) fails on the first read against an older database.
    This catches the typo direction, which is the silent one."""
    from packages.database.models import Base
    from packages.database.session import _ADDED_COLUMNS

    # Resolved from metadata, not a hardcoded map: the previous version listed two
    # tables by hand and failed the moment a third was added, which is a test that
    # breaks on correct changes.
    tables = {name: {c.name for c in table.columns} for name, table in Base.metadata.tables.items()}
    for table, column, _type in _ADDED_COLUMNS:
        assert table in tables, f"_ADDED_COLUMNS names unknown table {table!r}"
        # The column part can carry a type/DEFAULT suffix in the DDL; the name is first.
        assert column in tables[table], (
            f"_ADDED_COLUMNS lists {table}.{column!r}, which no model declares"
        )
