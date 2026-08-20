"""Consistency checks across the three hand-maintained per-model tables.

Adding a model touches REGISTRY (models/__init__.py), LANE_MAP (lanes.py)
and, for containerized models, the supervisor registry. Each table forgotten
fails silently at runtime (dropped at upload, or run in-process without a
container); these tests turn every mismatch into a loud pytest failure.
"""

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
        assert ASR_TRANSPORTS[asr_id] in ("stream", "chunks")


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
