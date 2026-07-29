"""Single source of truth for all runtime configuration.

Every service (API, worker, tests) imports `get_settings()` instead of
reading `os.environ` directly. Values are loaded from `.env` at the repo
root; nothing here is a substitute for setting a real value in `.env` —
these are dev-friendly defaults only, always overridable.

Moving to a new environment (staging, production, DGX Spark) means editing
`.env` only; no code changes.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database (Postgres) ---
    database_url: str = "postgresql+psycopg://postgres:pg123456@localhost:5432/diarization"

    # --- Queue (Redis / RQ) ---
    redis_url: str = "redis://localhost:6379/0"
    worker_concurrency: int = 1
    # RQ's own library default is 180s, which is shorter than several
    # per-model timeouts below (azure_batch_job_timeout_sec,
    # nemo_clustering_timeout_sec at 1800s) — those would never get a chance
    # to fire. This must stay >= the largest per-model timeout... AND, since
    # the supervisor (see "GPU container lifecycle" below) can now make a
    # job wait out a container cold-start before its own inference timeout
    # even starts counting, it must stay >= the largest (cold-start +
    # inference) SUM across managed models, not just the largest single
    # timeout. Worst case today: vibevoice at
    # vibevoice_cold_start_timeout_sec (600) + vibevoice_timeout_sec (5400)
    # = 6000 is the floor; padded here for margin.
    queue_job_timeout_sec: int = 6600

    # --- Primary storage lane: MinIO / S3 (local models) ---
    # Port 9010: on DGX Spark hosts running the Parakeet NIM containers, host
    # port 9000 is owned by parakeet-nim-str's HTTP API.
    s3_endpoint_url: str | None = "http://localhost:9010"
    s3_bucket: str = "diarization-audio"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # --- Azure Speech (real-time + batch diarization) ---
    azure_speech_key: str | None = None
    azure_speech_region: str | None = None
    azure_speech_endpoint: str | None = None
    azure_realtime_transcribe_timeout_sec: int = 600
    azure_batch_poll_interval_sec: int = 10
    azure_batch_job_timeout_sec: int = 1800
    azure_batch_time_to_live: str = "PT4H"
    azure_batch_max_speakers: int = 8
    locale: str = "en-US"

    # --- External recording API (S3-backed meeting-recording source) ---
    # The eval backend pulls audio from a recording API that streams raw bytes
    # behind a Bearer token (production: ADEO; local dev: tools/recording_api on
    # :8215). A token pasted with the URL (as a curl -H line) wins; this is the
    # fallback used when the pasted input carries none. The local replica also
    # enforces this exact token, so seeding and ingestion match by construction.
    recording_api_token: str | None = None
    recording_fetch_timeout_sec: int = 600

    # --- Azure lane storage: Blob (only touched when the Azure model is on) ---
    azure_storage_account_name: str | None = None
    azure_storage_account_key: str | None = None
    azure_storage_container_name: str = "diarization-audio"
    azure_storage_sas_read_expiry_hours: int = 48

    # --- API ---
    api_port: int = 8010
    cors_allow_origins: str = "http://localhost:5173"

    # --- Frontend (served via GET /config; no rebuild needed for this value) ---
    frontend_poll_interval_ms: int = 1500

    # --- Local diarization models (DGX Spark: cuda; dev machine: cpu) ---
    diarization_device: str = "cpu"
    # Required by pyannote community-1, a gated HuggingFace model. Accept its
    # conditions at https://huggingface.co/pyannote/speaker-diarization-community-1
    # then set this to a HuggingFace access token.
    huggingface_token: str | None = None
    # Directory holding sherpa-onnx's segmentation.onnx and embedding.onnx;
    # downloaded here on first use if missing (ungated, no token needed).
    sherpa_model_dir: str = "~/.cache/sherpa-onnx-diarization"

    # --- NVIDIA Parakeet-Sortformer NIM (DGX Spark local Triton/Riva containers) ---
    # Started via ./deploy/parakeet_nim_up.sh; gRPC only (speaker tags are not
    # exposed by the plain HTTP /v1/audio/transcriptions route).
    nim_str_grpc: str = "localhost:50051"
    nim_ofl_grpc: str = "localhost:50052"
    # HTTP health-check port only (used by the GPU supervisor's
    # /v1/health/ready poll — inference itself only ever uses the gRPC ports
    # above). Must match deploy/parakeet/parakeet-nim.env's
    # PARAKEET_NIM_HTTP_PORT / PARAKEET_NIM_OFL_HTTP_PORT — change both
    # together if 9000/9001 are taken by something else on the host.
    nim_str_health_port: int = 9000
    nim_ofl_health_port: int = 9001
    nim_language: str = "multi"
    nim_max_speakers: int = 8
    nim_grpc_timeout_sec: int = 600
    # Separate from nim_grpc_timeout_sec (which bounds one ASR call): this is
    # the GPU supervisor's cold-start budget while waiting for the
    # container's own /v1/health/ready to pass after a `docker start`.
    nim_cold_start_timeout_sec: int = 1800

    # --- NeMo Clustering Diarizer (custom container, no turnkey NIM ships this) ---
    # Started via ./deploy/nemo-clustering/nemo_clustering_up.sh; cascaded
    # MarbleNet VAD + TitaNet + spectral clustering, for meetings with more
    # speakers than the Sortformer NIMs above are tuned for.
    nemo_clustering_url: str = "http://localhost:9020"
    nemo_clustering_timeout_sec: int = 1800

    # --- 3D-Speaker CAM++ Clustering Diarizer (custom container, no turnkey NIM) ---
    # Started via ./deploy/3d-speaker-clustering/3d_speaker_clustering_up.sh;
    # ASR-free FSMN VAD + CAM++ speaker embeddings + clustering, no
    # transcription step.
    speaker3d_clustering_url: str = "http://localhost:9021"
    speaker3d_clustering_timeout_sec: int = 600
    # Budget for the container's /health/ready to pass after a `docker start`;
    # the first start downloads CAM++/FSMN-VAD checkpoints from ModelScope, so
    # this is separate from (and larger than) the per-request timeout above.
    # Mirrors nim_cold_start_timeout_sec's pattern.
    speaker3d_clustering_cold_start_timeout_sec: int = 1800

    # --- DiariZen (custom container, no turnkey NIM) ---
    # Started via ./deploy/diarizen/diarizen_up.sh; WavLM-Large + Conformer
    # local end-to-end diarization followed by global clustering. Pretrained
    # weights are CC BY-NC 4.0 (non-commercial/research use only).
    diarizen_url: str = "http://localhost:9022"
    diarizen_timeout_sec: int = 600

    # --- VibeVoice-ASR (custom container, no turnkey NIM) ---
    # Started via ./deploy/vibevoice/vibevoice_up.sh; Microsoft's 8B
    # decoder-only model doing ASR + diarization + timestamping in one
    # autoregressive pass. Autoregressive decoding over long audio is slow
    # (it generates a token per output word, not a fixed-cost forward pass),
    # hence the wide timeout -- a 32-minute file genuinely needs more than
    # 30 minutes of decode time (observed live: cut off at exactly 1800s),
    # so this must stay under queue_job_timeout_sec but well above realtime.
    # The timeout applies to BOTH the local container call and the remote
    # proxy call below.
    vibevoice_url: str = "http://localhost:9023"
    vibevoice_timeout_sec: int = 5400
    # Cold-start budget for the local container (health-ready wait after
    # `docker start`), separate from the inference timeout above -- measured
    # ~2 minutes live; mirrors nim_cold_start_timeout_sec's pattern.
    vibevoice_cold_start_timeout_sec: int = 600
    # Optional remote inference proxy (OpenAI-style chat-completions URL).
    # When set, the vibevoice runner calls this endpoint instead of the
    # local container, and vibevoice is EXCLUDED from the GPU residency
    # cap/supervisor entirely -- no docker start/stop, no local GPU slot,
    # exactly like the cloud-lane models. Leave unset to run locally.
    vibevoice_baseurl: str | None = None
    vibevoice_api_key: str | None = None
    # TLS verification for the remote proxy call above only. Defaults to
    # secure; set false only if the proxy sits behind a cert this host
    # doesn't trust (internal CA / self-signed).
    vibevoice_ssl_verify: bool = True

    # --- MOSS-Transcribe-Diarize (stock vLLM image; no custom container) ---
    # Started via ./deploy/moss-transcribe/moss_transcribe_up.sh; OpenMOSS's
    # 0.9B end-to-end model doing ASR + diarization + timestamping in one
    # autoregressive pass, served on vLLM's OpenAI-compatible transcription API.
    #
    # Same wide timeout as vibevoice, and for the same reason: being small does
    # NOT make it quick on long audio. Decode is autoregressive over the whole
    # transcript, so cost tracks output length, not model size. Measured on this
    # host: ~32 tokens/s generation, and a 32-minute file emits ~24k tokens =
    # ~13 minutes of decode (an RTF of ~0.4, not the ~0.06 a 30-second clip
    # suggests -- that figure is prefill-dominated and does not extrapolate).
    # At MOSS's ~90-minute ceiling that is ~2100s, so an earlier 1800s value
    # here would have cut off exactly the files this model exists to handle.
    # Stays under queue_job_timeout_sec (600 cold start + 5400 = 6000 <= 6600).
    moss_transcribe_url: str = "http://localhost:9024"
    moss_transcribe_timeout_sec: int = 5400
    # Cold-start budget for the container (health-ready wait after `docker
    # start`), separate from the inference timeout above. Only ~2GB of weights
    # to load, but vLLM's engine start + CUDA-graph capture dominates, so this
    # mirrors vibevoice_cold_start_timeout_sec rather than being scaled down
    # with the weights.
    moss_transcribe_cold_start_timeout_sec: int = 600

    # --- Live-speech transcription (the Live Speech panel) ---
    # Two ASR engines are available; the mode is chosen PER RUN from the panel,
    # not by a setting:
    #   online  -> TryHamsa STT over WSS. Audio LEAVES this host.
    #   offline -> the local cohere-transcribe vLLM container. Audio stays here.
    # The alignment stage (ctc-forced-aligner, below) is shared by both modes.
    # The chosen engine's id is stamped on each TranscriptResult row at enqueue
    # time, so a later run in the other mode never relabels an existing
    # transcript. A host only offers a mode whose credentials/endpoint are set.

    # --- TryHamsa STT (online mode) ---
    # A streaming WebSocket API, not a request/response one: audio is paced to
    # the server in 100ms chunks so its VAD can segment, which makes ASR
    # wall-clock a function of the recording's LENGTH (~50% of it), not of
    # model speed. hamsa_stt_chunk_sleep_sec is that pace.
    # HAMSA_STT_URL wins over HAMSA_STT_WS_URL when both are set (see
    # hamsa_ws_endpoint); both names exist because the reference client
    # accepted either.
    hamsa_stt_url: str | None = None
    hamsa_stt_ws_url: str | None = None
    hamsa_stt_key: str | None = None
    hamsa_stt_bearer_token: str | None = None
    # Per-service, mirroring vibevoice_ssl_verify -- deliberately NOT the bare
    # SSL_VERIFY some clients use, so turning verification off for this one
    # endpoint can never silently weaken another service's TLS.
    hamsa_stt_ssl_verify: bool = True
    hamsa_stt_sample_rate: int = 16000
    hamsa_stt_chunk_sleep_sec: float = 0.05
    # How long the socket may sit silent after the audio is sent before the
    # transcript is considered complete. Hamsa emits one message per detected
    # speech segment and gives no explicit end-of-stream signal.
    hamsa_stt_idle_timeout_sec: float = 5.0
    # Hard ceiling on one streaming session, so a hung socket fails on its own
    # instead of burning the whole queue_job_timeout_sec. Must comfortably
    # exceed 0.5x the longest recording: a 2-hour file streams for ~1 hour.
    hamsa_stt_session_timeout_sec: int = 5400

    # --- Cohere Transcribe Arabic (offline mode) ---
    # CohereLabs/cohere-transcribe-arabic-07-2026, a 2B Arabic/English ASR
    # model served on vLLM's OpenAI-compatible transcription API. Started via
    # ./deploy/cohere-transcribe/cohere_transcribe_up.sh.
    #
    # Deliberately NOT GPU-supervisor-managed: it consumes no residency slot,
    # is never evicted or idle-unloaded, and is expected to stay up once
    # started. Its footprint is bounded by the vLLM flags in its compose file
    # instead of by the cap.
    cohere_transcribe_url: str = "http://localhost:9025"
    cohere_transcribe_timeout_sec: int = 1800
    # Empty = auto-detect (correct for both English and Arabic). A non-empty
    # value FORCES that language as a decoder prompt the model obeys over the
    # audio, so "ar" makes English recordings hallucinate Arabic and "en" would
    # break Arabic ones. Only set it to force a known single-language batch.
    cohere_transcribe_language: str = ""

    # --- CTC forced aligner (in-process in the worker; both modes) ---
    # MahmoudAshraf97/ctc-forced-aligner's MMS-300m model, which romanizes
    # text before alignment and so handles code-switched audio. Runs on
    # diarization_device (cuda here), loaded once per worker process.
    ctc_aligner_language: str = "ara"
    ctc_aligner_batch_size: int = 4

    # --- GPU container lifecycle supervisor (DGX Spark residency cap) ---
    # Hard cap on how many of the local-lane model containers above may be
    # GPU-resident (started) at once. A request for a model beyond this cap
    # waits (re-enqueued) instead of proceeding or erroring. DGX Spark has
    # ~120GB unified memory and no MIG/MPS isolation; running all of them at
    # once caused OOM kills of unrelated infra containers — start
    # conservative and raise only after confirming headroom.
    max_resident_models: int = 2
    # How long a model may sit idle (zero active/queued jobs) before the
    # supervisor daemon stops its container to free the slot. An idle model
    # can still be evicted sooner than this if a competing request needs
    # the slot right away — this setting only governs unforced cleanup.
    idle_unload_timeout_sec: int = 600
    # Delay before a job denied a GPU slot (cap already full) is re-enqueued
    # to try again.
    admission_requeue_delay_sec: int = 10
    # Minimum time between retry attempts against a model stuck "unhealthy"
    # (its last cold-start attempt failed) — without this, a burst of
    # requests queued behind a genuinely broken container would hammer
    # `docker start` on every retry cycle.
    unhealthy_retry_backoff_sec: int = 120
    # apps/background_worker/supervisor/daemon.py sweep cadence: idle-unload
    # and staleness checks.
    supervisor_sweep_interval_sec: int = 15
    # Grace period passed to `docker stop --time` before Docker SIGKILLs a
    # container that hasn't exited on its own.
    container_stop_grace_sec: int = 30
    # Added on top of a model's own cold-start budget (reused from its
    # *_timeout_sec field above) as a floor for detecting a "starting" row
    # stuck past its deadline — covers the case where the supervisor daemon
    # itself was down when the deadline passed and only catches up later.
    supervisor_stale_grace_sec: int = 60

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def hamsa_ws_endpoint(self) -> str | None:
        """The Hamsa WebSocket URL, from either accepted setting name."""
        return self.hamsa_stt_url or self.hamsa_stt_ws_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
