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


def _split_csv(raw: str) -> list[str]:
    """A comma-separated .env value as a clean list.

    Drops empty entries and strips whitespace, so a trailing comma or a stray
    space never becomes a list item. That matters beyond tidiness: a TTS
    speaker name with a trailing comma is accepted by the vendor with a 200
    and then dropped mid-stream, which looks like a network fault rather than
    the config typo it is.
    """
    return [part.strip() for part in raw.split(",") if part.strip()]


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
    # "ar", and EMPTY IS NOT A NEUTRAL VALUE HERE. Do not "clean this up" back to
    # "" -- that is the bug this default exists to prevent.
    #
    # THIS MODEL HAS NO AUTO-DETECT. Probed 2026-08-31 against the container:
    #
    #     language=auto -> 400 "Unsupported language: 'auto'. Must be one of
    #     ['en','fr','de','es','pt','it','nl','pl','el','ar','ko','ja','vi','zh']"
    #
    # Omitting the field does not mean "detect it". It makes the model settle on
    # ONE language from the content, and which one depends on the audio:
    #
    #   mostly-Arabic conversation   -> Arabic (391 chars on the committed sample)
    #   code-switched within sentences -> ENGLISH, TRANSLATING the Arabic away
    #                                   (0 chars on a real Mixed-50/50 read-aloud,
    #                                    at 3 s, 10 s, 20 s, 30 s AND 45 s)
    #
    # The second case is what this surface normally produces, and it is not a
    # short-audio effect -- no chunk size avoids it. That is the whole reason this
    # default is not "".
    #
    # "ar" is what turns the Arabic path on, and it does NOT force everything to
    # Arabic: it code-switches, leaving English as English ("...coordination in
    # مصدر city hit 94%..."). On one real read-aloud recording it moved that
    # engine from 97.2% WER (0 Arabic characters) to 15.3% (372 Arabic + 307
    # Latin), i.e. from worst of three engines to best.
    #
    # The one measured cost: on PURE-ENGLISH audio "ar" is clean whole-file (0
    # Arabic characters over 54 s) but leaks Arabic into 6 of 18 3 s chunks. That
    # is a live-chunked English-only run only, and it is the reason this is called
    # out rather than silently set.
    cohere_transcribe_language: str = "ar"

    # --- Inception-STT (via the LiteLLM gateway) ---
    # A request/response transcription gateway, OpenAI-compatible
    # (/v1/audio/transcriptions: multipart `model` + `file` + optional
    # `language`; JSON back with text/audio_duration/word_timestamps).
    #
    # Field names mirror the .env keys already in use, which came from the
    # reference client this runner was ported from -- deliberately not renamed
    # to an inception_stt_* scheme, so .env and code read the same.
    litellm_base_url: str | None = None
    litellm_api_key: str | None = None
    stt_transcription_path: str = "/v1/audio/transcriptions"
    stt_model: str = "inception-stt"
    # "" or "auto" BOTH mean auto-detect, and both omit the field from the
    # request entirely (see STT_AUTO_LANGUAGES in the runner). A real language
    # code is a decoder prompt the model obeys over the audio; on the cohere
    # engine, forcing "ar" leaks Arabic into 6 of 18 short ENGLISH chunks (it is
    # clean on whole files). Mixed ar/en audio must never force one here.
    #
    # Unlike cohere-transcribe, this gateway's omit-the-field really is
    # auto-detect rather than "emit English" -- do not carry that finding across.
    stt_default_language: str = "auto"
    stt_timeout_seconds: float = 120
    # How long a piece of audio may be in ONE call to this gateway.
    #
    # This is not a comfort setting -- it decides how much of the transcript
    # exists at all. The gateway silently drops content in a length-dependent,
    # NON-MONOTONIC way: a longer clip can return fewer words than a clip it
    # strictly contains, with a 200 and no warning. Measured 2026-08-19 over one
    # fixed 100 s span of tests/samples/youtube_ar_32min_8spk.16k.wav, splitting
    # the SAME audio at different lengths:
    #
    #     segment   calls   total words   blank segments
    #        3 s      34         219          1/34
    #        5 s      20         201          1/20
    #        8 s      13          84          8/13
    #       10 s      10          28          8/10
    #       25 s       4          85          1/4
    #      100 s       1          51          0/1
    #
    # 3 s recovers 4.3x the words of a single call and 2.6x that of 25 s
    # segments; the 8-15 s band is the worst, returning nothing at all for most
    # pieces. The extra words are real content, not boundary double-counting
    # (0.5% local trigram repeats at 3 s), and the 25 s run visibly loses the
    # opening of its own first segment.
    #
    # So this defaults into the only regime measured to be reliable, and matches
    # live_chunk_default_sec below -- the live and stored paths chunk the same
    # way, which is also what makes their numbers comparable. Raising it trades
    # transcript away for fewer requests.
    batch_segment_seconds: float = 3
    batch_max_concurrency: int = 4
    # Send the WHOLE recording in one call on the stored-audio (batch) path,
    # instead of splitting it at batch_segment_seconds above.
    #
    # This deliberately overrides the table above, and it costs transcript. The
    # comparison surface wants both engines given the same input in batch mode --
    # cohere-transcribe's native mode is a single whole-file POST, so Inception
    # matching it is what makes that an apples-to-apples measurement. Measured
    # 2026-08-31 on one 65 s bilingual recording:
    #
    #     3 s segments -> 140 words (reference 144), WER 0.368
    #     whole file   ->  65 words, HTTP 200, no error
    #
    # ~54% of the transcript is discarded silently, which matches the 100 s row of
    # the table above (51 words vs 219). So an Inception batch WER is substantially
    # a measurement of this gateway's truncation, and the UI says so next to it.
    #
    # A flag, not a deletion: the split path is a hard-won safeguard and this is
    # the only way back to the 140-word result. It also drives the reported
    # transport (see `transport_for`), so the label can never disagree with what
    # actually ran. LIVE chunking is unaffected -- both engines still take 3 s
    # chunks there, which is the symmetric comparison on the live side.
    inception_batch_whole_file: bool = True
    # How long an idle connection to the gateway is kept alive for reuse.
    #
    # This is a latency setting, not a resource one. httpx's module-level
    # `post()` opens a new TCP+TLS connection per call, and measured against this
    # gateway that handshake was 73 ms of a 107 ms round trip — 68% of a figure
    # the scorecard attributes to the ENGINE. A pooled connection cut the median
    # to 35 ms. Must comfortably exceed the live chunk interval, or the
    # connection expires between chunks and every call pays the handshake again.
    stt_keepalive_expiry_sec: float = 30
    stt_max_connections: int = 16
    # TLS for the gateway call only. LITELLM_CA_BUNDLE (a path) is preferred
    # over disabling verification: it verifies properly against an internal CA
    # instead of trusting anything. VERIFY_SSL keeps the .env name already in
    # use, but is scoped to this one service like every other TLS flag here.
    verify_ssl: bool = True
    litellm_ca_bundle: str = ""

    # --- Pre-start the transcript surface's GPU containers ---
    # When true, the supervisor daemon starts every container-backed ASR engine
    # at boot, waits for each to report healthy, and then PINS them: they are
    # exempt from idle-unload for as long as the daemon runs.
    #
    # This exists because the transcript path cannot start a container itself.
    # Unlike the diarization path it never calls admission/ensure_ready, so a
    # local engine whose container is down reports "not configured" and can
    # never be enqueued -- and raising IDLE_UNLOAD_TIMEOUT_SEC does not help,
    # because idle is measured from `last_job_finished_at`, which for these
    # models is weeks old (moss 795h, vibevoice 697h as measured on 2026-09-01).
    #
    # It is a DELIBERATE OVERRIDE of GPU residency policy, and that is the whole
    # point of the flag, so be clear about what it costs:
    #   * pinned containers ignore MAX_RESIDENT_MODELS, so they can hold more
    #     slots than the cap allows;
    #   * they ignore `requires_exclusive_gpu`, which vibevoice sets. Both
    #     vibevoice and moss-transcribe were observed up, healthy and
    #     transcribing correctly at the same time on this GB10 host, so the flag
    #     is conservative here rather than wrong -- but on a smaller GPU
    #     co-residency is exactly what that flag exists to prevent, and vibevoice
    #     will fail to become healthy instead of failing loudly.
    #   * a diarization job for some OTHER model may then find no GPU room.
    #
    # Leave it false on a shared host. Turn it on when the box is dedicated to an
    # STT comparison run and you want every engine available at once.
    stt_prestart_containers: bool = False

    # --- Speechmatics (hosted batch ASR: submit a job, poll, fetch) ---
    speechmatics_api_key: str | None = None
    # The US endpoint. This key 401s against eu2.asr.api.speechmatics.com --
    # Speechmatics keys are region-scoped, so a "just point it at EU" change is a
    # silent auth failure, not a latency tweak.
    speechmatics_url: str = "https://asr.api.speechmatics.com/v2"
    # "ar_en" is Speechmatics' BILINGUAL Arabic+English pack ("Arabic and
    # English" per its own language_pack_info), not a fallback list, and it is
    # the only setting here that actually transcribes code-switched audio.
    #
    # Measured on a 61 s code-switched recording (2026-09-01):
    #   ar_en                -> WER 0.589, 119 deletions, English kept in Latin
    #   auto + expected      -> WER 0.729, 154 deletions, English GONE
    # Language identification ("auto") resolves ONE pack for the whole file
    # (predicted_language "ar", all 90 words tagged ar) and then transliterates
    # or drops every English passage. So "auto" is strictly worse here and is
    # deliberately not the default. Set this to a single code (e.g. "ar") only
    # to measure that regression on purpose.
    speechmatics_language: str = "ar_en"
    speechmatics_operating_point: str = "enhanced"
    speechmatics_poll_interval_sec: float = 3
    speechmatics_timeout_sec: float = 1800

    # --- Speechmatics REAL-TIME (WebSocket streaming) ---
    # A DIFFERENT product from the batch host above. Batch lives on
    # `asr.api.speechmatics.com` (jobs POST/poll/fetch); Real-Time lives on
    # `<region>.rt.speechmatics.com` (WebSocket v2). Same account credential,
    # different endpoint AND a different subdomain scheme (no `.api.` in the
    # RT host -- inserting one NXDOMAIN'd every probe until 2026-09-01).
    #
    # Region matches the JWT audience: our SPEECHMATICS_API_KEY mints a temp
    # RT token with aud=['eu','eu-1'], which is what `eu.rt.speechmatics.com`
    # accepts. `global` is a fallback that routes to a nearest cluster.
    speechmatics_rt_url: str = "wss://eu.rt.speechmatics.com/v2"
    # The Management-Plane host that mints short-lived RT tokens from the
    # long-lived API key. Kept in settings, not hardcoded, because the URL is
    # part of the account contract and moving accounts (staging/prod) is a
    # `.env` change here rather than a code change.
    speechmatics_rt_mint_url: str = "https://mp.speechmatics.com/v1/api_keys"
    # Seconds the minted temp key is valid for. Long enough to cover a full
    # read-aloud, short enough that a captured token from logs is useless
    # within minutes. Default matches Speechmatics' own SDK.
    speechmatics_rt_temp_key_ttl_sec: int = 300
    # `enable_partials=True` gets AddPartialTranscript frames alongside the
    # finals. Partials are visual feedback in the UI; ONLY AddTranscript is
    # recorded and scored (see live_relay.py). Turning partials off saves no
    # cost but hides the "engine is reacting" signal, so on is the default.
    speechmatics_rt_enable_partials: bool = True
    # `max_delay` (seconds) is how long the engine will wait before finalising
    # a segment even without silence. 2 s is the default the docs use and what
    # the 2026-09-01 probe measured cleanly with (time-to-first-partial 856 ms
    # for the 30 s pyannote sample). Lower gives faster finals at the cost of
    # more segment revisions; higher is smoother but laggier.
    speechmatics_rt_max_delay: float = 2
    # Socket-idle timeout: how long the relay waits for a server frame before
    # re-checking whether the client is done. Same shape as Hamsa's.
    speechmatics_rt_idle_timeout_sec: float = 30
    speechmatics_rt_session_timeout_sec: float = 3600

    # --- ElevenLabs Scribe (hosted, one whole-file POST) ---
    elevenlabs_api_key: str | None = None
    elevenlabs_stt_url: str = "https://api.elevenlabs.io/v1/speech-to-text"
    # Validated server-side: a bad id 400s listing what is valid, so a typo fails
    # loudly instead of silently falling back to an older model.
    elevenlabs_stt_model: str = "scribe_v2"
    # "auto" / "" omit `language_code`, which is what makes this engine
    # auto-detect. Measured: omitted on code-switched audio returned
    # language_code "ara" at 0.968 confidence AND kept the English in Latin
    # script (WER 0.564, the best of every engine probed). Forcing a single code
    # can only take that away, so auto is the default.
    elevenlabs_stt_language: str = "auto"
    elevenlabs_stt_timeout_sec: float = 600

    # --- ElevenLabs Scribe v2 REALTIME (WebSocket streaming) ---
    # A DIFFERENT product from `elevenlabs_stt_url` above. That one is the
    # whole-file batch POST used for stored audio; this one is the WebSocket
    # streaming endpoint used for the read-aloud surface's stream transport.
    # Different model (`scribe_v2_realtime`, distinct from `scribe_v2`), same
    # account credential (xi-api-key).
    #
    # Protocol probed end to end on 2026-09-01 against the real API with a
    # 30 s pyannote sample: on connect the server sends `session_started` with
    # the full accepted config; the client sends `input_audio_chunk` frames
    # carrying base64-encoded PCM16 in `audio_base_64` plus `commit` and
    # `sample_rate`; the server streams `partial_transcript` and
    # `committed_transcript` frames back. Raw binary makes the server close
    # cleanly with no error, so the base64-inside-JSON envelope is not
    # decoration -- see apps/background_worker/transcription/elevenlabs/live_relay.py.
    elevenlabs_stt_realtime_url: str = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    elevenlabs_stt_realtime_model: str = "scribe_v2_realtime"
    # "manual" | "vad". Manual means the client (us) commits explicitly on stop;
    # vad lets Scribe auto-commit on internal silence detection. Manual is the
    # default because the scorecard's committed segments are what get measured,
    # and letting a server-side VAD heuristic decide their cadence would move
    # the boundary this comparison is meant to hold fixed across engines.
    elevenlabs_stt_realtime_commit_strategy: str = "manual"
    # How long the relay waits for a server frame before re-checking whether the
    # client is done. Same shape as HAMSA_STT_IDLE_TIMEOUT_SEC.
    elevenlabs_stt_realtime_idle_timeout_sec: float = 15
    elevenlabs_stt_realtime_session_timeout_sec: float = 3600

    # --- ADEO Qwen3-ASR (hosted, per-model proxy on the ADEO inference host) ---
    # A per-model proxy URL, not a gateway with a model list: each model has its
    # own path and its own bearer token, so URL and key travel together.
    adeo_qwen3_asr_url: str | None = None
    adeo_qwen3_asr_api_key: str | None = None
    adeo_qwen3_asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    adeo_qwen3_asr_timeout_sec: float = 600
    # No language setting on purpose, and this is measured, not assumed. vLLM
    # VALIDATES `language` for this model and then the model IGNORES it: omitted,
    # "ar" and "en" returned byte-identical text (same sha256, 821 chars) on the
    # same code-switched clip, while "auto" and "ar,en" are rejected 400 against
    # a 57-code list. The field can therefore only ever break a request, never
    # change one, so it is never sent. The model code-switches natively.

    # --- ADEO Whisper (hosted, per-model proxy on the same ADEO host) ---
    adeo_whisper_url: str | None = None
    adeo_whisper_api_key: str | None = None
    # Empty omits `model`, which is what the working curl for this pod does. Its
    # /v1/models could not be read to confirm a served id (see below).
    adeo_whisper_model: str = ""
    # "auto" / "" omit `language`, which is how Whisper auto-detects.
    #
    # UNVERIFIED, unlike every other engine here: this pod answered 504
    # "Upstream service is unavailable" on every attempt over ~2 minutes on
    # 2026-09-01, including its own /v1/models, so its response shape was never
    # observed. It is implemented against the OpenAI transcription contract the
    # sibling Qwen3 pod on this same host is PROVEN to speak. Probe it once it is
    # up before trusting any figure it produces.
    adeo_whisper_language: str = "auto"
    adeo_whisper_timeout_sec: float = 600

    # --- TTS synthesis engines (script -> read-aloud audio) ---
    # Two synthesis engines, mirroring the two STT engines above: TryHamsa TTS
    # (streaming) and Inception-TTS (single-shot, via the same LiteLLM gateway
    # as inception-stt). Both are compared the same way the STT engines are:
    # each fed in its own native delivery mode, nothing smoothed to match.

    # --- TryHamsa TTS (streaming) ---
    hamsa_tts_api_url: str | None = None
    hamsa_tts_key: str | None = None
    hamsa_tts_bearer_token: str | None = None
    # A COMMA LIST, and entry 0 is the default voice. One field rather than a
    # separate default + list pair: the pair let the two disagree, and setting
    # the list value on the singular field produced a speaker name the pod
    # accepts with a 200 and then drops the stream on, which reads as a network
    # fault rather than a config typo. See hamsa_tts_speaker_options below.
    hamsa_tts_speaker: str = "Ruba"
    hamsa_tts_dialect: str = "uae"
    hamsa_tts_language_id: str = "ar"
    # ASSUMPTION, not a measurement: this API never states its sample rate
    # in-band, in the response headers or the payload. 16000 is chosen because
    # it matches this same vendor's own declared native STT rate
    # (hamsa_stt_sample_rate above) and the exact 3200-byte/100ms chunk-size
    # symmetry between this stream and Hamsa's STT input convention. Kept as a
    # settings field, not a code constant, precisely so it can be corrected if
    # that assumption turns out wrong.
    hamsa_tts_sample_rate: int = 16000
    # Per-service TLS switch, matching hamsa_stt_ssl_verify's own pattern
    # exactly: NOT the bare SSL_VERIFY some clients read, so disabling
    # verification for this one endpoint can never silently weaken another
    # service's TLS.
    hamsa_tts_ssl_verify: bool = True
    hamsa_tts_timeout_sec: float = 120

    # --- Inception-TTS (via the LiteLLM gateway; single-shot) ---
    tts_speech_path: str = "/v1/audio/speech"
    inception_tts_model: str = "inception-tts"
    # A COMMA LIST, entry 0 is the default. Same shape as hamsa_tts_speaker.
    inception_tts_voice: str = "alloy"
    # Confirmed by ffprobe against the real gateway: "wav" is genuinely honored
    # (real RIFF/WAVE, pcm_s16le, mono, 24000Hz) despite the gateway's
    # Content-Type header always claiming audio/mpeg. Never trust that header;
    # see the adapter's magic-byte sniff.
    inception_tts_response_format: str = "wav"
    # Must comfortably exceed the longest allowed script's synthesis time —
    # this engine has no meaningful streaming (~6.7s wall clock measured for a
    # 10-word Arabic sentence, whole body buffered before any bytes arrive).
    inception_tts_timeout_sec: float = 180
    # 422 guard so a runaway script fails fast at the request, not as a
    # gateway timeout minutes later.
    tts_max_input_chars: int = 4000

    # --- Script generation (read-aloud reference text) ---
    # An OpenAI-compatible chat-completions gateway, separate from the STT one
    # above with NO cross-default between them: a silent fallback between two
    # gateways is how the wrong model gets called.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    # No leading /v1: LLM_BASE_URL is expected to already carry its version
    # prefix (OpenRouter's "…/api/v1" does), whereas the STT gateway's base URL
    # does not and carries it in stt_transcription_path instead. The two
    # gateways genuinely differ here, which is why these are separate settings
    # and not one shared base URL.
    llm_chat_path: str = "/chat/completions"
    llm_max_tokens: int = 4096
    llm_timeout_sec: float = 180
    # Do NOT lower this to improve instruction-following on the even language mix;
    # it does the opposite. Measured, four runs each at one attempt, share of
    # scripts landing inside the 35-65% Arabic band:
    #   temp 0.2 -> 0/4  (94/81/78/92% Arabic)
    #   temp 0.5 -> 2/4  (71/48/44/74%)
    #   temp 1.0 -> 4/4  (46/62/57/46%)
    # At low temperature the model settles into its most-likely register, which is
    # Arabic-dominant with English nouns dropped in. Higher temperature is what
    # actually explores the alternating-clause structure the prompt asks for.
    llm_temperature: float = 1.0
    # How the prompt sizes a script for a requested number of minutes. Read-
    # aloud pace with natural pauses, not conversational speed. The word count
    # the API returns is always MEASURED from the generated text, never this
    # estimate.
    script_words_per_minute: int = 70
    # An even language split is requested but not reliably obeyed. Measured over
    # six runs at identical settings, the Arabic share came back 42/92/94% at one
    # minute and 78/78/46% at three — bimodal, not drifting: the model either
    # alternates clauses properly or reverts to Arabic sentences with English
    # nouns dropped in. Prompt wording alone did not fix it, so a script whose
    # measured split falls outside the band is regenerated. The band is a half-
    # width around 50%: 0.15 accepts 35-65% Arabic.
    script_mix_tolerance: float = 0.15
    # Total attempts, not retries. 1 disables the check entirely; the last
    # attempt is always returned even if it misses, because a script that is
    # slightly unbalanced still beats no script.
    script_mix_max_attempts: int = 3
    # The script-length slider's stops, the language-mix buttons and the
    # hard-case chips, all served to the browser by GET /config so none of
    # them is a literal in the frontend.
    script_length_options_min: str = "1,3,5,10"
    script_language_mixes: str = "ar,mixed-50-50,en"
    script_hard_cases: str = "proper-nouns,numbers-dates,emirati-dialect,fast-speech"

    # --- Live transcript sessions (the read-aloud comparison) ---
    # The recorder's PCM rate, served to the browser: the mic tap resamples to
    # it so both engines receive identical audio.
    live_record_sample_rate: int = 16000
    # Samples per microphone callback. A latency setting: at 16 kHz a 4096-sample
    # block is 256 ms, so PCM reached the streaming engine in bursts of ~2.6 of
    # its 100 ms frames rather than a smooth cadence, and a chunk was emitted up
    # to 256 ms after its audio existed. 1024 samples is 64 ms — under one engine
    # frame, so nothing waits on the block boundary.
    #
    # Must be a power of two between 256 and 16384 (Web Audio's constraint). The
    # waveform's render rate is deliberately NOT tied to this; see the recorder.
    live_record_block_samples: int = 1024
    # Chunk interval for engines that cannot stream, as a default and the
    # bounds of the UI control. This is the ONLY knob on chunking -- there is
    # no lookback or silence threshold, by design (see batch_segment_seconds).
    live_chunk_default_sec: float = 3
    live_chunk_min_sec: float = 1
    live_chunk_max_sec: float = 10
    # How long a live session's accumulated text and latencies survive in Redis
    # before expiry. Must exceed the longest recording anyone will read plus
    # the time to finalize it.
    live_session_ttl_sec: int = 7200
    # How long to wait for a streaming engine's socket to open before giving up.
    # Load-bearing: a misconfigured dev proxy drops the upgrade so the socket
    # neither opens nor errors, and without a deadline the recorder waits on it
    # forever and looks broken. A bounded wait turns that into a message.
    live_socket_open_timeout_sec: float = 10

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
        return _split_csv(self.cors_allow_origins)

    @property
    def hamsa_ws_endpoint(self) -> str | None:
        """The Hamsa WebSocket URL, from either accepted setting name."""
        return self.hamsa_stt_url or self.hamsa_stt_ws_url

    @property
    def stt_endpoint_url(self) -> str | None:
        """Full Inception-STT transcription URL, or None when unconfigured."""
        if not self.litellm_base_url:
            return None
        return f"{self.litellm_base_url.rstrip('/')}{self.stt_transcription_path}"

    @property
    def tts_speech_url(self) -> str | None:
        """Full Inception-TTS synthesis URL, or None when unconfigured."""
        if not self.litellm_base_url:
            return None
        return f"{self.litellm_base_url.rstrip('/')}{self.tts_speech_path}"

    @property
    def hamsa_tts_speaker_options(self) -> list[str]:
        """Every voice this host offers for hamsa-tts, in .env order.

        Never empty: an empty list would leave the dropdown unopenable and make
        `hamsa_tts_default_speaker` raise IndexError at request time, so a value
        that splits to nothing falls back to itself, stripped.
        """
        return _split_csv(self.hamsa_tts_speaker) or [self.hamsa_tts_speaker.strip()]

    @property
    def hamsa_tts_default_speaker(self) -> str:
        """The voice used when the caller does not choose one: entry 0.

        Reading position 0 rather than the raw field is what stops a list value
        being posted verbatim as one speaker name.
        """
        return self.hamsa_tts_speaker_options[0]

    @property
    def inception_tts_voice_options(self) -> list[str]:
        return _split_csv(self.inception_tts_voice) or [self.inception_tts_voice.strip()]

    @property
    def inception_tts_default_voice(self) -> str:
        return self.inception_tts_voice_options[0]

    @property
    def llm_chat_url(self) -> str | None:
        """Full script-generation chat-completions URL, or None when unconfigured.

        Tolerates LLM_BASE_URL already carrying the endpoint path. Writing the
        whole URL there is the natural reading of "base url", and the two
        gateways this repo talks to disagree about it (OpenRouter's own docs
        quote ".../api/v1/chat/completions"). Appending blindly turned that into
        a doubled path and a bare 404 with nothing to point at. Checking is one
        comparison; the alternative is a support question every time.
        """
        if not self.llm_base_url:
            return None
        base = self.llm_base_url.rstrip("/")
        if base.endswith(self.llm_chat_path.rstrip("/")):
            return base
        return f"{base}{self.llm_chat_path}"

    @property
    def litellm_httpx_verify(self) -> bool | str:
        """httpx `verify` for the gateway: a CA bundle path when one is set,
        otherwise the on/off flag. A bundle always wins, so setting one is
        enough to re-enable verification without also flipping VERIFY_SSL."""
        return self.litellm_ca_bundle or self.verify_ssl

    @property
    def script_length_options(self) -> list[int]:
        """The script-length slider's stops, in minutes."""
        return [int(part) for part in _split_csv(self.script_length_options_min)]

    @property
    def script_language_mix_options(self) -> list[str]:
        return _split_csv(self.script_language_mixes)

    @property
    def script_hard_case_options(self) -> list[str]:
        return _split_csv(self.script_hard_cases)


@lru_cache
def get_settings() -> Settings:
    return Settings()
