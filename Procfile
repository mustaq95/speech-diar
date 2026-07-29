web: uv run uvicorn apps.backend_api.main:app --port 8010
worker: ./scripts/run_workers.sh
supervisor: uv run python -m apps.background_worker.supervisor.daemon
recording: uv run uvicorn tools.recording_api.server:app --port 8215
# Offline transcription engine. Unlike the diarization containers this one has
# no GPU-supervisor lifecycle, so honcho owns it. Self-gating: it only starts
# the container when TRANSCRIPTION_MODE=offline (the mode that uses it) and
# otherwise idles, so there is nothing to comment out when switching modes.
cohere: ./deploy/cohere-transcribe/cohere_transcribe_honcho.sh
