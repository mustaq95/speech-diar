"""Thin FastAPI wrapper around Microsoft's VibeVoice-ASR.

Execution only: loads the model once at startup, then for each request runs it
against the uploaded audio and hands back the parsed segment list the
processor itself produces, untouched -- including the `Content` (transcript)
field this platform's contract has no place for. Dropping it is the worker
adapter's job (apps/background_worker/models/vibevoice/adapter.py), per this
platform's runner/adapter split: the runner boundary carries native output.

Model: microsoft/VibeVoice-ASR-HF (MIT, ungated) -- 8B decoder-only (Qwen2.5
backbone + VibeVoice's 7.5 Hz dual audio tokenizer). Unlike every other
engine in this lineup, diarization here is not acoustic clustering: the
speaker labels are generated text tokens conditioned on the transcript, so
the model emits ASR, diarization, and timestamps in one autoregressive pass
with no chunking and no external clustering stage.

Two consequences worth knowing before reading its output:

  - It does not handle overlapping speech. The output is a serialized stream,
    and the technical report (arXiv 2601.18184) states that where speakers
    talk simultaneously "the model tends to transcribe the dominant speaker,
    potentially missing secondary information." Expect near-zero overlap in
    its timeline where the acoustic models show plenty. That is the model.
  - Speaker count is not capped architecturally (speaker ids are emitted as
    text, not selected from a fixed set of output slots), but it was only
    evaluated up to ~8 speakers (AISHELL-4), and only in Chinese. Never pass
    a speaker-count hint anyway, per this platform's "nothing fabricated"
    rule -- and there is no argument to pass one to.

Greedy decoding, 32768 max_new_tokens, and bf16 all come from the model's own
generation_config.json / config.json; nothing is overridden here.
"""

import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from wave import open as wave_open

import torch
from fastapi import FastAPI, HTTPException, UploadFile

_MODEL_ID = os.environ.get("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-ASR-HF")

# One GPU, one resident 8B model: serialize inference rather than letting
# concurrent requests race for VRAM. The worker pool fans out N models in
# parallel, so this endpoint can genuinely be hit concurrently.
_lock = threading.Lock()
_processor = None
_model = None


def _wav_duration_sec(path: Path) -> float | None:
    with suppress(Exception):
        with wave_open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    # sdpa, not flash_attention_2 -- see the Dockerfile note on aarch64.
    _model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    _model.eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health/ready")
def health_ready() -> dict[str, str]:
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/diarize")
async def diarize(file: UploadFile) -> dict[str, object]:
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    work_dir = Path(tempfile.mkdtemp(prefix="vibevoice-req-"))
    try:
        audio_path = work_dir / (file.filename or "audio.wav")
        audio_path.write_bytes(await file.read())

        with _lock, torch.inference_mode():
            inputs = _processor.apply_transcription_request(
                audio=str(audio_path),
            ).to(_model.device, _model.dtype)
            output_ids = _model.generate(**inputs)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            parsed = _processor.decode(generated_ids, return_format="parsed")[0]

        # The model generates a *string that resembles* JSON; `return_format=
        # "parsed"` returns the raw string as-is when that string doesn't
        # parse. Surfacing that as a failure is deliberate: a silent fallback
        # would hand the worker something it would read as zero segments, and
        # a fabricated empty result is worse than a visible error.
        if not isinstance(parsed, list):
            raise HTTPException(
                status_code=500,
                detail=f"model output did not parse as segments: {parsed!r}",
            )

        return {
            "audio_duration_sec": _wav_duration_sec(audio_path),
            "segments": parsed,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
