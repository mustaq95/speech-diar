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

The model's output is generated text that only *looks* like JSON, and it does
not always parse -- see `_parse_segments`.
"""

import json
import os
import re
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


_OBJECT_RE = re.compile(r"\{[^{}]*\}")
_START_RE = re.compile(r'"Start"\s*:\s*(-?\d+(?:\.\d+)?)')
_END_RE = re.compile(r'"End"\s*:\s*(-?\d+(?:\.\d+)?)')
_SPEAKER_RE = re.compile(r'"Speaker"\s*:\s*(\d+)')


def _parse_segments(raw_text: str) -> list[dict] | None:
    """Recover the segment list from the model's generated text.

    VibeVoice-ASR does not emit JSON, it emits a *string that resembles* JSON,
    and that string is not always valid: an apostrophe or quote inside the
    transcribed `Content` is enough to break it (json.JSONDecodeError from
    transformers' own `extract_speaker_dict`, which does not catch it despite
    what the model card claims). Because generation is autoregressive on a GPU,
    the same audio can parse on one run and fail on the next.

    So: try strict JSON first. If that fails, fall back to reading the numeric
    fields straight out of each object with a regex. That is deliberately
    limited to `Start`, `End` and `Speaker` -- the fields the diarization
    contract actually needs -- and skips `Content` entirely, which is both the
    field that breaks the parse and the field this platform discards anyway.
    Nothing is invented: every number returned was emitted by the model.
    """
    body = raw_text.split("<|im_start|>assistant", 1)[-1]
    for marker in ("<|im_end|>", "<|endoftext|>"):
        body = body.split(marker, 1)[0]
    body = body.strip()

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        pass
    else:
        return parsed if isinstance(parsed, list) else None

    segments: list[dict] = []
    for match in _OBJECT_RE.finditer(body):
        chunk = match.group(0)
        start = _START_RE.search(chunk)
        end = _END_RE.search(chunk)
        if not (start and end):
            continue
        segment: dict = {"Start": float(start.group(1)), "End": float(end.group(1))}
        # Absent on non-speech events ([Silence], [Music], ...) -- leave it out
        # rather than inventing a speaker for them.
        speaker = _SPEAKER_RE.search(chunk)
        if speaker:
            segment["Speaker"] = int(speaker.group(1))
        segments.append(segment)

    return segments or None


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
    # eager, not sdpa and not flash_attention_2. Flash-attn has no aarch64
    # wheel (see the Dockerfile). SDPA looks like the obvious fallback but is
    # not an option either: attn_implementation applies to every submodule,
    # and VibeVoiceAcousticTokenizerEncoderModel has no SDPA path --
    # from_pretrained hard-raises a ValueError naming eager as the workaround.
    _model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
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
            raw_text = _processor.decode(generated_ids)[0]

        parsed = _parse_segments(raw_text)
        if parsed is None:
            raise HTTPException(
                status_code=500,
                detail=f"model output did not parse as segments: {raw_text!r}",
            )

        return {
            "audio_duration_sec": _wav_duration_sec(audio_path),
            "segments": parsed,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
