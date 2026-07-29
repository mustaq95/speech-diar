"""CTC forced aligner — execution code for the word-timing stage.

Runs MahmoudAshraf97/ctc-forced-aligner's MMS-300m model over the audio and the
ASR transcript, producing a start/end for each word. Shared by BOTH
transcription modes: whichever engine produced the text, the timings come from
here, so online and offline transcripts are directly comparable rather than
carrying two differently-derived timelines.

Returns the library's native span list untouched; only `adapter.py` reads it.

Why MMS and not the XLS-R path: MMS romanizes text before alignment, so a
code-switched recording (Arabic with English terms — routine in this
platform's material) still aligns. An Arabic-script-only CTC vocabulary
silently fails to place every Latin word.

torch and ctc_forced_aligner are heavy, GPU-oriented deps (`uv sync --extra
models`) and are imported lazily inside `_load_model()`, never at module scope:
this module is imported by the API process too, which must not require them.
The model is cached per worker process, exactly like
`apps/background_worker/models/pyannote_3_1/runner.py` — a cold load measured
~33 s on this host, which would otherwise be paid on every job. It holds ~1.2 GB
of GPU memory in each worker that has run an alignment; that footprint is
outside the GPU supervisor's residency cap, which only counts containers.

Configuration (`.env`, read via `packages/config/settings.py`):
  CTC_ALIGNER_LANGUAGE   — ISO-639-3 code passed to the romanizer (e.g. "ara")
  CTC_ALIGNER_BATCH_SIZE — emission-generation batch size
  DIARIZATION_DEVICE     — "cpu" or "cuda"; shared with the diarization models
"""

from typing import Any

from packages.config.settings import get_settings

#: The library's native output: one dict per word, with "text"/"start"/"end"/"score".
CtcAlignerRawOutput = list[dict[str, Any]]

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        import torch
        from ctc_forced_aligner import load_alignment_model

        settings = get_settings()
        device = settings.diarization_device
        # fp16 only on CUDA. On CPU it is both slower and numerically flaky,
        # and torchaudio's forced_align kernel expects fp32 there.
        dtype = torch.float16 if device == "cuda" else torch.float32
        _model, _tokenizer = load_alignment_model(device, dtype=dtype)
    return _model, _tokenizer


def run(audio_path: str, text: str) -> CtcAlignerRawOutput:
    """Align `text` against `audio_path`; return the native per-word spans.

    An empty transcript is a legitimate input (silent audio): return no spans
    rather than letting the library raise on an empty token list.
    """
    if not text.strip():
        return []

    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_audio,
        postprocess_results,
        preprocess_text,
    )

    settings = get_settings()
    model, tokenizer = _load_model()

    waveform = load_audio(audio_path, model.dtype, model.device)
    emissions, stride = generate_emissions(model, waveform, batch_size=settings.ctc_aligner_batch_size)
    tokens, normalized = preprocess_text(text, romanize=True, language=settings.ctc_aligner_language)
    segments, scores, blank = get_alignments(emissions, tokens, tokenizer)
    return postprocess_results(normalized, get_spans(tokens, segments, blank), stride, scores)
