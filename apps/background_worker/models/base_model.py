"""The abstract contract every diarization model must follow.

Each model lives in its own folder with two halves:

- ``runner.py``  — knows how to EXECUTE the model (load weights, call the
  API, ...) and returns the model's NATIVE output, whatever shape that is.
- ``adapter.py`` — the ONLY code allowed to understand that native shape;
  translates it into the shared ``DiarizationModelRun`` contract.

This is the backend twin of the frontend's adapter layer
(`apps/frontend/src/adapters/`): raw model output never crosses a service
boundary, only the unified contract does.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from packages.shared_contracts.schemas import DiarizationModelRun, normalize_model_run

TRaw = TypeVar("TRaw")


class ModelRunner(ABC, Generic[TRaw]):
    """Executes one diarization engine against an audio file."""

    #: Stable identifier, e.g. "pyannote". Must be unique across models.
    model_id: str

    #: False for pluggable stubs whose `run()` always raises NotImplementedError.
    #: `GET /models` uses this so the UI can list-but-disable unimplemented
    #: engines instead of hiding them or faking output.
    available: bool = True

    @abstractmethod
    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> TRaw:
        """Run the engine and return its NATIVE output (any shape)."""


class ModelAdapter(ABC, Generic[TRaw]):
    """Translates one engine's native output into the shared contract."""

    #: Static display metadata — known without running the model, so it can
    #: back `GET /models` and label a run before it has produced output.
    name: str
    short: str
    description: str

    @abstractmethod
    def adapt(self, raw: TRaw) -> DiarizationModelRun:
        """Map the native payload to ``DiarizationModelRun``. Must be pure."""

    def audio_duration_sec(self, raw: TRaw) -> float | None:
        """Total audio duration if the native payload carries it, else None."""
        return None

    def processing_ms(self, raw: TRaw) -> int | None:
        """The engine's own reported processing time in ms, if it carries one,
        else None (the pipeline falls back to worker-measured wall-clock)."""
        return None


class DiarizationModel(Generic[TRaw]):
    """One pluggable model = a runner + its adapter, registered by id."""

    def __init__(self, runner: ModelRunner[TRaw], adapter: ModelAdapter[TRaw]) -> None:
        self.runner = runner
        self.adapter = adapter

    @property
    def model_id(self) -> str:
        return self.runner.model_id

    @property
    def available(self) -> bool:
        return self.runner.available

    def process(self, audio_path: str, params: dict[str, Any] | None = None) -> DiarizationModelRun:
        raw = self.runner.run(audio_path, params)
        return normalize_model_run(self.adapter.adapt(raw))
