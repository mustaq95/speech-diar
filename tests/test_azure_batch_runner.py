"""Azure batch runner: the job timeout is a hard bound on paid API calls —
once the deadline passes, no poll GET goes out; the only remaining call is
the DELETE that cancels the server-side job."""

from types import SimpleNamespace
from typing import Any

import pytest
import requests

from apps.background_worker.models.azure_batch import runner
from apps.background_worker.models.azure_batch.runner import AzureBatchRunner

JOB_URL = "https://fake.example/speechtotext/v3.2/transcriptions/job-1"


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeRequests:
    """Recording double for the `requests` module; the job never finishes."""

    RequestException = requests.RequestException

    def __init__(self) -> None:
        self.post_calls = 0
        self.get_calls = 0
        self.delete_calls = 0

    def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        self.post_calls += 1
        return _FakeResponse({"self": JOB_URL})

    def get(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        self.get_calls += 1
        return _FakeResponse({"status": "Running"})

    def delete(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        self.delete_calls += 1
        return _FakeResponse({})


def test_batch_timeout_makes_no_paid_calls_past_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRequests()
    monkeypatch.setattr(runner, "requests", fake)
    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: SimpleNamespace(
            azure_speech_key="k",
            azure_speech_endpoint="https://fake.example",
            azure_speech_region=None,
            locale="en-US",
            azure_batch_max_speakers=8,
            azure_batch_time_to_live="PT1H",
            azure_batch_job_timeout_sec=0,
            azure_batch_poll_interval_sec=0,
        ),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        AzureBatchRunner().run("https://blob.example/x.wav")

    assert fake.post_calls == 1  # the pre-deadline submit
    assert fake.get_calls == 0  # no poll GET after the deadline
    assert fake.delete_calls == 1  # server-side job still cancelled
