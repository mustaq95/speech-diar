"""Unit test for apps/background_worker/supervisor/containers.py — the
docker CLI wrapper. Everything subprocess-level is stubbed; no test here
may shell out to a real `docker`.
"""

import pytest

from apps.background_worker.supervisor import containers


def test_stop_container_cli_timeout_outlives_the_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test, caught live: `docker stop --time N` lets the
    container take up to N seconds before SIGKILL, and a NIM/Triton
    container genuinely uses all of it — a flat CLI timeout equal to the
    grace made the client raise TimeoutExpired at the same instant the stop
    would have completed, killing the evicting job. The subprocess deadline
    must be strictly greater than the container's own grace window."""
    grace = 30
    fake_settings = type("S", (), {"container_stop_grace_sec": grace})()
    monkeypatch.setattr(containers, "get_settings", lambda: fake_settings)

    seen: dict = {}

    def fake_run(args, timeout):
        seen["args"] = args
        seen["timeout"] = timeout

    monkeypatch.setattr(containers, "_run", fake_run)

    containers.stop_container("parakeet-nim-str")

    assert seen["args"] == ["docker", "stop", "--time", str(grace), "parakeet-nim-str"]
    assert seen["timeout"] > grace
