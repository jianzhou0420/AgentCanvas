"""Phase 3 (Move 3) tests: subprocess→executor log/error push channel.

python -m pytest app/test_event_push.py -v
"""

from __future__ import annotations

import logging

import pytest

# ── executor endpoint → ErrorBus ──


def test_push_events_endpoint_emits_to_bus() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.execution import internal_events
    from app.errors import get_bus

    app = FastAPI()
    app.include_router(internal_events.router, prefix="/api/internal")

    before = len(get_bus().snapshot())
    with TestClient(app) as client:
        r = client.post(
            "/api/internal/events",
            json={
                "events": [
                    {
                        "severity": "error",
                        "message": "boom",
                        "code": "SUBPROC_NODE_FAIL",
                        "node_id": "x__y",
                        "execution_id": "e1",
                        "nodeset": "x",
                    },
                    {"severity": "warning", "message": "careful"},
                ]
            },
        )
    assert r.status_code == 200
    assert r.json()["count"] == 2

    snap = get_bus().snapshot()
    assert len(snap) >= before + 2
    err = [e for e in snap if e.message == "boom"][-1]
    assert err.severity == "error"
    assert err.source == "node"
    assert err.scope.get("node_id") == "x__y"
    assert err.scope.get("origin") == "subprocess"


def test_push_events_unknown_severity_falls_back_to_info() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.execution import internal_events
    from app.errors import get_bus

    app = FastAPI()
    app.include_router(internal_events.router, prefix="/api/internal")
    with TestClient(app) as client:
        client.post(
            "/api/internal/events",
            json={"events": [{"severity": "BOGUS", "message": "weird"}]},
        )
    env = [e for e in get_bus().snapshot() if e.message == "weird"][-1]
    assert env.severity == "info"


# ── subprocess push client ──


def test_emit_event_noop_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.server import event_push

    monkeypatch.delenv("AGENTCANVAS_EXECUTOR_URL", raising=False)
    with event_push._LOCK:
        event_push._BUFFER.clear()
    event_push.emit_event("info", "hi")
    assert event_push._BUFFER == []


def test_emit_event_buffers_and_flush_drops_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.server import event_push

    # Unreachable URL: emit buffers, flush attempts + drops (never raises).
    monkeypatch.setenv("AGENTCANVAS_EXECUTOR_URL", "http://127.0.0.1:1")
    with event_push._LOCK:
        event_push._BUFFER.clear()
    event_push.emit_event("warning", "careful", code="X", node_id="n1")
    assert len(event_push._BUFFER) == 1
    assert event_push._BUFFER[0]["message"] == "careful"
    event_push.flush()  # POST to :1 fails fast → buffer drained, no exception
    assert event_push._BUFFER == []


def test_log_bridge_skips_own_and_http_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.server import event_push

    monkeypatch.setenv("AGENTCANVAS_EXECUTOR_URL", "http://127.0.0.1:1")
    with event_push._LOCK:
        event_push._BUFFER.clear()
    h = event_push.SubprocessLogBridge(level=logging.WARNING)

    h.emit(logging.LogRecord("httpx", logging.WARNING, "f", 1, "skip me", None, None))
    h.emit(logging.LogRecord("agentcanvas.event-push", logging.ERROR, "f", 1, "skip", None, None))
    assert event_push._BUFFER == []

    h.emit(logging.LogRecord("some.nodeset", logging.ERROR, "f", 1, "real error", None, None))
    assert len(event_push._BUFFER) == 1
    assert event_push._BUFFER[0]["severity"] == "error"
    with event_push._LOCK:
        event_push._BUFFER.clear()
