"""One envelope, one sink interface, two adapters: the parent's ErrorBus,
the subprocess's PushSink, and the one log bridge over either."""

from __future__ import annotations

import logging

import pytest

from app.errors import ErrorBus, ErrorEnvelope, EventSink, LogBridgeHandler, get_bus, set_sink
from app.server.event_push import PushSink, install_push_sink

# ── the parent side: /api/internal/events republishes envelopes ──


def test_push_events_endpoint_republishes_envelopes() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.execution import internal_events

    app = FastAPI()
    app.include_router(internal_events.router, prefix="/api/internal")

    env = ErrorBus().from_exception(
        RuntimeError("boom"),
        source="node",
        code="SUBPROC_NODE_FAIL",
        scope={"node_id": "x__y", "execution_id": "e1", "nodeset": "x"},
    )
    before = len(get_bus().snapshot())
    with TestClient(app) as client:
        r = client.post(
            "/api/internal/events",
            json={"events": [env.to_dict(), {"severity": "warning", "message": "careful"}]},
        )
    assert r.status_code == 200
    assert r.json()["count"] == 2

    snap = get_bus().snapshot()
    assert len(snap) >= before + 2
    err = [e for e in snap if e.message == "boom"][-1]
    assert err.id == env.id and err.severity == "error" and err.source == "node"
    assert err.scope["node_id"] == "x__y" and err.scope["origin"] == "subprocess"
    assert "RuntimeError" in err.details["traceback"]


def test_push_events_unknown_severity_falls_back_to_info() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.execution import internal_events

    app = FastAPI()
    app.include_router(internal_events.router, prefix="/api/internal")
    with TestClient(app) as client:
        client.post(
            "/api/internal/events", json={"events": [{"severity": "BOGUS", "message": "weird"}]}
        )
    env = [e for e in get_bus().snapshot() if e.message == "weird"][-1]
    assert env.severity == "info" and env.title == "weird"


# ── the subprocess side: PushSink ──


def test_push_sink_buffers_warnings_and_flushes_errors_at_once() -> None:
    sink = PushSink("http://127.0.0.1:1/", nodeset="ns")  # unreachable: every push drops
    sink.emit(severity="warning", source="node", code="X", title="careful")
    assert sink.pending() == 1
    sink.from_exception(RuntimeError("boom"), source="node", code="SUBPROC_NODE_FAIL")
    assert sink.pending() == 0  # error → immediate flush (POST failed → dropped, no raise)
    sink.flush()
    assert sink.pending() == 0


def test_push_sink_stamps_origin_and_nodeset() -> None:
    sink = PushSink("http://127.0.0.1:1", nodeset="ns")
    env = sink.emit(severity="info", source="node", code="X", title="t", scope={"node_id": "n"})
    assert env.scope == {"node_id": "n", "origin": "subprocess", "nodeset": "ns"}
    assert sink.pending() == 1
    other = sink.emit(
        severity="info", source="node", code="X", title="t", scope={"nodeset": "mine"}
    )
    assert other.scope["nodeset"] == "mine"  # an explicit scope wins


def test_install_push_sink_is_a_noop_without_a_url() -> None:
    before = get_bus()
    assert install_push_sink(None, nodeset="x") is None
    assert install_push_sink("", nodeset="x") is None
    assert get_bus() is before


def test_install_push_sink_makes_it_the_process_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = get_bus()
    root = logging.getLogger()
    bridges_before = [h for h in root.handlers if isinstance(h, LogBridgeHandler)]
    for h in bridges_before:
        root.removeHandler(h)
    try:
        sink = install_push_sink("http://127.0.0.1:1", nodeset="ns")
        assert isinstance(sink, PushSink) and get_bus() is sink
        bridge = next(h for h in root.handlers if isinstance(h, LogBridgeHandler))
        assert bridge.level == logging.WARNING and bridge._sink is sink
        root.removeHandler(bridge)
    finally:
        set_sink(previous)
        for h in bridges_before:
            root.addHandler(h)


# ── the one log bridge, over either adapter ──


class _Capture(EventSink):
    def __init__(self) -> None:
        self.seen: list[ErrorEnvelope] = []

    def publish(self, env: ErrorEnvelope) -> None:
        self.seen.append(env)


def _record(name: str, level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord(name, level, "f", 1, msg, None, None)


def test_log_bridge_skips_own_and_http_loggers() -> None:
    sink = _Capture()
    h = LogBridgeHandler(sink, level=logging.WARNING)
    h.emit(_record("httpx", logging.WARNING, "skip me"))
    h.emit(_record("agentcanvas.event-push", logging.ERROR, "skip"))
    h.emit(_record("agentcanvas.errors", logging.ERROR, "skip"))
    h.emit(_record("uvicorn.access", logging.ERROR, "skip"))
    assert sink.seen == []
    h.emit(_record("some.nodeset", logging.ERROR, "real error"))
    assert [e.severity for e in sink.seen] == ["error"]
    assert sink.seen[0].source == "log" and sink.seen[0].code == "LOG_ERROR"
    assert sink.seen[0].scope["logger"] == "some.nodeset"


def test_log_bridge_never_raises_into_logging() -> None:
    class _Broken(EventSink):
        def publish(self, env: ErrorEnvelope) -> None:
            raise RuntimeError("sink down")

    h = LogBridgeHandler(_Broken(), level=logging.WARNING)
    h.emit(_record("x", logging.WARNING, "fine"))  # must not raise
