"""Endpoint tests for the nodeset source editor routes.

python -m pytest app/api/platform/test_components_source.py -v

Path-guard and layout logic are unit-tested in
``app/components/test_nodeset_source.py``; here we exercise the HTTP
surface: response shapes and the 404 / 400-syntax / 409-conflict paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.platform import components
from app.components.registry import WorkspaceComponentRegistry

_NODESET_SRC = '''
from app.components import BaseNodeSet

GREETING = "hi"


def _fmt(x):
    return f"{GREETING} {x}"


class _TmpDemoNode:
    node_type = "tmp_srcedit__demo"

    def forward(self, inputs):
        return _fmt(inputs)


class _TmpSrceditNodeSet(BaseNodeSet):
    name = "tmp_srcedit"
    description = "tmp nodeset for source-editor endpoint tests"

    def get_tools(self):
        return []
'''

_PROMPTS_SRC = '''PROMPT = "hello"


class _SibNode:
    node_type = "tmp_srcedit__sib"
'''


class _StubServices:
    def __init__(self, registry: WorkspaceComponentRegistry) -> None:
        self.workspace_component_registry = registry


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ws = tmp_path / "workspace"
    pkg = ws / "nodesets" / "method" / "tmp_srcedit"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(_NODESET_SRC)
    (pkg / "_prompts.py").write_text(_PROMPTS_SRC)

    registry = WorkspaceComponentRegistry(scan_dir=ws)
    registry.scan_all()
    monkeypatch.setattr(components, "get_services", lambda: _StubServices(registry))

    app = FastAPI()
    app.include_router(components.router, prefix="/api/components")
    with TestClient(app) as c:
        yield c
    registry.unregister_all()


def test_get_source_defaults_to_entry_file(client: TestClient) -> None:
    r = client.get("/api/components/nodesets/tmp_srcedit/source")
    assert r.status_code == 200
    body = r.json()
    assert body["is_package"] is True
    assert body["file"] == "__init__.py"
    assert body["files"] == ["__init__.py", "_prompts.py"]
    assert "_TmpSrceditNodeSet" in body["content"]
    assert body["mode"] == "local"
    assert isinstance(body["mtime_ns"], int)


def test_get_source_sibling_file(client: TestClient) -> None:
    r = client.get("/api/components/nodesets/tmp_srcedit/source", params={"file": "_prompts.py"})
    assert r.status_code == 200
    assert r.json()["content"] == _PROMPTS_SRC


def test_get_source_unknown_nodeset_404(client: TestClient) -> None:
    assert client.get("/api/components/nodesets/nope/source").status_code == 404


def test_get_source_escape_403(client: TestClient) -> None:
    r = client.get(
        "/api/components/nodesets/tmp_srcedit/source", params={"file": "../mapgpt.py"}
    )
    assert r.status_code == 403


# ── scoped view endpoints ──


def test_get_scoped_entry_file(client: TestClient) -> None:
    r = client.get(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        params={"node_type": "tmp_srcedit__demo"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == "__init__.py"
    kinds = {(s["kind"], s["name"]) for s in body["segments"]}
    assert ("class", "_TmpDemoNode") in kinds
    assert ("function", "_fmt") in kinds
    assert ("globals", "GREETING") in kinds
    assert ("class", "_TmpSrceditNodeSet") not in kinds  # not this node's slice


def test_get_scoped_searches_sibling_files(client: TestClient) -> None:
    r = client.get(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        params={"node_type": "tmp_srcedit__sib"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == "_prompts.py"
    kinds = {(s["kind"], s["name"]) for s in body["segments"]}
    assert ("class", "_SibNode") in kinds
    assert ("globals", "PROMPT") in kinds


def test_get_scoped_unknown_node_type_404(client: TestClient) -> None:
    r = client.get(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        params={"node_type": "tmp_srcedit__nope"},
    )
    assert r.status_code == 404


def test_put_scoped_splices_and_returns_fresh_segments(client: TestClient) -> None:
    got = client.get(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        params={"node_type": "tmp_srcedit__demo"},
    ).json()
    globals_seg = next(s for s in got["segments"] if s["kind"] == "globals")
    edited = {**globals_seg, "text": 'GREETING = "yo"\nEXTRA = 2\n'}
    r = client.put(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        json={
            "file": got["file"],
            "node_type": "tmp_srcedit__demo",
            "base_mtime_ns": got["mtime_ns"],
            "segments": [edited],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    fresh_kinds = {(s["kind"], s["name"]) for s in body["segments"]}
    assert ("globals", "GREETING, EXTRA") in fresh_kinds
    # rest of the file untouched
    whole = client.get("/api/components/nodesets/tmp_srcedit/source").json()["content"]
    assert 'GREETING = "yo"' in whole
    assert "class _TmpSrceditNodeSet(BaseNodeSet):" in whole


def test_put_scoped_syntax_error_400(client: TestClient) -> None:
    got = client.get(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        params={"node_type": "tmp_srcedit__demo"},
    ).json()
    cls_seg = next(s for s in got["segments"] if s["kind"] == "class")
    r = client.put(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        json={
            "file": got["file"],
            "node_type": "tmp_srcedit__demo",
            "base_mtime_ns": got["mtime_ns"],
            "segments": [{**cls_seg, "text": "class _TmpDemoNode(:\n"}],
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "syntax"


def test_put_scoped_stale_mtime_409(client: TestClient) -> None:
    got = client.get(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        params={"node_type": "tmp_srcedit__demo"},
    ).json()
    seg = got["segments"][0]
    r = client.put(
        "/api/components/nodesets/tmp_srcedit/source/scoped",
        json={
            "file": got["file"],
            "node_type": "tmp_srcedit__demo",
            "base_mtime_ns": 1,
            "segments": [seg],
        },
    )
    assert r.status_code == 409


def test_put_source_writes_file(client: TestClient) -> None:
    got = client.get(
        "/api/components/nodesets/tmp_srcedit/source", params={"file": "_prompts.py"}
    ).json()
    r = client.put(
        "/api/components/nodesets/tmp_srcedit/source",
        json={
            "file": "_prompts.py",
            "content": "PROMPT = 'edited'\n",
            "base_mtime_ns": got["mtime_ns"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["stale"] is False
    assert body["run_active"] is False
    again = client.get(
        "/api/components/nodesets/tmp_srcedit/source", params={"file": "_prompts.py"}
    ).json()
    assert again["content"] == "PROMPT = 'edited'\n"


def test_put_source_syntax_error_400(client: TestClient) -> None:
    got = client.get("/api/components/nodesets/tmp_srcedit/source").json()
    r = client.put(
        "/api/components/nodesets/tmp_srcedit/source",
        json={"file": "__init__.py", "content": "def f(:\n", "base_mtime_ns": got["mtime_ns"]},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["error"] == "syntax"
    assert detail["line"] == 1
    # nothing written
    fresh = client.get("/api/components/nodesets/tmp_srcedit/source").json()
    assert "_TmpSrceditNodeSet" in fresh["content"]


def test_put_source_stale_mtime_409(client: TestClient) -> None:
    r = client.put(
        "/api/components/nodesets/tmp_srcedit/source",
        json={"file": "__init__.py", "content": "x = 1\n", "base_mtime_ns": 12345},
    )
    assert r.status_code == 409


def test_put_source_missing_fields_400(client: TestClient) -> None:
    r = client.put(
        "/api/components/nodesets/tmp_srcedit/source", json={"content": "x = 1\n"}
    )
    assert r.status_code == 400
