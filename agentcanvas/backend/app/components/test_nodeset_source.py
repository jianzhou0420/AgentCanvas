"""Unit tests for nodeset_source — layout resolution + the write path guard.

Pure-function level (mirrors test_nodeset_watcher.py); the HTTP surface is
covered by ``api/platform/test_components_source.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.components.nodeset_source import (
    ScopedEditError,
    SourcePathError,
    SourceSyntaxError,
    extract_scoped_view,
    list_package_files,
    resolve_source,
    resolve_target,
    splice_segments,
    syntax_check,
)


def _mk(root: Path, rel: str, text: str = "# stub\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    """A miniature nodesets tree: one single-file nodeset + one package."""
    _mk(tmp_path, "nodesets/method/mapgpt.py")
    _mk(tmp_path, "nodesets/method/tooleqa/__init__.py")
    _mk(tmp_path, "nodesets/method/tooleqa/_prompts.py")
    _mk(tmp_path, "nodesets/method/tooleqa/sub/extra.py")
    _mk(tmp_path, "nodesets/method/tooleqa/__pycache__/junk.py")
    _mk(tmp_path, "nodesets/method/tooleqa/notes.txt")
    return tmp_path


def _roots(ws: Path) -> tuple[Path, ...]:
    return (ws / "nodesets",)


# ── resolve_source ──


def test_resolve_source_single_file(ws: Path) -> None:
    anchor, pkg = resolve_source(ws / "nodesets/method/mapgpt.py")
    assert anchor.name == "mapgpt.py"
    assert pkg is None


def test_resolve_source_package(ws: Path) -> None:
    _anchor, pkg = resolve_source(ws / "nodesets/method/tooleqa/__init__.py")
    assert pkg == ws / "nodesets/method/tooleqa"


# ── list_package_files ──


def test_list_package_files_sorted_and_filtered(ws: Path) -> None:
    files = list_package_files(ws / "nodesets/method/tooleqa")
    assert files == ["__init__.py", "_prompts.py", "sub/extra.py"]  # no __pycache__, no .txt


# ── resolve_target: happy paths ──


def test_target_single_file_own_name(ws: Path) -> None:
    src = ws / "nodesets/method/mapgpt.py"
    assert resolve_target(src, "mapgpt.py", _roots(ws)) == src.resolve()


def test_target_package_sibling_helper(ws: Path) -> None:
    src = ws / "nodesets/method/tooleqa/__init__.py"
    t = resolve_target(src, "_prompts.py", _roots(ws))
    assert t == (ws / "nodesets/method/tooleqa/_prompts.py").resolve()


def test_target_package_nested_file(ws: Path) -> None:
    src = ws / "nodesets/method/tooleqa/__init__.py"
    t = resolve_target(src, "sub/extra.py", _roots(ws))
    assert t == (ws / "nodesets/method/tooleqa/sub/extra.py").resolve()


# ── resolve_target: guard violations ──


def test_target_single_file_rejects_foreign_name(ws: Path) -> None:
    src = ws / "nodesets/method/mapgpt.py"
    with pytest.raises(SourcePathError):
        resolve_target(src, "navgpt.py", _roots(ws))


def test_target_rejects_parent_escape(ws: Path) -> None:
    src = ws / "nodesets/method/tooleqa/__init__.py"
    with pytest.raises(SourcePathError):
        resolve_target(src, "../mapgpt.py", _roots(ws))


def test_target_rejects_absolute_path(ws: Path) -> None:
    src = ws / "nodesets/method/tooleqa/__init__.py"
    outside = _mk(ws, "outside.py")
    with pytest.raises(SourcePathError):
        resolve_target(src, str(outside), _roots(ws))


def test_target_rejects_non_py(ws: Path) -> None:
    src = ws / "nodesets/method/tooleqa/__init__.py"
    with pytest.raises(SourcePathError):
        resolve_target(src, "notes.txt", _roots(ws))


def test_target_rejects_outside_roots(ws: Path) -> None:
    """Even a legal-looking layout outside the nodesets roots is refused."""
    stray = _mk(ws, "elsewhere/pkg/__init__.py")
    with pytest.raises(SourcePathError):
        resolve_target(stray, "__init__.py", _roots(ws))


def test_target_missing_file_is_not_created(ws: Path) -> None:
    src = ws / "nodesets/method/tooleqa/__init__.py"
    with pytest.raises(FileNotFoundError):
        resolve_target(src, "new_module.py", _roots(ws))
    assert not (ws / "nodesets/method/tooleqa/new_module.py").exists()


# ── syntax_check ──


def test_syntax_check_passes_valid() -> None:
    syntax_check("def f() -> int:\n    return 1\n")


def test_syntax_check_reports_line() -> None:
    with pytest.raises(SourceSyntaxError) as ei:
        syntax_check("def f(:\n    pass\n")
    assert ei.value.lineno == 1
    assert ei.value.msg


# ── scoped view ──

_SCOPED_SRC = '''\
"""Docstring."""

import math

PROMPT = "hello"
STRIDE: int = 3


def _helper_used(x):
    return _helper_inner(x) + STRIDE


def _helper_inner(x):
    return math.floor(x)


def _helper_unused(x):
    return x


class OtherNode:
    node_type = "demo__other"


@some.decorator
class DemoNode:
    node_type = "demo__main"

    def forward(self, inputs):
        return _helper_used(len(PROMPT))
'''


def test_scoped_view_picks_class_globals_and_reachable_functions() -> None:
    segs = extract_scoped_view(_SCOPED_SRC, "demo__main")
    assert segs is not None
    by_kind = {(s["kind"], s["name"]) for s in segs}
    assert ("class", "DemoNode") in by_kind
    assert ("function", "_helper_used") in by_kind
    assert ("function", "_helper_inner") in by_kind  # transitive
    assert ("function", "_helper_unused") not in by_kind
    assert ("class", "OtherNode") not in by_kind
    assert ("globals", "PROMPT, STRIDE") in by_kind
    # sorted by line, ranges match the original text
    starts = [s["start_line"] for s in segs]
    assert starts == sorted(starts)
    lines = _SCOPED_SRC.splitlines(keepends=True)
    for s in segs:
        assert s["text"] == "".join(lines[s["start_line"] - 1 : s["end_line"]])


def test_scoped_view_includes_decorator_line() -> None:
    segs = extract_scoped_view(_SCOPED_SRC, "demo__main")
    cls = next(s for s in segs if s["kind"] == "class")
    assert cls["text"].startswith("@some.decorator\n")


def test_scoped_view_none_when_class_not_here() -> None:
    assert extract_scoped_view(_SCOPED_SRC, "demo__absent") is None


def test_splice_roundtrip_identity() -> None:
    segs = extract_scoped_view(_SCOPED_SRC, "demo__main")
    assert splice_segments(_SCOPED_SRC, segs) == _SCOPED_SRC


def test_splice_applies_edit_and_shifts_lines() -> None:
    segs = extract_scoped_view(_SCOPED_SRC, "demo__main")
    globals_seg = next(s for s in segs if s["kind"] == "globals")
    globals_seg = {**globals_seg, "text": 'PROMPT = "hi"\nEXTRA = 1\nSTRIDE: int = 3\n'}
    out = splice_segments(_SCOPED_SRC, [globals_seg])
    assert 'PROMPT = "hi"' in out and "EXTRA = 1" in out
    assert "class DemoNode:" in out  # rest of file intact
    syntax_check(out)


def test_splice_rejects_overlap_and_out_of_range() -> None:
    with pytest.raises(ScopedEditError):
        splice_segments("a = 1\nb = 2\n", [
            {"start_line": 1, "end_line": 2, "text": "x = 0\n"},
            {"start_line": 2, "end_line": 2, "text": "y = 0\n"},
        ])
    with pytest.raises(ScopedEditError):
        splice_segments("a = 1\n", [{"start_line": 1, "end_line": 5, "text": "x\n"}])


# ── registry helpers (nodeset_mode / nodeset_source_info) ──

_NODESET_SRC = """
from app.components import BaseNodeSet


class _TmpSrceditNodeSet(BaseNodeSet):
    name = "tmp_srcedit"
    description = "tmp nodeset for source-editor tests"

    def get_tools(self):
        return []
"""


def test_nodeset_source_info_before_load(tmp_path: Path) -> None:
    """_source_file is resolvable for a discovered-but-never-loaded nodeset."""
    from app.components.registry import WorkspaceComponentRegistry

    src = _mk(tmp_path, "workspace/nodesets/method/tmp_srcedit.py", _NODESET_SRC)
    reg = WorkspaceComponentRegistry(scan_dir=tmp_path / "workspace")
    reg.scan_all()
    try:
        info = reg.nodeset_source_info("tmp_srcedit")
        assert info is not None
        assert info["source_file"] == src
        assert info["loaded"] is False
        assert info["mode"] == "local"
        assert info["requires_server"] is False
        assert reg.nodeset_source_info("no_such_nodeset") is None
    finally:
        reg.unregister_all()


def test_nodeset_mode_reflects_auto_servers(tmp_path: Path) -> None:
    from app.components.registry import WorkspaceComponentRegistry

    reg = WorkspaceComponentRegistry(scan_dir=tmp_path)
    assert reg.nodeset_mode("anything") == "local"
    reg._auto_servers["tmpns"] = object()  # untagged single-instance key
    reg._auto_servers["other#0"] = object()  # PB-1 tagged multi-worker key
    assert reg.nodeset_mode("tmpns") == "server"
    assert reg.nodeset_mode("other") == "server"
    assert reg.nodeset_mode("tmp") == "local"  # prefix of "tmpns" must not match
