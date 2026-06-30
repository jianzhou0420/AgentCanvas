"""Unit tests for the nodeset-source watcher's pure logic.

Covers file→nodeset attribution (``_resolve_nodeset_reimport``) and the
new/modified-file diff (``_changed_files``). The actual reload + broadcast loop
is integration-level (needs a live registry + event loop) and isn't covered
here.
"""

from __future__ import annotations

from pathlib import Path

from app.components.registry import WorkspaceComponentRegistry
from app.services.nodeset_watcher import _changed_files, roots_for


def _mk(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# stub\n")
    return p


def test_resolve_single_file(tmp_path: Path) -> None:
    r = WorkspaceComponentRegistry(str(tmp_path))
    f = _mk(tmp_path, "nodesets/method/explore_eqa.py")
    res = r._resolve_nodeset_reimport(f)
    assert res is not None
    assert res[0] == "nodesets/method"


def test_resolve_package(tmp_path: Path) -> None:
    r = WorkspaceComponentRegistry(str(tmp_path))
    _mk(tmp_path, "nodesets/method/tooleqa/__init__.py")
    f = _mk(tmp_path, "nodesets/method/tooleqa/explore.py")
    res = r._resolve_nodeset_reimport(f)
    assert res is not None
    assert res[0] == "nodesets/method"


def test_resolve_underscore_helper_is_none(tmp_path: Path) -> None:
    """Underscore helper modules are imported BY nodesets but not scanned —
    they can't be attributed to one nodeset, so the watcher leaves them
    unresolved (a manual /reload refreshes the parent; eval re-scans anyway)."""
    r = WorkspaceComponentRegistry(str(tmp_path))
    f = _mk(tmp_path, "nodesets/method/_explore_eqa_tsdf.py")
    assert r._resolve_nodeset_reimport(f) is None


def test_resolve_outside_nodesets_is_none(tmp_path: Path) -> None:
    r = WorkspaceComponentRegistry(str(tmp_path))
    f = _mk(tmp_path, "graphs/foo.json")
    assert r._resolve_nodeset_reimport(f) is None


def test_resolve_underscore_role_is_none(tmp_path: Path) -> None:
    r = WorkspaceComponentRegistry(str(tmp_path))
    f = _mk(tmp_path, "nodesets/_private/foo.py")
    assert r._resolve_nodeset_reimport(f) is None


def test_changed_files_detects_modified_and_new() -> None:
    baseline = {"a.py": 100, "b.py": 200}
    current = {"a.py": 100, "b.py": 999, "c.py": 300}  # b modified, c new, a same
    changed = _changed_files(baseline, current)
    assert changed == {Path("b.py"), Path("c.py")}


def test_changed_files_ignores_deletions() -> None:
    """v1 does not act on deletions — a removed file isn't in `current`."""
    baseline = {"a.py": 100, "b.py": 200}
    current = {"a.py": 100}  # b deleted
    assert _changed_files(baseline, current) == set()


def test_roots_includes_active_overlay(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    active = tmp_path / "active"
    r = WorkspaceComponentRegistry(str(frozen), active_dir=str(active))
    roots = roots_for(r)
    assert frozen / "nodesets" in roots
    assert active / "nodesets" in roots
