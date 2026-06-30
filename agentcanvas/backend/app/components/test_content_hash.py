"""Tests for content_hash.hash_nodeset_tree + resolve_overlay_source.

TODO #60 — verifies the hash util used to detect overlay-vs-frozen
nodeset source drift at eval submit time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .content_hash import hash_nodeset_tree, resolve_overlay_source

# ── Single-file nodesets ──


def test_single_file_identical(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("from __future__ import annotations\nclass X: pass\n")
    b.write_text("from __future__ import annotations\nclass X: pass\n")
    # Same name + same content → same hash. (Filename is part of the
    # digest stream, so distinct paths with same content + same basename
    # collide intentionally; reuse one file via cwd shift.)
    h_a = hash_nodeset_tree(a)
    a.rename(tmp_path / "x.py")
    (tmp_path / "x.py").rename(a)
    h_a2 = hash_nodeset_tree(a)
    assert h_a == h_a2


def test_single_file_whitespace_change(tmp_path: Path) -> None:
    f = tmp_path / "vlm.py"
    f.write_text("from __future__ import annotations\nclass X: pass\n")
    h1 = hash_nodeset_tree(f)
    f.write_text("from __future__ import annotations\nclass X:  pass\n")  # 2 spaces
    h2 = hash_nodeset_tree(f)
    assert h1 != h2


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_nodeset_tree(tmp_path / "nope.py")


# ── Package-mode nodesets ──


def test_package_identical(tmp_path: Path) -> None:
    p1 = tmp_path / "ns_a"
    p2 = tmp_path / "ns_b"
    for p in (p1, p2):
        p.mkdir()
        (p / "__init__.py").write_text("from .inner import Y\nclass X: pass\n")
        (p / "inner.py").write_text("class Y: pass\n")
        sub = p / "adapters"
        sub.mkdir()
        (sub / "__init__.py").write_text("\n")
        (sub / "foo.py").write_text("class Z: pass\n")
    h1 = hash_nodeset_tree(p1 / "__init__.py")
    h2 = hash_nodeset_tree(p2 / "__init__.py")
    assert h1 == h2, "byte-identical package trees should hash equal"


def test_package_modify_nested(tmp_path: Path) -> None:
    p = tmp_path / "ns"
    p.mkdir()
    (p / "__init__.py").write_text("class X: pass\n")
    sub = p / "models"
    sub.mkdir()
    (sub / "__init__.py").write_text("\n")
    (sub / "foo.py").write_text("class A: pass\n")

    h_before = hash_nodeset_tree(p / "__init__.py")
    (sub / "foo.py").write_text("class A: pass  # changed\n")
    h_after = hash_nodeset_tree(p / "__init__.py")
    assert h_before != h_after, "nested .py edit must perturb the hash"


def test_package_pycache_excluded(tmp_path: Path) -> None:
    p = tmp_path / "ns"
    p.mkdir()
    (p / "__init__.py").write_text("class X: pass\n")
    h1 = hash_nodeset_tree(p / "__init__.py")

    pycache = p / "__pycache__"
    pycache.mkdir()
    (pycache / "junk.py").write_text("# garbage\n")  # .py inside __pycache__
    (pycache / "__init__.cpython-310.pyc").write_bytes(b"\x00\x00\x00\x00")
    h2 = hash_nodeset_tree(p / "__init__.py")
    assert h1 == h2, "__pycache__/* must not contribute to the hash"


def test_package_non_py_files_ignored(tmp_path: Path) -> None:
    p = tmp_path / "ns"
    p.mkdir()
    (p / "__init__.py").write_text("class X: pass\n")
    h1 = hash_nodeset_tree(p / "__init__.py")
    (p / "README.md").write_text("docs\n")
    (p / "data.json").write_text("{}\n")
    h2 = hash_nodeset_tree(p / "__init__.py")
    assert h1 == h2, "non-.py files must not affect the hash"


def test_package_add_py_file_changes_hash(tmp_path: Path) -> None:
    p = tmp_path / "ns"
    p.mkdir()
    (p / "__init__.py").write_text("class X: pass\n")
    h1 = hash_nodeset_tree(p / "__init__.py")
    (p / "extra.py").write_text("class Y: pass\n")
    h2 = hash_nodeset_tree(p / "__init__.py")
    assert h1 != h2, "adding a .py file under the package must change the hash"


def test_package_rename_py_file_changes_hash(tmp_path: Path) -> None:
    p = tmp_path / "ns"
    p.mkdir()
    (p / "__init__.py").write_text("class X: pass\n")
    (p / "a.py").write_text("# same content\n")
    h1 = hash_nodeset_tree(p / "__init__.py")
    (p / "a.py").rename(p / "b.py")
    h2 = hash_nodeset_tree(p / "__init__.py")
    assert h1 != h2, "relative path is part of the hash stream"


# ── resolve_overlay_source ──


def test_resolve_overlay_present(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    aw = tmp_path / "aw"
    (ws / "nodesets" / "server").mkdir(parents=True)
    (aw / "nodesets" / "server").mkdir(parents=True)
    frozen = ws / "nodesets" / "server" / "vlm.py"
    overlay = aw / "nodesets" / "server" / "vlm.py"
    frozen.write_text("# frozen\n")
    overlay.write_text("# overlay\n")
    result = resolve_overlay_source(frozen, ws, aw)
    assert result == overlay.resolve()


def test_resolve_overlay_missing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    aw = tmp_path / "aw"
    (ws / "nodesets" / "server").mkdir(parents=True)
    aw.mkdir()
    frozen = ws / "nodesets" / "server" / "vlm.py"
    frozen.write_text("# frozen only\n")
    assert resolve_overlay_source(frozen, ws, aw) is None


def test_resolve_overlay_frozen_outside_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    aw = tmp_path / "aw"
    ws.mkdir()
    aw.mkdir()
    outside = tmp_path / "elsewhere" / "vlm.py"
    outside.parent.mkdir()
    outside.write_text("# outside\n")
    assert resolve_overlay_source(outside, ws, aw) is None
