"""Read/write resolution for nodeset source files (canvas source editor).

Pure functions — no registry or services dependency — so the layout rules
and the path guard are unit-testable in isolation. The API layer
(``api/platform/components.py``) resolves a nodeset name to its
``_source_file`` via the registry, then delegates here.

Layout rules mirror :mod:`content_hash`: a nodeset anchored at
``__init__.py`` is a package (every ``.py`` under its directory is fair
game); anything else is a single-file nodeset (only that file itself is
editable). Role-level underscore helpers (e.g. ``nodesets/method/_x.py``)
belong to no nodeset and are unreachable through this module.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

_EXCLUDED_DIR_NAMES = {"__pycache__"}


class SourcePathError(ValueError):
    """Requested file violates the nodeset-source path guard."""


class SourceSyntaxError(ValueError):
    """Submitted content is not valid Python."""

    def __init__(self, msg: str, lineno: int | None, offset: int | None) -> None:
        super().__init__(msg)
        self.msg = msg
        self.lineno = lineno
        self.offset = offset


def resolve_source(source_file: str | Path) -> tuple[Path, Path | None]:
    """Split a nodeset's ``_source_file`` into ``(anchor_file, package_dir)``.

    ``package_dir`` is the package directory when the anchor is an
    ``__init__.py``, else ``None`` (single-file nodeset).
    """
    anchor = Path(source_file)
    package_dir = anchor.parent if anchor.name == "__init__.py" else None
    return anchor, package_dir


def list_package_files(package_dir: Path) -> list[str]:
    """All ``.py`` files of a package as sorted relative POSIX paths.

    Excludes ``__pycache__`` (same rule as ``content_hash.hash_nodeset_tree``).
    """
    files: list[str] = []
    for p in package_dir.rglob("*.py"):
        rel = p.relative_to(package_dir)
        if any(part in _EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        files.append(rel.as_posix())
    return sorted(files)


def resolve_target(
    source_file: str | Path,
    rel_file: str,
    roots: Sequence[Path],
) -> Path:
    """Resolve ``rel_file`` against a nodeset's source anchor, guarded.

    Guard: the resolved target must stay inside the nodeset's own
    directory (package dir, or exactly the anchor for single-file
    nodesets), must live under one of the nodeset source ``roots``
    (``nodeset_watcher.roots_for``), and must be an existing ``.py``
    file — no file creation through this seam.

    Raises :class:`SourcePathError` on a guard violation and
    :class:`FileNotFoundError` when the path is legal but absent.
    """
    anchor, package_dir = resolve_source(source_file)

    if package_dir is None:
        if rel_file != anchor.name:
            raise SourcePathError(
                f"single-file nodeset only exposes {anchor.name!r}, not {rel_file!r}"
            )
        base = anchor.parent
    else:
        base = package_dir

    if Path(rel_file).is_absolute():
        raise SourcePathError(f"absolute paths are not allowed: {rel_file!r}")

    target = (base / rel_file).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as e:
        raise SourcePathError(f"{rel_file!r} escapes the nodeset directory") from e

    if target.suffix != ".py":
        raise SourcePathError(f"only .py files are editable, got {rel_file!r}")

    if not any(target.is_relative_to(root.resolve()) for root in roots):
        raise SourcePathError(f"{rel_file!r} resolves outside the nodeset source roots")

    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def syntax_check(content: str, filename: str = "<source>") -> None:
    """``ast.parse`` gate before any write; raises :class:`SourceSyntaxError`."""
    try:
        ast.parse(content, filename=filename)
    except SyntaxError as e:
        raise SourceSyntaxError(e.msg or "invalid syntax", e.lineno, e.offset) from e


# ── Scoped view (per-node source editing) ──
#
# The Source tab shows a node's *slice* of its defining file: module-level
# globals, the node's class, and the module-level functions that class
# (transitively) references. Each segment carries its original 1-based
# line range; edits are spliced back by range and the whole file is
# syntax-checked before writing.


class ScopedEditError(ValueError):
    """Segment ranges are invalid against the current file."""


def _stmt_start_line(node: ast.stmt) -> int:
    """First source line of a statement, decorators included."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def _is_node_type_class(node: ast.ClassDef, node_type: str) -> bool:
    """True if the class body assigns ``node_type = "<node_type>"``."""
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            targets = [stmt.target.id]
            value = stmt.value
        else:
            continue
        if (
            "node_type" in targets
            and isinstance(value, ast.Constant)
            and value.value == node_type
        ):
            return True
    return False


def extract_scoped_view(source_text: str, node_type: str) -> list[dict] | None:
    """A node's editable slice of its defining file, or ``None`` if the
    class isn't defined here (callers then try the package's sibling files).

    Returns segments sorted by line: ``{kind, name, start_line, end_line,
    text}`` with 1-based inclusive ranges. ``kind`` is ``"globals"`` (a
    contiguous run of module-level assignments), ``"function"`` (a
    module-level function the class transitively references by name), or
    ``"class"`` (the node's class itself).
    """
    tree = ast.parse(source_text)

    target_cls = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and _is_node_type_class(n, node_type)
        ),
        None,
    )
    if target_cls is None:
        return None

    # Module-level functions transitively referenced from the class body.
    module_funcs: dict[str, ast.stmt] = {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    needed: set[str] = set()
    frontier: list[ast.stmt] = [target_cls]
    while frontier:
        scope = frontier.pop()
        for ref in ast.walk(scope):
            if (
                isinstance(ref, ast.Name)
                and ref.id in module_funcs
                and ref.id not in needed
            ):
                needed.add(ref.id)
                frontier.append(module_funcs[ref.id])

    # Module-level assignments, grouped into contiguous runs.
    global_blocks: list[list[ast.stmt]] = []
    run: list[ast.stmt] = []
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            run.append(n)
        elif run:
            global_blocks.append(run)
            run = []
    if run:
        global_blocks.append(run)

    lines = source_text.splitlines(keepends=True)

    def segment(kind: str, name: str, start: int, end: int) -> dict:
        return {
            "kind": kind,
            "name": name,
            "start_line": start,
            "end_line": end,
            "text": "".join(lines[start - 1 : end]),
        }

    def assign_names(stmts: list[ast.stmt]) -> str:
        out: list[str] = []
        for s in stmts:
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            out.extend(t.id for t in targets if isinstance(t, ast.Name))
        return ", ".join(out)

    segments = [
        segment("globals", assign_names(block), block[0].lineno, block[-1].end_lineno)
        for block in global_blocks
    ]
    segments.extend(
        segment("function", name, _stmt_start_line(fn), fn.end_lineno)
        for name, fn in module_funcs.items()
        if name in needed
    )
    segments.append(
        segment(
            "class", target_cls.name, _stmt_start_line(target_cls), target_cls.end_lineno
        )
    )
    segments.sort(key=lambda s: s["start_line"])
    return segments


def splice_segments(source_text: str, edits: list[dict]) -> str:
    """Replace each ``{start_line, end_line, text}`` range with its new text.

    Ranges are 1-based inclusive, must lie within the file and must not
    overlap; raises :class:`ScopedEditError` otherwise. The caller is
    expected to ``syntax_check`` the result before writing.
    """
    lines = source_text.splitlines(keepends=True)
    ordered = sorted(edits, key=lambda e: int(e["start_line"]))
    prev_end = 0
    for e in ordered:
        start, end = int(e["start_line"]), int(e["end_line"])
        if start < 1 or end < start or end > len(lines):
            raise ScopedEditError(f"segment range {start}-{end} outside file")
        if start <= prev_end:
            raise ScopedEditError(f"segment range {start}-{end} overlaps a previous one")
        prev_end = end
    for e in sorted(ordered, key=lambda e: int(e["start_line"]), reverse=True):
        text = str(e["text"])
        if text and not text.endswith("\n"):
            text += "\n"
        lines[int(e["start_line"]) - 1 : int(e["end_line"])] = (
            text.splitlines(keepends=True)
        )
    return "".join(lines)
