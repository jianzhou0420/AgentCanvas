"""Content hashing for nodeset source trees.

TODO #60: lets ``/api/eval/v2/start`` detect whether an
``active_workspace_dir`` overlay has actually changed a ``shared``
nodeset's source vs. the frozen baseline. Only when the hash differs do
we spawn an ephemeral auto_host child for this eval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 16
_EXCLUDED_DIR_NAMES = {"__pycache__"}


def hash_nodeset_tree(source_file: str | Path) -> str:
    """SHA256 of a nodeset's source tree, anchored at ``source_file``.

    Two modes:

    * **Package mode** — ``source_file`` is ``__init__.py``. Hash every
      ``.py`` under the package directory recursively, sorted by relative
      POSIX path, with each entry contributing ``<rel-path>\\0<sha256>\\n``.
    * **Single-file mode** — anything else. Hash just that one file as
      ``<basename>\\0<sha256>\\n``.

    ``__pycache__`` directories are excluded. Non-``.py`` files are
    ignored (no JSON / YAML / weight files in the source tree of a
    nodeset — those live in ``data/`` or model dirs).

    Returns the hex digest of the resulting summary stream. Raises
    ``FileNotFoundError`` if ``source_file`` does not exist; that should
    be checked by the caller via ``Path.exists()`` first.
    """
    src = Path(source_file)
    if not src.exists():
        raise FileNotFoundError(src)

    h = hashlib.sha256()
    if src.name == "__init__.py":
        root = src.parent
        files: list[Path] = []
        for p in root.rglob("*.py"):
            if any(part in _EXCLUDED_DIR_NAMES for part in p.relative_to(root).parts):
                continue
            files.append(p)
        files.sort(key=lambda p: p.relative_to(root).as_posix())
        for p in files:
            rel = p.relative_to(root).as_posix()
            digest = _hash_file(p)
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            h.update(digest.encode("ascii"))
            h.update(b"\n")
    else:
        digest = _hash_file(src)
        h.update(src.name.encode("utf-8"))
        h.update(b"\x00")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_overlay_source(
    frozen_source: str | Path,
    workspace_root: str | Path,
    active_workspace_dir: str | Path,
) -> Path | None:
    """Map a frozen nodeset's ``_source_file`` to the overlay's equivalent.

    Returns the overlay path if it exists on disk, else ``None``.
    ``frozen_source`` must live under ``workspace_root``; otherwise we
    can't compute a relative path and return ``None``.
    """
    fp = Path(frozen_source).resolve()
    ws = Path(workspace_root).resolve()
    try:
        rel = fp.relative_to(ws)
    except ValueError:
        return None
    candidate = Path(active_workspace_dir).resolve() / rel
    return candidate if candidate.exists() else None
