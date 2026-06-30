"""Nodeset-source watcher — hot-reloads nodeset ``.py`` edits without a restart.

Companion to :mod:`graph_watcher` (which watches ``*.json`` graphs). AgentCanvas
is driven by coding agents that edit nodeset Python source directly on disk.
Eval subprocesses already re-scan source from disk on every launch
(``eval_subprocess_main._build_local_registry_for_run`` calls ``scan_all``), so a
nodeset code edit takes effect in the *next eval* for ``local`` / ``replicated``
/ ``env`` nodesets with no action. The gap this closes is the **long-lived
backend's own in-process registry** — live canvas Play, and ``shared``
auto_host servers an eval attaches to — which keeps the code it imported at
startup until an explicit ``/api/components/reload`` or a full restart.

This stdlib mtime poll (no extra dependency, no watcher thread — runs inside the
FastAPI event loop, like ``graph_watcher``) fingerprints ``nodesets/**/*.py``
and, on a *settled* change, calls
:meth:`WorkspaceComponentRegistry.hot_reload_nodeset_sources`, which reloads only the
affected nodesets and then broadcasts ``components_changed``.

Conservatism (honours "don't tear down GPU servers on every save"):
  * **Debounced** — acts only after sources stop changing for one extra tick,
    so a burst of saves (or a write-temp-then-rename) coalesces into one
    reload.
  * **Run-guarded** — defers while a canvas Play holds the ``ExecutionGuard``,
    so a nodeset instance is never swapped under a live in-process executor.
    (Subprocess evals don't hold the parent guard and own a separate registry,
    so they're unaffected.)
  * **Targeted** — only the changed nodesets are reloaded; auto-hosted SERVER
    nodesets (often GPU-backed, possibly serving a live eval) are re-discovered
    and flagged stale rather than torn down. Unrelated nodesets are untouched.

Not handled in v1: source *deletions* (a removed file's nodeset lingers until a
manual ``/reload``) and underscore helper modules shared across nodesets
(reported as ``unresolved`` — eval auto-picks-them-up; the parent needs a manual
``/reload``). Both are logged.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import WSMessage
from ..state import ExecutionGuard, ExecutionMode, broadcast

if TYPE_CHECKING:
    from ..components.registry import WorkspaceComponentRegistry

log = logging.getLogger("agentcanvas.nodeset_watcher")

_POLL_INTERVAL_SEC = 1.5


def roots_for(registry: WorkspaceComponentRegistry) -> tuple[Path, ...]:
    """Nodeset source roots: frozen workspace + optional active overlay."""
    roots = [registry._frozen_dir / "nodesets"]
    if registry._active_dir is not None:
        roots.append(registry._active_dir / "nodesets")
    return tuple(roots)


def _scan_mtimes(roots: tuple[Path, ...]) -> dict[str, int]:
    """Per-file fingerprint: ``{path: mtime_ns}`` for every ``*.py`` (incl. ``_``
    helpers — a helper edit must still trigger a reload attempt)."""
    out: dict[str, int] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            try:
                out[str(p)] = p.stat().st_mtime_ns
            except OSError:
                # Vanished mid-scan (concurrent agent edit) — next poll settles.
                continue
    return out


def _changed_files(baseline: dict[str, int], current: dict[str, int]) -> set[Path]:
    """New or modified files (deletions are not acted on in v1)."""
    return {Path(p) for p, m in current.items() if baseline.get(p) != m}


async def run_nodeset_watch_loop(
    registry: WorkspaceComponentRegistry,
    interval: float = _POLL_INTERVAL_SEC,
) -> None:
    """Poll nodeset source and hot-reload on a settled change.

    Cancel the task (on app shutdown) to stop. The initial signature is
    captured silently so startup never fires a spurious reload.
    """
    roots = roots_for(registry)
    baseline = _scan_mtimes(roots)
    log.info("nodeset watcher polling %s every %.1fs", [str(r) for r in roots], interval)
    pending: dict[str, int] | None = None  # snapshot awaiting stabilization
    while True:
        await asyncio.sleep(interval)
        try:
            current = _scan_mtimes(roots)
        except Exception:
            log.exception("nodeset watcher signature scan failed")
            continue

        if current == baseline:
            pending = None
            continue

        # Debounce: require the changed snapshot to be stable for one extra
        # tick before acting, so a burst of saves coalesces into one reload.
        if pending != current:
            pending = current
            continue

        # Stabilized. Defer while a canvas Play is in-process — don't swap a
        # nodeset instance under a live executor. Keep `pending` so we retry.
        if ExecutionGuard.current().get("mode") != ExecutionMode.idle.value:
            log.info("nodeset watcher: change settled but canvas active — deferring reload")
            continue

        changed = _changed_files(baseline, current)
        baseline = current
        pending = None
        if not changed:
            continue

        try:
            result = await registry.hot_reload_nodeset_sources(changed)
        except Exception:
            log.exception("nodeset watcher: hot-reload raised")
            continue

        log.info(
            "nodeset hot-reload: reloaded=%s stale_servers=%s discovered=%s unresolved=%s",
            result.get("reloaded"),
            result.get("stale_servers"),
            result.get("discovered"),
            result.get("unresolved"),
        )
        if result.get("stale_servers"):
            log.warning(
                "nodeset watcher: %s run as server-mode and were NOT auto-reloaded "
                "(GPU/eval-shared) — POST /api/components/reload to apply their new code",
                result["stale_servers"],
            )
        if result.get("unresolved"):
            log.info(
                "nodeset watcher: %s could not be attributed to a single nodeset "
                "(helper module or deletion) — eval picks them up on next launch; "
                "POST /api/components/reload to refresh the parent",
                result["unresolved"],
            )
        try:
            await broadcast(WSMessage(type="components_changed", data=result))
        except Exception:
            log.exception("nodeset watcher broadcast failed")
