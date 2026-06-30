"""Graph-directory watcher — pushes ``graphs_changed`` when files change on disk.

AgentCanvas is meant to be driven by coding agents (Claude Code) that create,
move, and delete graph JSON files directly on disk, outside the web UI. The
Explorer only re-fetches on mount / after a UI save, so those external edits
would otherwise go unseen until a manual refresh.

This runs a lightweight **stdlib mtime poll** inside the FastAPI event loop —
no extra dependency, no watcher thread (and thus none of the cross-event-loop
hazards a threaded watcher carries). It scans the two small graph roots every
``interval`` seconds, and only when the directory signature changes does it
broadcast a single ``graphs_changed`` WS frame. Polling naturally coalesces a
burst of FS events (e.g. a write-temp-then-rename) into one notification.

Latency is bounded by ``interval`` (~1.5s) — fine for "an agent just edited a
graph file"; not intended for sub-second sync.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..models import WSMessage
from ..state import broadcast

log = logging.getLogger("agentcanvas.graph_watcher")

_POLL_INTERVAL_SEC = 1.5


def _signature(roots: tuple[Path, ...]) -> frozenset[tuple[str, int]]:
    """Cheap directory fingerprint: (path, mtime_ns) for every *.json + dir."""
    entries: set[tuple[str, int]] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() or p.suffix == ".json":
                try:
                    entries.add((str(p), p.stat().st_mtime_ns))
                except OSError:
                    # File vanished mid-scan (concurrent agent edit) — skip;
                    # the next poll picks up the settled state.
                    continue
    return frozenset(entries)


async def run_graph_watch_loop(
    roots: tuple[Path, ...],
    interval: float = _POLL_INTERVAL_SEC,
) -> None:
    """Poll ``roots`` and broadcast ``graphs_changed`` on any change.

    Cancel the task (e.g. on app shutdown) to stop. The initial signature is
    captured silently so startup itself never fires a spurious notification.
    """
    last = _signature(roots)
    log.info("graph watcher polling %s every %.1fs", [str(r) for r in roots], interval)
    while True:
        await asyncio.sleep(interval)
        try:
            current = _signature(roots)
        except Exception:
            log.exception("graph watcher signature scan failed")
            continue
        if current != last:
            last = current
            try:
                await broadcast(WSMessage(type="graphs_changed", data={}))
            except Exception:
                log.exception("graph watcher broadcast failed")
