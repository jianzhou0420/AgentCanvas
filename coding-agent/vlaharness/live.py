from __future__ import annotations

"""Live-state writer for the VLA Harness tab.

The viewer is a page in the AgentCanvas frontend, but the run lives in a
different conda env talking to two auto_hosts on other ports — so instead of an
API route and a websocket topic, the harness drops a JSON file into the Vite
dev server's ``public/`` directory and the page polls it. Additive on both
sides: no backend endpoint, no shared server state.

Writes are atomic (tmp + rename) so a poll never reads a half-written file, and
frames are capped so the file stays small enough to fetch every second.
"""

import base64
import json
import os
import tempfile
import time
from typing import Any

DEFAULT_PATH = os.path.expanduser(
    "~/Desktop/Projects/AgentCanvas/agentcanvas/frontend/public/vla_live/state.json"
)
# One frame per dispatch, and only the most recent few episodes carry theirs.
# The page refetches the whole file every second, so frames are JPEG at viewing
# size — the same episode as PNG-384 was 7.9 MB, which is not a poll payload.
FRAME_EPISODES = 3
FRAMES_PER_EPISODE = 12


class LiveWriter:
    def __init__(self, path: str | None, arm: str, total: int, planner_model: str | None):
        self.path = path or DEFAULT_PATH
        self.enabled = bool(self.path)
        self.state: dict[str, Any] = {
            "arm": arm,
            "planner_model": planner_model,
            "total": total,
            "updated_at": "",
            "episodes": [],
        }
        if self.enabled:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.flush()

    # ── episode lifecycle ─────────────────────────────────────────────

    def start_episode(self, index: int, instruction: str, episode_id: Any = None) -> None:
        if not self.enabled:
            return
        self.state["episodes"].append(
            {
                "index": index,
                "episode_id": episode_id,
                "instruction": instruction,
                "status": "running",
                "events": [],
            }
        )
        self.flush()

    def _cur(self) -> dict[str, Any] | None:
        eps = self.state["episodes"]
        return eps[-1] if eps else None

    def plan(self, text: str) -> None:
        cur = self._cur()
        if not self.enabled or cur is None or not text.strip():
            return
        cur["events"].append({"t": "plan", "text": text.strip()})
        self.flush()

    def execute(self, sub: str, result: dict[str, Any], frame_b64: str | None) -> None:
        cur = self._cur()
        if not self.enabled or cur is None:
            return
        self._trim_frames()
        ev: dict[str, Any] = {
            "t": "execute",
            "sub": sub,
            "result": {k: v for k, v in result.items() if not k.startswith("_")},
        }
        if frame_b64:
            ev["frame"] = frame_b64
        cur["events"].append(ev)
        self.flush()

    def finish(self, forced: bool = False) -> None:
        cur = self._cur()
        if not self.enabled or cur is None:
            return
        cur["events"].append({"t": "finish", "forced": forced})
        self.flush()

    def end_episode(self, row: dict[str, Any]) -> None:
        cur = self._cur()
        if not self.enabled or cur is None:
            return
        cur.update(
            {
                "status": "done",
                "success": row.get("success"),
                "distance_to_goal": row.get("distance_to_goal"),
                "steps": row.get("steps"),
                "n_execute": row.get("n_execute"),
            }
        )
        self._trim_frames()
        self.flush()

    # ── plumbing ──────────────────────────────────────────────────────

    def _trim_frames(self) -> None:
        """Keep images only on the most recent episodes, and only the most
        recent dispatches within them — a thrashing episode can rack up 30."""
        for ep in self.state["episodes"][:-FRAME_EPISODES]:
            for ev in ep["events"]:
                ev.pop("frame", None)
        for ep in self.state["episodes"][-FRAME_EPISODES:]:
            framed = [ev for ev in ep["events"] if "frame" in ev]
            for ev in framed[:-FRAMES_PER_EPISODE]:
                ev.pop("frame", None)

    def flush(self) -> None:
        if not self.enabled:
            return
        self.state["updated_at"] = time.strftime("%H:%M:%S")
        d = os.path.dirname(self.path)
        try:
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f, ensure_ascii=False)
            os.replace(tmp, self.path)  # atomic — a poll never sees a partial file
        except Exception:
            pass


def shrink_for_view(b64: str, px: int = 288) -> str | None:
    """Downscale + JPEG for the viewer. Returns a full data: URI so the page can
    drop it straight into an <img src>."""
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if max(img.size) > px:
            s = px / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None
