from __future__ import annotations

"""Run artifacts in the coding-agent browse layout.

The VLA harness is a coding-agent harness like the others — an outer model
driving a tool through a loop — so its runs belong in the same browser, not in
a page of their own. That browser reads one layout:

    outputs/beta-vla-harness/<run_name>/
        summary.json          run metadata + aggregate metrics
        episode_{i}.jsonl     {"t": secs, "kind": ..., ...} one event per line

Same schema the eharness driver writes, so the existing reader serves it with
one line added to SOURCE_ROOTS. Every event the outer model produces goes in —
including `thinking`, because the whole point of watching this arm is seeing
how the planner reasons about what the policy just did.

Images stay out of the jsonl (they are in the live view); tool results here are
telemetry only, so an episode log stays a few hundred KB and greps cleanly.

last updated: 2026-08-07
"""

import json
import os
import time
from typing import Any

REPO_ROOT = os.path.expanduser("~/Desktop/Projects/AgentCanvas")
ROOT = os.path.join(REPO_ROOT, "outputs", "beta-vla-harness")


def _trunc(v: Any, n: int = 4000) -> Any:
    """Cap size WITHOUT changing type.

    The reader is typed against these fields — it does `accepted.join(...)`,
    `metrics.success + 0`. Stringifying a list to bound its length turns a
    render into a TypeError and blanks the page, so numbers stay numbers and
    lists stay lists; only oversized values degrade to a truncated string.
    """
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + f"…<+{len(v) - n}>"
    try:
        s = json.dumps(v, ensure_ascii=False, default=str)
    except Exception:
        return str(v)[:n]
    return v if len(s) <= n else s[:n] + f"…<+{len(s) - n}>"


class EventLog:
    """One run. Call `episode()` to open each episode's log."""

    def __init__(self, run_name: str, meta: dict[str, Any]) -> None:
        self.dir = os.path.join(ROOT, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.run_name = run_name
        self.meta = {"run_name": run_name, "cell": run_name, **meta}
        # The browser reads model/skill out of `config`.
        self.meta.setdefault("config", {"model": meta.get("model"),
                                        "skill": meta.get("arm")})
        self._f = None
        self._live: str | None = None
        self._seq = 0
        self._manifest: list[dict[str, Any]] = []
        self._t0 = time.time()
        self.write_summary([])

    # ── per-episode ───────────────────────────────────────────────────

    def episode(self, index: int) -> None:
        self.close()
        self._f = open(os.path.join(self.dir, f"episode_{index}.jsonl"), "w")
        self._t0 = time.time()
        self._live = os.path.join(self.dir, f"live_{index}")
        os.makedirs(self._live, exist_ok=True)
        self._seq = 0
        self._manifest: list[dict[str, Any]] = []

    def frames(self, out: dict[str, Any], full: bool = False) -> list[str]:
        """Dump one dispatch's views as obs_*.jpg — the layout the browser's
        frame endpoint serves. Named so lexical sort is chronological.

        `full` writes what the outer agent was actually shown: the sampled route
        plus the three current views, at native resolution and near-lossless.
        Those frames leave the model's context the moment the judgment is made,
        so this file is the only durable record of the evidence a decision rested
        on — a thumbnail would not be able to settle "was the archway visible?".
        `frames.json` indexes them so a later question can find them without
        anything re-entering a context window.
        """
        if getattr(self, "_live", None) is None:
            return []
        import base64
        import io

        try:
            from PIL import Image
        except Exception:
            return []

        self._seq += 1
        if full:
            # Exactly what the reviewer was shown for this dispatch: the segment
            # it walked, then the three views it ended in.
            steps = out.get("_leg_frame_steps") or []
            acts = out.get("_actions") or []
            items = [(f"seg{i:02d}", b) for i, b in enumerate(out.get("_frames_leg") or [])]
            items += [(k, out.get(f"_frame_{k}"))
                      for k in ("front", "left", "back", "right")]
            meta = {"step_in_dispatch": steps,
                    "action_at_step": [acts[k] if k < len(acts) else None for k in steps],
                    "episode_step_at_start": out.get("steps_at_start")}
        else:
            items = [(f"{i}", b) for i, b in enumerate(out.get("_frames_leg") or [])]
            items += [(k, out.get(f"_frame_{k}")) for k in ("left", "right")]
            meta = {}

        written = []
        for tag, b64 in items:
            if not b64:
                continue
            name = f"obs_{self._seq:03d}_{tag}.jpg"
            try:
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                if not full:
                    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                img.save(os.path.join(self._live, name), "JPEG",
                         quality=95 if full else 80)
                written.append(name)
            except Exception:
                pass

        if full and written:
            self._manifest.append({
                "dispatch": self._seq,
                "sub_instruction": out.get("sub_instruction"),
                "stop_reason": out.get("stop_reason"),
                "steps_used": out.get("steps_used"),
                "net_displacement_m": out.get("net_displacement_m"),
                "files": written,
                **meta,
            })
            with open(os.path.join(self._live, "frames.json"), "w") as f:
                json.dump(self._manifest, f, ensure_ascii=False, indent=2, default=str)
        return written

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def emit(self, kind: str, _cap: int = 4000, **fields: Any) -> None:
        """`_cap` raises the per-field size limit for events whose whole point
        is the text — a thinking trace cut at 4000 chars is not a trace."""
        if self._f is None:
            return
        row = {"t": round(time.time() - self._t0, 2), "kind": kind}
        row.update({k: _trunc(v, _cap) for k, v in fields.items() if v is not None})
        self._f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    # ── run-level ─────────────────────────────────────────────────────

    def write_summary(self, rows: list[dict[str, Any]]) -> None:
        """`episodes` is a LIST in this layout, not a count — the browser feeds
        it straight to display_aggregate, which rescores from `metrics` per
        episode rather than trusting any aggregate we write."""
        out = dict(self.meta)
        out["episodes"] = [
            {
                "index": r.get("index"),
                "episode_id": r.get("episode_id"),
                "scene_id": r.get("scene_id"),
                "instruction": r.get("instruction"),
                "metrics": {
                    k: r.get(k) for k in
                    ("success", "spl", "ndtw", "oracle_success", "distance_to_goal")
                },
                "agent": {
                    "called_stop": r.get("end_reason") == "planner_stop",
                    "dispatches": r.get("n_execute"),
                    "turns": r.get("planner_turns"),
                    "forced_finish": r.get("forced_finish"),
                    "legs": r.get("legs"),
                    "sub_instructions": r.get("sub_instructions"),
                },
                "steps": r.get("steps"),
                "end_reason": r.get("end_reason"),
                "wall_s": r.get("wall_s"),
            }
            for r in sorted(rows, key=lambda r: r.get("index", 0))
        ]
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
