from __future__ import annotations

"""HM-EQA replay parser with v1 smooth-mode support.

Inherits :class:`GenericReplayParser`'s step parsing (single-episode
log.jsonl, IMAGE-port frame extraction, lifted numeric outputs for
position/angle/floor_height). Adds:

- ``supports_smooth_mode() -> True``
- ``scene_id`` lifted from ``env_hmeqa__episode_info`` entries
- ``get_smooth_frame(...)`` that lerps pose between adjacent steps
  and calls a habitat-sim renderer subprocess (managed by
  :class:`ReplayRendererClient`).

The renderer subprocess is lazy-spawned on first frame request and
held warm for the FastAPI process lifetime; killed via
:meth:`shutdown` when the registry tears down.
"""


import math
import os
from pathlib import Path

from app.components import conda_env_python
from app.replay.interface import GenericReplayParser, ReplayEpisode
from app.replay.renderer_client import ReplayRendererClient

_THIS_DIR = Path(__file__).resolve().parent
_SHIM_PATH = (
    _THIS_DIR.parent.parent.parent.parent
    / "scripts"
    / "install"
    / "hmeqa_libs"
    / "nvidia_egl_workaround.so"
)


def _renderer_env() -> dict:
    env: dict = {}
    if _SHIM_PATH.exists():
        env["LD_PRELOAD"] = str(_SHIM_PATH)
    # Forward any HMEQA_* overrides the parent process has set so the
    # renderer points at the same scene root.
    for var in ("HMEQA_SCENE_ROOT", "HMEQA_DATA_ROOT"):
        if var in os.environ:
            env[var] = os.environ[var]
    return env


class HMEQAReplayParser(GenericReplayParser):
    """Replay parser for env_hmeqa with smooth-mode rendering."""

    def __init__(self) -> None:
        super().__init__("env_hmeqa")
        self._renderer: ReplayRendererClient | None = None
        # Cache the parsed episode so per-frame requests don't re-walk its
        # log.jsonl. Keyed by the episode log path. Bounded to one episode —
        # smooth-mode play is sequential and switching episodes invalidates.
        self._cached: tuple | None = None  # (log_path_str, ReplayEpisode)

    # ── v1 smooth-mode hooks ──────────────────────────────────────────

    def supports_smooth_mode(self) -> bool:
        return True

    async def get_smooth_frame(
        self,
        episode_log_path: Path,
        step_index: int,
        t: float,
    ) -> bytes:
        episode = self._get_or_parse(episode_log_path)
        steps = episode.steps
        if step_index < 0 or step_index >= len(steps) - 1:
            raise IndexError(
                f"step_index {step_index} out of range "
                f"(need [0, {len(steps) - 1}) for smooth interpolation)"
            )
        scene = episode.scene_id
        if not scene:
            raise RuntimeError(
                "scene_id missing from episode — cannot render. "
                "Add env_hmeqa__episode_info to the graph or supply scene_id."
            )

        start = steps[step_index]
        end = steps[step_index + 1]
        start_pos = _pose_position(start.info)
        end_pos = _pose_position(end.info)
        start_angle = _pose_angle(start.info)
        end_angle = _pose_angle(end.info)

        t = max(0.0, min(1.0, float(t)))
        pos_t = [
            start_pos[0] * (1 - t) + end_pos[0] * t,
            start_pos[1] * (1 - t) + end_pos[1] * t,
            start_pos[2] * (1 - t) + end_pos[2] * t,
        ]
        angle_t = _lerp_angle(start_angle, end_angle, t)

        client = self._ensure_renderer()
        return await client.post_for_bytes(
            "/render",
            {"scene": scene, "position": pos_t, "angle": angle_t},
        )

    async def shutdown(self) -> None:
        if self._renderer is not None:
            await self._renderer.stop()
            self._renderer = None
        self._cached = None

    # ── Parsing — augment Generic with scene_id ───────────────────────

    def parse(self, episode_log_path: Path) -> ReplayEpisode:
        episode = super().parse(episode_log_path)
        episode.supports_smooth = True
        episode.scene_id = self._extract_scene_from_log(episode_log_path)
        return episode

    def _extract_scene_from_log(self, episode_log_path: Path) -> str:
        for entry in self._read_entries(episode_log_path):
            if entry.get("node_type") == "env_hmeqa__episode_info":
                outputs = entry.get("outputs") or {}
                scene = outputs.get("scene")
                if isinstance(scene, str) and scene:
                    return scene
        return ""

    # ── Internals ─────────────────────────────────────────────────────

    def _get_or_parse(self, episode_log_path: Path) -> ReplayEpisode:
        key = str(episode_log_path)
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]
        episode = self.parse(episode_log_path)
        self._cached = (key, episode)
        return episode

    def _ensure_renderer(self) -> ReplayRendererClient:
        if self._renderer is None:
            self._renderer = ReplayRendererClient(
                renderer_file=_THIS_DIR / "hmeqa_renderer.py",
                class_name="HMEQARendererServer",
                python=conda_env_python("ac-hmeqa", "HMEQA_PYTHON"),
                env=_renderer_env(),
                startup_timeout=600,
            )
        return self._renderer


# ══════════════════════════════════════════════════════════════════════
# Pose helpers
# ══════════════════════════════════════════════════════════════════════


def _pose_position(info: dict) -> list:
    pose = info.get("pose")
    if isinstance(pose, dict):
        pos = pose.get("position")
        if isinstance(pos, list) and len(pos) == 3:
            return [float(v) for v in pos]
    raise RuntimeError(
        "missing pose.position in step info — cannot interpolate. "
        "Generic parser should have lifted this from env_hmeqa__step "
        "outputs; check the run was produced by a recent hmeqa nodeset."
    )


def _pose_angle(info: dict) -> float:
    angle = info.get("angle")
    if isinstance(angle, (int, float)):
        return float(angle)
    raise RuntimeError("missing 'angle' in step info — cannot interpolate")


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Shortest-path interpolation for yaw — wraps via (cos, sin) blend."""
    cos_t = math.cos(a) * (1 - t) + math.cos(b) * t
    sin_t = math.sin(a) * (1 - t) + math.sin(b) * t
    return math.atan2(sin_t, cos_t)
