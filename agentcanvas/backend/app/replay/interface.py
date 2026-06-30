"""Replay interface — env-agnostic timeline schema + parser ABC.

Each env nodeset that wants to support log replay declares a class
attribute ``replay_parser`` on its ``BaseNodeSet`` subclass pointing at a
module that contains a ``BaseReplayParser`` subclass. The module must be
importable from the main FastAPI process — i.e. it must not import the
env's underlying simulator (MatterSim, habitat-sim, etc.). It only reads
one episode's log.jsonl, which is plain JSON.

Each eval episode owns a self-contained dir
``outputs/eval_runs/{run_id}/episodes/ep{idx:04d}/`` holding ``log.jsonl``
+ ``assets/``; the parser is handed that episode's ``log.jsonl`` path
directly — no run-level splitting, no boundary detection.

The dataclasses defined here are the contract the frontend consumes.
``frame_url`` is a path relative to the episode's assets directory; the
frontend prepends ``/api/logs/{execution_id}/assets/`` to fetch the bytes,
where ``execution_id`` is ``{run_id}_ep{idx:04d}``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ReplayStep:
    step_index: int
    frame_url: str
    info: dict[str, Any] = field(default_factory=dict)
    # Smooth-mode pose (v1): position [x,y,z] habitat frame + yaw angle.
    # Populated by env-specific parsers when supports_smooth_mode() is True.
    render_params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayEpisode:
    episode_index: int
    episode_id: str
    instruction: str
    step_count: int
    steps: list[ReplayStep] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    supports_smooth: bool = False
    scene_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "step_count": self.step_count,
            "steps": [s.to_dict() for s in self.steps],
            "metrics": self.metrics,
            "supports_smooth": self.supports_smooth,
            "scene_id": self.scene_id,
        }


class BaseReplayParser(ABC):
    """Convert one episode's ``log.jsonl`` into a :class:`ReplayEpisode`.

    Each eval episode is persisted to its own dir, so the parser is given
    a single-episode log file — no boundary detection, no splitting.

    Subclass attribute:
        nodeset_name: must match the corresponding ``BaseNodeSet.name``.
    """

    nodeset_name: ClassVar[str]

    @abstractmethod
    def parse(self, episode_log_path: Path) -> ReplayEpisode:
        """Parse one episode's ``log.jsonl`` into a :class:`ReplayEpisode`.

        ``episode_log_path`` points at
        ``outputs/eval_runs/{run_id}/episodes/ep{idx:04d}/log.jsonl``.
        Raises ``FileNotFoundError`` if the log file is missing.
        """

    # ── v1 smooth-mode hooks (optional) ─────────────────────────────────

    def supports_smooth_mode(self) -> bool:
        """Whether this parser can render interpolated frames between steps.

        Default ``False`` — frontend hides the smooth toggle.
        """
        return False

    async def get_smooth_frame(
        self,
        episode_log_path: Path,
        step_index: int,
        t: float,
    ) -> bytes:
        """Render one JPEG between step ``step_index`` and ``step_index+1``.

        ``t`` is in [0, 1] — 0 = start pose, 1 = end pose. Default impl
        raises ``NotImplementedError``; the API surfaces this as a 404.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support smooth mode")

    async def shutdown(self) -> None:
        """Release any persistent resources (e.g. renderer subprocesses).

        Called from :class:`WorkspaceComponentRegistry.shutdown_all` on app shutdown.
        Default is a no-op.
        """
        return None


class GenericReplayParser(BaseReplayParser):
    """Default replay parser usable by any env nodeset.

    Conventions (override ``BaseReplayParser`` if your env breaks them):
        * Instruction / episode_id: read from the env's ``__reset``
          firing (normally the first entry in the episode log).
        * Per-step frame: any output port whose
          ``port_wire_types[port] == "IMAGE"``. Prefer the LAST such output
          produced by an env-prefixed entry in the step (latest view after
          the action took effect); fall back to the first IMAGE seen.
        * Per-step info: short string outputs flattened as
          ``"{node_label}.{key}"``. Env-prefixed entries also lift their
          string outputs into top-level info keys (``viewpoint_id``,
          ``heading``, etc. for whatever the env happens to emit).

    Bound at registration time to a specific nodeset name. Used as the
    automatic fallback for any ``env_*`` nodeset that doesn't declare its
    own ``replay_parser`` module.
    """

    # Output keys never lifted/flattened — too long or already shown elsewhere.
    SKIP_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"navigable_json", "directions", "extras_json", "position_json"}
    )
    MAX_STRING_LEN: ClassVar[int] = 800

    def __init__(self, nodeset_name: str) -> None:
        self.nodeset_name = nodeset_name
        self._env_prefix = f"{nodeset_name}__"
        self._reset_node_type = f"{nodeset_name}__reset"

    @staticmethod
    def _read_entries(episode_log_path: Path) -> list[dict[str, Any]]:
        """Read every JSON entry from one episode's ``log.jsonl``."""
        entries: list[dict[str, Any]] = []
        with episode_log_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return entries

    def parse(self, episode_log_path: Path) -> ReplayEpisode:
        if not episode_log_path.exists():
            raise FileNotFoundError(f"log.jsonl not found: {episode_log_path}")

        entries = self._read_entries(episode_log_path)

        # The env's reset firing carries instruction + episode_id. It is
        # normally the first entry, but scan to be robust to graphs that
        # fire a setup node ahead of reset.
        reset_outputs: dict[str, Any] = {}
        for entry in entries:
            if entry.get("node_type") == self._reset_node_type:
                reset_outputs = entry.get("outputs", {}) or {}
                break

        instruction = reset_outputs.get("instruction") or reset_outputs.get("question") or ""
        episode_id = reset_outputs.get("episode_id") or ""

        steps_by_num: dict[int, list[dict[str, Any]]] = {}
        for entry in entries:
            s = int(entry.get("step", 0))
            steps_by_num.setdefault(s, []).append(entry)

        steps: list[ReplayStep] = []
        for step_num in sorted(steps_by_num.keys()):
            step_entries = steps_by_num[step_num]
            frame_url = self._pick_frame(step_entries)
            info = self._collect_info(step_entries)
            info.setdefault("instruction", instruction)
            steps.append(
                ReplayStep(
                    step_index=step_num,
                    frame_url=frame_url,
                    info=info,
                )
            )

        # episode_index is run-context the parser doesn't have — the
        # replay router stamps it onto the payload after parse().
        return ReplayEpisode(
            episode_index=0,
            episode_id=episode_id,
            instruction=instruction,
            step_count=len(steps),
            steps=steps,
            metrics={},
        )

    def _pick_frame(self, entries: list[dict[str, Any]]) -> str:
        env_image: str | None = None
        any_image: str | None = None
        for entry in entries:
            wire_types = entry.get("port_wire_types") or {}
            outputs = entry.get("outputs") or {}
            node_type = entry.get("node_type", "")
            for port, wt in wire_types.items():
                if wt != "IMAGE":
                    continue
                asset = outputs.get(port)
                if not isinstance(asset, dict):
                    continue
                path = asset.get("path")
                if not path:
                    continue
                if any_image is None:
                    any_image = path
                if node_type.startswith(self._env_prefix):
                    env_image = path
        return env_image or any_image or ""

    def _collect_info(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        info: dict[str, Any] = {}
        for entry in entries:
            outputs = entry.get("outputs") or {}
            node_label = entry.get("node_label") or entry.get("node_type") or "node"
            node_type = entry.get("node_type", "")
            if node_type.startswith(self._env_prefix):
                # Lift env's own short outputs into top-level keys —
                # the env knows best what's load-bearing per step. v1
                # smooth-mode parsers read pose values straight from here.
                for key, val in outputs.items():
                    if key in self.SKIP_KEYS:
                        continue
                    if self._is_short_value(val):
                        info[key] = val
            for k, v in outputs.items():
                if not isinstance(v, str):
                    continue
                if len(v) > self.MAX_STRING_LEN:
                    continue
                if k in self.SKIP_KEYS:
                    continue
                info[f"{node_label}.{k}"] = v
        return info

    @staticmethod
    def _is_short_value(val: Any) -> bool:
        """Whether a value is a small primitive worth lifting into info.

        Accepts: short str, bool, int, float, list of <=4 numbers,
        dict whose values are all numeric / short-string.
        Skips numpy/asset dicts (carry ``__type``) and long strings.
        """
        if isinstance(val, str):
            return 0 < len(val) <= 200
        if isinstance(val, (bool, int, float)):
            return True
        if isinstance(val, list):
            return len(val) <= 4 and all(isinstance(x, (int, float, bool)) for x in val)
        if isinstance(val, dict):
            if "__type" in val:  # asset / ndarray serialization stub
                return False
            if len(val) > 6:
                return False
            for v in val.values():
                if isinstance(v, (int, float, bool)):
                    continue
                if isinstance(v, str) and len(v) <= 200:
                    continue
                if (
                    isinstance(v, list)
                    and len(v) <= 4
                    and all(isinstance(x, (int, float, bool)) for x in v)
                ):
                    continue
                return False
            return True
        return False
