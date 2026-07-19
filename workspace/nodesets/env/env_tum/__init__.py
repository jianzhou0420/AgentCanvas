from __future__ import annotations

"""EnvTumNodeSet — TUM RGB-D benchmark as a replay environment.

A *streaming replay* env: no simulator, no actions, just a pre-recorded RGB-D
sequence (Sturm et al., TUM RGB-D SLAM benchmark) fed one frame at a time so a
downstream SLAM system (``model_pyslam``) can track it live on the canvas.
Structurally it is the interactive-MDP env shape (``reset`` / ``step_*`` /
``observe_*`` / ``evaluate``) with a **degenerate action space**: ``step_replay``
takes no action — it just advances the frame cursor and reports when the
sequence is exhausted. Perception is agent-pulled via ``observe_egocentric``
(the same port shape as ``env_habitat``: rgb / depth / pose / intrinsics), so a
graph built against Habitat re-wires onto TUM with only the env swapped.

Why this exists (2026-07-09): the pySLAM canvas demo needs to *show off* SLAM —
long trajectory, loop closure, dense map. Riding a short VLN-CE policy walk
buries all of that; a classic benchmark sequence (fr3/long_office_household) is
pySLAM's home turf and runs on **CPU alone** (ORB2 + DBOW3, no GPU, no policy).
The ``model_pyslam`` backend is already validated on TUM fr1_xyz RGB-D, so this
nodeset simply supplies the frames the demo was missing.

Local mode (``server_python = None``): the reader is pure numpy + PIL, both in
the agentcanvas env — no dedicated conda env, no container. ``model_pyslam``
keeps its own container boundary; this nodeset just reads PNGs off disk.

Data layout: sequences live under ``data/tum/<rgbd_dataset_freiburgN_...>/``
(gitignored). ``download.sh`` fetches fr3/long_office_household. The env panel's
``split`` is the freiburg camera group (its intrinsics), ``episode`` the
sequence within it.

last updated: 2026-07-09
"""

import asyncio
import concurrent.futures
import contextlib
import logging
import os
from typing import Any, ClassVar

import numpy as np

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)

from ._reader import TumSequence, freiburg_group

log = logging.getLogger("agentcanvas.env_tum")


def _data_root() -> str:
    """Absolute path to ``data/tum/`` (env override: ``TUM_DATA_ROOT``)."""
    env = os.environ.get("TUM_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    # __file__ = workspace/nodesets/env/env_tum/__init__.py → four parents = repo root
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
    )
    return os.path.join(repo_root, "data", "tum")


# ══════════════════════════════════════════════════════════════════════
# TumEnvManager — singleton replay runtime (one active sequence + cursor)
# ══════════════════════════════════════════════════════════════════════


class TumEnvManager:
    """Holds the active TUM sequence, a frame cursor, and the episode placement.

    All public methods are blocking; call via the single-thread executor (PNG
    decode off the event loop). Cursor semantics: ``set_episode`` arms the
    sequence at cursor ``-1``; ``step`` advances first, so the first
    ``observe`` after the first ``step`` yields frame 0 — every frame is
    tracked, none skipped.
    """

    _instance: TumEnvManager | None = None

    def __init__(self) -> None:
        self._seq: TumSequence | None = None
        self._cursor: int = -1
        self._done: bool = False
        self._max_frames: int = 0  # 0 = whole sequence; >0 = head-truncate (quick tests)
        self._split: str = ""
        self._index: int = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="env_tum",
        )

    @classmethod
    def get(cls) -> TumEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return os.path.isdir(_data_root())

    # ── Sequence discovery (backs the env panel cascade) ──

    def _sequence_dirs(self) -> list[str]:
        root = _data_root()
        if not os.path.isdir(root):
            return []
        out = [
            d for d in sorted(os.listdir(root))
            if d.startswith("rgbd_dataset_") and os.path.isdir(os.path.join(root, d))
        ]
        return out

    def list_splits(self) -> list[str]:
        groups = sorted({freiburg_group(d) for d in self._sequence_dirs()})
        groups = [g for g in groups if g != "default"] or groups
        return groups or ["freiburg1", "freiburg2", "freiburg3"]

    def list_episodes(self, split: str) -> list[dict[str, Any]]:
        seqs = [d for d in self._sequence_dirs() if freiburg_group(d) == split]
        return [{"index": i, "sequence": s} for i, s in enumerate(seqs)]

    def _seq_dir_for(self, split: str, index: int) -> str | None:
        seqs = [d for d in self._sequence_dirs() if freiburg_group(d) == split]
        if 0 <= index < len(seqs):
            return os.path.join(_data_root(), seqs[index])
        return None

    # ── Lifecycle ──

    def initialize(self, **kwargs: Any) -> None:
        """Cheap — nothing to open until an episode is placed."""
        log.info("env_tum ready — data root %s (%d sequences)",
                 _data_root(), len(self._sequence_dirs()))

    def shutdown(self) -> None:
        self._seq = None

    def set_max_frames(self, cap: int) -> None:
        """Head-truncate the used frame count (0 = whole sequence). Rebuilds the
        effective length only; the associations are already indexed."""
        self._max_frames = max(0, int(cap or 0))

    # ── Episode control (backs env panel + reset) ──

    def set_episode(self, split: str, index: int) -> dict[str, Any]:
        seq_dir = self._seq_dir_for(split, index)
        if seq_dir is None:
            return {"error": f"no TUM sequence for split={split!r} index={index}"}
        self._seq = TumSequence(seq_dir)
        self._split, self._index = split, index
        self._cursor = -1
        self._done = False
        log.info("env_tum placed: %s (%d frames, group %s)",
                 self._seq.name, self._seq.total_frames, split)
        return self._metadata()

    def ensure_live(self) -> dict[str, Any]:
        """Reset semantics: a live sequence is read untouched; a done one (or an
        unplaced manager) is re-armed at cursor -1. Never chooses a sequence —
        placement is env-panel-owned (falls back to the first sequence on disk
        only when nothing has been placed yet, so a bare canvas Play works)."""
        if self._seq is None:
            seqs = self._sequence_dirs()
            if not seqs:
                return {"error": f"no TUM sequences under {_data_root()}"}
            grp = freiburg_group(seqs[0])
            return self.set_episode(grp, 0)
        if self._done:
            self._cursor = -1
            self._done = False
        return self._metadata()

    def _metadata(self) -> dict[str, Any]:
        if self._seq is None:
            return {"error": "no sequence placed"}
        return {
            "sequence": self._seq.name,
            "num_frames": self._seq.num_frames(self._max_frames),
            "total_frames": self._seq.total_frames,
            "intrinsics": self._seq.intrinsics,
            "split": self._split,
        }

    # ── Transition + perception ──

    def step(self) -> dict[str, Any]:
        """Advance the cursor one frame. Control signals only (no observation).

        ``terminated`` fires once the cursor reaches the final frame, so the
        loop tracks every frame and stops cleanly at the end of the sequence.
        """
        if self._seq is None:
            return {"reward": 0.0, "terminated": True, "truncated": False,
                    "info": {"error": "no sequence"}, "frame_index": -1}
        n = self._seq.num_frames(self._max_frames)
        if self._cursor < n - 1:
            self._cursor += 1
        terminated = self._cursor >= n - 1
        self._done = terminated
        info = {"frame_index": self._cursor, "num_frames": n, "sequence": self._seq.name}
        return {"reward": 0.0, "terminated": bool(terminated), "truncated": False,
                "info": info, "frame_index": self._cursor}

    def observe(self) -> dict[str, Any]:
        """Idempotent read of the current frame (never advances the cursor)."""
        if self._seq is None:
            return {"rgb": None, "depth": None, "pose": None, "intrinsics": None}
        fr = self._seq.load_frame(self._cursor)
        return {
            "rgb": fr["rgb"],
            "depth": fr["depth"],
            "pose": fr["pose"],
            "intrinsics": self._seq.intrinsics,
            "timestamp": fr["timestamp"],
        }

    def evaluate(self) -> dict[str, Any]:
        if self._seq is None:
            return {"error": "no sequence"}
        n = self._seq.num_frames(self._max_frames)
        return {
            "sequence": self._seq.name,
            "num_frames": n,
            "gt_path_length_m": round(self._seq.gt_path_length(self._max_frames), 4),
        }

    def get_total_episodes(self) -> int:
        return len(self._sequence_dirs())


def _mgr() -> TumEnvManager:
    return TumEnvManager.get()


async def _run(fn: Any, *args: Any) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_mgr().executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Nodes — gym-like env interface (replay variant)
#
#   reset               → sequence / num_frames / intrinsics  (metadata only)
#   step_replay         → advance cursor; reward/terminated/truncated/info + frame_index
#   observe_egocentric  → rgb / depth / pose(GT) / intrinsics  (pull, idempotent)
#   evaluate            → metrics (frame count + GT path length)
# ══════════════════════════════════════════════════════════════════════


class ResetTumTool(BaseCanvasNode):
    node_type = "env_tum__reset"
    display_name = "TUM: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="teal",
        config_fields=[
            ConfigField(
                "max_frames", "text",
                label="Max frames (blank/0 = whole sequence; head-truncate for quick tests)",
                default="0",
            ),
        ],
    )
    description = (
        "Ensure the placed TUM sequence is live (re-arm if finished) and emit its "
        "metadata: sequence name, frame count, camera intrinsics. No observation — "
        "pull the first frame via observe_egocentric."
    )
    category = "environment"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger reset (any value)", optional=True),
    ]
    output_ports = [
        PortDef("sequence", "TEXT", "Sequence name"),
        PortDef("num_frames", "ANY", "Number of frames to replay"),
        PortDef("intrinsics", "ANY", "Camera intrinsics {fx,fy,cx,cy,width,height}"),
        PortDef("episode_ok", "BOOL", "True once a sequence is armed"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        cap = 0
        with contextlib.suppress(TypeError, ValueError):
            cap = int(self.config.get("max_frames") or 0)
        _mgr().set_max_frames(cap)
        meta = await _run(_mgr().ensure_live)
        if "error" in meta:
            self._self_log("error", meta["error"])
            return {"sequence": "", "num_frames": 0, "intrinsics": None, "episode_ok": False}
        self._self_log("sequence", meta["sequence"])
        self._self_log("num_frames", meta["num_frames"])
        return {
            "sequence": meta["sequence"],
            "num_frames": meta["num_frames"],
            "intrinsics": meta["intrinsics"],
            "episode_ok": True,
        }


class StepReplayTumTool(BaseCanvasNode):
    node_type = "env_tum__step_replay"
    display_name = "TUM: Step (replay)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "Advance the replay cursor by one frame — the degenerate 'action' of a "
        "pre-recorded stream. Returns control signals only (terminated fires when "
        "the sequence is exhausted); pull the frame itself via observe_egocentric. "
        "Wire terminated → iterOut.stop and info → observe.trigger."
    )
    category = "environment"
    icon = "Play"
    input_ports = [
        PortDef("trigger", "ANY", "Advance one frame (any value — content ignored)", optional=True),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Always 0 (replay has no reward)"),
        PortDef("terminated", "BOOL", "True on the final frame of the sequence"),
        PortDef("truncated", "BOOL", "Always False (step-budget cutoff is the graph's job)"),
        PortDef("info", "ANY", "Diagnostics {frame_index, num_frames, sequence}"),
        PortDef("frame_index", "ANY", "Current frame index (also the loop-carry token)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().step)
        self._self_log("frame_index", result.get("frame_index"))
        self._self_log("terminated", result.get("terminated"))
        return result


class ObserveEgocentricTumTool(BaseCanvasNode):
    node_type = "env_tum__observe_egocentric"
    display_name = "TUM: Observe (egocentric)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "Pull the current RGB-D frame + ground-truth camera pose + intrinsics "
        "(idempotent read; never advances the cursor). Same port shape as "
        "env_habitat__observe_egocentric — depth is metric metres, pose is the TUM "
        "mocap ground truth (None when no mocap sample aligns to this frame)."
    )
    category = "environment"
    icon = "Eye"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Current RGB frame (HxWx3 uint8)"),
        PortDef("depth", "DEPTH", "Current depth map (HxW float32, metres)"),
        PortDef("pose", "POSE", "Ground-truth camera pose {position, orientation} or None"),
        PortDef("intrinsics", "ANY", "Camera intrinsics {fx,fy,cx,cy,width,height}"),
        PortDef("timestamp", "ANY", "Frame timestamp (seconds)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        obs = await _run(_mgr().observe)
        self._self_log("has_rgb", obs.get("rgb") is not None)
        self._self_log("has_pose", obs.get("pose") is not None)
        return obs


class EvaluateTumTool(BaseCanvasNode):
    node_type = "env_tum__evaluate"
    display_name = "TUM: Evaluate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    description = (
        "Post-hoc replay summary (after-loop band): frame count + ground-truth "
        "path length. The real SLAM accuracy metric (ATE/RPE) comes from "
        "model_pyslam__eval_trajectory, not here."
    )
    category = "evaluation"
    icon = "BarChart"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger evaluation (any value)", optional=True),
    ]
    output_ports = [
        PortDef("metrics", "METRICS", "Replay summary dict"),
        PortDef("num_frames", "TEXT", "Frames replayed"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().evaluate)
        if "error" in result:
            return {"metrics": {}, "num_frames": "0"}
        self._self_log("num_frames", result["num_frames"])
        self._self_log("gt_path_length_m", result["gt_path_length_m"])
        return {"metrics": result, "num_frames": str(result["num_frames"])}


# ══════════════════════════════════════════════════════════════════════
# TumEnvPanel — sequence selection (split = freiburg group, episode = sequence)
# ══════════════════════════════════════════════════════════════════════


class TumEnvPanel(BaseEnvPanel):
    """Two-field cascade ``split → episode_index``.

    ``split`` is the freiburg camera group (it fixes the intrinsics);
    ``episode_index`` picks a sequence within that group.
    """

    name = "env_tum"
    display_name = "TUM RGB-D (replay)"
    fields = [
        EnvPanelField("split", "select", "Camera"),
        EnvPanelField("episode_index", "select", "Sequence"),
    ]
    actions = [
        EnvPanelAction("play", "Play", side_effect="run_start"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop", "Stop", side_effect="run_stop", enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {"split": "", "episode_index": 0}

    def _mgr(self) -> TumEnvManager:
        return TumEnvManager.get()

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mgr().executor, fn, *args)

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "split": self._state.get("split", ""),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    async def on_load(self) -> dict[str, Any]:
        mgr = self._mgr()
        splits = await self._run(mgr.list_splits)
        split = self._state.get("split") or (splits[0] if splits else "")
        self._state["split"] = split
        episodes = await self._run(mgr.list_episodes, split) if split else []
        if not mgr.initialized:
            return {
                "available": False, "split": split, "episode_index": 0,
                "episode_count": 0, "splits": splits, "step_budget": 5000,
                "message": (
                    f"No TUM sequences under {_data_root()}. Run "
                    "workspace/nodesets/env/env_tum/download.sh to fetch "
                    "fr3/long_office_household."
                ),
            }
        return {
            "available": True,
            "split": split,
            "episode_index": int(self._state.get("episode_index", 0)),
            "episode_count": len(episodes),
            "splits": splits,
            "step_budget": 5000,  # long sequences (fr3/long_office ~2.5k frames)
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        if name == "split":
            self._state["split"] = str(value)
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                self._state["episode_index"] = int(value)
            except (TypeError, ValueError):
                self._state["episode_index"] = 0
            await self._run(
                self._mgr().set_episode,
                self._state["split"],
                int(self._state["episode_index"]),
            )
        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = self._episode_reset_payload()
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name in ("play", "reset"):
            result = await self._run(
                self._mgr().set_episode,
                self._state["split"],
                int(self._state["episode_index"]),
            )
            if isinstance(result, dict) and "error" in result:
                return {"ok": False, "side_effect": "none", "error": result["error"]}
            if name == "play":
                return {"ok": True, "side_effect": "run_start"}
            return {
                "ok": True,
                "side_effect": "signal",
                "signal_name": "episode_reset",
                "signal_payload": self._episode_reset_payload(),
            }
        if name in ("pause", "stop"):
            return {"ok": True, "side_effect": f"run_{name}"}
        return {"ok": False, "side_effect": "none", "error": f"Unknown action '{name}'"}

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        mgr = self._mgr()
        if field == "split":
            splits = await self._run(mgr.list_splits)
            return [{"value": s, "label": s} for s in splits]
        if field == "episode_index":
            episodes = await self._run(mgr.list_episodes, self._state.get("split", ""))
            return [{"value": e["index"], "label": f'{e["index"]}: {e["sequence"]}'}
                    for e in episodes]
        return []


# ══════════════════════════════════════════════════════════════════════
# EnvTumNodeSet — the nodeset binding (local mode: pure numpy + PIL)
# ══════════════════════════════════════════════════════════════════════


class EnvTumNodeSet(BaseNodeSet):
    """TUM RGB-D replay environment as a NodeSet.

    Local mode: the reader is pure numpy + PIL (both in the agentcanvas env), so
    no ``server_python`` / conda env / container. Stateful cursor per worker →
    ``parallelism = "replicated"``.
    """

    name = "env_tum"
    description = "TUM RGB-D SLAM benchmark — streaming replay environment"
    # No dedicated interpreter: the reader needs only numpy + PIL, already in the
    # framework env. Keep it local (model_pyslam owns the GPL container boundary).
    server_python: ClassVar[str | None] = None
    env_panel = TumEnvPanel
    parallelism: ClassVar[str] = "replicated"  # per-worker sequence + cursor
    # Pure disk I/O + PNG decode per step — sub-second; roomy headroom.
    default_per_step_budget_sec: ClassVar[float] = 5.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = TumEnvManager.get()

    def get_tools(self) -> list:
        return [
            ResetTumTool(),
            StepReplayTumTool(),
            ObserveEgocentricTumTool(),
            EvaluateTumTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        await _run(lambda: self._mgr.initialize(**kwargs))

    async def shutdown(self) -> None:
        await _run(self._mgr.shutdown)

    async def get_eval_metadata(self) -> dict:
        splits = await _run(self._mgr.list_splits)
        return {
            "env_name": "tum_rgbd",
            "datasets": ["TUM-RGBD"],
            "splits": splits,
            "episode_counts": {},
            "metrics": ["ate_rmse", "rpe_rmse", "num_frames"],
            "supports_set_episode": True,
            "step_budget": 5000,
        }
