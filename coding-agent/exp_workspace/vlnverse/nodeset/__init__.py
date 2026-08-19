from __future__ import annotations

"""EnvVLNVerseNodeSet — VLNVerse (Isaac Sim 5.1) VLN environment as a NodeSet.

Works both as a local in-process nodeset and as an auto-hosted server:
  Local:  POST /api/components/nodesets/env_vlnverse/load
  Server: POST /api/components/nodesets/env_vlnverse/load?mode=server

Architecture — three layers (same shape as env_habitat, different seam)
----------------------------------------------------------------------
1. ``VLNVerseEnvManager`` (singleton engine)
     Pure-python environment logic: episode data (``data/vlnverse/
     raw_data/final_splits``), kinematic agent pose + occupancy collision
     (``_kinematics``), per-episode metric accumulation (``_metrics``).
     Owns a threading.Lock and a single-thread ThreadPoolExecutor that
     pins all env work to one OS thread. Rendering is delegated to an
     out-of-process Isaac 5.1 worker over the msgpack seam
     (``_sim_backend`` → ``_isaac_worker.py`` under
     ``~/isaacsim5.1/python.sh``); this process NEVER imports
     ``omni.*``/``isaacsim``. That is the key difference from
     env_habitat, where habitat-sim lives in-process and drags the whole
     nodeset into the ac-vlnce env — here the manager runs anywhere and
     only the renderer needs Isaac.

2. Canvas tool nodes (thin ``BaseCanvasNode`` adapters)
     reset · step_discrete · step_hightolow · observe_egocentric ·
     observe_panorama · evaluate — the env-template four-verb contract
     (docs: nodesets/env/template.html). Each dispatches to the manager
     via ``_run_sync(...)`` and emits ``_self_log`` entries for
     two-layer observability (ADR-022).

3. ``EnvVLNVerseNodeSet`` (collection + lifecycle)
     ``get_tools()`` / ``initialize()`` / ``shutdown()`` /
     ``get_eval_metadata()`` + the ``VLNVerseEnvPanel`` control plane
     (dataset → split → episode cascade, ADR-025). ``server_python``
     steers ComponentRegistry to auto-host this module in the
     ac-vlnverse conda env when loaded in server mode.

VLNVerse specifics worth knowing
--------------------------------
- Movement is kinematic teleportation with occupancy snapping (freemap
  grid per scene); Isaac only renders. A "step" is a macro action, so
  the default episode budget is 20 steps (NavHarness MLLM eval), not
  habitat's 500 micro-steps.
- Upstream unit quirk (kept, test-pinned in ``_kinematics``): MOVE's
  ``angle`` is radians while ``elevation`` is DEGREES. All shipped
  episodes are single-floor (z == 0 across 16,867 episodes, audited
  2026-07-20), so elevation defaults to 0.
- Panorama views are CCW: dir_id 0 = current heading, increasing
  counter-clockwise (dir 3 = Left at n_views=12) — matches the
  NavHarness strip mapping, NOT habitat's clockwise names.
- Camera: 1024x1024, focal 10 mm / aperture 20 mm → 90° FOV,
  f_pixels = width/2 (NavHarness middle_layer camera model).
- ``shutdown()`` must reap the Isaac worker explicitly: it is a
  grandchild of the backend and NOT covered by auto_host's
  PR_SET_PDEATHSIG one-hop chain.

Vendored seam files: ``_sim_backend`` (msgpack RPC clients),
``_isaac_worker`` (Isaac-side, standalone), ``_kinematics`` /
``_quat`` / ``_metrics`` (pure logic). Sources + deviations documented
per file.

last updated: 2026-07-20
"""

import asyncio
import base64
import concurrent.futures
import contextlib
import glob
import gzip
import io
import json
import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

# Import is intentional at module load (not lazy) — mirrors env_habitat.
# Failure to import means the agentcanvas backend itself is broken.
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)

from ._kinematics import VLNVerseKinematics
from ._metrics import EpisodeMetricsAccumulator

# NOTE: ``._sim_backend`` (and its msgpack/msgpack_numpy deps) is imported
# lazily inside ``VLNVerseEnvManager.initialize`` — the registry scan must be
# able to import THIS module in the plain backend env, where msgpack_numpy may
# not be installed. Same pattern as env_habitat's lazy ``import habitat``.

log = logging.getLogger("agentcanvas.vlnverse")


# ══════════════════════════════════════════════════════════════════════
# Dataset wiring — fine / coarse × final_splits files
#
# The VLNVerse episode store is ``data/vlnverse/raw_data/final_splits/
# {dataset}_{split}.json[.gz]``. The three-field VLNVerseEnvPanel cascade
# (dataset → split → episode) picks the file; there is no per-split YAML
# (unlike VLN-CE) — the sim config is identical across splits and only
# the episode list changes, so a split switch never reboots Isaac.
# ══════════════════════════════════════════════════════════════════════

_DATASETS: list[str] = ["fine", "coarse"]

# Split suffixes exposed in the panel/eval, in display order (eval-relevant
# first). The final_splits dir also holds dialogue_subset / goalnav_subset /
# long_horizon* / interactive_subset families (different task semantics) and
# ``*.withheld`` challenge twins (goals=None, unscorable) — everything not in
# this list is out of contract (verified by data audit).
_SPLIT_ORDER: list[str] = [
    "val_unseen",
    "val",
    "subset",
    "subset_mini",
    "mini",
    "human",
    "challenge",
    "test",
    "train",
]
_ALLOWED_SPLITS: frozenset[str] = frozenset(_SPLIT_ORDER)

# NavHarness MLLM eval step budget (macro actions — see module docstring).
_DEFAULT_MAX_STEPS = 20
# Upstream success radius (goals.radius is 3.0 across every audited split).
_SUCCESS_DISTANCE_M = 3.0
# Discrete action space (bare/full surface parity with the habitat harness):
# 0=STOP · 1=FORWARD 0.25 m · 2=LEFT +15° · 3=RIGHT −15°.
_FORWARD_STEP_M = 0.25
_TURN_ANGLE_RAD = float(np.radians(15.0))


def _env_max_steps() -> int:
    """Episode step budget when the caller passes none.

    ``$VLNVERSE_MAX_STEPS`` wins; else the 20 macro-action default. The env
    var is the ONLY lever a server-mode run has: ``AutoServerApp`` calls
    ``initialize()`` with no kwargs, so an auto-hosted nodeset can never be
    handed ``max_steps``. A discrete bare/full run (0=STOP · 1=FWD 0.25 m ·
    2/3=turn 15°) needs the habitat-style 500 MICRO-step budget — at 20 it
    truncates after ~5 m of travel, long before the agent can act on the
    instruction.
    """
    raw = os.environ.get("VLNVERSE_MAX_STEPS")
    if not raw:
        return _DEFAULT_MAX_STEPS
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("Ignoring non-integer VLNVERSE_MAX_STEPS=%r", raw)
        return _DEFAULT_MAX_STEPS


def _data_root() -> str:
    """Resolve the VLNVerse data root.

    ``$VLNVERSE_DATA_ROOT`` wins; else ``<repo_root>/data/vlnverse``
    (``__file__`` lives at workspace/nodesets/env/env_vlnverse/__init__.py —
    four parents to the repo root).
    """
    explicit = os.environ.get("VLNVERSE_DATA_ROOT")
    if explicit:
        return explicit
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
    )
    return os.path.join(repo_root, "data", "vlnverse")


def _splits_dir() -> str:
    return os.path.join(_data_root(), "raw_data", "final_splits")


# (dataset → (splits-dir mtime, splits)) — the split-file set only changes
# when data is (re)installed, and the panel/eval paths re-enumerate often.
_dataset_splits_cache: dict[str, tuple[float, list[str]]] = {}


def _dataset_splits(dataset: str) -> list[str]:
    """Enumerate selectable splits for a dataset from the files on disk."""
    try:
        mtime = os.stat(_splits_dir()).st_mtime
    except OSError:
        return []  # data symlink not in place — nothing selectable, no cache
    cached = _dataset_splits_cache.get(dataset)
    if cached is not None and cached[0] == mtime:
        return list(cached[1])
    found: set[str] = set()
    for path in glob.glob(os.path.join(_splits_dir(), f"{dataset}_*.json")) + glob.glob(
        os.path.join(_splits_dir(), f"{dataset}_*.json.gz")
    ):
        name = os.path.basename(path)
        name = name[: -len(".json.gz")] if name.endswith(".json.gz") else name[: -len(".json")]
        suffix = name[len(dataset) + 1 :]
        if suffix in _ALLOWED_SPLITS:
            found.add(suffix)
    ordered = [s for s in _SPLIT_ORDER if s in found]
    _dataset_splits_cache[dataset] = (mtime, ordered)
    return list(ordered)


def _split_file(dataset: str, split: str) -> str:
    """Path of the split's episode file — plain .json preferred over .json.gz
    (audit: where both exist they are identical)."""
    base = os.path.join(_splits_dir(), f"{dataset}_{split}.json")
    if os.path.isfile(base):
        return base
    if os.path.isfile(base + ".gz"):
        return base + ".gz"
    raise FileNotFoundError(
        f"No episode file for dataset={dataset!r} split={split!r} under {_splits_dir()} "
        f"(tried {base}[.gz]) — is the data/vlnverse symlink in place? "
        f"(scripts/install/install_ac_vlnverse.sh)"
    )


def _normalize_episode(ep: dict) -> dict:
    """Canonicalize the episode's goal schema at load time — the single seam
    every consumer relies on afterwards.

    ``goals`` becomes ``{"position": [...], "radius": float} | None``:
      - dict with a position → kept (radius defaults to _SUCCESS_DISTANCE_M);
      - None/absent (the ``.withheld`` challenge twins) → None — distance
        metrics degrade to inf and are wire-sanitized by ``_finite_or_none``;
      - anything else (habitat-style goal LISTS live in the excluded
        long_horizon* family) → loud ValueError, never silent unscorability.
    """
    goals = ep.get("goals")
    if goals is None:
        ep["goals"] = None
    elif isinstance(goals, dict) and "position" in goals:
        radius = goals.get("radius")
        ep["goals"] = {
            "position": list(goals["position"]),
            # `is not None`, not `or`: a legitimate radius of 0 must survive.
            "radius": float(radius) if radius is not None else _SUCCESS_DISTANCE_M,
        }
    else:
        raise ValueError(
            f"Unsupported goals schema in episode {ep.get('episode_id')!r}: "
            f"{type(goals).__name__}"
        )
    return ep


# (dataset, split) → normalized episode list. Safe to share: episodes are
# read-only after _normalize_episode (the manager, panel, and kinematics only
# read fields; replicated eval workers are separate processes anyway).
_split_episodes_cache: dict[tuple[str, str], list[dict]] = {}


def _load_split_episodes(dataset: str, split: str) -> list[dict]:
    if split not in _ALLOWED_SPLITS:
        # Closes the /call surface bypass: without this guard a caller could
        # load e.g. "challenge.withheld" (goals=None → all-None metrics) that
        # the panel/eval enumeration deliberately excludes.
        raise ValueError(
            f"Split {split!r} is not an allowed VLNVerse split "
            f"(allowed: {_SPLIT_ORDER}) — the dialogue/goalnav/long_horizon/"
            f"interactive families and .withheld twins are out of contract."
        )
    key = (dataset, split)
    cached = _split_episodes_cache.get(key)
    if cached is not None:
        return cached
    path = _split_file(dataset, split)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        payload = json.load(f)
    episodes = payload["episodes"] if isinstance(payload, dict) else payload
    if not isinstance(episodes, list):
        raise ValueError(f"Malformed split file {path}: expected an episode list")
    episodes = [_normalize_episode(ep) for ep in episodes]
    _split_episodes_cache[key] = episodes
    return episodes


def _instruction_text(ep: dict) -> str:
    """Normalize the instruction to a plain string.

    fine → ``instruction.instruction_text`` is a str; coarse → it is a dict
    whose ``"formal"`` key carries the text (audit 2026-07-20). Anything
    unexpected degrades to ``str(...)`` rather than crashing.
    """
    inst = ep.get("instruction") or {}
    text = inst.get("instruction_text") if isinstance(inst, dict) else inst
    if isinstance(text, dict):
        return str(text.get("formal", ""))
    return str(text or "")


def _extract_episode_extras(ep: dict, dataset: str) -> dict[str, Any]:
    """Stable extras dict (habitat RxR-extras pattern): keeps the raw
    instruction payload + ids so downstream graphs can be dataset-agnostic."""
    inst = ep.get("instruction")
    return {
        "dataset": dataset,
        "raw_instruction": inst.get("instruction_text") if isinstance(inst, dict) else inst,
        "trajectory_id": ep.get("trajectory_id"),
        "scene_id_raw": ep.get("scene_id"),
    }


def _finite_or_none(v: Any) -> Any:
    """JSON-safe scalar: non-finite floats (inf when goals are absent) → None."""
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


# ══════════════════════════════════════════════════════════════════════
# VLNVerseEnvManager — singleton env runtime (renderer out-of-process)
# ══════════════════════════════════════════════════════════════════════


class VLNVerseEnvManager:
    """Manages a single VLNVerse episode runtime.

    All public methods are blocking and should be called via
    ``asyncio.run_in_executor(mgr.executor, fn)`` from async code.
    Single-thread executor keeps env state + worker RPC serialized.

    Method names deliberately mirror ``HabitatEnvManager`` — the auto_host
    ``POST /call/{fn}`` surface calls managers by method name, so keeping
    the names identical lets the coding-agent bridges / Human tab retarget
    from habitat to vlnverse by URL alone.
    """

    _instance: VLNVerseEnvManager | None = None

    def __init__(self) -> None:
        self._backend: Any = None  # SimBackend (Subprocess/Socket client)
        self._kin = VLNVerseKinematics(
            collision_threshold=0.1, use_occupancy_collision=True
        )
        self._acc: EpisodeMetricsAccumulator | None = None
        self._episodes: list[dict] = []
        self._dataset = "fine"
        self._split = "val_unseen"
        self._episode_index = 0
        self._episode_done = False
        self._terminated = False
        self._truncated = False
        self._step_count = 0
        self._max_steps = _DEFAULT_MAX_STEPS
        self._gpu_id = 0
        self._sim_cfg: dict[str, Any] | None = None
        self._current_scene: str | None = None
        # Per-episode success radius (goals.radius, canonicalized at load;
        # module constant is the goals-absent fallback).
        self._success_distance = _SUCCESS_DISTANCE_M
        # Frame cache (RenderedView). Explicit invalidation is the single
        # mechanism: every pose-changing action and every episode seat clears
        # it, so an observe after a move re-renders once, repeated observes
        # are free, and a post-STOP observe serves the cached terminal frame.
        self._current_view: Any = None
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vlnverse",
        )

    # ── Singleton access ──

    @classmethod
    def get(cls) -> VLNVerseEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Lifecycle ──

    def _build_sim_cfg(self) -> dict[str, Any]:
        """Serializable worker boot config — NavHarness ``_build_sim_cfg``
        defaults (RaytracedLighting @ 1024², focal 10 mm)."""
        return {
            "headless": os.environ.get("VLNVERSE_HEADLESS", "1") != "0",
            "gpu_id": int(self._gpu_id),
            "isaac_multi_gpu": False,
            "renderer": os.environ.get("VLNVERSE_RENDERER", "RaytracedLighting"),
            "anti_aliasing": 0,
            "denoiser": False,
            "camera_resolution": (1024, 1024),
            "focal_length": 10.0,
            # Single source of truth: the worker's own default is 2*focal —
            # this explicit value and that default must stay equal (90° FOV).
            "aperture": 20.0,
        }

    def _make_backend(self) -> Any:
        """Socket backend when ``$VLNVERSE_SIM_SOCKET`` names a pre-started
        worker (debug / warm-boot reuse); else spawn the worker under the
        Isaac 5.1 launcher (``$VLNVERSE_ISAAC_PYTHON``)."""
        from ._sim_backend import SocketIsaacSimBackend, SubprocessIsaacSimBackend

        socket_path = os.environ.get("VLNVERSE_SIM_SOCKET")
        if socket_path:
            log.info("Attaching to pre-started Isaac worker at %s", socket_path)
            return SocketIsaacSimBackend(socket_path)
        isaac_python = os.path.expanduser(
            os.environ.get("VLNVERSE_ISAAC_PYTHON", "~/isaacsim5.1/python.sh")
        )
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_isaac_worker.py")
        log.info("Spawning Isaac worker: %s %s", isaac_python, worker)
        return SubprocessIsaacSimBackend(isaac_python, worker)

    def initialize(
        self,
        dataset: str = "fine",
        split: str = "val_unseen",
        gpu_id: int = 0,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> dict:
        """Boot the Isaac worker, load the split, seat episode 0."""
        with self._lock:
            if self._backend is not None:
                log.warning("VLNVerseEnvManager already initialized — skipping")
                return self._get_episode_info_unlocked()

            if dataset not in _DATASETS:
                raise ValueError(f"Unknown dataset {dataset!r} (expected one of {_DATASETS})")

            self._dataset = dataset
            self._split = split
            self._gpu_id = int(gpu_id)
            self._max_steps = int(max_steps)
            self._sim_cfg = self._build_sim_cfg()

            self._episodes = _load_split_episodes(dataset, split)
            log.info(
                "Loaded %d episodes (dataset=%s split=%s)", len(self._episodes), dataset, split
            )

            backend = self._make_backend()
            log.info("Starting Isaac worker (cold boot is minute-scale) ...")
            backend.start(self._sim_cfg)
            self._backend = backend

            self._seat_episode_unlocked(0)
            info = self._get_episode_info_unlocked()
            log.info(
                "VLNVerse env ready — episode=%s scene=%s",
                info.get("episode_id"),
                info.get("scene_id"),
            )
            return info

    def switch_split(self, dataset: str, split: str) -> dict:
        """Swap the episode list (no Isaac reboot — sim config is
        split-independent) and seat episode 0 of the new split."""
        with self._lock:
            if self._backend is None:
                return {"error": "Environment not initialized"}
            if dataset not in _DATASETS:
                return {"error": f"Unknown dataset {dataset!r}"}
            try:
                self._episodes = _load_split_episodes(dataset, split)
            except ValueError as e:
                # Banned/unknown split: same {"error"} contract as the other
                # failure classes above — a /call client gets a clean payload,
                # not a 500.
                return {"error": str(e)}
            self._dataset = dataset
            self._split = split
            log.info(
                "Switched to dataset=%s split=%s (%d episodes)",
                dataset,
                split,
                len(self._episodes),
            )
            # Lazy seat: episode STATE only — the scene (20-30 s USD load
            # when it differs) is deferred to the first action that needs
            # the worker, so a panel cascade (dataset → split → episode →
            # play) doesn't pay for scenes the user never runs.
            self._seat_episode_unlocked(0, load_scene=False)
            return self._get_episode_info_unlocked()

    def shutdown(self) -> None:
        """Tear down the Isaac worker. Explicit reap: the worker is a
        grandchild of the backend process — auto_host's PR_SET_PDEATHSIG only
        covers the one-hop child, so relying on the signal chain would leak a
        multi-GB SimulationApp on nodeset unload."""
        with self._lock:
            if self._backend is not None:
                log.info("Shutting down Isaac worker")
                try:
                    self._backend.shutdown()
                finally:
                    self._backend = None
                self._current_view = None
                # A fresh worker boots with an EMPTY stage — forgetting this
                # would make _ensure_scene_unlocked treat the old scene as
                # still loaded after a re-initialize and render grey frames.
                self._current_scene = None
                # Post-shutdown evaluate()/queries error cleanly instead of
                # reporting the dead episode's metrics.
                self._acc = None

    @property
    def initialized(self) -> bool:
        return self._backend is not None

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ── Scene + episode seating (unlocked internals) ──

    def _scene_usd_path(self, scan: str) -> str:
        """Find the scene USD (NavHarness ``_resolve_scene_path`` semantics,
        rooted at data/vlnverse/scene)."""
        scene_dir = os.path.join(_data_root(), "scene", scan)
        candidates = [
            os.path.join(scene_dir, "start_result_navigation.usd"),
            os.path.join(scene_dir, "start_result_navigation.usda"),
            os.path.join(scene_dir, f"{scan}.usd"),
            os.path.join(scene_dir, f"{scan}.usda"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Scene file not found for {scan}. Tried: {candidates}")

    def _ensure_scene_unlocked(self, scan: str) -> None:
        """Load ``scan`` into the worker + its freemap into kinematics —
        no-op when already current (same-scene episode hops are instant;
        a cross-scene switch costs a USD load + stabilize in the worker)."""
        if scan == self._current_scene:
            return
        usd = self._scene_usd_path(scan)
        log.info("Loading scene %s (%s)", scan, usd)
        self._backend.load_scene(scan, usd)
        freemap = os.path.join(_data_root(), "scene", scan, "freemap.npy")
        self._kin.load_freemap(freemap)
        self._current_scene = scan

    def _seat_episode_unlocked(self, index: int, load_scene: bool = True) -> None:
        """Place the agent at episode ``index``'s start: scene, pose, goal,
        fresh metrics accumulator. The only place episode state is (re)armed.

        ``load_scene=False`` (split switching) defers the scene load to the
        first action that needs the worker (``_ensure_episode_scene_unlocked``)
        — a panel cascade otherwise pays up to two 20-30 s USD loads for
        scenes the user never runs. Batch-eval placement
        (``set_episode_by_index``) stays eager.
        """
        ep = self._episodes[index]
        if load_scene:
            self._ensure_scene_unlocked(ep["scan"])
        self._kin.reset_from_episode(ep)
        goals = ep.get("goals")
        self._success_distance = (
            goals["radius"] if goals else _SUCCESS_DISTANCE_M
        )
        self._acc = EpisodeMetricsAccumulator(
            initial_distance=self._kin.current_dist_to_goal()
        )
        self._episode_index = index
        self._episode_done = False
        self._terminated = False
        self._truncated = False
        self._step_count = 0
        self._invalidate_view_unlocked()

    def _ensure_episode_scene_unlocked(self) -> None:
        """Deferred half of a lazy (split-switch) seat: make sure the CURRENT
        episode's scene + freemap are live before rendering or moving.
        No-op when the scene is already current."""
        if self._episodes:
            self._ensure_scene_unlocked(self._episodes[self._episode_index]["scan"])

    # ── Frame cache ──

    def _invalidate_view_unlocked(self) -> None:
        self._current_view = None

    def _render_current_unlocked(self):
        """Front view at the current pose, cached until explicitly invalidated
        (every pose-changing action and episode seat clears it) — observe_*
        stays a pure idempotent read (env-template §5.3) while a post-move
        observe re-renders exactly once and a post-STOP observe serves the
        cached terminal frame."""
        if self._current_view is not None:
            return self._current_view
        self._ensure_episode_scene_unlocked()
        views = self._backend.capture_panoramic(
            self._kin.agent_position, self._kin.agent_heading, num_views=1
        )
        self._current_view = views[0]
        return self._current_view

    # ── Actions ──

    def _finish_step_unlocked(self, pose_changed: bool = True) -> None:
        """Post-action bookkeeping shared by both step families: count the
        step, apply the budget cutoff (truncated), and drop the frame cache
        when the pose moved (STOP keeps the terminal frame cached)."""
        self._step_count += 1
        if not self._episode_done and self._step_count >= self._max_steps:
            self._episode_done = True
            self._truncated = True
            if self._acc is not None:
                self._acc.set_end_reason("step_budget_exhausted")
        if pose_changed:
            self._invalidate_view_unlocked()

    def _step_result_unlocked(self, extra: dict | None = None) -> dict:
        state = self._get_agent_state_unlocked()
        result: dict[str, Any] = {
            "position": state["position"],
            "orientation": state["orientation"],
            "done": self._episode_done,
            "step_count": self._step_count,
            "reward": 0.0,  # VLN sparse reward — 0 each step (template §5.2 D)
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
        if extra:
            result.update(extra)
        if self._episode_done:
            result["metrics"] = self._evaluate_unlocked()
        return result

    def _move_unlocked(
        self,
        angle: float,
        elevation: float,
        distance: float,
        extra: dict,
        with_deviation: bool = True,
    ) -> dict:
        """Shared per-step body for both action families: guards, pre-action
        capture, kinematic move, accumulator record, budget bookkeeping.
        ``extra`` keys ride into the result verbatim; ``with_deviation``
        keeps the discrete family's result shape deviation-free."""
        if self._backend is None:
            return {"error": "Environment not initialized"}
        if self._episode_done:
            return {"error": "Episode already done", "done": True}
        self._ensure_episode_scene_unlocked()  # lazy split-switch seat
        pre_pos = self._kin.agent_position.copy()
        pre_dist = self._kin.current_dist_to_goal()
        collision, deviation = self._kin.move(angle, elevation, distance)
        self._acc.record_step(
            pre_pos, pre_dist, collided=collision, moved_distance=distance
        )
        self._finish_step_unlocked(pose_changed=True)
        extra = {**extra, "collision": collision}
        if with_deviation:
            extra["deviation_m"] = float(deviation)
        return self._step_result_unlocked(extra)

    def _stop_unlocked(self, extra: dict) -> dict:
        """Terminal STOP: record the no-move step, mark the episode done."""
        if self._backend is None:
            return {"error": "Environment not initialized"}
        if self._episode_done:
            return {"error": "Episode already done", "done": True}
        self._acc.record_step(
            self._kin.agent_position.copy(),
            self._kin.current_dist_to_goal(),
            collided=False,
            moved_distance=0.0,
        )
        self._acc.mark_stop()
        self._acc.set_end_reason("stop_called")
        self._episode_done = True
        self._terminated = True
        self._finish_step_unlocked(pose_changed=False)
        return self._step_result_unlocked(extra)

    def step(self, action: int) -> dict:
        """Discrete step: 0=STOP · 1=FWD 0.25 m · 2=LEFT +15° · 3=RIGHT −15°.

        Same mapping as the NavHarness bare/full harness surface, so methods
        written against habitat's step_discrete transfer unchanged.
        """
        with self._lock:
            action = int(action)
            if action == 0:
                return self._stop_unlocked({"action": 0, "collision": False})
            if action not in (1, 2, 3):
                return {"error": f"Unknown discrete action {action} (expected 0-3)"}
            angle = {1: 0.0, 2: _TURN_ANGLE_RAD, 3: -_TURN_ANGLE_RAD}[action]
            distance = _FORWARD_STEP_M if action == 1 else 0.0
            return self._move_unlocked(
                angle, 0.0, distance, {"action": action}, with_deviation=False
            )

    def step_hightolow(self, angle: float, distance: float, elevation: float = 0.0) -> dict:
        """MOVE macro: rotate ``angle`` rad, advance ``distance`` m at
        ``elevation`` DEG (upstream rad/deg asymmetry — see module docstring).

        Boundary kept from habitat: ``angle == 0 and distance == 0`` is the
        canonical "no movement" STOP signal — dispatched to ``step(0)`` so
        the metrics register a called stop instead of a budget timeout.
        """
        if float(angle) == 0.0 and float(distance) == 0.0:
            return self.step(0)

        with self._lock:
            return self._move_unlocked(
                float(angle),
                float(elevation),
                float(distance),
                {
                    "angle": float(angle),
                    "distance": float(distance),
                    "elevation": float(elevation),
                },
            )

    # ── Metrics ──

    def _evaluate_unlocked(self) -> dict:
        if self._acc is None:
            return {"error": "No episode seated"}
        ep = self._episodes[self._episode_index]
        metrics = self._acc.final_metrics(
            episode_id=ep.get("episode_id"),
            current_dist_to_goal=self._kin.current_dist_to_goal(),
            reference_path=self._kin.reference_path,
            success_distance=self._success_distance,  # per-episode goals.radius
        )
        # goals=None episodes yield inf distances — JSON-sanitize (→ None)
        # rather than crash the wire (data audit: .withheld splits).
        return {k: _finite_or_none(v) for k, v in metrics.items()}

    def evaluate(self) -> dict:
        """Current episode metrics on demand — callable mid-episode (e.g.
        after a graph-side budget cap), same contract as habitat's."""
        with self._lock:
            if self._acc is None:
                return {"error": "Environment not initialized"}
            return self._evaluate_unlocked()

    # ── Rendering / observation ──

    def current_frame(self) -> dict:
        """Numpy RGB-D of the current pose (cached, pure read for
        observe_egocentric). Returns {rgb (H,W,3) u8, depth (H,W) f32}."""
        with self._lock:
            if self._backend is None:
                return {"error": "Environment not initialized"}
            view = self._render_current_unlocked()
            return {"rgb": np.asarray(view.rgb), "depth": np.asarray(view.depth)}

    def get_observations(self) -> dict:
        """Base64 form of the current frame (auto_host /call parity with
        habitat's get_observations)."""
        with self._lock:
            if self._backend is None:
                return {"error": "Environment not initialized"}
            view = self._render_current_unlocked()
            rgb = np.asarray(view.rgb)
            return {
                "rgb_base64": self.encode_rgb_base64(rgb),
                "depth_base64": self.encode_depth_base64(np.asarray(view.depth)),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            }

    def get_observation(self) -> dict:
        return self.get_observations()

    def render_panorama_rgbd(self, n_views: int = 12) -> dict:
        """Aligned RGB+Depth panorama for waypoint prediction / wp bridges.

        dir_id 0 = current heading, increasing CCW in (360/n)° steps
        (VLNVerse convention — the worker composes +yaw per view).
        """
        # The node's slider floors at 4, but the /call surface can pass
        # 0/negative — an empty capture would IndexError on the cache seed.
        n_views = max(1, int(n_views))
        with self._lock:
            if self._backend is None:
                return {"error": "Environment not initialized"}
            self._ensure_episode_scene_unlocked()
            rendered = self._backend.capture_panoramic(
                self._kin.agent_position, self._kin.agent_heading, num_views=n_views
            )
            # Seed the front-view cache: dir 0 IS the current-heading view and
            # rendering doesn't move the pose, so a following
            # observe_egocentric is free.
            self._current_view = rendered[0]
            views = []
            for i, view in enumerate(rendered):
                rgb = np.asarray(view.rgb)
                depth = np.asarray(view.depth)
                views.append(
                    {
                        "dir_id": i,
                        "heading_deg": round((i * 360.0 / len(rendered))) % 360,
                        "rgb_base64": self.encode_rgb_base64(rgb),
                        "depth_base64": self.encode_depth_base64(depth),
                        "depth_raw_base64": self.encode_depth_raw_base64(depth),
                    }
                )
            return {"views": views, "n_views": len(views)}

    def render_panorama(self, n_views: int = 4) -> dict:
        """Labeled composite panorama grid (single image). CCW direction
        names — 90° is Left here, unlike habitat's clockwise table."""
        n_views = max(1, int(n_views))  # /call can pass 0 — see render_panorama_rgbd
        with self._lock:
            if self._backend is None:
                return {"error": "Environment not initialized"}

            direction_names = {
                4: ["Front", "Left", "Back", "Right"],
                8: [
                    "Front",
                    "Front-Left",
                    "Left",
                    "Back-Left",
                    "Back",
                    "Back-Right",
                    "Right",
                    "Front-Right",
                ],
                12: [
                    "Front(0°)",
                    "Front-Left(30°)",
                    "Front-Left(60°)",
                    "Left(90°)",
                    "Back-Left(120°)",
                    "Back-Left(150°)",
                    "Back(180°)",
                    "Back-Right(210°)",
                    "Back-Right(240°)",
                    "Right(270°)",
                    "Front-Right(300°)",
                    "Front-Right(330°)",
                ],
                24: [f"{i * 15}°" for i in range(24)],
            }.get(n_views, [f"View_{i}" for i in range(n_views)])

            self._ensure_episode_scene_unlocked()
            rendered = self._backend.capture_panoramic(
                self._kin.agent_position, self._kin.agent_heading, num_views=n_views
            )
            # Seed the front-view cache (dir 0 = current heading, pose
            # unchanged) — same rationale as render_panorama_rgbd.
            self._current_view = rendered[0]
            # Composite mode's product is the single stitched image — per-view
            # payloads carry only the labels (no 12× PNG encodes).
            views = [
                {
                    "direction": direction_names[i],
                    "heading_deg": round((i * 360.0 / len(rendered))) % 360,
                }
                for i in range(len(rendered))
            ]
            composite = self._build_composite(
                [np.asarray(v.rgb, dtype=np.uint8) for v in rendered],
                [v["direction"] for v in views],
            )
            return {"views": views, "composite": composite, "n_views": len(views)}

    # ── Queries ──

    def get_agent_state(self) -> dict:
        with self._lock:
            return self._get_agent_state_unlocked()

    def get_state(self) -> dict:
        return self.get_agent_state()

    def _get_agent_state_unlocked(self) -> dict:
        if self._acc is None:
            return {"error": "Environment not initialized"}
        pos = [float(v) for v in self._kin.agent_position]
        w, x, y, z = (float(v) for v in self._kin.agent_rotation_quat)
        # Wire ordering [x, y, z, w] — habitat parity (kinematics stores wxyz).
        return {
            "position": pos,
            "orientation": [x, y, z, w],
            "heading": float(self._kin.agent_heading),
        }

    def get_cam_intrinsics(self) -> dict | None:
        """Pinhole intrinsics from the worker camera model: focal 10 mm on a
        20 mm square aperture → 90° FOV, f_pixels = width/2 (NavHarness
        middle_layer derivation)."""
        cfg = self._sim_cfg
        if cfg is None:
            return None
        h, w = (int(v) for v in cfg.get("camera_resolution", (1024, 1024)))
        focal = float(cfg.get("focal_length", 10.0))
        # fx = w/2 is verified correct: the worker
        # explicitly calls set_horizontal/vertical_aperture(cfg["aperture"]
        # or 2*focal = 20 mm) — Isaac's 20.955 mm default never applies —
        # giving FOV = 2·atan(ap/(2f)) = 90° and f_px = (W/2)/tan(45°) = W/2,
        # the exact model NavHarness middle_layer.py states. Isaac version
        # unit quirks (mm vs cm in set_focal_length) cancel because focal and
        # aperture go through the same wrapper and FOV depends on their ratio.
        aperture = float(cfg.get("aperture", 2 * focal))
        fx = w * focal / aperture
        fy = h * focal / aperture
        return {"fx": fx, "fy": fy, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}

    def get_instruction_text(self) -> str | None:
        """Current episode's natural-language instruction (per-episode
        metadata, NOT a per-step sensor) — normalized to a plain string."""
        with self._lock:
            if not self._episodes:
                return None
            return _instruction_text(self._episodes[self._episode_index])

    def get_episode_info(self) -> dict:
        with self._lock:
            return self._get_episode_info_unlocked()

    def _get_episode_info_unlocked(self) -> dict:
        if not self._episodes:
            return {"error": "Environment not initialized"}
        ep = self._episodes[self._episode_index]
        info: dict[str, Any] = {
            "episode_id": str(ep.get("episode_id", "")),
            "scene_id": str(ep.get("scan", "")),
            "step_count": self._step_count,
            "done": self._episode_done,
            "instruction": _instruction_text(ep),
            "language": None,  # cross-nodeset schema parity (habitat/MP3D)
            "extras": _extract_episode_extras(ep, self._dataset),
        }
        # goals is canonical dict-or-None after _normalize_episode; the wire
        # shape stays a list-of-one (habitat parity).
        if ep.get("goals"):
            info["goals"] = [dict(ep["goals"])]
        return info

    def reset_episode(self) -> dict:
        """Re-arm the CURRENT episode in place (idempotent ensure-live —
        template decision I; episode selection stays panel-owned)."""
        with self._lock:
            if self._backend is None:
                return {"error": "Environment not initialized"}
            self._seat_episode_unlocked(self._episode_index)
            info = self._get_episode_info_unlocked()
            info.update(self._get_agent_state_unlocked())
            return info

    def reset(self) -> dict:
        return self.reset_episode()

    def set_episode_by_index(self, index: int) -> dict:
        with self._lock:
            if self._backend is None:
                return {"error": "Environment not initialized"}
            index = int(index)
            if index < 0 or index >= len(self._episodes):
                return {"error": f"Index {index} out of range (0-{len(self._episodes) - 1})"}
            self._seat_episode_unlocked(index)
            info = self._get_episode_info_unlocked()
            info.update(self._get_agent_state_unlocked())
            return info

    def set_episode(self, index: int) -> dict:
        return self.set_episode_by_index(index)

    def get_total_episodes(self) -> int:
        with self._lock:
            return len(self._episodes)

    def get_current_episode(self) -> dict | None:
        with self._lock:
            if not self._episodes:
                return None
            return self._episodes[self._episode_index]

    def get_episodes_list(self, offset: int = 0, limit: int = 50, light: bool = False) -> dict:
        """Episode listing. ``light=True`` returns ids only — the panel's
        episode dropdown labels need index + scene, not 10k instruction
        payloads built under the env lock."""
        with self._lock:
            if not self._episodes:
                return {"episodes": [], "total": 0}
            eps = self._episodes[int(offset) : int(offset) + int(limit)]
            result = []
            for i, ep in enumerate(eps):
                entry = {
                    "index": int(offset) + i,
                    "episode_id": str(ep.get("episode_id", "")),
                    "scene_id": str(ep.get("scan", "")),
                }
                if not light:
                    entry.update(
                        {
                            "instruction": _instruction_text(ep),
                            "language": None,
                            "extras": _extract_episode_extras(ep, self._dataset),
                        }
                    )
                result.append(entry)
            return {"episodes": result, "total": len(self._episodes)}

    def get_episodes(self, offset: int = 0, limit: int = 50) -> dict:
        return self.get_episodes_list(offset, limit)

    def get_status(self) -> dict:
        with self._lock:
            if self._backend is None:
                return {"initialized": False}
            ep = self._episodes[self._episode_index] if self._episodes else {}
            return {
                "initialized": True,
                "dataset": self._dataset,
                "split": self._split,
                "step_count": self._step_count,
                "done": self._episode_done,
                "episode_id": str(ep.get("episode_id", "")),
                "scene_id": str(ep.get("scan", "")),
                "total_episodes": len(self._episodes),
                "max_steps": self._max_steps,
            }

    # ── Encoding helpers (verbatim from env_habitat) ──

    @staticmethod
    def encode_rgb_base64(rgb: np.ndarray) -> str:
        from PIL import Image

        # asarray, not astype: copy-free when the input is already uint8.
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # Depth arrives FINITE from the capture boundary — _sim_backend clamps
    # Isaac's non-finite no-hit pixels to a fixed far clip there, so every
    # consumer (DEPTH port, raw_obs, both encoders) sees the same values and
    # the min/max normalization below is safe.

    @staticmethod
    def encode_depth_base64(depth: np.ndarray) -> str:
        from PIL import Image

        d = np.squeeze(depth)
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 1e-6:
            d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d, dtype=np.uint8)
        img = Image.fromarray(d_norm, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def encode_depth_raw_base64(depth: np.ndarray) -> str:
        # 16-bit PNG depth in millimetres — preserves absolute metric depth
        # for consumers that unproject (waypoint predictors, SpatialBot-style
        # packers). Separate from encode_depth_base64's 8-bit viewer form.
        from PIL import Image

        d = np.squeeze(depth).astype(np.float32)
        d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
        img = Image.fromarray(d_mm, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _build_composite(images: list[np.ndarray], labels: list[str]) -> np.ndarray | None:
        """Stitched label-annotated grid as a numpy RGB array. Deviation from
        habitat (which returns base64): the only local consumer is the node's
        IMAGE port — encoding to PNG/b64 and decoding straight back in the
        same process wasted ~1 s per composite call."""
        from PIL import Image, ImageDraw

        n = len(images)
        if n == 0:
            return None

        h, w = images[0].shape[:2]
        label_h = 20
        border = 2  # thin black border around each sub-image

        if n <= 4:
            cols, rows = 2, 2
        elif n <= 8:
            cols, rows = 4, 2
        elif n <= 12:
            cols, rows = 4, 3
        else:
            cols, rows = 6, 4

        cell_w = w + border * 2
        cell_h = h + label_h + border * 2
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for i, (img_arr, label) in enumerate(zip(images, labels)):
            row, col = divmod(i, cols)
            x = col * cell_w + border
            y = row * cell_h + border
            draw.text((x + w // 2 - len(label) * 3, y + 2), label, fill=(255, 255, 100))
            canvas.paste(Image.fromarray(img_arr), (x, y + label_h))

        return np.asarray(canvas, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# Nodes — gym-like env interface (see docs: nodesets/env/template.html)
#
#   reset                episode metadata only (no observation)
#   step_discrete        action:ACTION  → reward / terminated / truncated / info
#   step_hightolow       angle,distance[,elevation] → same four ports
#   observe_egocentric   → rgb, depth, pose, intrinsics, raw_obs, instruction
#   observe_panorama     → views, directions, n_views, composite
#   evaluate             → metrics, success, spl   (thin metric sink)
#
# Episode lifecycle (episode selection) is env panel-owned
# (VLNVerseEnvPanel.set_episode); ``reset`` only revives. step nodes return
# NO observation — perception is pulled via the observe_* family.
# ══════════════════════════════════════════════════════════════════════


def _get_env() -> VLNVerseEnvManager:
    return VLNVerseEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    env_mgr = _get_env()
    return await asyncio.get_running_loop().run_in_executor(env_mgr.executor, fn, *args)


def _vlnverse_step_info(result: dict) -> dict:
    """Assemble the gym ``info`` dict from a manager step result."""
    info: dict[str, Any] = {
        "step_count": result.get("step_count"),
        "position": result.get("position"),
        "orientation": result.get("orientation"),
        "collision": result.get("collision"),
    }
    if "deviation_m" in result:
        info["deviation_m"] = result["deviation_m"]
    if result.get("metrics"):
        info["metrics"] = result["metrics"]
    return info


class ResetVLNVerseTool(BaseCanvasNode):
    node_type = "env_vlnverse__reset"
    display_name = "VLNVerse: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Ensure a live episode (re-arm if done) — emit instruction + ids, no observation"
    category = "environment"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger reset (any value)", optional=True),
    ]
    output_ports = [
        PortDef("instruction", "TEXT", "Navigation instruction"),
        PortDef("episode_id", "TEXT", "Episode ID"),
        PortDef("scene_id", "TEXT", "Scene ID (kujiale scan)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Episode placement is env panel-owned (set_episode); reset ensures
        # the placed episode is live. A done episode (canvas re-run) is
        # re-armed in place; a live one — the batch-eval path — is read
        # without disturbance. Rollover lives HERE, never in observe_*.
        mgr = _get_env()
        if mgr._episode_done:
            await _run_sync(mgr.reset_episode)
        info = await _run_sync(mgr.get_episode_info)
        self._self_log("episode_id", info.get("episode_id"))
        self._self_log("instruction", (info.get("instruction") or "")[:200])
        self._self_log("scene_id", info.get("scene_id"))
        return {
            "instruction": info.get("instruction", ""),
            "episode_id": str(info.get("episode_id", "")),
            "scene_id": str(info.get("scene_id", "")),
        }


class StepDiscreteVLNVerseTool(BaseCanvasNode):
    node_type = "env_vlnverse__step_discrete"
    display_name = "VLNVerse: Step (discrete)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Advance one tick with a discrete action (0=STOP, 1=FWD 0.25m, 2=LEFT 15°, 3=RIGHT 15°)"
    category = "environment"
    icon = "Play"
    input_ports = [
        PortDef("action", "ACTION", "Discrete action (0-3)"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called"),
        PortDef("truncated", "BOOL", "Step-budget cutoff"),
        PortDef("info", "ANY", "Per-step diagnostics + terminal metrics"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        action = int(inputs.get("action", 1))
        result = await _run_sync(_get_env().step, action)
        terminated = bool(result.get("terminated", result.get("done", False)))
        truncated = bool(result.get("truncated", False))
        info = _vlnverse_step_info(result)

        from app.standard.actions import ACTION_NAMES

        self._self_log("action", action)
        self._self_log("action_name", ACTION_NAMES.get(action, "UNKNOWN"))
        self._self_log("terminated", terminated)
        self._self_log("truncated", truncated)
        self._self_log("collision", result.get("collision"))
        self._self_log("step_count", result.get("step_count"))
        if (terminated or truncated) and result.get("metrics"):
            self._self_log("metrics", result["metrics"])
        return {
            "reward": result.get("reward", 0.0),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }


class StepHighToLowVLNVerseTool(BaseCanvasNode):
    """MOVE macro action — rotate by angle then advance distance.

    The native VLNVerse action (NavHarness MOVE / action 4). Unit quirk
    kept from upstream (test-pinned in ``_kinematics``): ``angle`` is
    RADIANS but ``elevation`` is DEGREES. All shipped episodes are
    single-floor, so elevation defaults to 0 and is an optional port.
    """

    node_type = "env_vlnverse__step_hightolow"
    display_name = "VLNVerse: Step (HIGHTOLOW)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Rotate by angle (rad) then advance distance (m); optional elevation (DEGREES)"
    category = "environment"
    icon = "Compass"
    input_ports = [
        PortDef("angle", "TEXT", "Rotation in radians (yaw, +CCW) — scalar float"),
        PortDef("distance", "TEXT", "Forward distance in metres — scalar float"),
        PortDef(
            "elevation",
            "TEXT",
            "Elevation angle in DEGREES (upstream unit quirk; default 0 — all "
            "shipped episodes are single-floor)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called (angle=0, distance=0)"),
        PortDef("truncated", "BOOL", "Step-budget cutoff"),
        PortDef("info", "ANY", "Diagnostics + terminal metrics (incl. collision)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        angle = float(inputs.get("angle") or 0.0)
        distance = float(inputs.get("distance") or 0.0)
        elevation = float(inputs.get("elevation") or 0.0)
        result = await _run_sync(_get_env().step_hightolow, angle, distance, elevation)
        terminated = bool(result.get("terminated", result.get("done", False)))
        truncated = bool(result.get("truncated", False))
        info = _vlnverse_step_info(result)
        self._self_log("angle_rad", angle)
        self._self_log("distance_m", distance)
        if elevation:
            self._self_log("elevation_deg", elevation)
        self._self_log("collision", result.get("collision"))
        self._self_log("terminated", terminated)
        self._self_log("truncated", truncated)
        if (terminated or truncated) and result.get("metrics"):
            self._self_log("metrics", result["metrics"])
        return {
            "reward": result.get("reward", 0.0),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }


class ObserveEgocentricVLNVerseTool(BaseCanvasNode):
    node_type = "env_vlnverse__observe_egocentric"
    display_name = "VLNVerse: Observe (egocentric)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Pull the current first-person observation: RGB, depth, pose, intrinsics"
    category = "environment"
    icon = "Eye"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Current RGB observation (1024², 90° FOV)"),
        PortDef("depth", "DEPTH", "Current depth map (metres, float32)"),
        PortDef("pose", "POSE", "Agent position + orientation"),
        PortDef("intrinsics", "ANY", "Camera intrinsics {fx,fy,cx,cy,width,height} or None"),
        PortDef("raw_obs", "ANY", "Raw observation dict {rgb, depth} (for policy nodes)"),
        PortDef(
            "instruction_text",
            "TEXT",
            "Raw natural-language instruction for the current episode (per-episode "
            "metadata; feed a tokenizer/VLM node)",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Pure read — no lifecycle action. A finished episode observed again
        # returns the terminal frame; re-arming is reset's job. The manager
        # caches the frame per pose, so this renders at most once per move.
        mgr = _get_env()
        frame = await _run_sync(mgr.current_frame)
        rgb = depth = None
        if isinstance(frame, dict) and "error" not in frame:
            rgb = np.asarray(frame["rgb"], dtype=np.uint8)
            depth = np.asarray(frame["depth"], dtype=np.float32)
        pose = await _run_sync(mgr.get_agent_state)
        if not isinstance(pose, dict) or "error" in pose:
            pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
        intrinsics = mgr.get_cam_intrinsics()
        instruction_text = await _run_sync(mgr.get_instruction_text)
        self._self_log("has_rgb", rgb is not None)
        self._self_log("has_depth", depth is not None)
        if rgb is not None:
            self._self_log("rgb_shape", list(rgb.shape))
        return {
            "rgb": rgb,
            "depth": depth,
            "pose": pose,
            "intrinsics": intrinsics,
            "raw_obs": {"rgb": rgb, "depth": depth},
            "instruction_text": instruction_text,
        }


class ObservePanoramaVLNVerseTool(BaseCanvasNode):
    """Multi-view aligned RGB-D panorama at the agent's position.

    dir_id 0 = current heading, increasing CCW in (360/n)° steps
    (dir 3 = Left at n_views=12) — the NavHarness strip convention.
    Expensive (one Isaac render + warmup per view) — pull on demand only.
    """

    node_type = "env_vlnverse__observe_panorama"
    display_name = "VLNVerse: Observe (panorama)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Multi-view aligned RGB+Depth panorama at the agent's position (CCW)"
    category = "environment"
    icon = "Camera"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-render (optional)", optional=True),
    ]
    output_ports = [
        PortDef(
            "views",
            "ANY",
            "List of {dir_id, heading_deg, rgb_base64, depth_base64, depth_raw_base64} "
            "(views_rgbd mode)",
        ),
        PortDef("directions", "TEXT", "Per-view heading labels (JSON)"),
        PortDef("n_views", "ANY", "Number of views returned"),
        PortDef("composite", "IMAGE", "Stitched panorama grid (composite mode)"),
    ]
    config_fields = [
        ConfigField(
            name="representation",
            field_type="select",
            label="Representation",
            default="views_rgbd",
            options=[
                {"value": "views_rgbd", "label": "Aligned RGB-D views (waypoint predictor)"},
                {"value": "composite", "label": "Stitched composite grid (single image)"},
            ],
        ),
        ConfigField(
            name="n_views",
            field_type="slider",
            label="Number of views",
            default=12,
            min=4,
            max=24,
            step=4,
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        cfg = self.config or {}
        representation = cfg.get("representation", "views_rgbd")
        n_views = int(cfg.get("n_views", 12))
        mgr = _get_env()

        if representation == "composite":
            result = await _run_sync(mgr.render_panorama, n_views)
            if "error" in result:
                self._self_log("error", result["error"])
                return {"views": [], "directions": "[]", "n_views": 0, "composite": None}
            # Manager hands back the numpy canvas directly (no b64 round-trip);
            # views carry labels only in composite mode.
            composite_arr = result.get("composite")
            directions = json.dumps(result.get("views", []))
            self._self_log("n_views", result.get("n_views", 0))
            self._self_log(
                "composite_shape",
                list(composite_arr.shape) if composite_arr is not None else None,
            )
            return {
                "views": result.get("views", []),
                "directions": directions,
                "n_views": result.get("n_views", 0),
                "composite": composite_arr,
            }

        result = await _run_sync(mgr.render_panorama_rgbd, n_views)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"views": [], "directions": "[]", "n_views": 0, "composite": None}
        views = result.get("views", [])
        directions = json.dumps(
            [{"dir_id": v.get("dir_id"), "heading_deg": v.get("heading_deg")} for v in views]
        )
        self._self_log("n_views", result.get("n_views", 0))
        return {
            "views": views,
            "directions": directions,
            "n_views": result.get("n_views", 0),
            "composite": None,
        }


class EvaluateVLNVerseTool(BaseCanvasNode):
    """Thin metric sink over the env's terminal state (after-loop band).

    Emits the accumulator's metric dict (success, spl, oracle_success,
    ndtw, colrate, ...). Wire ``trigger`` from the loop's
    ``iter_out.final_stop`` so it fires exactly once when the episode ends
    (env-template decision E). Read-only; works whether the episode ended
    by STOP (terminated) or by step-budget truncation.
    """

    node_type = "env_vlnverse__evaluate"
    display_name = "VLNVerse: Evaluate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    description = "Pull current episode metrics (SR/SPL/nDTW/colrate) without stepping"
    category = "evaluation"
    icon = "BarChart"
    input_ports = [
        PortDef("trigger", "TEXT", "Trigger evaluation (any value)", optional=True),
    ]
    output_ports = [
        PortDef("metrics", "METRICS", "VLNVerse episode metrics dict"),
        PortDef("success", "TEXT", "1 if agent ended within 3 m of goal, 0 otherwise"),
        PortDef("spl", "TEXT", "Success weighted by Path Length"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_env()
        result = await _run_sync(mgr.evaluate)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"metrics": {}, "success": "0", "spl": "0"}

        success = result.get("success", 0)
        spl = result.get("spl", 0)
        self._self_log("success", success)
        self._self_log("spl", f"{spl:.3f}" if isinstance(spl, (int, float)) else str(spl))
        for k in ("ndtw", "colrate", "distance_to_goal", "path_length"):
            if result.get(k) is not None:
                v = result[k]
                self._self_log(k, f"{v:.3f}" if isinstance(v, (int, float)) else str(v))
        return {
            "metrics": result,
            "success": str(int(success)) if isinstance(success, (int, float)) else "0",
            "spl": f"{spl:.4f}" if isinstance(spl, (int, float)) else "0",
        }


# ══════════════════════════════════════════════════════════════════════
# VLNVerseEnvPanel — episode/split control plane for the canvas panel
# Implements BaseEnvPanel; registered via EnvVLNVerseNodeSet.env_panel.
# ══════════════════════════════════════════════════════════════════════


class VLNVerseEnvPanel(BaseEnvPanel):
    """Canvas env panel for the VLNVerse environment.

    Three-field cascade (``dataset → split → episode_index``) — mirrors
    HabitatEnvPanel. Changing dataset resets split + episode; changing
    split swaps the episode list (no Isaac reboot). Each field change
    emits an ``episode_reset`` signal so any ``lifetime="episode"`` state
    container clears downstream.
    """

    name = "env_vlnverse"
    display_name = "VLNVerse (Isaac)"
    fields = [
        EnvPanelField("dataset", "select", "Dataset"),
        EnvPanelField("split", "select", "Split"),
        EnvPanelField("episode_index", "select", "Episode"),
    ]
    actions = [
        EnvPanelAction("play", "Play", side_effect="run_start"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop", "Stop", side_effect="run_stop", enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "dataset": "fine",
            "split": "val_unseen",
            "episode_index": 0,
        }

    # ── Helpers ──

    def _mgr(self) -> VLNVerseEnvManager:
        return VLNVerseEnvManager.get()

    async def _run(self, fn, *args):
        mgr = self._mgr()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(mgr.executor, fn, *args)

    async def _episode_info(self, index: int) -> dict[str, Any]:
        mgr = self._mgr()
        data = await self._run(mgr.get_episodes_list, index, 1)
        eps = data.get("episodes", []) if isinstance(data, dict) else []
        return eps[0] if eps else {"error": "Index out of range"}

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "dataset": self._state.get("dataset", ""),
            "split": self._state.get("split", ""),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    # ── Lifecycle hooks ──

    async def on_load(self) -> dict[str, Any]:
        # Server mode: the env runs in a spawned subprocess, so the local
        # manager singleton is empty by design (no control-plane bridge yet).
        ctx = getattr(self, "_context", {}) or {}
        dataset = self._state.get("dataset", "fine")
        splits = _dataset_splits(dataset)
        if ctx.get("mode") == "server":
            return {
                "available": False,
                "dataset": dataset,
                "split": "",
                "episode_index": 0,
                "episode_count": 0,
                "datasets": list(_DATASETS),
                "splits": splits,
                "message": (
                    "VLNVerse is running in server mode (subprocess). Episode "
                    "control from this panel is not yet supported."
                ),
            }
        mgr = self._mgr()
        if not mgr.initialized:
            return {
                "available": False,
                "dataset": dataset,
                "split": "",
                "episode_index": 0,
                "episode_count": 0,
                "datasets": list(_DATASETS),
                "splits": splits,
                "message": (
                    "VLNVerse environment not initialized. Load env_vlnverse "
                    "from the NodeSet Manager to enable episode control."
                ),
            }

        status = await self._run(mgr.get_status)
        current_index = mgr._episode_index
        self._state["split"] = status.get("split", self._state.get("split"))
        self._state["dataset"] = status.get("dataset", dataset)
        self._state["episode_index"] = current_index

        ep_info = await self._episode_info(current_index)
        return {
            "available": True,
            "dataset": self._state["dataset"],
            "split": self._state["split"],
            "episode_index": current_index,
            "episode_count": status.get("total_episodes", 0),
            "datasets": list(_DATASETS),
            "splits": _dataset_splits(self._state["dataset"]),
            "step_budget": status.get("max_steps", _DEFAULT_MAX_STEPS),
            "current_episode": ep_info,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        if name == "dataset":
            new_dataset = str(value)
            if new_dataset not in _DATASETS:
                return await self.on_load()
            prev_dataset = self._state.get("dataset")
            self._state["dataset"] = new_dataset
            splits = _dataset_splits(new_dataset)
            self._state["split"] = splits[0] if splits else ""
            self._state["episode_index"] = 0
            # A dataset change re-routes the episode file — reload the list
            # so a following split push that happens to equal the new
            # default isn't short-circuited (habitat's RxR/R2R lesson).
            if prev_dataset and prev_dataset != new_dataset and self._state["split"]:
                await self._switch_split(self._state["split"])
        elif name == "split":
            new_split = str(value)
            if new_split and new_split != self._state.get("split"):
                await self._switch_split(new_split)
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                new_index = int(value)
            except (TypeError, ValueError):
                new_index = 0
            self._state["episode_index"] = new_index
            ep_info = await self._episode_info(new_index)
            state = await self.on_load()
            # on_load rebinds episode_index from the env's current episode,
            # clobbering the value just armed when the env hasn't been reset
            # yet. Restore it so the subsequent on_action("play") reads the
            # new index — without this, batch eval reuses episode 0 for
            # every dispatched index (habitat's fix, kept verbatim).
            self._state["episode_index"] = new_index
            state["episode_index"] = new_index
            state["current_episode"] = ep_info
            state["side_effect"] = "signal"
            state["signal_name"] = "episode_reset"
            state["signal_payload"] = self._episode_reset_payload()
            return state
        else:
            self._state[name] = value
            return await self.on_load()

        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = self._episode_reset_payload()
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        mgr = self._mgr()
        if name in ("play", "reset"):
            if not mgr.initialized:
                return {
                    "ok": False,
                    "side_effect": "none",
                    "error": "VLNVerse environment not initialized",
                }
            await self._run(mgr.set_episode_by_index, int(self._state["episode_index"]))
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
        if field == "dataset":
            return [{"value": d, "label": d} for d in _DATASETS]
        if field == "split":
            splits = _dataset_splits(self._state.get("dataset", "fine"))
            return [{"value": s, "label": s} for s in splits]
        if field == "episode_index":
            if not mgr.initialized:
                return []
            data = await self._run(mgr.get_episodes_list, 0, 10000, True)  # light
            episodes = data.get("episodes", []) if isinstance(data, dict) else []
            return [
                {
                    "value": ep.get("index", i),
                    "label": "{}: {}".format(ep.get("index", i), ep.get("scene_id", "")),
                }
                for i, ep in enumerate(episodes)
            ]
        return []

    # ── Split switching ──

    async def _switch_split(self, split: str) -> None:
        """Swap the episode list for the cascade's (dataset, split). Cheap:
        no Isaac reboot — the scene switches lazily on episode seating."""
        mgr = self._mgr()
        dataset = self._state.get("dataset", "fine")
        if split not in _dataset_splits(dataset):
            log.warning(
                "VLNVerseEnvPanel: split %r not valid for dataset %r", split, dataset
            )
            return
        if not mgr.initialized:
            # Not booted yet — remember the selection for panel display only.
            # NOTE: initialize() takes its dataset/split from the registry
            # load kwargs and does NOT consult panel state; a pre-init
            # selection here does not steer the eventual boot.
            self._state["split"] = split
            return
        result = await self._run(mgr.switch_split, dataset, split)
        if isinstance(result, dict) and "error" in result:
            log.error("VLNVerseEnvPanel: switch_split failed: %s", result["error"])
            return
        self._state["split"] = split


# ══════════════════════════════════════════════════════════════════════
# EnvVLNVerseNodeSet — the unified nodeset
# ══════════════════════════════════════════════════════════════════════


class EnvVLNVerseNodeSet(BaseNodeSet):
    """VLNVerse (Isaac Sim 5.1) environment as a NodeSet.

    Exposes ``env_mgr`` semantics compatible with env_habitat's manager
    surface (same /call/{fn} names) — the sim itself renders in a
    separate Isaac 5.1 worker process, so this nodeset's own process
    stays numpy-light and can load locally OR auto-host in the
    ``ac-vlnverse`` conda env (``?mode=server``).
    """

    name = "env_vlnverse"
    description = "VLNVerse (Isaac Sim 5.1) VLN environment — kujiale scenes"
    server_python = conda_env_python("ac-vlnverse", "VLNVERSE_PYTHON")
    env_panel = VLNVerseEnvPanel
    statefulness = "stateful"  # Stateful env: per-worker scene + agent pose.
    # A step is a macro action (turn+walk) but the render cost sits in
    # observe_*; panorama = 12 Isaac renders with warmup. 60s/step leaves
    # headroom for cross-scene episode switches (USD load + stabilize)
    # that happen inside set_episode during batch eval.
    default_per_step_budget_sec = 60.0
    # RTX RaytracedLighting SimulationApp steady state (author preset —
    # measurement wins once calibrated on this machine).
    expected_vram_mb = 6000

    def __init__(self) -> None:
        super().__init__()
        self._mgr = VLNVerseEnvManager.get()

    def get_tools(self) -> list:
        return [
            ResetVLNVerseTool(),
            StepDiscreteVLNVerseTool(),
            StepHighToLowVLNVerseTool(),
            ObserveEgocentricVLNVerseTool(),
            ObservePanoramaVLNVerseTool(),
            EvaluateVLNVerseTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Initialize the VLNVerse environment (boots the Isaac worker).

        Kwargs:
            dataset:   "fine" (default) or "coarse".
            split:     Split name (default "val_unseen").
            gpu_id:    CUDA device index for the Isaac worker (default 0).
            max_steps: Episode step budget (default 20 — macro actions).
        """
        if self._mgr.initialized:
            log.info("VLNVerse already initialized — skipping")
            return
        dataset = kwargs.get("dataset", "fine")
        split = kwargs.get("split", "val_unseen")
        gpu_id = kwargs.get("gpu_id", 0)
        max_steps = kwargs.get("max_steps", _env_max_steps())
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            self._mgr.initialize,
            dataset,
            split,
            gpu_id,
            max_steps,
        )
        log.info("EnvVLNVerseNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        """Eval metadata for the VLNVerse environment.

        Splits are the union of both datasets' filtered lists so the Eval
        page can list every selectable option; episode_counts covers the
        currently-loaded split only.
        """
        by_dataset = {d: _dataset_splits(d) for d in _DATASETS}
        splits = list(dict.fromkeys(s for ds in by_dataset.values() for s in ds))
        metadata = {
            "env_name": "vlnverse_isaac",
            "datasets": list(_DATASETS),
            "splits": splits,
            # Per-dataset lists so eval-page combinatorics can be validated up
            # front — the flat union can pair ("coarse", <fine-only split>).
            "splits_by_dataset": by_dataset,
            "episode_counts": {},
            "metrics": [
                "success",
                "spl",
                "oracle_success",
                "stop_success",
                "success_efficiency",
                "ndtw",
                "colrate",
                "distance_to_goal",
                "path_length",
            ],
            "supports_set_episode": self._mgr.initialized,
            "step_budget": self._mgr.max_steps,
        }
        if self._mgr.initialized:
            with contextlib.suppress(Exception):
                metadata["episode_counts"] = {
                    self._mgr._split: self._mgr.get_total_episodes()
                }
        return metadata

    async def shutdown(self) -> None:
        # Via the executor, not inline: shutdown contends on the manager lock,
        # which an in-flight deferred scene load can hold for 20-30 s — inline
        # that would freeze the FastAPI event loop (no /health) during teardown.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._mgr.executor, self._mgr.shutdown)
