from __future__ import annotations

"""Matterport3D env nodeset — discrete panoramic simulator wrapper.

Env-output contract (enforce this — no prose in env output):

    observe()       → {viewpoint_id, scan_id, heading_deg, elevation_deg,
                       navigable: {vp_id: {heading_rad, elevation_rad, distance}},
                       scene_descriptions: list[str] | None,
                       scene_objects: list[dict] | None,
                       scene_summary: str}
    get_navigable() → {vp_id: {heading_rad, elevation_rad, distance}}  (via observe)
    navigate_to(vp) → {success: bool, reached_viewpoint, target_viewpoint?,
                       trajectory: list[vp_id], error?, reason?}
    step(action)    → (internal) records structured trajectory entry, no prose

Invariant: no prompt fragments. No NavGPT-specific shapes. No compass strings.
All LLM-facing formatting lives in ``workspace/nodesets/navgpt_mp3d_tools.py``.

---

EnvMP3DNodeSet — Matterport3D discrete panoramic navigation as a unified NodeSet.

Works both as a local in-process nodeset and as an auto-hosted server:
  Local:  POST /api/components/nodesets/env_mp3d/load
  Server: POST /api/components/nodesets/env_mp3d/load?mode=server

Wraps MatterSim (Matterport3DSimulator) for the R2R/R4R discrete nav tasks.
Connectivity graphs live in ``third_party/Matterport3DSimulator/connectivity/``.
R2R episode JSON files live in ``third_party/Matterport3DSimulator/tasks/R2R/data/``.

last updated: 2026-04-14
"""


import asyncio
import base64
import concurrent.futures
import contextlib
import functools
import glob
import io
import json
import logging
import math
import os
import random
import threading
from typing import Any, ClassVar

import networkx as nx
import numpy as np
from PIL import Image

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)
from app.components.env_panel import BaseEnvPanel, EnvPanelAction, EnvPanelField

log = logging.getLogger("agentcanvas.matterport3d")

# Default data paths (overridable via env vars)
# __file__ lives at workspace/nodesets/env/env_mp3d/__init__.py — four parents
# to reach the repo root.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
)
_MP3D_SIM_ROOT = os.environ.get(
    "MP3D_SIM_ROOT",
    os.path.join(_REPO_ROOT, "third_party", "Matterport3DSimulator"),
)
_CONNECTIVITY_DIR = os.path.join(_MP3D_SIM_ROOT, "connectivity")

# Canonical step-1 tasks root (`fetch_episodes_vln.sh` output). R2R retains the
# legacy submodule path as a fallback for fresh clones that skipped the
# installer — every other dataset is canonical-only.
_TASKS_ROOT = os.environ.get(
    "MP3D_TASKS_ROOT",
    os.path.join(_REPO_ROOT, "data", "mp3d", "tasks"),
)
_R2R_DATA_DIR = os.path.join(_MP3D_SIM_ROOT, "tasks", "R2R", "data")

_DATASET_DIRS: dict[str, list[str]] = {
    "R2R": [os.path.join(_TASKS_ROOT, "R2R"), _R2R_DATA_DIR],
    "R4R": [os.path.join(_TASKS_ROOT, "R4R")],
    "RxR": [os.path.join(_TASKS_ROOT, "RxR")],
    "REVERIE": [os.path.join(_TASKS_ROOT, "REVERIE")],
    "CVDN": [os.path.join(_TASKS_ROOT, "CVDN")],
    "NDH": [os.path.join(_TASKS_ROOT, "NDH")],
}

# Per-dataset split-name glob, evaluated relative to the dataset dir.
# ``{split}`` expands to the split token reported via ``list_splits``.
_DATASET_FILE_PATTERNS: dict[str, str] = {
    "R2R": "R2R_{split}.json",
    "R4R": "R4R_{split}.json",
    "RxR": "rxr_{base}_guide.jsonl.gz",  # {base} strips the _<lang> suffix
    "REVERIE": "REVERIE_{split}.json",
    "CVDN": "{split}.json",
    "NDH": "{split}.json",
}

# BCP-47 prefix table for RxR split-name language suffix (`val_unseen_en`).
_RXR_LANG_MAP: dict[str, str] = {
    "en": "en",  # matches en-IN, en-US
    "hi": "hi",
    "te": "te",
}


def _resolve_dataset_dir(dataset: str) -> str | None:
    """Return the first existing directory for *dataset*, or None."""
    for path in _DATASET_DIRS.get(dataset, []):
        if os.path.isdir(path):
            return path
    return None


def _load_first_existing_json(paths: list[str]) -> Any:
    """Load the first readable JSON file from *paths*, or return None."""
    for path in paths:
        if not (path and os.path.isfile(path)):
            continue
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            log.exception("Failed to load JSON from %s", path)
    return None


def _count_json_records(path: str) -> int:
    """Return ``len(json.load(f))`` with exception guard, or 0 on failure."""
    try:
        with open(path) as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        log.warning("Could not count records in %s", path)
        return 0


def _resolve_mp3d_data_path() -> str:
    """Resolve the MP3D dataset root (parent of ``v1/scans/``).

    Precedence:
      1. ``MP3D_DATA_PATH`` — parent of ``v1/scans/`` (our native var).
      2. ``MATTERPORT_DATA_DIR`` — historically points *at* ``v1/scans/``
         per MatterSim README and ``install_ac_mp3d.sh``; parent-of is used.
      3. ``{REPO_ROOT}/data/mp3d`` — repo default (post data-layout unification).
    """
    if env := os.environ.get("MP3D_DATA_PATH"):
        return env
    if env := os.environ.get("MATTERPORT_DATA_DIR"):
        normalised = os.path.normpath(env)
        # Anchor on path separator so /data/my_scans etc. are NOT stripped.
        return os.path.dirname(normalised) if normalised.endswith(os.sep + "scans") else env
    return os.path.join(_REPO_ROOT, "data", "mp3d")


# ══════════════════════════════════════════════════════════════════════
# MP3DEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class MP3DEnvManager:
    """Manages a single Matterport3D Simulator instance.

    All public methods are blocking and should be called via
    ``asyncio.run_in_executor(mgr.executor, fn)`` from async code.
    A single-thread executor enforces MatterSim's thread affinity.

    Episode loading:
        Call ``load_r2r_episodes(split)`` to populate ``_episodes``.
        Then ``new_episode()`` cycles through them.  If no R2R data is
        available the manager falls back to random starts via
        ``sim.newRandomEpisode()``.

    Viewing angles:
        MatterSim in discretized mode exposes 36 fixed views:
          - elevations: -30°, 0°, +30°  (rows 0-11, 12-23, 24-35)
          - headings:   0°..330° in 30° steps per elevation row
        Horizon-level views are viewIndex 12-23 (elevation ≈ 0).
    """

    _instance: MP3DEnvManager | None = None

    def __init__(self) -> None:
        self._sim = None
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mp3d",
        )
        self._current_scan: str = ""
        self._current_viewpoint: str = ""
        self._step_count: int = 0
        self._episode_done: bool = False
        # Unified dataset-aware episode store:
        #   self._episodes["R2R"]["val_unseen"]       → [flattened ep dicts]
        #   self._episodes["RxR"]["val_unseen_en"]    → [...]
        #   self._episodes["REVERIE"]["val_seen"]     → [...]
        # Each flattened record conforms to the base schema:
        #   {instr_id, path_id, scan, heading, path, instruction, distance,
        #    dataset, extras}
        self._episodes: dict[str, dict[str, list[dict]]] = {}
        # Split availability (dataset → split → count) populated at init without
        # retaining episode data. Used by the env panel to list splits.
        self._episode_counts: dict[str, dict[str, int]] = {}
        self._current_split_idx: int = -1  # legacy mattersim-mode index
        self._width: int = 640
        self._height: int = 480
        self._vfov_deg: float = 60.0
        self._depth_enabled: bool = True
        self._dataset: str = "R2R"
        self._split: str = "val_unseen"

        # ── Graph navigation infrastructure (additive, coexists with MatterSim) ──
        self._graphs: dict = {}  # scan_id → nx.Graph
        self._shortest_paths: dict = {}  # scan_id → {src → {dst → [path]}}
        self._shortest_distances: dict = {}  # scan_id → {src → {dst → float}}
        self._graph_viewpoint: str = ""
        self._graph_heading: float = 0.0
        self._graph_elevation: float = 0.0
        self._graph_scan: str = ""
        self._graph_trajectory: list = []
        self._graph_episode: dict | None = None
        self._graph_history: list = []

    # ── Singleton access ──

    @classmethod
    def get(cls) -> MP3DEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Lifecycle ──

    def initialize(
        self,
        width: int = 640,
        height: int = 480,
        vfov: float = 60.0,
        preloading: bool = False,
        depth_enabled: bool = False,
        dataset: str = "R2R",
        split: str = "val_unseen",
        cache_size: int = 10,
    ) -> dict:
        """Create and initialize the MatterSim simulator.

        Args:
            width:         Camera resolution width (pixels).
            height:        Camera resolution height (pixels).
            vfov:          Vertical field of view in degrees.
            preloading:    If True, preloads all panoramas into RAM (~50 GB).
                           Set False for fast startup during development.
            depth_enabled: Enable depth output alongside RGB.
            dataset:       Default dataset (R2R / R4R / RxR / REVERIE / CVDN / NDH).
            split:         Default split within the dataset. RxR adds a language
                           suffix (val_unseen_en / val_unseen_hi / val_unseen_te).
            cache_size:    LRU cache size for loaded panoramas.

        Returns:
            Status dict with initialized flag and episode count.
        """
        with self._lock:
            if self._sim is not None:
                log.warning("MP3DEnvManager already initialized — skipping")
                return self._get_status_unlocked()

            self._width = width
            self._height = height
            self._vfov_deg = vfov
            self._depth_enabled = depth_enabled
            self._dataset = dataset
            self._split = split

            try:
                import MatterSim
            except ImportError as exc:
                raise ImportError(
                    "MatterSim not found. Build with: cd third_party/Matterport3DSimulator && "
                    "mkdir build && cd build && cmake .. && make -j4"
                ) from exc

            sim = MatterSim.Simulator()
            sim.setCameraResolution(width, height)
            sim.setCameraVFOV(math.radians(vfov))
            sim.setPreloadingEnabled(preloading)
            sim.setDepthEnabled(depth_enabled)
            sim.setBatchSize(1)
            sim.setCacheSize(cache_size)
            sim.setElevationLimits(math.radians(-40), math.radians(40))
            sim.setDiscretizedViewingAngles(True)

            # Validate data paths before initializing (P3.5)
            data_path = _resolve_mp3d_data_path()
            scans_dir = os.path.join(data_path, "v1", "scans")
            if not os.path.isdir(scans_dir):
                raise RuntimeError(
                    f"MP3D scans dir not found: {scans_dir}. "
                    f"Set MP3D_DATA_PATH to the dataset root (parent of v1/scans/, "
                    f"e.g. {os.path.join(_REPO_ROOT, 'data', 'mp3d')}) "
                    f"or MATTERPORT_DATA_DIR to v1/scans/ itself "
                    f"(e.g. {os.path.join(_REPO_ROOT, 'data', 'mp3d', 'v1', 'scans')}). "
                    f"Default repo layout: {os.path.join(_REPO_ROOT, 'data', 'mp3d', 'v1', 'scans')}"
                )
            if not os.listdir(scans_dir):
                raise RuntimeError(
                    f"MP3D scans dir is empty: {scans_dir}. "
                    f"Download matterport_skybox_images per scan via the signed-ToU "
                    f"download_mp.py from the Matterport3D dataset repo."
                )
            if _resolve_dataset_dir("R2R") is None:
                raise RuntimeError(
                    f"No R2R data found under {_TASKS_ROOT}/R2R or the legacy "
                    f"submodule path {_R2R_DATA_DIR}. Run "
                    f"`bash scripts/data/fetch_episodes_vln.sh --r2r`."
                )

            # Point simulator at scans directory (MatterSim expects the
            # parent of <scan>/matterport_skybox_images/, i.e. v1/scans).
            if os.path.isdir(_CONNECTIVITY_DIR):
                sim.setDatasetPath(scans_dir)
                sim.setNavGraphPath(_CONNECTIVITY_DIR)
                log.info("MP3D scans dir: %s", scans_dir)
                log.info("MP3D connectivity dir: %s", _CONNECTIVITY_DIR)
            else:
                log.warning(
                    "Connectivity dir not found: %s — sim may fail",
                    _CONNECTIVITY_DIR,
                )

            sim.initialize()
            self._sim = sim
            log.info(
                "MatterSim initialized (%dx%d vfov=%.0f° depth=%s preload=%s)",
                width,
                height,
                vfov,
                depth_enabled,
                preloading,
            )
            log.info("Observation data cache: LRU maxsize=4096")

            # Walk every dataset dir and record per-split episode counts.
            # Data payloads are not retained — env panel populates on demand.
            self._scan_datasets_and_splits()

            # Eager-load the requested (dataset, split) so initial play works.
            self._load_episodes(dataset, split)

            # Start first episode (best-effort — scene data may not be downloaded)
            try:
                self._start_first_episode_unlocked()
            except (ValueError, RuntimeError) as exc:
                log.warning(
                    "Could not start initial episode (scene data may not be "
                    "downloaded yet): %s — server will start without an active "
                    "episode. Call new_episode() when data is available.",
                    exc,
                )

            return self._get_status_unlocked()

    def shutdown(self) -> None:
        """Close the simulator and release resources (P3.8)."""
        with self._lock:
            if self._sim is not None:
                log.info("Shutting down MatterSim")
                # MatterSim has no explicit close() — release the reference
                self._sim = None
        # Shut down the thread pool outside the lock to avoid deadlock with in-flight tasks
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    @property
    def initialized(self) -> bool:
        return self._sim is not None

    @property
    def executor(self) -> concurrent.futures.Executor:
        if self._executor is None:
            raise RuntimeError("MP3DEnvManager has been shut down — executor is gone")
        return self._executor

    @property
    def current_graph_viewpoint(self) -> str:
        """Public accessor for the current graph navigation viewpoint (P3.9)."""
        return self._graph_viewpoint

    # ── Private helpers ──

    # ── Dataset discovery (lightweight, filename-only) ──

    def _scan_datasets_and_splits(self) -> None:
        """Populate ``_episode_counts[dataset][split]`` with sentinel ``-1`` for
        every (dataset, split) pair discoverable on disk.

        Discovery is purely filename-based — **no JSON parsing, no gzip
        streaming** — so init cost is ``O(number of files)`` even with 20 GB
        of RxR data on disk. Real episode counts fill in lazily on first
        ``get_episode_count()`` call (usually driven by the env panel
        panel's ``on_load``).
        """
        self._episode_counts.clear()
        for dataset in _DATASET_DIRS:
            ds_dir = _resolve_dataset_dir(dataset)
            if ds_dir is None:
                continue
            splits = self._discover_splits_in_dir(dataset, ds_dir)
            if splits:
                self._episode_counts[dataset] = {s: -1 for s in splits}
                log.info(
                    "Dataset %-8s: %d splits discovered (%s)",
                    dataset,
                    len(splits),
                    ", ".join(splits[:4]) + ("..." if len(splits) > 4 else ""),
                )

    @staticmethod
    def _discover_splits_in_dir(dataset: str, ds_dir: str) -> list[str]:
        """Return sorted split names from filenames only (no file reads)."""
        try:
            fnames = sorted(os.listdir(ds_dir))
        except OSError:
            return []

        splits: list[str] = []
        if dataset == "R2R":
            splits = [
                f[len("R2R_") : -len(".json")]
                for f in fnames
                if f.startswith("R2R_") and f.endswith(".json")
            ]
        elif dataset == "R4R":
            splits = [
                f[len("R4R_") : -len(".json")]
                for f in fnames
                if f.startswith("R4R_") and f.endswith(".json")
            ]
        elif dataset == "RxR":
            # Filename-based discovery only. Each rxr_{base}_guide.jsonl.gz
            # is multilingual; we surface three split rows per base file
            # (_en/_hi/_te). Empty-language slots return 0 on real load.
            bases = [
                f[len("rxr_") : -len("_guide.jsonl.gz")]
                for f in fnames
                if f.startswith("rxr_") and f.endswith("_guide.jsonl.gz")
            ]
            for base in bases:
                for lang in _RXR_LANG_MAP:
                    splits.append(f"{base}_{lang}")
        elif dataset == "REVERIE":
            splits = [
                f[len("REVERIE_") : -len(".json")]
                for f in fnames
                if f.startswith("REVERIE_") and f.endswith(".json")
            ]
        elif dataset in ("CVDN", "NDH"):
            splits = [f[: -len(".json")] for f in fnames if f.endswith(".json")]
        return sorted(splits)

    # ── Episode loading (dispatcher + per-dataset loaders) ──

    def _load_episodes(self, dataset: str, split: str) -> None:
        """Populate ``self._episodes[dataset][split]`` with flattened episodes.

        Idempotent: re-calling for an already-loaded (dataset, split) is a
        no-op. Must be called while holding ``self._lock``.
        """
        if self._episodes.get(dataset, {}).get(split) is not None:
            return
        ds_dir = _resolve_dataset_dir(dataset)
        if ds_dir is None:
            log.warning("Dataset dir missing for %s", dataset)
            self._episodes.setdefault(dataset, {})[split] = []
            return

        loader = {
            "R2R": self._load_r2r,
            "R4R": self._load_r4r,
            "RxR": self._load_rxr,
            "REVERIE": self._load_reverie,
            "CVDN": self._load_cvdn,
            "NDH": self._load_ndh,
        }.get(dataset)
        if loader is None:
            log.warning("No loader registered for dataset %s", dataset)
            self._episodes.setdefault(dataset, {})[split] = []
            return

        try:
            records = loader(ds_dir, split)
        except Exception:
            log.exception("Loader failed for %s/%s", dataset, split)
            records = []

        self._episodes.setdefault(dataset, {})[split] = records
        if records:
            self._current_split_idx = 0
        log.info("Loaded %d episodes: %s/%s", len(records), dataset, split)

    def _explode_instructions(
        self,
        dataset: str,
        item: dict,
        instructions: list[str],
        extras_for: Any = None,
    ) -> list[dict]:
        """Emit one normalised record per instruction. Shared by R2R/R4R/REVERIE."""
        out: list[dict] = []
        for j, instr in enumerate(instructions):
            extras = {}
            if callable(extras_for):
                extras = extras_for(item, j) or {}
            out.append(
                {
                    "instr_id": f"{item.get('path_id', item.get('id', ''))}_{j}",
                    "path_id": item.get("path_id", item.get("id", "")),
                    "scan": item["scan"],
                    "heading": float(item.get("heading", 0.0)),
                    "path": item.get("path", []),
                    "instruction": instr,
                    "distance": float(item.get("distance", 0.0)),
                    "dataset": dataset,
                    "extras": extras,
                }
            )
        return out

    def _path_distance(self, scan: str, path: list[str]) -> float:
        """Sum connectivity-graph shortest-path distances along *path*."""
        if len(path) < 2:
            return 0.0
        try:
            self.graph_ensure_scan(scan)
        except ValueError:
            return 0.0
        sd = self._shortest_distances.get(scan)
        if not sd:
            return 0.0
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            try:
                total += float(sd[a][b])
            except KeyError:
                return 0.0
        return total

    # -- R2R --
    def _load_r2r(self, ds_dir: str, split: str) -> list[dict]:
        candidate_paths = [
            os.path.join(ds_dir, f"R2R_{split}.json"),
            os.path.join(ds_dir, f"R2R_{split}_enc.json"),
            os.environ.get(f"R2R_{split.upper()}_JSON", ""),
        ]
        raw = _load_first_existing_json(candidate_paths)
        if raw is None:
            return []
        records: list[dict] = []
        for item in raw:
            instructions = item.get("instructions") or [item.get("instruction", "")]
            records.extend(self._explode_instructions("R2R", item, instructions))
        return records

    # -- R4R --
    def _load_r4r(self, ds_dir: str, split: str) -> list[dict]:
        raw = _load_first_existing_json([os.path.join(ds_dir, f"R4R_{split}.json")])
        if raw is None:
            return []

        def _extras(item: dict, _j: int) -> dict:
            return {
                "first_path_id": item.get("first_path_id"),
                "second_path_id": item.get("second_path_id"),
                "shortest_path": item.get("shortest_path"),
                "shortest_path_distance": item.get("shortest_path_distance"),
            }

        records: list[dict] = []
        for item in raw:
            instructions = item.get("instructions") or [item.get("instruction", "")]
            records.extend(self._explode_instructions("R4R", item, instructions, _extras))
        return records

    # -- RxR (one jsonl.gz per base split, filtered by BCP-47 language) --
    def _load_rxr(self, ds_dir: str, split: str) -> list[dict]:
        # Split names encode language: "val_unseen_en" → base="val_unseen", lang="en"
        for lang_token in _RXR_LANG_MAP:
            suffix = f"_{lang_token}"
            if split.endswith(suffix):
                base = split[: -len(suffix)]
                lang_prefix = _RXR_LANG_MAP[lang_token]
                break
        else:
            log.warning(
                "RxR split %r is missing a language suffix (_en/_hi/_te)",
                split,
            )
            return []

        path = os.path.join(ds_dir, f"rxr_{base}_guide.jsonl.gz")
        if not os.path.isfile(path):
            log.warning("RxR file missing: %s", path)
            return []

        pose_trace_root = os.path.join(ds_dir, "pose_traces", f"rxr_{base}")

        records: list[dict] = []
        try:
            import gzip

            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    lang = (rec.get("language") or "").split("-", 1)[0].lower()
                    if lang != lang_prefix:
                        continue
                    scan = rec["scan"]
                    ep_path = rec.get("path", [])
                    instr_id = str(rec.get("instruction_id", rec.get("path_id", "")))
                    pose_trace_path = os.path.join(
                        pose_trace_root,
                        f"{instr_id}_guide_pose_trace.npz",
                    )
                    records.append(
                        {
                            "instr_id": instr_id,
                            "path_id": rec.get("path_id"),
                            "scan": scan,
                            "heading": float(rec.get("heading", 0.0)),
                            "path": ep_path,
                            "instruction": rec.get("instruction", ""),
                            "distance": self._path_distance(scan, ep_path),
                            "dataset": "RxR",
                            "extras": {
                                "language": rec.get("language"),
                                "instruction_id": rec.get("instruction_id"),
                                "annotator_id": rec.get("annotator_id"),
                                "timed_instruction": rec.get("timed_instruction"),
                                "edit_distance": rec.get("edit_distance"),
                                "pose_trace_path": pose_trace_path
                                if os.path.isfile(pose_trace_path)
                                else None,
                            },
                        }
                    )
        except Exception:
            log.exception("Failed to stream RxR file %s", path)
            return []
        return records

    # -- REVERIE --
    def _load_reverie(self, ds_dir: str, split: str) -> list[dict]:
        raw = _load_first_existing_json([os.path.join(ds_dir, f"REVERIE_{split}.json")])
        if raw is None:
            return []
        bbox_dir = os.path.join(ds_dir, "BBox")

        def _extras(item: dict, j: int) -> dict:
            scan = item["scan"]
            last_vp = item.get("path", [""])[-1] if item.get("path") else ""
            bbox_path = os.path.join(bbox_dir, f"{scan}_{last_vp}.json")
            instrs_l = item.get("instructions_l") or []
            return {
                "objId": item.get("objId"),
                "reverie_id": item.get("id"),
                "ix": item.get("ix"),
                "instructions_l": instrs_l[j] if j < len(instrs_l) else None,
                "bbox_path": bbox_path if os.path.isfile(bbox_path) else None,
            }

        records: list[dict] = []
        for item in raw:
            instructions = item.get("instructions") or [item.get("instruction", "")]
            records.extend(self._explode_instructions("REVERIE", item, instructions, _extras))
        return records

    # -- CVDN --
    def _load_cvdn(self, ds_dir: str, split: str) -> list[dict]:
        raw = _load_first_existing_json([os.path.join(ds_dir, f"{split}.json")])
        if raw is None:
            return []
        records: list[dict] = []
        for item in raw:
            scan = item["scan"]
            path = item.get("planner_nav_steps") or item.get("nav_steps") or []
            oracle_turns = [
                m.get("message", "")
                for m in (item.get("dialog_history") or [])
                if m.get("role") == "oracle"
            ]
            instruction = "\n".join(t for t in oracle_turns if t)
            # Heading lives inside nav_camera[0].message[0]
            heading = 0.0
            try:
                heading = float(item["nav_camera"][0]["message"][0]["heading"])
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            idx = item.get("idx", item.get("game_idx", len(records)))
            records.append(
                {
                    "instr_id": str(idx),
                    "path_id": idx,
                    "scan": scan,
                    "heading": heading,
                    "path": path,
                    "instruction": instruction,
                    "distance": self._path_distance(scan, path),
                    "dataset": "CVDN",
                    "extras": {
                        "dialog_history": item.get("dialog_history"),
                        "target": item.get("target"),
                        "start_pano": item.get("start_pano"),
                        "end_panos": item.get("end_panos"),
                        "nav_steps": item.get("nav_steps"),
                    },
                }
            )
        return records

    # -- NDH (CVDN-derived, structurally distinct) --
    def _load_ndh(self, ds_dir: str, split: str) -> list[dict]:
        raw = _load_first_existing_json([os.path.join(ds_dir, f"{split}.json")])
        if raw is None:
            return []
        records: list[dict] = []
        for item in raw:
            scan = item["scan"]
            path = item.get("planner_path") or item.get("player_path") or []
            oracle_turns = [
                m.get("message", "")
                for m in (item.get("dialog_history") or [])
                if m.get("role") == "oracle"
            ]
            instruction = "\n".join(t for t in oracle_turns if t)
            start_pano = item.get("start_pano") or {}
            heading = 0.0
            if isinstance(start_pano, dict):
                try:
                    heading = float(start_pano.get("heading", 0.0))
                except (TypeError, ValueError):
                    pass
            game_idx = item.get("game_idx", item.get("idx", len(records)))
            inst_idx = item.get("inst_idx", 0)
            records.append(
                {
                    "instr_id": f"{game_idx}_{inst_idx}",
                    "path_id": game_idx,
                    "scan": scan,
                    "heading": heading,
                    "path": path,
                    "instruction": instruction,
                    "distance": self._path_distance(scan, path),
                    "dataset": "NDH",
                    "extras": {
                        "dialog_history": item.get("dialog_history"),
                        "target": item.get("target"),
                        "start_pano": start_pano,
                        "end_panos": item.get("end_panos"),
                        "nav_history": item.get("nav_history"),
                        "inst_idx": inst_idx,
                        "R2R_success": item.get("R2R_success"),
                        "R2R_oracle_success": item.get("R2R_oracle_success"),
                    },
                }
            )
        return records

    def _start_first_episode_unlocked(self) -> None:
        """Start the first loaded episode or a random one (must hold _lock)."""
        episodes = self._episodes.get(self._dataset, {}).get(self._split, [])
        if episodes:
            ep = episodes[0]
            scan = ep["scan"]
            vp = ep["path"][0]
            heading = ep.get("heading", 0.0)
            self._sim.newEpisode([scan], [vp], [heading], [0.0])
            self._current_scan = scan
            self._current_viewpoint = vp
        else:
            # Gather available scans from connectivity directory
            scans = self._get_available_scans_unlocked()
            if scans:
                self._sim.newRandomEpisode([scans[0]])
                state = self._sim.getState()[0]
                self._current_scan = state.scanId
                self._current_viewpoint = state.location.viewpointId
            else:
                log.warning("No scans found — skipping initial episode start")
                return

        self._step_count = 0
        self._episode_done = False
        log.info(
            "Started episode: scan=%s viewpoint=%s",
            self._current_scan,
            self._current_viewpoint,
        )

    def _get_available_scans_unlocked(self) -> list[str]:
        """Return list of scan IDs from connectivity directory (must hold _lock)."""
        if not os.path.isdir(_CONNECTIVITY_DIR):
            return []
        return [
            os.path.basename(p).replace("_connectivity.json", "")
            for p in sorted(glob.glob(os.path.join(_CONNECTIVITY_DIR, "*_connectivity.json")))
        ]

    def _get_state_unlocked(self):
        """Return raw MatterSim state object (must hold _lock)."""
        if self._sim is None:
            return None
        return self._sim.getState()[0]

    def get_episode_info(self) -> dict:
        """Return current episode metadata including R2R instruction if loaded."""
        with self._lock:
            return self._get_episode_info_unlocked()

    def _get_episode_info_unlocked(self) -> dict:
        """Return episode metadata (must hold _lock).

        Prefers graph-mode state (``_graph_scan`` / ``_graph_episode``) when
        set — that's what the BaseEnvPanel's Play action writes via
        ``set_episode`` → ``graph_new_episode``. Legacy MatterSim-mode state
        (``_current_scan``, ``_current_split_idx``) is used only as a
        fallback for tools that haven't been migrated to graph mode.
        """
        info: dict[str, Any] = {
            "scan_id": self._current_scan,
            "viewpoint_id": self._current_viewpoint,
            "step_count": self._step_count,
            "done": self._episode_done,
        }

        if self._graph_scan and self._graph_episode:
            ep = self._graph_episode
            info["scan_id"] = self._graph_scan
            info["viewpoint_id"] = self._graph_viewpoint
            info["episode_id"] = str(ep.get("instr_id") or ep.get("path_id") or "")
            info["instruction"] = ep.get("instruction", "")
            info["path"] = ep.get("path", [])
            info["distance"] = ep.get("distance", 0.0)
            info["dataset"] = ep.get("dataset", self._dataset)
            info["extras"] = ep.get("extras", {})
            return info

        episodes = self._episodes.get(self._dataset, {}).get(self._split, [])
        if self._current_split_idx >= 0 and episodes:
            idx = self._current_split_idx
            ep = episodes[idx]
            info["episode_id"] = str(ep.get("instr_id") or ep.get("path_id") or idx)
            info["instruction"] = ep.get("instruction", "")
            info["path"] = ep.get("path", [])
            info["distance"] = ep.get("distance", 0.0)
            info["dataset"] = ep.get("dataset", self._dataset)
            info["extras"] = ep.get("extras", {})
        else:
            info["episode_id"] = ""
            info["instruction"] = ""
            info["dataset"] = self._dataset
            info["extras"] = {}

        return info

    def render_panorama_with_elevations(
        self,
        n_headings: int = 8,
        elevations: list[float] | None = None,
    ) -> dict:
        """Capture a panoramic sweep with multiple elevation levels (NavGPT style).

        Renders ``n_headings`` x ``len(elevations)`` views by rotating heading
        in equal steps and tilting camera to each elevation angle.  Default
        parameters reproduce the NavGPT paper: 8 headings x 3 elevations
        (-30°, 0°, +30°) = 24 views total.

        Args:
            n_headings:  Number of horizontal heading steps (default 8 → 45° apart).
            elevations:  Elevation angles in degrees (default [-30, 0, 30]).

        Caller must ensure sim is in a discretized pose before invocation.
        Restores heading/elevation on return.

        Returns:
            ``{"views": [...], "n_views": int}``. Each view dict carries
            ``rgb_base64`` (+ ``depth_base64`` when depth enabled),
            ``heading_deg``, ``elevation_deg``, ``direction`` and the native
            MatterSim ``view_index``. No composite is built.
        """
        if elevations is None:
            elevations = [-30, 0, 30]

        direction_names = {
            4: ["Front", "Right", "Back", "Left"],
            8: [
                "Front",
                "Front-Right",
                "Right",
                "Back-Right",
                "Back",
                "Back-Left",
                "Left",
                "Front-Left",
            ],
        }.get(n_headings, [f"{round(i * 360 / n_headings)}°" for i in range(n_headings)])

        elevation_suffixes = {-30: "(down)", 0: "(ahead)", 30: "(up)"}

        with self._lock:
            if self._sim is None:
                return {"error": "Simulator not initialized"}

            state = self._get_state_unlocked()
            initial_heading = float(state.heading)
            initial_elevation = float(state.elevation)

            views: list[dict] = []

            for elev_deg in elevations:
                elev_rad = math.radians(elev_deg)
                suffix = elevation_suffixes.get(elev_deg, f"({elev_deg:+d}°)")

                for i in range(n_headings):
                    angle_step = 2.0 * math.pi / n_headings
                    target_heading = i * angle_step

                    # Compute deltas from current state
                    current_state = self._sim.getState()[0]
                    delta_h = target_heading - float(current_state.heading)
                    delta_h = (delta_h + math.pi) % (2 * math.pi) - math.pi
                    delta_e = elev_rad - float(current_state.elevation)

                    self._sim.makeAction([0], [delta_h], [delta_e])

                    state = self._get_state_unlocked()
                    rgb = np.array(state.rgb, copy=False)
                    rgb = rgb[:, :, ::-1].copy()  # BGR → RGB

                    heading_deg = round(math.degrees(float(state.heading)) % 360, 1)
                    dir_name = direction_names[i] if i < len(direction_names) else f"{heading_deg}°"
                    label = f"{dir_name} {suffix}"

                    view_entry: dict[str, Any] = {
                        "direction": label,
                        "heading_deg": heading_deg,
                        "elevation_deg": round(math.degrees(float(state.elevation)), 1),
                        "view_index": state.viewIndex,
                        "rgb_base64": self.encode_rgb_base64(rgb),
                    }
                    if self._depth_enabled and state.depth is not None:
                        depth_raw = np.array(state.depth, copy=True)  # uint16 HxW
                        view_entry["depth_base64"] = self.encode_depth_base64(depth_raw)
                    views.append(view_entry)

            # Restore original heading and elevation
            current_state = self._sim.getState()[0]
            restore_h = initial_heading - float(current_state.heading)
            restore_h = (restore_h + math.pi) % (2 * math.pi) - math.pi
            restore_e = initial_elevation - float(current_state.elevation)
            self._sim.makeAction([0], [restore_h], [restore_e])

            # Per-view primitive only — no composite. Each view dict carries
            # rgb_base64 (+ depth_base64 when enabled), heading_deg,
            # elevation_deg and the native MatterSim view_index, which is all
            # downstream method nodes need to consume views directly.
            return {
                "views": views,
                "n_views": len(views),
            }

    def _get_status_unlocked(self) -> dict:
        """Return manager status dict (must hold _lock)."""
        episodes = self._episodes.get(self._dataset, {}).get(self._split, [])
        return {
            "initialized": self._sim is not None,
            "scan": self._current_scan,
            "viewpoint": self._current_viewpoint,
            "step_count": self._step_count,
            "done": self._episode_done,
            "episodes_loaded": len(episodes),
            "dataset": self._dataset,
            "split": self._split,
            "episode_counts": {
                d: {s: (c if c >= 0 else None) for s, c in splits.items()}
                for d, splits in self._episode_counts.items()
            },
        }

    def get_status(self) -> dict:
        """Return manager status dict (thread-safe public accessor, P3.3)."""
        with self._lock:
            return self._get_status_unlocked()

    # ── Env panel API (callable via run_in_executor) ──

    def list_datasets(self) -> list[str]:
        """Return alphabetical list of datasets with at least one split on disk."""
        with self._lock:
            return sorted(self._episode_counts.keys())

    def list_splits(self, dataset: str) -> list[str]:
        """Return sorted split names for *dataset*."""
        with self._lock:
            return sorted(self._episode_counts.get(dataset, {}).keys())

    def get_episode_count(self, dataset: str, split: str) -> int:
        """Return episode count for (dataset, split).

        Filename-based discovery populates ``_episode_counts`` with sentinel
        ``-1``; on first call for a given split we run the real loader and
        cache the resulting record count. Subsequent calls are O(1).
        """
        with self._lock:
            count = self._episode_counts.get(dataset, {}).get(split, 0)
            if count != -1:
                return count
            self._load_episodes(dataset, split)
            data = self._episodes.get(dataset, {}).get(split, [])
            count = len(data)
            self._episode_counts.setdefault(dataset, {})[split] = count
            return count

    def peek_episode(self, dataset: str, split: str, index: int) -> dict:
        """Return episode metadata at *index* for (dataset, split), loading if needed."""
        with self._lock:
            self._load_episodes(dataset, split)
            data = self._episodes.get(dataset, {}).get(split, [])
            if not data or index < 0 or index >= len(data):
                return {}
            ep = data[index]
            return {
                "instr_id": ep.get("instr_id", ""),
                "scan": ep.get("scan", ""),
                "path_id": ep.get("path_id", ""),
                "heading": ep.get("heading", 0.0),
                "instruction": ep.get("instruction", ""),
                "path_len": len(ep.get("path", [])),
                "dataset": ep.get("dataset", dataset),
                "extras": ep.get("extras", {}),
            }

    def list_episodes(self, dataset: str, split: str, limit: int = 2000) -> list[dict]:
        """Return lightweight episode list for (dataset, split), loading if needed."""
        with self._lock:
            self._load_episodes(dataset, split)
            data = self._episodes.get(dataset, {}).get(split, [])
            return [
                {"index": i, "instr_id": ep["instr_id"], "scan": ep["scan"]}
                for i, ep in enumerate(data[:limit])
            ]

    def set_episode(self, dataset: str, split: str, index: int) -> dict:
        """Wrap graph_new_episode for env panel use; returns ok/error dict."""
        try:
            result = self.graph_new_episode(dataset, split, index)
            return {"ok": True, "episode_info": result}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    def get_current_episode(self) -> dict:
        """Return current episode state for the env panel panel."""
        with self._lock:
            return {
                "dataset": self._dataset,
                "split": self._split,
                "index": self._current_split_idx,
                "episode_info": self._get_episode_info_unlocked(),
            }

    # ── Image encoding helpers ──

    @staticmethod
    def encode_rgb_base64(rgb: np.ndarray) -> str:
        """Encode an RGB uint8 HxWx3 array as a base64 PNG string."""
        img = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def encode_depth_base64(depth: np.ndarray) -> str:
        """Encode a MatterSim depth map (uint16, units of 0.25 mm) as PNG.

        Converts to meters (divide by 4000), then normalises to 0–255.
        Returns a grayscale PNG as a base64 ASCII string.
        """
        d = np.squeeze(depth).astype(np.float32)
        d_m = d / 4000.0  # uint16 → metres
        d_min, d_max = d_m.min(), d_m.max()
        if d_max - d_min > 1e-6:
            d_norm = ((d_m - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d_m, dtype=np.uint8)
        img = Image.fromarray(d_norm, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ── Graph Navigation (viewpoint-ID-based, additive to MatterSim) ──

    def graph_ensure_scan(self, scan: str) -> None:
        """Lazy-load connectivity graph + shortest paths for *scan*."""
        if scan in self._graphs:
            return
        G = _load_scan_graph(_CONNECTIVITY_DIR, scan)
        if G is None:
            raise ValueError(f"Connectivity not found for scan '{scan}'")
        self._graphs[scan] = G
        self._shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self._shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        log.info("Loaded graph for scan %s (%d nodes)", scan, G.number_of_nodes())

    def graph_load(self, dataset: str, split: str) -> None:
        """Ensure episodes for (dataset, split) are loaded. Idempotent."""
        with self._lock:
            self._load_episodes(dataset, split)

    def graph_new_episode(
        self,
        dataset: str = "R2R",
        split: str = "val_unseen",
        index: int | None = None,
    ) -> dict:
        """Start a new episode on the viewpoint graph.

        Args:
            dataset: Dataset name (R2R / R4R / RxR / REVERIE / CVDN / NDH).
            split:   Split name within the dataset. RxR splits carry a
                     language suffix (val_unseen_en / _hi / _te).
            index:   Episode index (None = random).

        Returns:
            Dict with instruction, scan_id, start_viewpoint, episode_id,
            ground_truth_path, dataset, split, extras.
        """
        with self._lock:
            self._load_episodes(dataset, split)
            data = self._episodes.get(dataset, {}).get(split, [])
            if not data:
                raise ValueError(
                    f"No episodes for {dataset}/{split}. Available: "
                    f"{ {d: list(s.keys()) for d, s in self._episode_counts.items()} }"
                )
            idx = index % len(data) if index is not None else random.randint(0, len(data) - 1)
            ep = data[idx]

        # graph_ensure_scan acquires connectivity; called outside lock to
        # avoid nested locking during first-time Dijkstra builds.
        self.graph_ensure_scan(ep["scan"])

        with self._lock:
            self._dataset = dataset
            self._split = split
            self._current_split_idx = idx
            self._graph_scan = ep["scan"]
            self._graph_viewpoint = ep["path"][0]
            self._graph_heading = float(ep.get("heading", 0.0))
            self._graph_elevation = 0.0
            self._graph_trajectory = [self._graph_viewpoint]
            self._graph_episode = ep
            self._graph_history = []

        return {
            "instruction": ep.get("instruction", ""),
            "instr_id": ep.get("instr_id", ""),
            "scan_id": ep["scan"],
            "start_viewpoint": ep["path"][0],
            "episode_index": idx,
            "ground_truth_path": json.dumps(ep.get("path", [])),
            "dataset": ep.get("dataset", dataset),
            "split": split,
            "extras": ep.get("extras", {}),
        }

    def graph_get_observation(self) -> dict:
        """Get structured env observation at the current graph viewpoint.

        Returns raw env data only — no vision, no prose, no compass
        formatting. Online vision (BLIP-2 / Faster R-CNN) is wired
        directly to format/init/scratchpad consumers at the graph layer;
        pre-computed scene files are exposed here as a fallback for
        offline-mode graphs.
        """
        if not self._graph_scan:
            return {"error": "No active graph episode"}
        G = self._graphs.get(self._graph_scan)
        if G is None:
            return {"error": f"Graph not loaded for scan {self._graph_scan}"}
        nav = _compute_navigable(G, self._graph_viewpoint)

        obs_data = _load_observation_data(self._graph_scan, self._graph_viewpoint) or {}
        scene_descs = obs_data.get("detail")
        scene_summary = obs_data.get("summary") or ""
        objects = obs_data.get("objects")

        pos = G.nodes[self._graph_viewpoint]["position"]
        position_xyz = [float(pos[0]), float(pos[1]), float(pos[2])]

        return {
            "viewpoint_id": self._graph_viewpoint,
            "scan_id": self._graph_scan,
            "position": position_xyz,  # current agent [x,y,z] in MP3D sim coords
            "heading_rad": self._graph_heading,
            "elevation_rad": self._graph_elevation,
            "heading_deg": math.degrees(self._graph_heading),
            "elevation_deg": math.degrees(self._graph_elevation),
            "navigable": nav,  # dict[vp_id, {heading, elevation, distance, position}] — radians
            "scene_descriptions": scene_descs,  # list[str] | None — pre-computed fallback
            "scene_objects": objects,  # list[dict] | None — pre-computed fallback
            "scene_summary": scene_summary,
        }

    def graph_navigate(self, target: str) -> dict:
        """Navigate to a viewpoint by ID on the graph.

        Supports adjacent (direct step) and non-adjacent (shortest-path walk)
        navigation.  When target is "STOP", signals episode done without moving.

        Returns dict with new_viewpoint, turned_angle, success, and optionally done.
        """
        if not self._graph_scan:
            return {"success": False, "error": "No active graph episode"}
        G = self._graphs.get(self._graph_scan)
        if G is None:
            return {"success": False, "error": f"Graph missing for {self._graph_scan}"}

        # STOP detection
        if target.strip().upper() in ("STOP", "FINISHED", "FINISHED!"):
            return {
                "success": True,
                "new_viewpoint": self._graph_viewpoint,
                "turned_angle": "0.0",
                "done": True,
            }

        # Validate target exists in graph — include navigable IDs for error feedback
        if target not in G:
            nav = _compute_navigable(G, self._graph_viewpoint)
            nav_ids = list(nav.keys())
            return {
                "success": False,
                "error": (
                    f"ViewpointID '{target}' is not valid, agent not moved. "
                    f"DO NOT fabricate nonexistent IDs. "
                    f"The navigable viewpoints you can choose from current "
                    f"viewpoints are: {nav_ids}."
                ),
                "new_viewpoint": self._graph_viewpoint,
                "turned_angle": "0",
            }

        nav = _compute_navigable(G, self._graph_viewpoint)

        heading_before = self._graph_heading

        if target in nav:
            # Adjacent — direct step
            self._graph_step_to(target, nav[target])
        else:
            # Non-adjacent — walk shortest path
            try:
                path = self._shortest_paths[self._graph_scan][self._graph_viewpoint][target]
            except KeyError:
                return {
                    "success": False,
                    "error": f"No path from '{self._graph_viewpoint}' to '{target}'",
                    "trajectory": list(self._graph_trajectory),
                }
            # Walk each intermediate step (preserves trajectory for metrics)
            mid_path_break = False
            for waypoint in path[1:]:
                wp_nav = _compute_navigable(G, self._graph_viewpoint)
                if waypoint in wp_nav:
                    self._graph_step_to(waypoint, wp_nav[waypoint])
                else:
                    log.warning(
                        "Shortest-path waypoint %s not navigable from %s",
                        waypoint,
                        self._graph_viewpoint,
                    )
                    mid_path_break = True
                    break

            if mid_path_break:
                # Compute total heading change up to the point we stopped
                total_turned = math.degrees(self._graph_heading - heading_before)
                while total_turned > 180:
                    total_turned -= 360
                while total_turned <= -180:
                    total_turned += 360
                return {
                    "success": False,
                    "reason": "waypoint_unreachable",
                    "reached_viewpoint": self._graph_viewpoint,
                    "target_viewpoint": target,
                    "turned_angle": str(round(total_turned, 1)),
                    "trajectory": list(self._graph_trajectory),
                }

        # Compute total heading change from before navigation to after
        total_turned = math.degrees(self._graph_heading - heading_before)
        while total_turned > 180:
            total_turned -= 360
        while total_turned <= -180:
            total_turned += 360

        return {
            "success": True,
            "reached_viewpoint": target,
            "new_viewpoint": self._graph_viewpoint,
            "turned_angle": str(round(total_turned, 1)),
            "trajectory": list(self._graph_trajectory),
        }

    def _graph_step_to(self, target: str, info: dict) -> None:
        """Internal: move to an adjacent viewpoint, updating heading/trajectory.

        Records a structured step entry. Narrative prose is built by the
        agent-side history formatter, not here.
        """
        old_heading = self._graph_heading
        self._graph_heading = info["heading"]
        self._graph_elevation = info["elevation"]
        self._graph_viewpoint = target
        self._graph_trajectory.append(target)

        turned = self._graph_heading - old_heading
        while turned > math.pi:
            turned -= 2 * math.pi
        while turned <= -math.pi:
            turned += 2 * math.pi

        self._graph_history.append(
            {
                "step": len(self._graph_history) + 1,
                "from_viewpoint": self._graph_trajectory[-2]
                if len(self._graph_trajectory) >= 2
                else None,
                "to_viewpoint": target,
                "from_heading_rad": old_heading,
                "to_heading_rad": self._graph_heading,
                "turned_rad": turned,
                "elevation_rad": self._graph_elevation,
            }
        )

    def graph_evaluate(self) -> dict:
        """Compute R2R evaluation metrics for the current graph episode."""
        if not self._graph_episode:
            return {"error": "No active graph episode"}
        gt = self._graph_episode["path"]
        scan = self._graph_scan
        if scan not in self._shortest_distances:
            return {"error": f"Distances not loaded for {scan}"}
        return _eval_trajectory(self._shortest_distances[scan], self._graph_trajectory, gt)

    def graph_render_panorama(self, n_headings: int = 12) -> dict:
        """Render a panorama at the current graph viewpoint via MatterSim.

        Temporarily positions MatterSim at ``_graph_viewpoint`` with
        ``_graph_heading``, then renders ``n_headings`` x 3 elevation views
        ([-30, 0, 30]) and returns the per-view primitive (no composite).

        Requires MatterSim to be initialized (call ``initialize()`` first).
        Always elevation-capable so candidate views carry the correct
        elevation row for downstream perception / candidate→view mapping.
        """
        with self._lock:
            if self._sim is None:
                return {"error": "MatterSim not initialized — call initialize() first"}
            if not self._graph_scan or not self._graph_viewpoint:
                return {"error": "No active graph episode"}

            # Position MatterSim at the current graph viewpoint
            try:
                self._sim.newEpisode(
                    [self._graph_scan],
                    [self._graph_viewpoint],
                    [self._graph_heading],
                    [self._graph_elevation],
                )
            except (ValueError, OSError, RuntimeError) as exc:
                log.warning("graph_render_panorama: MatterSim positioning failed: %s", exc)
                return {"error": f"MatterSim render failed: {exc}"}

        # Always elevation-capable (acquires lock internally).
        try:
            return self.render_panorama_with_elevations(
                n_headings=n_headings, elevations=[-30, 0, 30]
            )
        except Exception as exc:
            log.warning("graph_render_panorama: render failed: %s", exc)
            return {"error": f"Panorama render failed: {exc}"}


# ══════════════════════════════════════════════════════════════════════
# Graph Navigation Utilities (from NavGPT — viewpoint graph infrastructure)
# ══════════════════════════════════════════════════════════════════════

ERROR_MARGIN = 3.0  # metres — standard R2R success threshold


def _load_scan_graph(connectivity_dir: str, scan: str) -> nx.Graph | None:
    """Load a single scan's connectivity graph into a NetworkX Graph."""
    path = os.path.join(connectivity_dir, f"{scan}_connectivity.json")
    if not os.path.exists(path):
        return None

    def _dist(a: dict, b: dict) -> float:
        return math.sqrt(
            (a["pose"][3] - b["pose"][3]) ** 2
            + (a["pose"][7] - b["pose"][7]) ** 2
            + (a["pose"][11] - b["pose"][11]) ** 2
        )

    with open(path) as f:
        data = json.load(f)

    G = nx.Graph()
    for item in data:
        if item["included"]:
            pos = np.array([item["pose"][3], item["pose"][7], item["pose"][11]])
            G.add_node(item["image_id"], position=pos)
    for item in data:
        if not item["included"]:
            continue
        for j, conn in enumerate(item["unobstructed"]):
            if conn and data[j]["included"]:
                G.add_edge(item["image_id"], data[j]["image_id"], weight=_dist(item, data[j]))
    return G


def _compute_navigable(
    graph: nx.Graph,
    viewpoint_id: str,
) -> dict:
    """Compute navigable viewpoints with heading / elevation / distance."""
    if viewpoint_id not in graph:
        return {}
    pos = graph.nodes[viewpoint_id]["position"]
    navigable: dict = {}
    for neighbor in graph.neighbors(viewpoint_id):
        n_pos = graph.nodes[neighbor]["position"]
        dx = n_pos[0] - pos[0]
        dy = n_pos[1] - pos[1]
        dz = n_pos[2] - pos[2]
        xy_dist = max(math.sqrt(dx**2 + dy**2), 1e-8)
        xyz_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        heading = math.atan2(dx, dy)
        elevation = math.atan2(dz, xy_dist)
        navigable[neighbor] = {
            "heading": heading,
            "elevation": elevation,
            "distance": xyz_dist,
            "position": [float(n_pos[0]), float(n_pos[1]), float(n_pos[2])],
        }
    return navigable


# ── NavGPT observation data (pre-computed scene descriptions) ──

# Configurable root for pre-computed NavGPT observation data.
# Default layout produced by workspace/nodesets/_upstream/navgpt/fetch_data.sh.
_NAVGPT_OBS_ROOT = os.environ.get(
    "NAVGPT_OBS_DIR",
    os.path.join(_REPO_ROOT, "data", "navgpt", "r2r"),
)


@functools.lru_cache(maxsize=4096)
def _load_observation_data(scan: str, viewpoint: str) -> dict | None:
    """Try to load pre-computed NavGPT observation data for a viewpoint.

    Looks for three JSON sources matching the original NavGPT data layout:
      - observations_list_summarized/{scan}.json  → detail (8-direction descriptions)
      - observations_summarized/{scan}_summarized.json → summary (1-sentence)
      - objects_list/{scan}.json → objects per direction

    Returns dict with keys ``detail``, ``summary``, ``objects`` or ``None``
    when data files are not available.  Results are cached via LRU (maxsize=4096).
    """
    obs_dir = os.path.join(_NAVGPT_OBS_ROOT, "observations_list_summarized")
    sum_dir = os.path.join(_NAVGPT_OBS_ROOT, "observations_summarized")
    obj_dir = os.path.join(_NAVGPT_OBS_ROOT, "objects_list")

    detail_path = os.path.join(obs_dir, f"{scan}.json")
    if not os.path.isfile(detail_path):
        return None

    try:
        with open(detail_path) as f:
            detail = json.load(f).get(viewpoint)
        summary_path = os.path.join(sum_dir, f"{scan}_summarized.json")
        summary = None
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f).get(viewpoint)
        objects = None
        obj_path = os.path.join(obj_dir, f"{scan}.json")
        if os.path.isfile(obj_path):
            with open(obj_path) as f:
                objects = json.load(f).get(viewpoint)

        return {"detail": detail, "summary": summary, "objects": objects}
    except Exception:
        return None


# ── R2R Evaluation Metrics ──


def _cal_dtw(
    sd: dict,
    prediction: list,
    reference: list,
    threshold: float = ERROR_MARGIN,
) -> dict:
    n, m = len(prediction), len(reference)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = sd[prediction[i - 1]][reference[j - 1]]
            dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
    dtw_val = float(dtw[n][m])
    ndtw = math.exp(-dtw_val / (threshold * m)) if m > 0 else 0.0
    success = float(sd[prediction[-1]][reference[-1]] < threshold)
    return {"nDTW": ndtw, "SDTW": success * ndtw}


def _cal_cls(
    sd: dict,
    prediction: list,
    reference: list,
    threshold: float = ERROR_MARGIN,
) -> float:
    def _length(nodes: list) -> float:
        return sum(sd[a][b] for a, b in zip(nodes[:-1], nodes[1:]))

    if len(prediction) < 1 or len(reference) < 1:
        return 0.0
    coverage = float(
        np.mean([math.exp(-min(sd[u][v] for v in prediction) / threshold) for u in reference])
    )
    expected = coverage * _length(reference)
    if expected == 0:
        return 0.0
    return coverage * expected / (expected + abs(expected - _length(prediction)))


def _eval_trajectory(sd: dict, pred: list, gt: list) -> dict:
    """Full R2R evaluation metrics for one trajectory."""
    if not pred or not gt:
        return {"error": "empty trajectory or ground truth"}
    scores: dict = {}
    scores["nav_error"] = sd[pred[-1]][gt[-1]]
    scores["oracle_error"] = min(sd[p][gt[-1]] for p in pred)
    scores["trajectory_steps"] = float(len(pred) - 1)
    scores["trajectory_length"] = sum(sd[a][b] for a, b in zip(pred[:-1], pred[1:]))
    gt_len = sum(sd[a][b] for a, b in zip(gt[:-1], gt[1:]))
    scores["success"] = float(scores["nav_error"] < ERROR_MARGIN)
    scores["spl"] = scores["success"] * gt_len / max(scores["trajectory_length"], gt_len, 0.01)
    scores["oracle_success"] = float(scores["oracle_error"] < ERROR_MARGIN)
    scores.update(_cal_dtw(sd, pred, gt))
    scores["CLS"] = _cal_cls(sd, pred, gt)
    return scores


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers (mirrors habitat.py pattern)
# ══════════════════════════════════════════════════════════════════════


def _get_env() -> MP3DEnvManager:
    return MP3DEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    """Run a blocking MP3DEnvManager method on its dedicated thread."""
    env_mgr = _get_env()
    return await asyncio.get_running_loop().run_in_executor(env_mgr.executor, fn, *args)


def _resolve_node_config(node: BaseCanvasNode, ctx: Any) -> dict:
    """Merge class-level self.config with per-instance ctx.node_config (P3.4)."""
    base = getattr(node, "config", None) or {}
    override = getattr(ctx, "node_config", None) or {}
    return {**base, **override}


# ══════════════════════════════════════════════════════════════════════
# MP3D Canvas Tools
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Graph Navigation Canvas Tools (viewpoint-ID-based, additive)
# ══════════════════════════════════════════════════════════════════════


class MP3DStepWaypointTool(BaseCanvasNode):
    """Navigate to a viewpoint by its ID on the MP3D connectivity graph.

    Pure environment node — receives a clean viewpoint ID (already parsed
    by an upstream method node like ``navgpt_mp3d_tools__parse_action``)
    and performs graph navigation.

    Adjacent targets: direct single step.
    Non-adjacent targets: walks the shortest path, recording all
    intermediate viewpoints in the trajectory.

    Writes ``trajectory`` to graph_state for evaluation.  All agent-specific
    logic (LLM parsing, scratchpad, history) lives in method nodes.

    The ``done`` key is included in the output ONLY when the input is
    "STOP" — this prevents the executor's ``done``-key scan from
    triggering premature termination.
    """

    node_type = "env_mp3d__step_waypoint"
    display_name = "MP3D: Step (waypoint)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "Navigate to a viewpoint ID (or STOP) on the MP3D graph — returns control signals only"
    )
    category = "environment"
    icon = "Navigation"
    input_ports = [
        PortDef("viewpoint_id", "TEXT", "Target viewpoint ID or STOP"),
    ]
    output_ports = [
        # gym-like contract
        PortDef("reward", "ANY", "Per-step reward (scalar; 0 — VLN sparse)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called"),
        PortDef("truncated", "BOOL", "Step-budget cutoff (graph step_budget enforces)"),
        PortDef("info", "ANY", "Nav result: {new_viewpoint, turned_angle, success, error}"),
        # mp3d-specific nav-result extras (also inside info) — methods may wire these directly
        PortDef("new_viewpoint", "TEXT", "Viewpoint after navigation"),
        PortDef("turned_angle", "TEXT", "Degrees turned"),
        PortDef("success", "TEXT", "Whether navigation succeeded"),
        PortDef("error", "TEXT", "Error message on failure (empty on success)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        vp_id = str(inputs.get("viewpoint_id", "")).strip()
        if not vp_id:
            return {
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {
                    "success": False,
                    "new_viewpoint": "",
                    "turned_angle": "0",
                    "error": "Empty viewpoint ID",
                },
                "new_viewpoint": "",
                "turned_angle": "0",
                "success": False,
                "error": "Empty viewpoint ID",
            }

        mgr = _get_env()

        result = await asyncio.get_running_loop().run_in_executor(
            mgr.executor,
            mgr.graph_navigate,
            vp_id,
        )

        # Trajectory accumulation (environment responsibility — needed for evaluation)
        gs = getattr(ctx, "graph_state", None)
        if gs and result.get("success") is True and result.get("done") is not True:
            with contextlib.suppress(Exception):
                gs.write("trajectory", result.get("trajectory", []))

        terminated = bool(result.get("done") is True)  # done is only True on STOP
        new_vp = result.get("new_viewpoint") or result.get("reached_viewpoint", "")
        self._self_log("target", vp_id)
        self._self_log("success", result.get("success"))
        self._self_log("new_viewpoint", new_vp)
        self._self_log("terminated", terminated)

        info = {
            "new_viewpoint": new_vp,
            "turned_angle": result.get("turned_angle", "0"),
            "success": result.get("success", False),
            "error": result.get("error", ""),
        }
        return {
            "reward": 0.0,
            "terminated": terminated,
            "truncated": False,
            "info": info,
            "new_viewpoint": new_vp,
            "turned_angle": result.get("turned_angle", "0"),
            "success": result.get("success", False),
            "error": result.get("error", ""),
        }


class MP3DObservationTool(BaseCanvasNode):
    """Return structured observation at current graph viewpoint.

    Env-contract node: returns raw env data only — no prose, no compass
    formatting, no vision data. Online vision (BLIP-2 / Faster R-CNN) and
    offline pre-computed scene descriptions are wired directly into
    ``NavGPTObservationFormatNode`` via explicit ports; this node stays
    pure env access.
    """

    node_type = "env_mp3d__observe_navigable"
    display_name = "MP3D: Observe (navigable)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = "Pull structured observation at current graph viewpoint: viewpoint, heading, navigable neighbours"
    category = "environment"
    icon = "Eye"
    input_ports = [
        PortDef("trigger", "TEXT", "Trigger (any value)", optional=True),
    ]
    output_ports = [
        PortDef("viewpoint_id", "TEXT", "Current viewpoint ID"),
        PortDef("scan_id", "TEXT", "Current scan ID"),
        PortDef("heading", "TEXT", "Current heading in degrees (string)"),
        PortDef("navigable_json", "TEXT", "Navigable viewpoints as JSON (radian values)"),
        PortDef("position_json", "TEXT", "Agent 3D position as JSON [x,y,z] (MP3D sim coords)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_env()

        result = await asyncio.get_running_loop().run_in_executor(
            mgr.executor,
            mgr.graph_get_observation,
        )

        nav = result.get("navigable") or {}
        navigable_count = len(nav)
        position = result.get("position") or [0.0, 0.0, 0.0]
        self._self_log("viewpoint_id", result.get("viewpoint_id"))
        self._self_log("heading", result.get("heading_deg"))
        self._self_log("scan_id", result.get("scan_id"))
        self._self_log("navigable_count", navigable_count)
        self._self_log("position", position)

        def _round_nav_value(vv: Any) -> Any:
            # Preserve list-typed fields (e.g. neighbor position [x,y,z])
            # while rounding scalar heading/elevation/distance.
            if isinstance(vv, (list, tuple)):
                return [round(float(x), 6) for x in vv]
            return round(float(vv), 6)

        return {
            "viewpoint_id": result.get("viewpoint_id", ""),
            "scan_id": result.get("scan_id", ""),
            "heading": str(round(result.get("heading_deg", 0.0), 1)),
            "navigable_json": json.dumps(
                {k: {kk: _round_nav_value(vv) for kk, vv in v.items()} for k, v in nav.items()}
            ),
            "position_json": json.dumps([round(float(c), 6) for c in position]),
        }


class MP3DEvaluateTool(BaseCanvasNode):
    """Compute R2R evaluation metrics for the current graph episode.

    Reads the trajectory accumulated during navigation and compares it
    against the ground-truth path using graph shortest-path distances.

    Metrics: SR, SPL, nDTW, SDTW, CLS, nav_error, oracle_success.
    """

    node_type = "env_mp3d__evaluate"
    display_name = "MP3D: Evaluate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    description = "Compute R2R metrics (SR, SPL, nDTW, SDTW) for current episode"
    category = "evaluation"
    icon = "BarChart"
    input_ports = [
        PortDef("trigger", "TEXT", "Trigger evaluation (any value)", optional=True),
    ]
    output_ports = [
        PortDef("metrics", "TEXT", "Full metrics as JSON"),
        PortDef("success", "TEXT", "Whether agent reached goal (SR)"),
        PortDef("spl", "TEXT", "Success weighted by Path Length"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_env()
        result = await asyncio.get_running_loop().run_in_executor(
            mgr.executor,
            mgr.graph_evaluate,
        )

        if "error" in result:
            self._self_log("error", result["error"])
            return {"metrics": json.dumps(result), "success": "0", "spl": "0"}

        self._self_log("success", result.get("success"))
        self._self_log("spl", f"{result.get('spl', 0):.3f}")
        self._self_log("nav_error", f"{result.get('nav_error', 0):.2f}m")
        self._self_log("nDTW", f"{result.get('nDTW', 0):.3f}")

        # Persist the parsed metrics dict to node state so BatchEvalRunner's
        # _collect_metrics can harvest it. Output ports stay stringified for
        # the TEXT wire contract the viewer/sink expects.
        ctx.metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}

        return {
            "metrics": json.dumps(result, indent=2),
            "success": str(result.get("success", 0)),
            "spl": str(round(result.get("spl", 0), 4)),
        }


class MP3DGraphPanoramaTool(BaseCanvasNode):
    """Render a panorama at the current graph viewpoint via MatterSim.

    Bridges graph-mode navigation with MatterSim rendering.  Positions
    MatterSim at the current ``_graph_viewpoint`` / ``_graph_heading``,
    renders ``n_headings`` x 3 elevation views and emits the per-view
    primitive (``views`` + ``view_meta`` + ``depth_views``) — ready for
    BLIP-2 captioning and Faster R-CNN object detection, which consume
    the per-view images directly (no composite to reverse-engineer).

    Requires both graph mode (for navigation) AND MatterSim initialized
    (for rendering).  Use ``initialize()`` on the env_mp3d nodeset first.
    """

    node_type = "env_mp3d__observe_panorama"
    display_name = "MP3D: Observe (panorama)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="teal",
        config_fields=[
            ConfigField(
                "n_headings",
                "select",
                label="Headings",
                default=12,
                options=[
                    {"value": 8, "label": "8 headings x 3 elev (24 views)"},
                    {"value": 12, "label": "12 headings x 3 elev (36 views)"},
                ],
            ),
        ],
    )
    description = (
        "Render per-view panorama primitive at current graph viewpoint (requires MatterSim)"
    )
    category = "environment"
    icon = "Scan"
    input_ports = [
        PortDef("trigger", "TEXT", "Trigger (any value fires rendering)", optional=True),
    ]
    output_ports = [
        PortDef("views", "LIST[IMAGE]", "Per-view RGB images (one per rendered view)"),
        PortDef(
            "view_meta",
            "TEXT",
            "JSON list aligned 1:1 with views: [{view_index, heading_deg, elevation_deg, direction}]",
        ),
        PortDef(
            "depth_views",
            "LIST[DEPTH]",
            "Per-view depth (float32 metres), aligned with views; empty if depth disabled",
            optional=True,
        ),
        PortDef("scan_id", "TEXT", "Current scan ID (for downstream cache lookups)", optional=True),
        PortDef(
            "viewpoint_id",
            "TEXT",
            "Current viewpoint ID (for downstream cache lookups)",
            optional=True,
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        n_headings = int(_resolve_node_config(self, ctx).get("n_headings", 12))

        mgr = _get_env()
        result = await asyncio.get_running_loop().run_in_executor(
            mgr.executor,
            mgr.graph_render_panorama,
            n_headings,
        )

        if "error" in result:
            self._self_log("error", result["error"])
            return {
                "views": [],
                "view_meta": "[]",
                "depth_views": [],
                "scan_id": "",
                "viewpoint_id": "",
            }

        rgb_list, depth_list, view_meta = _pack_views(result.get("views", []))

        self._self_log("n_views", result.get("n_views", 0))
        self._self_log("viewpoint", mgr.current_graph_viewpoint)
        self._self_log("scan", mgr._graph_scan)
        self._self_log("views_count", len(rgb_list))
        self._self_log("depth_views_count", len(depth_list))

        return {
            "views": rgb_list,
            "view_meta": view_meta,
            "depth_views": depth_list,
            "scan_id": str(mgr._graph_scan or ""),
            "viewpoint_id": str(mgr.current_graph_viewpoint or ""),
        }


# ══════════════════════════════════════════════════════════════════════
# MP3DEnvPanel — canvas panel env panel for R2R episode selection
# ══════════════════════════════════════════════════════════════════════


class MP3DEnvPanel(BaseEnvPanel):
    """Canvas panel env panel for the Matterport3D environment.

    Exposes a three-field cascade: ``dataset → split → episode_index``.
    Changing dataset resets split + episode; changing split resets
    episode. All three emit an ``episode_reset`` signal so any
    ``lifetime="episode"`` state container clears downstream.
    """

    name = "env_mp3d"
    display_name = "Matterport3D"
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
            "dataset": "R2R",
            "split": "val_unseen",
            "episode_index": 0,
        }

    def _mgr(self) -> MP3DEnvManager:
        return _get_env()

    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mgr()._executor, fn, *args)

    async def on_load(self) -> dict[str, Any]:
        ctx = getattr(self, "_context", {}) or {}
        if ctx.get("mode") == "server":
            return {
                "available": False,
                "dataset": "",
                "split": "",
                "episode_index": 0,
                "episode_count": 0,
                "datasets": [],
                "splits": [],
                "message": (
                    "Matterport3D is running in server mode (subprocess). Episode "
                    "control from this panel is not yet supported."
                ),
            }
        mgr = self._mgr()
        datasets = await self._run(mgr.list_datasets)
        if not datasets:
            return {
                "available": False,
                "dataset": "",
                "split": "",
                "episode_index": 0,
                "episode_count": 0,
                "datasets": [],
                "splits": [],
                "message": (
                    "Matterport3D environment not initialized. Load env_mp3d from "
                    "the NodeSet Manager to enable episode control."
                ),
            }

        dataset = self._state["dataset"]
        if dataset not in datasets:
            dataset = datasets[0]
            self._state["dataset"] = dataset

        splits = await self._run(mgr.list_splits, dataset)
        split = self._state["split"]
        if split not in splits:
            split = splits[0] if splits else ""
            self._state["split"] = split

        index = int(self._state["episode_index"])
        episode_count = await self._run(mgr.get_episode_count, dataset, split) if split else 0
        current_episode = await self._run(mgr.peek_episode, dataset, split, index) if split else {}
        return {
            "available": True,
            "dataset": dataset,
            "split": split,
            "episode_index": index,
            "episode_count": episode_count,
            "datasets": datasets,
            "splits": splits,
            "current_episode": current_episode,
            "step_budget": 30,
        }

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "dataset": self._state.get("dataset", ""),
            "split": self._state.get("split", ""),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        if name == "dataset":
            self._state["dataset"] = str(value)
            self._state["split"] = ""  # refilled by on_load
            self._state["episode_index"] = 0
        elif name == "split":
            self._state["split"] = str(value)
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                self._state["episode_index"] = int(value)
            except (TypeError, ValueError):
                self._state["episode_index"] = 0
        else:
            self._state[name] = value
            return await self.on_load()

        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = self._episode_reset_payload()
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name in ("play", "reset"):
            mgr = self._mgr()
            result = await self._run(
                mgr.set_episode,
                self._state["dataset"],
                self._state["split"],
                int(self._state["episode_index"]),
            )
            if result["ok"]:
                if name == "play":
                    return {"ok": True, "side_effect": "run_start"}
                return {
                    "ok": True,
                    "side_effect": "signal",
                    "signal_name": "episode_reset",
                    "signal_payload": self._episode_reset_payload(),
                }
            return {"ok": False, "side_effect": "none", "error": result.get("error")}
        if name in ("pause", "stop"):
            return {"ok": True, "side_effect": f"run_{name}"}
        return {"ok": False, "side_effect": "none", "error": f"Unknown action '{name}'"}

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        mgr = self._mgr()
        if field == "dataset":
            datasets = await self._run(mgr.list_datasets)
            return [{"value": d, "label": d} for d in datasets]
        if field == "split":
            splits = await self._run(mgr.list_splits, self._state["dataset"])
            return [{"value": s, "label": s} for s in splits]
        if field == "episode_index":
            episodes = await self._run(
                mgr.list_episodes,
                self._state["dataset"],
                self._state["split"],
            )
            return [
                {
                    "value": e["index"],
                    "label": f"{e['index']}: {e['scan']} ({e['instr_id']})",
                }
                for e in episodes
            ]
        return []


# ══════════════════════════════════════════════════════════════════════
# Gym-style facade — reset / step
#
# Bundle (episode_info + graph_panorama + observation) into one ``reset``
# and (graph_navigate + graph_panorama + observation) into one ``step``.
# Mirrors the Habitat / VLN-CE convention where an env exposes just two
# user-facing methods; agents that want the underlying primitives still
# have ``env_mp3d__graph_panorama``, ``env_mp3d__observation``, etc.
# ══════════════════════════════════════════════════════════════════════


_RESET_STEP_N_HEADINGS = [
    {"value": 8, "label": "8 headings x 3 elev (24 views)"},
    {"value": 12, "label": "12 headings x 3 elev (36 views)"},
]


def _decode_view_rgb(b64: str | None) -> np.ndarray | None:
    """Decode a single per-view RGB PNG (base64) to a uint8 HxWx3 array."""
    if not b64:
        return None
    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _decode_view_depth(b64: str | None) -> np.ndarray | None:
    """Decode a single per-view uint16-grayscale PNG to float32 metres."""
    if not b64:
        return None
    raw = base64.b64decode(b64)
    u16 = np.array(Image.open(io.BytesIO(raw)), dtype=np.uint16)
    return u16.astype(np.float32) / 4000.0


def _pack_views(views: list[dict]) -> tuple[list, list, str]:
    """Decode the manager's render output into the per-view wire contract.

    Returns ``(rgb_list, depth_list, view_meta_json)`` where:
      - ``rgb_list``    — list[np.ndarray] per-view RGB (LIST[IMAGE])
      - ``depth_list``  — list[np.ndarray] per-view depth metres (LIST[DEPTH]);
                          empty when depth is disabled / absent
      - ``view_meta``   — JSON list aligned 1:1 with ``rgb_list``:
                          ``[{view_index, heading_deg, elevation_deg, direction}, …]``
    """
    rgb_list: list = []
    depth_list: list = []
    meta: list[dict] = []
    for v in views:
        rgb = _decode_view_rgb(v.get("rgb_base64"))
        if rgb is None:
            continue
        rgb_list.append(rgb)
        meta.append(
            {
                "view_index": v.get("view_index", 0),
                "heading_deg": v.get("heading_deg", 0.0),
                "elevation_deg": v.get("elevation_deg", 0.0),
                "direction": v.get("direction", ""),
            }
        )
        depth = _decode_view_depth(v.get("depth_base64"))
        if depth is not None:
            depth_list.append(depth)
    return rgb_list, depth_list, json.dumps(meta)


def _pack_navigable(nav: dict, view_meta_json: str | None = None) -> str:
    """Serialize the graph navigable dict to JSON.

    When ``view_meta_json`` is supplied (the per-view metadata emitted
    alongside the render), each candidate gains a ``view_index`` pointing at
    the rendered view nearest its (heading, elevation) — this is what lets a
    method map candidate → image by exact index instead of cropping a grid.
    """

    def _pack_value(vv):
        if isinstance(vv, (list, tuple)):
            return [round(float(x), 6) for x in vv]
        return round(float(vv), 6)

    meta: list[dict] = []
    if view_meta_json:
        with contextlib.suppress(Exception):
            parsed = json.loads(view_meta_json)
            if isinstance(parsed, list):
                meta = [m for m in parsed if isinstance(m, dict)]

    def _nearest_view_index(heading_rad: float, elevation_rad: float):
        if not meta:
            return None
        target_h = math.degrees(heading_rad) % 360.0
        target_e = math.degrees(elevation_rad)

        def _hdist(a: float, b: float) -> float:
            d = abs((a - b) % 360.0)
            return min(d, 360.0 - d)

        best = min(
            meta,
            key=lambda m: (
                _hdist(float(m.get("heading_deg", 0.0)), target_h)
                + abs(float(m.get("elevation_deg", 0.0)) - target_e)
            ),
        )
        return best.get("view_index", 0)

    packed: dict[str, dict] = {}
    for k, v in (nav or {}).items():
        entry = {kk: _pack_value(vv) for kk, vv in v.items()}
        vi = _nearest_view_index(float(v.get("heading", 0.0)), float(v.get("elevation", 0.0)))
        if vi is not None:
            entry["view_index"] = int(vi)
        packed[k] = entry
    return json.dumps(packed)


def _heading_to_pose(heading_deg: float) -> dict:
    """Yaw-only POSE satisfying wire_types.is_valid_pose().

    MP3D graph mode has no continuous position — fill zeros and encode
    heading as a quaternion around the vertical axis.
    """
    h = math.radians(heading_deg or 0.0)
    return {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, math.sin(h / 2), math.cos(h / 2)],
    }


class MP3DResetTool(BaseCanvasNode):
    """Gym-style ``reset()``: emit the full initial observation bundle.

    One-shot replacement for ``(episode_info → graph_panorama →
    observation)`` seed chains. Fires once at run start (no required
    inputs) and produces every piece an LLM-VLN agent needs to plan
    the first action: instruction, scan/episode IDs, the starting
    viewpoint + heading + navigable neighbours, the per-view panorama
    primitive (``views`` + ``view_meta`` + ``depth_views``), and a
    ``navigable_json`` whose candidates each carry a ``view_index``
    pointing at the rendered view nearest that candidate's direction.
    """

    node_type = "env_mp3d__reset"
    display_name = "MP3D: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="teal",
        config_fields=[
            ConfigField(
                "n_headings",
                "select",
                label="Headings",
                default=12,
                options=_RESET_STEP_N_HEADINGS,
            ),
        ],
    )
    description = "Begin episode — emit instruction + ids (metadata only, no observation)"
    category = "environment"
    icon = "RotateCcw"
    input_ports: ClassVar[list] = [
        PortDef("trigger", "TEXT", "Optional trigger (any value fires the reset)", optional=True),
    ]
    output_ports = [
        PortDef("instruction", "TEXT", "Navigation instruction for current episode"),
        PortDef("episode_id", "TEXT", "Episode ID"),
        PortDef("scan_id", "TEXT", "Scan ID"),
        PortDef("dataset", "TEXT", "Dataset tag (R2R / R4R / RxR / REVERIE / CVDN / NDH)"),
        PortDef(
            "extras_json",
            "TEXT",
            "Dataset-specific extras (language, objId, dialog_history, …) as JSON",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Episode placement is env panel-owned; reset only reads metadata.
        # Perception (panorama / navigable) is pulled via the observe_* nodes.
        mgr = _get_env()
        loop = asyncio.get_running_loop()

        if not mgr.initialized:
            log.warning("MP3DEnvManager not initialized — lazy-initializing with defaults")
            await loop.run_in_executor(
                mgr.executor,
                mgr.initialize,
                640,
                480,
                60.0,
                False,
                False,
                "R2R",
                "val_unseen",
                10,
            )

        info = await loop.run_in_executor(mgr.executor, mgr.get_episode_info)
        instruction = str(info.get("instruction", ""))
        episode_id = str(info.get("episode_id", ""))
        scan_id = str(info.get("scan_id", ""))
        dataset = str(info.get("dataset", ""))
        extras = info.get("extras") or {}

        self._self_log("episode_id", episode_id)
        self._self_log("scan_id", scan_id)
        self._self_log("dataset", dataset)
        self._self_log("instruction_preview", instruction[:200])
        if extras:
            self._self_log("extras_keys", sorted(extras.keys()))

        return {
            "instruction": instruction,
            "episode_id": episode_id,
            "scan_id": scan_id,
            "dataset": dataset,
            "extras_json": json.dumps(extras, default=str),
        }


# ══════════════════════════════════════════════════════════════════════
# EnvMP3DNodeSet — the unified nodeset
# ══════════════════════════════════════════════════════════════════════


class EnvMP3DNodeSet(BaseNodeSet):
    """Matterport3D discrete panoramic navigation as a NodeSet.

    Exposes the R2R navigation interface through 7 graph-mode canvas nodes:
    reset, step (gym facade) + episode_info, navigate_to, observation,
    evaluate, graph_panorama (fine-grained). All image output is the
    per-view primitive (views + view_meta + depth_views), never a composite.

    Works both locally (in-process) and as an auto-hosted server
    (``?mode=server`` — runs under ``MP3D_PYTHON`` interpreter).

    Initialization kwargs (passed via /load endpoint or initialize()):
        width:        Camera width in pixels (default 640).
        height:       Camera height in pixels (default 480).
        vfov:         Vertical field-of-view in degrees (default 60).
        preloading:   Preload all panoramas into RAM (default False).
        depth:        Enable depth output (default True).
        split:        R2R split to load (default val_unseen).
        cache_size:   Panorama LRU cache size (default 10).
    """

    name = "env_mp3d"
    description = "Matterport3D discrete panoramic navigation (R2R)"
    server_python = conda_env_python("ac-mp3d", "MP3D_PYTHON")
    env_panel = MP3DEnvPanel
    parallelism = "replicated"  # Stateful simulator: per-worker scene + agent pose.

    def __init__(self) -> None:
        super().__init__()
        self._mgr = MP3DEnvManager.get()

    def get_tools(self) -> list:
        return [
            # gym-like env interface (see docs: nodesets/env/template.html)
            MP3DResetTool(),  # env_mp3d__reset (metadata only)
            MP3DStepWaypointTool(),  # env_mp3d__step_waypoint
            MP3DObservationTool(),  # env_mp3d__observe_navigable
            MP3DGraphPanoramaTool(),  # env_mp3d__observe_panorama
            MP3DEvaluateTool(),  # env_mp3d__evaluate
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Initialize the MatterSim simulator.

        Kwargs:
            width:       Camera width (default 640).
            height:      Camera height (default 480).
            vfov:        Vertical FOV degrees (default 60).
            preloading:  Preload all panos into RAM (default False).
            depth:       Enable depth sensor (default False). Requires
                         per-viewpoint ``_skybox_depth_small.png`` files,
                         which are not in the shipped zips — MatterSim
                         aborts panorama rendering if they're missing.
            dataset:     Default dataset (R2R / R4R / RxR / REVERIE / CVDN / NDH).
            split:       Default split within the dataset (RxR: add _en / _hi / _te).
            cache_size:  Panorama cache size (default 10).
        """
        if self._mgr.initialized:
            log.info("MP3DEnvManager already initialized — skipping")
            return

        width = int(kwargs.get("width", 640))
        height = int(kwargs.get("height", 480))
        vfov = float(kwargs.get("vfov", 60.0))
        preloading = bool(kwargs.get("preloading", False))
        depth = bool(kwargs.get("depth", True))
        dataset = str(kwargs.get("dataset", "R2R"))
        split = str(kwargs.get("split", "val_unseen"))
        cache_size = int(kwargs.get("cache_size", 10))

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            self._mgr.initialize,
            width,
            height,
            vfov,
            preloading,
            depth,
            dataset,
            split,
            cache_size,
        )
        log.info(
            "EnvMP3DNodeSet initialized (dataset=%s split=%s %dx%d)",
            dataset,
            split,
            width,
            height,
        )

        # Auto-seed a default episode so a freshly-loaded nodeset is
        # immediately ready for reset/step — callers can skip the
        # Play click in the typical "load → run" flow. They can still
        # override via the env panel's Play action.
        try:
            await loop.run_in_executor(
                self._mgr.executor,
                self._mgr.set_episode,
                dataset,
                split,
                0,
            )
            log.info("EnvMP3DNodeSet seeded default episode %s/%s index=0", dataset, split)
        except Exception as exc:
            log.warning("EnvMP3DNodeSet default episode seed failed: %s", exc)

    async def shutdown(self) -> None:
        """Shutdown the MatterSim simulator."""
        self._mgr.shutdown()
        log.info("EnvMP3DNodeSet shut down")

    async def get_eval_metadata(self) -> dict:
        """Return evaluation metadata for the Matterport3D benchmark.

        Static split/metric info is always available. Dynamic episode
        counts and the dataset index populate after initialization.
        """
        metadata: dict[str, Any] = {
            "env_name": "matterport3d",
            "datasets": ["R2R", "R4R", "RxR", "REVERIE", "CVDN", "NDH"],
            "splits": ["train", "val_seen", "val_unseen", "test"],
            "episode_counts": {},
            "metrics": [
                "success_rate",
                "spl",  # Success weighted by Path Length
                "ndtw",  # Normalised Dynamic Time Warping
                "sdtw",  # Success weighted by normalised DTW
                "path_length",
                "oracle_success",
                "trajectory_length",
            ],
            "supports_set_episode": True,
            "step_budget": 30,  # R2R standard: ≤30 actions
            "action_space": "discrete",
            "view_angles": 36,
            "elevation_levels": 3,
        }

        if self._mgr.initialized:
            status = self._mgr.get_status()
            metadata["episode_counts"] = status.get("episode_counts", {})
            metadata["current_dataset"] = status.get("dataset", "")
            metadata["current_split"] = status.get("split", "")
            metadata["current_scan"] = status["scan"]
            metadata["current_viewpoint"] = status["viewpoint"]

        return metadata
