"""Human-performance run manager — an interactive human driver for env_habitat.

Backs the Human tab. Where ``CodingAgentRunner`` spawns an auto_host *and* a
non-interactive agent driver, this one spawns *only* the ``env_habitat``
auto_host and then drives it one keypress at a time from the browser:

    load_episode(i) -> panel field push + play + reset + first frame
    step(action)    -> env_habitat__step_discrete + observe (new frame + pose)
    stop()          -> step(0) if still live, then env_habitat__evaluate

It talks to the auto_host over the exact same HTTP surface the
beta-coding-agent driver uses (``/env-panel/field/{name}``,
``/env-panel/action/{name}``, ``/call/{fn}``), so the metrics come from
habitat's own ruler — SR / OSR / NE / nDTW / SPL identical to the agent runs.

One habitat env = one GPU, so this owns at most one server and one live
session at a time. Every human action + trajectory coordinate is persisted
under ``outputs/human/{split}/``:

    episode_{i}.jsonl   full trajectory (meta, per-step action + pose, metrics)
    summary.json        per-episode records (tested/success/metrics) + aggregate

The habitat interpreter/source are resolved from the already-discovered
``env_habitat`` nodeset in the workspace registry — this module never imports
workspace or habitat code (framework import boundary), mirroring
``CodingAgentRunner``.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("agentcanvas.human")

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "human"

NODESET_NAME = "env_habitat"

# Discrete action space — identical to the coding-agent experiments.
ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
# Human runs render at the experiment resolution by default (512 px RGB).
DEFAULT_RGB_RESOLUTION = 512
# Metrics surfaced to the UI aggregate, in display order.
AGG_METRIC_KEYS = ("success", "oracle_success", "distance_to_goal", "ndtw", "spl")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


class HumanRunner:
    """Singleton service (lifespan-owned) managing the human-test env + session."""

    _HEALTH_TIMEOUT = 3.0  # seconds; liveness probe against the auto_host
    # Shown when the auto_host has vanished mid-session. The usual culprit is a
    # backend auto-reload: ``uvicorn --reload`` restarts the worker on any .py
    # change and the habitat subprocess (armed to die with its parent) goes with
    # it. Recovery is a fresh Start Session; the durable fix is no --reload.
    _DEAD_MSG = (
        "Habitat env server is not reachable — it was most likely killed by a "
        "backend auto-reload (uvicorn --reload restarts the worker on any .py "
        "change and takes the env subprocess down with it). Click Start Session "
        "to relaunch. For uninterrupted sessions, run the backend without --reload."
    )

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._server: Any = None  # BaseServer for the habitat auto_host
        self._url: str | None = None
        self._server_state = "idle"  # idle | starting | ready | error | stopped
        self._server_error: str | None = None
        self._split = "rand100"
        self._lock = threading.Lock()  # serialize env HTTP + session mutation
        # Live session (one episode at a time); None between episodes.
        self._session: dict[str, Any] | None = None

    # ── server lifecycle ──────────────────────────────────────────────

    def start_server(self, split: str = "rand100") -> dict[str, Any]:
        """Spawn (or reuse) the env_habitat auto_host and block until it's ready.

        BLOCKING — call via ``asyncio.to_thread`` so it runs on a pooled worker
        thread that outlives the request. This is deliberate, not laziness:
        ``BaseServer`` + auto_host arm ``PR_SET_PDEATHSIG``, which on Linux fires
        when the *spawning thread* dies (not the whole process). Spawning from a
        short-lived ``threading.Thread`` would therefore SIGTERM the env the
        instant that thread returned — even on a clean start. A pooled to_thread
        worker persists for the process lifetime, so the env stays up. A cold
        scene load takes only a few seconds; callers still poll server_status().
        """
        with self._lock:
            if self._server_state == "starting":
                return self._server_status_unlocked()
            # A live 'ready' server is reused; a stale 'ready' (env killed by a
            # backend reload) or any idle/error/stopped state (re)spawns fresh.
            if self._server_state == "ready" and self._url and self._health(self._url):
                return self._server_status_unlocked()
            self._teardown()
            self._split = split or "rand100"
            self._server_state = "starting"
            self._server_error = None
            self._session = None

        try:
            self._spawn(self._split)  # blocks: BaseServer.start waits for /health
            with self._lock:
                self._server_state = "ready"
            log.info("human env_habitat ready at %s (split=%s)", self._url, self._split)
        except Exception as exc:  # noqa: BLE001 — surface to the UI, never crash lifespan
            log.exception("human env_habitat start failed")
            with self._lock:
                self._server_error = str(exc)
                self._server_state = "error"
                self._teardown()
        return self.server_status()

    def _spawn(self, split: str) -> None:
        from ..server.base_server import BaseServer

        nodeset = getattr(self._registry, "_discovered_nodesets", {}).get(NODESET_NAME)
        if nodeset is None:
            raise RuntimeError(f"nodeset {NODESET_NAME!r} not discovered by the registry")
        nodeset_cls = type(nodeset)
        python = getattr(nodeset_cls, "server_python", None) or sys.executable
        source_file = getattr(nodeset, "_source_file", None)
        if source_file is None:
            raise RuntimeError(f"nodeset {NODESET_NAME!r} has no _source_file")

        backend_dir = str(REPO_ROOT / "agentcanvas" / "backend")
        port = _free_port()
        server = BaseServer(
            name="human_env_habitat",
            command=[
                python,
                "-m",
                "app.server.auto_host",
                "--file",
                str(source_file),
                "--class",
                nodeset_cls.__name__,
                "--port",
                str(port),
            ],
            port=port,
            startup_timeout=getattr(nodeset_cls, "startup_timeout", 1800),
            working_dir=backend_dir,
            env={"PYTHONPATH": f"{backend_dir}:{REPO_ROOT}"},
        )
        server.start()
        url = f"http://127.0.0.1:{port}"
        # Publish the handle before the field pushes so a push failure still
        # leaves a tearable server (no orphaned habitat holding the GPU).
        self._server = server
        self._url = url
        # Dataset + split are pushed once here (a split push re-initializes the
        # env against its YAML — the expensive part); per-episode work only ever
        # pushes episode_index afterward.
        self._panel_field(url, "dataset", "R2R-CE")
        self._panel_field(url, "split", split)

    def stop_server(self) -> dict[str, Any]:
        """Tear down the auto_host (frees the GPU) and drop any live session."""
        with self._lock:
            self._session = None
            self._teardown()
            self._server_state = "stopped"
        log.info("human env_habitat stopped")
        return self.server_status()

    def _teardown(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                log.exception("human auto_host stop raised")
            self._server = None
        self._url = None

    def shutdown(self) -> None:
        """App-exit hook (lifespan teardown)."""
        with self._lock:
            self._session = None
            self._teardown()

    # ── liveness ──────────────────────────────────────────────────────

    def _health(self, url: str) -> bool:
        """True iff the auto_host answers GET /health 200 (never raises)."""
        try:
            resp = requests.get(f"{url}/health", timeout=self._HEALTH_TIMEOUT)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _mark_env_dead(self) -> None:
        """Fold a vanished auto_host into a recoverable error state (caller
        holds the lock). Drops the live session and clears the server handle so
        the next Start Session respawns cleanly."""
        self._server_state = "error"
        self._server_error = self._DEAD_MSG
        self._session = None
        self._teardown()

    # ── env HTTP helpers (driver-side; identical surface to run_episodes.py) ──
    #
    # A dropped connection means the auto_host vanished (see _DEAD_MSG) — it is
    # re-raised as a clean RuntimeError so the API returns a readable 409 rather
    # than a raw urllib3 ``Max retries exceeded`` 500. A non-2xx status is a real
    # env-side error and surfaces as-is via raise_for_status().

    def _req(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            resp = requests.request(method, url, timeout=600, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(self._DEAD_MSG) from exc
        resp.raise_for_status()
        return resp

    def _panel_field(self, url: str, name: str, value: Any) -> None:
        self._req("POST", f"{url}/env-panel/field/{name}", json={"value": value})

    def _panel_action(self, url: str, name: str) -> None:
        self._req("POST", f"{url}/env-panel/action/{name}", json={"params": {}})

    def _call(
        self, url: str, fn: str, inputs: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"inputs": inputs}
        if config:
            body["config"] = config
        return self._req("POST", f"{url}/call/{fn}", json=body).json()["outputs"]

    def _observe(self, url: str) -> tuple[str | None, list | None]:
        """Pull the current egocentric RGB (base64 PNG) + agent position."""
        outputs = self._call(url, "env_habitat__observe_egocentric", {})
        rgb = outputs.get("rgb")
        pose = outputs.get("pose") or {}
        position = pose.get("position") if isinstance(pose, dict) else None
        orientation = pose.get("orientation") if isinstance(pose, dict) else None
        return rgb, ({"position": position, "orientation": orientation}
                     if position is not None else None)

    # ── episode session ───────────────────────────────────────────────

    def load_episode(
        self, index: int, rgb_resolution: int = DEFAULT_RGB_RESOLUTION
    ) -> dict[str, Any]:
        """Place + arm episode ``index`` and return instruction + first frame.

        Re-loading an already-tested episode starts a fresh trajectory (the
        record is overwritten on the next stop — this is the UI's "re-test").
        """
        with self._lock:
            url = self._require_ready()
            if index < 0 or index > 99:
                raise ValueError(f"episode index out of range 0-99: {index}")
            # panel field push -> set_episode; play + reset arm the placed
            # episode. reset carries the RGB-resolution override in its config.
            self._panel_field(url, "episode_index", index)
            self._panel_action(url, "play")
            reset_config = {"rgb_resolution": str(rgb_resolution)} if rgb_resolution else None
            ep = self._call(url, "env_habitat__reset", {"trigger": "human"}, reset_config)
            instruction = ep.get("instruction", "")
            rgb, pose = self._observe(url)

            session = {
                "index": index,
                "episode_id": ep.get("episode_id"),
                "scene_id": ep.get("scene_id"),
                "instruction": instruction,
                "rgb_resolution": rgb_resolution,
                "step_count": 0,
                "actions": [],       # ordered action ints
                "trajectory": [],    # per-step {action, action_name, position, orientation}
                "done": False,
                "called_stop": False,
                "end_reason": None,
                "metrics": None,
                "t0": time.time(),
            }
            self._session = session

            # Fresh trajectory file (overwrites any prior test of this episode).
            run_dir = self._run_dir(create=True)
            traj_path = run_dir / f"episode_{index}.jsonl"
            with traj_path.open("w") as fh:
                fh.write(json.dumps({
                    "t": 0.0, "kind": "episode_meta", "index": index,
                    "episode_id": session["episode_id"], "scene_id": session["scene_id"],
                    "instruction": instruction, "split": self._split,
                    "rgb_resolution": rgb_resolution,
                    "start_position": pose.get("position") if pose else None,
                }) + "\n")

            return {
                "index": index,
                "episode_id": session["episode_id"],
                "scene_id": session["scene_id"],
                "instruction": instruction,
                "frame": rgb,
                "position": pose.get("position") if pose else None,
                "step_count": 0,
                "done": False,
            }

    def step(self, action: int) -> dict[str, Any]:
        """Execute one discrete movement action (1/2/3) and return the new frame.

        STOP (0) is not accepted here — it goes through ``stop()`` so the confirm
        gate + evaluation stay in one place.
        """
        with self._lock:
            url = self._require_ready()
            session = self._require_session()
            if action not in (1, 2, 3):
                raise ValueError(f"step action must be 1/2/3 (got {action}); use stop() for STOP")
            if session["done"]:
                raise RuntimeError("episode already over — call stop to evaluate")

            out = self._call(url, "env_habitat__step_discrete", {"action": action})
            terminated = bool(out.get("terminated"))
            truncated = bool(out.get("truncated"))
            rgb, pose = self._observe(url)

            session["step_count"] += 1
            session["actions"].append(action)
            entry = {
                "step": session["step_count"], "action": action,
                "action_name": ACTION_NAMES.get(action, "?"),
                "position": pose.get("position") if pose else None,
                "orientation": pose.get("orientation") if pose else None,
                "terminated": terminated, "truncated": truncated,
            }
            session["trajectory"].append(entry)
            self._append_traj({"t": round(time.time() - session["t0"], 2),
                               "kind": "step", **entry})

            if terminated or truncated:
                # A budget truncation ends the episode without a human STOP.
                session["done"] = True
                session["end_reason"] = "budget" if truncated else "terminated"

            return {
                "frame": rgb,
                "position": pose.get("position") if pose else None,
                "step_count": session["step_count"],
                "done": session["done"],
                "end_reason": session["end_reason"],
            }

    def stop(self) -> dict[str, Any]:
        """Issue STOP (if still live) then evaluate; persist the record."""
        with self._lock:
            url = self._require_ready()
            session = self._require_session()

            if not session["done"]:
                out = self._call(url, "env_habitat__step_discrete", {"action": 0})
                session["step_count"] += 1
                session["actions"].append(0)
                _, pose = self._observe(url)
                entry = {
                    "step": session["step_count"], "action": 0, "action_name": "STOP",
                    "position": pose.get("position") if pose else None,
                    "orientation": pose.get("orientation") if pose else None,
                    "terminated": bool(out.get("terminated")),
                    "truncated": bool(out.get("truncated")),
                }
                session["trajectory"].append(entry)
                self._append_traj({"t": round(time.time() - session["t0"], 2),
                                   "kind": "stop", **entry})
                session["called_stop"] = True
                session["done"] = True
                session["end_reason"] = "stop"

            metrics_out = self._call(url, "env_habitat__evaluate", {"trigger": "human"})
            metrics = metrics_out.get("metrics") or {}
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
            session["metrics"] = metrics
            self._append_traj({"t": round(time.time() - session["t0"], 2),
                               "kind": "metrics", "metrics": metrics})

            self._persist_record(session)
            return {
                "index": session["index"],
                "metrics": metrics,
                "step_count": session["step_count"],
                "called_stop": session["called_stop"],
                "end_reason": session["end_reason"],
                "done": True,
            }

    # ── persistence ───────────────────────────────────────────────────

    def _run_dir(self, create: bool = False) -> Path:
        split = self._split if all(c.isalnum() or c in "_-." for c in self._split) else "rand100"
        d = OUTPUT_ROOT / split
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def _append_traj(self, entry: dict[str, Any]) -> None:
        session = self._session
        if session is None:
            return
        path = self._run_dir(create=True) / f"episode_{session['index']}.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _persist_record(self, session: dict[str, Any]) -> None:
        run_dir = self._run_dir(create=True)
        summary_path = run_dir / "summary.json"
        data: dict[str, Any] = {"split": self._split, "episodes": []}
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
            except (OSError, ValueError):
                data = {"split": self._split, "episodes": []}
        record = {
            "index": session["index"],
            "episode_id": session["episode_id"],
            "scene_id": session["scene_id"],
            "instruction": session["instruction"],
            "metrics": session["metrics"] or {},
            "num_steps": session["step_count"],
            "num_actions": len(session["actions"]),
            "called_stop": session["called_stop"],
            "end_reason": session["end_reason"],
            "tested": True,
            "tested_at": time.time(),
            "wall_sec": round(time.time() - session["t0"], 1),
        }
        episodes = [e for e in data.get("episodes", []) if e.get("index") != session["index"]]
        episodes.append(record)
        episodes.sort(key=lambda e: e.get("index", 0))
        data["split"] = self._split
        data["episodes"] = episodes
        data["aggregate"] = self._aggregate(episodes)
        data["updated_at"] = time.time()
        summary_path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
        agg: dict[str, Any] = {"tested": len(episodes)}
        buckets: dict[str, list[float]] = {}
        for e in episodes:
            for k, v in (e.get("metrics") or {}).items():
                if isinstance(v, bool):
                    v = float(v)
                if isinstance(v, (int, float)):
                    buckets.setdefault(k, []).append(float(v))
            buckets.setdefault("num_steps", []).append(float(e.get("num_steps") or 0))
        for k, vals in buckets.items():
            if vals:
                agg[k] = round(sum(vals) / len(vals), 4)
        return agg

    # ── read surface ──────────────────────────────────────────────────

    def status(self, split: str | None = None) -> dict[str, Any]:
        """Per-episode tested/success records + aggregate for a split."""
        use = split or self._split
        if not all(c.isalnum() or c in "_-." for c in use):
            use = "rand100"
        summary_path = OUTPUT_ROOT / use / "summary.json"
        episodes: list[dict[str, Any]] = []
        aggregate: dict[str, Any] | None = None
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
                aggregate = data.get("aggregate")
                for e in data.get("episodes", []):
                    m = e.get("metrics") or {}
                    episodes.append({
                        "index": e.get("index"),
                        "success": m.get("success"),
                        "oracle_success": m.get("oracle_success"),
                        "distance_to_goal": m.get("distance_to_goal"),
                        "ndtw": m.get("ndtw"),
                        "spl": m.get("spl"),
                        "num_steps": e.get("num_steps"),
                        "called_stop": e.get("called_stop"),
                        "tested": e.get("tested", True),
                    })
            except (OSError, ValueError):
                pass
        return {"split": use, "episodes": episodes, "aggregate": aggregate}

    def _server_status_unlocked(self) -> dict[str, Any]:
        return {
            "state": self._server_state,
            "error": self._server_error,
            "split": self._split,
            "url": self._url,
            "session": self._session_view(),
        }

    def server_status(self) -> dict[str, Any]:
        # Probe liveness OUTSIDE the lock so a routine poll never blocks an
        # in-flight step; a live episode holds the lock for its whole HTTP
        # sequence, so this can't race a load/step (it waits for it first).
        with self._lock:
            state, url = self._server_state, self._url
        if state == "ready" and url and not self._health(url):
            with self._lock:
                if self._server_state == "ready":  # re-check under lock
                    self._mark_env_dead()
        with self._lock:
            return self._server_status_unlocked()

    def _session_view(self) -> dict[str, Any] | None:
        s = self._session
        if s is None:
            return None
        return {
            "index": s["index"],
            "episode_id": s["episode_id"],
            "scene_id": s["scene_id"],
            "instruction": s["instruction"],
            "step_count": s["step_count"],
            "done": s["done"],
            "called_stop": s["called_stop"],
            "end_reason": s["end_reason"],
            "metrics": s["metrics"],
            "rgb_resolution": s["rgb_resolution"],
        }

    # ── guards ────────────────────────────────────────────────────────

    def _require_ready(self) -> str:
        if self._server_state != "ready" or self._url is None:
            raise RuntimeError(f"env not ready (state={self._server_state})")
        return self._url

    def _require_session(self) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("no episode loaded")
        return self._session
