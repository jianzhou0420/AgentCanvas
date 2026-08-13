"""Shared episode loop for all harnesses — the 90% the three legacy drivers
duplicated, collected once.

Per episode: place via the env-panel HTTP surface, read the instruction
driver-side, build the briefing (prompts.py), hand a fresh EpisodeContext to
the harness adapter, and let it run ONE clean session while every event goes
through the single EventSink (so the episode_{i}.jsonl vocabulary is enforced
by construction). Afterwards read habitat's own measures via
``env_habitat__evaluate`` — the same ruler as the verified baselines.

Artifact layout is byte-compatible with the legacy drivers (episode_{i}.jsonl
+ raw/ + live_{i}/ + summary.json) and lands in the SAME per-harness output
roots, so the Coding-Agent Monitor needs no changes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from cells import (BENCHMARK_FROZEN, STD_FROZEN,
                   WP_MAX_MOVES, WP_THINK_BUDGET, CellSpec)
from prompts import (FIRST_PROMPT, HMEQA_FIRST_PROMPT, HYBRID_FIRST_PROMPT,
                     LIBERO_FIRST_PROMPT, LIBERO_TOOLBOX_FIRST_PROMPT,
                     LIBERO_TOOLBOX_VISION_FIRST_PROMPT, build_briefing)

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "coding-agent" / "bridges" / "mcp_bridge.py"
WP_BRIDGE_PATH = REPO_ROOT / "coding-agent" / "bridges" / "wp_bridge.py"
# Real Unitree Go2. Same tool surface as mcp_bridge (observe/step/look_around),
# but its HTTP peer is go2_host.py on the robot's own machine rather than a
# local habitat auto_host — CycloneDDS is layer-2, so the SDK cannot run here.
GO2_BRIDGE_PATH = REPO_ROOT / "coding-agent" / "bridges" / "go2_bridge.py"
# HM-EQA (explore-eqa, Ren et al. 2024): multiple-choice EQA on HM3D. The
# episode ends by answer("A".."D") instead of STOP — the answer rides the
# tool-result channel (EventSink.last_step_result) back to the driver, which
# passes it to env_hmeqa__evaluate as pred_letter.
HMEQA_BRIDGE_PATH = REPO_ROOT / "coding-agent" / "bridges" / "hmeqa_bridge.py"
# Agent-selected hybrid surface: primitive step() AND waypoint goto() in one
# toolface, with the look-then-move gate that makes the choice of lens the
# choice of interface. Same habitat auto_host peer as mcp_bridge, plus the
# waypoint predictor that wp uses.
HYBRID_BRIDGE_PATH = REPO_ROOT / "coding-agent" / "bridges" / "hybrid_bridge.py"
# LIBERO manipulation (Liu et al. 2023): the arm line. Same two-tool minimal
# surface, but step() takes the env's native 7-D continuous control ticks and
# there is no terminal action — LIBERO detects success from scene state, so
# the episode ends on task success or budget exhaustion (libero_bridge.py).
LIBERO_BRIDGE_PATH = REPO_ROOT / "coding-agent" / "bridges" / "libero_bridge.py"


def env_verb_prefix(benchmark: str) -> str:
    """Nodeset verb namespace for ANY benchmark — the driver's one place that
    turns a benchmark name into the ``env_*`` prefix its auto_host answers on.

    Also the value of the server's ``/health`` ``name``, which is what makes
    the pre-flight peer check (a mismatched server places the WRONG episodes)
    a one-liner for every line rather than a per-family special case.
    """
    if benchmark == "hmeqa":
        return "env_hmeqa"
    if benchmark == "vlnverse":
        return "env_vlnverse"
    if benchmark == "libero":
        return "env_libero"
    return "env_habitat"


# ── habitat auto_host HTTP helpers (driver-side; not visible to the agent) ──


def panel_field(server_url: str, name: str, value: Any) -> None:
    resp = requests.post(
        f"{server_url}/env-panel/field/{name}", json={"value": value}, timeout=600
    )
    resp.raise_for_status()


def panel_action(server_url: str, name: str) -> None:
    resp = requests.post(
        f"{server_url}/env-panel/action/{name}", json={"params": {}}, timeout=600
    )
    resp.raise_for_status()


def call_function(
    server_url: str, fn: str, inputs: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"inputs": inputs}
    if config:
        body["config"] = config
    resp = requests.post(f"{server_url}/call/{fn}", json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()["outputs"]


# ── serialization helpers ──


def json_safe(obj: Any, _depth: int = 0) -> Any:
    """Coerce SDK messages / codex events / options objects into JSON; base64
    image blobs are elided to a marker (frames live in live_*/)."""
    if _depth > 12:
        return "<max-depth>"
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"_type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = json_safe(getattr(obj, f.name), _depth + 1)
        return out
    if isinstance(obj, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x, _depth + 1) for x in obj]
    if isinstance(obj, bytes):
        return f"<bytes {len(obj)}>"
    if isinstance(obj, str):
        # long, space-free string = base64 blob → elide; prose keeps spaces
        if len(obj) > 4000 and " " not in obj[:200]:
            return f"<blob {len(obj)} chars elided>"
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return str(obj)


# Restored 2026-08-01 after the two-machine merge dropped it (claude_sdk.py
# still calls is_rate_limited at episode finalization — every claude-sdk
# episode NameError'd at the result step without this). Source: 6d0bf63
# (std-v2 rate-limit resilience). NOTE: that commit's worker-level
# retry/backoff loop did NOT survive the merge and still needs a re-merge.
# "session limit"/"usage limit" = the 5h subscription window (resets on a clock,
# not transient) — retry can't clear it within the window, but tagging it here
# routes it to error="rate_limited" -> excluded (not scored 0 as a nav failure).
RATE_LIMIT_MARKERS = ("temporarily limiting", "rate limited", "overloaded",
                      "rate_limit", "429", "session limit", "usage limit",
                      "hit your session")
RATE_LIMIT_MAX_ATTEMPTS = 6
RATE_LIMIT_BASE_BACKOFF = 30     # seconds, exponential per attempt
RATE_LIMIT_MAX_BACKOFF = 300     # per-backoff cap


def is_rate_limited(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in RATE_LIMIT_MARKERS)


_TOOL_SCHEMAS_CACHE: dict[tuple[bool, ...], Any] = {}


async def bridge_tool_schemas(bare: bool, wp: bool = False, go2: bool = False,
                              hmeqa: bool = False,
                              hmeqa_tilt: bool = True,
                              auto_observe: bool = False,
                              hybrid: bool = False,
                              libero: bool = False,
                              toolbox: bool = False,
                              toolbox_gt: bool = True) -> Any:
    """The bridge's own tool definitions, introspected in-process from the
    bridge module the sessions actually talk to (mcp_bridge.py, or
    wp_bridge.py for the wp condition, hybrid_bridge.py for the hybrid
    condition, go2_bridge.py for the real robot, hmeqa_bridge.py for
    HM-EQA — whose step description also keys off the tilt mask — or
    libero_bridge.py for the LIBERO arm line; mini's port is
    byte-equivalent, gated by check_equivalence.py). Cached per
    (bare, wp, go2, hmeqa, hmeqa_tilt, auto_observe, hybrid, libero); never
    raises (logging must not break a run).
    ``auto_observe`` mirrors the session's HABITAT_AUTO_OBSERVE so the recorded
    step()/goto() descriptions match."""
    key = (bare, wp, go2, hmeqa, hmeqa_tilt, auto_observe, hybrid, libero,
           toolbox, toolbox_gt)
    if key in _TOOL_SCHEMAS_CACHE:
        return _TOOL_SCHEMAS_CACHE[key]
    bridge_path = (GO2_BRIDGE_PATH if go2
                   else HMEQA_BRIDGE_PATH if hmeqa
                   else LIBERO_BRIDGE_PATH if libero
                   else HYBRID_BRIDGE_PATH if hybrid
                   else WP_BRIDGE_PATH if wp else BRIDGE_PATH)
    # Each bridge reads its own prefix; introspecting go2 with HABITAT_BARE set
    # would silently return the non-bare toolset. (hybrid is a habitat bridge,
    # so it stays on HABITAT_BARE.)
    bare_var = ("GO2_BARE" if go2
                else "HMEQA_BARE" if hmeqa
                else "LIBERO_BARE" if libero else "HABITAT_BARE")
    saved = os.environ.get(bare_var)
    saved_tilt = os.environ.get("HMEQA_TILT")
    saved_ao = os.environ.get("HABITAT_AUTO_OBSERVE")
    saved_lao = os.environ.get("LIBERO_AUTO_OBSERVE")
    saved_ltb = os.environ.get("LIBERO_TOOLBOX")
    saved_ltg = os.environ.get("LIBERO_TOOLBOX_GT")
    try:
        os.environ[bare_var] = "1" if bare else "0"
        os.environ["HABITAT_AUTO_OBSERVE"] = "1" if auto_observe else "0"
        if libero:
            os.environ["LIBERO_AUTO_OBSERVE"] = "1" if auto_observe else "0"
            os.environ["LIBERO_TOOLBOX"] = "1" if toolbox else "0"
            os.environ["LIBERO_TOOLBOX_GT"] = "1" if toolbox_gt else "0"
        if hmeqa:
            os.environ["HMEQA_TILT"] = "1" if hmeqa_tilt else "0"
        spec = importlib.util.spec_from_file_location("_bridge_introspect", bridge_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tools = await mod.mcp.list_tools()
        _TOOL_SCHEMAS_CACHE[key] = json_safe([
            {"name": getattr(t, "name", None),
             "description": getattr(t, "description", None),
             "input_schema": getattr(t, "inputSchema", None)}
            for t in tools
        ])
    except Exception as exc:  # noqa: BLE001 — logging must never break a run
        _TOOL_SCHEMAS_CACHE[key] = {"error": f"tool-schema introspection failed: {exc!r}"}
    finally:
        for _name, _saved in ((bare_var, saved), ("HMEQA_TILT", saved_tilt),
                              ("HABITAT_AUTO_OBSERVE", saved_ao),
                              ("LIBERO_AUTO_OBSERVE", saved_lao),
                              ("LIBERO_TOOLBOX", saved_ltb),
                              ("LIBERO_TOOLBOX_GT", saved_ltg)):
            if _saved is None:
                os.environ.pop(_name, None)
            else:
                os.environ[_name] = _saved
    return _TOOL_SCHEMAS_CACHE[key]


# ── event sink: the one writer of the episode_{i}.jsonl vocabulary ──


class EventSink:
    """Wraps one episode's trajectory file. Adapters emit through this only,
    so the curated vocabulary (thinking / assistant_text / tool_use /
    tool_result / system_init / driver_error / result / exit) stays uniform
    across harnesses. Also tracks the agent stats every harness needs:
    per-tool call counts and the last parsed step() result."""

    def __init__(self, traj_path: Path) -> None:
        self._fh = traj_path.open("w")
        self._t0 = time.time()
        self.tool_calls: dict[str, int] = {}
        self.last_step_result: dict[str, Any] | None = None

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "tool_use":
            short = str(payload.get("name", "")).rsplit("__", 1)[-1]
            self.tool_calls[short] = self.tool_calls.get(short, 0) + 1
        elif kind == "tool_result":
            parsed = self._parse_step_result(payload.get("texts") or [])
            if parsed is not None:
                self.last_step_result = parsed
        elif kind == "exit":
            # mini delivers the episode-ending step result as the exit
            # message content (Submitted), not as a tool_result — without
            # this the final STOP never reaches the episode record.
            content = payload.get("content")
            parsed = self._parse_step_result([content] if isinstance(content, str) else [])
            if parsed is not None:
                self.last_step_result = parsed
        self._fh.write(json.dumps(
            {"t": round(time.time() - self._t0, 2), "kind": kind, **payload}) + "\n")
        self._fh.flush()  # live tail -f must see every event as it happens

    @staticmethod
    def _parse_step_result(texts: list[str]) -> dict[str, Any] | None:
        for text in texts:
            try:
                data = json.loads(text)
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict) and "steps_taken_total" in data:
                return data
        return None

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def close(self) -> None:
        self._fh.close()


# ── adapter contract ──


@dataclass
class EpisodeContext:
    index: int
    instruction: str
    briefing: str                 # rendered system prompt / user-message briefing
    first_prompt: str
    server_url: str
    bare: bool
    model: str
    max_turns: int
    step_budget: int
    episode_timeout: int
    workdir: Path
    live_dir: Path
    raw_dir: Path
    wp: bool = False       # waypoint action space (wp_bridge.py)
    hybrid: bool = False   # primitive + waypoint in one surface (hybrid_bridge.py)
    wp_server_url: str = ""  # waypoint-predictor auto_host (wp / hybrid cells)
    wp_max_moves: int = 30   # decision-step cap enforced by wp_bridge (wp only)
    go2: bool = False      # real Unitree Go2 (go2_bridge.py); server_url points
                           # at go2_host.py on the robot's machine, not localhost
    benchmark: str = "r2r"  # r2r | hmeqa | vlnverse | libero
    hmeqa_tilt: bool = True  # hmeqa only: camera-tilt actions 4/5 on the
                             # toolface (cells tilt_actions; --set tilt_actions=0
                             # masks them — bridge + briefing follow together)
    max_budget_usd: float | None = None  # per-episode USD fuse (sdk harness only;
                                         # CLI --max-budget-usd, hmeqa frozen $18)
    hybrid: bool = False   # primitive + waypoint in one surface (hybrid_bridge.py)
    # auto-observe: step()/goto() carry the resulting view (observe() first-look
    # only). RxR default (its long instructions double the turn count under the
    # classic alternation); off for R2R-CE so its frozen baselines still hold.
    auto_observe: bool = False
    # libero loaded-toolbox surface (atomic reads + GT readout + servo macros;
    # cells condition libero_toolbox — see libero_bridge.py TOOLBOX)
    toolbox: bool = False
    # toolbox GT switch: False = pixel_to_3d replaces get_objects
    # (condition libero_toolbox_vision)
    toolbox_gt: bool = True
    extra: dict[str, Any] = field(default_factory=dict)  # harness-specific knobs

    @property
    def turn_budget(self) -> int:
        """Bridge broadcast/STOP-gate budget: off (0) in the bare condition."""
        return 0 if self.bare else self.max_turns

    @property
    def bridge_path(self) -> Path:
        """Stdio bridge module for this condition's action space / embodiment."""
        if self.go2:
            return GO2_BRIDGE_PATH
        if self.benchmark == "hmeqa":
            return HMEQA_BRIDGE_PATH
        if self.benchmark == "libero":
            return LIBERO_BRIDGE_PATH
        if self.hybrid:
            return HYBRID_BRIDGE_PATH
        return WP_BRIDGE_PATH if self.wp else BRIDGE_PATH

    def bridge_env(self) -> dict[str, str]:
        """Env for harnesses that spawn the stdio bridge, prefixed per bridge.

        The prefix is not cosmetic: a bridge that does not recognise the name
        falls back to its default, and for TURN_BUDGET that default is 0 —
        which silently disables the STOP gate and the budget broadcast rather
        than failing. So go2 gets GO2_*, and the two sets never mix.
        """
        if self.go2:
            return {
                "GO2_SERVER_URL": self.server_url,
                "GO2_STEP_BUDGET": str(self.step_budget),
                "GO2_TURN_BUDGET": str(self.turn_budget),
                "GO2_BARE": "1" if self.bare else "0",
                "GO2_LIVE_DIR": str(self.live_dir),
            }
        if self.benchmark == "hmeqa":
            return {
                "HMEQA_SERVER_URL": self.server_url,
                "HMEQA_STEP_BUDGET": str(self.step_budget),
                "HMEQA_TURN_BUDGET": str(self.turn_budget),
                "HMEQA_BARE": "1" if self.bare else "0",
                "HMEQA_TILT": "1" if self.hmeqa_tilt else "0",
                "HMEQA_LIVE_DIR": str(self.live_dir),
            }
        if self.benchmark == "libero":
            return {
                "LIBERO_SERVER_URL": self.server_url,
                "LIBERO_STEP_BUDGET": str(self.step_budget),
                "LIBERO_TURN_BUDGET": str(self.turn_budget),
                "LIBERO_BARE": "1" if self.bare else "0",
                "LIBERO_AUTO_OBSERVE": "1" if self.auto_observe else "0",
                "LIBERO_TOOLBOX": "1" if self.toolbox else "0",
                "LIBERO_TOOLBOX_GT": "1" if self.toolbox_gt else "0",
                "LIBERO_LIVE_DIR": str(self.live_dir),
            }
        env = {
            "HABITAT_SERVER_URL": self.server_url,
            # env_habitat for the R2R/RxR lines, env_vlnverse for the Isaac
            # line — same two verbs, same port shapes, so mcp_bridge only
            # needs the namespace. wp/hybrid do NOT read this: their pano
            # dir_id convention is habitat's clockwise one and the waypoint
            # predictor is habitat-trained, so those surfaces stay
            # habitat-only (refused for vlnverse at cell-build time).
            "HABITAT_VERB_PREFIX": env_verb_prefix(self.benchmark),
            "HABITAT_STEP_BUDGET": str(self.step_budget),
            "HABITAT_TURN_BUDGET": str(self.turn_budget),
            "HABITAT_BARE": "1" if self.bare else "0",
            "HABITAT_AUTO_OBSERVE": "1" if self.auto_observe else "0",
            "HABITAT_LIVE_DIR": str(self.live_dir),
        }
        if self.wp or self.hybrid:  # both need the waypoint predictor
            env["HABITAT_WP_SERVER_URL"] = self.wp_server_url
            env["HABITAT_WP_MAX_MOVES"] = str(self.wp_max_moves)
            if self.extra.get("wp_predict_fn"):
                env["HABITAT_WP_PREDICT_FN"] = str(self.extra["wp_predict_fn"])
        return env


@dataclass
class SessionOutcome:
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    turns: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HarnessAdapter(Protocol):
    name: str            # "claude-sdk" | "mini-swe" | "codex"
    inherent: dict       # recorded harness-inherent facts (thinking, auth, caps)

    def prepare(self, spec: CellSpec) -> None:
        """Once per run, BEFORE any episode: auth guards, version pins into
        self.inherent, and any serving stack the harness owns (mini brings up
        ollama for local models — see mini_swe.py). Raise to abort the cell:
        a misconfigured server that silently degrades is worse than no run."""

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        """Harness-specific block merged into the session_inputs event."""

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        """Run ONE clean session; emit events through sink only."""

    def finalize(self, run_dir: Path) -> dict[str, Any]:
        """Optional. Once per run, AFTER the last episode: whatever audit the
        harness alone can produce (mini slices its serve log for the exact
        per-request prompt-token counts). Merged into summary.run_stats."""


class VramSampler:
    """Peak GPU memory over a run. Silent no-op where nvidia-smi is absent, so
    it costs nothing on the API-billed cells."""

    def __init__(self, period_s: float = 0.5) -> None:
        self.peak_mib = 0
        self._period = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._period):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.split()
                if out:
                    self.peak_mib = max(self.peak_mib, int(out[0]))
            except Exception:  # noqa: BLE001 — sampling must never break a run
                pass

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_mib or None


# ── episode + run loops ──


async def run_episode(
    adapter: HarnessAdapter, spec: CellSpec, cfg: dict[str, Any],
    url: str, index: int, run_dir: Path,
) -> dict[str, Any]:
    # Blocking HTTP rides to_thread so parallel workers never stall the loop
    # (first play on a cold server can hold a scene load for ~30s).
    if spec.go2:
        # Real robot, minimal episode loop (user decision 2026-07-20): no dataset
        # to place into and no env-panel on the host — the instruction is
        # operator-supplied and reset just stands the dog up and zeroes counters.
        instruction = str(cfg["extra"].get("instruction") or "")
        if not instruction:
            raise RuntimeError(
                "go2 cell needs --set instruction=... — there is no dataset to read one from"
            )
        ep = await asyncio.to_thread(
            call_function, url, "env_go2__reset", {"trigger": "driver"}
        )
    elif spec.benchmark == "hmeqa":
        # Same placement flow (panel episode_index + play). reset's
        # `question` port is the RAW text (the verified explore_eqa_hmeqa
        # graph depends on that — it appends the choices itself), so the
        # formatted multiple-choice question the briefing's {question} slot
        # needs is built HERE from question + choices, mirroring
        # explore-eqa's own formatting ("\nA. <choice>").
        await asyncio.to_thread(panel_field, url, "episode_index", index)
        await asyncio.to_thread(panel_action, url, "play")
        ep = await asyncio.to_thread(
            call_function, url, "env_hmeqa__reset", {"trigger": "driver"}
        )
        raw_q = str(ep.get("question") or "").strip()
        choices = ep.get("choices") or []
        if not raw_q or not choices:
            raise RuntimeError(
                f"hmeqa reset returned question={raw_q!r} choices={choices!r} "
                f"(episode {index}) — episode placement failed?"
            )
        instruction = raw_q + "".join(
            f"\n{letter}. {c}" for letter, c in zip("ABCD", choices)
        )
    elif spec.benchmark == "libero":
        # Flat episode index over the suite: task_id = k % tasks_per_suite,
        # init state = k // tasks_per_suite (episodes 0-9 = every task once).
        # The panel's task_id / episode_index cascade places via set_episode;
        # reset (ensure_live) then reads the placed episode untouched. The
        # suite itself was selected once per run via the split field.
        tasks_per_suite = int(cfg["tasks_per_suite"])
        await asyncio.to_thread(
            panel_field, url, "task_id", index % tasks_per_suite)
        await asyncio.to_thread(
            panel_field, url, "episode_index", index // tasks_per_suite)
        ep = await asyncio.to_thread(
            call_function, url, "env_libero__reset", {"trigger": "driver"}
        )
        instruction = str(ep.get("instruction") or "")
        if not instruction:
            raise RuntimeError(
                f"libero reset returned empty instruction (episode {index}) "
                "— episode placement failed?"
            )
    else:
        await asyncio.to_thread(panel_field, url, "episode_index", index)
        await asyncio.to_thread(panel_action, url, "play")

        # habitat-r2r / RxR-CE and vlnverse share this branch: both are
        # instruction-following VLN, both emit the instruction straight off
        # reset. Only the verb namespace differs (vlnverse has no rgb knob —
        # it renders at its own 1024²/90° and ignores the config key).
        reset_config = {"rgb_resolution": str(cfg["rgb_resolution"])}
        ep = await asyncio.to_thread(
            call_function, url, f"{env_verb_prefix(spec.benchmark)}__reset",
            {"trigger": "driver"}, reset_config,
        )
        instruction = ep["instruction"]

    auto_observe = bool(cfg.get("auto_observe", False))
    toolbox = bool(cfg.get("toolbox", False))
    toolbox_gt = bool(cfg.get("toolbox_gt", True))
    briefing = build_briefing(
        instruction, cfg["step_budget"], bare=spec.bare,
        wp=spec.wp, wp_max_moves=cfg.get("wp_max_moves", 30), go2=spec.go2,
        benchmark=spec.benchmark,
        hmeqa_tilt=bool(cfg.get("tilt_actions", True)),
        auto_observe=auto_observe, hybrid=spec.hybrid, toolbox=toolbox,
        toolbox_gt=toolbox_gt,
    )

    workdir = run_dir / f"workdir_{index}"
    workdir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ctx = EpisodeContext(
        index=index,
        instruction=instruction,
        briefing=briefing,
        first_prompt=(HYBRID_FIRST_PROMPT if spec.hybrid
                      else HMEQA_FIRST_PROMPT if spec.benchmark == "hmeqa"
                      else (LIBERO_TOOLBOX_FIRST_PROMPT if toolbox_gt
                            else LIBERO_TOOLBOX_VISION_FIRST_PROMPT)
                      if spec.benchmark == "libero" and toolbox
                      else LIBERO_FIRST_PROMPT if spec.benchmark == "libero"
                      else FIRST_PROMPT),
        server_url=url,
        bare=spec.bare,
        model=spec.model_id,
        max_turns=cfg["max_turns"],
        step_budget=cfg["step_budget"],
        episode_timeout=cfg["episode_timeout"],
        workdir=workdir,
        live_dir=run_dir / f"live_{index}",
        raw_dir=raw_dir,
        benchmark=spec.benchmark,
        hmeqa_tilt=bool(cfg.get("tilt_actions", True)),
        max_budget_usd=cfg.get("max_budget_usd"),
        wp=spec.wp,
        go2=spec.go2,
        hybrid=spec.hybrid,
        auto_observe=auto_observe,
        toolbox=toolbox,
        toolbox_gt=toolbox_gt,
        wp_server_url=cfg.get("wp_server") or "",
        wp_max_moves=cfg.get("wp_max_moves", 30),
        extra=dict(cfg.get("extra") or {}),
    )

    sink = EventSink(run_dir / f"episode_{index}.jsonl")
    try:
        sink.emit("episode_meta", {
            "index": index, "episode_id": ep.get("episode_id"),
            "scene_id": ep.get("scene_id"), "instruction": instruction,
        })
        sink.emit("session_inputs", {
            "cell": spec.name,
            "harness": adapter.name,
            "model": spec.model_id,
            "system_prompt": briefing,
            "first_prompt": FIRST_PROMPT,
            "tool_schemas": await bridge_tool_schemas(
                spec.bare, spec.wp, spec.go2,
                spec.benchmark == "hmeqa",
                bool(cfg.get("tilt_actions", True)),
                auto_observe, spec.hybrid,
                libero=spec.benchmark == "libero",
                toolbox=toolbox, toolbox_gt=toolbox_gt),
            **json_safe(adapter.describe(ctx)),
        })

        outcome = await adapter.run(ctx, sink)

        sink.emit("result", {"result": json_safe({
            "usage": outcome.usage, "cost_usd": outcome.cost_usd,
            "turns": outcome.turns, "error": outcome.error, **outcome.extra,
        })})

        # Evaluate while the trajectory file is still open so the metrics land
        # in the log itself. Driver-side; the agent never sees it. On go2 there
        # is no ground truth, hence no evaluate — success is a human judgment
        # made from the recording, so metrics stay empty rather than fabricated.
        metrics: dict[str, Any] = {}
        if not spec.go2:
            if spec.benchmark == "hmeqa":
                # The agent's letter rides the tool-result channel (see
                # hmeqa_bridge.answer — its result carries steps_taken_total,
                # so it lands as the sink's last parsed step result). An
                # episode that never answered evaluates "" -> success 0,
                # the benchmark's own zero.
                evaluate_fn = "env_hmeqa__evaluate"
                evaluate_inputs: dict[str, Any] = {
                    "pred_letter": str(
                        (sink.last_step_result or {}).get("answer") or "")
                }
            elif spec.benchmark == "libero":
                # Success lives in the env manager's episode state (LIBERO's
                # own done flag) — no agent-side terminal to read back.
                evaluate_fn = "env_libero__evaluate"
                evaluate_inputs = {"trigger": "driver"}
            else:
                evaluate_fn = f"{env_verb_prefix(spec.benchmark)}__evaluate"
                evaluate_inputs = {"trigger": "driver"}
            try:
                metrics_out = await asyncio.to_thread(
                    call_function, url, evaluate_fn, evaluate_inputs
                )
                metrics = metrics_out.get("metrics") or {}
                if isinstance(metrics, str):
                    metrics = json.loads(metrics)
            except Exception as exc:  # noqa: BLE001
                sink.emit("driver_error", {"error": f"evaluate failed: {exc!r}"})
        sink.emit("episode_metrics", {"metrics": metrics})
    finally:
        wall = sink.elapsed
        sink.close()

    last = sink.last_step_result or {}
    episode: dict[str, Any] = {
        "index": index,
        "episode_id": ep.get("episode_id"),
        "scene_id": ep.get("scene_id"),
        "instruction": instruction,
        "metrics": metrics,
        "agent": {
            "tool_calls": sink.tool_calls,
            "env_steps": last.get("steps_taken_total", 0),
            "end_reason": last.get("end_reason"),
            # deliberate termination: STOP on the nav lines, answer() on
            # hmeqa — same "the agent chose to end" semantics either way
            "called_stop": last.get("end_reason") in ("stop_called",
                                                      "answer_submitted"),
            "num_turns": outcome.turns,
            "usage": outcome.usage,
            "total_cost_usd": outcome.cost_usd,
            **json_safe(outcome.extra),
        },
        "wall_sec": round(wall, 1),
    }
    if spec.benchmark == "hmeqa" and last.get("answer"):
        episode["agent"]["answer"] = last.get("answer")
    if outcome.error:
        episode["error"] = outcome.error
    return episode


def is_scored(rec: dict[str, Any]) -> bool:
    """True if the episode is a real, scored navigation attempt that counts
    toward SR.

    INCLUDES a turn-exhausted episode (hit the SDK ``max_turns`` cap without
    calling STOP): it navigated, was evaluated, ``success`` is 0.0 — a
    legitimate FAILURE, not an error to hide (dropping it inflates SR).

    EXCLUDES two kinds of non-attempts:
    - genuine infra failures (timeout / crash) that produced no metrics
      (``metrics == {}`` → ``success is None``);
    - rate-limit / 'limit exceeded' casualties — errored WITHOUT taking a
      single navigation step (``error`` set + ``env_steps`` 0), so the model
      never really attempted the task. (A genuine turn-exhausted run has
      ``env_steps`` > 0 and stays counted.)
    """
    if (rec.get("metrics") or {}).get("success") is None:
        return False
    if rec.get("error") and not ((rec.get("agent") or {}).get("env_steps") or 0):
        return False
    return True


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"episode_count": len(episodes)}
    numeric: dict[str, list[float]] = {}
    for rec in episodes:
        for key, value in (rec.get("metrics") or {}).items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
        numeric.setdefault("env_steps", []).append(float(rec["agent"].get("env_steps", 0)))
    for key, values in numeric.items():
        if values:
            agg[key] = round(sum(values) / len(values), 4)
    agg["stop_rate"] = round(
        sum(1 for r in episodes if r["agent"].get("called_stop")) / max(1, len(episodes)), 4
    )
    return agg


def format_episodes(indices: list[int]) -> str:
    """Inverse of parse_episodes: [7,8,9,14,44] -> '7-9,14,44'."""
    xs = sorted(set(indices))
    if not xs:
        return ""
    out: list[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = x
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def parse_episodes(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part:
            indices.append(int(part))
    return indices


async def run_cell(
    adapter: HarnessAdapter, spec: CellSpec, servers: list[str],
    episodes_spec: str | None = None, run_name: str | None = None,
    extra: dict[str, Any] | None = None, wp_server: str | None = None,
) -> dict[str, Any]:
    """Run (or resume) one cell. Existing episode records in the run dir's
    summary are kept; requested indices are re-run and replace their records."""
    # Each benchmark line carries its own frozen config — hm3d / mp3d / ovon
    # are peers of the habitat-r2r std line, not variants of it.
    cfg = dict(BENCHMARK_FROZEN.get(spec.benchmark, STD_FROZEN))
    if spec.max_turns:  # local GPU gets the bigger cap; rented compute takes 100
        cfg["max_turns"] = spec.max_turns
    cfg["extra"] = dict(extra or {})
    # Off-board (nonstd) overrides via --set: `dataset`/`split` retarget the
    # eval (e.g. RxR-CE / rand100_en) and `max_turns` lifts the SDK turn cap
    # (RxR's long instructions need >100 — episode turn-exhausts otherwise).
    # Popped out of `extra` so they don't leak into the harness/model config;
    # absent → the frozen defaults for this benchmark line.
    # (Merged 2026-08-01: this loop is the superset of the separate max_turns /
    # split pops that used to sit here — one place, not three.)
    for _k in ("dataset", "split", "max_turns", "max_budget_usd",
               "episode_timeout"):
        if _k in cfg["extra"]:
            cfg[_k] = cfg["extra"].pop(_k)
    if "max_turns" in cfg:
        cfg["max_turns"] = int(cfg["max_turns"])
    if cfg.get("max_budget_usd") is not None:
        cfg["max_budget_usd"] = float(cfg["max_budget_usd"])
    if "episode_timeout" in cfg:
        cfg["episode_timeout"] = int(cfg["episode_timeout"])
    # hmeqa tilt mask: `--nonstd --set tilt_actions=0` drops actions 4/5 from
    # the toolface (bridge validation + tool description + briefing together).
    if "tilt_actions" in cfg["extra"]:
        cfg["tilt_actions"] = bool(int(cfg["extra"].pop("tilt_actions")))
    # auto-observe: step()/goto() carry the resulting view so observe() is a
    # first-look only — halves the per-move tool calls, which RxR's long
    # instructions otherwise inflate. Default ON only for RxR-CE; R2R-CE keeps
    # the classic observe/step alternation (and its frozen baselines). Override
    # either way with `--set auto_observe=1|0`. The bridge (HABITAT_AUTO_OBSERVE)
    # and the prompt are both driven off this single flag so they can't disagree.
    # libero toolbox surface: cells knob (condition libero_toolbox) — popped
    # so it drives the bridge/briefing/allowed-tools trio, never the model.
    cfg["toolbox"] = str(
        cfg["extra"].pop("toolbox", 0)).lower() in ("1", "true", "yes")
    # GT switch within the toolbox: 0 = pixel_to_3d instead of get_objects
    # (condition libero_toolbox_vision)
    cfg["toolbox_gt"] = str(
        cfg["extra"].pop("toolbox_gt", 1)).lower() not in ("0", "false", "no")
    _ao = cfg["extra"].pop("auto_observe", None)
    if spec.hybrid:
        # hybrid is ALWAYS separate (non-auto-observe): the model must choose
        # which lens to look through — forward camera (observe) vs waypoint
        # panorama (observe_waypoints) — and that choice IS the interface
        # choice, so a view can never be auto-coupled into a move result.
        cfg["auto_observe"] = False
    else:
        cfg["auto_observe"] = (
            str(_ao).lower() in ("1", "true", "yes") if _ao is not None
            else cfg.get("dataset") == "RxR-CE"
        )
    cfg["wp_server"] = wp_server if (spec.wp or spec.hybrid) else None
    if spec.wp or spec.hybrid:
        # decision-step budget (wp_bridge-enforced; hybrid ignores it and lets
        # the 500-step budget bind, but the value is still recorded). Overridable
        # via --set wp_max_moves=N (popped so it doesn't reach the model).
        cfg["wp_max_moves"] = int(cfg["extra"].pop("wp_max_moves", WP_MAX_MOVES))
        # force substantive thinking (overridable via --set think_budget=N)
        cfg["extra"].setdefault("think_budget", WP_THINK_BUDGET)

    # May raise (bad auth, unpinnable serving context) — that is the point.
    adapter.prepare(spec)

    for url in servers:
        health = requests.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        if spec.go2:
            # go2_host's /health has no "name"; what matters is the feedback
            # feed — without it every step runs open-loop and the calibration
            # numbers stop applying, so refuse rather than degrade silently.
            h = health.json()
            print(f"[std] {url} healthy: go2 feedback={h.get('feedback')} "
                  f"step_m={h.get('step_m')} turn_deg={h.get('turn_deg')}")
            if not (h.get("dry_run") or h.get("feedback")):
                raise RuntimeError(f"{url}: go2 host has no state feedback "
                                   "(closed-loop unavailable) — check the dog link")
        else:
            server_name = health.json()["name"]
            print(f"[std] {url} healthy: {server_name}")
            # hmeqa needs an env_hmeqa auto_host, vlnverse an env_vlnverse
            # one, libero an env_libero one — a mismatched server would place
            # the wrong episodes, so refuse rather than degrade silently. The
            # habitat lines are checked too now that the prefix is computed
            # rather than assumed.
            want = env_verb_prefix(spec.benchmark)
            if server_name != want:
                raise RuntimeError(
                    f"{url} serves {server_name!r} but cell {spec.name} "
                    f"({spec.benchmark}) needs {want!r}"
                )

    if spec.wp or spec.hybrid:
        # vlnverse rides the primitive surface only, because the wp surface is
        # wired to SmartWay and VLNVerse HAS ITS OWN waypoint predictor. Both
        # mismatches below would degrade SILENTLY rather than fail:
        #
        # (1) Pano convention: vlnverse numbers dir_id counter-clockwise (the
        #     NavHarness strip convention) where habitat numbers it clockwise,
        #     so wp_bridge's dir_id arithmetic maps candidates to mirrored
        #     headings.
        # (2) Wrong predictor, NOT an uncalibrated one. The two descend from
        #     the same model (Hong et al. CVPR'22 — transformer/waypoint_bert.py
        #     is byte-identical between the two trees), but everything around
        #     it differs: SmartWay adds ID_CrossAttention and hardcodes
        #     120 angles / 12 imgs, loads best.pth over a gibson-2plus depth
        #     encoder, and is trained on habitat MP3D/Gibson. VLNVerse ships
        #     checkpoints trained on THESE Isaac renders
        #     (internnav/model/waypoint_predictor/checkpoints/wp-train-cv*,
        #     over gibson-4plus-mp3d-train-val-test) — and NavHarness runs the
        #     `_worgb` depth-only variant, which sidesteps the RGB domain gap
        #     entirely.
        #
        # So the path to lifting this is NOT "recalibrate SmartWay for Isaac":
        # it is a second predictor nodeset (or a ckpt/encoder switch on the
        # existing engine, since the transformer core is shared) plus the
        # re-derived pano convention. Tracked as the M5 follow-up.
        if spec.benchmark == "vlnverse":
            raise RuntimeError(
                f"cell {spec.name}: wp/hybrid is wired to the SmartWay "
                "predictor (habitat-trained, CW pano). VLNVerse has its own "
                "predictor checkpoints — serving those is the M5 follow-up. "
                "Use a bare/std vlnverse cell."
            )
        # a wp/hybrid cell without its predictor would silently degrade — refuse
        if not wp_server:
            raise RuntimeError("wp/hybrid cell needs --wp-server (waypoint-predictor auto_host)")
        health = requests.get(f"{wp_server}/health", timeout=10)
        health.raise_for_status()
        print(f"[std] {wp_server} healthy: {health.json()['name']} (waypoint predictor)")

    run_name = run_name or spec.name
    run_dir = spec.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    # Graceful drain (restored 2026-08-03 from 6d0bf63 — the two-machine merge
    # dropped it while stdrun.py kept advertising the `drain` command):
    # `touch <run_dir>/DRAIN` (or `stdrun.py drain <cell>`, or SIGUSR1) asks
    # every worker to finish its current episode, flush, and exit WITHOUT
    # pulling a new one — in-flight work is never cut, un-pulled indices stay
    # pending for the next resume.
    drain = asyncio.Event()
    drain_sentinel = run_dir / "DRAIN"
    if drain_sentinel.exists():
        drain_sentinel.unlink()  # clear a stale sentinel from a prior run
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, drain.set)
    except (NotImplementedError, RuntimeError):
        pass  # signal handlers unavailable on this platform / thread

    # resume: keep prior records for indices not being re-run
    prior: dict[int, dict[str, Any]] = {}
    if summary_path.exists():
        try:
            for rec in json.loads(summary_path.read_text()).get("episodes", []):
                prior[int(rec["index"])] = rec
        except Exception:  # noqa: BLE001 — a corrupt summary must not block reruns
            pass

    indices = parse_episodes(episodes_spec or cfg["episodes"])
    print(f"[std] cell={spec.name} model={spec.model_id} eps={indices[0]}-{indices[-1]} "
          f"workers={len(servers)} -> {run_dir}")
    if not spec.go2:  # no env-panel on the go2 host, and no dataset to select
        for url in servers:
            # ovon's panel has no dataset selector (one dataset; the split IS
            # the experimental axis) — its FROZEN block carries dataset=None.
            if cfg.get("dataset"):
                panel_field(url, "dataset", cfg["dataset"])
            panel_field(url, "split", cfg["split"])

    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in indices:
        queue.put_nowait(index)

    episodes: dict[int, dict[str, Any]] = dict(prior)
    write_lock = asyncio.Lock()
    run_stats: dict[str, Any] = {}  # VRAM peak + the harness's own audit

    async def flush_summary() -> None:
        async with write_lock:
            ordered = [episodes[i] for i in sorted(episodes)]
            summary = {
                "run_name": run_name,
                "cell": spec.name,
                "harness": adapter.name,
                "harness_inherent": adapter.inherent,
                "config": {**{k: v for k, v in cfg.items() if k != "extra"},
                           "model": spec.model_id, "bare": spec.bare,
                           "extra": json_safe(cfg["extra"])},
                "servers": servers,
                "run_stats": run_stats,
                "aggregate": aggregate([e for e in ordered if is_scored(e)]),
                "episodes": ordered,
            }
            summary_path.write_text(json.dumps(summary, indent=2))

    async def worker(position: int, url: str) -> None:
        await asyncio.sleep(position * 2)  # stagger cold-server scene loads
        while True:
            if drain.is_set() or drain_sentinel.exists():
                print(f"[std] worker {position} ({url}) draining — "
                      f"no new episode pulled")
                return
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            print(f"[std] episode {index} starting on {url}")
            # Rate-limit retry (restored 2026-08-03 from 6d0bf63 — the merge
            # dropped it, so a throttled episode landed as a contaminated
            # partial record instead of re-running; the libero full ep0-9 run
            # was the casualty that surfaced it): back off OUTSIDE the timed
            # scope and re-run the episode fresh, so the wait never eats the
            # episode's wall-clock budget.
            episode: dict[str, Any] = {}
            for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
                try:
                    episode = await asyncio.wait_for(
                        run_episode(adapter, spec, cfg, url, index, run_dir),
                        timeout=cfg["episode_timeout"] + 600,  # backstop over in-session caps
                    )
                except asyncio.TimeoutError:
                    print(f"[std] episode {index} TIMED OUT (backstop)")
                    episode = {"index": index, "error": "timeout", "metrics": {},
                               "agent": {"env_steps": 0, "called_stop": False, "tool_calls": {}}}
                except Exception as exc:  # noqa: BLE001 — one bad episode must not kill the run
                    tag = "rate_limited" if is_rate_limited(repr(exc)) else repr(exc)
                    print(f"[std] episode {index} FAILED: {exc!r}")
                    episode = {"index": index, "error": tag, "metrics": {},
                               "agent": {"env_steps": 0, "called_stop": False, "tool_calls": {}}}
                if episode.get("error") != "rate_limited" or attempt == RATE_LIMIT_MAX_ATTEMPTS:
                    break
                backoff = min(RATE_LIMIT_BASE_BACKOFF * (2 ** (attempt - 1)),
                              RATE_LIMIT_MAX_BACKOFF)
                print(f"[std] episode {index} RATE-LIMITED (attempt {attempt}/"
                      f"{RATE_LIMIT_MAX_ATTEMPTS}) — backing off {backoff}s "
                      f"(episode countdown paused) then retrying")
                await asyncio.sleep(backoff)  # OUTSIDE wait_for: excluded from episode_timeout
            episodes[index] = episode
            await flush_summary()
            m = episode.get("metrics") or {}
            print(f"[std] episode {index} done: success={m.get('success')} "
                  f"spl={m.get('spl')} steps={episode['agent'].get('env_steps')} "
                  f"stop={episode['agent'].get('called_stop')}")

    vram = VramSampler()
    vram.start()
    try:
        await asyncio.gather(*(worker(i, url) for i, url in enumerate(servers)))
    finally:
        run_stats["vram_peak_mib"] = vram.stop()
        finalize = getattr(adapter, "finalize", None)
        if callable(finalize):
            try:
                run_stats.update(finalize(run_dir) or {})
            except Exception as exc:  # noqa: BLE001 — audit must not lose a run
                run_stats["finalize_error"] = repr(exc)
        await flush_summary()

    if drain.is_set() or drain_sentinel.exists():
        left: list[int] = []
        while not queue.empty():
            left.append(queue.get_nowait())
        print(f"[std] DRAINED — in-flight episodes finished; "
              f"{len(left)} un-run, resume with --episodes {format_episodes(left)}"
              if left else "[std] DRAINED — queue already empty")
    if drain_sentinel.exists():
        drain_sentinel.unlink()

    final = aggregate([e for e in episodes.values() if is_scored(e)])
    print(f"[std] cell complete -> {summary_path}")
    if run_stats:
        print(f"[std] run stats: {json.dumps(run_stats)}")
    print(json.dumps(final, indent=2))
    # Full statistics report (charts + tables, after the metric block) for
    # every run — user decision 2026-07-23. Best-effort: a failed report
    # must never lose a completed run.
    try:
        from ac_support.run_stats import generate as _generate_stats
        print(f"[std] stats report -> {_generate_stats(run_dir)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[std] stats report failed (non-fatal): {exc!r}")
    return final
