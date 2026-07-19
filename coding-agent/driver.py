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

from cells import STD_FROZEN, WP_MAX_MOVES, WP_THINK_BUDGET, CellSpec
from prompts import FIRST_PROMPT, assert_std_skill_freeze, build_briefing

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "beta-coding-agent" / "mcp_bridge.py"
WP_BRIDGE_PATH = REPO_ROOT / "beta-coding-agent" / "wp_bridge.py"


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


_TOOL_SCHEMAS_CACHE: dict[tuple[bool, bool], Any] = {}


async def bridge_tool_schemas(bare: bool, wp: bool = False) -> Any:
    """The bridge's own tool definitions, introspected in-process from the
    bridge module the sessions actually talk to (mcp_bridge.py, or
    wp_bridge.py for the wp condition; mini's port is byte-equivalent, gated
    by check_equivalence.py). Cached per (bare, wp); never raises (logging
    must not break a run)."""
    key = (bare, wp)
    if key in _TOOL_SCHEMAS_CACHE:
        return _TOOL_SCHEMAS_CACHE[key]
    bridge_path = WP_BRIDGE_PATH if wp else BRIDGE_PATH
    saved = os.environ.get("HABITAT_BARE")
    try:
        os.environ["HABITAT_BARE"] = "1" if bare else "0"
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
        if saved is None:
            os.environ.pop("HABITAT_BARE", None)
        else:
            os.environ["HABITAT_BARE"] = saved
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
    skill: str | None
    model: str
    max_turns: int
    step_budget: int
    episode_timeout: int
    workdir: Path
    live_dir: Path
    raw_dir: Path
    persona: bool = False  # ablation: keep the harness's stock persona
    wp: bool = False       # waypoint action space (wp_bridge.py)
    wp_server_url: str = ""  # waypoint-predictor auto_host (wp cells only)
    wp_max_moves: int = 30   # decision-step cap enforced by wp_bridge (wp only)
    extra: dict[str, Any] = field(default_factory=dict)  # harness-specific knobs

    @property
    def turn_budget(self) -> int:
        """Bridge broadcast/STOP-gate budget: off (0) in the bare condition."""
        return 0 if self.bare else self.max_turns

    @property
    def bridge_path(self) -> Path:
        """Stdio bridge module for this condition's action space."""
        return WP_BRIDGE_PATH if self.wp else BRIDGE_PATH

    def bridge_env(self) -> dict[str, str]:
        """HABITAT_* env for harnesses that spawn the stdio bridge."""
        env = {
            "HABITAT_SERVER_URL": self.server_url,
            "HABITAT_STEP_BUDGET": str(self.step_budget),
            "HABITAT_TURN_BUDGET": str(self.turn_budget),
            "HABITAT_BARE": "1" if self.bare else "0",
            "HABITAT_LIVE_DIR": str(self.live_dir),
        }
        if self.wp:
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

# Rate-limit retry — subscription throttling ("Server is temporarily limiting
# requests") rides through transiently. A throttled session is NOT a navigation
# result: the worker backs off OUTSIDE the timed scope and re-runs the episode
# fresh, so the wait never counts against the episode's wall-clock budget (user
# constraint 2026-07-17: pause the 2400s countdown while waiting). Exhausted
# retries keep error="rate_limited" -> excluded by aggregate() (not scored 0).
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


async def run_episode(
    adapter: HarnessAdapter, spec: CellSpec, cfg: dict[str, Any],
    url: str, index: int, run_dir: Path,
) -> dict[str, Any]:
    # Blocking HTTP rides to_thread so parallel workers never stall the loop
    # (first play on a cold server can hold a scene load for ~30s).
    await asyncio.to_thread(panel_field, url, "episode_index", index)
    await asyncio.to_thread(panel_action, url, "play")

    reset_config = {"rgb_resolution": str(cfg["rgb_resolution"])}
    ep = await asyncio.to_thread(
        call_function, url, "env_habitat__reset", {"trigger": "driver"}, reset_config
    )
    instruction = ep["instruction"]

    briefing, skill_md5 = build_briefing(
        instruction, cfg["step_budget"], bare=spec.bare, skill=spec.skill,
        wp=spec.wp, wp_max_moves=cfg.get("wp_max_moves", 30),
    )

    workdir = run_dir / f"workdir_{index}"
    workdir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ctx = EpisodeContext(
        index=index,
        instruction=instruction,
        briefing=briefing,
        first_prompt=FIRST_PROMPT,
        server_url=url,
        bare=spec.bare,
        skill=spec.skill,
        model=spec.model_id,
        max_turns=cfg["max_turns"],
        step_budget=cfg["step_budget"],
        episode_timeout=cfg["episode_timeout"],
        workdir=workdir,
        live_dir=run_dir / f"live_{index}",
        raw_dir=raw_dir,
        persona=spec.persona,
        wp=spec.wp,
        wp_server_url=cfg.get("wp_server") or "",
        wp_max_moves=cfg.get("wp_max_moves", 30),
        extra=dict(cfg.get("extra") or {}),
    )

    sink = EventSink(run_dir / f"episode_{index}.jsonl")
    try:
        sink.emit("episode_meta", {
            "index": index, "episode_id": ep.get("episode_id"),
            "scene_id": ep.get("scene_id"), "instruction": instruction,
            "skill": spec.skill,
        })
        sink.emit("session_inputs", {
            "cell": spec.name,
            "harness": adapter.name,
            "model": spec.model_id,
            "skill": spec.skill,
            "skill_md5": skill_md5,
            "persona": spec.persona,
            "system_prompt": briefing,
            "first_prompt": FIRST_PROMPT,
            "tool_schemas": await bridge_tool_schemas(spec.bare, spec.wp),
            **json_safe(adapter.describe(ctx)),
        })

        outcome = await adapter.run(ctx, sink)

        sink.emit("result", {"result": json_safe({
            "usage": outcome.usage, "cost_usd": outcome.cost_usd,
            "turns": outcome.turns, "error": outcome.error, **outcome.extra,
        })})

        # Evaluate while the trajectory file is still open so the metrics land
        # in the log itself. Driver-side; the agent never sees it.
        metrics: dict[str, Any] = {}
        try:
            metrics_out = await asyncio.to_thread(
                call_function, url, "env_habitat__evaluate", {"trigger": "driver"}
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
            "called_stop": last.get("end_reason") == "stop_called",
            "num_turns": outcome.turns,
            "usage": outcome.usage,
            "total_cost_usd": outcome.cost_usd,
            **json_safe(outcome.extra),
        },
        "wall_sec": round(wall, 1),
    }
    if outcome.error:
        episode["error"] = outcome.error
    return episode


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Truncation counts, infra failure doesn't (口径 2026-07-15):
    an episode with evaluated metrics is scored as-is (cap-hit / cost
    truncations land here, error tag or not); an unevaluated timeout scores
    success=0; other unevaluated errors (account block, server crash) are
    excluded as non-runs and reported via "excluded"."""
    def _engaged(e: dict[str, Any]) -> bool:
        # evaluate() runs even when the session died at spawn (infra failure
        # leaves spawn-position metrics) — only score error-tagged records
        # where the agent actually did something
        a = e.get("agent") or {}
        return bool(a.get("env_steps") or a.get("tool_calls") or a.get("called_stop"))

    scored: list[dict[str, Any]] = []
    for e in episodes:
        if e.get("error") == "rate_limited":
            continue  # subscription throttle, not a navigation result -> excluded
        if e.get("metrics") and (not e.get("error") or _engaged(e)):
            scored.append(e)
        elif e.get("error") == "timeout":
            scored.append({"metrics": {"success": 0.0},
                           "agent": e.get("agent") or {}})
    agg: dict[str, Any] = {"episode_count": len(scored),
                           "excluded": len(episodes) - len(scored)}
    numeric: dict[str, list[float]] = {}
    for rec in scored:
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
        sum(1 for r in scored if r["agent"].get("called_stop")) / max(1, len(scored)), 4
    )
    return agg


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


async def run_cell(
    adapter: HarnessAdapter, spec: CellSpec, servers: list[str],
    episodes_spec: str | None = None, run_name: str | None = None,
    extra: dict[str, Any] | None = None, wp_server: str | None = None,
) -> dict[str, Any]:
    """Run (or resume) one cell. Existing episode records in the run dir's
    summary are kept; requested indices are re-run and replace their records."""
    cfg = dict(STD_FROZEN)
    if spec.max_turns:  # local GPU gets the bigger cap; rented compute takes 100
        cfg["max_turns"] = spec.max_turns
    cfg["extra"] = dict(extra or {})
    cfg["wp_server"] = wp_server if spec.wp else None
    if spec.wp:
        cfg["wp_max_moves"] = WP_MAX_MOVES
        # force substantive thinking (overridable via --set think_budget=N)
        cfg["extra"].setdefault("think_budget", WP_THINK_BUDGET)
    if spec.skill:
        skill_md5 = assert_std_skill_freeze(spec.skill)
        print(f"[std] skill {spec.skill} md5 {skill_md5} (frozen OK)")

    # May raise (bad auth, unpinnable serving context) — that is the point.
    adapter.prepare(spec)

    for url in servers:
        health = requests.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        print(f"[std] {url} healthy: {health.json()['name']}")

    if spec.wp:
        # a wp cell without its predictor would silently degrade — refuse
        if not wp_server:
            raise RuntimeError("wp cell needs --wp-server (waypoint-predictor auto_host)")
        health = requests.get(f"{wp_server}/health", timeout=10)
        health.raise_for_status()
        print(f"[std] {wp_server} healthy: {health.json()['name']} (waypoint predictor)")

    run_name = run_name or spec.name
    run_dir = spec.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    # graceful drain: `touch <run_dir>/DRAIN` (or `stdrun.py drain <cell>`, or
    # send SIGUSR1) asks every worker to finish its current episode, flush, and
    # exit WITHOUT pulling a new one — in-flight work is never cut, un-pulled
    # indices stay pending for the next resume. Lets us stop a batch at an
    # episode boundary instead of hard-killing it (which loses in-flight work).
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
    for url in servers:
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
                           "skill": spec.skill, "persona": spec.persona,
                           "extra": json_safe(cfg["extra"])},
                "servers": servers,
                "run_stats": run_stats,
                # aggregate() applies the board口径 itself (timeout scores 0,
                # rate_limited/infra errors are excluded) — do not pre-filter.
                "aggregate": aggregate(ordered),
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
            # Rate-limit retry: back off OUTSIDE the timed scope and re-run the
            # episode fresh, so the wait never eats the episode's wall-clock
            # budget. Each attempt gets a full episode_timeout; the backoff sleep
            # is not counted against it (the "paused countdown").
            episode = None
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

    final = aggregate(list(episodes.values()))
    print(f"[std] cell complete -> {summary_path}")
    if run_stats:
        print(f"[std] run stats: {json.dumps(run_stats)}")
    print(json.dumps(final, indent=2))
    return final
