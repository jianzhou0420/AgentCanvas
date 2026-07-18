"""mini-swe-agent adapter — the open ReAct loop.

Session block ported from beta-react-harness/run_episodes.py; the harness
modules themselves (toolset / model / env / nav_agent) are imported from
beta-react-harness — they stay the single implementation, still gated by
check_equivalence.py. litellm bills through the provider API key.

Prompt delivery note: the shared driver hands us the RENDERED briefing;
mini's DefaultAgent runs its templates through jinja, so we wrap the text in
{% raw %} — rendering is then the identity function and the system prompt
stays byte-equal to the SDK cell (modulo jinja's stripped trailing newline,
the already-recorded difference).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from driver import EpisodeContext, EventSink, SessionOutcome

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_DIR = REPO_ROOT / "beta-react-harness"

# Serving contract for locally-hosted models. ollama's default context is 4096:
# every request past it is silently truncated, the run completes, the numbers
# look plausible and are garbage. So the adapter — not a wrapper script the
# standard path can be run without — owns the server and pins this.
SERVE_CTX = 131072
OLLAMA_URL = "http://127.0.0.1:11434"
SERVE_LOG = REPO_ROOT / "outputs" / "beta-react-harness" / "_ollama_serve.log"

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
if str(MINI_DIR) not in sys.path:
    sys.path.insert(0, str(MINI_DIR))


def _jinja_raw(text: str) -> str:
    return "{% raw %}" + text + "{% endraw %}"


def _ollama_bin() -> str | None:
    return shutil.which("ollama") or next(
        (str(p) for p in [Path.home() / "ollama" / "bin" / "ollama"] if p.exists()), None
    )


def _get(url: str, timeout: float = 2.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:  # noqa: BLE001
        return None


def _post(url: str, payload: dict, timeout: float = 10.0) -> dict | None:
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001
        return None


def _ollama_pid() -> str | None:
    pids = subprocess.run(["pgrep", "-f", "ollama serve"],
                          capture_output=True, text=True).stdout.split()
    return pids[0] if pids else None


def _serving_ctx_of_running_ollama() -> int | None:
    """OLLAMA_CONTEXT_LENGTH the live server was started with. None = no server;
    0 = server up but the variable is unset, i.e. the 4096 truncation default."""
    if _get(f"{OLLAMA_URL}/api/version") is None:
        return None
    pid = _ollama_pid()
    if pid is None:
        return 0
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return 0
    for kv in environ:
        if kv.startswith(b"OLLAMA_CONTEXT_LENGTH="):
            return int(kv.split(b"=", 1)[1])
    return 0


def _we_own_the_log() -> bool:
    """Is the live server writing to OUR log? If not we cannot slice its exact
    per-request token counts, and finalize() would silently return nothing —
    the same silent-degradation class as an unpinned context. So this forces a
    restart rather than quietly shipping a run with no token accounting."""
    pid = _ollama_pid()
    if pid is None:
        return False
    try:
        return Path(f"/proc/{pid}/fd/1").resolve() == SERVE_LOG.resolve()
    except OSError:
        return False


class MiniSweAdapter:
    name = "mini-swe"

    def __init__(self) -> None:
        self.inherent: dict[str, Any] = {
            "auth": "provider API key (litellm billing)",
            "thinking": "not configured (plain completion)",
            "turn_cap": "hard (agent step_limit)",
            # mini's philosophy is "pack the full context every call"; the ONLY
            # bolt-on is image_window (model.py) = keep newest K frames, collapse
            # older to a text stub. K=0 (the current cell default) → full visual
            # history → linear request growth → the Anthropic cells can breach the
            # 32 MB request cap. This is a harness-inherent difference vs the SDK's
            # recent-window auto-eviction. See docs developer-guide/coding-agent/
            # harness-notes.
            "vision_context": "image_window=K newest frames (default 0 = full history, unbounded)",
        }
        self._log_offset = 0  # where this cell's stretch of the serve log starts

    def prepare(self, spec) -> None:
        # wp reaches the env through toolset.WaypointToolSet (in-process mirror
        # of wp_bridge.py, gated by check_equivalence.py) — no MCP subprocess,
        # same as bare/nav here. The predictor auto_host is validated driver-side.
        try:
            import minisweagent
            self.inherent["mini_version"] = getattr(minisweagent, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
        self._check_auth(spec.model_id, spec.extra_dict)
        if spec.model_id.startswith("ollama"):
            self._ensure_ollama(spec.model_id.split("/", 1)[1])

    # ── serving stack (local models only) ──

    def _ensure_ollama(self, model_tag: str) -> None:
        binary = _ollama_bin()
        if binary is None:
            raise RuntimeError("ollama binary not found — a local cell cannot run")

        if _serving_ctx_of_running_ollama() != SERVE_CTX or not _we_own_the_log():
            subprocess.run(["pkill", "-f", "ollama serve"], check=False)
            for _ in range(30):
                if _get(f"{OLLAMA_URL}/api/version", 1.0) is None:
                    break
                time.sleep(1)
            SERVE_LOG.parent.mkdir(parents=True, exist_ok=True)
            log = SERVE_LOG.open("a")
            subprocess.Popen(  # noqa: S603
                [binary, "serve"], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OLLAMA_CONTEXT_LENGTH": str(SERVE_CTX)},
                start_new_session=True)
            for _ in range(60):
                if _get(f"{OLLAMA_URL}/api/version", 1.0) is not None:
                    break
                time.sleep(1)

        got = _serving_ctx_of_running_ollama()
        if got != SERVE_CTX:
            raise RuntimeError(
                f"ollama is serving at context {got} (need {SERVE_CTX}) — refusing to "
                "run: past its window ollama truncates silently and the run would look "
                "fine while being worthless"
            )
        print(f"[mini] ollama serving at {SERVE_CTX} ctx (pinned)")

        self._log_offset = SERVE_LOG.stat().st_size if SERVE_LOG.exists() else 0

        # The serving layer, read back FROM THE SERVER — the only place it is real.
        show = _post(f"{OLLAMA_URL}/api/show", {"model": model_tag}) or {}
        info = show.get("model_info") or {}
        sampling: dict[str, str] = {}
        for line in (show.get("parameters") or "").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                sampling[parts[0]] = parts[1].strip()

        # Reproducibility guard, same principle as the context pin: verify, don't
        # trust. litellm's ollama route does not support presence_penalty and
        # drop_params=True makes it vanish silently, so sampling the harness
        # "sets" may never reach the server. The Modelfile is therefore the single
        # source of truth — and this asserts what it actually says.
        deterministic = sampling.get("temperature") == "0" or "seed" in sampling
        if not deterministic:
            raise RuntimeError(
                f"{model_tag} serves with sampling {sampling or '(ollama defaults)'} "
                "— neither temperature=0 nor a seed is pinned, so every episode is a "
                "different random sample and no result here is reproducible. Bake the "
                "sampling into a Modelfile (`ollama create <tag>-greedy -f ...`): the "
                "harness cannot pin presence_penalty through litellm."
            )
        print(f"[mini] sampling pinned server-side: {sampling}")

        self.inherent["serving"] = {
            "backend": "ollama",
            "version": subprocess.run([binary, "--version"], capture_output=True,
                                      text=True).stdout.strip(),
            "served_context": SERVE_CTX,
            "native_context": next(
                (v for k, v in info.items() if k.endswith("context_length")), None),
            "quantization": (show.get("details") or {}).get("quantization_level"),
            "sampling": sampling,
            "deterministic": deterministic,
            "note": "sampling comes from the MODELFILE (read back from /api/show), "
                    "not from the harness — litellm's ollama route drops "
                    "presence_penalty, so pinning it client-side would be a no-op",
        }

    def finalize(self, run_dir: Path) -> dict[str, Any]:
        """Exact per-request prompt-token counts, straight from llama.cpp, for
        this cell's stretch of the serve log. Proves the context window was never
        crossed — the claim nothing else in the stack can make. Says so out loud
        when it cannot be produced; a missing audit must not look like a clean one."""
        if "serving" not in self.inherent:
            return {}  # not a local cell — nothing to audit
        if not SERVE_LOG.exists():
            return {"prompt_tokens": {"unavailable": f"{SERVE_LOG} missing"}}
        with SERVE_LOG.open(errors="ignore") as fh:
            fh.seek(self._log_offset)
            toks = [int(m.group(1)) for line in fh
                    if (m := re.search(r"task\.n_tokens = (\d+)", line))]
        if not toks:
            return {"prompt_tokens": {
                "unavailable": "no `task.n_tokens` lines in this cell's slice of "
                               "the serve log — the server is not the one we started"}}
        return {"prompt_tokens": {
            "llm_requests": len(toks), "peak": max(toks),
            "mean": round(sum(toks) / len(toks)), "sum": sum(toks),
            "served_context": SERVE_CTX,
            "peak_pct_of_window": round(100 * max(toks) / SERVE_CTX, 1),
            "over_32k": sum(1 for t in toks if t > 32768),
            "over_64k": sum(1 for t in toks if t > 65536),
        }}

    # ── auth ──

    @staticmethod
    def _is_local_model(model: str, extra: dict) -> bool:
        """Locally served model: an explicit api_base (ollama / vLLM / any
        OpenAI-compatible server) or a local litellm route prefix."""
        return bool(extra.get("api_base")) or any(
            model.startswith(p) for p in ("ollama", "hosted_vllm/", "openai/")
        )

    def _is_local(self, ctx: EpisodeContext) -> bool:
        return self._is_local_model(ctx.model, ctx.extra)

    def _check_auth(self, model_id: str, extra: dict) -> None:
        if self._is_local_model(model_id, extra):
            return  # local server — no provider key involved
        model = model_id.lower()
        if any(s in model for s in ("anthropic", "claude")) \
                and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — litellm needs it (API billing)"
            )
        if model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set — litellm needs it (API billing)"
            )

    def _knobs(self, ctx: EpisodeContext) -> dict[str, Any]:
        set_cache_control = (
            "default_end"
            if any(s in (ctx.model or "").lower()
                   for s in ["anthropic", "sonnet", "opus", "claude"])
            else None
        )
        model_kwargs: dict[str, Any] = {"drop_params": True}
        if ctx.extra.get("api_base"):
            model_kwargs["api_base"] = ctx.extra["api_base"]
        return {
            "cost_limit": ctx.extra.get("cost_limit", 5.0),
            "image_window": ctx.extra.get("image_window", 0),
            "set_cache_control": set_cache_control,
            "model_kwargs": model_kwargs,
            # local servers have no litellm price entry — a hard cost lookup
            # would kill the run; tokens still land in the trajectory
            "cost_tracking": "ignore_errors" if self._is_local(ctx) else "default",
        }

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        knobs = self._knobs(ctx)
        return {
            "agent_config": {
                "step_limit": ctx.max_turns,
                "cost_limit": knobs["cost_limit"],
                "wall_time_limit_seconds": ctx.episode_timeout,
            },
            "model_config": {"image_window": knobs["image_window"],
                             "set_cache_control": knobs["set_cache_control"],
                             "model_kwargs": knobs["model_kwargs"],
                             "cost_tracking": knobs["cost_tracking"]},
        }

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        # mini's loop is synchronous; ride a thread so parallel workers overlap.
        return await asyncio.to_thread(self._run_sync, ctx, sink)

    def _run_sync(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        from env import HabitatEnvironment
        from model import NavToolsModel
        from nav_agent import NavAgent

        knobs = self._knobs(ctx)  # auth + serving already settled in prepare()
        env = HabitatEnvironment(
            server_url=ctx.server_url,
            bare=ctx.bare,
            step_budget=ctx.step_budget,
            turn_budget=ctx.turn_budget,
            live_dir=str(ctx.live_dir),
            wp=ctx.wp,
            wp_server_url=ctx.wp_server_url,
            wp_max_moves=ctx.wp_max_moves,
            wp_predict_fn=ctx.extra.get("wp_predict_fn", "smartway_waypoint__predict"),
        )
        model = NavToolsModel(
            model_name=ctx.model,
            tools=env.toolset.tool_schemas(),
            image_window=knobs["image_window"],
            set_cache_control=knobs["set_cache_control"],
            model_kwargs=knobs["model_kwargs"],
            cost_tracking=knobs["cost_tracking"],
        )
        agent = NavAgent(
            model,
            env,
            system_template=_jinja_raw(ctx.briefing),
            instance_template=_jinja_raw(ctx.first_prompt),
            step_limit=ctx.max_turns,
            cost_limit=knobs["cost_limit"],
            wall_time_limit_seconds=ctx.episode_timeout,
            output_path=ctx.raw_dir / f"episode_{ctx.index}.traj.json",
            event_hook=sink.emit,
        )

        error: str | None = None
        exit_info: dict[str, Any] = {}
        try:
            exit_info = agent.run(task=ctx.instruction)
        except Exception as exc:  # noqa: BLE001 — run() already logged + saved
            error = repr(exc)
            sink.emit("driver_error", {"error": error})

        return SessionOutcome(
            usage=None,  # litellm cost tracking only; no per-token usage dump
            cost_usd=round(agent.cost, 4),
            turns=agent.n_calls,
            error=error,
            extra={
                "exit_status": exit_info.get("exit_status"),
                "toolset_counts": dict(env.toolset.calls_by_tool),
                "toolset_env_steps": env.toolset.steps_taken,
                "toolset_end_reason": env.toolset.end_reason,
            },
        )
