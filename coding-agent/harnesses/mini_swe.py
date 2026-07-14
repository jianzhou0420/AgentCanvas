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
import os
import sys
from pathlib import Path
from typing import Any

from driver import EpisodeContext, EventSink, SessionOutcome

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_DIR = REPO_ROOT / "beta-react-harness"

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
if str(MINI_DIR) not in sys.path:
    sys.path.insert(0, str(MINI_DIR))


def _jinja_raw(text: str) -> str:
    return "{% raw %}" + text + "{% endraw %}"


class MiniSweAdapter:
    name = "mini-swe"

    def __init__(self) -> None:
        self.inherent: dict[str, Any] = {
            "auth": "provider API key (litellm billing)",
            "thinking": "not configured (plain completion)",
            "turn_cap": "hard (agent step_limit)",
        }

    def prepare(self) -> None:
        try:
            import minisweagent
            self.inherent["mini_version"] = getattr(minisweagent, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _is_local(ctx: EpisodeContext) -> bool:
        """Locally served model: an explicit api_base (ollama / vLLM / any
        OpenAI-compatible server) or a local litellm route prefix."""
        return bool(ctx.extra.get("api_base")) or any(
            ctx.model.startswith(p) for p in ("ollama", "hosted_vllm/", "openai/")
        )

    def _check_auth(self, ctx: EpisodeContext) -> None:
        if self._is_local(ctx):
            return  # local server — no provider key involved
        model = ctx.model.lower()
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

        self._check_auth(ctx)
        knobs = self._knobs(ctx)
        env = HabitatEnvironment(
            server_url=ctx.server_url,
            bare=ctx.bare,
            step_budget=ctx.step_budget,
            turn_budget=ctx.turn_budget,
            live_dir=str(ctx.live_dir),
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
