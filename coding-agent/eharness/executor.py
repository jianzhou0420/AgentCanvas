"""Executor slot — run ONE subgoal as a bounded sub-session (插桩点 ②).

The sub-session is a fresh mini NavAgent over the SAME wrapped toolset: its
context is born with the brief and dies with the receipt; whatever it learned
survives only through state.json / keyframes / the receipt (which is the whole
point — spec §5.1: for the executor tier, budget IS compaction).

The executor is pluggable by construction: everything session-specific enters
through ``agent_factory``; the default builds the mini stack (NavToolsModel +
NavAgent). A claude-sdk executor plugs in here later without touching the
wrapper or the planner; a future VLA slots in as ``execute(subinstruction,
max_steps)`` with the outer layer holding termination (vla-as-tool §5 M1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from eharness.prompts_eh import SUB_ADDENDUM
from eharness.receipts import Receipt
from eharness.wrapper import HarnessedToolset


def _jinja_raw(text: str) -> str:
    return "{% raw %}" + text + "{% endraw %}"


class SubgoalEnvironment:
    """mini Environment duck type over the wrapped toolset: ends the inner
    agent's run (Submitted) when the SUBGOAL closes, not just the episode."""

    def __init__(self, wrapper: HarnessedToolset) -> None:
        self.toolset = wrapper

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        from minisweagent.exceptions import Submitted
        result = self.toolset.execute(action.get("tool", ""), action.get("args") or {})
        output = {"content": result.content, "info": result.info}
        info = result.info if isinstance(result.info, dict) else {}
        if info.get("episode_over") or info.get("subgoal_over"):
            if info.get("episode_over"):
                status = info.get("end_reason") or "episode_over"
            elif info.get("preempt"):
                status = "preempted"
            elif info.get("subgoal_budget_exhausted"):
                status = "subgoal_budget_exhausted"
            else:
                status = "subgoal_done"
            raise Submitted({
                "role": "exit",
                "content": json.dumps({"end_reason": status,
                                       **{k: v for k, v in info.items()
                                          if isinstance(v, (str, int, bool, float))}}),
                "extra": {"exit_status": status, "submission": "",
                          "final_info": info},
            })
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def serialize(self) -> dict:
        return {"info": {"config": {"environment_type": "eharness.SubgoalEnvironment"}}}


def default_agent_factory(*, wrapper: HarnessedToolset, system: str, task: str,
                          model_name: str, model_knobs: dict[str, Any],
                          turn_cap: int, output_path: Path,
                          event_hook: Callable[[str, dict], None]):
    """The mini executor: NavToolsModel + NavAgent over SubgoalEnvironment.
    Imports resolve because harnesses.mini_swe put MINI_DIR on sys.path."""
    from model import NavToolsModel
    from nav_agent import NavAgent

    env = SubgoalEnvironment(wrapper)
    model = NavToolsModel(
        model_name=model_name,
        tools=wrapper.tool_schemas(),
        image_window=model_knobs.get("image_window", 0),
        set_cache_control=model_knobs.get("set_cache_control"),
        model_kwargs=model_knobs.get("model_kwargs", {"drop_params": True}),
        cost_tracking=model_knobs.get("cost_tracking", "default"),
    )
    agent = NavAgent(
        model, env,
        system_template=_jinja_raw(system),
        instance_template=_jinja_raw(task),
        step_limit=turn_cap,
        cost_limit=model_knobs.get("cost_limit", 5.0),
        output_path=output_path,
        event_hook=event_hook,
    )
    return agent


def run_subgoal(
    wrapper: HarnessedToolset,
    *,
    subgoal: str,
    budget: int,
    briefing: str,
    model_name: str,
    model_knobs: dict[str, Any],
    turn_cap: int,
    output_path: Path,
    event_hook: Callable[[str, dict], None],
    agent_factory: Callable[..., Any] = default_agent_factory,
) -> tuple[Receipt, float, int]:
    """Run one sub-session; returns (receipt, cost_usd, turns)."""
    wrapper.begin_subgoal(subgoal, budget)
    system = briefing + SUB_ADDENDUM.format(subgoal=subgoal, budget=budget)
    agent = agent_factory(
        wrapper=wrapper, system=system, task=subgoal, model_name=model_name,
        model_knobs=model_knobs, turn_cap=turn_cap, output_path=output_path,
        event_hook=event_hook)

    exit_status = "subgoal_turn_cap"
    try:
        exit_info = agent.run(task=subgoal)
        exit_status = (exit_info or {}).get("exit_status") or exit_status
    except Exception as exc:  # noqa: BLE001 — a dead sub-session becomes a receipt
        exit_status = f"sub_error:{type(exc).__name__}"
        event_hook("driver_error", {"error": repr(exc), "where": "sub-session"})

    receipt = wrapper.end_subgoal(exit_status)
    receipt.turns_used = getattr(agent, "n_calls", 0)
    return receipt, float(getattr(agent, "cost", 0.0) or 0.0), receipt.turns_used
