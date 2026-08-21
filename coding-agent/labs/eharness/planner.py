"""Planner — the commander loop. O(1) context by construction.

Every turn's context is REBUILT from scratch: system prompt + one user message
rendered from (instruction, state block, heartbeat, last event). Nothing
accumulates, so the per-turn request size is flat in episode length — the
property the whole delegation design exists to buy (spec §1). The state block
is to the planner what the disk is to Claude Code: it dares to have no
history because the conclusions are already committed (spec §5.2).

The planner has NO camera and NO movement tools (registry as code): it wants
to look → recall stored keyframes, or delegate a worker whose receipt gets
verified. finish() runs the pre-stop chain through the wrapper — the same
gate the sub-agent faces.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from eharness.prompts_eh import PLANNER_SYSTEM, PLANNER_TURN
from eharness.state_block import StateBlock
from eharness.wrapper import HarnessedToolset, _text

PLANNER_TOOLS: list[dict[str, Any]] = [
    {"name": "delegate", "description":
        "Send the worker to execute one concrete subgoal. budget = robot "
        "steps it may spend (20-80 is typical).",
     "input_schema": {"type": "object", "properties": {
         "subgoal": {"type": "string"},
         "budget": {"type": "integer"}},
         "required": ["subgoal"]}},
    {"name": "recall", "description":
        "View stored keyframes by landmark name, frame number, or 'dead end'.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}}, "required": ["query"]}},
    {"name": "verify", "description":
        "Ask the verifier whether the current camera view supports a claim.",
     "input_schema": {"type": "object", "properties": {
         "claim": {"type": "string"}}, "required": ["claim"]}},
    {"name": "finish", "description":
        "Declare the task complete and stop the robot (verified — a rejected "
        "finish returns the reason instead of stopping).",
     "input_schema": {"type": "object", "properties": {}}},
]


def _to_litellm_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": t["input_schema"]}} for t in tools]


class Planner:
    def __init__(
        self,
        wrapper: HarnessedToolset,
        state: StateBlock,
        *,
        model_name: str,
        model_kwargs: dict[str, Any],
        run_subgoal_fn: Callable[..., tuple],   # bound executor.run_subgoal
        emit: Callable[[str, dict], None],
        step_budget: int,
        max_planner_turns: int = 24,
        default_subgoal_budget: int = 60,
        subgoal_turn_cap: int = 30,
        complete_fn: Callable[..., Any] | None = None,   # test injection
    ) -> None:
        self.wrapper = wrapper
        self.state = state
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.run_subgoal_fn = run_subgoal_fn
        self.emit = emit
        self.step_budget = step_budget
        self.max_planner_turns = max_planner_turns
        self.default_subgoal_budget = default_subgoal_budget
        self.subgoal_turn_cap = subgoal_turn_cap
        if complete_fn is None:
            import litellm
            complete_fn = litellm.completion
        self.complete = complete_fn

        self.turns = 0
        self.sub_cost = 0.0
        self.sub_turns = 0
        self.receipts: list[Any] = []
        self._last_event: list[dict[str, Any]] = [_text("(episode start)")]
        self._noparse_streak = 0

    # ── the loop ──

    def run(self) -> dict[str, Any]:
        while not self.wrapper.episode_over:
            if self.turns >= self.max_planner_turns:
                self._finish(forced=True)
                break
            self.turns += 1
            name, args = self._decide()
            if name is None:
                if self._noparse_streak >= 3:
                    self._finish(forced=True)
                    break
                continue
            if name == "delegate":
                self._delegate(args)
            elif name == "recall":
                self._recall(args)
            elif name == "verify":
                self._verify(args)
            elif name == "finish":
                self._finish()
            else:
                self._last_event = [_text(
                    f"unknown tool '{name}'; use delegate/recall/verify/finish")]
        return {
            "planner_turns": self.turns,
            "sub_turns": self.sub_turns,
            "sub_cost": round(self.sub_cost, 4),
            "receipts": len(self.receipts),
            "verify_calls": self.wrapper.verify_calls,
            "guard_trips": self.wrapper.guards.state.trips_total,
            "end_reason": self.wrapper.end_reason,
        }

    # ── one decision (fresh context every time — this IS the O(1) claim) ──

    def _decide(self) -> tuple[str | None, dict[str, Any]]:
        user_content: list[dict[str, Any]] = [_text(PLANNER_TURN.format(
            instruction=self.state.instruction,
            state=self.state.render_full(),
            heartbeat=self.wrapper.heartbeat.render(),
            steps_used=self.wrapper.steps_taken,
            step_budget=self.step_budget,
            turn=self.turns, max_turns=self.max_planner_turns,
            last_event="Last event below." if self._last_event else "",
        ))] + self._last_event
        messages = [{"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": user_content}]
        try:
            resp = self.complete(
                model=self.model_name, messages=messages,
                tools=_to_litellm_tools(PLANNER_TOOLS), **self.model_kwargs)
        except Exception as exc:  # noqa: BLE001
            self.emit("driver_error", {"error": repr(exc), "where": "planner"})
            self._noparse_streak += 1
            return None, {}
        msg = resp.choices[0].message
        if getattr(msg, "content", None):
            self.emit("assistant_text", {"text": str(msg.content)[:2000],
                                         "tier": "planner"})
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            self._noparse_streak += 1
            self._last_event = [_text(
                "Your reply had no tool call. Respond with exactly one tool "
                "call (delegate/recall/verify/finish).")]
            return None, {}
        self._noparse_streak = 0
        call = calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except (ValueError, TypeError):
            args = {}
        self.emit("tool_use", {"id": getattr(call, "id", None),
                               "name": call.function.name, "input": args,
                               "tier": "planner"})
        return call.function.name, args if isinstance(args, dict) else {}

    # ── planner tools ──

    def _delegate(self, args: dict[str, Any]) -> None:
        subgoal = str(args.get("subgoal", "")).strip()
        if not subgoal:
            self._last_event = [_text("delegate needs a subgoal")]
            return
        remaining = max(0, self.step_budget - self.wrapper.steps_taken)
        budget = int(args.get("budget") or self.default_subgoal_budget)
        budget = max(5, min(budget, remaining or 5))   # harness clamps, model proposes
        receipt, cost, turns = self.run_subgoal_fn(
            subgoal=subgoal, budget=budget, turn_cap=self.subgoal_turn_cap)
        self.sub_cost += cost
        self.sub_turns += turns
        # on_subagent_return: a 'reached' claim earns a V1 check before it
        # steers planning (23/30 claimed success is the number this exists for)
        if receipt.claim == "reached" and self.wrapper.judge_model is not None:
            supported, reason = self.wrapper.verify(
                f"The robot has just completed: {subgoal}")
            receipt.verdict = ("pass" if supported else
                               "no verdict" if supported is None else
                               f"FAILED: {reason}")
        before = self.state.snapshot()
        self.receipts.append(receipt)
        self.state.receipts_tail.append({
            "subgoal": subgoal, "claim": receipt.claim,
            "verdict": receipt.verdict, "steps_used": receipt.steps_used,
            "not_done": receipt.not_done})
        del self.state.receipts_tail[:-5]
        self.state.save()
        StateBlock.assert_monotone(before, self.state.snapshot())  # pre_compact
        self.emit("tool_result", {"tier": "planner", "texts": [receipt.render()]})
        self._last_event = [_text(receipt.render())]

    def _recall(self, args: dict[str, Any]) -> None:
        result = self.wrapper.execute("recall", {"query": args.get("query", "")})
        self.emit("tool_result", {"tier": "planner", "texts": [
            p.get("text", "<image>") for p in result.content]})
        self._last_event = list(result.content)   # images ride into next turn

    def _verify(self, args: dict[str, Any]) -> None:
        supported, reason = self.wrapper.verify(str(args.get("claim", "")))
        text = (f"verdict: {'supported' if supported else 'not supported' if supported is not None else 'no verdict'}"
                f" — {reason}")
        self.emit("tool_result", {"tier": "planner", "texts": [text]})
        self._last_event = [_text(text)]

    def _finish(self, forced: bool = False) -> None:
        if forced:
            self.emit("driver_error", {
                "error": "planner cap reached — salvage stop", "where": "planner"})
        name, args = self.wrapper.stop_action()
        result = self.wrapper.execute(name, args)
        info = result.info if isinstance(result.info, dict) else {}
        if info.get("stop_vetoed") and not forced:
            # the judge may compel work, never the conclusion: the veto comes
            # back to the planner as information, and a second finish executes.
            texts = [p.get("text", "") for p in result.content
                     if isinstance(p, dict) and p.get("type") == "text"]
            self._last_event = [_text(
                "finish was withheld by verification: " + " ".join(texts)[:400]
                + " Delegate a correction, or call finish again to override.")]
            return
        if info.get("stop_vetoed") and forced:   # forced salvage: push through
            self.wrapper.execute(name, args)
        self.emit("tool_result", {"tier": "planner", "texts": [
            json.dumps({k: v for k, v in info.items()
                        if isinstance(v, (str, int, float, bool))})]})
