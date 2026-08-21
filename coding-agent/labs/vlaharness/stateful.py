from __future__ import annotations

"""The stateful L3 loop.

    plan_route(instruction) ─► legs + termination condition
    repeat:
        execute(next instruction)          the 2B policy rolls out
        judge(state, frames+motors)        fresh session, O(1) context
        state.record_segment(...)          the dispatch becomes durable memory
        state.propose(judgment)            harness applies; model only proposed
        continue / recover / finish

The harness is the only writer of state, and every belief update rides on the
same call that picks the next action — no extra round trips to maintain memory.

Two guards that are the harness's job, not the model's:

  * **`finish` is gated.** A judgment of "finish" that names no landmark, or
    that arrives on a `budget` stop (the policy was cut off mid-movement, it
    decided nothing), is downgraded to `continue`. The model proposes
    termination; the harness rules on it.
  * **The route can always end.** If turns or the step budget run out, STOP is
    fired where the robot stands. An episode that never finishes scores zero,
    which is strictly worse than finishing in the wrong place.

last updated: 2026-08-07
"""

import os
from typing import Any

from vlaharness.judge import judge, plan_route
from vlaharness.state import NavState

VALID = ("continue", "recover", "drive", "finish")

# What counts as "the policy thinks it has arrived at the end of the route".
ARRIVAL_MIN_DISPLACEMENT_M = 2.0


def _arrival_candidate(state: NavState, telemetry: dict[str, Any]) -> bool:
    """The policy stopped itself, after really moving, on the route's last leg."""
    if telemetry.get("stop_reason") != "policy_stop":
        return False
    if (telemetry.get("net_displacement_m") or 0.0) < ARRIVAL_MIN_DISPLACEMENT_M:
        return False
    return state.cursor >= len(state.legs) - 1


def _next_instruction(state: NavState, decision: str, proposed: str | None) -> str | None:
    """What actually goes to the policy. A proposal is used when it is a real
    movement; otherwise fall back to the route."""
    text = (proposed or "").strip()
    bare = text.lower().rstrip(".!").strip()
    if text and bare not in ("stop", "continue", "keep going", "wait", "finish", "null", "none"):
        return text
    if decision == "recover":
        return None                      # a recovery with no instruction is not one
    if state.cursor < len(state.legs):
        return state.current_leg
    return state.terminate or None


def run_stateful_episode(
    tools_impl,
    instruction: str,
    *,
    model: str = "claude-sonnet-5",
    max_turns: int = 24,
    live=None,
    log=None,
    live_dir: str | None = None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {"turns": 0, "tool_calls": [], "planner_text": [],
                             "judgments": []}

    try:
        views0 = tools_impl.current_views()
    except Exception:
        views0 = None
    plan = plan_route(instruction, model=model, views=views0)
    state = NavState(
        instruction=instruction,
        legs=plan["legs"],
        terminate=plan["terminate"],
        step_budget=tools_impl.step_budget,
        path=os.path.join(live_dir, "state.json") if live_dir else None,
    )
    trace["legs"] = state.legs
    trace["terminate"] = state.terminate
    if log is not None:
        log.emit("subgoal_parse", segments=state.legs, terminate=state.terminate,
                 raw=plan.get("raw"))
    if live is not None:
        live.plan(f"route: {' | '.join(state.legs)}\nstop when: {state.terminate}")
    state.save()

    nxt: str | None = state.current_leg or instruction
    pending_drive: list[str] | None = None

    for _ in range(max_turns):
        if tools_impl.episode_over or not nxt:
            break
        trace["turns"] += 1

        drive_actions = pending_drive
        pending_drive = None
        if drive_actions:
            if log is not None:
                log.emit("tool_use", name="drive", input={"actions": drive_actions})
            out = tools_impl.drive(drive_actions)
        else:
            if log is not None:
                log.emit("tool_use", name="execute", input={"sub_instruction": nxt})
            out = tools_impl.execute(nxt)
        clean = {k: v for k, v in out.items() if not k.startswith("_")}
        called = "drive" if drive_actions else "execute"
        trace["tool_calls"].append(
            {"name": called,
             "input": {"actions": drive_actions} if drive_actions
                      else {"sub_instruction": nxt},
             "result": clean}
        )
        if log is not None:
            log.emit("tool_result", name=called, result=clean,
                     frames=log.frames(out) or None)
        if live is not None:
            from vlaharness.live import shrink_for_view

            frames = out.get("_frames_leg") or []
            live.execute(nxt, clean, shrink_for_view(frames[-1]) if frames else None)
        if clean.get("error"):
            break

        actions = out.get("_actions") or []
        verdict, raw = judge(state, out, actions, model=model)
        decision = str(verdict.get("decision") or "").strip().lower()
        if decision not in VALID:
            decision = "recover"

        # ── the harness rules on termination, the model only proposes ──
        saw = str(verdict.get("saw") or "").strip()
        if decision == "finish":
            if clean.get("stop_reason") == "budget":
                decision, verdict["gate"] = "continue", "budget stop is not arrival"
            elif not saw or saw.startswith("<"):
                decision, verdict["gate"] = "continue", "named no landmark"
        elif decision == "recover" and _arrival_candidate(state, clean) \
                and not str(verdict.get("contradiction") or "").strip():
            # The measured failure mode of this arm: the policy walks the route,
            # fires its own STOP at the goal, and the judge — unable to name the
            # landmark in three narrow views — calls it off-route and walks away
            # from a win. Episode 1 lost exactly this way, from a pose the
            # policy-alone baseline scored as a success. Overriding a
            # last-leg arrival now requires naming contradicting evidence.
            decision, verdict["gate"] = "finish", "last-leg arrival, no contradiction named"

        trace["judgments"].append({"decision": decision, **{
            k: verdict.get(k) for k in
            ("leg_verdict", "saw", "current_place", "near_goal", "gate", "next_instruction")}})
        trace["planner_text"].append(f"[{decision}] {saw}")
        if log is not None:
            log.emit("judgment", decision=decision, leg_verdict=verdict.get("leg_verdict"),
                     saw=saw, gate=verdict.get("gate"), raw=raw)
        if live is not None:
            live.plan(f"[{decision}] {saw}")

        state.record_segment(
            asked=("[harness drove] " + clean.get("sub_instruction", "")) if drive_actions else nxt,
            telemetry=clean, actions=actions, verdict=decision, saw=saw)
        accepted = state.propose(verdict)
        if log is not None and accepted:
            log.emit("state_write", accepted=accepted)

        if decision == "finish":
            break
        if decision == "drive":
            acts = [str(a).upper().strip() for a in (verdict.get("actions") or [])]
            acts = [a for a in acts if a in ("MOVE_FORWARD", "FORWARD",
                                             "TURN_LEFT", "TURN_RIGHT")]
            if acts:
                pending_drive = acts
                state.save()
                continue
            decision = "recover"      # a drive with no actions is not a drive
        if decision == "continue" and verdict.get("leg_verdict") == "done":
            state.advance()
        # A recovery loop that has stopped paying for itself is worse than an
        # imperfect route: it burns the episode budget standing still. Three
        # dispatches that did not move the robot means it is wedged, and no
        # re-wording has freed it — the first stateful episode spent 17 of its
        # 18 turns in exactly this loop. Take the next leg instead.
        elif decision == "recover" and state.consecutive_stalls >= 3:
            state.advance()
            if log is not None:
                log.emit("state_write", accepted=["forced_advance: recovery loop broken"])
        state.save()

        nxt = _next_instruction(state, decision, verdict.get("next_instruction"))
        if nxt is None and state.cursor < len(state.legs):
            nxt = state.current_leg

    if not tools_impl.episode_over:
        out = tools_impl.finish()
        forced = trace["turns"] >= max_turns or not trace["judgments"] or \
            trace["judgments"][-1]["decision"] != "finish"
        trace["tool_calls"].append({"name": "finish", "input": {}, "forced": forced})
        trace["forced_finish"] = forced
        if log is not None:
            log.emit("tool_use", name="finish", input={}, forced=forced)
            log.emit("tool_result", name="finish", result=out)
        if live is not None:
            live.finish(forced=forced)

    state.save()
    trace["state"] = {
        "visited": state.visited, "landmarks": state.landmarks,
        "ruled_out": state.ruled_out, "lessons": state.lessons,
        "segments": state.done, "cursor": state.cursor, "legs": state.legs,
    }
    return trace
