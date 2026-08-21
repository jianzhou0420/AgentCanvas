from __future__ import annotations

"""The L3 agent loop: dispatch a subtask, review what came back, rewrite the
state block, issue the next one. Verify before ending.

    bootstrap(mission, current views)     → termination test, state block, line 1
    repeat:
        execute(line)  or  drive(actions)     the 2B policy, or the harness
        review(state, 14 frames, telemetry)   fresh session, O(1) context
        state.record(...)                     harness writes the facts
        state.rewrite(...)                    the model replaces its own block
        continue / drive / finish
    verify(state, current views)          → the only path to STOP
        · may `look` first: a free 360° sweep at the current pose. No env steps,
          no pose change — a panorama is a render. So "I cannot see it" is
          always a reason to look, never a verdict.
        · may `step`: a few raw motor actions of its own, for a final
          adjustment that is not worth handing back to a policy that cannot
          backtrack.

There is no pre-planned leg list and no cursor. The mission is decomposed
*while it is being carried out*: the next subtask is written each turn from the
state block and what is currently visible, so it can name something the original
sentence never did — "walk to the gap between the bar and the chairs" — and it
can be re-aimed the moment the situation changes. The one thing fixed at the
start is the termination test, because that is what gets verified.

Four harness rules, each one paid for by a measured failure:

  * **`finish` on a `budget` stop is refused.** The policy was cut off mid-motion
    at the step cap; it decided nothing, so there is nothing to ratify.
  * **A policy stop that walked ≥2 m, reported an observation, and named no
    contradiction is an arrival**, even when the model wanted to keep going.
    Episode 1 of the first stateful run was judged off-route at the exact pose
    where the policy alone scored a success, and the correction walked a won
    episode away. The "reported an observation" clause is not decoration: this
    rule turns *not objecting* into *agreeing*, so it must not be satisfiable by
    an answer that never arrived. One did, and finished an episode in a bathroom
    hallway.
  * **Three dispatches with no movement means wedged, not misaddressed.** The
    harness turns the robot itself rather than letting the model spend the
    budget re-phrasing. One run spent 17 of 18 turns in that loop.
  * **A rejection with no remedy attached ends the episode.** "Not there" and no
    instruction, no motor actions, nothing to try is not a plan; looping on it
    burns two model calls per pass and stops in the same place regardless.

last updated: 2026-08-09
"""

import os
from typing import Any

from vlaharness.agent_judge import bootstrap, review, verify
from vlaharness.agent_state import STALL_ACT, AgentState

VALID = ("continue", "drive", "finish")

# A policy stop this far from where the dispatch started is a decision, not a
# stumble — see the arrival rule above.
ARRIVAL_MIN_DISPLACEMENT_M = 2.0
TURN_DEG = 15                      # matches toolset.TURN_ANGLE_DEG

MAX_VETOES = 2                     # rejections before the episode ends anyway
SWEEP_VIEWS = 8                    # frames in the verifier's free 360° look
# Motor bursts nothing reviews before the next decision. The main loop's unstick
# is deliberately NOT on this budget: a review always follows it, which is the
# entire point of it.
MAX_BLIND_BURSTS = 2
# STALL_ACT is imported from agent_state, where it sits beside the threshold at
# which the block warns the model. Warn then act, two numbers, one place.

# Thinking is archived, never replayed. It goes to the event log so a run is
# auditable — how the agent reasoned about a dispatch is the whole point of
# watching this arm — and nowhere near a later call's context.
THINK_CAP = 20_000

DRIVEABLE = ("MOVE_FORWARD", "FORWARD", "TURN_LEFT", "TURN_RIGHT")

# Below one dispatch of env steps, the loop has nothing left to offer. Its whole
# value is dispatch-then-review-then-correct; with less budget than a single
# rollout, the next dispatch cannot be corrected, so handing one out is a blind
# commitment of the remaining steps. Stop and verify instead — verification can
# still make a small bounded adjustment, and the episode is scored on where the
# robot stands, not on how much budget it spent.
FINISH_MARGIN_STEPS = 60

# Each name switches OFF one design choice, so the bundle can be measured one
# axis at a time instead of as a single take-it-or-leave-it arm. Runs record
# which were active, so a result row says what produced it.
ABLATIONS = {
    "verify": "no verification call — the review's `finish` ends the episode",
    "sweep":  "verifier cannot take the free 360 look",
    "drive":  "no motor actions at all: no model drive, no unstick, no verify step",
    "back":   "three current views instead of four (no back view)",
    "frames": "3 frames per segment instead of 10",
}


def unstick(n: int) -> list[str]:
    """The harness's blind manoeuvre when the policy will not move, escalating.

    90°, then 180°, then 90° the other way. A fixed 90° left every time is worse
    than it looks: four of them walk the robot right back to the heading it was
    already wedged on, having spent 24 steps to show it the same four views.
    Escalating guarantees each attempt lands somewhere the last one did not.
    """
    plan = [("TURN_LEFT", 6), ("TURN_LEFT", 12), ("TURN_RIGHT", 6)]
    direction, count = plan[min(n, len(plan) - 1)]
    return [direction] * count


def _clean_actions(raw: Any) -> list[str]:
    return [a for a in (str(x).upper().strip() for x in (raw or [])) if a in DRIVEABLE]


def _motor(tools_impl, actions: list[str], *, purpose: str, label: str,
           state, trace: dict[str, Any], log) -> dict[str, Any]:
    """One blind motor burst, executed and recorded the same way wherever it
    came from. Three places used to do this with three copies of the same eight
    lines and three separate ideas of what to write into the ledger."""
    if log is not None:
        log.emit("tool_use", name="drive", input={"actions": actions})
    out = tools_impl.drive(actions)
    clean = {k: v for k, v in out.items() if not k.startswith("_")}
    trace["tool_calls"].append({"name": "drive", "input": {"actions": actions},
                                "result": clean, "purpose": purpose})
    if log is not None:
        log.emit("tool_result", name="drive", result=clean,
                 frames=log.frames(out, full=True) or None)
    state.record(asked=f"[{label}] {' '.join(actions)}", telemetry=clean,
                 actions=out.get("_actions") or [], decision=label, driver="harness")
    state.save()
    return clean


def _usable(text: Any) -> str | None:
    """A next_instruction has to be an English movement, not an acknowledgement
    and not a motor command.

    The policy is a language model: handed the string "MOVE_FORWARD" it does not
    step forward, it tries to interpret an instruction that says nothing about
    where to go. Measured — it walked 15 steps and 3.24 m on exactly that, in a
    direction nobody chose. Motor intent belongs in `actions`, which the harness
    executes; this channel is for sentences.
    """
    s = " ".join(str(text or "").split())
    bare = s.lower().rstrip(".!").strip()
    if not s or bare in ("stop", "continue", "keep going", "wait", "finish",
                         "null", "none", "n/a"):
        return None
    if all(w.strip(",.").upper() in DRIVEABLE or w.strip(",.").upper() == "STOP"
           for w in s.split()):
        return None                        # a motor sequence in the wrong channel
    return s


def run_agent_episode(
    tools_impl,
    instruction: str,
    *,
    model: str = "claude-sonnet-5",
    max_turns: int = 24,
    live=None,
    log=None,
    live_dir: str | None = None,
    ablate: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    trace: dict[str, Any] = {"turns": 0, "tool_calls": [], "planner_text": [],
                             "judgments": [], "payloads": [],
                             "ablate": sorted(ablate), "cost": {}}
    can_drive = "drive" not in ablate

    # ── bootstrap ─────────────────────────────────────────────────────
    try:
        views = tools_impl.current_views()
    except Exception:
        views = {}
    opening, raw, thinking, report = bootstrap(instruction, views, model=model)
    trace["payloads"].append({"call": "bootstrap", **report})
    if log is not None and thinking:
        log.emit("thinking", _cap=THINK_CAP, call="bootstrap",
                 chars=len(thinking), text=thinking)

    state = AgentState(
        mission=instruction,
        terminate=str(opening.get("terminate") or "").strip(),
        step_budget=tools_impl.step_budget,
        path=os.path.join(live_dir, "state.json") if live_dir else None,
    )
    state.rewrite(opening.get("state"))
    nxt = _usable(opening.get("next_instruction")) or instruction.strip()

    trace["terminate"] = state.terminate
    trace["bootstrap_failed"] = bool(opening.get("parse_failed")) or not state.terminate
    # No `legs` key on purpose: there is no route plan in this arm. The row's
    # `sub_instructions` is the real record of how the mission got decomposed.
    if log is not None:
        log.emit("bootstrap", terminate=state.terminate, first_instruction=nxt,
                 failed=trace["bootstrap_failed"] or None,
                 state={f: getattr(state, f) for f in
                        ("progress", "here", "layout", "ruled_out", "remaining")},
                 payload=report, raw=raw)
    if live is not None:
        live.plan(f"done when: {state.terminate}\nfirst: {nxt}")
    state.save()

    pending_drive: list[str] | None = None
    unsticks = 0

    # ── the loop ──────────────────────────────────────────────────────
    for _ in range(max_turns):
        if tools_impl.episode_over or (not nxt and not pending_drive):
            break
        trace["turns"] += 1

        drive_actions, pending_drive = pending_drive, None
        called = "drive" if drive_actions else "execute"
        tool_input = {"actions": drive_actions} if drive_actions else {"sub_instruction": nxt}
        if log is not None:
            log.emit("tool_use", name=called, input=tool_input)
        out = tools_impl.drive(drive_actions) if drive_actions else tools_impl.execute(nxt)

        clean = {k: v for k, v in out.items() if not k.startswith("_")}
        trace["tool_calls"].append({"name": called, "input": tool_input, "result": clean})
        if log is not None:
            log.emit("tool_result", name=called, result=clean,
                     frames=log.frames(out, full=True) or None)
        if live is not None:
            from vlaharness.live import shrink_for_view

            frames = out.get("_frames_leg") or []
            live.execute(nxt or " ".join(drive_actions or []), clean,
                         shrink_for_view(frames[-1]) if frames else None)
        if clean.get("error"):
            break

        # ── review: judge, rewrite, decide — one call ─────────────────
        actions = out.get("_actions") or []
        verdict, raw, thinking, report = review(state, out, actions, model=model)
        trace["payloads"].append({"call": "review", "turn": trace["turns"], **report})
        if log is not None and thinking:
            log.emit("thinking", _cap=THINK_CAP, call="review", turn=trace["turns"],
                     chars=len(thinking), text=thinking)

        decision = str(verdict.get("decision") or "").strip().lower()
        if decision not in VALID:
            decision = "continue"
        saw = str(verdict.get("saw") or "").strip()
        gate = None

        # The arrival gate below turns "did not object" into "agreed". That is
        # only sound if the model actually looked and actually reported — an
        # answer that never arrived, or one that names nothing it could see,
        # satisfies "no contradiction" vacuously. Measured: an API error on the
        # first review of an episode was promoted straight to finish, in a
        # bathroom hallway. So the gate now needs a *positive* observation, not
        # merely the absence of a negative one.
        reported = bool(saw) and not saw.startswith("<") and not verdict.get("parse_failed")

        if verdict.get("parse_failed"):
            gate = "review failed; no gate applied"
        elif decision == "finish" and clean.get("stop_reason") == "budget":
            decision, gate = "continue", "budget stop is not arrival"
        elif decision != "finish" and reported \
                and clean.get("stop_reason") == "policy_stop" \
                and (clean.get("net_displacement_m") or 0.0) >= ARRIVAL_MIN_DISPLACEMENT_M \
                and not str(verdict.get("contradiction") or "").strip():
            decision, gate = "finish", "policy walked and stopped; observation reported, no contradiction"

        state.record(asked=(" ".join(drive_actions) if drive_actions else nxt),
                     telemetry=clean, actions=actions, decision=decision,
                     driver="harness" if drive_actions else "policy")
        changed = state.rewrite(verdict.get("state"))
        if isinstance(verdict.get("near_goal"), bool):
            state.near_goal = verdict["near_goal"]

        trace["judgments"].append({"decision": decision, "gate": gate, "saw": saw,
                                   "leg_verdict": verdict.get("verdict"),
                                   "next_instruction": verdict.get("next_instruction")})
        trace["planner_text"].append(f"[{decision}] {saw}")
        if log is not None:
            log.emit("judgment", decision=decision, leg_verdict=verdict.get("verdict"),
                     saw=saw, gate=gate, payload=report, raw=raw)
            if changed:
                log.emit("state_write", accepted=changed)
        if live is not None:
            live.plan(f"[{decision}] {saw}")

        if decision == "finish":
            break

        if decision == "drive":
            acts = _clean_actions(verdict.get("actions")) if can_drive else []
            if acts:
                pending_drive, nxt = acts, None
                state.save()
                continue
            decision = "continue"          # a drive with no actions is not one

        # Wedged: re-phrasing has stopped paying for itself, change the pose.
        # The counter clears itself when a harness burst is recorded, so there
        # is no manual reset here to fall out of step with `record`.
        if can_drive and state.consecutive_stalls >= STALL_ACT:
            pending_drive, nxt = unstick(unsticks), None
            if log is not None:
                log.emit("state_write",
                         accepted=[f"harness unstick #{unsticks + 1}: "
                                   f"{len(pending_drive) * TURN_DEG}° left"])
            unsticks += 1
            state.save()
            continue

        # Out of room to dispatch-and-correct: hand over to verification
        # rather than spend the last of the budget on a rollout nobody can
        # review the result of.
        if (clean.get("steps_remaining") or 0) < FINISH_MARGIN_STEPS:
            trace["ended_dispatching"] = "budget below one dispatch"
            if log is not None:
                log.emit("state_write",
                         accepted=[f"stopped dispatching: "
                                   f"{clean.get('steps_remaining')} steps left"])
            break

        # No usable line and no drive — send the mission itself, which is the
        # one phrasing this policy was actually trained on and the only string
        # here known to be a navigable instruction.
        #
        # `state.remaining` used to sit in between, and it was wrong: that field
        # is a description of what is left to do ("find the sink, it should be
        # past the stove"), written to be read by the planner next turn. Handing
        # it to the policy sends a status report where a command belongs — the
        # same category error as sending it "MOVE_FORWARD", one level subtler
        # because it is at least English.
        nxt = _usable(verdict.get("next_instruction")) or instruction.strip()
        state.save()

    # ── verification: the only path to STOP ───────────────────────────
    if not tools_impl.episode_over and trace["turns"] >= max_turns:
        # Out of turns. Verifying here would be theatre — the answer cannot
        # change what happens, and an episode that never fires STOP scores zero.
        out = tools_impl.finish()
        trace["tool_calls"].append({"name": "finish", "input": {}, "forced": True})
        trace["forced_finish"] = True
        trace["verification"] = {"skipped": "turn budget exhausted"}
        if log is not None:
            log.emit("tool_use", name="finish", input={}, forced=True)
            log.emit("tool_result", name="finish", result=out)
        if live is not None:
            live.finish(forced=True)

    if "verify" in ablate and not tools_impl.episode_over:
        # The proposer terminates unchecked. This is the control that says
        # what the separate verification call is worth.
        out = tools_impl.finish()
        trace["tool_calls"].append({"name": "finish", "input": {}, "forced": False})
        trace["verification"] = {"skipped": "ablated"}
        if log is not None:
            log.emit("tool_use", name="finish", input={}, forced=False)
            log.emit("tool_result", name="finish", result=out)
        if live is not None:
            live.finish(forced=False)

    sweep: dict[str, Any] | None = None
    swept = False
    blind_left = MAX_BLIND_BURSTS if can_drive else 0
    while not tools_impl.episode_over:
        try:
            views = tools_impl.current_views()
        except Exception:
            views = {}
        checked, raw, thinking, report = verify(
            state, views, model=model, sweep=sweep,
            can_look=(not swept) and "sweep" not in ablate)
        trace["payloads"].append({"call": "verify", **report})
        if log is not None and thinking:
            log.emit("thinking", _cap=THINK_CAP, call="verify",
                     chars=len(thinking), text=thinking)
        sweep = None

        # ── look before concluding ────────────────────────────────────
        # A panorama is a render, not a move: no env steps, no pose change, no
        # distance to the goal given up. Which makes "I cannot see the landmark"
        # a reason to sweep, never a verdict — a handful of fixed views
        # routinely does not contain it, and concluding from that has already
        # cost this harness finished episodes.
        if checked.get("look") and not swept and "sweep" not in ablate \
                and not tools_impl.episode_over:
            try:
                sweep = tools_impl.look_around(SWEEP_VIEWS)
                swept = True
                if log is not None:
                    log.emit("verify_look", views=SWEEP_VIEWS, env_steps_spent=0,
                             why=checked.get("why"), payload=report, raw=raw)
                continue
            except Exception as exc:  # noqa: BLE001
                sweep = None
                if log is not None:
                    log.emit("verify_look", error=repr(exc)[:160])

        ok = bool(checked.get("satisfied"))

        # What a rejection is actually offering. A veto is only worth spending
        # if it comes with something to do about it: `step` (the verifier's own
        # motors), an English line for the policy, or motor actions it put in
        # the English field by mistake.
        nudge = _clean_actions(checked.get("step"))
        raw_next = checked.get("next_instruction")
        more = _usable(raw_next)
        misfiled = [] if more else _clean_actions(
            str(raw_next or "").replace(",", " ").split())
        # A remedy the harness cannot carry out is not a remedy. Asking for a
        # motor burst with the blind budget already spent leaves the robot
        # exactly where it stands, so counting it as "actionable" buys another
        # verification round that can only produce the same answer.
        actionable = bool(more or ((nudge or misfiled) and blind_left > 0))

        # Terminate when accepted, when the veto allowance is spent, or when the
        # rejection has no remedy attached. That last one used to loop instead:
        # "not there" with nothing to do about it burned the remaining vetoes,
        # two model calls apiece, and finished in the same spot anyway.
        stop_reason_here = ("accepted" if ok else
                            "veto allowance spent" if state.vetoes >= MAX_VETOES else
                            "rejected with no remedy offered" if not actionable else None)

        if log is not None:
            log.emit("verification", satisfied=ok, why=checked.get("why"),
                     swept=swept, actionable=actionable, ended=stop_reason_here,
                     parse_failed=checked.get("parse_failed"), payload=report, raw=raw)
        trace["verification"] = {"satisfied": ok, "why": checked.get("why"),
                                 "swept": swept, "ended": stop_reason_here,
                                 "forced_accept": bool(stop_reason_here) and not ok,
                                 "parse_failed": bool(checked.get("parse_failed"))}

        if stop_reason_here:
            out = tools_impl.finish()
            trace["tool_calls"].append({"name": "finish", "input": {}, "forced": not ok})
            trace["forced_finish"] = not ok
            if log is not None:
                log.emit("tool_use", name="finish", input={}, forced=not ok)
                log.emit("tool_result", name="finish", result=out)
            if live is not None:
                live.finish(forced=not ok)
            break

        # ── the verifier's own hands ──────────────────────────────────
        # `step` is not a veto. "It is right there, two steps left" is a
        # different claim from "we are in the wrong place", and charging a veto
        # for it pushes the model toward the expensive alternative — handing
        # back to a policy that cannot backtrack.
        #
        # These bursts and the misfiled-motor case share one budget because they
        # share the property that matters: the robot moves and *nothing reviews
        # the result before the next decision*. The main loop's unstick is not
        # on this budget — a review always follows it, which is the point of it.
        if nudge and blind_left > 0:
            blind_left -= 1
            swept = False                  # new pose; the old sweep is stale
            if log is not None:
                log.emit("verify_step", actions=nudge, why=checked.get("why"),
                         blind_left=blind_left)
            _motor(tools_impl, nudge, purpose="verify_step", label="adjust",
                   state=state, trace=trace, log=log)
            continue

        state.vetoes += 1
        swept = False
        if more:
            if log is not None:
                log.emit("tool_use", name="execute", input={"sub_instruction": more})
            out = tools_impl.execute(more)
            clean = {k: v for k, v in out.items() if not k.startswith("_")}
            trace["tool_calls"].append({"name": "execute",
                                        "input": {"sub_instruction": more}, "result": clean})
            if log is not None:
                log.emit("tool_result", name="execute", result=clean,
                         frames=log.frames(out, full=True) or None)
            state.record(asked=more, telemetry=clean, actions=out.get("_actions") or [],
                         decision="verify-retry")
            state.save()
        elif misfiled and blind_left > 0:
            blind_left -= 1                # it meant a movement, wrong field
            _motor(tools_impl, misfiled, purpose="verify_correction", label="adjust",
                   state=state, trace=trace, log=log)

    if "finish" not in [c["name"] for c in trace["tool_calls"]]:
        # The env ended the episode underneath us — the step budget ran out
        # mid-rollout. Nothing was chosen and no STOP was fired; the score is
        # wherever the policy happened to be. Say so rather than leaving the
        # row looking like a clean finish.
        trace["ended_without_stop"] = tools_impl.end_reason or "unknown"

    state.save()
    trace["state"] = {f: getattr(state, f) for f in
                      ("progress", "here", "layout", "ruled_out", "remaining",
                       "terminate", "near_goal", "vetoes")}
    trace["state"]["ledger"] = state.ledger
    return trace
