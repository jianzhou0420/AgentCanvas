from __future__ import annotations

"""The L3 agent loop, v2 — every termination is typed, every write is sourced.

Same shape as ``agent_loop``: dispatch a subtask, review what came back, rewrite
state, verify before ending. What changed is everything the n=100 run showed was
load-bearing and wrong.

**Termination is three-state, and a forced STOP is not a pass.** The old loop
finished with `forced=True` after the veto allowance ran out and recorded it
next to genuine acceptances, so "the verifier agreed" and "we had to stop
somewhere" were the same row. Now every episode ends in exactly one of
VERIFIED_SUCCESS / VERIFIED_FAILURE / UNRESOLVED_STOP, and only the first counts
as a positive when scoring the verifier.

**A missing next_instruction is a state transition, not a replay.** The old
fallback re-sent the entire mission. In the full run that path fired 0 times; in
the no-drive ablation it fired **371 times across 29 episodes, one of them 23
times**, which is most of why that arm looked worse — it was measuring a
degenerate replay loop, not the absence of motor actions. There is no silent
fallback here: near-goal verifies, wedged recovers, a parse failure retries, and
anything else asks for one constrained replan.

**The arrival gate needs the mission to be nearly over.** It used to fire on any
2-metre policy stop with an observation attached, so "walked into the kitchen"
could be sent to the verifier while the mission ended at the sink. It now also
requires the terminal clause to be active and the model to have flagged
near-goal.

**Corrections are micro closed-loop.** At most two actions, then re-observe.
Forward motion is off by default: episodes where the verifier drove scored 41.2%
where the policy alone scored 64.7% on the same episodes.

**Wedged means not moving AND not seeing anything new.** Net displacement alone
called a deliberate turn a stall.

last updated: 2026-08-10
"""

import os
from typing import Any

from vlaharness.agent_judge2 import (MICRO_ACTIONS, UNRESOLVED_STOP, VERIFIED_FAILURE,
                                     VERIFIED_SUCCESS, adjudicate, bootstrap, review,
                                     verify)
from vlaharness.agent_state2 import STALL_ACT, EvidenceState

VALID = ("continue", "drive", "finish")
DRIVEABLE = ("MOVE_FORWARD", "FORWARD", "TURN_LEFT", "TURN_RIGHT")
TURN_ONLY = ("TURN_LEFT", "TURN_RIGHT")

ARRIVAL_MIN_DISPLACEMENT_M = 2.0
MAX_VETOES = 2
SWEEP_VIEWS = 8
MAX_BLIND_BURSTS = 3          # micro bursts are 2 actions, so this is ≤6 steps
THINK_CAP = 20_000

ABLATIONS = {
    "verify":      "no verification call — the review's `finish` ends the episode",
    "sweep":       "verifier cannot take the free 360 look",
    "drive":       "no motor actions at all: no model drive, no unstick, no verify step",
    "verify-step": "verifier may look but not move (isolates the correction channel)",
    "forward":     "verifier corrections may turn but never walk",
    "back":        "three current views instead of four",
    "frames":      "3 frames per segment instead of 10",
    "evidence":    "state writes accepted without provenance (the old ungated behaviour)",
}


def _clean(raw: Any, allow: tuple[str, ...] = DRIVEABLE, cap: int = 12) -> list[str]:
    return [a for a in (str(x).upper().strip() for x in (raw or [])) if a in allow][:cap]


def _usable(text: Any) -> str | None:
    """An instruction has to be English movement — not an acknowledgement, not a
    motor command, and not the whole mission read back."""
    s = " ".join(str(text or "").split())
    bare = s.lower().rstrip(".!").strip()
    if not s or bare in ("stop", "continue", "keep going", "wait", "finish",
                         "null", "none", "n/a"):
        return None
    if all(w.strip(",.").upper() in DRIVEABLE or w.strip(",.").upper() == "STOP"
           for w in s.split()):
        return None
    return s


def _handles(names: list[str]) -> list[str]:
    return [n[:-4] if n.endswith(".jpg") else n for n in (names or [])]


def run_agent_episode2(
    tools, instruction: str, *, model: str = "claude-sonnet-5", max_turns: int = 24,
    live=None, log=None, live_dir: str | None = None,
    ablate: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    can_drive = "drive" not in ablate
    can_vstep = can_drive and "verify-step" not in ablate
    can_fwd = can_vstep and "forward" not in ablate
    gated = "evidence" not in ablate
    # Derived, not hardcoded: below one rollout of budget the loop cannot
    # dispatch-and-correct, so it has nothing left to offer.
    finish_margin = max(20, int(getattr(tools, "max_steps", 50) * 1.2))

    trace: dict[str, Any] = {
        "turns": 0, "tool_calls": [], "planner_text": [], "judgments": [],
        "payloads": [], "ablate": sorted(ablate),
        "counts": {"planner_calls": 0, "review_calls": 0, "verify_calls": 0,
                   "parse_failures": 0, "forced_stops": 0, "unverified_stops": 0,
                   "verify_looks": 0, "verify_steps": 0,
                   "verify_turn_steps": 0, "verify_forward_steps": 0,
                   "state_writes_accepted": 0, "state_writes_refused": 0},
    }
    C = trace["counts"]

    def emit(kind, **kw):
        if log is not None:
            log.emit(kind, **kw)

    def think(t, **kw):
        if log is not None and t:
            log.emit("thinking", _cap=THINK_CAP, chars=len(t), text=t, **kw)

    def probe():
        d = tools.probe_distance()
        return {"_goal_m": d} if d is not None else {}

    # ── bootstrap ────────────────────────────────────────────────────
    views = tools.current_views()
    vh = _handles(log.frames(views, full=True) if log else [])
    C["planner_calls"] += 1
    opening, raw, th, rep = bootstrap(instruction, views, vh, model)
    trace["payloads"].append({"call": "bootstrap", **rep})
    think(th, call="bootstrap")
    if opening is None:
        C["parse_failures"] += 1

    state = EvidenceState(
        mission=instruction,
        terminate=str((opening or {}).get("terminate") or "").strip(),
        clauses=[str(c).strip() for c in ((opening or {}).get("clauses") or []) if str(c).strip()],
        step_budget=tools.step_budget,
        path=os.path.join(live_dir, "state.json") if live_dir else None,
    )
    ok, bad = state.propose((opening or {}).get("state"),
                            evidence_pool=set(vh) if gated else set(),
                            swept=not gated)
    C["state_writes_accepted"] += len(ok); C["state_writes_refused"] += len(bad)
    nxt = _usable((opening or {}).get("next_instruction")) or instruction.strip()
    trace["bootstrap_failed"] = opening is None or not state.terminate
    trace["terminate"] = state.terminate
    trace["clauses"] = state.clauses
    emit("bootstrap", terminate=state.terminate, clauses=state.clauses,
         first_instruction=nxt, failed=trace["bootstrap_failed"] or None,
         accepted=ok or None, refused=bad or None, payload=rep, raw=raw)
    state.save()

    pending: list[str] | None = None
    unsticks = 0
    prev_hash: int | None = None
    replans = 0

    # ── the loop ─────────────────────────────────────────────────────
    for _ in range(max_turns):
        if tools.episode_over or (not nxt and not pending):
            break
        trace["turns"] += 1
        drive_actions, pending = pending, None
        called = "drive" if drive_actions else "execute"
        tin = {"actions": drive_actions} if drive_actions else {"sub_instruction": nxt}
        emit("tool_use", name=called, input=tin)
        out = tools.drive(drive_actions) if drive_actions else tools.execute(nxt)
        out.update(probe())
        clean = {k: v for k, v in out.items() if not k.startswith("_")}
        clean["_goal_m"] = out.get("_goal_m")
        trace["tool_calls"].append({"name": called, "input": tin, "result": clean})
        names = log.frames(out, full=True) if log else []
        emit("tool_result", name=called, result=clean, frames=names or None,
             ledger=out.get("_ledger"))
        if live is not None:
            from vlaharness.live import shrink_for_view
            f = out.get("_frames_leg") or []
            live.execute(nxt or " ".join(drive_actions or []), clean,
                         shrink_for_view(f[-1]) if f else None)
        if clean.get("error"):
            break

        actions = out.get("_actions") or []
        C["review_calls"] += 1
        verdict, raw, th, rep = review(state, out, actions, _handles(names), model)
        trace["payloads"].append({"call": "review", "turn": trace["turns"], **rep})
        think(th, call="review", turn=trace["turns"])

        if verdict is None:
            C["parse_failures"] += 1
            decision, gate = "continue", "review failed; no gate applied"
            saw = ""
        else:
            decision = str(verdict.get("decision") or "").strip().lower()
            decision = decision if decision in VALID else "continue"
            saw = str((verdict.get("receipt") or {}).get("claim") or "").strip()
            gate = None

        state.record(asked=(" ".join(drive_actions) if drive_actions else nxt),
                     telemetry=clean, actions=actions, decision=decision,
                     driver="harness" if drive_actions else "policy",
                     frame_hash=out.get("_frame_hash"), prev_hash=prev_hash)
        prev_hash = out.get("_frame_hash")

        if verdict is not None:
            ok, bad = state.propose(verdict.get("state"),
                                    evidence_pool=set(_handles(names)) if gated else set(),
                                    swept=not gated)
            C["state_writes_accepted"] += len(ok); C["state_writes_refused"] += len(bad)
            if bad:
                emit("state_refused", refused=bad)
            if ok:
                emit("state_write", accepted=ok)

        # ── arrival gate: now needs the mission to be nearly over ────
        if verdict is not None and gate is None:
            if decision == "finish" and clean.get("stop_reason") == "budget":
                decision, gate = "continue", "budget stop is not arrival"
            elif (decision != "finish"
                  and clean.get("stop_reason") == "policy_stop"
                  and (clean.get("net_displacement_m") or 0.0) >= ARRIVAL_MIN_DISPLACEMENT_M
                  and saw and state.near_goal and state.terminal_clause_active
                  and not str(verdict.get("contradiction") or "").strip()):
                decision, gate = "finish", "policy stop on the terminal clause, near goal"

        trace["judgments"].append({"decision": decision, "gate": gate, "saw": saw,
                                   "verdict": (verdict or {}).get("verdict"),
                                   "receipt": (verdict or {}).get("receipt")})
        emit("judgment", decision=decision, gate=gate, saw=saw,
             verdict=(verdict or {}).get("verdict"),
             receipt=(verdict or {}).get("receipt"), payload=rep, raw=raw)

        if decision == "finish":
            break
        if decision == "drive":
            acts = _clean(verdict.get("actions")) if can_drive else []
            if acts:
                pending, nxt = acts, None
                state.save(); continue
            decision = "continue"

        if can_drive and state.wedged:
            pending, nxt = (["TURN_LEFT"] * 6 if unsticks == 0 else
                            ["TURN_LEFT"] * 12 if unsticks == 1 else ["TURN_RIGHT"] * 6), None
            emit("state_write", accepted=[f"harness unstick #{unsticks + 1}"])
            unsticks += 1
            state.save(); continue

        if (clean.get("steps_remaining") or 0) < finish_margin:
            trace["ended_dispatching"] = "budget below one dispatch"
            emit("state_write", accepted=[f"stopped dispatching: "
                                          f"{clean.get('steps_remaining')} steps left"])
            break

        # ── missing instruction is a transition, never a mission replay ──
        cand = _usable((verdict or {}).get("next_instruction"))
        if cand is None:
            if state.near_goal:
                trace["ended_dispatching"] = "no instruction, near goal → verify"
                emit("state_write", accepted=["missing_next_instruction → verify (near goal)"])
                break
            if verdict is None and replans < 2:
                replans += 1
                emit("state_write", accepted=["missing_next_instruction → retry review"])
                continue                       # re-dispatch nothing; re-review next turn
            if replans < 2:
                replans += 1
                cand = state.next_objective.strip() or None
                emit("state_write",
                     accepted=[f"missing_next_instruction → constrained replan #{replans}"])
            if cand is None:
                trace["ended_dispatching"] = "no usable instruction and no replan left"
                break
        nxt = cand
        state.save()

    # ── termination ──────────────────────────────────────────────────
    outcome, why, swept = UNRESOLVED_STOP, "loop exited", False
    if not tools.episode_over and "verify" in ablate:
        outcome, why = UNRESOLVED_STOP, "verification ablated"
    elif not tools.episode_over and trace["turns"] >= max_turns:
        outcome, why = UNRESOLVED_STOP, "turn budget exhausted before verification"
    elif not tools.episode_over:
        tools.verifying(True)
        sweep = None
        blind_left = MAX_BLIND_BURSTS if can_vstep else 0
        while True:
            views = tools.current_views()
            names = _handles(log.frames(views, full=True) if log else [])
            C["verify_calls"] += 1
            v, raw, th, rep = verify(state, views, names, model, sweep=sweep,
                                     can_look=(not swept) and "sweep" not in ablate)
            trace["payloads"].append({"call": "verify", **rep})
            think(th, call="verify")
            sweep = None
            if v is None:
                C["parse_failures"] += 1

            if v is not None and v.get("look") and not swept and "sweep" not in ablate:
                sweep = tools.look_around(SWEEP_VIEWS)
                swept = True
                C["verify_looks"] += 1
                emit("verify_look", views=SWEEP_VIEWS, env_steps_spent=0,
                     why=v.get("why"), payload=rep, raw=raw)
                continue

            outcome, why = adjudicate(v, swept=swept)
            emit("verification", outcome=outcome, why=why,
                 target_visible=(v or {}).get("target_visible"),
                 relation_holds=(v or {}).get("relation_holds"),
                 close_enough=(v or {}).get("close_enough"),
                 confidence=(v or {}).get("confidence"), swept=swept,
                 goal_m_probe=tools.probe_distance(), payload=rep, raw=raw)
            if outcome != VERIFIED_FAILURE:
                break

            micro = _clean((v or {}).get("step"),
                           allow=DRIVEABLE if can_fwd else TURN_ONLY, cap=MICRO_ACTIONS)
            if micro and blind_left > 0:
                blind_left -= 1
                C["verify_steps"] += 1
                C["verify_turn_steps"] += sum(1 for a in micro if a.startswith("TURN"))
                C["verify_forward_steps"] += sum(1 for a in micro if not a.startswith("TURN"))
                before = tools.probe_distance()
                emit("verify_step", actions=micro, why=(v or {}).get("why"),
                     goal_m_before=before, blind_left=blind_left)
                o = tools.drive(micro)
                after = tools.probe_distance()
                emit("verify_step_result", goal_m_after=after,
                     delta_m=(None if (before is None or after is None) else round(after - before, 3)))
                state.record(asked=f"[adjust] {' '.join(micro)}",
                             telemetry={**{k: x for k, x in o.items() if not k.startswith("_")},
                                        "_goal_m": after},
                             actions=o.get("_actions") or [], decision="adjust",
                             driver="harness")
                swept = False
                state.save(); continue

            state.vetoes += 1
            if state.vetoes >= MAX_VETOES:
                outcome, why = UNRESOLVED_STOP, "veto allowance spent"
                break
            more = _usable((v or {}).get("next_instruction"))
            if not more:
                outcome, why = VERIFIED_FAILURE, why + " (no remedy offered)"
                break
            emit("tool_use", name="execute", input={"sub_instruction": more})
            o = tools.execute(more)
            cl = {k: x for k, x in o.items() if not k.startswith("_")}
            emit("tool_result", name="execute", result=cl,
                 frames=(log.frames(o, full=True) if log else None) or None)
            state.record(asked=more, telemetry={**cl, "_goal_m": tools.probe_distance()},
                         actions=o.get("_actions") or [], decision="verify-retry")
            swept = False
            state.save()
        tools.verifying(False)

    if not tools.episode_over:
        o = tools.finish()
        forced = outcome != VERIFIED_SUCCESS
        C["forced_stops"] += 1 if forced else 0
        C["unverified_stops"] += 1 if outcome == UNRESOLVED_STOP else 0
        trace["tool_calls"].append({"name": "finish", "input": {}, "forced": forced})
        emit("tool_use", name="finish", input={}, forced=forced, outcome=outcome)
        emit("tool_result", name="finish", result=o)
        if live is not None:
            live.finish(forced=forced)
    else:
        trace["ended_without_stop"] = tools.end_reason or "unknown"

    trace["outcome"] = outcome
    trace["outcome_why"] = why
    trace["forced_finish"] = outcome != VERIFIED_SUCCESS
    trace["counts"]["ledger"] = {
        "policy_env_steps": tools.policy_env_steps,
        "harness_drive_steps": tools.harness_drive_steps,
        "verification_steps": tools.verification_steps,
        "render_only_observations": tools.render_only_observations,
    }
    state.save()
    trace["state"] = {
        "terminate": state.terminate, "clauses": state.clauses,
        "current_place": state.current_place, "next_objective": state.next_objective,
        "near_goal": state.near_goal, "vetoes": state.vetoes,
        "beliefs": [b.__dict__ for b in state.beliefs],
        "progress": state.progress, "refused": state.rejected_writes,
        "ledger": state.ledger,
    }
    return trace
