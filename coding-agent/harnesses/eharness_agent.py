"""eharness adapter — the embodied-harness shell over the mini executor.

Serving, auth, and the token audit are the mini adapter's (composition, not
copy): one ollama instance serves planner, sub-agent, AND judge — same model,
stateless HTTP, so "two tiers" costs zero extra VRAM (user ruling 2026-08-04:
the tiers separate context and permissions, not weights).

Artifact layout matches every other harness (episode_{i}.jsonl / raw/ /
live_{i}/) plus the harness's own memory files inside live_{i}/:
state.json · keyframes.jsonl · receipts.jsonl · heartbeat.json · memory.md.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from driver import EpisodeContext, EventSink, SessionOutcome
from harnesses.mini_swe import MiniSweAdapter  # also puts mini/ on sys.path


# §16.5: the private splitters (SPLIT_SYSTEM / split_instruction_llm /
# split_instruction) moved into eharness.resolver — the ONE parser every
# executor reads. Nothing here parses instructions anymore.


class EharnessAdapter:
    name = "eharness"

    def __init__(self) -> None:
        self._mini = MiniSweAdapter()
        self.inherent: dict[str, Any] = {
            "shell": "eharness (two-tier planner/sub-agent, shared model)",
            "verification": "V0 frame-hash guards + V1 clean-context judge "
                            "(pre_stop, receipt, on-demand)",
            "memory": "state.json + keyframes.jsonl + memory.md under live_dir "
                      "(disk is the source of truth; no coordinates)",
            "context": "planner O(1) fresh-per-turn; sub-sessions bounded by "
                       "subgoal budget; state injected via tool results",
            # §18-7: paradigm honesty — two_tier is its OWN experimental
            # axis, not the multiturn contract wearing a different name.
            # Cells pin image_window=6 explicitly (cells.py EH_EXTRA), so
            # two-tier workers run a last-6 image window and the planner is
            # a fresh single-turn request each round; neither is the
            # 12-cycle compiler, and runs must be labelled accordingly.
            "two_tier_context": (
                "worker last-K image window (cells pin K=6 — a 6-frame "
                "ablation, NOT the 12-cycle contract); planner fresh-per-"
                "turn O(1) with only the latest event — NOT multiturn"),
        }

    # serving/auth/audit ride the mini adapter unchanged
    def prepare(self, spec) -> None:
        self._mini.prepare(spec)
        self.inherent["executor"] = dict(self._mini.inherent)

    def refine_instruction(self, *, model: str, system: str, user: str,
                           extra: dict | None = None) -> str | None:
        """§17: ONE short text-only call with the run's own nav model —
        the same transport the resolver uses (local ollama goes native
        with think off; anything else through litellm). Returns the raw
        model text; None on any failure (the caller falls back whole)."""
        try:
            from eharness.resolver import _default_transport
            kwargs = {}
            if extra and extra.get("api_base"):
                kwargs["api_base"] = str(extra["api_base"])
            # §20.1: the Refiner fails FAST — 45 s, not the resolver's 300
            return _default_transport(model, kwargs or None, system, user,
                                      timeout_s=45.0)
        except Exception:  # noqa: BLE001 — a preprocessor never costs a run
            return None

    def finalize(self, run_dir: Path) -> dict[str, Any]:
        return self._mini.finalize(run_dir)

    def _knobs(self, ctx: EpisodeContext) -> dict[str, Any]:
        base = self._mini._knobs(ctx)  # noqa: SLF001 — deliberate composition
        return {
            **base,
            # bounded visual context for sub-sessions (arm B′ by default —
            # last-K with the event-stub upgrade landing in model layer later)
            "image_window": int(ctx.extra.get("image_window", 6)),
            "subgoal_budget": int(ctx.extra.get("subgoal_budget", 60)),
            "subgoal_turn_cap": int(ctx.extra.get("subgoal_turn_cap", 30)),
            "planner_max_turns": int(ctx.extra.get("planner_max_turns", 24)),
            "verify": str(ctx.extra.get("verify", "1")) not in ("0", "false"),
            # solo (default, 2026-08-04 solo-harness ruling) | two_tier (S4 arm)
            "paradigm": str(ctx.extra.get("paradigm", "solo")),
            "compact_at": int(ctx.extra.get("compact_at", 12000)),
            "auto_view": str(ctx.extra.get("auto_view", "1")) not in ("0", "false"),
            "verify_moves": str(ctx.extra.get("verify_moves", "1")) not in ("0", "false"),
            "tail_msgs": int(ctx.extra.get("tail_msgs", 24)),
            "reattach_k": int(ctx.extra.get("reattach_k", 3)),
            # §12 context contract knobs (legacy_l2=1 restores the old cut)
            "history_frames": int(ctx.extra.get("history_frames", 12)),
            "recent_turns": int(ctx.extra.get("recent_turns", 6)),
            "legacy_l2": str(ctx.extra.get("legacy_l2", "0")) in ("1", "true"),
            # thinking ON = same footing as every pre-v2.6 run; 0 is the
            # 27× judge speed lever, kept as an ablation knob
            "judge_think": str(ctx.extra.get("judge_think", "1")) not in ("0", "false"),
            # 0 = off (every run before 2026-08-05); >0 blocks a move every
            # N moves until the model states segment-done / stop-condition
            "reflect_every": int(ctx.extra.get("reflect_every", 0)),
        }

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {"eharness_config": self._knobs(ctx)}

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        return await asyncio.to_thread(self._run_sync, ctx, sink)

    def _run_sync(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        from env import HabitatEnvironment  # mini/ is on sys.path via mini_swe

        from eharness.executor import run_subgoal
        from eharness.planner import Planner
        from eharness.state_block import StateBlock
        from eharness.wrapper import HarnessedToolset

        knobs = self._knobs(ctx)
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
            hybrid=ctx.hybrid,
            # dwp rides the `extra` channel rather than a new EpisodeContext
            # field: the frozen cells' contract stays untouched, and the knob
            # is set the same way judge_think and eh_bridge are.
            dwp=str(ctx.extra.get("dwp", "")) not in ("", "0", "false"),
            dwp_sam_url=str(ctx.extra.get("sam_url", "")),
            dwp_max_moves=int(ctx.extra.get("dwp_max_moves", 60)),
            dwp_landmark_every=int(ctx.extra.get("landmark_every", 3)),
            dwp_range_cap_m=float(ctx.extra.get("range_cap_m", 0.0)),
            dwp_places=int(ctx.extra.get("places", 0)),
            instruction=ctx.instruction,
        )

        state = StateBlock.load(ctx.live_dir / "state.json")
        state.instruction = ctx.instruction
        # §16.5: the ONE resolver — the driver already wrote the episode
        # record into the live dir; load it (resolve only as a fallback for
        # driver-less invocations). §16.4: the record guarantees ≥1
        # actionable segment, so near_stop can never arm at frame 0. Gate
        # on MISSING SEGMENTS alone (review P2), and rehydrate the record's
        # landmarks even when the state already carries a route (review P1:
        # the else-branch silently degraded SAM queries to scraped words).
        from eharness.resolver import load_record, resolve_or_load
        if not state.sub_instructions:
            rec = resolve_or_load(ctx.live_dir, ctx.instruction,
                                  model_name=ctx.model,
                                  model_kwargs=knobs["model_kwargs"])
            state.sub_instructions = rec.segments
            state.terminate = rec.terminate
            sink.emit("subgoal_parse", {"segments": state.sub_instructions,
                                        "terminate": state.terminate,
                                        "landmarks": rec.landmarks,
                                        "source": rec.source})
        else:
            rec = load_record(ctx.live_dir)
        landmark_words = (rec.landmarks if rec is not None
                          and rec.instruction == ctx.instruction else [])
        # The depth toolset grounds landmark nouns, which only exist after the
        # split. Detector-friendly phrases from the splitter beat words scraped
        # out of the instruction: SAM 3 returns nothing for "bar" and finds the
        # counter immediately for "bar counter" (measured on EP0, same frame).
        if hasattr(env.toolset, "set_route"):
            env.toolset.set_route(list(state.sub_instructions), state.terminate,
                                  landmarks=landmark_words)
        # One free 360° look before the first move: the detector's baseline of
        # what was HERE at the start. Costs zero env steps (observe_panorama
        # renders without stepping the sim) and gives the verifier a step-0
        # ledger entry, so "has it passed the pool?" stops being re-decided
        # from pixels every time.
        opening_parts: list = []
        # §16.2: the RESIDENT bootstrap first — frame 0 fused into the
        # accumulated map, candidates proposed on THIS toolset, snapshot
        # and artifact published. Geometry alone suffices: with SAM off or
        # landmarks empty the model still opens with current RGB + map +
        # candidates instead of deciding blind.
        booted = False
        if hasattr(env.toolset, "bootstrap"):
            try:
                boot = env.toolset.bootstrap()
                opening_parts = list(boot.get("parts") or [])
                booted = bool(opening_parts)
                sink.emit("bootstrap", {
                    "candidates": len(boot["artifact"]["candidates"]),
                    "candidate_epoch": boot["artifact"]["candidate_epoch"],
                    "sensor_frame": boot["artifact"]["sensor_frame"],
                    "images": len(boot["artifact"].get("images") or {})})
            except Exception as exc:  # noqa: BLE001 — degrade to the survey
                sink.emit("bootstrap", {"error": repr(exc)})
        if hasattr(env.toolset, "opening_survey"):
            survey = (env.toolset.opening_survey(skip_frontal=True)
                      if booted else env.toolset.opening_survey())
            if survey.get("seen"):
                sink.emit("opening_survey", survey)
            # speak it before the first decision — otherwise turn 1 is taken
            # blind (EP0: a 45° turn into a wall, first logged frame at step 3)
            if survey.get("sentence"):
                state.surroundings = survey["sentence"]
                state.save()
            # APPEND — the bootstrap's frontal+map lead the first message;
            # the survey adds the panorama views behind them
            opening_parts = opening_parts + (survey.get("parts") or [])
        state.step_budget = ctx.step_budget
        state.save()

        wrapper = HarnessedToolset(
            env.toolset,
            state=state,
            live_dir=ctx.live_dir,
            judge_model=ctx.model if knobs["verify"] else None,
            judge_kwargs={**knobs["model_kwargs"],
                          "judge_think": knobs["judge_think"]},
            emit=sink.emit,
            subgoal_turn_cap=knobs["subgoal_turn_cap"],
            # solo: the SoloModel owns the ephemeral state message; appending
            # renders to tool results would duplicate (and persist stale) state
            inject_state=(knobs["paradigm"] != "solo"),
            # §3.4: the mini/solo path stores image REFS and the compiler
            # materialises the survivors; the SDK bridge keeps inline images
            # (its provider needs them in the tool result itself)
            store_refs=(knobs["paradigm"] == "solo"),
            auto_view=knobs["auto_view"],
            verify_moves=knobs["verify_moves"],
            reflect_every=knobs["reflect_every"],
        )

        if knobs["paradigm"] == "solo":
            return self._run_solo(ctx, sink, knobs, wrapper, state,
                                  opening_parts=opening_parts)

        sub_raw = ctx.raw_dir / f"episode_{ctx.index}_subgoals"
        sub_raw.mkdir(parents=True, exist_ok=True)
        counter = {"k": 0}

        def run_one(*, subgoal: str, budget: int, turn_cap: int):
            counter["k"] += 1
            return run_subgoal(
                wrapper,
                subgoal=subgoal, budget=budget,
                briefing=ctx.briefing,
                model_name=ctx.model,
                model_knobs={
                    "image_window": knobs["image_window"],
                    "set_cache_control": knobs["set_cache_control"],
                    "model_kwargs": knobs["model_kwargs"],
                    "cost_tracking": knobs["cost_tracking"],
                    "cost_limit": knobs["cost_limit"],
                },
                turn_cap=turn_cap,
                output_path=sub_raw / f"sub_{counter['k']:02d}.traj.json",
                event_hook=sink.emit,
            )

        planner = Planner(
            wrapper, state,
            model_name=ctx.model,
            model_kwargs=knobs["model_kwargs"],
            run_subgoal_fn=run_one,
            emit=sink.emit,
            step_budget=ctx.step_budget,
            max_planner_turns=knobs["planner_max_turns"],
            default_subgoal_budget=knobs["subgoal_budget"],
            subgoal_turn_cap=knobs["subgoal_turn_cap"],
        )

        error: str | None = None
        summary: dict[str, Any] = {}
        try:
            summary = planner.run()
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            sink.emit("driver_error", {"error": error, "where": "eharness"})

        state.sink_cold(ctx.live_dir / "memory.md")   # pre_compact discipline
        return SessionOutcome(
            usage=None,
            cost_usd=round(planner.sub_cost, 4),
            turns=planner.turns + planner.sub_turns,
            error=error,
            extra={
                "exit_status": summary.get("end_reason"),
                "toolset_counts": dict(env.toolset.calls_by_tool),
                "toolset_env_steps": env.toolset.steps_taken,
                "toolset_end_reason": env.toolset.end_reason,
                "eharness": summary,
            },
        )

    def _run_solo(self, ctx: EpisodeContext, sink: EventSink,
                  knobs: dict[str, Any], wrapper, state,
                  opening_parts: list | None = None) -> SessionOutcome:
        """The solo paradigm: one harnessed loop for the whole episode
        (solo-harness.html §8). Guards / pre-stop / keyframes / state come
        from the wrapper; bounded context + L2 come from SoloModel."""
        from eharness.solo import run_solo

        counts = {"compact": 0, "guard": 0, "verdict": 0, "state_write": 0}

        def emit(kind: str, payload: dict) -> None:
            if kind in counts:
                counts[kind] += 1
            sink.emit(kind, payload)

        wrapper._emit = emit  # count wrapper-side events (guard/verdict/state_write) too

        exit_info, agent = run_solo(
            wrapper,
            briefing=ctx.briefing,
            task=ctx.instruction,
            model_name=ctx.model,
            opening_parts=opening_parts,
            model_knobs={
                "image_window": knobs["image_window"],
                "compact_at": knobs["compact_at"],
                "tail_msgs": knobs["tail_msgs"],
                "reattach_k": knobs["reattach_k"],
                "history_frames": knobs["history_frames"],
                "recent_turns": knobs["recent_turns"],
                "legacy_l2": knobs["legacy_l2"],
                "set_cache_control": knobs["set_cache_control"],
                "model_kwargs": knobs["model_kwargs"],
                "cost_tracking": knobs["cost_tracking"],
                "cost_limit": knobs["cost_limit"],
            },
            turn_cap=ctx.max_turns,
            output_path=ctx.raw_dir / f"episode_{ctx.index}.traj.json",
            event_hook=emit,
        )
        state.sink_cold(ctx.live_dir / "memory.md")
        error = exit_info.get("error")
        return SessionOutcome(
            usage=None,
            cost_usd=round(float(getattr(agent, "cost", 0.0) or 0.0), 4),
            turns=getattr(agent, "n_calls", 0),
            error=error,
            extra={
                "exit_status": exit_info.get("exit_status"),
                "toolset_counts": dict(wrapper.inner.calls_by_tool),
                "toolset_env_steps": wrapper.inner.steps_taken,
                "toolset_end_reason": wrapper.inner.end_reason,
                "eharness": {
                    "paradigm": "solo",
                    "compactions": counts["compact"],
                    "guard_events": counts["guard"],
                    "verdicts": counts["verdict"],
                    "state_writes": counts["state_write"],
                    "verify_calls": wrapper.verify_calls,
                    "keyframes": len(wrapper.frames.keyframes()),
                    "frames": len(wrapper.frames.frames),
                    "state_cursor": f"{state.cursor}/{len(state.sub_instructions)}",
                },
            },
        )
