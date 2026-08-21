#!/usr/bin/env python
from __future__ import annotations

"""Episode driver for the VLA harness.

    L3  planner decomposes, dispatches, judges each stop, recovers, finishes
    L1  --mode clauses: no model in the loop; the instruction is split offline
        and the policy's STOP advances to the next clause (the control that
        separates "the agent helped" from "decomposition helped")
    L0  --mode whole: the whole instruction, one execute — a parity check that
        the harness reproduces the policy-alone baseline

All three write the same row schema, so they are directly comparable.

    python run.py --n 3 --live                  # L3, subscription via Agent SDK
    python run.py --n 100 --mode clauses --live # L1 control
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlaharness.planner import build_client, run_episode  # noqa: E402
from vlaharness.toolset import VlaToolSet  # noqa: E402


def _cost(payloads) -> dict:
    """What one episode of outer-model calls actually consumed. Aggregated
    here rather than logged per call so a run's price is one column."""
    out = {"calls": 0, "usd": 0.0, "in_tok": 0, "out_tok": 0, "api_ms": 0,
           "ctx_tok": 0, "max_ctx_tok": 0, "failed_calls": 0}
    for p in payloads or []:
        m = p.get("call_meta") or {}
        out["calls"] += 1
        out["failed_calls"] += 1 if m.get("failed") else 0
        for k, key in (("usd", "cost_usd"), ("in_tok", "in_tok"),
                       ("out_tok", "out_tok"), ("api_ms", "api_ms"),
                       ("ctx_tok", "ctx_tok")):
            out[k] += (m.get(key) or 0)
        out["max_ctx_tok"] = max(out["max_ctx_tok"], m.get("ctx_tok") or 0)
    out["usd"] = round(out["usd"], 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://127.0.0.1:9240"))
    ap.add_argument("--policy-url", default=os.environ.get("POLICY_URL", "http://127.0.0.1:9230"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dataset", default="R2R-CE")
    ap.add_argument("--split", default="rand100")
    ap.add_argument("--planner-model", default="claude-sonnet-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-steps", type=int, default=50, help="policy steps per execute()")
    ap.add_argument("--step-budget", type=int, default=400, help="env steps per episode")
    ap.add_argument("--max-turns", type=int, default=24, help="planner turns per episode")
    ap.add_argument("--mode", default="planner",
                    choices=["agent", "agent2", "stateful", "planner", "clauses",
                             "clauses_rule", "whole"],
                    help="agent=L3 rewritten state block, no pre-planned legs, 14 "
                         "native-res frames per review · stateful=L3 monotone state "
                         "block + O(1) judge · planner=L3 conversational · "
                         "clauses=L1 (model decomposes, then leaves) · "
                         "clauses_rule=L1 rule splitter · whole=parity vs L0")
    ap.add_argument("--run-name", default=None, help="artifact dir under outputs/beta-vla-harness")
    ap.add_argument("--live", nargs="?", const="", default=None,
                    help="write live state for the VLA Harness tab (optional path)")
    ap.add_argument("--api", action="store_true",
                    help="drive the planner via the Messages API (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--rgb-size", default=None, help="e.g. 720x640 or 512 (default: upstream rig)")
    ap.add_argument("--rgb-hfov", default=None, help="e.g. 110 or 90")
    ap.add_argument("--rgb-height", default=None, help="camera height in m, e.g. 0.5 or 1.25")
    ap.add_argument("--ablate", default="",
                    help="agent mode only: comma-separated design axes to switch OFF, "
                         "so the bundle can be measured one at a time. "
                         "verify,sweep,drive,back,frames — see agent_loop.ABLATIONS")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm = {"agent": "l3_agent", "agent2": "l3_agent2", "stateful": "l3_stateful",
           "planner": "l3_planner", "clauses": "l1_llm",
           "clauses_rule": "l1_clauses", "whole": "l0_parity"}[args.mode]

    from vlaharness.agent_loop2 import ABLATIONS as A2
    from vlaharness.agent_loop import ABLATIONS as A1
    ABLATIONS = A2 if args.mode == "agent2" else A1

    ablate = frozenset(a.strip() for a in args.ablate.split(",") if a.strip())
    unknown = ablate - set(ABLATIONS)
    if unknown:
        raise SystemExit(f"unknown --ablate axes {sorted(unknown)}; "
                         f"choose from {sorted(ABLATIONS)}")
    base_arm = arm
    if ablate:
        # An ablated run is a different arm and must not silently overwrite
        # the full one's results.
        arm = arm + "_no_" + "_".join(sorted(ablate))
        print("ablating: " + "; ".join(f"{a} — {ABLATIONS[a]}" for a in sorted(ablate)),
              flush=True)
    out_path = args.out or os.path.expanduser(
        f"~/Desktop/Projects/DeepStack/outputs/vla_{arm}_r2rce_rand100/result.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # The Claude Code OAuth credential 429s against the raw Messages API, so
    # the planner arm goes through the Agent SDK — the same subscription path
    # the repo's other billed runs use. --api forces the Messages API instead.
    client, auth = (None, "none")
    use_sdk = args.mode == "planner" and not args.api
    if args.mode in ("agent", "agent2", "stateful"):
        auth = "sdk:subscription"
    elif args.mode == "planner":
        if use_sdk:
            auth = "sdk:subscription"
        else:
            client, auth = build_client(args.planner_model)
    print(f"[{arm}] planner={args.planner_model if client else '-'} effort={args.effort} "
          f"auth={auth} max_steps={args.max_steps} -> {out_path}", flush=True)

    done: dict[int, dict] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["index"]] = r
        print(f"resuming: {len(done)} episodes on disk", flush=True)

    live = None
    if args.live is not None:
        from vlaharness.live import LiveWriter
        live = LiveWriter(args.live or None, arm, args.n,
                          args.planner_model if args.mode == "planner" else None)
        print(f"live view: VLA Harness tab on :5173 -> {live.path}", flush=True)

    from vlaharness.eventlog import EventLog

    log = EventLog(args.run_name or f"std_vla_{arm}_sonnet5", {
        "harness": "vla",
        "arm": arm,
        "model": args.planner_model if args.mode in
                 ("agent", "agent2", "stateful", "planner", "clauses") else None,
        "split": f"{args.dataset} {args.split}",
        "harness_inherent": {
            "policy": "qwen3vl-omega-current3-vlm (2B) via policy_vla_nav nodeset",
            "rig": "720x640 · hfov 110 · camera height 0.5 m (upstream evaluator defaults)",
            "tool": "execute(sub_instruction) rolls the policy out; its STOP ends the "
                    "rollout, never the episode. finish() is the only terminator.",
            "arm": {
                "l3_agent2": "v2: typed evidence-backed state with a negative-fact "
                             "gate, three-state termination (VERIFIED_SUCCESS / "
                             "VERIFIED_FAILURE / UNRESOLVED_STOP), schema-checked "
                             "calls with retry + backup verifier, micro closed-loop "
                             "corrections, frame-similarity stall detection, and no "
                             "silent mission replay.",
                "l3_agent": "no pre-planned legs — the outer model writes the next "
                            "subtask each turn from a state block it REWRITES in full, "
                            "reviewing 10 frames of THAT dispatch plus the 4 views at "
                            "the pose it ended in, native 720x640. A separate "
                            "verification call is the only path to STOP, and it may "
                            "take a free 360 sweep or drive a few steps itself. Frames "
                            "are archived at full quality and never re-enter context.",
                "l3_stateful": "outer model plans a route + termination condition, then "
                               "judges every dispatch in a fresh O(1) session against a "
                               "disk-backed state block; frames leave context, prose stays",
                "l3_planner": "outer model decomposes, judges each stop against the "
                              "traversed path, recovers, decides termination",
                "l1_llm": "outer model writes the legs, then leaves the loop entirely",
                "l1_clauses": "rule-based split; no model anywhere",
                "l0_parity": "whole instruction, one dispatch — parity check vs baseline",
            }[base_arm] + ("" if not ablate else
               "  ABLATED: " + "; ".join(f"{a} — {ABLATIONS[a]}" for a in sorted(ablate))),
            "auth": "claude subscription via Agent SDK",
        },
        "max_steps_per_dispatch": args.max_steps,
        "step_budget": args.step_budget,
    })
    print(f"artifacts: {log.dir}", flush=True)

    from vlaharness.toolset import RIG

    # Every arm defaults to the rig the policy was TRAINED on (0.5 m / 110 /
    # 720x640). Raising the camera to AgentCanvas's 1.25 m costs 8 SR points on
    # the policy alone — measured paired, n=100, 57.0 -> 49.0 — so an arm run
    # there is measuring the harness through a handicapped actuator, and the
    # two effects cannot be separated afterwards. Rig shift is its own
    # experiment (`--rgb-height 1.25`), not a default.
    rig = dict(RIG)
    for key, val in (("rgb_resolution", args.rgb_size), ("rgb_hfov", args.rgb_hfov),
                     ("rgb_height_m", args.rgb_height)):
        if val:
            rig[key] = str(val)
    if rig != RIG:
        print(f"rig override: {rig}", flush=True)
    log.meta["rig"] = rig

    probe = VlaToolSet(args.env_url, args.policy_url, rig=rig)
    if args.dataset:
        probe.panel_field("dataset", args.dataset)
    probe.panel_field("split", args.split)

    rows = list(done.values())
    for index in range(args.start, args.start + args.n):
        if index in done:
            continue
        # The agent arm reviews ONE DISPATCH at a time — 10 frames sampled from
        # that segment, not from the episode — at native resolution, and needs
        # the three current views to really be current, so it turns off the
        # shrink and asks for one extra render at the end of each rollout.
        # path_frames=0 because the episode-wide sample is unused here and
        # encoding it would be pure waste.
        extra = ({"leg_frames": 3 if "frames" in ablate else 10,
                  "path_frames": 0, "frame_px": 0, "observe_at_end": True,
                  "back_view": "back" not in ablate}
                 if args.mode in ("agent", "agent2") else {})
        tools = VlaToolSet(
            args.env_url, args.policy_url,
            max_steps=args.max_steps, step_budget=args.step_budget, rig=rig,
            **extra,
        )
        t0 = time.time()
        try:
            info = tools.reset_episode(index)
            instruction = info.get("instruction") or ""
            log.episode(index)
            log.emit("episode_meta", index=index, episode_id=info.get("episode_id"),
                     scene_id=info.get("scene_id"), instruction=instruction)
            legs, legs_raw = None, None
            if args.mode == "clauses":
                from vlaharness.decompose import decompose

                legs, legs_raw = decompose(instruction, model=args.planner_model)
                log.emit("decompose", model=args.planner_model, segments=legs, raw=legs_raw)
            if live is not None:
                live.start_episode(index, instruction, info.get("episode_id"))
            if args.mode == "agent2":
                from vlaharness.agent_loop2 import run_agent_episode2

                trace = run_agent_episode2(
                    tools, instruction, model=args.planner_model,
                    max_turns=args.max_turns, live=live, log=log,
                    live_dir=getattr(log, "_live", None), ablate=ablate,
                )
            elif args.mode == "agent":
                from vlaharness.agent_loop import run_agent_episode

                trace = run_agent_episode(
                    tools, instruction, model=args.planner_model,
                    max_turns=args.max_turns, live=live, log=log,
                    live_dir=getattr(log, "_live", None), ablate=ablate,
                )
            elif args.mode == "stateful":
                from vlaharness.stateful import run_stateful_episode

                trace = run_stateful_episode(
                    tools, instruction, model=args.planner_model,
                    max_turns=args.max_turns, live=live, log=log,
                    live_dir=getattr(log, "_live", None),
                )
            elif use_sdk:
                from vlaharness.planner_sdk import run_episode_sdk

                trace = run_episode_sdk(
                    tools, instruction,
                    model=args.planner_model, effort=args.effort,
                    max_turns=args.max_turns, live=live, log=log,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
            else:
                trace = run_episode(
                    client, tools, instruction,
                    model=args.planner_model, effort=args.effort,
                    max_turns=args.max_turns,
                    mode="clauses" if args.mode == "clauses_rule" else args.mode,
                    live=live, legs=legs, log=log,
                )
            metrics = tools.metrics()
        except Exception as exc:  # noqa: BLE001 — one bad episode must not kill the run
            print(f"  ep{index}: FAILED {exc!r}", flush=True)
            continue

        row = {
            "index": index,
            "arm": arm,
            "episode_id": info.get("episode_id"),
            "scene_id": info.get("scene_id"),
            "instruction": instruction,
            "success": float(metrics.get("success", 0.0) or 0.0),
            "spl": float(metrics.get("spl", 0.0) or 0.0),
            "oracle_success": float(metrics.get("oracle_success", 0.0) or 0.0),
            "distance_to_goal": float(metrics.get("distance_to_goal", 0.0) or 0.0),
            "ndtw": float(metrics.get("ndtw", 0.0) or 0.0),
            "steps": tools.steps_taken,
            "end_reason": tools.end_reason,
            "planner_turns": trace.get("turns", 0),
            "n_execute": sum(1 for c in trace["tool_calls"] if c["name"] == "execute"),
            "forced_finish": trace.get("forced_finish", False),
            "planner_error": trace.get("error"),
            "planner_result": trace.get("result"),
            "legs": trace.get("legs") or legs,
            "legs_raw": legs_raw,
            "terminate": trace.get("terminate"),
            "judgments": trace.get("judgments"),
            "state": trace.get("state"),
            "verification": trace.get("verification"),
            "bootstrap_failed": trace.get("bootstrap_failed"),
            "ended_without_stop": trace.get("ended_without_stop"),
            "ended_dispatching": trace.get("ended_dispatching"),
            "ablate": sorted(ablate) or None,
            "outcome": trace.get("outcome"),
            "outcome_why": trace.get("outcome_why"),
            "counts": trace.get("counts"),
            "clauses": trace.get("clauses"),
            "cost": _cost(trace.get("payloads")),
            "payloads": trace.get("payloads"),
            "sub_instructions": [
                c["input"].get("sub_instruction")
                for c in trace["tool_calls"] if c["name"] == "execute"
            ],
            "calls": tools.calls,
            "planner_text": trace.get("planner_text", []),
            "wall_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        log.emit("episode_end", **{k: row[k] for k in
                 ("success", "spl", "oracle_success", "distance_to_goal", "ndtw",
                  "steps", "end_reason", "n_execute", "wall_s")})
        log.write_summary(rows)
        if live is not None:
            live.end_episode(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        n = len(rows)
        sr = 100 * sum(r["success"] for r in rows) / n
        print(
            f"  ep{index:3d} {'OK ' if row['success'] else '.  '} "
            f"steps={row['steps']:3d} ne={row['distance_to_goal']:5.2f} "
            f"exec={row['n_execute']:2d} {str(row['end_reason']):22s} "
            f"{row['wall_s']:6.1f}s | SR={sr:.1f}% ({n})",
            flush=True,
        )

    if rows:
        n = len(rows)
        print(f"\n=== {arm} · R2R-CE {args.split} · n={n} ===")
        for k, label in (("success", "SR"), ("spl", "SPL"), ("oracle_success", "OSR"),
                         ("ndtw", "nDTW"), ("distance_to_goal", "NE")):
            v = sum(r.get(k, 0.0) for r in rows) / n
            print(f"  {label:4s} {v:6.2f} m" if k == "distance_to_goal" else f"  {label:4s} {v * 100:6.2f}")
        print(f"  steps  {sum(r['steps'] for r in rows) / n:6.1f}")
        print(f"  exec   {sum(r['n_execute'] for r in rows) / n:6.1f} per episode")
        ends: dict[str, int] = {}
        for r in rows:
            ends[str(r["end_reason"])] = ends.get(str(r["end_reason"]), 0) + 1
        print(f"  end    {ends}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
