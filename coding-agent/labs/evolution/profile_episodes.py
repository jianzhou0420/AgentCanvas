"""profile_episodes — mechanical failure profiler for a run dir (eharness-evo Stage-1).

The program version of the hand diagnosis done on evo9b_allin_blocked01 ep2/ep3:
read episode_*.jsonl, reduce each episode to a few cheap signals, and tag the
dominant failure shape. No GT beyond the driver's own metrics, no pixels — the
action stream, the tool results, and the student's own thinking text are enough
to flag where to look. Tags are hypotheses for the engineer, not verdicts.

Usage:
  python profile_episodes.py RUN_DIR [--json OUT] [--md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

ACT = {0: "S", 1: "F", 2: "L", 3: "R", 4: "L", 5: "R", 6: "L"}  # macros fold into L/R
SELF_REPORT = re.compile(
    r"(turning in circles|in a loop|stuck|dead[- ]end|same (view|wall|area) again|"
    r"back (to|at|in) (the|where)|keep getting|frustrating|again\b.*\b(wall|hallway|door))",
    re.I)


def load(run_dir: str):
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    by_idx = {e["index"]: e for e in summ.get("episodes", [])}
    for f in sorted(glob.glob(os.path.join(run_dir, "episode_*.jsonl")),
                    key=lambda p: int(re.findall(r"episode_(\d+)", p)[0])):
        idx = int(re.findall(r"episode_(\d+)", f)[0])
        evs = []
        for line in open(f):
            try:
                evs.append(json.loads(line))
            except Exception:
                pass
        yield idx, by_idx.get(idx), evs


def profile(idx: int, rec: dict | None, evs: list[dict]) -> dict:
    calls: list[list[int]] = []
    blocked_calls = 0
    revisit_hits = 0
    remembers = 0
    self_reports = 0
    for e in evs:
        k = e.get("kind")
        if k == "tool_use":
            name = e.get("name") or ""
            inp = e.get("input") or {}
            if "step" in name and isinstance(inp.get("actions"), list):
                calls.append([int(a) for a in inp["actions"] if isinstance(a, (int, float))])
            elif "remember" in name:
                remembers += 1
        elif k == "tool_result":
            for t in (e.get("texts") or []):
                if '"forward_blocked"' in t and '"blocked": 0' not in t:
                    blocked_calls += 1
                if "seen_before" in t:
                    revisit_hits += 1
        elif k == "thinking":
            if SELF_REPORT.search(e.get("text") or ""):
                self_reports += 1

    flat = [a for c in calls for a in c]
    n = len(flat)
    mix = Counter(ACT.get(a, "?") for a in flat)
    # longest consecutive run of same-direction turns across calls (spin)
    best = cur = 0
    prev = None
    for a in flat:
        s = ACT.get(a)
        if s in ("L", "R") and s == prev:
            cur += 1
        elif s in ("L", "R"):
            cur = 1
        else:
            cur = 0
        prev = s if s in ("L", "R") else None
        best = max(best, cur)
    # ping-pong: call-level L-run followed by R-run (or vice versa), both >= 3
    runs = ["".join(ACT.get(a, "?") for a in c) for c in calls]
    pingpong = 0
    for a, b in zip(runs, runs[1:]):
        if (set(a) == {"L"} and set(b) == {"R"} or set(a) == {"R"} and set(b) == {"L"}) \
                and len(a) >= 3 and len(b) >= 3:
            pingpong += 1
    fwd_only_calls = sum(1 for r in runs if r and set(r) == {"F"})
    single_action_calls = sum(1 for c in calls if len(c) == 1)

    m = (rec or {}).get("metrics", {}) if rec else {}
    success = m.get("success")
    dtg = m.get("distance_to_goal")
    steps = m.get("steps_taken")
    oracle = m.get("oracle_success")
    stopped = any(0 in c for c in calls)

    tags = []
    if success == 1.0:
        tags.append("success")
    else:
        if oracle == 1.0 and not stopped:
            tags.append("reached-no-stop")
        elif stopped and (dtg or 0) > 3:
            tags.append("wrong-stop")
        if best >= 24:
            tags.append(f"spin(max_turn_run={best})")
        if pingpong >= 3:
            tags.append(f"pingpong(x{pingpong})")
        if blocked_calls >= 3 and fwd_only_calls >= 0.5 * max(1, len(runs)):
            tags.append(f"wall-push(blocked_calls={blocked_calls})")
        if self_reports >= 3:
            tags.append(f"self-reported-loop(x{self_reports})")
        if steps is not None and steps >= 500:
            tags.append("budget-exhausted")
        if not tags:
            tags.append("other")
    return {
        "ep": idx, "success": success, "dtg": round(dtg, 2) if dtg is not None else None,
        "steps": steps, "calls": len(calls),
        "acts_per_call": round(n / max(1, len(calls)), 2),
        "fwd_pct": round(100 * mix["F"] / max(1, n)), "L_pct": round(100 * mix["L"] / max(1, n)),
        "R_pct": round(100 * mix["R"] / max(1, n)),
        "max_turn_run": best, "pingpong": pingpong, "blocked_calls": blocked_calls,
        "single_action_calls": single_action_calls, "remembers": remembers,
        "revisit_hits": revisit_hits, "self_reports": self_reports, "stopped": stopped,
        "tags": tags,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = [profile(i, r, e) for i, r, e in load(a.run_dir)]
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, ensure_ascii=False)
    hdr = ["ep", "ok", "dtg", "steps", "calls", "a/call", "F%", "L%", "R%", "spin", "pp", "blk", "1act", "mem", "rev", "self", "tags"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(x) for x in [
            r["ep"], int(r["success"] or 0), r["dtg"], r["steps"], r["calls"], r["acts_per_call"],
            r["fwd_pct"], r["L_pct"], r["R_pct"], r["max_turn_run"], r["pingpong"], r["blocked_calls"],
            r["single_action_calls"], r["remembers"], r["revisit_hits"], r["self_reports"],
            ",".join(r["tags"])]))
    tagc = Counter(t.split("(")[0] for r in rows for t in r["tags"])
    sr = sum(1 for r in rows if r["success"] == 1.0) / max(1, len(rows))
    print(f"\n{len(rows)} eps  SR={sr:.2f}  tag counts: {dict(tagc)}")


if __name__ == "__main__":
    main()
