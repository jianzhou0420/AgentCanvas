#!/usr/bin/env python3
"""Analyze an Agent-selected Hybrid Interface run.

Standard navigation metrics (SR / SPL / nDTW / NE / OSR) plus the interface-use
metrics the hybrid experiment is about:
  - fail / timeout counts (+ failure breakdown by end_reason)
  - primitive(step) vs waypoint(goto) call ratio
  - interface switches per episode
  - how often a waypoint move is followed by a primitive correction or a STOP,
    and how each episode makes its final approach (primitive vs waypoint)

Reads only the run's own artifacts: summary.json (metrics + agent.tool_calls +
end_reason/error) and live_<i>/actions.log (the ordered, executed tool trace the
bridge writes). Usage:  python analyze_hybrid.py <run_dir> [--json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def move_seq(run_dir: Path, idx: int) -> list[str]:
    """Ordered EXECUTED tool trace for episode idx from actions.log:
    each entry is one of 'obs' | 'pano' | 'step' | 'goto' | 'stop'."""
    al = run_dir / f"live_{idx}" / "actions.log"
    seq: list[str] = []
    if not al.exists():
        return seq
    for line in al.open():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("observe"):
            seq.append("obs")
        elif d.get("observe_waypoints"):
            seq.append("pano")
        elif "goto" in d:
            seq.append("goto")
        elif "step" in d:
            seq.append("step")
        elif d.get("stopped"):
            seq.append("stop")
    return seq


def is_scored(e: dict) -> bool:
    """A real, scored navigation attempt (mirrors driver.is_scored)."""
    m = e.get("metrics") or {}
    if m.get("success") is None:
        return False
    if e.get("error") and not ((e.get("agent") or {}).get("env_steps") or 0):
        return False
    return True


def analyze(run_dir: Path) -> dict:
    d = json.loads((run_dir / "summary.json").read_text())
    eps = d.get("episodes", [])
    scored = [e for e in eps if is_scored(e)]

    def mean(key: str):
        vals = [e["metrics"].get(key) for e in scored
                if isinstance(e["metrics"].get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    succ = [e for e in scored if e["metrics"].get("success") == 1]
    fail = [e for e in scored if e["metrics"].get("success") == 0]
    timeout = [e for e in eps if e.get("error") == "timeout"]
    errored = [e for e in eps if e.get("error") and e.get("error") != "timeout"
               and not is_scored(e)]
    fail_reasons = Counter((e.get("agent") or {}).get("end_reason") for e in fail)

    tot = {"step": 0, "goto": 0, "observe": 0, "observe_waypoints": 0}
    per_ep = []
    for e in scored:
        tc = (e.get("agent") or {}).get("tool_calls") or {}
        for k in tot:
            tot[k] += tc.get(k, 0)
        seq = move_seq(run_dir, e["index"])
        moves = [x for x in seq if x in ("step", "goto", "stop")]
        switches = sum(1 for i in range(len(moves) - 1)
                       if {moves[i], moves[i + 1]} == {"step", "goto"})
        g2s = sum(1 for i in range(len(moves) - 1)
                  if moves[i] == "goto" and moves[i + 1] == "step")
        g2stop = sum(1 for i in range(len(moves) - 1)
                     if moves[i] == "goto" and moves[i + 1] == "stop")
        arrive = moves[-2] if len(moves) >= 2 and moves[-1] == "stop" else None
        per_ep.append({
            "idx": e["index"], "id": e.get("episode_id"),
            "success": e["metrics"].get("success"),
            "step": tc.get("step", 0), "goto": tc.get("goto", 0),
            "observe": tc.get("observe", 0), "observe_waypoints": tc.get("observe_waypoints", 0),
            "switches": switches, "goto_then_step": g2s, "goto_then_stop": g2stop,
            "arrive_via": arrive,
        })

    moves_total = tot["step"] + tot["goto"]
    usage = {
        "primitive_step_calls": tot["step"],
        "waypoint_goto_calls": tot["goto"],
        "observe_forward_calls": tot["observe"],
        "observe_waypoints_calls": tot["observe_waypoints"],
        "primitive_frac_of_moves": round(tot["step"] / moves_total, 3) if moves_total else None,
        "waypoint_frac_of_moves": round(tot["goto"] / moves_total, 3) if moves_total else None,
        "avg_switches_per_episode": round(sum(x["switches"] for x in per_ep) / len(per_ep), 2) if per_ep else None,
        "episodes_with_a_switch": sum(1 for x in per_ep if x["switches"] > 0),
        "goto_then_primitive_correction": sum(x["goto_then_step"] for x in per_ep),
        "goto_then_stop": sum(x["goto_then_stop"] for x in per_ep),
        "episodes_arriving_via_primitive": sum(1 for x in per_ep if x["arrive_via"] == "step"),
        "episodes_arriving_via_waypoint": sum(1 for x in per_ep if x["arrive_via"] == "goto"),
    }
    return {
        "run": run_dir.name,
        "aggregate": {"episodes": len(eps), "scored": len(scored), "SR": mean("success"),
                      "SPL": mean("spl"), "nDTW": mean("ndtw"), "OSR": mean("oracle_success"),
                      "NE_m": mean("distance_to_goal")},
        "outcomes": {"success": len(succ), "fail": len(fail), "timeout": len(timeout),
                     "errored_infra": len(errored), "fail_end_reasons": dict(fail_reasons)},
        "interface_usage": usage,
        "per_episode": per_ep,
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv
    if not args:
        sys.exit("usage: python analyze_hybrid.py <run_dir> [--json]")
    out = analyze(Path(args[0]))
    if as_json:
        print(json.dumps(out, indent=2))
        return
    a, o, u = out["aggregate"], out["outcomes"], out["interface_usage"]
    print(f"=== {out['run']} ===")
    print(f"nav: n={a['episodes']} scored={a['scored']} | SR={a['SR']} SPL={a['SPL']} "
          f"nDTW={a['nDTW']} OSR={a['OSR']} NE={a['NE_m']}m")
    print(f"outcomes: success={o['success']} fail={o['fail']} timeout={o['timeout']} "
          f"errored={o['errored_infra']} | fail_by_reason={o['fail_end_reasons']}")
    print("interface usage:")
    print(f"  primitive step calls = {u['primitive_step_calls']}  "
          f"waypoint goto calls = {u['waypoint_goto_calls']}  "
          f"(primitive frac of moves = {u['primitive_frac_of_moves']}, "
          f"waypoint = {u['waypoint_frac_of_moves']})")
    print(f"  looks: observe(forward) = {u['observe_forward_calls']}  "
          f"observe_waypoints = {u['observe_waypoints_calls']}")
    print(f"  switches/episode avg = {u['avg_switches_per_episode']}  "
          f"({u['episodes_with_a_switch']} episodes switched at least once)")
    print(f"  goto -> primitive correction = {u['goto_then_primitive_correction']}  "
          f"goto -> STOP = {u['goto_then_stop']}")
    print(f"  final approach: via primitive = {u['episodes_arriving_via_primitive']}  "
          f"via waypoint = {u['episodes_arriving_via_waypoint']}")


if __name__ == "__main__":
    main()
