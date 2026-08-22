"""supervisor — the eharness-evo campaign program (what "self-evolution" IS).

A plain state machine over a campaign directory. Each `step` does ONE bounded
phase, writes state.json, and returns; re-running the same command resumes.
The LLM engineer is not a session: it is called at most twice per generation
(DIAGNOSE, PROPOSE), each as a fresh, contract-bound SDK call whose output is
validated by code. Everything else — rollouts, profiling, clustering, gates,
promotion — is deterministic Python.

    ROLLOUT -> PROFILE -> DIAGNOSE -> PROPOSE -> GATE -> PROMOTE -> ROLLOUT ...
                                                           \\-> COMPLETE (budget)

Usage:
  python supervisor.py init  ROOT --parent evo9b_allin_blocked01 --episodes 0-24
  python supervisor.py step  ROOT [--dry-run] [--engine dry|sdk] [--no-run]
  python supervisor.py show  ROOT

Campaign root layout (all append-only except state.json):
  manifest.json        frozen: parent arm, episodes, alpha, budgets, engine
  state.json           the ONLY mutable file: phase / generation / current_arm / candidate
  gen_NNN/profile.json | clusters.json | diagnosis/{input,output}.json | proposal/{input,output}.json
  ledger/gates.jsonl | ledger/promotions.jsonl | ledger/events.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # .../AgentCanvas
CA = REPO / "coding-agent"
RUNS = REPO / "outputs" / "beta-react-harness"
PY = os.environ.get("EVO_PYTHON", os.path.expanduser("~/miniconda3/envs/agentcanvas/bin/python"))
sys.path.insert(0, str(HERE))
from paired_gate import load_board, compare          # noqa: E402
from profile_episodes import load as load_eps, profile  # noqa: E402

PHASES = ["ROLLOUT", "PROFILE", "DIAGNOSE", "PROPOSE", "GATE", "PROMOTE", "COMPLETE"]
NEXT = {"ROLLOUT": "PROFILE", "PROFILE": "DIAGNOSE", "DIAGNOSE": "PROPOSE",
        "PROPOSE": "GATE", "GATE": "PROMOTE", "PROMOTE": "ROLLOUT"}


# ── tiny persistence primitives (append-only ledger + atomic state) ──
def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"t": time.strftime("%F %T"), **row}, ensure_ascii=False) + "\n")


class Campaign:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.state = json.loads((root / "state.json").read_text())

    def save(self) -> None:
        _atomic_write(self.root / "state.json", self.state)

    def log(self, **row) -> None:
        _append(self.root / "ledger" / "events.jsonl", {"gen": self.state["generation"],
                                                        "phase": self.state["phase"], **row})

    @property
    def gen_dir(self) -> Path:
        d = self.root / f"gen_{self.state['generation']:03d}"
        d.mkdir(exist_ok=True)
        return d

    def transition(self, to: str, **updates) -> None:
        assert to in PHASES, to
        self.state.update(updates)
        self.state["phase"] = to
        self.save()


# ── phases ──
def ph_rollout(c: Campaign, dry: bool, no_run: bool) -> str:
    """Make sure the current arm has a board on the campaign episodes."""
    arm = c.state["current_arm"]
    run = RUNS / arm
    eps = c.manifest["episodes"]
    done = load_board(run)["n_total"] if (run / "summary.json").exists() else 0
    want = _count_eps(eps)
    if done >= want:
        c.log(note=f"{arm}: board present ({done}/{want} eps)")
        return f"board present: {arm} ({done}/{want})"
    if dry or no_run:
        return f"WOULD RUN: stdrun.py run {arm} --episodes {eps}  (have {done}/{want})"
    cmd = [PY, str(CA / "stdrun.py"), "run", arm, "--episodes", eps]
    c.log(cmd=" ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True,
                   env={**os.environ, "MINI_SERVE_CTX": os.environ.get("MINI_SERVE_CTX", "65536")})
    return f"ran {arm} on {eps}"


def ph_profile(c: Campaign, dry: bool, **_) -> str:
    """Deterministic failure profiling + clustering by dominant tag (no LLM)."""
    arm = c.state["current_arm"]
    rows = [profile(i, r, e) for i, r, e in load_eps(str(RUNS / arm))]
    _atomic_write(c.gen_dir / "profile.json", rows)
    clusters: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["success"] == 1.0:
            continue
        tag = next((t.split("(")[0] for t in r["tags"]
                    if t.split("(")[0] in ("spin", "pingpong", "wall-push", "reached-no-stop",
                                           "wrong-stop", "self-reported-loop")), r["tags"][0])
        clusters[tag].append(r["ep"])
    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    out = [{"tag": t, "members": m, "size": len(m)} for t, m in ranked]
    _atomic_write(c.gen_dir / "clusters.json", out)
    sr = sum(1 for r in rows if r["success"] == 1.0) / max(1, len(rows))
    c.log(sr=round(sr, 3), n=len(rows), clusters=[(t, len(m)) for t, m in ranked])
    return f"{arm}: {len(rows)} eps SR={sr:.2f}; clusters " + ", ".join(f"{t}:{len(m)}" for t, m in ranked)


DIAG_CONTRACT = {
    "cluster_tag": "<copy from input>",
    "outcome": "what happens (observable)",
    "immediate_trigger": "the event right before it goes wrong",
    "root_cause": "one mechanism, or 'inconclusive: ...'",
    "competing_hypotheses": ["at least two distinct alternatives"],
    "evidence": ["ep N: <quote or frame id>", "..."],
    "falsifier": "what observation would refute root_cause",
    "proposed_delta": {"tier": "T1|T2|T3|T4", "one_line": "exactly one change to the arm"},
    "confidence": 0.0,
}


def ph_diagnose(c: Campaign, dry: bool, engine: str, **_) -> str:
    """One contract-bound engineer call on the top cluster (or a dry input dump)."""
    clusters = json.loads((c.gen_dir / "clusters.json").read_text())
    if not clusters:
        c.log(note="no failure clusters")
        return "no failures to diagnose"
    top = clusters[0]
    rows = {r["ep"]: r for r in json.loads((c.gen_dir / "profile.json").read_text())}
    arm = c.state["current_arm"]
    inp = {
        "campaign": c.root.name, "generation": c.state["generation"], "arm": arm,
        "cluster": top, "member_profiles": [rows[e] for e in top["members"]],
        "evidence_paths": {str(e): {"jsonl": str(RUNS / arm / f"episode_{e}.jsonl"),
                                    "frames_dir": str(RUNS / arm / f"live_{e}")}
                           for e in top["members"]},
        "read_budget": 12,
        "output_schema": DIAG_CONTRACT,
        "rules": ["read at least 2 member transcripts and 2 frames before answering",
                  "root_cause may be 'inconclusive'", "no scene names, no coordinates in the delta",
                  "proposed_delta must be ONE change: a tool (T1), a text (T2), a rule (T3) or a loop organ (T4)"],
    }
    d = c.gen_dir / "diagnosis"; d.mkdir(exist_ok=True)
    _atomic_write(d / "input.json", inp)
    if engine == "dry" or dry:
        c.log(note="diagnosis input written (dry engine)")
        return (f"engineer call #1 prepared: cluster '{top['tag']}' x{top['size']} -> {d/'input.json'}"
                f"\n      contract keys: {list(DIAG_CONTRACT)}  (engine=dry: not calling the LLM)")
    out = _engineer_call(inp, "diagnose", d)
    return f"diagnosis: {out.get('root_cause','?')[:100]}"


def ph_propose(c: Campaign, dry: bool, engine: str, **_) -> str:
    """Engineer call #2: fork the parent arm with exactly one delta; code validates."""
    d = c.gen_dir / "diagnosis" / "output.json"
    parent = c.state["current_arm"]
    child = f"{parent.split('__')[0]}__g{c.state['generation'] + 1:03d}"
    p = c.gen_dir / "proposal"; p.mkdir(exist_ok=True)
    inp = {"parent_arm_dir": str(CA / "exp_workspace" / parent), "child_arm": child,
           "diagnosis": json.loads(d.read_text()) if d.exists() else None,
           "contract": ["copy the parent folder to exp_workspace/" + child,
                        "change EXACTLY ONE thing (prompts.py text block, or an extra flag in exp.py)",
                        "register the cell under the child name; max_turns and all frozen knobs inherited",
                        "no coordinates / scene names / episode-specific content"]}
    _atomic_write(p / "input.json", inp)
    if engine == "dry" or dry:
        c.log(note="proposal input written (dry engine)")
        return (f"engineer call #2 prepared: fork {parent} -> {child} with ONE delta -> {p/'input.json'}"
                f"\n      validator afterwards: folder exists, diff touches prompts.py/exp.py only, forbidden-token scan")
    _engineer_call(inp, "propose", p)
    ok, why = validate_child(parent, child)
    if not ok:
        _append(c.root / "ledger" / "events.jsonl", {"rejected_candidate": child, "why": why})
        return f"candidate rejected by validator: {why}"
    c.state["candidate"] = child
    c.save()
    return f"candidate {child} accepted by validator"


def validate_child(parent: str, child: str) -> tuple[bool, str]:
    pd, cd = CA / "exp_workspace" / parent, CA / "exp_workspace" / child
    if not (cd / "exp.py").exists() or not (cd / "prompts.py").exists():
        return False, "child folder incomplete"
    changed = [f for f in ("prompts.py", "exp.py")
               if (pd / f).read_bytes() != (cd / f).read_bytes()]
    extra = [q.name for q in cd.iterdir() if q.name not in ("exp.py", "prompts.py", "README.md", "__pycache__")]
    if extra:
        return False, f"unexpected files in child: {extra}"
    txt = (cd / "prompts.py").read_text().lower()
    for bad in ("x=", "y=", "coordinate", "position (", "scene_id", "episode_id"):
        if bad in txt:
            return False, f"forbidden token {bad!r} in prompts.py"
    return True, f"changed: {changed}"


def ph_gate(c: Campaign, dry: bool, no_run: bool, **_) -> str:
    cand, parent = c.state.get("candidate"), c.state["current_arm"]
    if not cand:
        return "no candidate -> skip gate"
    crun = RUNS / cand
    eps = c.manifest["episodes"]
    if not (crun / "summary.json").exists() or load_board(crun)["n_total"] < _count_eps(eps):
        if dry or no_run:
            return f"WOULD RUN candidate board: stdrun.py run {cand} --episodes {eps}"
        subprocess.run([PY, str(CA / "stdrun.py"), "run", cand, "--episodes", eps], cwd=REPO, check=True,
                       env={**os.environ, "MINI_SERVE_CTX": os.environ.get("MINI_SERVE_CTX", "65536")})
    rep = compare(load_board(crun), load_board(RUNS / parent), c.manifest["alpha"])
    d = rep["discordant"]
    directional = (d["candidate_wins"] - d["parent_wins"]) >= c.manifest["screen_min_net_wins"]
    rep["decision"] = {"significant": rep["significant"], "directional": directional,
                       "passed": rep["significant"] or directional}
    _append(c.root / "ledger" / "gates.jsonl", {"gen": c.state["generation"], **rep})
    c.state["last_gate"] = rep["decision"]; c.save()
    return (f"gate {cand} vs {parent}: ΔSR {rep['sr']['delta']:+.3f}, discordant {d['total']} "
            f"({d['candidate_wins']}/{d['parent_wins']}), p={rep['p_one_sided_exact_mcnemar']} "
            f"-> {'PASS' if rep['decision']['passed'] else 'FAIL'}")


def ph_promote(c: Campaign, dry: bool, **_) -> str:
    cand = c.state.get("candidate")
    g = c.state.get("last_gate") or {}
    if cand and g.get("passed"):
        _append(c.root / "ledger" / "promotions.jsonl",
                {"gen": c.state["generation"], "from": c.state["current_arm"], "to": cand, "gate": g})
        c.state["lineage"].append(cand)
        c.state["current_arm"] = cand
        msg = f"PROMOTED {cand}"
    else:
        msg = f"kept {c.state['current_arm']} (candidate {cand} not promoted)"
    c.state["candidate"] = None
    c.state["generation"] += 1
    if c.state["generation"] >= c.manifest["max_generations"]:
        c.transition("COMPLETE", outcome="generation budget exhausted")
        return msg + " -> COMPLETE"
    c.save()
    return msg


def _engineer_call(inp: dict, stage: str, outdir: Path) -> dict:
    """Contract-bound Claude call (Agent SDK). Fresh process, read-only tools,
    output = one JSON object, validated by the caller. (sdk engine)"""
    raise NotImplementedError("engine=sdk: wire claude_agent_sdk here (see core/harnesses/claude_sdk.py); "
                              "dry engine writes input.json only")


def _count_eps(spec: str) -> int:
    n = 0
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); n += int(b) - int(a) + 1
        elif part.strip():
            n += 1
    return n


# ── CLI ──
def cmd_init(a) -> None:
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    _atomic_write(root / "manifest.json", {
        "parent_arm": a.parent, "episodes": a.episodes, "alpha": 0.025,
        "screen_min_net_wins": 3, "max_generations": a.max_generations, "engine": a.engine,
        "created": time.strftime("%F %T"), "note": "screening tier campaign; every number is a hypothesis"})
    _atomic_write(root / "state.json", {"phase": "ROLLOUT", "generation": 0,
                                        "current_arm": a.parent, "candidate": None,
                                        "lineage": [a.parent], "last_gate": None})
    print(f"initialised {root}  parent={a.parent} episodes={a.episodes}")


def cmd_step(a) -> None:
    c = Campaign(Path(a.root))
    ph = c.state["phase"]
    fn = {"ROLLOUT": ph_rollout, "PROFILE": ph_profile, "DIAGNOSE": ph_diagnose,
          "PROPOSE": ph_propose, "GATE": ph_gate, "PROMOTE": ph_promote}.get(ph)
    if fn is None:
        print(f"[{ph}] campaign complete: {c.state.get('outcome')}"); return
    msg = fn(c, dry=a.dry_run, engine=a.engine or c.manifest["engine"], no_run=a.no_run)
    print(f"[gen {c.state['generation']} · {ph}] {msg}")
    if ph == "PROMOTE" and c.state["phase"] == "COMPLETE":
        return
    if a.dry_run and ph in ("ROLLOUT", "GATE") and msg.startswith("WOULD RUN"):
        print("      (dry-run: not advancing past a phase that needs a board)")
        return
    c.transition(NEXT[ph])
    print(f"      -> next phase {c.state['phase']}")


def cmd_show(a) -> None:
    c = Campaign(Path(a.root))
    print(json.dumps({"manifest": c.manifest, "state": c.state}, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("root"); i.add_argument("--parent", required=True)
    i.add_argument("--episodes", default="0-24"); i.add_argument("--max-generations", type=int, default=5)
    i.add_argument("--engine", default="dry", choices=("dry", "sdk"))
    s = sub.add_parser("step"); s.add_argument("root"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--engine", default=None, choices=("dry", "sdk")); s.add_argument("--no-run", action="store_true")
    w = sub.add_parser("show"); w.add_argument("root")
    a = ap.parse_args()
    {"init": cmd_init, "step": cmd_step, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
