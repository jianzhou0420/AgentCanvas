"""supervisor — the eharness-evo campaign program (what "self-evolution" IS).

A plain state machine over a campaign directory. Each `step` does ONE bounded
phase, writes state.json, and returns; `run` loops `step` unattended. The LLM
engineer is not a session: it is called at most twice per generation
(DIAGNOSE, PROPOSE), each as a fresh, contract-bound Claude Agent SDK call whose
output is validated by code. Everything else — rollouts, profiling, clustering,
gates, promotion, rollback — is deterministic Python.

    ROLLOUT -> PROFILE -> DIAGNOSE -> PROPOSE -> GATE -> PROMOTE -> ROLLOUT ...
                                                           \\-> COMPLETE (budget)

  python supervisor.py init ROOT --parent ARM [--candidate ARM] [--episodes 0-49] [--max-generations 4]
  python supervisor.py step ROOT [--dry-run] [--engine dry|sdk]
  python supervisor.py run  ROOT [--engine sdk] [--max-hours 12]
  python supervisor.py show ROOT
  python supervisor.py rollback ROOT --to ARM --reason "..."

Campaign root (append-only except state.json):
  manifest.json  state.json  gen_NNN/{profile,clusters}.json
  gen_NNN/diagnosis/{input,output}.json + attempt-N/  gen_NNN/proposal/{input,output}.json + attempt-N/
  ledger/{events,gates,promotions}.jsonl
Arms are never edited: a child is a NEW exp_workspace folder; promotion moves
state.current_arm; rollback moves it back. That is the whole trick.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CA = REPO / "coding-agent"
RUNS = REPO / "outputs" / "beta-react-harness"
PY = os.environ.get("EVO_PYTHON", os.path.expanduser("~/miniconda3/envs/agentcanvas/bin/python"))
ENGINEER_MODEL = os.environ.get("EVO_ENGINEER_MODEL", "claude-opus-5")
sys.path.insert(0, str(HERE))
from paired_gate import compare, load_board          # noqa: E402
from profile_episodes import load as load_eps, profile  # noqa: E402

PHASES = ["ROLLOUT", "PROFILE", "DIAGNOSE", "PROPOSE", "GATE", "PROMOTE", "COMPLETE"]
NEXT = {"ROLLOUT": "PROFILE", "PROFILE": "DIAGNOSE", "DIAGNOSE": "PROPOSE",
        "PROPOSE": "GATE", "GATE": "PROMOTE", "PROMOTE": "ROLLOUT"}
ORGAN_FLAGS = {
    "blocked_signal": "step result reports moved_m / forward_blocked / collided_last (+factual note)",
    "turn_macros": "step accepts 4/5/6 = left90/right90/turn-around and reports turned_deg",
    "memo": "remember(text) tool; the model's own notes are echoed in every step/observe result",
    "revisit": "observe() says when the current view matches one seen >=8 steps ago",
}


# ── persistence primitives ──
def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"t": time.strftime("%F %T"), **row}, ensure_ascii=False) + "\n")


def _eps(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _fmt_eps(idx: list[int]) -> str:
    idx = sorted(set(idx)); parts = []
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        parts.append(f"{idx[i]}-{idx[j]}" if j > i else str(idx[i]))
        i = j + 1
    return ",".join(parts)


def _done_indices(run_dir: Path) -> set[int]:
    f = run_dir / "summary.json"
    if not f.exists():
        return set()
    d = json.loads(f.read_text())
    return {int(e["index"]) for e in d.get("episodes", []) if (e.get("metrics") or {}).get("success") is not None}


def _wait_idle(cell: str) -> None:
    """Never run two boards at once (one env server, one GPU)."""
    while True:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        if "stdrun.py run " not in ps:
            return
        time.sleep(60)


def _run_board(cell: str, missing: list[int], log) -> None:
    _wait_idle(cell)
    cmd = [PY, str(CA / "stdrun.py"), "run", cell, "--episodes", _fmt_eps(missing)]
    log(cmd=" ".join(cmd))
    env = {**os.environ, "MINI_SERVE_CTX": os.environ.get("MINI_SERVE_CTX", "65536")}
    subprocess.run(cmd, cwd=REPO, check=True, env=env)


class Campaign:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.state = json.loads((root / "state.json").read_text())

    def save(self) -> None:
        _atomic_write(self.root / "state.json", self.state)

    def log(self, **row) -> None:
        _append(self.root / "ledger" / "events.jsonl",
                {"gen": self.state["generation"], "phase": self.state["phase"], **row})

    @property
    def gen_dir(self) -> Path:
        d = self.root / f"gen_{self.state['generation']:03d}"
        d.mkdir(exist_ok=True)
        return d

    def transition(self, to: str, **updates) -> None:
        self.state.update(updates); self.state["phase"] = to; self.save()


# ── phases ──
def ph_rollout(c: Campaign, dry: bool, **_) -> str:
    arm = c.state["current_arm"]
    want = _eps(c.manifest["episodes"])
    if not dry:
        _wait_idle(arm)  # another board may still be writing this run dir
    missing = sorted(set(want) - _done_indices(RUNS / arm))
    if not missing:
        c.log(note=f"{arm}: board complete ({len(want)} eps)")
        return f"board complete: {arm} ({len(want)} eps)"
    if dry:
        return f"WOULD RUN: stdrun.py run {arm} --episodes {_fmt_eps(missing)}"
    _run_board(arm, missing, c.log)
    return f"ran {arm} on {_fmt_eps(missing)}"


def ph_profile(c: Campaign, dry: bool, **_) -> str:
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
    _atomic_write(c.gen_dir / "clusters.json", [{"tag": t, "members": m, "size": len(m)} for t, m in ranked])
    sr = sum(1 for r in rows if r["success"] == 1.0) / max(1, len(rows))
    c.log(sr=round(sr, 3), n=len(rows), clusters=[(t, len(m)) for t, m in ranked])
    return f"{arm}: {len(rows)} eps SR={sr:.2f}; clusters " + ", ".join(f"{t}:{len(m)}" for t, m in ranked)


DIAG_CONTRACT = {
    "cluster_tag": "<copy from input>",
    "outcome": "what happens, observable",
    "immediate_trigger": "the event right before it goes wrong",
    "root_cause": "ONE mechanism, or 'inconclusive: ...'",
    "competing_hypotheses": ["at least two distinct alternatives"],
    "evidence": ["ep N step S: <quote or frame file>", "..."],
    "falsifier": "what observation would refute root_cause",
    "proposed_delta": {"tier": "T1|T2|T3|T4",
                       "one_line": "exactly ONE change to the arm (a skill text edit, or toggling one organ flag)"},
    "confidence": 0.0,
}

DIAG_SYSTEM = (
    "You are the evolution engineer of a navigation harness for a frozen small model "
    "(Qwen3.5-9B, bare front-view + step tool). You diagnose ONE failure cluster from the "
    "archives and answer with exactly one JSON object (no prose around it). Read-only: do "
    "not modify any file. Distinguish outcome / immediate trigger / root cause; give at "
    "least two competing hypotheses; 'inconclusive' is a legal root_cause. Never propose "
    "coordinates, scene names or episode-specific content. The proposed delta must be ONE "
    "change: a skill-text edit in prompts.py, or toggling one organ flag in exp.py "
    f"(available flags: {json.dumps(ORGAN_FLAGS)})."
)

PROPOSE_SYSTEM = (
    "You are the evolution engineer. Create the child arm folder described in the input "
    "JSON by COPYING the parent folder's prompts.py and exp.py and applying EXACTLY ONE "
    "change that implements the diagnosis's proposed_delta. Allowed changes: (a) edit one "
    "skill paragraph / add one short paragraph in prompts.py SKILLS_BLOCK (English, no "
    "coordinates, no scene names); or (b) add/remove ONE organ flag in exp.py extras "
    f"(available: {list(ORGAN_FLAGS)}). In exp.py set name=<child_arm> and keep "
    "max_turns=200 and everything else. Do not touch any other file in the repository. "
    "Do not run boards or servers. After writing, verify registration with Bash: "
    "<python> -c \"import sys; sys.path.insert(0,'coding-agent'); from core.cells import get_cell; "
    "print(get_cell('<child_arm>'))\" using the python path given in the input. Finish with "
    "exactly one JSON object: {\"child_arm\": ..., \"delta_tier\": ..., \"delta_summary\": ..., "
    "\"files_changed\": [...]}."
)


def _largest_json(text: str) -> dict | None:
    best = None
    for m in re.finditer(r"\{", text):
        try:
            obj, end = json.JSONDecoder().raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and (best is None or end > best[1]):
            best = (obj, end)
    return best[0] if best else None


def _engineer_call(inp: dict, stage: str, outdir: Path, system: str, tools: list[str],
                   max_turns: int, required: list[str]) -> dict:
    """One contract-bound Claude Agent SDK call. Fresh client, bounded turns,
    transcript + output persisted under outdir/attempt-N/. Validated; one retry."""
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
                                  TextBlock, ToolUseBlock)
    _atomic_write(outdir / "input.json", inp)
    attempts = sorted(outdir.glob("attempt-*"))
    for n in range(len(attempts), len(attempts) + 2):
        adir = outdir / f"attempt-{n}"; adir.mkdir(parents=True, exist_ok=True)
        prompt = (f"Stage: {stage}. The input contract is in {outdir/'input.json'} — read it first. "
                  f"Output schema is its 'output_schema' field (if present). "
                  f"When done, reply with the single JSON object only.")
        if n > len(attempts):
            prompt += " Previous attempt did not end with a valid JSON object — this time reply with ONLY the JSON."

        async def _one() -> str:
            opts = ClaudeAgentOptions(system_prompt=system, setting_sources=[],
                                      tools=tools, allowed_tools=tools, permission_mode="bypassPermissions",
                                      max_turns=max_turns, max_budget_usd=5.0,
                                      model=ENGINEER_MODEL, cwd=str(REPO))
            texts: list[str] = []
            with (adir / "transcript.jsonl").open("w") as tr:
                async with ClaudeSDKClient(options=opts) as client:
                    await client.query(prompt)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for b in message.content:
                                if isinstance(b, TextBlock):
                                    texts.append(b.text)
                                    tr.write(json.dumps({"text": b.text[:4000]}, ensure_ascii=False) + "\n")
                                elif isinstance(b, ToolUseBlock):
                                    tr.write(json.dumps({"tool": b.name, "input": json.loads(json.dumps(b.input, default=str))[:1] if False else str(b.input)[:600]}, ensure_ascii=False) + "\n")
                        tr.flush()
            return "\n".join(texts)

        text = asyncio.run(_one())
        (adir / "final_text.txt").write_text(text)
        out = _largest_json(text)
        if out and all(k in out for k in required):
            _atomic_write(outdir / "output.json", {"attempt": n, **out})
            return out
        _append(outdir / "rejected.jsonl", {"attempt": n, "why": "no valid JSON with required keys",
                                             "tail": text[-300:]})
    raise RuntimeError(f"{stage}: engineer produced no valid output after 2 attempts")


def ph_diagnose(c: Campaign, dry: bool, engine: str, **_) -> str:
    clusters = json.loads((c.gen_dir / "clusters.json").read_text())
    if not clusters:
        c.log(note="no failure clusters"); return "no failures to diagnose"
    top = clusters[0]
    rows = {r["ep"]: r for r in json.loads((c.gen_dir / "profile.json").read_text())}
    arm = c.state["current_arm"]
    inp = {
        "campaign": c.root.name, "generation": c.state["generation"], "arm": arm,
        "arm_dir": str(CA / "exp_workspace" / arm),
        "cluster": top, "member_profiles": [rows[e] for e in top["members"]],
        "evidence_paths": {str(e): {"transcript_jsonl": str(RUNS / arm / f"episode_{e}.jsonl"),
                                    "frames_dir": str(RUNS / arm / f"live_{e}")} for e in top["members"]},
        "how_to_read": "transcript lines are JSON with kind in {thinking, tool_use, tool_result, ...}; "
                       "frames are live_N/obs_XXXX_stepSSS.png (view at least two with Read)",
        "read_budget": 12, "output_schema": DIAG_CONTRACT,
        "rules": ["read >=2 member transcripts and >=2 frames before answering",
                  "root_cause may be 'inconclusive'", "no scene names / coordinates in the delta",
                  "proposed_delta = ONE change (skill text edit, or one organ flag)"],
    }
    d = c.gen_dir / "diagnosis"
    if (d / "output.json").exists():
        return "diagnosis already present (resume)"
    if engine == "dry" or dry:
        _atomic_write(d / "input.json", inp); c.log(note="diagnosis input written (dry)")
        return f"engineer call #1 prepared: cluster '{top['tag']}' x{top['size']} (dry)"
    out = _engineer_call(inp, "diagnose", d, DIAG_SYSTEM, ["Read", "Glob", "Grep"], 40,
                         ["root_cause", "proposed_delta"])
    c.log(diagnosis=out.get("root_cause", "")[:200], delta=out.get("proposed_delta"))
    return f"diagnosis: {str(out.get('root_cause',''))[:120]} | delta: {out.get('proposed_delta')}"


def ph_propose(c: Campaign, dry: bool, engine: str, **_) -> str:
    if c.state.get("candidate"):
        c.log(note=f"candidate pre-supplied: {c.state['candidate']}")
        return f"candidate pre-supplied: {c.state['candidate']} (skipping engineer call #2)"
    parent = c.state["current_arm"]
    child = f"{c.manifest['child_prefix']}{c.state['generation'] + 1:02d}"
    p = c.gen_dir / "proposal"
    d = c.gen_dir / "diagnosis" / "output.json"
    inp = {"parent_arm": parent, "parent_arm_dir": str(CA / "exp_workspace" / parent),
           "child_arm": child, "child_arm_dir": str(CA / "exp_workspace" / child),
           "python": PY, "diagnosis": json.loads(d.read_text()) if d.exists() else None,
           "organ_flags": ORGAN_FLAGS,
           "contract": ["copy parent prompts.py + exp.py into child_arm_dir",
                        "EXACTLY ONE change implementing diagnosis.proposed_delta",
                        "exp.py: name=child_arm, max_turns=200, extras inherit (+/- one flag at most)",
                        "no other repository file may change; do not run boards"]}
    if engine == "dry" or dry:
        _atomic_write(p / "input.json", inp); c.log(note="proposal input written (dry)")
        return f"engineer call #2 prepared: fork {parent} -> {child} (dry)"
    out = _engineer_call(inp, "propose", p, PROPOSE_SYSTEM,
                         ["Read", "Glob", "Grep", "Write", "Edit", "Bash"], 60, ["child_arm"])
    ok, why = validate_child(parent, child)
    c.log(proposal=out, validator=why, ok=ok)
    if not ok:
        _append(c.root / "ledger" / "events.jsonl", {"rejected_candidate": child, "why": why})
        return f"candidate {child} REJECTED by validator: {why}"
    c.state["candidate"] = child; c.save()
    return f"candidate {child} accepted: {why}"


def validate_child(parent: str, child: str) -> tuple[bool, str]:
    pd, cd = CA / "exp_workspace" / parent, CA / "exp_workspace" / child
    if not (cd / "exp.py").exists() or not (cd / "prompts.py").exists():
        return False, "child folder incomplete"
    extra = [q.name for q in cd.iterdir() if q.name not in ("exp.py", "prompts.py", "README.md", "__pycache__")]
    if extra:
        return False, f"unexpected files in child: {extra}"
    changed = [f for f in ("prompts.py", "exp.py") if (pd / f).read_bytes() != (cd / f).read_bytes()]
    if not changed:
        return False, "child identical to parent"
    txt = (cd / "prompts.py").read_text().lower()
    for bad in ("coordinate", "scene_id", "episode_id", "x=", "y="):
        if bad in txt:
            return False, f"forbidden token {bad!r} in prompts.py"
    r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'coding-agent'); from core.cells import get_cell; "
                        f"s=get_cell('{child}'); assert s.max_turns==200; print(s.extra)"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"registration failed: {r.stderr[-300:]}"
    # git hygiene: nothing outside the child folder may have changed
    g = subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True, text=True).stdout
    stray = [l for l in g.splitlines() if l.strip() and f"exp_workspace/{child}/" not in l
             and not l.startswith("??") and "labs/evolution" not in l]
    if stray:
        return False, f"stray modifications outside child: {stray[:5]}"
    return True, f"changed {changed}; extras {r.stdout.strip()}"


def ph_gate(c: Campaign, dry: bool, **_) -> str:
    cand, parent = c.state.get("candidate"), c.state["current_arm"]
    if not cand:
        return "no candidate -> skip gate"
    want = _eps(c.manifest["episodes"])
    if not dry:
        _wait_idle(cand)
    missing = sorted(set(want) - _done_indices(RUNS / cand))
    if missing:
        if dry:
            return f"WOULD RUN candidate board: {cand} --episodes {_fmt_eps(missing)}"
        _run_board(cand, missing, c.log)
    rep = compare(load_board(RUNS / cand), load_board(RUNS / parent), c.manifest["alpha"])
    d = rep["discordant"]
    directional = (d["candidate_wins"] - d["parent_wins"]) >= c.manifest["screen_min_net_wins"]
    rep["decision"] = {"significant": rep["significant"], "directional": directional,
                       "passed": bool(rep["significant"] or directional)}
    _append(c.root / "ledger" / "gates.jsonl", {"gen": c.state["generation"], **rep})
    c.state["last_gate"] = rep["decision"]; c.save()
    return (f"gate {cand} vs {parent}: SR {rep['sr']['candidate']:.2f} vs {rep['sr']['parent']:.2f} "
            f"(Δ{rep['sr']['delta']:+.3f}), discordant {d['total']} ({d['candidate_wins']}/{d['parent_wins']}), "
            f"p={rep['p_one_sided_exact_mcnemar']} -> {'PASS' if rep['decision']['passed'] else 'FAIL'}")


def ph_promote(c: Campaign, dry: bool, **_) -> str:
    cand = c.state.get("candidate"); g = c.state.get("last_gate") or {}
    if cand and g.get("passed"):
        _append(c.root / "ledger" / "promotions.jsonl",
                {"gen": c.state["generation"], "from": c.state["current_arm"], "to": cand, "gate": g})
        c.state["lineage"].append(cand); c.state["current_arm"] = cand
        msg = f"PROMOTED {cand}"
    else:
        c.state.setdefault("rejected", []).append(cand)
        msg = f"kept {c.state['current_arm']} (candidate {cand} not promoted)"
    c.state["candidate"] = None; c.state["last_gate"] = None
    c.state["generation"] += 1
    if c.state["generation"] >= c.manifest["max_generations"]:
        c.transition("COMPLETE", outcome="generation budget exhausted"); return msg + " -> COMPLETE"
    c.save(); return msg


PHASE_FN = {"ROLLOUT": ph_rollout, "PROFILE": ph_profile, "DIAGNOSE": ph_diagnose,
            "PROPOSE": ph_propose, "GATE": ph_gate, "PROMOTE": ph_promote}


def do_step(root: Path, dry: bool, engine: str | None) -> bool:
    """Returns True if the campaign can continue."""
    c = Campaign(root); ph = c.state["phase"]
    if ph == "COMPLETE":
        print(f"[COMPLETE] {c.state.get('outcome')}"); return False
    engine = engine or c.manifest["engine"]
    msg = PHASE_FN[ph](c, dry=dry, engine=engine)
    print(f"[{time.strftime('%H:%M')} gen {c.state['generation']} · {ph}] {msg}", flush=True)
    c.log(result=msg[:300])
    if c.state["phase"] == "COMPLETE":
        return False
    if dry and msg.startswith("WOULD RUN"):
        print("      (dry-run stops at a phase that needs a board)"); return False
    c.transition(NEXT[ph]); return True


def cmd_init(a) -> None:
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    _atomic_write(root / "manifest.json", {
        "parent_arm": a.parent, "episodes": a.episodes, "alpha": 0.025, "screen_min_net_wins": 3,
        "max_generations": a.max_generations, "engine": a.engine, "child_prefix": a.child_prefix,
        "engineer_model": ENGINEER_MODEL, "created": time.strftime("%F %T")})
    _atomic_write(root / "state.json", {"phase": "ROLLOUT", "generation": 0, "current_arm": a.parent,
                                        "candidate": a.candidate, "lineage": [a.parent],
                                        "rejected": [], "last_gate": None})
    print(f"initialised {root}: parent={a.parent} candidate={a.candidate} episodes={a.episodes}")


def cmd_run(a) -> None:
    t_end = time.time() + a.max_hours * 3600
    fails = 0
    while time.time() < t_end:
        try:
            if not do_step(Path(a.root), dry=False, engine=a.engine):
                break
            fails = 0
        except Exception as ex:  # noqa: BLE001 — overnight robustness: log, back off, retry
            fails += 1
            _append(Path(a.root) / "ledger" / "events.jsonl", {"error": repr(ex)[:500], "fails": fails})
            print(f"[error] {ex!r} (fail {fails}/3)", flush=True)
            if fails >= 3:
                c = Campaign(Path(a.root)); c.transition("COMPLETE", outcome=f"aborted: {ex!r}"[:300])
                break
            time.sleep(120)


def cmd_rollback(a) -> None:
    c = Campaign(Path(a.root))
    assert a.to in c.state["lineage"], f"{a.to} not in lineage {c.state['lineage']}"
    _append(c.root / "ledger" / "promotions.jsonl",
            {"gen": c.state["generation"], "rollback_from": c.state["current_arm"], "to": a.to, "reason": a.reason})
    c.state["current_arm"] = a.to; c.state["candidate"] = None; c.state["last_gate"] = None
    c.transition("ROLLOUT")
    print(f"rolled back to {a.to}; next phase ROLLOUT")


def cmd_show(a) -> None:
    c = Campaign(Path(a.root))
    print(json.dumps({"manifest": c.manifest, "state": c.state}, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("root"); i.add_argument("--parent", required=True)
    i.add_argument("--candidate", default=None); i.add_argument("--episodes", default="0-49")
    i.add_argument("--max-generations", type=int, default=4); i.add_argument("--engine", default="sdk", choices=("dry", "sdk"))
    i.add_argument("--child-prefix", default="evo9b_auto_g")
    s = sub.add_parser("step"); s.add_argument("root"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--engine", default=None, choices=("dry", "sdk"))
    r = sub.add_parser("run"); r.add_argument("root"); r.add_argument("--engine", default=None, choices=("dry", "sdk"))
    r.add_argument("--max-hours", type=float, default=14)
    w = sub.add_parser("show"); w.add_argument("root")
    b = sub.add_parser("rollback"); b.add_argument("root"); b.add_argument("--to", required=True); b.add_argument("--reason", default="")
    a = ap.parse_args()
    {"init": cmd_init, "step": lambda a: do_step(Path(a.root), a.dry_run, a.engine), "run": cmd_run,
     "show": cmd_show, "rollback": cmd_rollback}[a.cmd](a)


if __name__ == "__main__":
    main()
