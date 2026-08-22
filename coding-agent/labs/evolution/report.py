"""report — human-readable summary of an eharness-evo campaign root (morning read)."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
st = json.loads((root / "state.json").read_text()); mf = json.loads((root / "manifest.json").read_text())
print(f"# campaign {root.name}   phase={st['phase']} gen={st['generation']} current_arm={st['current_arm']}")
print(f"lineage: {' -> '.join(st['lineage'])}   rejected: {st.get('rejected')}   episodes={mf['episodes']}")
gates = [json.loads(l) for l in (root / "ledger" / "gates.jsonl").read_text().splitlines()] if (root / "ledger" / "gates.jsonl").exists() else []
for g in gates:
    d = g["discordant"]
    print(f"- gate gen{g['gen']}: {g['candidate']} vs {g['parent']}  SR {g['sr']['candidate']:.2f} vs {g['sr']['parent']:.2f} "
          f"(Δ{g['sr']['delta']:+.3f})  discordant {d['total']} ({d['candidate_wins']}/{d['parent_wins']})  p={g['p_one_sided_exact_mcnemar']}  "
          f"-> {'PASS' if g['decision']['passed'] else 'FAIL'}")
for gd in sorted(root.glob("gen_*")):
    print(f"\n## {gd.name}")
    cl = gd / "clusters.json"
    if cl.exists():
        print("clusters:", ", ".join(f"{c['tag']}:{c['size']}" for c in json.loads(cl.read_text())))
    dg = gd / "diagnosis" / "output.json"
    if dg.exists():
        o = json.loads(dg.read_text())
        print("diagnosis:", str(o.get("root_cause"))[:300]); print("delta:", o.get("proposed_delta"))
    pr = gd / "proposal" / "output.json"
    if pr.exists():
        print("proposal:", json.loads(pr.read_text()))
ev = root / "ledger" / "events.jsonl"
if ev.exists():
    errs = [l for l in ev.read_text().splitlines() if '"error"' in l]
    print(f"\nevents: {len(ev.read_text().splitlines())} lines, errors: {len(errs)}")
    for e in errs[-3:]:
        print("  ", e[:200])
