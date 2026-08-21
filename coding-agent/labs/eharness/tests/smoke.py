"""eharness smoke — the whole shell end-to-end, no LLM, no habitat.

Run:  cd coding-agent && python -m eharness.smoke

What it proves (plumbing, not navigation): a scripted planner delegates over a
fake wp-style env; the sub-session (scripted too, driven straight through the
wrapper) observes / moves / writes state; the V0 stall guard trips on repeated
identical views and escalates into a preempted receipt; the pre-stop V1 gate
(scripted judge) vetoes the first finish and admits the second; receipts are
judged and committed; the monotone-retention assert holds; every memory file
lands under live_dir. Exits non-zero on any failed check.
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # coding-agent/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harnesses" / "mini"))  # toolset import

from PIL import Image as PILImage  # noqa: E402

from eharness.frames import ahash, hamming  # noqa: E402
from eharness.planner import Planner  # noqa: E402
from eharness.receipts import Receipt  # noqa: E402
from eharness.state_block import StateBlock  # noqa: E402
from eharness.wrapper import HarnessedToolset  # noqa: E402
from toolset import ToolResult, png_part, text_part  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    CHECKS.append((name, ok))
    print(("  ok  " if ok else "  FAIL") + " · " + name)


def scene_png(seed: int) -> bytes:
    """Deterministic 64×64 'view' — different seeds → visually different."""
    img = PILImage.new("RGB", (64, 64))
    px = img.load()
    for x in range(64):
        for y in range(64):
            px[x, y] = ((x * (seed + 3)) % 256, (y * (seed * 7 + 1)) % 256,
                        (x * y * (seed + 1)) % 256)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeWpToolset:
    """Minimal wp-surface stand-in: observe / goto / stop, scene changes with
    position; a 'wall' region where goto stops changing the view."""

    def __init__(self, live_dir: Path) -> None:
        self.live_dir = live_dir
        self.calls_by_tool = {"observe": 0, "goto": 0, "stop": 0}
        self.steps_taken = 0
        self.episode_over = False
        self.end_reason = None
        self.pos = 0
        self.stuck = False   # test hook: freeze the scene → stall guard food

    def tool_schemas(self):
        return [
            {"name": "observe", "description": "look", "input_schema":
                {"type": "object", "properties": {}}},
            {"name": "goto", "description": "move", "input_schema":
                {"type": "object", "properties": {"waypoint": {"type": "integer"}},
                 "required": ["waypoint"]}},
            {"name": "stop", "description": "stop", "input_schema":
                {"type": "object", "properties": {}}},
        ]

    def execute(self, name: str, args: dict) -> ToolResult:
        self.calls_by_tool[name] = self.calls_by_tool.get(name, 0) + 1
        if name == "observe":
            return ToolResult(
                content=[png_part(scene_png(self.pos)),
                         text_part(json.dumps({"waypoints": {"1": "Front"},
                                               "steps_taken_total": self.steps_taken}))],
                info={"kind": "observe"})
        if name == "goto":
            if not self.stuck:
                self.pos += 1
            self.steps_taken += 9
            return ToolResult(
                content=[png_part(scene_png(self.pos)),
                         text_part(json.dumps({"steps_taken_total": self.steps_taken,
                                               "episode_over": False}))],
                info={"kind": "goto", "steps_taken_total": self.steps_taken,
                      "episode_over": False})
        if name == "stop":
            self.episode_over = True
            self.end_reason = "stop_called"
            self.steps_taken += 1
            return ToolResult(
                content=[text_part(json.dumps({"episode_over": True,
                                               "end_reason": "stop_called"}))],
                info={"kind": "stop", "episode_over": True,
                      "end_reason": "stop_called",
                      "steps_taken_total": self.steps_taken})
        return ToolResult(content=[text_part("unknown")], info={"error": name})


def scripted_judge(verdicts: list[bool]):
    """Pops one verdict per call; True = supported."""
    calls: list[str] = []

    def fake_judge(claim, context, images, *, model_name, model_kwargs=None):
        calls.append(claim)
        v = verdicts.pop(0) if verdicts else True
        return v, ("looks right" if v else "view does not match the goal")
    fake_judge.calls = calls
    return fake_judge


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="eh_smoke_"))
    live = tmp / "live_0"
    events: list[tuple[str, dict]] = []

    # ── unit level ──────────────────────────────────────────────
    a, b = scene_png(1), scene_png(2)
    check("ahash separates different scenes", hamming(ahash(a), ahash(b)) > 8)
    check("ahash stable on identical scene", hamming(ahash(a), ahash(scene_png(1))) == 0)

    sb = StateBlock(path=tmp / "state.json")
    sb.instruction = "walk down the hallway past the stairs, then enter the room on the right and stop next to the sofa"
    sb.sub_instructions = ["walk down the hallway past the stairs",
                           "enter the room on the right",
                           "stop next to the sofa"]
    sb.apply_update({"landmark": "the stairs", "progress_note": "in the hallway"},
                    source="test", frame=0)
    sb.apply_update({"dead_end": "left door is a closet"}, source="test", frame=1)
    before = sb.snapshot()
    sb.apply_update({"advance_subgoal": True, "progress_note": "past the stairs"},
                    source="test", frame=2)
    StateBlock.assert_monotone(before, sb.snapshot())
    check("state: landmark + negative fact + advance", sb.cursor == 1
          and "the stairs" in sb.landmarks and len(sb.negative_facts) == 1)
    try:
        StateBlock.assert_monotone(sb.snapshot(),
                                   {"visited": set(), "landmarks": set(),
                                    "negative_facts": set()})
        check("monotone assert catches loss", False)
    except AssertionError:
        check("monotone assert catches loss", True)
    sb2 = StateBlock.load(tmp / "state.json")
    check("state survives disk round-trip", sb2.cursor == 1
          and "the stairs" in sb2.landmarks)

    # ── the shell, end to end ───────────────────────────────────
    import eharness.wrapper as wmod
    fake_judge = scripted_judge([False, True, True])  # veto 1st stop; pass rest
    wmod.run_judge = fake_judge          # patch the V1 seam

    def fake_judge_stop(claim, context, images, *, model_name,
                        model_kwargs=None):
        v, why = fake_judge(claim, context, images, model_name=model_name,
                            model_kwargs=model_kwargs)
        return v, why, ("" if v else "move closer to the goal, then stop")
    wmod.run_judge_stop = fake_judge_stop  # pre-stop gate seam (v2.9)
    wmod.run_judge_milestone = (           # milestone seam (v3.0): all evidenced
        lambda sub, images, *, model_name, model_kwargs=None, **_kw:
        (True, "scripted evidence"))

    inner = FakeWpToolset(live)
    state = StateBlock(path=live / "state.json")
    state.instruction = sb.instruction
    state.sub_instructions = list(sb.sub_instructions)
    state.save()
    w = HarnessedToolset(inner, state=state, live_dir=live,
                         judge_model="scripted", emit=lambda k, p: events.append((k, p)))

    # schemas: state fields ride every tool; recall present; finish_subgoal gated
    schemas = {s["name"]: s for s in w.tool_schemas()}
    check("schemas extended with state fields",
          "landmark" in schemas["goto"]["input_schema"]["properties"])
    check("recall registered, finish_subgoal gated off outside subgoal",
          "recall" in schemas and "finish_subgoal" not in schemas)

    # sub-session 1: moves + decision-coupled writes + finish_subgoal
    w.begin_subgoal("walk down the hallway past the stairs", budget=40)
    check("finish_subgoal appears inside a subgoal",
          any(s["name"] == "finish_subgoal" for s in w.tool_schemas()))
    w.execute("observe", {})
    r = w.execute("goto", {"waypoint": 1, "landmark": "the stairs",
                           "progress_note": "hallway, stairs on my left"})
    check("state render appended to tool result",
          # §14.2: the injected copy carries a version header
          # ("[STATE v1 — supersedes …]"); the bare "[STATE]" form is the
          # mini/solo path only — accept the family, not one spelling
          any(p.get("text", "").lstrip().startswith("[STATE")
              for p in r.content
              if isinstance(p, dict) and p.get("type") == "text"))
    w.execute("goto", {"waypoint": 1, "advance_subgoal": True,
                       "progress_note": "past the stairs now"})
    w.execute("finish_subgoal", {"claim": "reached",
                                 "what_i_see": "end of hallway"})
    rec1 = w.end_subgoal("subgoal_done")
    check("receipt 1 reached, steps counted", rec1.claim == "reached"
          and rec1.steps_used == 18 and rec1.turns_used == 0)
    check("keyframes promoted on events",
          len(w.frames.keyframes()) >= 2)
    check("heartbeat file written", (live / "heartbeat.json").exists())

    # sub-session 2: freeze the scene → stall guard must trip and escalate
    inner.stuck = True
    w.begin_subgoal("enter the room on the right", budget=40)
    preempted = False
    for _ in range(12):
        res = w.execute("goto", {"waypoint": 1})
        info = res.info if isinstance(res.info, dict) else {}
        if info.get("preempt"):
            preempted = True
            break
        w.execute("observe", {})
    check("stall guard trips and escalates", preempted)
    rec2 = w.end_subgoal("preempted")
    check("preempted receipt claims blocked", rec2.claim == "blocked")
    guard_events = [p for k, p in events if k == "guard"]
    check("guard events emitted", len(guard_events) >= 1)

    # recall by landmark name returns the tagged frame (stage3 P0-3:
    # explicit schema — kind names the index, no string guessing)
    rr = w.execute("recall", {"kind": "landmark", "query": "stairs"})
    check("recall(kind='landmark', query='stairs') returns image + caption",
          any(p.get("type") == "image_url" for p in rr.content))
    rr_legacy = w.execute("recall", {"query": "stairs"})
    check("legacy one-string recall still resolves",
          any(p.get("type") == "image_url" for p in rr_legacy.content))

    # pre-stop gate: first stop vetoed (scripted judge False), second passes
    inner.stuck = False
    veto = w.execute("stop", {})
    vinfo = veto.info if isinstance(veto.info, dict) else {}
    check("first stop vetoed by V1", vinfo.get("stop_vetoed") is True
          and not inner.episode_over)
    done = w.execute("stop", {})
    check("second stop executes", inner.episode_over
          and done.info.get("episode_over") is True)
    check("verdict events emitted",
          any(k == "verdict" for k, _ in events))

    # ── planner loop over a fresh episode (scripted completion) ──
    events2: list[tuple[str, dict]] = []
    inner2 = FakeWpToolset(live)
    state2 = StateBlock(path=live / "state.json")
    state2.instruction = sb.instruction
    state2.sub_instructions = list(sb.sub_instructions)
    w2 = HarnessedToolset(inner2, state=state2, live_dir=live,
                          judge_model="scripted",
                          emit=lambda k, p: events2.append((k, p)))

    class Msg:
        def __init__(self, name, args):
            self.content = ""
            self.tool_calls = [type("TC", (), {
                "id": "tc1",
                "function": type("F", (), {"name": name,
                                           "arguments": json.dumps(args)})()})()]

    class Resp:
        def __init__(self, name, args):
            self.choices = [type("C", (), {"message": Msg(name, args)})()]

    script = [("delegate", {"subgoal": "walk down the hallway", "budget": 30}),
              ("recall", {"query": "stairs"}),
              ("verify", {"claim": "the robot is near the sofa"}),
              ("finish", {})]

    def fake_complete(**kwargs):
        name, args = script.pop(0) if script else ("finish", {})
        return Resp(name, args)

    def scripted_sub(*, subgoal, budget, turn_cap):
        # a sub-session that just walks and reports — through the wrapper
        w2.begin_subgoal(subgoal, budget)
        w2.execute("observe", {})
        w2.execute("goto", {"waypoint": 1, "landmark": "the stairs"})
        w2.execute("finish_subgoal", {"claim": "reached", "what_i_see": "hallway end"})
        return w2.end_subgoal("subgoal_done"), 0.01, 3

    planner = Planner(w2, state2, model_name="scripted", model_kwargs={},
                      run_subgoal_fn=scripted_sub,
                      emit=lambda k, p: events2.append((k, p)),
                      step_budget=500, complete_fn=fake_complete)
    summary = planner.run()
    check("planner: episode ends via finish", inner2.episode_over)
    check("planner: receipt committed + judged",
          len(planner.receipts) == 1 and planner.receipts[0].verdict == "pass")
    check("planner: summary counts sane",
          summary["planner_turns"] == 4 and summary["receipts"] == 1)
    check("receipts.jsonl + keyframes.jsonl + state.json on disk",
          all((live / f).exists() for f in
              ("receipts.jsonl", "keyframes.jsonl", "state.json")))

    # O(1) render: state render bounded regardless of activity
    check("state render bounded", len(state2.render_full()) < 3500)

    # ── solo: SoloModel compaction (no LLM — only message preparation) ──
    from eharness.compactor import SoloModel
    events3: list[tuple[str, dict]] = []
    inner3 = FakeWpToolset(live)
    state3 = StateBlock(path=live / "state.json")
    state3.instruction = sb.instruction
    state3.sub_instructions = list(sb.sub_instructions)
    w3 = HarnessedToolset(inner3, state=state3, live_dir=live,
                          judge_model=None, inject_state=False,
                          emit=lambda k, p: events3.append((k, p)))
    msgs: list[dict] = [{"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"}]
    for i in range(10):
        r = w3.execute("goto", {"waypoint": 1,
                                **({"landmark": "the stairs"} if i == 2 else {})})
        msgs.append({"role": "assistant", "content": f"thinking {i}",
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "goto",
                                                  "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": list(r.content)})
    check("frame markers ride tool results",
          any(isinstance(p, dict) and p.get("text", "").startswith("[frame#")
              for m in msgs if isinstance(m.get("content"), list)
              for p in m["content"]))
    model3 = SoloModel(harness=w3, emit=lambda k, p: events3.append((k, p)),
                       model_name="scripted", tools=w3.tool_schemas(),
                       image_window=2, compact_at=800, tail_msgs=4,
                       reattach_k=1, model_kwargs={},
                       cost_tracking="ignore_errors",
                       legacy_l2=True)   # this section smokes the LEGACY cut;
                                         # the §12 path has its own suite
    prepared = model3._prepare_messages_for_api(msgs)
    n_img = sum(1 for m in prepared if isinstance(m.get("content"), list)
                for pt in m["content"]
                if isinstance(pt, dict) and pt.get("type") == "image_url")
    texts = " ".join(str(pt.get("text", "")) for m in prepared
                     if isinstance(m.get("content"), list)
                     for pt in m["content"] if isinstance(pt, dict))
    check("solo: ephemeral state message last",
          "[CURRENT STATE" in str(prepared[-1].get("content")))
    check("solo: L2 boundary present", "[COMPACTED:" in texts)
    check("solo: images bounded (window + reattach)", n_img <= 2 + 1)
    check("solo: event stubs carry recall handles", "recall(" in texts)
    check("solo: estimate shrank",
          SoloModel._estimate(prepared) < SoloModel._estimate(msgs))
    check("solo: compact event emitted",
          any(k == "compact" for k, _ in events3))
    # one goto was (correctly) blocked by the no-progress guard at ~60 steps,
    # so executed gotos — not attempts — is the ground truth for image count
    executed = inner3.calls_by_tool["goto"]
    check("solo: no-progress guard blocked exactly one attempt", executed == 9)
    check("solo: stored history untouched (recompute-per-call)",
          sum(1 for m in msgs if isinstance(m.get("content"), list)
              for pt in m["content"]
              if isinstance(pt, dict) and pt.get("type") == "image_url") == executed)

    shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
          + (f" — FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
