"""§9.2 — the CYCLE-based context contract, proven on synthetic transcripts.

Every check below maps to a row of the multiturn revision plan: whole cycles
in place, original reasoning retained, unselected cycles gone entirely,
12 history + CURRENT + latest map with 14 enforced, refs materialised at
compile time, and the mandatory slots surviving an imageless error turn.
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eharness.context import ContextPolicy, assemble  # noqa: E402
from eharness.state_block import StateBlock  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
    "DwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class FakeFrame:
    def __init__(self, idx):
        self.idx, self.step = idx, idx * 3
        self.png_path = f"frames/frame_{idx:06d}.png"


class FakeFrames:
    def __init__(self, n, live_dir=None):
        self.frames = [FakeFrame(i) for i in range(n)]
        self.live_dir = live_dir

    def last_frame(self):
        return self.frames[-1] if self.frames else None

    def png_bytes(self, idx):
        return PNG if 0 <= idx < len(self.frames) else None

    def stub_text(self, idx):
        return f"[frame#{idx} elided · recall({idx}) to view]"


def img_part():
    b64 = base64.b64encode(PNG).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}


def ref_part(idx):
    return {"type": "image_ref", "ref": f"frames/frame_{idx:06d}.png"}


def cycle(fid, *, n, with_map=True, refs=False, imageless=False,
          extra_other=0):
    """One complete cycle: assistant (reasoning + tool call) + tool result."""
    a = {"role": "assistant",
         "content": f"I will take place 1 (turn {n}).",
         "reasoning_content": f"thinking about turn {n}: the corridor looks "
                              "open to the left, the map agrees.",
         "tool_calls": [{"id": f"c{n}", "type": "function",
                         "function": {"name": "goto",
                                      "arguments": '{"place": 1}'}}]}
    content = []
    if not imageless:
        content += [{"type": "text", "text": "IMAGE 1 — current egocentric RGB"},
                    ref_part(fid) if refs else img_part(),
                    {"type": "text", "text": f"[frame#{fid}]"}]
        if with_map:
            content += [{"type": "text",
                         "text": "IMAGE 2 — accumulated top-down memory (x)"},
                        {"type": "image_ref", "ref": f"maps/map_{n:05d}.png",
                         "map": True} if refs else img_part()]
    for _ in range(extra_other):
        content.append(img_part())
    content.append({"type": "text", "text": json.dumps({"ok": True, "n": n})})
    t = {"role": "tool", "tool_call_id": f"c{n}", "content": content}
    return [a, t]


def transcript(n_cycles, frames, **kw):
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "task"}]
    per = max(1, (len(frames.frames) - 1) // max(1, n_cycles))
    for t in range(n_cycles):
        fid = min(len(frames.frames) - 1, (t + 1) * per)
        msgs += cycle(fid, n=t, **kw)
    return msgs


def images_in(msgs):
    out = []
    for mi, m in enumerate(msgs):
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image_url",
                                                             "image_ref"):
                    out.append((mi, m.get("role"), p))
    return out


def texts_of(msgs):
    return " ".join(str(p.get("text", "")) for m in msgs
                    if isinstance(m.get("content"), list)
                    for p in m["content"] if isinstance(p, dict))


print("── 100-frame episode → ≤12 history cycles, early/mid/late, ordered ─")
fr = FakeFrames(101)
msgs = transcript(30, fr)
out, man = assemble(msgs, fr, ContextPolicy())
ids = man["history_cycle_ids"]
check("at most 12 history cycles retained", len(ids) == 12, str(ids))
check("first and last visual history cycles included",
      ids[0] == 0 and ids[-1] == 28, f"{ids[0]}..{ids[-1]}")
check("spread covers early, middle and late",
      any(i < 5 for i in ids) and any(12 <= i <= 18 for i in ids)
      and any(i > 24 for i in ids))
out2, man2 = assemble(msgs, fr, ContextPolicy())
check("deterministic — same input, same selection",
      man2["history_cycle_ids"] == ids)
check("history frames ride along in manifest",
      len(man["history_frame_ids"]) == 12, str(man["history_frame_ids"]))

print()
print("── cycles stay WHOLE and IN PLACE ──────────────────────────────────")
roles = [m.get("role") for m in out]
orphan = any(m.get("role") == "tool" and (i == 0 or out[i - 1].get("role")
             not in ("assistant", "tool"))
             for i, m in enumerate(out))
check("no orphan tool result", not orphan, " ".join(roles[:12]) + " …")
imgs = images_in(out)
user_imgs = [x for x in imgs if x[1] == "user"]
check("history images live in their OWN tool results, none in a user block",
      all(r == "tool" for _mi, r, _p in imgs) and not user_imgs,
      f"{len(imgs)} images, roles {sorted(set(r for _m, r, _p in imgs))}")
kept_assistants = [m for m in out if m.get("role") == "assistant"]
check("retained assistants keep their ORIGINAL reasoning",
      all(isinstance(m.get("reasoning_content"), str)
          and m["reasoning_content"].startswith("thinking about")
          for m in kept_assistants),
      f"{len(kept_assistants)} assistants, "
      f"{man['reasoning_chars_retained']} reasoning chars")
check("tool calls stay paired with their ids",
      all(m.get("tool_calls") for m in kept_assistants))
check("dropped cycles vanish whole (assistant count = retained cycles)",
      len(kept_assistants) == man["retained_cycles"],
      f"{len(kept_assistants)} vs {man['retained_cycles']}")
check("…and the elision is spoken, not silent",
      "not shown" in texts_of(out) and man["dropped_turns"] == 17,
      f"dropped {man['dropped_turns']}")

print()
print("── the visual slots: 12 + CURRENT + latest map, 14 enforced ────────")
check("hard cap holds", man["image_count"] <= 14,
      f"{man['image_count']} images")
check("exactly one CURRENT frame (the newest cycle's own view)",
      man["current_frame_id"] == 90, str(man["current_frame_id"]))
texts = texts_of(out)
n_map_labels = sum(1 for m in out if isinstance(m.get("content"), list)
                   for i, p in enumerate(m["content"])
                   if isinstance(p, dict) and p.get("type") == "image_url"
                   and i > 0 and str(m["content"][i - 1].get("text", ""))
                   .startswith("IMAGE 2 — accumulated"))
check("exactly one live accumulated map", n_map_labels == 1,
      f"{n_map_labels} live maps, {texts.count('[MAP UPDATED')} stubs")

# a recall flood in the newest turn must not break the cap
msgs2 = transcript(30, fr)
msgs2[-1]["content"] = list(msgs2[-1]["content"]) + [img_part()] * 6
out3, man3 = assemble(msgs2, fr, ContextPolicy())
check("recall/other images pay out of the history quota, cap still holds",
      man3["image_count"] <= 14 and len(man3["history_cycle_ids"]) <= 6,
      f"{man3['image_count']} images, "
      f"{len(man3['history_cycle_ids'])} history cycles")

print()
print("── refs: stored transcript is light, payload is real ───────────────")
tdir = Path(tempfile.mkdtemp(prefix="ctx_"))
(tdir / "frames").mkdir()
(tdir / "maps").mkdir()
fr2 = FakeFrames(13, live_dir=tdir)
for i in range(13):
    (tdir / f"frames/frame_{i:06d}.png").write_bytes(PNG)
for n in range(12):
    (tdir / f"maps/map_{n:05d}.png").write_bytes(PNG)
msgs = transcript(12, fr2, refs=True)
stored_b64 = sum(1 for _mi, _r, p in images_in(msgs)
                 if p.get("type") == "image_url")
check("stored transcript holds refs, not base64", stored_b64 == 0,
      f"{stored_b64} base64 images in storage")
out4, man4 = assemble(msgs, fr2, ContextPolicy(), live_dir=tdir)
mat = images_in(out4)
check("compile materialises every surviving image",
      all(p.get("type") == "image_url" for _m, _r, p in mat) and len(mat) > 4,
      f"{len(mat)} data-URI images, 0 refs left"
      if all(p.get("type") == "image_url" for _m, _r, p in mat)
      else str([p.get('type') for _m, _r, p in mat]))
missing_dir = Path(tempfile.mkdtemp(prefix="ctx_missing_"))
out_m, _ = assemble(transcript(3, FakeFrames(4), refs=True), FakeFrames(4),
                    ContextPolicy(), live_dir=missing_dir)
check("a missing file becomes an honest stub, not a hole",
      "unavailable" in texts_of(out_m)
      and not any(p.get("type") == "image_ref"
                  for _m, _r, p in images_in(out_m)))

print()
print("── mandatory slots survive an imageless error turn ─────────────────")
fr3 = FakeFrames(20, live_dir=tdir)
msgs = transcript(8, fr3)
msgs += cycle(0, n=99, imageless=True)      # error turn: no image, no map
out5, man5 = assemble(msgs, fr3, ContextPolicy())
texts5 = texts_of(out5)
check("CURRENT is re-attached from the stable file",
      man5["current_frame_id"] == 19 and "carried no image" in texts5,
      str(man5["current_frame_id"]))
check("the latest known map is re-attached to the newest turn",
      "latest known" in texts5)

print()
print("── StateBlock stays pure; legacy L2 stays gone ─────────────────────")
sb = StateBlock(instruction="go", terminate="stop at the bar corner")
sb.pending_question = "same couch as v2?"
_ = sb.render()
check("render is pure — the question survives", sb.pending_question != "")
check("no [COMPACTED: marker in the default payload",
      "[COMPACTED:" not in texts)
check("manifest proves the shape", all(
    k in man for k in ("retained_cycles", "history_cycle_ids",
                       "history_frame_ids", "current_frame_id", "map_id",
                       "image_count", "reasoning_chars_retained",
                       "dropped_turns", "context_build_ms")),
    json.dumps({k: man[k] for k in ("retained_cycles", "image_count",
                                    "dropped_turns")}))

print()
print("── §14.15 · provider shape validators (synthetic-level) ─────────────")
# Real provider replay (Claude thinking blocks, OpenAI strict mode, Qwen
# templates) is out of scope here — these are pure-local structural checks
# of the shapes every provider rejects outright.


def claude_shape_ok(msgs):
    """tool_use/tool_result pairing: an assistant's tool_calls must be
    answered by the immediately following contiguous tool messages, exactly
    once each; a tool message may not appear anywhere else. reasoning
    (thinking) may only ride assistant messages."""
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "assistant":
            want = sorted(str(tc.get("id")) for tc in (m.get("tool_calls") or []))
            j = i + 1
            got = []
            while j < len(msgs) and msgs[j].get("role") == "tool":
                got.append(str(msgs[j].get("tool_call_id")))
                j += 1
            if want != sorted(got):
                return False
            i = j
        else:
            if m.get("reasoning_content") or m.get("role") == "tool":
                return False
            i += 1
    return True


def openai_shape_ok(msgs):
    """role=tool must immediately follow the assistant (or sibling tool of
    the same batch) whose tool_calls contain its tool_call_id, once each."""
    pending: set = set()
    for m in msgs:
        if m.get("role") == "assistant":
            if pending:
                return False          # previous batch left unanswered
            pending = {str(tc.get("id")) for tc in (m.get("tool_calls") or [])}
        elif m.get("role") == "tool":
            tid = str(m.get("tool_call_id"))
            if tid not in pending:
                return False
            pending.discard(tid)
        elif pending:
            return False              # non-tool message inside a batch
    return not pending


def generic_shape_ok(msgs):
    """no two consecutive assistant messages; no empty content without
    tool_calls."""
    prev = None
    for m in msgs:
        if m.get("role") == "assistant" and prev == "assistant":
            return False
        c = m.get("content")
        empty = (c is None or c == "" or c == [])
        if empty and not m.get("tool_calls"):
            return False
        prev = m.get("role")
    return True


def shapes(name, msgs):
    check(f"{name}: Claude shape (pairing, thinking placement)",
          claude_shape_ok(msgs))
    check(f"{name}: OpenAI shape (tool follows its call)",
          openai_shape_ok(msgs))
    check(f"{name}: generic shape (no dangling/empty turns)",
          generic_shape_ok(msgs))


shapes("30-cycle payload", out)
shapes("imageless-error payload", out5)

print()
print("── §14.15 · one assistant, MULTIPLE tool calls ──────────────────────")


def multi_cycle(fid, n):
    a = {"role": "assistant", "content": f"two probes (turn {n}).",
         "reasoning_content": f"thinking about turn {n}: split attention.",
         "tool_calls": [
             {"id": f"m{n}a", "type": "function",
              "function": {"name": "goto", "arguments": '{"place": 1}'}},
             {"id": f"m{n}b", "type": "function",
              "function": {"name": "recall", "arguments": '{"query": "bar"}'}}]}
    t1 = {"role": "tool", "tool_call_id": f"m{n}a", "content": [
        {"type": "text", "text": "IMAGE 1 — current egocentric RGB"},
        img_part(), {"type": "text", "text": f"[frame#{fid}]"},
        {"type": "text", "text": json.dumps({"ok": True, "n": n})}]}
    t2 = {"role": "tool", "tool_call_id": f"m{n}b",
          "content": [{"type": "text", "text": "nothing recalled"}]}
    return [a, t1, t2]


frM = FakeFrames(101)
msgsM = [{"role": "system", "content": "sys"},
         {"role": "user", "content": "task"}]
for t in range(30):
    fid = min(100, (t + 1) * 3)
    msgsM += multi_cycle(fid, t) if t % 3 == 0 else cycle(fid, n=t)
outM, manM = assemble(msgsM, frM, ContextPolicy())
kept_multi = [m for m in outM if m.get("role") == "assistant"
              and len(m.get("tool_calls") or []) == 2]
check("retained multi-call cycles keep BOTH tool calls", len(kept_multi) >= 2,
      f"{len(kept_multi)} multi-call assistants retained")
tool_ids = [str(m.get("tool_call_id")) for m in outM if m.get("role") == "tool"]
check("each retained call answered exactly once",
      all(tool_ids.count(str(tc["id"])) == 1 for m in kept_multi
          for tc in m["tool_calls"]))
dropped_multi = [t for t in range(30) if t % 3 == 0
                 and t not in manM["history_cycle_ids"] and t != 29]
check("a dropped multi-call cycle vanishes whole (no orphan of either id)",
      bool(dropped_multi) and all(
          f"m{k}a" not in tool_ids and f"m{k}b" not in tool_ids
          for k in dropped_multi),
      f"dropped {dropped_multi[:4]}…")
shapes("multi-call payload", outM)

print()
print("── §14.15 · FormatError / retry / correction at cycle boundaries ────")
dangle = {"role": "assistant", "content": "I'll goto 2.",
          "reasoning_content": "thinking: retry after the format error.",
          "tool_calls": [{"id": "zz9", "type": "function",
                          "function": {"name": "goto", "arguments": "{"}}]}
corr = {"role": "user",
        "content": "FormatError: unterminated arguments — send a valid call"}
frF = FakeFrames(20)
base = [{"role": "system", "content": "sys"},
        {"role": "user", "content": "task"}]
msgsF = list(base)
for t in range(3):
    msgsF += cycle(5 + 3 * t, n=t)
msgsF += [{"role": "user", "content": "FormatError: bad tool call — retry"}]
# ^ rides cycle 2 (the splitter appends non-assistant turns to the open cycle)
msgsF += [dangle, corr]                     # newest cycle: dangling call
outF, manF = assemble(msgsF, frF, ContextPolicy())
kept_ids = [str(tc.get("id")) for m in outF if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])]
check("dangling tool_call is dropped from the compiled payload",
      "zz9" not in kept_ids, str(kept_ids))
check("the correction text itself survives",
      "FormatError: unterminated" in " ".join(
          str(m.get("content")) for m in outF))
shapes("format-error payload", outF)

zombie_a = {"role": "assistant", "content": "(free text, parse failed)",
            "reasoning_content": "thinking: mumble."}
zombie_t = {"role": "tool", "tool_call_id": "ghost1",
            "content": [{"type": "text", "text": "stale result"}]}
msgsZ = list(base)
for t in range(3):
    msgsZ += cycle(5 + 3 * t, n=t)
msgsZ += [zombie_a, zombie_t]               # newest cycle: orphan tool result
outZ, _manZ = assemble(msgsZ, frF, ContextPolicy())
orphanZ = [m for m in outZ if m.get("role") == "tool"
           and str(m.get("tool_call_id")) == "ghost1"]
check("orphan tool result never reaches a provider as role=tool",
      not orphanZ)
check("…but its content is carried forward",
      "stale result" in texts_of(outZ))
# review P2: with the newest cycle's result re-rolled to user, the
# mandatory-slot fallbacks must still attach INSIDE the newest cycle — the
# old role=='tool' scan walked past it and glued the CURRENT view onto a
# history turn, mid-transcript, as if it were an old action's outcome.
ghost_at = next(i for i, m in enumerate(outZ)
                if "stale result" in str(m.get("content")))
cur_at = [i for i, m in enumerate(outZ)
          if "[your CURRENT view" in str(m.get("content"))]
check("CURRENT-view fallback attaches to the newest (re-rolled) turn",
      cur_at == [ghost_at], f"ghost at {ghost_at}, fallback at {cur_at}")
map_at = [i for i, m in enumerate(outZ)
          if "latest known — this turn produced none" in str(m.get("content"))]
check("latest-map fallback attaches there too, never to history",
      not map_at or map_at == [ghost_at],
      f"map fallback at {map_at}")
check("internal _reroled_tool tag never reaches the provider payload",
      all("_reroled_tool" not in m for m in outZ))
shapes("orphan-result payload", outZ)

print()
print("── §14.15 · recall+survey+imageless+STOP together, cap holds ────────")
frS = FakeFrames(20, live_dir=tdir)
msgsS = [{"role": "system", "content": "sys"},
         {"role": "user", "content": [{"type": "text", "text": "task"},
                                      img_part(), img_part()]}]
for t in range(18):
    msgsS += cycle(min(19, t + 1), n=t)
msgsS += cycle(0, n=98, imageless=True)               # error turn
msgsS += cycle(0, n=99, imageless=True, extra_other=3)  # STOP turn + recalls
outS, manS = assemble(msgsS, frS, ContextPolicy())
tS = texts_of(outS)
check("hard cap holds with every slot type in play",
      manS["image_count"] <= 14, f"{manS['image_count']} images")
check("opening survey elided from later turns",
      "opening survey view shown on the first decision" in tS
      and not any(r == "user" and mi < 2 for mi, r, _p in images_in(outS)))
check("CURRENT fallback fires exactly once",
      tS.count("carried no image") == 1)
check("latest-map fallback fires exactly once, no double map",
      tS.count("latest known") == 1)
shapes("combined-slots payload", outS)

print()
print("── §14.15 · flood → FINAL deterministic degradation ────────────────")
frD = FakeFrames(101)
msgsD = transcript(20, frD)
msgsD[-1]["content"] = list(msgsD[-1]["content"]) + [img_part()] * 30
outD, manD = assemble(msgsD, frD, ContextPolicy())
check("cap enforced even with zero history slots left",
      manD["image_count"] <= 14, f"{manD['image_count']} images")
check("final degradation is recorded in the manifest",
      manD["images_degraded_final"] > 0, str(manD["images_degraded_final"]))
outD2, manD2 = assemble(msgsD, frD, ContextPolicy())
check("final degradation is deterministic (same input → same payload)",
      texts_of(outD) == texts_of(outD2)
      and manD["image_count"] == manD2["image_count"]
      and manD["images_degraded_final"] == manD2["images_degraded_final"])
check("the CURRENT frame and live map survive the flood",
      manD["current_frame_id"] is not None
      and texts_of(outD).count("[image elided — over the image budget]")
      == manD["images_degraded_final"])
shapes("flood payload", outD)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("context contract (cycles): all passed")
