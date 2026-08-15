"""Stdio MCP bridge with the eharness organs INSIDE — the S4 executor port.

The universal-shell thesis made literal (implementation-plan v2): the organs
live at the TOOL boundary, so any executor that calls tools gets them. For
the Claude Agent SDK the tool boundary is this bridge process, so the
HarnessedToolset wraps the HabitatToolSet right here:

    SDK (sonnet, subscription auth)  →  MCP stdio  →  THIS BRIDGE
        →  HarnessedToolset (guards · veer · milestones · brake · bell ·
           state block · journey — inject_state=True: the [STATE] header
           rides every tool result, the designed closed-loop channel)
        →  HabitatToolSet  →  env_habitat server

The judge runs on the LOCAL model (ollama qwen) — no API key, and the
executor/judge split gives the perception-diversity the reception-desk
failure showed we need. The SDK keeps its own context management; the
model-layer organs (L2 compaction, hygiene) simply don't apply here.

Env contract (set by driver.bridge_env when extra.eh_bridge=1):
  HABITAT_SERVER_URL / HABITAT_VERB_PREFIX / HABITAT_STEP_BUDGET /
  HABITAT_TURN_BUDGET / HABITAT_BARE / HABITAT_AUTO_OBSERVE /
  HABITAT_LIVE_DIR — same names as mcp_bridge.py.
  EH_INSTRUCTION  — the episode instruction (for the state block + split).
  EH_JUDGE_MODEL  — litellm name for the judge (default local qwen 9b).
  EH_JUDGE_THINK  — "0" turns judge thinking off (default on).
  EH_SPLIT_MODEL  — bare ollama tag for the structured split.
  OLLAMA_URL      — for the split call (default http://127.0.0.1:11434).

Import side effects are limited to path setup — the wrapper is built lazily
on the first tool call, so the driver's tool-schema introspection (which
imports this module with a bare env) never triggers a model call.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

_CODING_AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODING_AGENT))
sys.path.insert(0, str(_CODING_AGENT / "harnesses" / "mini"))

# §14.9: descriptions come from the CapabilityManifest, never hand-written —
# the bridge's private step text taught the BARE surface to "cover ground
# with goto" (a verb its server does not register), and its dwp step/goto
# texts drifted from what mini registers verbatim (review P1)
from eharness.capabilities import (  # noqa: E402
    DWP_STEP_DESC,
    GOTO_DESC,
    STEP_DESC_BARE,
)

SERVER_URL = os.environ.get("HABITAT_SERVER_URL", "http://127.0.0.1:9200")
STEP_BUDGET = int(os.environ.get("HABITAT_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("HABITAT_TURN_BUDGET", "0"))
BARE = os.environ.get("HABITAT_BARE") == "1"
LIVE_DIR = (Path(os.environ["HABITAT_LIVE_DIR"])
            if os.environ.get("HABITAT_LIVE_DIR") else None)
INSTRUCTION = os.environ.get("EH_INSTRUCTION", "")
JUDGE_MODEL = os.environ.get("EH_JUDGE_MODEL",
                             "ollama_chat/qwen3.5:9b-bf16-std")
JUDGE_THINK = os.environ.get("EH_JUDGE_THINK", "1") != "0"
SPLIT_MODEL = os.environ.get("EH_SPLIT_MODEL", "qwen3.5:9b-bf16-std")
# The geometry line, ported to the SDK executor. Without these the bridge builds
# a bare HabitatToolSet, so a frontier model would be measured on an action
# space three versions old — and the whole point of running it is to see whether
# the harness's help changes sign with capability, which needs the SAME organs.
DWP = os.environ.get("EH_DWP") == "1"
PLACES = int(os.environ.get("EH_PLACES", "0"))
SAM_URL = os.environ.get("EH_SAM_URL", "")
LANDMARK_EVERY = int(os.environ.get("EH_LANDMARK_EVERY", "3"))
# the blocking self-assessment organ (wrapper reflect gate): same knob and
# default as the mini path — dropping it silently stripped the organ from
# every SDK arm while the cells claimed organ parity (review P2)
REFLECT_EVERY = int(os.environ.get("EH_REFLECT_EVERY", "0"))
# 0 = event-driven SAM (the mini run default) — parity with the mini arm
SAM_EVERY = int(os.environ.get("EH_SAM_INTERMEDIATE_EVERY", "0"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

mcp = FastMCP("habitat-env")

_t0 = time.time()
_wrapper: Any = None


def _emit(kind: str, payload: dict) -> None:
    if LIVE_DIR is None:
        return
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        with (LIVE_DIR / "events.jsonl").open("a") as fh:
            fh.write(json.dumps({"t": round(time.time() - _t0, 2),
                                 "kind": kind, **payload},
                                ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — logging must never break a run
        pass


def _ensure() -> Any:
    """Build toolset + state + wrapper on first use (never at import)."""
    global _wrapper
    if _wrapper is not None:
        return _wrapper
    from toolset import HabitatToolSet

    from eharness.state_block import StateBlock
    from eharness.wrapper import HarnessedToolset

    if DWP:
        from depth_toolset import DepthWaypointToolSet
        inner = DepthWaypointToolSet(
            SERVER_URL, instruction=INSTRUCTION, bare=BARE,
            step_budget=STEP_BUDGET, turn_budget=TURN_BUDGET,
            live_dir=LIVE_DIR, sam_url=SAM_URL,
            landmark_every=LANDMARK_EVERY, places=PLACES,
            sam_intermediate_every=SAM_EVERY)
    else:
        inner = HabitatToolSet(
            SERVER_URL, bare=BARE, step_budget=STEP_BUDGET,
            turn_budget=TURN_BUDGET, live_dir=LIVE_DIR)
    state = (StateBlock.load(LIVE_DIR / "state.json")
             if LIVE_DIR else StateBlock())
    if INSTRUCTION and not state.instruction:
        state.instruction = INSTRUCTION
    # §16.5: the ONE resolver. The driver normally wrote the record into
    # the live dir before this process even started — load it; only a
    # bridge run without a driver (manual smoke) resolves here, with the
    # same module, same fallback, same validation. Gate on MISSING
    # SEGMENTS alone (review P2: legacy state with segments=[] but a
    # terminate string skipped this and armed near_stop at frame 0).
    rec = None
    if state.instruction:
        from eharness.resolver import load_record, resolve_or_load
        if not state.sub_instructions:
            rec = resolve_or_load(LIVE_DIR, state.instruction,
                                  model_name=f"ollama_chat/{SPLIT_MODEL}",
                                  model_kwargs={"api_base": OLLAMA_URL})
            state.sub_instructions = rec.segments
            state.terminate = rec.terminate
            _emit("subgoal_parse", {"segments": rec.segments,
                                    "terminate": rec.terminate,
                                    "landmarks": rec.landmarks,
                                    "source": rec.source})
        else:
            rec = load_record(LIVE_DIR)
    # the detector needs the route's nouns, same as the mini path — and the
    # SAME nouns: rehydrated UNCONDITIONALLY (review P1: inside the guard, a
    # bridge restart silently dropped SAM queries and terminate wiring)
    if DWP and hasattr(inner, "set_route") and state.sub_instructions:
        inner.set_route(list(state.sub_instructions), state.terminate,
                        landmarks=(rec.landmarks if rec is not None
                                   and rec.instruction == state.instruction
                                   else None))
    state.step_budget = STEP_BUDGET
    if LIVE_DIR:
        state.path = LIVE_DIR / "state.json"
    state.save()
    _wrapper = HarnessedToolset(
        inner, state=state, live_dir=LIVE_DIR,
        judge_model=JUDGE_MODEL,
        judge_kwargs={"drop_params": True, "judge_think": JUDGE_THINK},
        emit=_emit,
        reflect_every=REFLECT_EVERY,
        inject_state=True,      # the closed-loop channel: [STATE] rides results
        auto_view=True, verify_moves=True)
    return _wrapper


def _to_mcp(result: Any) -> list:
    """ToolResult content → FastMCP payload (images + text)."""
    from eharness.wrapper import _png_from_part
    out: list[Any] = []
    texts: list[str] = []
    for part in (result.content or []):
        png = _png_from_part(part)
        if png is not None:
            if texts:
                out.append("\n".join(texts))
                texts = []
            out.append(Image(data=png, format="png"))
        elif isinstance(part, dict) and part.get("type") == "text":
            texts.append(str(part.get("text", "")))
    if texts:
        out.append("\n".join(texts))
    return out or ["(empty result)"]


_STATE_FIELDS_DOC = (
    "Optional state-write fields (your shared memory — the [STATE] header in "
    "every result is rebuilt from them): progress_note (one line: where you "
    "are / how the current segment is going), advance_subgoal (true when the "
    "current route segment is COMPLETE), landmark (a place you can now "
    "identify), dead_end (a checked, ruled-out direction), expect (where "
    "this move should land you — checked against the outcome view), "
    "surroundings (your understanding of what is around you right now — "
    "update whenever the scene changes), near_goal (true the moment you "
    "believe the final goal is in sight — arms arrival verification), "
    "revoke_dead_end (name a dead end you now believe was wrong — it is "
    "struck from the list with the revocation on record)."
)


# §14.3: on the geometry (dwp) surface observe and look_around are HARNESS
# stages, not model tools — they must not exist at the SERVER level either.
# allowed_tools in the SDK options is only an adapter-side filter; a Codex or
# any other MCP client discovering this server would still see them, so the
# canonical registry (eharness.capabilities.mcp_visible_tools) decides what
# gets registered at all.
if not DWP:
  @mcp.tool(description=(
      "Look through the robot's forward-facing camera. Returns the current "
      "egocentric RGB view plus the [STATE] header. Pure read — no step cost."))
  def observe() -> list:
      w = _ensure()
      return _to_mcp(w.execute("observe", {}))


@mcp.tool(description=(
    (DWP_STEP_DESC if DWP else STEP_DESC_BARE) + " " + _STATE_FIELDS_DOC))
def step(actions: list[int],
         progress_note: str | None = None,
         advance_subgoal: bool | None = None,
         landmark: str | None = None,
         dead_end: str | None = None,
         expect: str | None = None,
         surroundings: str | None = None,
         near_goal: bool | None = None,
         revoke_dead_end: str | None = None,
         segment_done: bool | None = None,
         at_goal: bool | None = None,
         answer: str | None = None) -> list:
    w = _ensure()
    args: dict[str, Any] = {"actions": actions}
    for k, v in (("progress_note", progress_note),
                 ("advance_subgoal", advance_subgoal),
                 ("landmark", landmark), ("dead_end", dead_end),
                 ("expect", expect), ("surroundings", surroundings),
                 ("near_goal", near_goal),
                 ("revoke_dead_end", revoke_dead_end),
                 # the reflect gate's [STOP AND ANSWER] demands these —
                 # without them the SDK arm could never answer it (review P2)
                 ("segment_done", segment_done), ("at_goal", at_goal),
                 ("answer", answer)):
        if v is not None:
            args[k] = v
    return _to_mcp(w.execute("step", args))


# Registered ONLY on the geometry line. On the bare surface the inner toolset
# has no goto handler, so exposing the verb there would offer the model an
# action that can only fail.
if DWP:
  @mcp.tool(description=GOTO_DESC + " " + _STATE_FIELDS_DOC)
  def goto(place: int,
           progress_note: str | None = None,
           advance_subgoal: bool | None = None,
           landmark: str | None = None,
           dead_end: str | None = None,
           expect: str | None = None,
           surroundings: str | None = None,
           near_goal: bool | None = None,
           revoke_dead_end: str | None = None,
           segment_done: bool | None = None,
           at_goal: bool | None = None,
           answer: str | None = None) -> list:
      """Only registered when the geometry line is on — see EH_DWP.

      Ported late and it cost a run: the toolset was wired to the bridge but this
      verb was not, so a frontier model was measured on an action space that
      computed the candidates, printed them in the observe payload, and then gave
      it no way to take one. "It never used the menu" means nothing when there is
      no menu."""
      w = _ensure()
      args: dict[str, Any] = {"place": place}
      for k, v in (("progress_note", progress_note),
                   ("advance_subgoal", advance_subgoal),
                   ("landmark", landmark), ("dead_end", dead_end),
                   ("expect", expect), ("surroundings", surroundings),
                   ("near_goal", near_goal),
                   ("revoke_dead_end", revoke_dead_end),
                   ("segment_done", segment_done), ("at_goal", at_goal),
                   ("answer", answer)):
          if v is not None:
              args[k] = v
      return _to_mcp(w.execute("goto", args))

  # face withdrawn from the model surface (user ruling 2026-08-12 evening):
  # goto + step ARE the surface; step's own 2/3 turns cover direction
  # changes. The inner _tool_face survives for internal/human use only.


@mcp.tool(description=(
    "When forward is blocked: YOU pick the side that looks open; the "
    "harness turns 15° that way (30° if still blocked) plus a forward step "
    "FOR you and returns the outcome view."))
def veer(direction: str) -> list:
    w = _ensure()
    return _to_mcp(w.execute("veer", {"direction": direction}))


if not DWP:
  @mcp.tool(description=(
      "Scan your surroundings: four views (ahead / right / behind / left), "
      "each labelled with the exact turn sequence to FACE it. Heading "
      "restored afterwards. Costs about 24 env steps."))
  def look_around() -> list:
      w = _ensure()
      return _to_mcp(w.execute("look_around", {}))


@mcp.tool(description=(
    "Replay stored keyframes by landmark name, frame number ([frame#N] tags "
    "mark each view) or 'segment N' (a completed route segment's filmstrip). "
    "Free — costs no steps."))
def recall(query: str) -> list:
    w = _ensure()
    return _to_mcp(w.execute("recall", {"query": query}))


if __name__ == "__main__":
    # §16.2: the RESIDENT bootstrap — before this process answers a single
    # tool call, the SAME toolset instance the model will drive takes its
    # first look (frame-0 fuse, candidate table, snapshot, artifact). The
    # adapter reads live_dir/bootstrap.json for the first message, so the
    # numbers in the model's first image ARE this instance's numbers —
    # goto(1) works on turn one instead of "no places are current".
    # mcp.run() only starts serving afterwards, so the SDK's connected-wait
    # doubles as the bootstrap barrier.
    if DWP and os.environ.get("EH_BOOTSTRAP", "1") == "1":
        # a STALE artifact is worse than none: on any re-boot, drop the old
        # file first so the adapter can only ever serve numbers a FRESH
        # successful boot produced (review P2 — a failed re-bootstrap left
        # the previous process's candidates looking live)
        if LIVE_DIR is not None:
            try:
                (LIVE_DIR / "bootstrap.json").unlink(missing_ok=True)
            except OSError:
                pass
        try:
            _w = _ensure()
            if hasattr(_w.inner, "bootstrap"):
                _boot = _w.inner.bootstrap()
                _emit("bootstrap", {
                    "candidates": len(_boot["artifact"]["candidates"]),
                    "candidate_epoch": _boot["artifact"]["candidate_epoch"],
                    "sensor_frame": _boot["artifact"]["sensor_frame"]})
                # step-0 landmark ledger, same as the mini arm (review P2:
                # the verifier organs held different evidence per arm) —
                # skip_frontal: the bootstrap already owns the frontal
                if hasattr(_w.inner, "opening_survey"):
                    _survey = _w.inner.opening_survey(skip_frontal=True)
                    if _survey.get("seen"):
                        _emit("opening_survey", {
                            "views": _survey.get("views"),
                            "seen": _survey.get("seen")})
                    if _survey.get("sentence"):
                        _w.state.surroundings = _survey["sentence"]
                        _w.state.save()
        except Exception as exc:  # noqa: BLE001 — a failed boot must not kill
            # the server; the adapter falls back to the text-only first
            # prompt and says so honestly
            _emit("bootstrap", {"error": str(exc)[:200]})
    mcp.run()
