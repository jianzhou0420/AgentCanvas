"""Tier-1 byte-equivalence gate: threestep2 (decomp) vs threestep (mono).

Feeds identical hand-crafted inputs + initial graph_state into the mono
``judge_decide`` and the decomp's ``judge_verdict → resolve_action →
update_nav_state`` pipeline (run in graph-edge order), then byte-compares every
output port AND every state write. The judge LLM is monkey-patched to a canned
raw verdict so the comparison is deterministic, model-free, and fast.

Run:  PYTHONPATH=agentcanvas/backend:. <agentcanvas-python> -m pytest \
          workspace/nodesets/method/threestep2/test_equivalence.py -q
or:   PYTHONPATH=agentcanvas/backend:. <agentcanvas-python> \
          workspace/nodesets/method/threestep2/test_equivalence.py
"""

from __future__ import annotations

import asyncio
import json

from workspace.nodesets.method.threestep import JudgeDecideNode
from workspace.nodesets.method.threestep2 import (
    JudgeVerdictNode,
    ResolveActionNode,
    UpdateNavStateNode,
)

ABILITIES = "continue,stay,backtrack,look_around"
_CANNED_RAW = ""  # set per-scenario; both patched _judge_llm methods return it


async def _fake_judge_llm(self, profile, user_prompt, images_b64, max_tokens):
    return _CANNED_RAW


# Patch both nodes' LLM call (the only non-determinism).
JudgeDecideNode._judge_llm = _fake_judge_llm
JudgeVerdictNode._judge_llm = _fake_judge_llm


class FakeGS:
    def __init__(self, d):
        self.d = dict(d)

    def read(self, k):
        return self.d.get(k)

    def write(self, k, v):
        self.d[k] = v


class FakeCtx:
    def __init__(self, gs, step):
        self.graph_state = gs
        self.step = step


_STATE_KEYS = (
    "nav_history",
    "current_action_idx",
    "chosen_images_b64",
    "chosen_descriptions",
    "move_stack",
    "last_chosen_vp",
    "stuck_directions",
)
_OUT_KEYS = ("angle", "distance", "stop", "decision", "decision_thought")


def _cfg(node):
    node.config = {"llm_profile": "x", "enabled_abilities": ABILITIES, "max_tokens": 2000}
    return node


def _run_mono(scn):
    gs = FakeGS(scn["state"])
    ctx = FakeCtx(gs, scn.get("step", 0))
    node = _cfg(JudgeDecideNode())
    out = asyncio.run(node.forward(dict(scn["inputs"]), ctx))
    return {k: out.get(k) for k in _OUT_KEYS}, {k: gs.d.get(k) for k in _STATE_KEYS}


def _run_decomp(scn):
    gs = FakeGS(scn["state"])
    ctx = FakeCtx(gs, scn.get("step", 0))
    inp = scn["inputs"]
    jv = _cfg(JudgeVerdictNode())
    ra = _cfg(ResolveActionNode())
    un = _cfg(UpdateNavStateNode())

    v = asyncio.run(
        jv.forward(
            {
                "completion_estimation": inp.get("completion_estimation"),
                "pred_vp": inp.get("pred_vp"),
                "pred_thought": inp.get("pred_thought"),
                "current_action": inp.get("current_action"),
                "current_landmarks_join": inp.get("current_landmarks_join"),
                "history_traj": inp.get("history_traj"),
                "action_idx": inp.get("action_idx"),
                "num_actions": inp.get("num_actions"),
                "views": inp.get("views"),
                "candidates": inp.get("candidates"),
            },
            ctx,
        )
    )
    a = asyncio.run(
        ra.forward(
            {
                "decision": v["decision"],
                "completion_estimation": inp.get("completion_estimation"),
                "pred_vp": inp.get("pred_vp"),
                "candidates": inp.get("candidates"),
                "action_idx": inp.get("action_idx"),
                "num_actions": inp.get("num_actions"),
            },
            ctx,
        )
    )
    asyncio.run(
        un.forward(
            {
                "decision": v["decision"],
                "completion_estimation": inp.get("completion_estimation"),
                "action_mode": a["action_mode"],
                "stop": a["stop"],
                "move_angle": a["move_angle"],
                "move_distance": a["move_distance"],
                "pred_vp": inp.get("pred_vp"),
                "views": inp.get("views"),
                "entry": inp.get("entry"),
                "action_idx": inp.get("action_idx"),
                "num_actions": inp.get("num_actions"),
                "candidates": inp.get("candidates"),
            },
            ctx,
        )
    )
    out = {
        "angle": a["angle"],
        "distance": a["distance"],
        "stop": a["stop"],
        "decision": v["decision"],
        "decision_thought": v["decision_thought"],
    }
    return out, {k: gs.d.get(k) for k in _STATE_KEYS}


def _views(*dir_ids):
    return [{"dir_id": d, "rgb_base64": f"img{d}"} for d in dir_ids]


def _entry(step, vp, obs, th):
    return json.dumps({"step": step, "viewpoint": vp, "observation": obs, "thought": th})


# ── scenarios ──────────────────────────────────────────────────────────────
# Each: name · canned judge raw · initial state · judge_decide inputs.

CANDS = {"6": [1.5, 1.0], "0": [0.05, 1.0], "3": [1.0, 1.25], "7": [3.3, 0.75]}


def _scn(name, raw, state, inputs, step=0):
    base_in = {
        "completion_estimation": "No",
        "pred_vp": "6",
        "pred_thought": "navigator thought",
        "current_action": "walk to the kitchen",
        "current_landmarks_join": "the kitchen, the sink",
        "history_traj": "Step 0 start position. ",
        "action_idx": 1,
        "num_actions": 4,
        "candidates": CANDS,
        "views": _views(0, 6, 3, 7),
        "entry": _entry(0, "6", "a hallway", "go forward"),
    }
    base_in.update(inputs)
    return {"name": name, "raw": raw, "state": dict(state), "inputs": base_in, "step": step}


PRIOR_HIST = [{"step": 0, "viewpoint": "3", "observation": "o0", "thought": "t0"}]

SCENARIOS = [
    # gated OFF (no judge call) — completion No / Unknown
    _scn(
        "No@mid",
        "",
        {"current_action_idx": 1, "nav_history": PRIOR_HIST},
        {"completion_estimation": "No", "action_idx": 1},
    ),
    _scn(
        "Unknown@0_firststep",
        "",
        {},
        {"completion_estimation": "Unknown", "action_idx": 0, "num_actions": 3},
    ),
    _scn(
        "No@last_no_stop",
        "",
        {"current_action_idx": 3},
        {"completion_estimation": "No", "action_idx": 3, "num_actions": 4},
    ),
    # gated ON — continue, idx<last  → advance + reset history
    _scn(
        "Yes_continue@mid",
        "Decision: Continue\nConfidence: 8",
        {"current_action_idx": 1, "nav_history": PRIOR_HIST},
        {"completion_estimation": "Yes", "action_idx": 1, "num_actions": 4},
    ),
    # continue@last → rules convert to stay → should_stop fires
    _scn(
        "Yes_continue@last",
        "Decision: Continue\nConfidence: 8",
        {"current_action_idx": 3, "nav_history": PRIOR_HIST},
        {"completion_estimation": "Yes", "action_idx": 3, "num_actions": 4},
    ),
    # stay, idx<last → keep + append
    _scn(
        "Yes_stay@mid",
        "Decision: Stay\nConfidence: 8",
        {"current_action_idx": 1, "nav_history": PRIOR_HIST},
        {"completion_estimation": "Yes", "action_idx": 1, "num_actions": 4},
    ),
    # stay@last → should_stop
    _scn(
        "Yes_stay@last",
        "Decision: Stay\nConfidence: 8",
        {"current_action_idx": 2},
        {"completion_estimation": "Yes", "action_idx": 2, "num_actions": 3},
    ),
    # backtrack, nonempty move_stack → reverse + pop
    _scn(
        "Yes_backtrack_nonempty",
        "Decision: Backtrack\nConfidence: 8",
        {
            "current_action_idx": 2,
            "nav_history": PRIOR_HIST,
            "move_stack": [[0.5, 1.0], [1.2, 1.0]],
            "chosen_images_b64": ["s", "a", "b"],
        },
        {"completion_estimation": "Yes", "action_idx": 2, "num_actions": 4},
    ),
    # backtrack, empty move_stack → reverse [0,0]
    _scn(
        "Yes_backtrack_empty",
        "Decision: Backtrack\nConfidence: 8",
        {"current_action_idx": 1, "move_stack": []},
        {"completion_estimation": "Yes", "action_idx": 1, "num_actions": 3},
    ),
    # look_around → normal move + keep
    _scn(
        "Yes_look_around",
        "Decision: Look Around\nConfidence: 8",
        {"current_action_idx": 1, "nav_history": PRIOR_HIST},
        {"completion_estimation": "Yes", "action_idx": 1, "num_actions": 4},
    ),
    # low confidence (<5) → look_around override
    _scn(
        "Yes_lowconf",
        "Decision: Continue\nConfidence: 3",
        {"current_action_idx": 1, "nav_history": PRIOR_HIST},
        {"completion_estimation": "Yes", "action_idx": 1, "num_actions": 4},
    ),
    # pred_vp not in candidates → forward (0,0), gated off
    _scn(
        "No_badvp",
        "",
        {"current_action_idx": 1},
        {"completion_estimation": "No", "pred_vp": "9", "action_idx": 1},
    ),
    # empty entry → no append
    _scn(
        "No_noentry",
        "",
        {"current_action_idx": 1, "nav_history": PRIOR_HIST},
        {"completion_estimation": "No", "action_idx": 1, "entry": "{}"},
    ),
]


def _check(scn):
    global _CANNED_RAW
    _CANNED_RAW = scn["raw"]
    mono_out, mono_state = _run_mono(scn)
    deco_out, deco_state = _run_decomp(scn)
    assert mono_out == deco_out, (
        f"[{scn['name']}] OUTPUT mismatch:\n  mono={mono_out}\n  deco={deco_out}"
    )
    assert mono_state == deco_state, (
        f"[{scn['name']}] STATE mismatch:\n  mono={mono_state}\n  deco={deco_state}"
    )


def test_byte_equivalence():
    for scn in SCENARIOS:
        _check(scn)


if __name__ == "__main__":
    ok = 0
    for scn in SCENARIOS:
        try:
            _check(scn)
            print(f"  PASS  {scn['name']}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {scn['name']}\n{e}")
    print(f"\n{ok}/{len(SCENARIOS)} scenarios byte-equivalent")
    raise SystemExit(0 if ok == len(SCENARIOS) else 1)
