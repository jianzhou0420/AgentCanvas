"""I/O equivalence tests: smartway decomposed pipeline == smartway_mono monolith.

For each scenario we feed the **same** inputs and graph_state snapshot into
both pipelines, then byte-compare every output port and every state write.
This is a faster, deterministic, GPU/API-free alternative to end-to-end
eval-parity (which is necessary for paper-fidelity but slow + stochastic).

Run from the backend (so `app.*` is importable):

    cd agentcanvas/backend && \
      PYTHONPATH=../..:.  python -m pytest \
        ../../workspace/nodesets/method/smartway/test_equivalence.py -v

The UUID-determinism fixture (`deterministic_uuids`) is autouse: both
``smartway_mono`` and ``smartway`` modules' ``uuid.uuid4`` are monkey-patched
with a fresh counter at the start of each test so the freshly-minted
node-IDs match byte-for-byte across mono and decomp.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

# Ensure workspace + backend are importable.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agentcanvas" / "backend"))


# ───────────────────────────── fakes ─────────────────────────────────


class FakeGraphState:
    """Dict-backed graph_state stand-in. Mirrors the read/write surface."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._d: dict[str, Any] = dict(initial or {})

    def read(self, key: str) -> Any:
        return self._d.get(key)

    def write(self, key: str, value: Any) -> None:
        self._d[key] = value

    def snapshot(self) -> dict[str, Any]:
        # Deep-copy via json round-trip won't survive numpy etc., but state
        # values here are all JSON-able scalars / lists / dicts / strings.
        return json.loads(json.dumps(self._d))


class FakeCtx:
    def __init__(self, step: int, gs: FakeGraphState) -> None:
        self.step = step
        self.graph_state = gs


# ───────────────────────────── fixtures ──────────────────────────────


# Resettable counter shared by mono and decomp runs in the same test.
# Helpers call `_reset_uuid_counter()` at the top of each runner so
# both pipelines see UUIDs 1..N for the same logical N waypoints.
_uuid_counter: list[int] = [0]


def _det_uuid4() -> uuid.UUID:
    _uuid_counter[0] += 1
    return uuid.UUID(int=_uuid_counter[0])


def _reset_uuid_counter() -> None:
    _uuid_counter[0] = 0


@pytest.fixture(autouse=True)
def deterministic_uuids(monkeypatch):
    """Replace uuid.uuid4 in BOTH nodeset modules with a deterministic
    counter so freshly-minted node-IDs match across mono and decomp.

    Patches the *module-bound* `uuid` symbol each nodeset imported, not
    the global `uuid` module — targets only production code under test.
    Test runners must call `_reset_uuid_counter()` at the top of each
    pipeline run so mono+decomp see the same UUID sequence.
    """
    from workspace.nodesets.method import smartway as s_dec
    from workspace.nodesets.method import smartway_mono as s_mono

    monkeypatch.setattr(s_mono.uuid, "uuid4", _det_uuid4)
    monkeypatch.setattr(s_dec.uuid, "uuid4", _det_uuid4)
    _reset_uuid_counter()
    yield
    _reset_uuid_counter()


def _tiny_b64_png() -> str:
    """8x8 red square — small enough for fast tests, real enough for PIL."""
    import numpy as np
    from PIL import Image

    arr = np.full((8, 8, 3), [255, 0, 0], dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ───────────────────────────── runners ───────────────────────────────


def _run_mono_pre_llm(inputs: dict, t: int, gs_initial: dict) -> tuple[dict, dict]:
    from workspace.nodesets.method.smartway_mono import SmartwayMonoPlanStepNode

    _reset_uuid_counter()
    gs = FakeGraphState(gs_initial)
    ctx = FakeCtx(t, gs)
    out = asyncio.run(SmartwayMonoPlanStepNode().forward(inputs, ctx))
    return out, gs.snapshot()


def _run_decomp_pre_llm(inputs: dict, t: int, gs_initial: dict) -> tuple[dict, dict]:
    from workspace.nodesets.method.smartway import (
        SmartwayAssemblePromptNode,
        SmartwayBuildActionOptionsNode,
        SmartwayBuildImagesNode,
        SmartwayUpdateTopologyNode,
    )

    _reset_uuid_counter()
    gs = FakeGraphState(gs_initial)
    ctx = FakeCtx(t, gs)

    ut_out = asyncio.run(
        SmartwayUpdateTopologyNode().forward({"candidates": inputs["candidates"]}, ctx)
    )

    bao_out = asyncio.run(
        SmartwayBuildActionOptionsNode().forward(
            {
                "candidates_enriched": ut_out["candidates_enriched"],
                "candidate_node_indices": ut_out["candidate_node_indices"],
                "last_backtrack": ut_out["last_backtrack"],
                "tags": inputs.get("tags") or {},
            },
            ctx,
        )
    )

    ap_out = asyncio.run(
        SmartwayAssemblePromptNode().forward(
            {
                "instruction": inputs.get("instruction", ""),
                "history_snap": ut_out["history_snap"],
                "planning_snap": ut_out["planning_snap"],
                "full_options": bao_out["full_options"],
            },
            ctx,
        )
    )

    bi_out = asyncio.run(
        SmartwayBuildImagesNode().forward(
            {
                "candidates_enriched": ut_out["candidates_enriched"],
                "candidate_node_indices": ut_out["candidate_node_indices"],
            },
            ctx,
        )
    )

    combined = {
        "task_description": ap_out["task_description"],
        "prompt": ap_out["prompt"],
        "images": bi_out["images"],
        "image_labels": bi_out["image_labels"],
        "only_options": bao_out["only_options"],
        "only_actions": bao_out["only_actions"],
        "candidates_dict": bao_out["candidates_dict"],
    }
    return combined, gs.snapshot()


def _run_mono_post_llm(inputs: dict, t: int, gs_initial: dict) -> tuple[dict, dict]:
    from workspace.nodesets.method.smartway_mono import SmartwayMonoDecideActionNode

    gs = FakeGraphState(gs_initial)
    ctx = FakeCtx(t, gs)
    out = asyncio.run(SmartwayMonoDecideActionNode().forward(inputs, ctx))
    return out, gs.snapshot()


def _run_decomp_post_llm(inputs: dict, t: int, gs_initial: dict) -> tuple[dict, dict]:
    from workspace.nodesets.method.smartway import (
        SmartwayParseResponseNode,
        SmartwayResolveActionNode,
    )

    gs = FakeGraphState(gs_initial)
    ctx = FakeCtx(t, gs)

    pr_out = asyncio.run(
        SmartwayParseResponseNode().forward(
            {
                "response": inputs.get("response", ""),
                "only_options": inputs.get("only_options", "[]"),
            },
            ctx,
        )
    )

    ra_out = asyncio.run(
        SmartwayResolveActionNode().forward(
            {
                "picked_index": pr_out["picked_index"],
                "is_stop": pr_out["is_stop"],
                "candidates_dict": inputs.get("candidates_dict", "{}"),
            },
            ctx,
        )
    )

    combined = {
        "picked_index": pr_out["picked_index"],
        "is_stop": pr_out["is_stop"],
        "is_return": ra_out["is_return"],
        "angle": ra_out["angle"],
        "distance": ra_out["distance"],
        "thought": pr_out["thought"],
        "new_planning": pr_out["new_planning"],
    }
    return combined, gs.snapshot()


# ───────────────────────────── assertions ────────────────────────────


def _assert_images_equal(a: list, b: list) -> None:
    import numpy as np

    assert len(a) == len(b), f"image-list lengths differ: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), f"image[{i}] not ndarray"
        assert x.shape == y.shape, f"image[{i}] shape: {x.shape} vs {y.shape}"
        assert np.array_equal(x, y), f"image[{i}] pixels differ"


def _assert_pre_llm_equiv(
    mono_out: dict, mono_state: dict, decomp_out: dict, decomp_state: dict
) -> None:
    # Outputs
    assert mono_out["task_description"] == decomp_out["task_description"]
    assert mono_out["prompt"] == decomp_out["prompt"], (
        f"\nMONO PROMPT:\n{mono_out['prompt']}\n\nDECOMP PROMPT:\n{decomp_out['prompt']}"
    )
    _assert_images_equal(mono_out["images"], decomp_out["images"])
    assert mono_out["image_labels"] == decomp_out["image_labels"]
    assert mono_out["only_options"] == decomp_out["only_options"]
    assert mono_out["only_actions"] == decomp_out["only_actions"]
    assert mono_out["candidates_dict"] == decomp_out["candidates_dict"]
    # State (only compare keys this segment owns — backtrack-clear differs by design)
    for k in ("nodes_list", "graph", "trajectory"):
        assert mono_state.get(k) == decomp_state.get(k), f"state[{k}] mismatch"


def _assert_post_llm_equiv(
    mono_out: dict, mono_state: dict, decomp_out: dict, decomp_state: dict
) -> None:
    for k in (
        "picked_index",
        "is_stop",
        "is_return",
        "angle",
        "distance",
        "thought",
        "new_planning",
    ):
        assert mono_out[k] == decomp_out[k], f"output[{k}]: {mono_out[k]!r} vs {decomp_out[k]!r}"
    for k in ("planning", "last_picked_index", "last_picked_type"):
        assert mono_state.get(k) == decomp_state.get(k), f"state[{k}] mismatch"


# ───────────────────────────── pre-LLM scenarios ─────────────────────


def _gs0() -> dict:
    """Episode-init state container (after Initialize)."""
    return {
        "history": "",
        "planning": ["Navigation has just started, with no planning yet."],
        "backtrack": False,
        "nodes_list": [],
        "graph": {},
        "trajectory": [],
        "last_distance": 0.0,
    }


PRE_LLM_SCENARIOS = {
    "step0_basic": {
        "t": 0,
        "gs": _gs0(),
        "inputs": {
            "instruction": "Walk to the kitchen.",
            "candidates": {
                0: {"angle": 0.5, "distance": 1.2, "rgb_base64": ""},
                1: {"angle": -0.3, "distance": 2.0, "rgb_base64": ""},
            },
            "tags": {0: "stove, kitchen", 1: "hallway"},
        },
    },
    "step5_return_eligible": {
        "t": 5,
        "gs": {
            **_gs0(),
            "history": "step 0: turn slight left to Place 1...",
            "last_distance": 2.0,
        },
        "inputs": {
            "instruction": "Walk to the kitchen.",
            "candidates": {
                0: {"angle": 0.2, "distance": 1.5, "rgb_base64": ""},
                1: {"angle": -1.0, "distance": 1.8, "rgb_base64": ""},
                2: {"angle": 1.5, "distance": 1.0, "rgb_base64": ""},
            },
            "tags": {0: "door", 1: "lamp", 2: "wall"},
        },
    },
    "step5_backtrack_latched": {
        "t": 5,
        "gs": {
            **_gs0(),
            "history": "step 0: turn around to Place 1...",
            "backtrack": True,
            "last_distance": 2.0,
        },
        "inputs": {
            "instruction": "Walk to the kitchen.",
            "candidates": {
                0: {"angle": 0.2, "distance": 1.5, "rgb_base64": ""},
                1: {"angle": -1.0, "distance": 1.8, "rgb_base64": ""},
            },
            "tags": {0: "door", 1: "lamp"},
        },
    },
    "step2_stop_threshold": {
        "t": 2,
        "gs": {
            **_gs0(),
            "history": "step 0: go forward, step 1: turn slight left",
            "last_distance": 1.5,
        },
        "inputs": {
            "instruction": "Walk to the kitchen.",
            "candidates": {
                0: {"angle": 0.0, "distance": 2.0, "rgb_base64": ""},
                1: {"angle": 1.0, "distance": 1.5, "rgb_base64": ""},
            },
            "tags": {0: "fridge", 1: "stove"},
        },
    },
}


@pytest.mark.parametrize("name", list(PRE_LLM_SCENARIOS.keys()))
def test_pre_llm_equivalence(name: str) -> None:
    sc = PRE_LLM_SCENARIOS[name]
    mono_out, mono_state = _run_mono_pre_llm(sc["inputs"], sc["t"], sc["gs"])
    decomp_out, decomp_state = _run_decomp_pre_llm(sc["inputs"], sc["t"], sc["gs"])
    _assert_pre_llm_equiv(mono_out, mono_state, decomp_out, decomp_state)


def test_pre_llm_with_real_rgb() -> None:
    """RGB decode path: confirm images + image_labels arrays match
    (covers the second porting bug fix — labels must be 1:1 with images
    and use episode-global node_index, not per-step j)."""
    rgb = _tiny_b64_png()
    inputs = {
        "instruction": "Walk to the kitchen.",
        "candidates": {
            0: {"angle": 0.2, "distance": 1.5, "rgb_base64": rgb},
            1: {"angle": -1.0, "distance": 1.8, "rgb_base64": rgb},
        },
        "tags": {0: "door", 1: "lamp"},
    }
    gs = {**_gs0(), "history": "step 0: go forward", "last_distance": 1.5}
    mono_out, mono_state = _run_mono_pre_llm(inputs, 5, gs)
    decomp_out, decomp_state = _run_decomp_pre_llm(inputs, 5, gs)
    _assert_pre_llm_equiv(mono_out, mono_state, decomp_out, decomp_state)
    # Sanity: both should have produced real images + matching labels.
    assert len(mono_out["images"]) == 2
    assert mono_out["image_labels"] == decomp_out["image_labels"]


# ───────────────────────────── post-LLM scenarios ────────────────────


# A handcrafted candidates_dict matching what build_action_options
# emits (the JSON-stringified manifest used by resolve_action).
_CANDS_3WP = json.dumps(
    {
        0: {"angle": 0.5, "distance": 1.2, "type": "waypoint"},
        1: {"angle": -0.3, "distance": 2.0, "type": "waypoint"},
        2: {"angle": 1.5, "distance": 1.0, "type": "waypoint"},
    }
)

_CANDS_2WP_PLUS_RETURN = json.dumps(
    {
        0: {"angle": 0.2, "distance": 1.5, "type": "waypoint"},
        1: {"angle": -1.0, "distance": 1.8, "type": "waypoint"},
        2: {"angle": 3.14159265, "distance": 2.0, "type": "return"},
    }
)


POST_LLM_SCENARIOS = {
    "fenced_step0_pick_B": {
        "t": 0,
        "gs": {"planning": ["Navigation has just started, with no planning yet."]},
        "inputs": {
            "response": '```json\n{"Thought": "Go to door.", "New Planning": "Move B.", "Action": "B"}\n```',
            "only_options": json.dumps(["A", "B", "C"]),
            "candidates_dict": _CANDS_3WP,
        },
    },
    "bare_step5_pick_C": {
        "t": 5,
        "gs": {"planning": ["init"]},
        "inputs": {
            "response": '{"Thought": "Pick C.", "New Planning": "Approach.", "Action": "C"}',
            "only_options": json.dumps(["A", "B", "C", "D"]),
            "candidates_dict": _CANDS_3WP,
        },
    },
    "prose_wrapped_pick_A_means_stop_at_t5": {
        # t>=2: parse_json_action subtracts 1, so picked letter "A" → 0 - 1 = -1 (STOP).
        "t": 5,
        "gs": {"planning": ["init"]},
        "inputs": {
            "response": (
                "Sure, here is my decision:\n"
                '{"Thought": "Goal reached.", "New Planning": "Stop here.", "Action": "A"}\n'
                "Hope that helps."
            ),
            "only_options": json.dumps(["A", "B", "C", "D"]),
            "candidates_dict": _CANDS_3WP,
        },
    },
    "malformed_fallback_stop_at_t5": {
        # t>=2 + parse failure → index defaults to 0 → -1 (STOP).
        "t": 5,
        "gs": {"planning": ["init"]},
        "inputs": {
            "response": "I cannot follow the JSON format right now sorry.",
            "only_options": json.dumps(["A", "B", "C"]),
            "candidates_dict": _CANDS_3WP,
        },
    },
    "return_picked_at_t5": {
        # only_options has stop+3 waypoints+return = 5 letters.
        # Pick D (idx 3 after -=1 = 2 → candidates_dict["2"] which is type="return").
        "t": 5,
        "gs": {"planning": ["init"]},
        "inputs": {
            "response": '{"Thought": "Backtrack.", "New Planning": "Try left branch.", "Action": "D"}',
            "only_options": json.dumps(["A", "B", "C", "D"]),
            "candidates_dict": _CANDS_2WP_PLUS_RETURN,
        },
    },
}


@pytest.mark.parametrize("name", list(POST_LLM_SCENARIOS.keys()))
def test_post_llm_equivalence(name: str) -> None:
    sc = POST_LLM_SCENARIOS[name]
    mono_out, mono_state = _run_mono_post_llm(sc["inputs"], sc["t"], sc["gs"])
    decomp_out, decomp_state = _run_decomp_post_llm(sc["inputs"], sc["t"], sc["gs"])
    _assert_post_llm_equiv(mono_out, mono_state, decomp_out, decomp_state)
