"""SmartWay nodeset — Phase-2 decomposition of smartway_monolith.

Same upstream as smartway_mono (SmartWay-Code @ daa2dd8,
arxiv 2503.10069). Splits the validated 3-node monolith along natural seams:

    plan_step           →  update_topology    (sole writer: nodes_list/graph/trajectory;
                                               state snapshots for downstream)
                           build_action_options (pure: per-cand "Place N" + stop letters)
                           assemble_prompt    (pure: user-prompt template render)
                           build_images       (pure: RGB decode + image_labels)

    decide_action       →  parse_response     (JSON parse + extract; sole writer: planning)
                           resolve_action     (picked → angle/distance/is_return)

    update_history      →  update_history     (unchanged; sole writer: history/backtrack/
                                               last_distance — incl. backtrack=is_return,
                                               removing the read-clear-at-top pattern)

smartway_mono remains the ground-truth for equivalence testing — see
``test_equivalence.py``. State-key ownership is single-writer per key after
the split.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import uuid
from contextlib import suppress
from typing import Any, ClassVar

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    NodeUIConfig,
    PortDef,
)
from workspace.nodesets.method.smartway_mono._prompts import (
    DEFAULT_PLANNING,
    assemble_prompt,
    build_task_description,
    make_action_prompts,
    parse_json_action,
    parse_json_planning,
    prepend_stop_options,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers (duplicated minimally from smartway_mono — avoid cross-import for
# helpers that are intentionally per-nodeset implementation details).
# ═══════════════════════════════════════════════════════════════════════


def _read(gs: Any, key: str, default: Any) -> Any:
    if gs is None:
        return default
    try:
        v = gs.read(key)
        return v if v is not None else default
    except Exception:
        return default


def _write(gs: Any, key: str, value: Any) -> None:
    if gs is None:
        return
    with suppress(Exception):
        gs.write(key, value)


def _decode_rgb(b64: str):
    if not b64:
        return None
    try:
        import numpy as np
        from PIL import Image

        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(img, dtype="uint8")
    except Exception:
        return None


def _coerce_candidates(raw: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for k, v in raw.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            entry = dict(v)
            entry.setdefault("type", "waypoint")
            try:
                entry["angle"] = float(entry.get("angle", 0.0))
                entry["distance"] = float(entry.get("distance", 0.0))
            except (TypeError, ValueError):
                continue
            out[idx] = entry
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                out[idx] = {
                    "angle": float(v[0]),
                    "distance": float(v[1]),
                    "rgb_base64": "",
                    "type": "waypoint",
                }
            except (TypeError, ValueError):
                continue
    return out


# ═══════════════════════════════════════════════════════════════════════
# Pre-LLM segment (4 nodes)
# ═══════════════════════════════════════════════════════════════════════


class SmartwayUpdateTopologyNode(BaseCanvasNode):
    """First method node post-iter_in. Reads state snapshot, builds the
    enriched candidates dict (with optional synthetic return), and is the
    SOLE WRITER of ``nodes_list`` / ``graph`` / ``trajectory``.

    Mirrors the topology-bookkeeping portion of upstream
    ``make_action_prompt_backtrackv2`` (one_stage_prompt_manager.py:65-95,
    119-120). The synthetic-return merge mirrors lines 81-90.

    State snapshots (``history_snap`` / ``planning_snap``) are emitted as
    explicit wires so downstream nodes don't re-read graph_state mid-iter.
    """

    node_type: ClassVar[str] = "smartway__update_topology"
    display_name: ClassVar[str] = "SmartWay: Update Topology"
    description: ClassVar[str] = (
        "Sole writer of nodes_list/graph/trajectory; emits enriched candidates "
        "+ state snapshots for the rest of the iteration."
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Network"
    input_ports: ClassVar[list] = [
        PortDef("candidates", "ANY", "{idx: {angle, distance, rgb_base64}} from waypoint_predict"),
    ]
    output_ports: ClassVar[list] = [
        PortDef(
            "candidates_enriched", "ANY", "{idx: {angle, distance, rgb_base64, type, src_idx}}"
        ),
        PortDef(
            "candidate_node_indices",
            "ANY",
            "List[int] episode-global node indices, 1:1 with enriched keys",
        ),
        PortDef("last_backtrack", "BOOL", "backtrack latch snapshot at iter start"),
        PortDef("history_snap", "TEXT", "history string snapshot"),
        PortDef("planning_snap", "TEXT", "planning[-1] snapshot"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        raw_cands = inputs.get("candidates") or {}
        t = int(getattr(ctx, "step", 0)) if ctx else 0

        gs = getattr(ctx, "graph_state", None) if ctx else None

        last_backtrack = bool(_read(gs, "backtrack", False))
        history = str(_read(gs, "history", ""))
        planning = _read(gs, "planning", [DEFAULT_PLANNING])
        if not isinstance(planning, list) or not planning:
            planning = [DEFAULT_PLANNING]
        last_distance = float(_read(gs, "last_distance", 0.0))

        cands_in = _coerce_candidates(raw_cands)
        sorted_ids = sorted(cands_in.keys())
        candidates_dict: dict[int, dict[str, Any]] = {}
        for src_idx in sorted_ids:
            entry = cands_in[src_idx]
            candidates_dict[len(candidates_dict)] = {
                "angle": float(entry.get("angle", 0.0)),
                "distance": float(entry.get("distance", 0.0)),
                "rgb_base64": str(entry.get("rgb_base64", "")),
                "type": "waypoint",
                "src_idx": src_idx,
            }

        # synthetic return option (one_stage_prompt_manager.py:81-90)
        if not last_backtrack and last_distance != 0.0 and t != 0:
            synth_idx = len(candidates_dict)
            candidates_dict[synth_idx] = {
                "angle": math.pi,
                "distance": last_distance,
                "rgb_base64": "",
                "type": "return",
                "src_idx": -1,
            }

        # Topology bookkeeping — sole-writer responsibility for this node.
        nodes_list = list(_read(gs, "nodes_list", []))
        graph = dict(_read(gs, "graph", {}))
        trajectory = list(_read(gs, "trajectory", []))

        current_vp = str(uuid.uuid4())
        if current_vp not in nodes_list:
            nodes_list.append(current_vp)
        trajectory.append(current_vp)
        cand_vpids: list[str] = []
        candidate_node_indices: list[int] = []
        for _j in candidates_dict:
            wp_id = str(uuid.uuid4())
            cand_vpids.append(wp_id)
            if wp_id not in nodes_list:
                nodes_list.append(wp_id)
            candidate_node_indices.append(nodes_list.index(wp_id))
        if current_vp not in graph:
            graph[current_vp] = cand_vpids

        _write(gs, "nodes_list", nodes_list)
        _write(gs, "graph", graph)
        _write(gs, "trajectory", trajectory)

        planning_latest = str(planning[-1])

        self._self_log("step", t)
        self._self_log("last_backtrack", last_backtrack)
        self._self_log("n_candidates", len(candidates_dict))
        self._self_log(
            "offered_return",
            any(c["type"] == "return" for c in candidates_dict.values()),
        )

        return {
            "candidates_enriched": candidates_dict,
            "candidate_node_indices": candidate_node_indices,
            "last_backtrack": last_backtrack,
            "history_snap": history,
            "planning_snap": planning_latest,
        }


class SmartwayBuildActionOptionsNode(BaseCanvasNode):
    """Pure: build per-candidate "Place N" phrases + prepend stop letters.

    Mirrors ``make_action_prompts`` (the inner loop of
    one_stage_prompt_manager.py:98-113) + ``make_action_options_backtrack``
    (line 148-165).
    """

    node_type: ClassVar[str] = "smartway__build_action_options"
    display_name: ClassVar[str] = "SmartWay: Build Action Options"
    description: ClassVar[str] = "Per-candidate 'Place N' phrases + letter-prefix + stop option."
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "ListChecks"
    input_ports: ClassVar[list] = [
        PortDef("candidates_enriched", "ANY", "From update_topology"),
        PortDef("candidate_node_indices", "ANY", "From update_topology"),
        PortDef("last_backtrack", "BOOL", "From update_topology"),
        PortDef("tags", "ANY", "{src_idx: tag_string} from perception_tag"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("full_options", "ANY", "Letter-prefixed options list (for the prompt)"),
        PortDef("only_options", "TEXT", "JSON list of letters"),
        PortDef("only_actions", "TEXT", "JSON list of action phrases (no stop)"),
        PortDef(
            "candidates_dict", "TEXT", "JSON {idx: {angle, distance, type}} for resolve_action"
        ),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        candidates_dict = inputs.get("candidates_enriched") or {}
        node_indices = list(inputs.get("candidate_node_indices") or [])
        last_backtrack = bool(inputs.get("last_backtrack", False))
        tags_in = inputs.get("tags") or {}
        t = int(getattr(ctx, "step", 0)) if ctx else 0

        # Re-build res_list from candidates + tags. tags_in keyed by src_idx.
        res_list: list[str] = []
        for j in candidates_dict:
            entry = candidates_dict[j]
            if entry["type"] == "return":
                res_list.append("")
                continue
            src_idx = entry.get("src_idx", j)
            tag = tags_in.get(src_idx, tags_in.get(str(src_idx), ""))
            res_list.append(str(tag))

        action_prompts = make_action_prompts(
            candidates_dict,
            res_list,
            node_indices,
            t=t,
            last_backtrack=last_backtrack,
        )
        full_options, only_options = prepend_stop_options(action_prompts, t=t)

        cands_out = {
            j: {
                "angle": v["angle"],
                "distance": v["distance"],
                "type": v["type"],
            }
            for j, v in candidates_dict.items()
        }

        self._self_log("n_options", len(full_options))
        self._self_log("only_options", only_options)

        return {
            "full_options": full_options,
            "only_options": json.dumps(only_options),
            "only_actions": json.dumps(action_prompts),
            "candidates_dict": json.dumps(cands_out),
        }


class SmartwayAssemblePromptNode(BaseCanvasNode):
    """Pure: template render. Verbatim ``make_r2r_json_prompts``
    (one_stage_prompt_manager.py:202-245) — emits both the static
    ``task_description`` (system prompt) and the per-step user prompt.
    """

    node_type: ClassVar[str] = "smartway__assemble_prompt"
    display_name: ClassVar[str] = "SmartWay: Assemble Prompt"
    description: ClassVar[str] = "Render user prompt template + emit static task_description."
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "FileText"
    input_ports: ClassVar[list] = [
        PortDef("instruction", "TEXT", "Navigation instruction"),
        PortDef("history_snap", "TEXT", "From update_topology"),
        PortDef("planning_snap", "TEXT", "From update_topology"),
        PortDef("full_options", "ANY", "Letter-prefixed options from build_action_options"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("task_description", "TEXT", "System prompt (verbatim)"),
        PortDef("prompt", "TEXT", "User prompt"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        instruction = str(inputs.get("instruction", ""))
        history_snap = str(inputs.get("history_snap", ""))
        planning_snap = str(inputs.get("planning_snap", DEFAULT_PLANNING))
        full_options = inputs.get("full_options") or []
        t = int(getattr(ctx, "step", 0)) if ctx else 0

        user_prompt = assemble_prompt(
            instruction=instruction,
            history=history_snap,
            planning_latest=planning_snap,
            action_options=list(full_options),
            t=t,
        )

        self._self_log("step", t)
        self._self_log("prompt_preview", user_prompt[-300:])

        return {
            "task_description": build_task_description(),
            "prompt": user_prompt,
        }


class SmartwayBuildImagesNode(BaseCanvasNode):
    """Pure: decode per-candidate RGB tiles + emit parallel
    ``Image {node_index}:`` labels.

    Mirrors the image/img_id construction in upstream ``gpt_infer_back_track``
    (api.py:48-95) — the label-before-image pattern that makes the model's
    "Place N corresponds to Image N" mapping interpretable.
    """

    node_type: ClassVar[str] = "smartway__build_images"
    display_name: ClassVar[str] = "SmartWay: Build Images"
    description: ClassVar[str] = "Decode RGB tiles + emit 'Image N:' captions (parallel arrays)."
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Image"
    input_ports: ClassVar[list] = [
        PortDef("candidates_enriched", "ANY", "From update_topology"),
        PortDef("candidate_node_indices", "ANY", "From update_topology"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("images", "LIST[IMAGE]", "Per-candidate RGB tiles"),
        PortDef(
            "image_labels",
            "LIST[TEXT]",
            'Per-image "Image {node_index}:" captions — 1:1 with images.',
        ),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        candidates_dict = inputs.get("candidates_enriched") or {}
        node_indices = list(inputs.get("candidate_node_indices") or [])

        images: list = []
        image_labels: list[str] = []
        for j in candidates_dict:
            rgb_b64 = candidates_dict[j].get("rgb_base64", "")
            if not rgb_b64:
                continue
            arr = _decode_rgb(rgb_b64)
            if arr is not None:
                images.append(arr)
                image_labels.append(f"Image {node_indices[j]}:")

        self._self_log("n_images", len(images))

        return {"images": images, "image_labels": image_labels}


# ═══════════════════════════════════════════════════════════════════════
# Post-LLM segment (2 nodes)
# ═══════════════════════════════════════════════════════════════════════


class SmartwayParseResponseNode(BaseCanvasNode):
    """Post-LLM #1: tolerant JSON parse → extract Action letter index
    + Thought + New Planning. SOLE WRITER of ``planning`` state.

    Mirrors ``parse_json_action`` + ``parse_json_planning``
    (one_stage_prompt_manager.py:284-309), plus the markdown-fence /
    embedded-prose stripping our port needs (LLMs sometimes wrap JSON).
    """

    node_type: ClassVar[str] = "smartway__parse_response"
    display_name: ClassVar[str] = "SmartWay: Parse Response"
    description: ClassVar[str] = (
        "Parse LLM JSON → picked_index/thought/new_planning; sole writer of planning."
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Parentheses"
    input_ports: ClassVar[list] = [
        PortDef("response", "TEXT", "LLM JSON response text"),
        PortDef("only_options", "TEXT", "JSON list of letters from build_action_options"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("picked_index", "TEXT", "Integer index (or -1 for STOP)"),
        PortDef("is_stop", "BOOL", "True when picked_index == -1"),
        PortDef("thought", "TEXT", "LLM Thought field"),
        PortDef("new_planning", "TEXT", "LLM New Planning field"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        response = str(inputs.get("response", "")).strip()
        t = int(getattr(ctx, "step", 0)) if ctx else 0

        only_options: list[str] = []
        with suppress(json.JSONDecodeError, TypeError):
            v = json.loads(str(inputs.get("only_options", "[]")))
            if isinstance(v, list):
                only_options = [str(x) for x in v]

        cleaned = response
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        elif "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if 0 <= start < end:
                cleaned = cleaned[start : end + 1]

        parsed: dict[str, Any] = {}
        parse_error: str | None = None
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                parsed = {}
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = f"json: {exc}"

        picked = parse_json_action(parsed, only_options, t=t)
        new_planning = parse_json_planning(parsed)
        thought = str(parsed.get("Thought", ""))
        is_stop = picked == -1

        gs = getattr(ctx, "graph_state", None) if ctx else None
        planning = _read(gs, "planning", [DEFAULT_PLANNING])
        if not isinstance(planning, list):
            planning = [DEFAULT_PLANNING]
        planning = [*list(planning), new_planning]
        _write(gs, "planning", planning)

        if parse_error:
            self._self_log("parse_error", parse_error)
        self._self_log("picked_index", picked)
        self._self_log("is_stop", is_stop)
        self._self_log("thought", thought[:200])
        self._self_log("new_planning", new_planning[:200])

        return {
            "picked_index": str(picked),
            "is_stop": is_stop,
            "thought": thought,
            "new_planning": new_planning,
        }


class SmartwayResolveActionNode(BaseCanvasNode):
    """Post-LLM #2: picked_index → angle/distance/is_return via
    ``candidates_dict`` lookup. Writes step-scratch state
    ``last_picked_index`` / ``last_picked_type``.

    Mirrors the ``make_equiv_action`` portion (base_il_trainer.py:167-204)
    — the index → env-action translation.
    """

    node_type: ClassVar[str] = "smartway__resolve_action"
    display_name: ClassVar[str] = "SmartWay: Resolve Action"
    description: ClassVar[str] = "picked_index + candidates_dict → angle / distance / is_return."
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Navigation"
    input_ports: ClassVar[list] = [
        PortDef("picked_index", "TEXT", "From parse_response"),
        PortDef("is_stop", "BOOL", "From parse_response"),
        PortDef("candidates_dict", "TEXT", "JSON manifest from build_action_options"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("angle", "TEXT", "Action angle in radians"),
        PortDef("distance", "TEXT", "Action distance in metres"),
        PortDef("is_return", "BOOL", "True when picked option was synthetic return"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            picked = int(str(inputs.get("picked_index", "-1")))
        except (TypeError, ValueError):
            picked = -1
        is_stop = bool(inputs.get("is_stop", False))

        cands: dict[str, dict[str, Any]] = {}
        with suppress(json.JSONDecodeError, TypeError):
            v = json.loads(str(inputs.get("candidates_dict", "{}")))
            if isinstance(v, dict):
                cands = v

        is_return = False
        angle = 0.0
        distance = 0.0
        if not is_stop:
            entry = cands.get(str(picked)) or cands.get(picked)  # type: ignore[arg-type]
            if isinstance(entry, dict):
                angle = float(entry.get("angle", 0.0))
                distance = float(entry.get("distance", 0.0))
                is_return = bool(entry.get("type") == "return")
            else:
                self._self_log("equiv_lookup_miss", picked)

        gs = getattr(ctx, "graph_state", None) if ctx else None
        _write(gs, "last_picked_index", str(picked))
        _write(
            gs,
            "last_picked_type",
            "return" if is_return else ("stop" if is_stop else "waypoint"),
        )

        self._self_log("picked_index", picked)
        self._self_log("is_return", is_return)
        self._self_log("angle_rad", angle)
        self._self_log("distance_m", distance)

        return {
            "angle": f"{angle:.6f}",
            "distance": f"{distance:.6f}",
            "is_return": is_return,
        }


# ═══════════════════════════════════════════════════════════════════════
# Post-step (unchanged from smartway_mono)
# ═══════════════════════════════════════════════════════════════════════


class SmartwayUpdateHistoryNode(BaseCanvasNode):
    """Post-step: append history; SOLE WRITER of
    ``history`` / ``backtrack`` / ``last_distance``.

    Mirrors ``make_history`` (one_stage_prompt_manager.py:168-184) +
    ``last_distance`` update (base_il_trainer.py:55-57). Unlike smartway_mono,
    this is now the sole writer of ``backtrack`` — ``backtrack = is_return``
    every step, container ``initial_value=False`` handles iter 0. No
    read-clear pattern at iter top.
    """

    node_type: ClassVar[str] = "smartway__update_history"
    display_name: ClassVar[str] = "SmartWay: Update History"
    description: ClassVar[str] = (
        "Append step entry to history; backtrack=is_return; update last_distance."
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "ClipboardList"
    input_ports: ClassVar[list] = [
        PortDef("picked_index", "TEXT", "From parse_response"),
        PortDef("is_stop", "BOOL", "From parse_response"),
        PortDef("is_return", "BOOL", "From resolve_action"),
        PortDef("only_actions", "TEXT", "JSON action_prompts (no stop) from build_action_options"),
        PortDef("distance", "TEXT", "From resolve_action"),
        PortDef(
            "step_done",
            "BOOL",
            "env step done flag (gates updates on episode end)",
            optional=True,
        ),
    ]
    output_ports: ClassVar[list] = [
        PortDef("history", "TEXT", "Updated history text"),
        PortDef("backtrack", "BOOL", "Updated latch value"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            picked = int(str(inputs.get("picked_index", "-1")))
        except (TypeError, ValueError):
            picked = -1
        is_stop = bool(inputs.get("is_stop", False))
        is_return = bool(inputs.get("is_return", False))

        only_actions: list[str] = []
        with suppress(json.JSONDecodeError, TypeError):
            v = json.loads(str(inputs.get("only_actions", "[]")))
            if isinstance(v, list):
                only_actions = [str(x) for x in v]

        try:
            distance = float(str(inputs.get("distance", "0")))
        except (TypeError, ValueError):
            distance = 0.0

        if is_stop:
            last_action = "stop"
        elif 0 <= picked < len(only_actions):
            last_action = only_actions[picked]
        else:
            last_action = ""

        t = int(getattr(ctx, "step", 0)) if ctx else 0

        gs = getattr(ctx, "graph_state", None) if ctx else None
        prior = str(_read(gs, "history", ""))

        if t == 0 or not prior:
            new_hist = f"step {t}: {last_action}"
        else:
            new_hist = prior + f", step {t}: {last_action}"

        new_backtrack = is_return

        _write(gs, "history", new_hist)
        _write(gs, "backtrack", new_backtrack)

        if not is_stop and distance > 0:
            _write(gs, "last_distance", distance)

        self._self_log("step", t)
        self._self_log("picked", picked)
        self._self_log("last_action", last_action[:120])
        self._self_log("new_backtrack", new_backtrack)
        self._self_log("history_preview", new_hist[-200:])

        return {"history": new_hist, "backtrack": new_backtrack}


# ═══════════════════════════════════════════════════════════════════════
# Nodeset registration
# ═══════════════════════════════════════════════════════════════════════


class SmartwayNodeSet(BaseNodeSet):
    """SmartWay — Phase-2 decomposition (7 method nodes)."""

    name = "smartway"
    description = (
        "SmartWay decomposed: update_topology / build_action_options / "
        "assemble_prompt / build_images / parse_response / resolve_action / "
        "update_history."
    )

    def get_tools(self) -> list:
        return [
            SmartwayUpdateTopologyNode(),
            SmartwayBuildActionOptionsNode(),
            SmartwayAssemblePromptNode(),
            SmartwayBuildImagesNode(),
            SmartwayParseResponseNode(),
            SmartwayResolveActionNode(),
            SmartwayUpdateHistoryNode(),
        ]
