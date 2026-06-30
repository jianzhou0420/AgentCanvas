"""SmartWay nodeset — Phase-1 monolith port (IROS 2025 zero-shot VLN-CE).

Upstream: https://github.com/sxyxs/SmartWay-Code @ daa2dd8 (MIT)
Re-fetch: workspace/nodesets/_upstream/smartway-code/fetch_upstream.sh
Paper:    https://arxiv.org/abs/2503.10069

This is a `/implement-graph` Phase-1 monolith. The per-step body that
upstream splits into five `OneStagePromptManager` methods collapses into
**three** method-side nodes here — the minimum the graph topology allows
given that an LLM call and an env step are interposed inside one
iteration:

  plan_step       (segment 1, pre-LLM)   prompt_assembly + bookkeeping
  decide_action   (segment 2, post-LLM,  parse_json_action + parse_json_planning
                   pre-step)             + make_equiv_action (merged)
  update_history  (segment 3, post-step) make_history + last_distance update

Compared to the older `workspace/nodesets/smartway/` (5 nodes), `decide_action`
absorbs both `parse_action` and `equiv_action` — they sit in the same
topological slot (between `llmCall` and `step_hightolow`) with no external
boundary between them, so the monolith-first methodology keeps them as
one node until A/B-ing or architect mutation demands the seam.

The backtrack latch (the load-bearing rule — don't offer return twice in
a row) survives unchanged: `plan_step` reads + clears `graph_state.backtrack`
at the top, `update_history` sets it back to `True` iff the picked option
was the synthetic return phrase.

Expected `graph_state` entries (declared on the graph, lifetime=episode
unless noted):

  history                str    initial ""
  planning               ANY    initial [DEFAULT_PLANNING]
  backtrack              BOOL   initial False    (the latch)
  nodes_list             ANY    initial []
  graph                  ANY    initial {}
  trajectory             ANY    initial []
  last_distance          ANY    initial 0.0
  last_picked_index      TEXT   lifetime=step
  last_picked_type       TEXT   lifetime=step
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

from ._prompts import (
    DEFAULT_PLANNING,
    assemble_prompt,
    build_task_description,
    make_action_prompts,
    parse_json_action,
    parse_json_planning,
    prepend_stop_options,
)


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


class SmartwayMonoPlanStepNode(BaseCanvasNode):
    """Pre-LLM segment: assemble prompt + per-candidate images + manifest.

    Mirrors `make_action_prompt_backtrackv2` + `make_r2r_json_prompts`
    (one_stage_prompt_manager.py:44 + :202). Reads + clears the
    `backtrack` latch, builds `candidates_dict` (with optional synthetic
    return when eligible), letters-prefixes options, decodes RGB tiles
    for the multi-image GPT-4o call, and updates the topology bookkeeping
    state (`nodes_list` / `graph` / `trajectory`).
    """

    node_type: ClassVar[str] = "smartway_mono__plan_step"
    display_name: ClassVar[str] = "SmartWay (mono): Plan Step"
    description: ClassVar[str] = (
        "Pre-LLM: read+clear backtrack latch, build candidates manifest "
        "(with optional synthetic return), render prompt + image list."
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Network"
    input_ports: ClassVar[list] = [
        PortDef("instruction", "TEXT", "Navigation instruction"),
        PortDef("candidates", "ANY", "{idx: {angle, distance, rgb_base64}}"),
        PortDef("tags", "ANY", "{idx: tag_string} from RAM+"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("task_description", "TEXT", "System prompt (verbatim)"),
        PortDef("prompt", "TEXT", "User prompt"),
        PortDef("images", "LIST[IMAGE]", "Per-candidate RGB tiles"),
        PortDef(
            "image_labels",
            "LIST[TEXT]",
            'Per-image "Image {node_index}:" captions — 1:1 with images, '
            "matches the `Place {node_index}` IDs in the prompt text "
            "(mirrors upstream gpt_infer_back_track image_list/img_id loop).",
        ),
        PortDef("only_options", "TEXT", "JSON list of letter ids"),
        PortDef("only_actions", "TEXT", "JSON list of action phrases (no stop)"),
        PortDef("candidates_dict", "TEXT", "JSON {idx: {angle, distance, type}}"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        instruction = str(inputs.get("instruction", ""))
        raw_cands = inputs.get("candidates") or {}
        tags_in = inputs.get("tags") or {}
        t = int(getattr(ctx, "step", 0)) if ctx else 0

        gs = getattr(ctx, "graph_state", None) if ctx else None

        backtrack = bool(_read(gs, "backtrack", False))
        last_backtrack = backtrack
        _write(gs, "backtrack", False)

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

        # one_stage_prompt_manager.py:81-90 — synthetic return option
        if not last_backtrack and last_distance != 0.0 and t != 0:
            synth_idx = len(candidates_dict)
            candidates_dict[synth_idx] = {
                "angle": math.pi,
                "distance": last_distance,
                "rgb_base64": "",
                "type": "return",
                "src_idx": -1,
            }

        res_list: list[str] = []
        for j in candidates_dict:
            entry = candidates_dict[j]
            if entry["type"] == "return":
                res_list.append("")
                continue
            src_idx = entry.get("src_idx", j)
            tag = tags_in.get(src_idx, tags_in.get(str(src_idx), ""))
            res_list.append(str(tag))

        # Topology bookkeeping (one_stage_prompt_manager.py:65-95, 119-120)
        # MUST run before make_action_prompts so candidate node_indices are
        # known when building the per-candidate "Place {node_idx}" strings.
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
            # node_index = nodes_list[0].index(waypoint_id) — upstream line 106.
            # Just-appended fresh UUID → equals len(nodes_list)-1.
            candidate_node_indices.append(nodes_list.index(wp_id))
        if current_vp not in graph:
            graph[current_vp] = cand_vpids

        _write(gs, "nodes_list", nodes_list)
        _write(gs, "graph", graph)
        _write(gs, "trajectory", trajectory)

        action_prompts = make_action_prompts(
            candidates_dict,
            res_list,
            candidate_node_indices,
            t=t,
            last_backtrack=last_backtrack,
        )
        full_options, only_options = prepend_stop_options(action_prompts, t=t)

        planning_latest = str(planning[-1])
        user_prompt = assemble_prompt(
            instruction=instruction,
            history=history,
            planning_latest=planning_latest,
            action_options=full_options,
            t=t,
        )

        # Build images + parallel "Image {node_index}:" labels — upstream
        # gpt_infer_back_track (api.py:48-95) inserts a text label before
        # each image so the model can map "Place N" prompt text → image N.
        # Without these labels the action options say "Place 7 corresponds
        # to Image 7" but the model can't tell which image is Image 7.
        # See cand_inputs['cand_index'][0] used as img_id in upstream.
        images: list = []
        image_labels: list[str] = []
        for j in candidates_dict:
            rgb_b64 = candidates_dict[j].get("rgb_base64", "")
            if not rgb_b64:
                continue
            arr = _decode_rgb(rgb_b64)
            if arr is not None:
                images.append(arr)
                image_labels.append(f"Image {candidate_node_indices[j]}:")

        cands_out = {
            j: {
                "angle": v["angle"],
                "distance": v["distance"],
                "type": v["type"],
            }
            for j, v in candidates_dict.items()
        }

        self._self_log("step", t)
        self._self_log("last_backtrack", last_backtrack)
        self._self_log("n_candidates", len(candidates_dict))
        self._self_log("n_images", len(images))
        self._self_log(
            "offered_return", any(c["type"] == "return" for c in candidates_dict.values())
        )
        self._self_log("only_options", only_options)
        self._self_log("prompt_preview", user_prompt[-300:])

        return {
            "task_description": build_task_description(),
            "prompt": user_prompt,
            "images": images,
            "image_labels": image_labels,
            "only_options": json.dumps(only_options),
            "only_actions": json.dumps(action_prompts),
            "candidates_dict": json.dumps(cands_out),
        }


class SmartwayMonoDecideActionNode(BaseCanvasNode):
    """Post-LLM segment: parse JSON → picked option → (angle, distance).

    Collapses upstream `parse_json_action` + `parse_json_planning` +
    `make_equiv_action` into one post-LLM node — they share the same
    topological slot (between `llmCall` and `step_hightolow`) with no
    external boundary between them.

    * Tolerant JSON parse on the LLM response (strips markdown fences,
      extracts outermost ``{...}`` if embedded in prose).
    * Letter → index via `only_options.index(); after which
      ``index -= 1`` when ``t >= 2`` (so ``-1`` = STOP).
    * Index → ``candidates_dict[idx]`` → ``angle`` / ``distance``.
    * Appends ``New Planning`` to the planning list in `graph_state`.
    * Surfaces ``is_return`` so `update_history` knows to set the latch.
    """

    node_type: ClassVar[str] = "smartway_mono__decide_action"
    display_name: ClassVar[str] = "SmartWay (mono): Decide Action"
    description: ClassVar[str] = (
        "Parse LLM JSON → picked_index → (angle, distance); append planning; "
        "surface is_stop / is_return for downstream."
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Navigation"
    input_ports: ClassVar[list] = [
        PortDef("response", "TEXT", "LLM JSON response text"),
        PortDef("only_options", "TEXT", "JSON list of letters from plan_step"),
        PortDef("candidates_dict", "TEXT", "JSON candidates manifest from plan_step"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("picked_index", "TEXT", "Integer index (or -1 for STOP)"),
        PortDef("is_stop", "BOOL", "True when picked_index == -1"),
        PortDef("is_return", "BOOL", "True when picked option was synthetic return"),
        PortDef("angle", "TEXT", "Action angle in radians"),
        PortDef("distance", "TEXT", "Action distance in metres"),
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
        planning = _read(gs, "planning", [DEFAULT_PLANNING])
        if not isinstance(planning, list):
            planning = [DEFAULT_PLANNING]
        planning = [*list(planning), new_planning]
        _write(gs, "planning", planning)
        _write(gs, "last_picked_index", str(picked))
        _write(
            gs, "last_picked_type", "return" if is_return else ("stop" if is_stop else "waypoint")
        )

        if parse_error:
            self._self_log("parse_error", parse_error)
        self._self_log("picked_index", picked)
        self._self_log("is_stop", is_stop)
        self._self_log("is_return", is_return)
        self._self_log("angle_rad", angle)
        self._self_log("distance_m", distance)
        self._self_log("thought", thought[:200])
        self._self_log("new_planning", new_planning[:200])

        return {
            "picked_index": str(picked),
            "is_stop": is_stop,
            "is_return": is_return,
            "angle": f"{angle:.6f}",
            "distance": f"{distance:.6f}",
            "thought": thought,
            "new_planning": new_planning,
        }


class SmartwayMonoUpdateHistoryNode(BaseCanvasNode):
    """Post-step segment: append history; set the backtrack latch.

    Mirrors `make_history` (one_stage_prompt_manager.py:168-184). Appends
    ``"step N: <action>"`` to `history`; sets `backtrack=True` iff the
    picked option was the synthetic return — the next `plan_step` reads
    + clears the latch, suppressing the return option that step.

    Also updates `last_distance` (base_il_trainer.py:55-57) so the next
    step's `plan_step` knows whether to offer a return at all.
    """

    node_type: ClassVar[str] = "smartway_mono__update_history"
    display_name: ClassVar[str] = "SmartWay (mono): Update History"
    description: ClassVar[str] = (
        "Append 'step N: <action>' to history; set backtrack=True if return "
        "picked; update last_distance for next-step return-option offer."
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "ClipboardList"
    input_ports: ClassVar[list] = [
        PortDef("picked_index", "TEXT", "Integer index from decide_action"),
        PortDef("is_stop", "BOOL", "True when picked_index == -1"),
        PortDef("is_return", "BOOL", "True when picked was synthetic return"),
        PortDef("only_actions", "TEXT", "JSON action_prompts (no stop) from plan_step"),
        PortDef("distance", "TEXT", "Picked distance (from decide_action)"),
        PortDef(
            "step_done", "BOOL", "env step done flag (gates updates on episode end)", optional=True
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


class SmartwayMonoNodeSet(BaseNodeSet):
    """SmartWay (IROS 2025) — Phase-1 monolith port (3 method nodes)."""

    name = "smartway_mono"
    description = (
        "SmartWay monolith port: pre-LLM plan_step, post-LLM decide_action "
        "(parse+equiv merged), post-step update_history (with backtrack latch)."
    )

    def get_tools(self) -> list:
        return [
            SmartwayMonoPlanStepNode(),
            SmartwayMonoDecideActionNode(),
            SmartwayMonoUpdateHistoryNode(),
        ]
