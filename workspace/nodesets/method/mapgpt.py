"""MapGPT nodeset — ACL 2024 zero-shot VLN.

Source: https://github.com/chen-judge/MapGPT @ 7c642f4 (no upstream LICENSE).
Paper:  https://arxiv.org/abs/2401.07314
Re-fetch upstream: workspace/nodesets/_upstream/mapgpt/fetch_upstream.sh

Linguistic-form topological map embedded in the LLM prompt plus
adaptive multi-step planning carried across iterations. Prompts,
parser regex, and rollout constants are verbatim from the source.

The reference's ``make_action_prompt`` + ``make_action_options`` +
``make_r2r_prompts`` pipeline is split across four nodes here. Direction
phrases ("turn left", "go forward") require both current-heading and
candidate-heading, which both live at the env boundary, so they are
frozen into ``candidates_json`` inside ``observe`` and ride downstream
as static records — earlier splits that deferred direction computation
to a view-serializer stage lost access to the current heading.

  observe         — *env → features*, pure. Per candidate, compute the
                    direction word from ``cand_heading - cur_heading`` and
                    pick the candidate's image as the env per-view
                    ``views[view_index]`` (nearest heading+elevation; the
                    env already attaches ``view_index`` to each navigable
                    record). Emits ``candidates_json`` (records) + an
                    aligned ``candidate_tiles`` list.
  update_map      — *single owner of the topo_map state*. Folds the
                    current viewpoint and candidate records into
                    ``topo_map`` (first-visit adjacency, tile cache),
                    writes back, and emits the merged ``topo_snapshot``
                    for downstream nodes in the same iter.
  build_options   — *option assembler*. From ``topo_snapshot`` + the
                    raw candidates, build the letter-prefixed options
                    text and a manifest ``[{letter, vp, phrase}, …]``.
                    Stop gating: when ``ctx.step >= stop_after``,
                    prepends a ``{letter:"A", vp:"STOP",
                    phrase:"stop"}`` record.
  render_prompt   — *prompt + image pack*. Reads ``history`` /
                    ``planning`` from state, formats trajectory / map /
                    supplementary text from ``topo_snapshot``, and
                    packs ``node_imgs`` into ``image_list`` with
                    ``"Image i:"`` labels aligned to Place IDs.
  parse_action    — *manifest consumer*. Reads letter from LLM, looks
                    up the record, emits ``viewpoint_id`` + the raw
                    ``action_phrase`` chosen. Writes new planning.
  update_history  — *state writer*. Appends
                    ``step t: <action_phrase>`` to ``history`` — same
                    payload shape as the reference's ``make_history``.
  image_budget    — *guard*. Emits ``done=True`` when ``image_count``
                    exceeds the budget.
  system_prompt   — *static emitter*. Verbatim JSON-mode system
                    prompt (gpt-4o).

Expected graph-state entries (declared in the graph JSON):
  topo_map  — LastWrite ANY, lifetime="run" (dict-of-four-lists)
  history   — LastWrite TEXT, lifetime="run"
  planning  — LastWrite TEXT, lifetime="run", initial=_DEFAULT_PLANNING
"""

from __future__ import annotations

import copy
import json
import math
import re
from contextlib import suppress
from typing import Any, ClassVar

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)

# ═══════════════════════════════════════════════════════════════════════
# Verbatim constants — character-for-character from the MapGPT source.
# ═══════════════════════════════════════════════════════════════════════

# System prompt, JSON mode (gpt-4o).
# Source: GPT/one_stage_prompt_manager.py:218-236 (make_r2r_json_prompts).
# The reference repo also ships a free-text variant (make_r2r_prompts, lines
# 169-187) tailored for gpt-4-vision-preview's weaker JSON-mode support — we
# only use the gpt-4o JSON path, so the STR variant is omitted.
_SYSTEM_PROMPT = (
    "You are an embodied robot that navigates in the real world."
    " You need to explore between some places marked with IDs and ultimately"
    " find the destination to stop. At each step, a series of images"
    " corresponding to the places you have explored and have observed will be"
    " provided to you.\n"
    "'Instruction' is a global, step-by-step detailed guidance, but you might"
    " have already executed some of the commands. You need to carefully"
    " discern the commands that have not been executed yet.\n"
    "'History' represents the places you have explored in previous steps"
    " along with their corresponding images. It may include the correct"
    " landmarks mentioned in the 'Instruction' as well as some past erroneous"
    " explorations.\n"
    "'Trajectory' represents the ID info of the places you have explored."
    " You start navigating from Place 0.\n"
    "'Map' refers to the connectivity between the places you have explored"
    " and other places you have observed.\n"
    "'Supplementary Info' records some places and their corresponding images"
    " you have ever seen but have not yet visited. These places are only"
    " considered when there is a navigation error, and you decide to"
    " backtrack for further exploration.\n"
    "'Previous Planning' records previous long-term multi-step planning info"
    " that you can refer to now.\n"
    "'Action options' are some actions that you can take at this step.\n"
    "For each provided image of the places, you should combine the"
    " 'Instruction' and carefully examine the relevant information, such as"
    " scene descriptions, landmarks, and objects. You need to align"
    " 'Instruction' with 'History' (including corresponding images) to"
    " estimate your instruction execution progress and refer to 'Map' for"
    " path planning. Check the Place IDs in the 'History' and 'Trajectory',"
    " avoiding repeated exploration that leads to getting stuck in a loop,"
    " unless it is necessary to backtrack to a specific place.\n"
    "If you can already see the destination, estimate the distance between"
    " you and it. If the distance is far, continue moving and try to stop"
    " within 1 meter of the destination.\n"
    "Your answer should be JSON format and must include three fields:"
    " 'Thought', 'New Planning', and 'Action'. You need to combine"
    " 'Instruction', 'Trajectory', 'Map', 'Supplementary Info', your past"
    " 'History', 'Previous Planning', 'Action options', and the provided"
    " images to think about what to do next and why, and complete your"
    " thinking into 'Thought'.\n"
    "Based on your 'Map', 'Previous Planning' and current 'Thought', you also"
    " need to update your new multi-step path planning to 'New Planning'.\n"
    "At the end of your output, you must provide a single capital letter in"
    " the 'Action options' that corresponds to the action you have decided to"
    " take, and place only the letter into 'Action', such as \"Action: A\"."
)

# one_stage_prompt_manager.py:189 / :238
_INIT_HISTORY = "The navigation has just begun, with no history."

# one_stage_prompt_manager.py:14
_DEFAULT_PLANNING = "Navigation has just started, with no planning yet."

# one_stage_prompt_manager.py:163
_NO_SUPP = "Nothing yet."

# STOP sentinel shared between build_prompt and parse_action via cand_vpids_json.
_STOP_SENTINEL = "STOP"

# Initial empty topological map — four parallel lists mirroring the source
# ``self.nodes_list`` / ``self.graph`` / ``self.trajectory`` / ``self.node_imgs``.
_TOPO_MAP_INITIAL: dict[str, Any] = {
    "nodes_list": [],
    "graph": {},
    "trajectory": [],
    "node_imgs": [],
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers — verbatim ports of the one_stage_prompt_manager.py methods.
# ═══════════════════════════════════════════════════════════════════════


def _get_action_concept(rel_heading: float, rel_elevation: float) -> str:
    """Heading delta → English direction phrase.

    Verbatim port of ``OneStagePromptManager.get_action_concept``
    (one_stage_prompt_manager.py:16-39). Inputs are in radians.
    """
    if rel_elevation > 0:
        return "go up"
    if rel_elevation < 0:
        return "go down"
    if rel_heading < 0:
        if rel_heading >= -math.pi / 2:
            return "turn left"
        if rel_heading < -math.pi / 2 and rel_heading > -math.pi * 3 / 2:
            return "turn around"
        return "turn right"
    if rel_heading > 0:
        if rel_heading <= math.pi / 2:
            return "turn right"
        if rel_heading > math.pi / 2 and rel_heading < math.pi * 3 / 2:
            return "turn around"
        return "turn left"
    return "go forward"  # rel_heading == 0


def _make_map_text(
    nodes_list: list[str],
    graph: dict[str, list[str]],
    trajectory: list[str],
) -> tuple[str, str, str]:
    """Serialize the topo-map to ``(trajectory_text, graph_text, supp_text)``.

    Verbatim port of ``OneStagePromptManager.make_map_prompt``
    (one_stage_prompt_manager.py:124-165).
    """
    no_dup_nodes: list[str] = []
    trajectory_text = "Place"
    graph_text = ""

    candidate_nodes = graph.get(trajectory[-1], []) if trajectory else []

    for node in trajectory:
        node_index = nodes_list.index(node)
        trajectory_text += f" {node_index}"

        if node not in no_dup_nodes:
            no_dup_nodes.append(node)

            adj_text = ""
            adjacent_nodes = graph.get(node, [])
            for adj_node in adjacent_nodes:
                adj_index = nodes_list.index(adj_node)
                adj_text += f" {adj_index},"

            graph_text += f"\nPlace {node_index} is connected with Places{adj_text}"[:-1]

    graph_supp_text = ""
    supp_exist = None
    for node_index, node in enumerate(nodes_list):
        if node in trajectory or node in candidate_nodes:
            continue
        supp_exist = True
        graph_supp_text += f"\nPlace {node_index}, which is corresponding to Image {node_index}"

    if supp_exist is None:
        graph_supp_text = _NO_SUPP

    return trajectory_text, graph_text, graph_supp_text


def _nearest_view(
    view_meta: list[dict],
    target_heading_rad: float,
    target_elevation_rad: float,
) -> int | None:
    """Index into ``view_meta`` of the view nearest a candidate's direction.

    env_mp3d emits the per-view primitive (``views`` + ``view_meta``); the
    candidate's full image is simply ``views[idx]`` at the returned index —
    no cropping. Nearest is by combined heading + elevation distance, which
    with 12 headings x 3 elevations lands on the correct elevation row
    (mirrors the upstream ``pointId``-keyed per-view image).
    """
    if not view_meta:
        return None

    target_h = math.degrees(target_heading_rad) % 360.0
    target_e = math.degrees(target_elevation_rad)

    def _wrap_dist(a: float, b: float) -> float:
        d = abs((a - b) % 360.0)
        return min(d, 360.0 - d)

    best_idx, _ = min(
        enumerate(view_meta),
        key=lambda kv: (
            _wrap_dist(float(kv[1].get("heading_deg", 0.0)), target_h)
            + abs(float(kv[1].get("elevation_deg", 0.0)) - target_e)
        ),
    )
    return best_idx


# ═══════════════════════════════════════════════════════════════════════
# Node — System Prompt (str or json mode)
# ═══════════════════════════════════════════════════════════════════════


class MapGPTSystemPromptNode(BaseCanvasNode):
    """Emit the verbatim MapGPT JSON-mode system prompt (gpt-4o)."""

    node_type: ClassVar[str] = "mapgpt__system_prompt"
    display_name: ClassVar[str] = "MapGPT: System Prompt"
    description: ClassVar[str] = "Verbatim MapGPT JSON-mode system prompt (gpt-4o)"
    category: ClassVar[str] = "skill"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "AlignLeft"
    input_ports: ClassVar[list] = []
    output_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "System prompt text"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        self._self_log("length", len(_SYSTEM_PROMPT))
        return {"text": _SYSTEM_PROMPT}


# ═══════════════════════════════════════════════════════════════════════
# Node — Observe  (env obs → per-candidate {direction, tile} records)
# ═══════════════════════════════════════════════════════════════════════


class MapGPTObserveNode(BaseCanvasNode):
    """Compute per-candidate direction phrases and pick each candidate's view.

    Pure transform of the env observation — no state I/O. The direction
    word ``"turn left" / "go forward" / …`` is computed from
    ``cand_heading - cur_heading`` and frozen into ``candidates_json``,
    so downstream nodes do not need access to the current heading. The
    candidate's image is the env per-view ``views[view_index]`` — picked by
    nearest (heading, elevation) against ``view_meta``, no cropping. When
    the env already attached ``view_index`` to the navigable record it is
    used directly.
    """

    node_type: ClassVar[str] = "mapgpt__observe"
    display_name: ClassVar[str] = "MapGPT: Observe"
    description: ClassVar[str] = (
        "Env obs → per-candidate {vp, direction, heading, elev, view_index} + aligned views"
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Eye"
    input_ports: ClassVar[list] = [
        PortDef("viewpoint_id", "TEXT", "Current viewpoint ID"),
        PortDef("heading", "TEXT", "Current heading in degrees"),
        PortDef(
            "navigable_json", "TEXT", "Navigable viewpoints JSON (radian headings + view_index)"
        ),
        PortDef("views", "LIST[IMAGE]", "Per-view panorama images from env"),
        PortDef("view_meta", "TEXT", "Per-view metadata JSON aligned 1:1 with views"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("current_vp", "TEXT", "Current viewpoint ID (forwarded)"),
        PortDef(
            "candidates_json",
            "TEXT",
            'JSON [{"vp","direction","heading","elev","view_index"}, …]',
        ),
        PortDef(
            "candidate_tiles",
            "LIST[IMAGE]",
            "Per-candidate view images aligned 1:1 with candidates_json",
        ),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        vp = str(inputs.get("viewpoint_id", "")).strip()
        navigable_json = str(inputs.get("navigable_json", "{}"))
        raw_views = inputs.get("views")
        views: list[Any] = list(raw_views) if isinstance(raw_views, list) else []
        try:
            view_meta = json.loads(str(inputs.get("view_meta", "[]")))
            if not isinstance(view_meta, list):
                view_meta = []
        except (json.JSONDecodeError, TypeError):
            view_meta = []

        try:
            cur_heading_rad = math.radians(float(str(inputs.get("heading", "0"))))
        except (TypeError, ValueError):
            cur_heading_rad = 0.0
        try:
            navigable: dict[str, dict[str, float]] = json.loads(navigable_json)
        except (json.JSONDecodeError, TypeError):
            navigable = {}

        candidates: list[dict[str, Any]] = []
        tiles: list[Any] = []
        for cand_vp, nav in navigable.items():
            if not isinstance(nav, dict):
                continue
            cand_abs_heading = float(nav.get("heading", 0.0))
            cand_abs_elev = float(nav.get("elevation", 0.0))
            direction = _get_action_concept(
                cand_abs_heading - cur_heading_rad,
                cand_abs_elev,
            )
            # Prefer the env-attached view_index; else resolve nearest view.
            vi = nav.get("view_index")
            if vi is None:
                vi = _nearest_view(view_meta, cand_abs_heading, cand_abs_elev)
            tile = views[vi] if isinstance(vi, int) and 0 <= vi < len(views) else None
            candidates.append(
                {
                    "vp": cand_vp,
                    "direction": direction,
                    "heading": cand_abs_heading,
                    "elev": cand_abs_elev,
                    "view_index": vi if vi is not None else -1,
                }
            )
            tiles.append(tile)

        self._self_log("current_vp", vp)
        self._self_log("n_candidates", len(candidates))
        self._self_log("directions", [c["direction"] for c in candidates])
        self._self_log("view_indices", [c["view_index"] for c in candidates])
        return {
            "current_vp": vp,
            "candidates_json": json.dumps(candidates),
            "candidate_tiles": tiles,
        }


# ═══════════════════════════════════════════════════════════════════════
# Node — Update Map  (single owner of topo_map graph_state)
# ═══════════════════════════════════════════════════════════════════════


class MapGPTUpdateMapNode(BaseCanvasNode):
    """Fold current obs into ``topo_map`` state and emit ``topo_snapshot``.

    Sole writer of ``topo_map``. Appends current_vp to ``trajectory``
    (and to ``nodes_list`` if new), folds candidate vps into
    ``nodes_list`` / ``node_imgs`` (refreshing the tile slot when seen
    again), and records adjacency on first visit. The merged dict is
    written back to ``graph_state["topo_map"]`` AND emitted as
    ``topo_snapshot`` for downstream nodes in the same iter.
    """

    node_type: ClassVar[str] = "mapgpt__update_map"
    display_name: ClassVar[str] = "MapGPT: Update Map"
    description: ClassVar[str] = (
        "Fold obs into topo_map state (first-visit adjacency + tile cache); emit snapshot"
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "Network"
    input_ports: ClassVar[list] = [
        PortDef("current_vp", "TEXT", "Current viewpoint ID"),
        PortDef("candidates_json", "TEXT", "Candidate records JSON from observe"),
        PortDef("candidate_tiles", "LIST[IMAGE]", "Per-candidate panorama tiles"),
    ]
    output_ports: ClassVar[list] = [
        PortDef(
            "topo_snapshot",
            "ANY",
            "{nodes_list, graph, trajectory, node_imgs} after this step's update",
        ),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        vp = str(inputs.get("current_vp", "")).strip()
        try:
            candidates = json.loads(str(inputs.get("candidates_json", "[]")))
            if not isinstance(candidates, list):
                candidates = []
        except (json.JSONDecodeError, TypeError):
            candidates = []
        raw_tiles = inputs.get("candidate_tiles")
        tiles: list[Any] = list(raw_tiles) if isinstance(raw_tiles, list) else []

        topo: dict[str, Any] = copy.deepcopy(_TOPO_MAP_INITIAL)
        gs = getattr(ctx, "graph_state", None) if ctx else None
        if gs:
            with suppress(Exception):
                prior_topo = gs.read("topo_map")
                if isinstance(prior_topo, dict) and prior_topo.get("nodes_list") is not None:
                    topo = copy.deepcopy(prior_topo)

        nodes_list: list[str] = topo["nodes_list"]
        graph: dict[str, list[str]] = topo["graph"]
        trajectory: list[str] = topo["trajectory"]
        node_imgs: list[Any] = topo["node_imgs"]

        # Fold current viewpoint (reference lines 54-60).
        if vp and vp not in nodes_list:
            nodes_list.append(vp)
            node_imgs.append(None)
        if vp:
            trajectory.append(vp)

        # Fold candidates: append new vps, refresh tile slot when seen again.
        cand_vpids: list[str] = []
        for i, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue
            cand_vp = str(cand.get("vp", ""))
            if not cand_vp:
                continue
            cand_vpids.append(cand_vp)
            tile = tiles[i] if i < len(tiles) else None
            if cand_vp not in nodes_list:
                nodes_list.append(cand_vp)
                node_imgs.append(tile)
            else:
                idx = nodes_list.index(cand_vp)
                if tile is not None:
                    node_imgs[idx] = tile

        # First-visit adjacency (reference lines 85-87).
        first_visit = vp and vp not in graph
        if first_visit:
            graph[vp] = cand_vpids

        new_topo = {
            "nodes_list": nodes_list,
            "graph": graph,
            "trajectory": trajectory,
            "node_imgs": node_imgs,
        }
        if gs:
            with suppress(Exception):
                gs.write("topo_map", new_topo)

        self._self_log("current_vp", vp)
        self._self_log("first_visit", bool(first_visit))
        self._self_log("nodes_list_size", len(nodes_list))
        self._self_log("trajectory_len", len(trajectory))
        self._self_log("n_candidates", len(cand_vpids))
        return {"topo_snapshot": new_topo}


# ═══════════════════════════════════════════════════════════════════════
# Node — Build Options  (letter prefix + stop gating + manifest)
# ═══════════════════════════════════════════════════════════════════════


class MapGPTBuildOptionsNode(BaseCanvasNode):
    """Format the LLM action options + manifest from ``topo_snapshot``.

    Pulls Place IDs from ``topo_snapshot.nodes_list`` and joins them
    with direction phrases from ``candidates_json`` to produce the
    reference's ``"<direction> to Place i which is corresponding to
    Image i"`` per-candidate strings, then letter-prefixes the list.
    Stop gating: when ``ctx.step >= stop_after``, prepends
    ``{letter:"A", vp:"STOP", phrase:"stop"}``.
    """

    node_type: ClassVar[str] = "mapgpt__build_options"
    display_name: ClassVar[str] = "MapGPT: Build Options"
    description: ClassVar[str] = (
        "Letter-prefix candidates + stop gating → options_text + options_json manifest"
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "List"
    input_ports: ClassVar[list] = [
        PortDef("topo_snapshot", "ANY", "Topo snapshot from update_map"),
        PortDef("candidates_json", "TEXT", "Candidate records JSON from observe"),
    ]
    output_ports: ClassVar[list] = [
        PortDef(
            "options_text",
            "TEXT",
            "Action options bracketed list, e.g. \"['A. stop', 'B. turn left to Place 3 …']\"",
        ),
        PortDef(
            "options_json",
            "TEXT",
            'JSON [{"letter":"A","vp":"<id>|STOP","phrase":"<action_phrase>"}, …]',
        ),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField(
                "stop_after",
                "slider",
                label="Stop-after step",
                default=3,
                min=0,
                max=15,
                step=1,
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        topo = inputs.get("topo_snapshot") or {}
        if not isinstance(topo, dict):
            topo = {}
        nodes_list: list[str] = list(topo.get("nodes_list", []) or [])
        try:
            candidates = json.loads(str(inputs.get("candidates_json", "[]")))
            if not isinstance(candidates, list):
                candidates = []
        except (json.JSONDecodeError, TypeError):
            candidates = []

        step_t = int(getattr(ctx, "step", 0)) if ctx else 0
        stop_after = int(self.config.get("stop_after", 3))

        cand_phrases: list[str] = []
        vp_aligned: list[str] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cand_vp = str(cand.get("vp", ""))
            direction = str(cand.get("direction", ""))
            if cand_vp not in nodes_list:
                # update_map is the sole arbiter of nodes_list — a missing index
                # means the candidate was rejected upstream; drop it silently.
                continue
            idx = nodes_list.index(cand_vp)
            cand_phrases.append(f"{direction} to Place {idx} which is corresponding to Image {idx}")
            vp_aligned.append(cand_vp)

        action_prompts: list[str] = list(cand_phrases)
        vps: list[str] = list(vp_aligned)
        phrases: list[str] = list(cand_phrases)
        if stop_after and step_t >= stop_after:
            action_prompts = ["stop", *action_prompts]
            vps = [_STOP_SENTINEL, *vps]
            phrases = ["stop", *phrases]

        full_options = [f"{chr(j + 65)}. {action_prompts[j]}" for j in range(len(action_prompts))]
        options_text = "[" + ", ".join(f"'{o}'" for o in full_options) + "]"

        records = [
            {"letter": chr(j + 65), "vp": vps[j], "phrase": phrases[j]}
            for j in range(len(action_prompts))
        ]

        self._self_log("step", step_t)
        self._self_log("n_options", len(records))
        self._self_log("stop_available", bool(vps and vps[0] == _STOP_SENTINEL))
        self._self_log("options_preview", full_options[:6])
        return {
            "options_text": options_text,
            "options_json": json.dumps(records),
        }


# ═══════════════════════════════════════════════════════════════════════
# Node — Render Prompt  (7-field user prompt + image pack)
# ═══════════════════════════════════════════════════════════════════════


class MapGPTRenderPromptNode(BaseCanvasNode):
    """Render the 7-field MapGPT user prompt + per-Place image list.

    Reads ``history`` / ``planning`` from state, formats trajectory /
    map / supplementary text via ``_make_map_text``, and packs every
    cached tile from ``topo_snapshot.node_imgs`` into ``image_list``
    with ``"Image i:"`` labels aligned to the Place ID (= index in
    ``nodes_list``) — verbatim ``GPT/api.py:22-44``.
    """

    node_type: ClassVar[str] = "mapgpt__render_prompt"
    display_name: ClassVar[str] = "MapGPT: Render Prompt"
    description: ClassVar[str] = (
        "Render 7-field user prompt + image pack from topo_snapshot + history/planning state"
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "AlignLeft"
    input_ports: ClassVar[list] = [
        PortDef("instruction", "TEXT", "Navigation instruction"),
        PortDef("topo_snapshot", "ANY", "Topo snapshot from update_map"),
        PortDef("options_text", "TEXT", "Action options text from build_options"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("prompt", "TEXT", "Assembled 7-field user prompt"),
        PortDef("image_list", "LIST[IMAGE]", "Per-Place images for the LLM"),
        PortDef(
            "image_labels",
            "LIST[TEXT]",
            "Place-ID labels aligned 1:1 with image_list (e.g. 'Image 2:')",
        ),
        PortDef("image_count", "TEXT", "Length of image_list (string int)"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        instruction = str(inputs.get("instruction", ""))
        options_text = str(inputs.get("options_text", "[]"))
        topo = inputs.get("topo_snapshot") or {}
        if not isinstance(topo, dict):
            topo = {}
        nodes_list: list[str] = list(topo.get("nodes_list", []) or [])
        graph: dict[str, list[str]] = dict(topo.get("graph", {}) or {})
        trajectory: list[str] = list(topo.get("trajectory", []) or [])
        node_imgs: list[Any] = list(topo.get("node_imgs", []) or [])

        step_t = int(getattr(ctx, "step", 0)) if ctx else 0

        history = _INIT_HISTORY
        planning = _DEFAULT_PLANNING
        gs = getattr(ctx, "graph_state", None) if ctx else None
        if gs:
            if step_t > 0:
                with suppress(Exception):
                    raw_hist = gs.read("history")
                    if isinstance(raw_hist, str) and raw_hist:
                        history = raw_hist
            with suppress(Exception):
                raw_plan = gs.read("planning")
                if isinstance(raw_plan, str) and raw_plan:
                    planning = raw_plan

        trajectory_text, graph_text, supp_text = _make_map_text(nodes_list, graph, trajectory)

        image_list: list = []
        image_labels: list[str] = []
        for i, img in enumerate(node_imgs):
            if img is not None:
                image_list.append(img)
                image_labels.append(f"Image {i}:")

        prompt = (
            f"Instruction: {instruction}\n"
            f"History: {history}\n"
            f"Trajectory: {trajectory_text}\n"
            f"Map:{graph_text}\n"
            f"Supplementary Info: {supp_text}\n"
            f"Previous Planning:\n"
            f"{planning}\n"
            f"Action options (step {step_t}): {options_text}"
        )

        self._self_log("step", step_t)
        self._self_log("image_count", len(image_list))
        self._self_log("image_labels_preview", image_labels[:6])
        self._self_log("prompt_length", len(prompt))
        return {
            "prompt": prompt,
            "image_list": image_list,
            "image_labels": image_labels,
            "image_count": str(len(image_list)),
        }


# ═══════════════════════════════════════════════════════════════════════
# Node — Parse Action  (consumer of build_prompt's manifest)
# ═══════════════════════════════════════════════════════════════════════


class MapGPTParseActionNode(BaseCanvasNode):
    """Parse the JSON LLM response; resolve letter → ``{vp, phrase}`` via manifest.

    ``options_json`` is a single manifest ``[{letter, vp, phrase}, …]``.
    Lookup by letter; ``vp == "STOP"`` ⇒ ``is_stop=True``; the chosen
    record's ``phrase`` is emitted as ``action_phrase`` for
    downstream history logging. Fallback on parse failure: record at
    index 0 (which is stop when ``step_t >= stop_after``, first
    candidate otherwise — matches the reference's index-shift trick).

    Also extracts ``new_planning`` and writes it back to ``planning``
    state (carries across iterations as ``Previous Planning``).
    """

    node_type: ClassVar[str] = "mapgpt__parse_action"
    display_name: ClassVar[str] = "MapGPT: Parse Action"
    description: ClassVar[str] = (
        "Parse JSON LLM response → {viewpoint_id, action_phrase}; update planning state"
    )
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "GitBranch"
    input_ports: ClassVar[list] = [
        PortDef("response", "TEXT", "LLM response text (JSON)"),
        PortDef("options_json", "TEXT", "Manifest [{letter, vp, phrase}, …] from plan_step"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("viewpoint_id", "TEXT", "Target viewpoint ID or 'STOP'"),
        PortDef("action_phrase", "TEXT", "Raw action phrase chosen (for history)"),
        PortDef("thought", "TEXT", "Extracted Thought text"),
        PortDef("new_planning", "TEXT", "Extracted New Planning text"),
        PortDef("is_stop", "BOOL", "True when the chosen option is stop"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        response = str(inputs.get("response", "")).strip()
        options: list[dict] = []
        with suppress(json.JSONDecodeError, TypeError):
            raw = json.loads(str(inputs.get("options_json", "[]")))
            if isinstance(raw, list):
                options = [r for r in raw if isinstance(r, dict)]

        letters = [str(rec.get("letter", "")) for rec in options]

        output_index = 0
        parse_error: str | None = None
        parse_source = "fallback"
        thought = ""
        new_planning = ""
        letter = ""

        # Tier 1 — parse_json_action (one_stage_prompt_manager.py:336-352).
        # Tolerate markdown fences (```json ... ```) and JSON embedded in
        # prose by extracting the outermost {...} block before decoding.
        cleaned = response
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        elif "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < end:
                cleaned = cleaned[start : end + 1]
        try:
            data = json.loads(cleaned)
            letter = str(data.get("Action", "")).strip().rstrip(".").upper()
            thought = str(data.get("Thought", ""))
            new_planning = str(data.get("New Planning", ""))
            if letter in letters:
                parse_source = "json"
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            parse_error = f"json: {exc}"

        # Tier 2 — plain-text fallback. LLMs drift to free-form answers once
        # the JSON system prompt is no longer in recent context, so match
        # common letter-emphasis patterns on the raw response.
        if letter not in letters:
            m = (
                re.search(r"\*\*\s*([A-Z])[.\s\)\*]", response)
                or re.search(r"(?im)^\s*Action[:\s]+([A-Z])\b", response)
                or re.search(r"(?im)^\s*([A-Z])[.\)]\s", response)
            )
            if m and m.group(1) in letters:
                letter = m.group(1)
                parse_source = "regex"

        if letter in letters:
            output_index = letters.index(letter)

        # --- Resolve letter → record via the manifest -------------------
        if 0 <= output_index < len(options):
            record = options[output_index]
        elif options:
            record = options[0]
        else:
            record = {"vp": _STOP_SENTINEL, "phrase": "stop"}

        vp_id = str(record.get("vp", _STOP_SENTINEL))
        action_phrase = str(record.get("phrase", ""))
        is_stop = vp_id == _STOP_SENTINEL

        gs = getattr(ctx, "graph_state", None) if ctx else None
        if gs and new_planning:
            with suppress(Exception):
                gs.write("planning", new_planning)

        if parse_error:
            self._self_log("parse_error", parse_error)
        self._self_log("parse_source", parse_source)
        self._self_log("output_index", output_index)
        self._self_log("resolved_vp", vp_id)
        self._self_log("action_phrase", action_phrase)
        self._self_log("is_stop", is_stop)
        self._self_log("thought", thought[:200])
        self._self_log("new_planning", new_planning[:200])
        return {
            "viewpoint_id": vp_id,
            "action_phrase": action_phrase,
            "thought": thought,
            "new_planning": new_planning,
            "is_stop": is_stop,
        }


# ═══════════════════════════════════════════════════════════════════════
# Node — Update History  (state writer)
# ═══════════════════════════════════════════════════════════════════════


class MapGPTUpdateHistoryNode(BaseCanvasNode):
    """Append the executed action phrase to the ``history`` state entry.

    Verbatim port of ``OneStagePromptManager.make_history``
    (one_stage_prompt_manager.py:114-122). The reference prepends
    ``'stop'`` to ``only_actions`` and indexes by the chosen letter;
    we receive the exact same payload as ``action_phrase`` from
    ``parse_action``, so no lookup is needed here.
    """

    node_type: ClassVar[str] = "mapgpt__update_history"
    display_name: ClassVar[str] = "MapGPT: Update History"
    description: ClassVar[str] = "Append 'step t: <action_phrase>' to history state"
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "ClipboardList"
    input_ports: ClassVar[list] = [
        PortDef("action_phrase", "TEXT", "Raw action phrase from parse_action"),
        PortDef("thought", "TEXT", "Thought text for trace", optional=True),
    ]
    output_ports: ClassVar[list] = [
        PortDef("history", "TEXT", "Updated history text"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        action_phrase = str(inputs.get("action_phrase", "")).strip() or "step"
        step_t = int(getattr(ctx, "step", 0)) if ctx else 0

        prior = ""
        gs = getattr(ctx, "graph_state", None) if ctx else None
        if gs:
            with suppress(Exception):
                raw = gs.read("history")
                if isinstance(raw, str):
                    prior = raw

        if step_t == 0 or not prior:
            new_hist = f"step {step_t}: {action_phrase}"
        else:
            new_hist = prior + f", step {step_t}: {action_phrase}"

        if gs:
            with suppress(Exception):
                gs.write("history", new_hist)

        self._self_log("step", step_t)
        self._self_log("action_phrase", action_phrase)
        self._self_log("history_preview", new_hist[-200:])
        return {"history": new_hist}


# ═══════════════════════════════════════════════════════════════════════
# Node — Image Budget Guard
# ═══════════════════════════════════════════════════════════════════════


class MapGPTImageBudgetNode(BaseCanvasNode):
    """Emit ``done=True`` when the Place-image count exceeds the budget.

    Verbatim semantic of the hard-abort at ``vln/gpt_agent.py:133-136``:
    ``if len(image_list) > 20: a_t = [0]``.
    """

    node_type: ClassVar[str] = "mapgpt__image_budget"
    display_name: ClassVar[str] = "MapGPT: Image Budget"
    description: ClassVar[str] = "Emit done=True when image_count exceeds the budget"
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "AlertTriangle"
    input_ports: ClassVar[list] = [
        PortDef("image_count", "TEXT", "Current image_list length (stringified int)"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("done", "BOOL", "True when budget exceeded"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="rose",
        config_fields=[
            ConfigField(
                "max_images",
                "slider",
                label="Max images",
                default=20,
                min=1,
                max=50,
                step=1,
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            n = int(str(inputs.get("image_count", "0")))
        except (TypeError, ValueError):
            n = 0
        max_images = int(self.config.get("max_images", 20))
        exceeded = n > max_images
        self._self_log("image_count", n)
        self._self_log("max_images", max_images)
        self._self_log("exceeded", exceeded)
        if exceeded:
            return {"done": True}
        # Mirror env_mp3d__navigate_to: omit the key when not done so the
        # executor's done-scan does not terminate early.
        return {}


# ═══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ═══════════════════════════════════════════════════════════════════════


class MapgptNodeSet(BaseNodeSet):
    name = "mapgpt"
    description = (
        "MapGPT (ACL 2024) nodeset — observe (env→features) / update_map (state) / "
        "build_options (manifest) / render_prompt (7-field prompt) split, "
        "manifest-driven action parser, history writer + image-budget guard"
    )

    def get_tools(self) -> list:
        return [
            MapGPTSystemPromptNode(),
            MapGPTObserveNode(),
            MapGPTUpdateMapNode(),
            MapGPTBuildOptionsNode(),
            MapGPTRenderPromptNode(),
            MapGPTParseActionNode(),
            MapGPTUpdateHistoryNode(),
            MapGPTImageBudgetNode(),
        ]
