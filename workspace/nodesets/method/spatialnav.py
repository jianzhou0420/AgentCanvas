"""SpatialNav method nodes — SpatialNav-MP3D discrete (M2 Phase 3).

Fork of ``navgpt_mp3d_tools.py``'s four reasoning nodes with
SpatialNav-specific deltas:

- ``spatialnav__observation_format`` — 8-compass observation +
  SSG-sourced ``local_objects`` / ``interest_objects`` / ``room_type``
  enrichment, wrapped with the SpatialNav ``**Current Top-down Map**:
  [SSG map image]`` / ``**Current Viewpoint**:`` section headers.
  (The SSG map IMAGE itself flows via a separate port on ``llmCall``;
  the text placeholder keeps the trace shape comparable to NavGPT.)

- ``spatialnav__parse_action`` — intentional divergence from NavGPT:
  ``Final Answer:`` is prioritised **above** ``Action:`` (reversed from
  NavGPT's raise-on-both), matching
  ``SpatialAgentOutputParser.parse()`` at
  ``tmp/_reference/spatialnav/agent.py`` L93-128. See
  ``tmp/_reference/spatialnav/NOTICE.md`` → Known Divergences.

- ``spatialnav__scratchpad_writer`` — ``max_scratchpad_length`` defaults
  to 4096 (was 7000 in NavGPT; matches ``SpatialVLNAgent`` L160);
  appends the ``Current Top-down Map: [SSG map image]`` label so the
  text trace remains readable without the image port.

- ``spatialnav__init_observation`` — fork of NavGPT's init; additionally
  emits ``scan_id_out`` and ``path_id_out`` pass-through ports so the
  Initialize path can drive ``ssg__reset_episode`` and a first-step
  ``ssg__query_objects`` call.

All four nodes are stateless (per-call) — graph-state carries the
accumulated scratchpad between iterations as with NavGPT.

Fork parent:
    workspace/nodesets/navgpt_mp3d_tools.py

Upstream reference:
    tmp/_reference/spatialnav/agent.py  (SpatialVLNAgent,
    SpatialAgentOutputParser, SpatialSceneGraphNav)
    tmp/_reference/spatialnav/prompts.py  (VLN_SPATIAL_GPT5_PROMPT_V2)
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.spatialnav")


# ══════════════════════════════════════════════════════════════════════
# Constants / shared helpers — forked from navgpt_mp3d_tools with
# SpatialNav-specific values.
# ══════════════════════════════════════════════════════════════════════

# SpatialVLNAgent.max_scratchpad_length = 4096 (was 7000 in NavGPT).
_DEFAULT_MAX_SCRATCHPAD_LENGTH = 4096

# Literal label that stands in for the SSG map IMAGE in the TEXT trace.
# The actual map flows on the ``scene_graph: IMAGE`` port into ``llmCall``;
# this label keeps text diffs readable and matches the upstream prompt
# shape (``\n\t**Current Top-down Map**: <ImageHere>``).
_SSG_MAP_TEXT_LABEL = "[SSG map image]"


def _lr_label(h_deg: float) -> str:
    """Format heading as ``right X.XX`` / ``left X.XX``."""
    if h_deg > 0:
        return f"right {h_deg:.2f}"
    if h_deg < 0:
        return f"left {-h_deg:.2f}"
    return "right 0.00"


def _normalize_heading_deg(h_deg: float) -> float:
    h = h_deg
    while h > 180:
        h -= 360
    while h <= -180:
        h += 360
    return h


def _format_turned_angle(delta_deg: float, curr_heading_deg: float) -> str:
    curr = _normalize_heading_deg(curr_heading_deg)
    prev = _normalize_heading_deg(curr - delta_deg)
    return (
        f"Turn heading direction {abs(delta_deg):.2f} degrees "
        f"from {_lr_label(prev)} to {_lr_label(curr)}."
    )


# ══════════════════════════════════════════════════════════════════════
# SpatialNav parse — Final Answer prioritised over Action
# (intentional divergence from NavGPT, see NOTICE.md).
# ══════════════════════════════════════════════════════════════════════


def _parse_spatialnav_action(raw: str) -> tuple[str, bool]:
    """Extract viewpoint ID / STOP from SpatialNav-formatted LLM output.

    Priority order (mirrors ``SpatialAgentOutputParser.parse``):
        1. ``Final Answer:`` — if present, treat the step as terminal
           regardless of any ``Action:`` / ``Action Input:`` that may
           also appear. Return ``(answer, is_stop=True)``.
        2. ``Action: ... Action Input: "<32-hex>"`` — emit the viewpoint
           ID; ``is_stop=False``.
        3. Fallback 32-char hex anywhere in string; ``is_stop=False``.
        4. Stop keyword fallback (``STOP`` / ``FINISHED``).

    Differs from NavGPT's parser (``navgpt_mp3d_tools__parse_action``)
    where ``Action Input`` wins on ties. This is the key
    SpatialNav-vs-NavGPT reasoning difference documented in
    ``tmp/_reference/spatialnav/NOTICE.md``.
    """
    if not raw:
        return "", False

    # 1. Final Answer priority — SpatialNav's defining divergence.
    fa = re.search(r"Final Answer\s*:\s*(.*)", raw, re.DOTALL)
    if fa:
        answer = fa.group(1).strip().strip('"').strip("'")
        # Terminal step regardless of content; map recognised stop keywords
        # to the canonical "STOP" token the env expects.
        upper = answer.upper()
        if upper in ("STOP", "FINISHED", "FINISHED!") or "STOP" in upper:
            return "STOP", True
        # If it's a 32-char hex, still terminal (agent is saying "I'm done,
        # stopping at this viewpoint"). Downstream navigate_to treats it as
        # a terminal navigation target.
        hex_match = re.match(r"([a-fA-F0-9]{32})", answer)
        if hex_match:
            return hex_match.group(1), True
        # Free-form final answer — treat as stop.
        return "STOP", True

    # 2. Action / Action Input format (ReAct tool-calling).
    m = re.search(
        r'Action\s*\d*\s*:[\s]*(.*?)\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*"?([a-fA-F0-9]{32})"?',
        raw,
        re.DOTALL,
    )
    if m:
        return m.group(2).strip(), False

    # 3. Fallback — any 32-char hex.
    hex_m = re.search(r"[a-fA-F0-9]{32}", raw)
    if hex_m:
        return hex_m.group(0), False

    # 4. Stop keyword fallback.
    upper = raw.upper()
    if "FINISHED" in upper or "STOP" in upper:
        return "STOP", True

    return raw, False


# ══════════════════════════════════════════════════════════════════════
# 8-compass observation formatter (forked from navgpt with SSG fields)
# ══════════════════════════════════════════════════════════════════════


def _merge_rcnn_to_sectors(raw_objects: list, n_headings: int = 8) -> list:
    """Convert RCNN per-view detections to per-sector dict format.

    Copy of ``navgpt_mp3d_tools._merge_rcnn_to_sectors`` — duplicated
    (not imported) to keep spatialnav nodeset independently loadable.
    """
    n_views = len(raw_objects)
    if n_views == 0:
        return [{} for _ in range(n_headings)]
    n_elevs = n_views // n_headings if n_views >= n_headings else 1

    sectors: list = []
    for h in range(n_headings):
        sector_objects: dict = {}
        sector_center_deg = h * (360.0 / n_headings)
        for e in range(n_elevs):
            view_idx = h + e * n_headings
            if view_idx >= n_views:
                continue
            for obj in raw_objects[view_idx]:
                name = obj.get("name", "unknown")
                abs_heading_rad = math.radians(
                    sector_center_deg + obj.get("rel_heading_deg", 0),
                )
                distance = obj.get("estimated_distance_m", 3.0)
                confidence = obj.get("confidence", 0)
                if name not in sector_objects or confidence > sector_objects[name].get("_conf", 0):
                    sector_objects[name] = {
                        "heading": abs_heading_rad,
                        "distance": distance,
                        "_conf": confidence,
                    }
        for v in sector_objects.values():
            v.pop("_conf", None)
        sectors.append(sector_objects)
    return sectors


def _format_observation_compass(
    navigable: dict,
    current_heading_rad: float,
    scene_descriptions: list | None = None,
    objects_per_sector: list | None = None,
) -> str:
    """NavGPT-style 8-compass observation formatter. Byte-identical to
    ``navgpt_mp3d_tools._format_observation_compass`` — duplicated here
    to keep this nodeset independently loadable.
    """
    heading_deg = math.degrees(current_heading_rad)

    def _normalize(angle: float) -> float:
        while angle > 180:
            angle -= 360
        while angle <= -180:
            angle += 360
        return angle

    def _lr(angle: float) -> str:
        return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

    directions = [
        "Front",
        "Front Right",
        "Right",
        "Rear Right",
        "Rear",
        "Rear Left",
        "Left",
        "Front Left",
    ]
    range_idx = int((heading_deg - 22.5) // 45) + 1
    obs_idx = [(i + range_idx) % 8 for i in range(8)]

    candidate_range: dict[int, dict] = {}
    for vp_id, vp_data in navigable.items():
        vp_heading_deg = math.degrees(vp_data["heading"])
        vp_range_idx = int((vp_heading_deg - 22.5) // 45) + 1
        rel_heading = _normalize(vp_heading_deg - heading_deg)
        vp_desc = f"{_lr(rel_heading)}, {vp_data['distance']:.2f}m"
        candidate_range.setdefault(vp_range_idx, {})[vp_id] = vp_desc

    angle_ranges = [
        (angle - 22.5 - heading_deg, angle + 22.5 - heading_deg) for angle in range(0, 360, 45)
    ]

    formatted: list[str] = []
    for direction, idx in zip(directions, obs_idx, strict=False):
        rel1 = _normalize(angle_ranges[idx][0])
        rel2 = _normalize(angle_ranges[idx][1])
        s = f"{direction}, range ({_lr(rel1)} to {_lr(rel2)}): "
        if scene_descriptions and idx < len(scene_descriptions) and scene_descriptions[idx]:
            s += f"\n'{scene_descriptions[idx]}'"
        else:
            s += "\n(no scene description available)"
        if objects_per_sector and idx < len(objects_per_sector) and objects_per_sector[idx]:
            obj_dict = {}
            for obj_name, obj_data in objects_per_sector[idx].items():
                obj_heading = obj_data.get("heading", 0)
                if isinstance(obj_heading, (int, float)):
                    rel_obj = _normalize(math.degrees(obj_heading) - heading_deg)
                else:
                    rel_obj = 0.0
                obj_dist = obj_data.get("distance", 0)
                obj_dict[obj_name] = f"{_lr(rel_obj)}, {obj_dist:.2f}m"
            s += f"\n{direction} Objects in 3m: {obj_dict}"
        else:
            s += f"\n{direction} Objects in 3m: None"
        if candidate_range.get(idx):
            s += f"\n{direction} Navigable Viewpoints:{candidate_range[idx]}"
        else:
            s += f"\n{direction} Navigable Viewpoints: None"
        formatted.append(s)

    return "\n".join(formatted)


# ══════════════════════════════════════════════════════════════════════
# Node: spatialnav__observation_format
# ══════════════════════════════════════════════════════════════════════


class SpatialNavObservationFormatNode(BaseCanvasNode):
    """Format MP3D observation as the SpatialNav prompt body.

    Wraps the NavGPT 8-compass observation with the SpatialNav section
    headers (``**Current Top-down Map**:`` + ``**Current Viewpoint**:``)
    and optionally enriches the body with SSG-sourced
    ``local_objects`` / ``interest_objects`` / ``room_type`` fields
    pulled from ``ssg__query_objects``.

    The ``scene_graph`` IMAGE itself does **not** flow through this
    node — it travels on a separate port (``scene_graph: IMAGE``) on
    ``llmCall``. The literal ``[SSG map image]`` label here is a
    stable textual placeholder that keeps the trace readable and makes
    prompt-template interpolation trivial.

    Fork of ``navgpt_mp3d_tools__observation_format``.
    """

    node_type = "spatialnav__observation_format"
    display_name = "SpatialNav: Observation Format"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "use_relative_angle",
                "toggle",
                "Use relative angles only (skip explicit orientation line; matches paper V2)",
                default=True,
            ),
            ConfigField(
                "use_surround_objects",
                "toggle",
                "Enrich observation with SSG local_objects / interest_objects / room_type",
                default=True,
            ),
        ],
    )
    description = (
        "Format env observation as SpatialNav 8-compass prompt text "
        "(SSG map label + optional local/interest objects + room type)"
    )
    category = "processing"
    icon = "Eye"
    input_ports = [
        PortDef("heading", "TEXT", "Current heading in degrees"),
        PortDef("navigable_json", "TEXT", "Navigable viewpoints as JSON"),
        PortDef(
            "scene_descriptions_json",
            "TEXT",
            "Scene descriptions as JSON list (optional)",
            optional=True,
        ),
        PortDef(
            "scene_objects_json",
            "TEXT",
            "Scene objects as JSON (raw or per-sector) (optional)",
            optional=True,
        ),
        PortDef(
            "elevation",
            "TEXT",
            "Current elevation in degrees (optional; only used if use_relative_angle=false)",
            optional=True,
        ),
        PortDef(
            "local_objects_json",
            "TEXT",
            "SSG local objects within visibility radius (from ssg__query_objects)",
            optional=True,
        ),
        PortDef(
            "interest_objects_json",
            "TEXT",
            "SSG instruction-aligned interest objects (from ssg__query_objects)",
            optional=True,
        ),
        PortDef(
            "room_type",
            "TEXT",
            "Current room type from SSG (e.g. 'bedroom')",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "observation",
            "TEXT",
            "SpatialNav observation body (map-label + 8-compass viewpoint + SSG enrichment)",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        config = getattr(self, "config", None) or {}
        use_relative_angle = bool(config.get("use_relative_angle", True))
        use_surround_objects = bool(config.get("use_surround_objects", True))

        heading_raw = str(inputs.get("heading", "0")).strip()
        elevation_raw = str(inputs.get("elevation", "0") or "0").strip()
        try:
            heading_deg = float(heading_raw)
        except (ValueError, TypeError):
            heading_deg = 0.0
        try:
            elevation_deg = float(elevation_raw)
        except (ValueError, TypeError):
            elevation_deg = 0.0
        current_heading_rad = math.radians(heading_deg)

        try:
            nav = json.loads(inputs.get("navigable_json") or "{}")
        except (ValueError, TypeError) as exc:
            self._self_log("navigable_parse_error", str(exc))
            nav = {}

        scene_descs: list | None = None
        raw_descs = inputs.get("scene_descriptions_json")
        if raw_descs:
            try:
                parsed = json.loads(raw_descs)
                if isinstance(parsed, list) and parsed:
                    scene_descs = parsed
            except (ValueError, TypeError) as exc:
                self._self_log("scene_descs_parse_error", str(exc))

        objects: list | None = None
        raw_objects = inputs.get("scene_objects_json")
        if raw_objects:
            try:
                parsed = json.loads(raw_objects)
                if isinstance(parsed, list) and parsed:
                    objects = parsed
            except (ValueError, TypeError) as exc:
                self._self_log("scene_objects_parse_error", str(exc))

        if objects is not None and len(objects) > 0 and isinstance(objects[0], list):
            objects = _merge_rcnn_to_sectors(objects)

        env_feature = _format_observation_compass(
            nav,
            current_heading_rad,
            scene_descs,
            objects,
        )

        # SSG enrichment — local objects / interest objects / room type.
        enrichment_lines: list[str] = []
        if use_surround_objects:
            local_raw = inputs.get("local_objects_json") or "[]"
            interest_raw = inputs.get("interest_objects_json") or "[]"
            room_type = (inputs.get("room_type") or "").strip()
            try:
                local_objects = json.loads(local_raw) if local_raw else []
            except (ValueError, TypeError):
                local_objects = []
            try:
                interest_objects = json.loads(interest_raw) if interest_raw else []
            except (ValueError, TypeError):
                interest_objects = []

            if room_type:
                enrichment_lines.append(f"\t**Current Room Type**: {room_type}")
            if local_objects:
                names = [o.get("category", "unknown") for o in local_objects if isinstance(o, dict)]
                if names:
                    enrichment_lines.append(
                        f"\t**Nearby Objects (SSG, within radius)**: {', '.join(names)}"
                    )
            if interest_objects:
                enrichment_lines.append(
                    f"\t**Instruction-relevant Objects**: {', '.join(interest_objects)}"
                )

        # Compose with SpatialNav section headers. The literal
        # "[SSG map image]" label is a placeholder — the real image
        # flows via the ``scene_graph`` IMAGE port on llmCall.
        if use_relative_angle:
            observation = (
                f"\n\t**Current Top-down Map**: {_SSG_MAP_TEXT_LABEL}"
                f"\n\t**Current Viewpoint**:\n{env_feature}"
            )
        else:
            orientation = f"heading: {heading_deg:.2f}, elevation: {elevation_deg:.2f}"
            observation = (
                f"\n\t**Current Top-down Map**: {_SSG_MAP_TEXT_LABEL}"
                f"\n\t**Current Orientation**:\n{orientation}"
                f"\n\t**Current Viewpoint**:\n{env_feature}"
            )

        if enrichment_lines:
            observation += "\n" + "\n".join(enrichment_lines)

        self._self_log("observation_length", len(observation))
        self._self_log("observation_preview", observation[:400])
        self._self_log("navigable_count", len(nav))
        self._self_log(
            "enrichment_lines", [ln.split(":", 1)[0].strip("\t* ") for ln in enrichment_lines]
        )

        return {"observation": observation}


# ══════════════════════════════════════════════════════════════════════
# Node: spatialnav__parse_action
# ══════════════════════════════════════════════════════════════════════


class SpatialNavParseActionNode(BaseCanvasNode):
    """Parse viewpoint ID / STOP from LLM output — SpatialNav rules.

    **Key divergence from NavGPT**: ``Final Answer:`` wins over
    ``Action:`` / ``Action Input:``. This matches
    ``SpatialAgentOutputParser.parse`` at
    ``tmp/_reference/spatialnav/agent.py`` L93-128 where the upstream
    code explicitly documents the reversal with ``# if final answer
    exists, prioritize it``.

    Fork of ``navgpt_mp3d_tools__parse_action``.
    """

    node_type = "spatialnav__parse_action"
    display_name = "SpatialNav: Parse Action"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="indigo")
    description = "Extract viewpoint ID or STOP from LLM response (Final Answer > Action)"
    category = "processing"
    icon = "GitBranch"
    input_ports = [
        PortDef("llm_response", "TEXT", "Raw orchestrator LLM output"),
    ]
    output_ports = [
        PortDef("viewpoint_id", "TEXT", "Extracted viewpoint ID (or STOP)"),
        PortDef("is_stop", "BOOL", "True when agent signals Final Answer / STOP / Finished"),
        PortDef("thought", "TEXT", "Extracted thought text before Final Answer/Action"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        llm_response = str(inputs.get("llm_response", "")).strip()
        if not llm_response:
            return {"viewpoint_id": "", "is_stop": False, "thought": ""}

        vp_id, is_stop = _parse_spatialnav_action(llm_response)

        # Extract thought: everything after "Thought:" up to
        # "Action:" / "Final Answer:" / EOF.
        thought = llm_response
        thought_match = re.search(
            r"(?:Thought)\s*:\s*(.+?)(?:\n(?:Action|Final\s+Answer)\s*:|$)",
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        self._self_log("parsed_action", vp_id)
        self._self_log("is_stop", is_stop)
        self._self_log("thought", thought[:200] if thought else "")

        return {"viewpoint_id": vp_id, "is_stop": is_stop, "thought": thought}


# ══════════════════════════════════════════════════════════════════════
# Node: spatialnav__init_observation
# ══════════════════════════════════════════════════════════════════════


class SpatialNavInitObservationNode(BaseCanvasNode):
    """Seed init_observation / history / scratchpad for step 0.

    Forked from ``navgpt_mp3d_tools__init_observation`` with two
    additional pass-through outputs (``scan_id_out``, ``path_id_out``)
    so the Initialize path can drive ``ssg__reset_episode`` and a
    step-0 ``ssg__query_objects`` without needing to pull from the
    episode-info node twice.
    """

    node_type = "spatialnav__init_observation"
    display_name = "SpatialNav: Init Observation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="indigo")
    description = "Seed init_observation + initial history for SpatialNav orchestrator"
    category = "processing"
    icon = "Play"
    input_ports = [
        PortDef("observation", "TEXT", "SpatialNav-formatted observation text"),
        PortDef("viewpoint_id", "TEXT", "Starting viewpoint ID"),
        PortDef("summary", "TEXT", "1-sentence scene summary"),
        PortDef("scan_id", "TEXT", "Current scan ID (for SSG reset)", optional=True),
        PortDef(
            "path_id",
            "TEXT",
            "Current episode path_id (for SSG interest-object lookup)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "init_observation",
            "TEXT",
            "Step-0 init observation (1-sentence scene summary, paper §3.4)",
        ),
        PortDef(
            "history_0",
            "TEXT",
            "Step-0 history line swapped into {init_observation} on step >= 1",
        ),
        PortDef(
            "scratchpad_init",
            "TEXT",
            "Seed empty scratchpad '' for step-0 (routed via Initialize into iter_in.init_ports)",
        ),
        PortDef("observation_out", "TEXT", "Pass-through observation (for sequencing)"),
        PortDef("viewpoint_id_out", "TEXT", "Pass-through viewpoint ID"),
        PortDef("scan_id_out", "TEXT", "Pass-through scan ID (drives ssg__reset_episode)"),
        PortDef("path_id_out", "TEXT", "Pass-through path ID (drives ssg__query_objects)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        observation = str(inputs.get("observation", ""))
        vp_id = str(inputs.get("viewpoint_id", ""))
        summary = str(inputs.get("summary", "")) or "(scene description unavailable)"
        scan_id = str(inputs.get("scan_id", ""))
        path_id = str(inputs.get("path_id", ""))

        init_obs = f"\nThe scene from the viewpoint is a {summary}"
        init_history = (
            f"Navigation start, no actions taken yet.\n"
            f'Current viewpoint "{vp_id}": '
            f"Scene from the viewpoint is a {summary}"
        )

        self._self_log("viewpoint_id", vp_id)
        self._self_log("scan_id", scan_id)
        self._self_log("path_id", path_id)
        self._self_log("summary", summary[:200])

        return {
            "init_observation": init_obs,
            "history_0": init_history,
            "scratchpad_init": "",
            "observation_out": observation,
            "viewpoint_id_out": vp_id,
            "scan_id_out": scan_id,
            "path_id_out": path_id,
        }


# ══════════════════════════════════════════════════════════════════════
# Node: spatialnav__scratchpad_writer
# ══════════════════════════════════════════════════════════════════════


class SpatialNavScratchpadWriterNode(BaseCanvasNode):
    """Build SpatialNav ReAct scratchpad entry from LLM response +
    observation, with SSG map label embedded in the observation text.

    Forked from ``navgpt_mp3d_tools__scratchpad_writer`` with:
    - ``max_scratchpad_length`` ConfigField (default 4096, was 7000
      hardcoded in NavGPT); matches ``SpatialVLNAgent`` L160.
    - ``Current Top-down Map: [SSG map image]`` label prefix in the
      observation so the text trace remains readable (the actual map
      image flows to the LLM via a separate IMAGE port).
    """

    node_type = "spatialnav__scratchpad_writer"
    display_name = "SpatialNav: Scratchpad Writer"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "max_scratchpad_length",
                "slider",
                "Max scratchpad length (chars; SpatialVLNAgent default is 4096)",
                default=_DEFAULT_MAX_SCRATCHPAD_LENGTH,
                min=1024,
                max=16384,
                step=256,
            ),
        ],
    )
    description = "Build SpatialNav ReAct scratchpad entry (map-labelled observation, len=4096)"
    category = "processing"
    icon = "NotebookPen"
    input_ports = [
        PortDef("scratchpad_in", "TEXT", "Current scratchpad from iter_in"),
        PortDef("llm_response", "TEXT", "Raw orchestrator LLM output"),
        PortDef("observation", "TEXT", "8-compass observation from graph_observe"),
        PortDef("viewpoint_id", "TEXT", "Current viewpoint ID after navigation"),
        PortDef("summary", "TEXT", "1-sentence scene summary"),
        PortDef("success", "TEXT", "Navigation success: 'true' or 'false'"),
        PortDef("error", "TEXT", "Error message from navigate (optional)", optional=True),
        PortDef(
            "turned_angle",
            "TEXT",
            "Signed heading delta in degrees (from navigate_to)",
            optional=True,
        ),
        PortDef("heading", "TEXT", "Current heading in degrees after navigation", optional=True),
    ]
    output_ports = [
        PortDef("scratchpad_out", "TEXT", "Updated scratchpad with SSG-labelled observation"),
        PortDef("observation_out", "TEXT", "Pass-through observation"),
        PortDef("viewpoint_id_out", "TEXT", "Pass-through viewpoint ID"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        config = getattr(self, "config", None) or {}
        max_len = int(config.get("max_scratchpad_length", _DEFAULT_MAX_SCRATCHPAD_LENGTH))

        current = str(inputs.get("scratchpad_in", ""))
        llm_response = str(inputs.get("llm_response", ""))
        observation = str(inputs.get("observation", ""))
        vp_id = str(inputs.get("viewpoint_id", ""))
        summary = str(inputs.get("summary", "")) or "(scene description unavailable)"
        success = str(inputs.get("success", "true"))
        error = str(inputs.get("error", ""))
        turned_raw = str(inputs.get("turned_angle", "0")).strip()
        heading_raw = str(inputs.get("heading", "0")).strip()

        current += llm_response

        if success == "true":
            try:
                delta = float(turned_raw)
            except ValueError:
                delta = 0.0
            try:
                curr_heading = float(heading_raw)
            except ValueError:
                curr_heading = 0.0
            turned_str = _format_turned_angle(delta, curr_heading)

            # SpatialNav observation narration — map label first (the
            # image itself is sent on the scene_graph IMAGE port at
            # the next prompt build, not here).
            current += (
                f"\nObservation: \n{turned_str}"
                f"\nCurrent Top-down Map: {_SSG_MAP_TEXT_LABEL}"
                f'\nCurrent viewpoint "{vp_id}": '
                f"Scene from the viewpoint is a {summary}"
                f"\nThought:"
            )
        else:
            current += (
                f"\nObservation: {error or 'Navigation failed.'}"
                f"\nCurrent Top-down Map: {_SSG_MAP_TEXT_LABEL}"
                f"\nCurrent Viewpoint:\n{observation}"
                f"\nThought:"
            )

        if len(current) > max_len:
            # Match SpatialVLNAgent.get_full_inputs L209-L211: when
            # truncation kicks in, prefix with "... ..." elision.
            current = "... ..." + current[-(max_len - 7) :]

        self._self_log("success", success)
        self._self_log("scratchpad_length", len(current))
        self._self_log("viewpoint_id", vp_id)
        self._self_log("max_len", max_len)

        return {
            "scratchpad_out": current,
            "observation_out": observation,
            "viewpoint_id_out": vp_id,
        }


# ══════════════════════════════════════════════════════════════════════
# NodeSet wrapper
# ══════════════════════════════════════════════════════════════════════


class SpatialNavNodeSet(BaseNodeSet):
    """SpatialNav method nodes — discrete MP3D reasoning layer.

    Four reasoning nodes, forked from ``navgpt_mp3d_tools.py`` with
    SpatialNav-specific deltas (parse-priority reversal,
    max_scratchpad_length=4096, SSG map label in observation,
    scan/path pass-through for SSG wiring).

    Pairs with ``workspace/nodesets/ssg.py`` (SSG data nodes) to form
    the full SpatialNav-MP3D discrete stack. Graph topology lives in
    ``workspace/graphs/spatialnav_mp3d.json`` (M2 Phase 4).

    Load:
        POST /api/components/nodesets/spatialnav/load
    """

    name = "spatialnav"
    display_name = "SpatialNav Method"
    description = (
        "SpatialNav method nodes — observation_format (SSG-aware), "
        "parse_action (Final Answer > Action), scratchpad_writer (4096), init_observation"
    )

    def get_tools(self) -> list:
        return [
            SpatialNavObservationFormatNode(),
            SpatialNavParseActionNode(),
            SpatialNavInitObservationNode(),
            SpatialNavScratchpadWriterNode(),
        ]

    async def initialize(self) -> None:
        log.info("SpatialNav method nodeset initialised (4 reasoning nodes)")

    async def shutdown(self) -> None:
        log.info("SpatialNav method nodeset shutdown")
