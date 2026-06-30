"""BasicAgentNodeSet — general-purpose tools for LLM-based VLN agents.

Provides 13 canvas nodes across 6 categories:
  1. Scratch Pad (working memory): note_write, note_read, note_list
  2. Web Grounding: web_search, web_fetch
  3. Vision: image_analyze
  4. Spatial Math: measure_distance, compute_heading
  5. Episode Context: get_instruction, get_step_count, get_history
  6. Observation Encoding: obs_to_text, frame_sample

Load:  POST /api/components/nodesets/basic_agent/load
(nodeset name matches the ``basic_agent__*`` node_type prefix so
``ensure_nodesets_for_graph`` auto-load resolves — ADR-components-003;
renamed from ``basic_agent_tools`` 2026-06-10)

last updated: 2026-06-10
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.basic-agent")

ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}


# ══════════════════════════════════════════════════════════════════════
# Shared scratch pad state
# ══════════════════════════════════════════════════════════════════════

# Module-level dict keyed by pad_id (from node config).
# All scratch pad nodes with the same pad_id share the same dict.
# Cleared by BasicAgentNodeSet.shutdown().
_scratch_pads: dict[str, dict[str, str]] = {}


def _get_pad(config: dict) -> dict[str, str]:
    """Get or create the scratch pad for a given pad_id."""
    pad_id = config.get("pad_id", "default")
    if pad_id not in _scratch_pads:
        _scratch_pads[pad_id] = {}
    return _scratch_pads[pad_id]


# ══════════════════════════════════════════════════════════════════════
# Category 1: Scratch Pad
# ══════════════════════════════════════════════════════════════════════


class NoteWriteNode(BaseCanvasNode):
    """Write a key-value note to the scratch pad."""

    node_type = "basic_agent__note_write"
    display_name = "Scratch Pad: Write"
    description = "Save a named note to the scratch pad (shared within pad_id)"
    category = "tool"
    icon = "StickyNote"
    input_ports = [
        PortDef("key", "TEXT", "Note key / name"),
        PortDef("content", "TEXT", "Note content"),
    ]
    output_ports = [
        PortDef("ok", "BOOL", "True if written successfully"),
    ]
    ui_config = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("pad_id", "text", label="Pad ID", default="default"),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        key = str(inputs.get("key", ""))
        content = str(inputs.get("content", ""))
        if not key:
            return {"ok": False}
        pad = _get_pad(self.config)
        pad[key] = content
        return {"ok": True}


class NoteReadNode(BaseCanvasNode):
    """Read a note from the scratch pad by key."""

    node_type = "basic_agent__note_read"
    display_name = "Scratch Pad: Read"
    description = "Read a named note from the scratch pad"
    category = "tool"
    icon = "StickyNote"
    input_ports = [
        PortDef("key", "TEXT", "Note key to read"),
    ]
    output_ports = [
        PortDef("content", "TEXT", "Note content (empty string if not found)"),
    ]
    ui_config = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("pad_id", "text", label="Pad ID", default="default"),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        key = str(inputs.get("key", ""))
        pad = _get_pad(self.config)
        return {"content": pad.get(key, "")}


class NoteListNode(BaseCanvasNode):
    """List all keys in the scratch pad."""

    node_type = "basic_agent__note_list"
    display_name = "Scratch Pad: List"
    description = "List all note keys in the scratch pad"
    category = "tool"
    icon = "StickyNote"
    input_ports = [
        PortDef("trigger", "TEXT", "Optional trigger to re-fire", optional=True),
    ]
    output_ports = [
        PortDef("keys", "TEXT", "JSON list of note key names"),
    ]
    ui_config = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("pad_id", "text", label="Pad ID", default="default"),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        pad = _get_pad(self.config)
        return {"keys": json.dumps(list(pad.keys()))}


# ══════════════════════════════════════════════════════════════════════
# Category 2: Web Grounding
# ══════════════════════════════════════════════════════════════════════


class WebSearchNode(BaseCanvasNode):
    """Search the web via DuckDuckGo HTML-lite (no API key needed)."""

    node_type = "basic_agent__web_search"
    display_name = "Web: Search"
    description = "Search the web for information (DuckDuckGo)"
    category = "tool"
    icon = "Globe"
    input_ports = [
        PortDef("query", "TEXT", "Search query"),
    ]
    output_ports = [
        PortDef("results", "TEXT", "Search results as numbered text"),
    ]
    ui_config = NodeUIConfig(
        color="sky",
        config_fields=[
            ConfigField(
                "max_results", "slider", label="Max results", default=5, min=1, max=10, step=1
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import httpx

        query = str(inputs.get("query", "")).strip()
        if not query:
            return {"results": "(empty query)"}

        max_results = int(self.config.get("max_results", 5))

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                            " (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        ),
                    },
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            return {"results": f"(search failed: {str(e)[:200]})"}

        # Parse result titles and snippets from DuckDuckGo HTML-lite
        results = []
        # Match result blocks: title in <a class="result__a">, snippet in <a class="result__snippet">
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)

        for i, title in enumerate(titles[:max_results]):
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
            results.append(f"{i + 1}. {title.strip()}\n   {snippet}")

        if not results:
            return {"results": f"(no results found for: {query})"}

        return {"results": "\n\n".join(results)}


class WebFetchNode(BaseCanvasNode):
    """Fetch a URL and extract readable text content."""

    node_type = "basic_agent__web_fetch"
    display_name = "Web: Fetch"
    description = "Fetch a web page and extract text content"
    category = "tool"
    icon = "Globe"
    input_ports = [
        PortDef("url", "TEXT", "URL to fetch"),
    ]
    output_ports = [
        PortDef("content", "TEXT", "Extracted text content"),
    ]
    ui_config = NodeUIConfig(
        color="sky",
        config_fields=[
            ConfigField(
                "max_chars", "slider", label="Max chars", default=4000, min=500, max=20000, step=500
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import httpx

        url = str(inputs.get("url", "")).strip()
        if not url:
            return {"content": "(empty URL)"}

        max_chars = int(self.config.get("max_chars", 4000))

        try:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                max_redirects=3,
            ) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                            " (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        ),
                    },
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            return {"content": f"(fetch failed: {str(e)[:200]})"}

        # Strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + "... (truncated)"

        return {"content": text if text else "(empty page)"}


# ══════════════════════════════════════════════════════════════════════
# Category 3: Vision
# ══════════════════════════════════════════════════════════════════════


class ImageAnalyzeNode(BaseCanvasNode):
    """On-demand VLM image analysis."""

    node_type = "basic_agent__image_analyze"
    display_name = "Vision: Analyze"
    description = "Analyze an image with the active VLM (on-demand vision)"
    category = "tool"
    icon = "ScanEye"
    input_ports = [
        PortDef("image", "IMAGE", "Image to analyze"),
        PortDef("prompt", "TEXT", "Analysis prompt", optional=True),
    ]
    output_ports = [
        PortDef("response", "TEXT", "VLM response text"),
    ]
    ui_config = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "temperature",
                "slider",
                label="Temperature",
                default=0.3,
                min=0.0,
                max=1.0,
                step=0.1,
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        from app.llm import get_llm_config, vlm_complete
        from app.standard.wire_types import image_to_base64

        llm_config = get_llm_config()
        if not llm_config:
            return {"response": "(no LLM profile active — cannot call VLM)"}

        # Handle both numpy array and base64 string inputs
        image = inputs.get("image")
        if image is None:
            return {"response": "(no image provided)"}

        import numpy as np

        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        elif isinstance(image, str) and len(image) > 100:
            image_b64 = image
        else:
            return {"response": "(invalid image input)"}

        prompt = inputs.get("prompt") or self.config.get(
            "prompt",
            "Describe this image concisely for a navigation agent.",
        )
        temp = self.config.get("temperature", 0.3)

        try:
            response = await vlm_complete(
                llm_config,
                prompt,
                [image_b64],
                max_tokens=768,
                temperature=temp,
            )
            return {"response": response or "(VLM returned no response)"}
        except Exception as e:
            return {"response": f"(VLM error: {str(e)[:200]})"}


# ══════════════════════════════════════════════════════════════════════
# Category 4: Spatial Math
# ══════════════════════════════════════════════════════════════════════


class MeasureDistanceNode(BaseCanvasNode):
    """Euclidean distance between two agent states."""

    node_type = "basic_agent__measure_distance"
    display_name = "Spatial: Measure Distance"
    description = "Calculate Euclidean distance between two positions"
    category = "tool"
    icon = "Ruler"
    input_ports = [
        PortDef("from_pose", "POSE", "Start pose"),
        PortDef("to_pose", "POSE", "End pose"),
    ]
    output_ports = [
        PortDef("distance", "TEXT", "Distance in meters"),
    ]
    ui_config = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        from_p = inputs.get("from_pose", {})
        to_p = inputs.get("to_pose", {})
        pos_a = from_p.get("position", [0, 0, 0]) if isinstance(from_p, dict) else [0, 0, 0]
        pos_b = to_p.get("position", [0, 0, 0]) if isinstance(to_p, dict) else [0, 0, 0]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b, strict=True)))
        return {"distance": f"{dist:.4f}"}


class ComputeHeadingNode(BaseCanvasNode):
    """Compute heading from current position to target."""

    node_type = "basic_agent__compute_heading"
    display_name = "Spatial: Compute Heading"
    description = "Compute absolute and relative heading to a target position"
    category = "tool"
    icon = "Compass"
    input_ports = [
        PortDef("current", "POSE", "Current agent pose (position + orientation)"),
        PortDef("target", "POSE", "Target pose / position"),
    ]
    output_ports = [
        PortDef("heading", "TEXT", "JSON with absolute_deg and relative_deg"),
    ]
    ui_config = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        current = inputs.get("current", {})
        target = inputs.get("target", {})
        if not isinstance(current, dict) or not isinstance(target, dict):
            return {"heading": json.dumps({"error": "invalid state input"})}

        pos_a = current.get("position", [0, 0, 0])
        pos_b = target.get("position", [0, 0, 0])

        # Habitat uses Y-up: horizontal plane is XZ, forward is -Z
        dx = pos_b[0] - pos_a[0]
        dz = pos_b[2] - pos_a[2]
        absolute_rad = math.atan2(dx, -dz)
        absolute_deg = math.degrees(absolute_rad) % 360

        # Extract agent yaw from quaternion orientation
        quat = current.get("orientation", [0, 0, 0, 1])
        agent_yaw_deg = _quat_to_yaw_deg(quat)

        relative_deg = (absolute_deg - agent_yaw_deg) % 360
        if relative_deg > 180:
            relative_deg -= 360  # normalize to [-180, 180]

        return {
            "heading": json.dumps(
                {
                    "absolute_deg": round(absolute_deg, 1),
                    "relative_deg": round(relative_deg, 1),
                    "interpretation": _interpret_heading(relative_deg),
                }
            )
        }


def _quat_to_yaw_deg(quat: list) -> float:
    """Convert [qx, qy, qz, qw] quaternion to yaw in degrees (Habitat Y-up)."""
    if len(quat) < 4:
        return 0.0
    qx, qy, qz, qw = quat
    # Yaw around Y axis: atan2(2*(qw*qy + qx*qz), 1 - 2*(qy*qy + qz*qz))
    # Simplified for Habitat's convention
    siny_cosp = 2.0 * (qw * qy + qx * qz)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad) % 360


def _interpret_heading(relative_deg: float) -> str:
    """Human-readable heading interpretation."""
    abs_deg = abs(relative_deg)
    if abs_deg < 15:
        return "ahead"
    elif abs_deg < 60:
        return "slightly left" if relative_deg < 0 else "slightly right"
    elif abs_deg < 120:
        return "left" if relative_deg < 0 else "right"
    elif abs_deg < 165:
        return "behind-left" if relative_deg < 0 else "behind-right"
    else:
        return "behind"


# ══════════════════════════════════════════════════════════════════════
# Category 5: Episode Context
# ══════════════════════════════════════════════════════════════════════


def _try_get_habitat_mgr() -> Any:
    """Try to access HabitatEnvManager via the component registry.

    Uses the same pattern as policy_cma.py — avoids importing the habitat
    module directly, which would fail under the agentcanvas env (ADR-020).
    """
    try:
        from app.state import get_services

        for ns in get_services().workspace_component_registry._live_nodesets.values():
            mgr = getattr(ns, "_mgr", None)
            if mgr is not None and getattr(mgr, "initialized", False):
                return mgr
    except Exception:
        pass
    return None


class GetInstructionNode(BaseCanvasNode):
    """Get the current episode's navigation instruction."""

    node_type = "basic_agent__get_instruction"
    display_name = "Episode: Get Instruction"
    description = "Retrieve the navigation instruction for the current episode"
    category = "tool"
    icon = "FileText"
    input_ports = []  # seed node
    output_ports = [
        PortDef("instruction", "TEXT", "Navigation instruction text"),
    ]
    ui_config = NodeUIConfig(color="rose")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _try_get_habitat_mgr()
        if mgr is None:
            return {"instruction": "(Habitat not loaded)"}

        import asyncio

        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(mgr.executor, mgr.get_episode_info)
        return {"instruction": info.get("instruction", "")}


class GetStepCountNode(BaseCanvasNode):
    """Get the current iteration step count."""

    node_type = "basic_agent__get_step_count"
    display_name = "Episode: Get Step Count"
    description = "Get the current iteration step number"
    category = "tool"
    icon = "Hash"
    input_ports = [
        PortDef("trigger", "TEXT", "Optional trigger to re-fire", optional=True),
    ]
    output_ports = [
        PortDef("count", "TEXT", "Current step number"),
    ]
    ui_config = NodeUIConfig(color="rose")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {"count": str(ctx.step)}


class GetHistoryNode(BaseCanvasNode):
    """Accumulate and retrieve action history across iterations."""

    node_type = "basic_agent__get_history"
    display_name = "Episode: Get History"
    description = "Accumulate action history and retrieve as formatted text"
    category = "tool"
    icon = "ClipboardList"
    input_ports = [
        PortDef("action", "ACTION", "Action taken this step", optional=True),
        PortDef("response", "TEXT", "LLM response this step", optional=True),
    ]
    output_ports = [
        PortDef("history", "TEXT", "Formatted action history"),
    ]
    ui_config = NodeUIConfig(
        color="rose",
        config_fields=[
            ConfigField(
                "max_history", "slider", label="Max entries", default=50, min=5, max=200, step=5
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Initialize persistent state on first firing
        if ctx.basic_history is None:
            ctx.basic_history = []

        action = inputs.get("action")
        response = inputs.get("response")

        # Record this step's action (if provided)
        if action is not None:
            action_int = int(action)
            entry = {
                "step": ctx.step,
                "action": action_int,
                "action_name": ACTION_NAMES.get(action_int, "UNKNOWN"),
            }
            # Extract a short thought from the LLM response
            if response:
                thought = str(response)[:120]
                if len(str(response)) > 120:
                    thought += "..."
                entry["thought"] = thought
            ctx.basic_history.append(entry)

        max_entries = int(self.config.get("max_history", 50))
        entries = ctx.basic_history[-max_entries:]

        if not entries:
            return {"history": "(no actions yet)"}

        lines = []
        for e in entries:
            line = f"Step {e['step']}: {e['action_name']}"
            if "thought" in e:
                line += f" — {e['thought']}"
            lines.append(line)

        return {"history": "\n".join(lines)}


# ══════════════════════════════════════════════════════════════════════
# Category 6: Observation Encoding
# ══════════════════════════════════════════════════════════════════════


class ObsToTextNode(BaseCanvasNode):
    """Format pose + caption into a readable observation text block."""

    node_type = "basic_agent__obs_to_text"
    display_name = "Obs: To Text"
    description = "Format pose and image caption into a prompt-ready observation block"
    category = "tool"
    icon = "Eye"
    input_ports = [
        PortDef("pose", "POSE", "Agent pose (position + orientation)", optional=True),
        # caption is required so the node fires exactly once per step, after the
        # (slow) captioner returns — optional pose/extra arrivals alone must not
        # trigger an early fire that double-fires downstream LLM nodes.
        PortDef("caption", "TEXT", "Image caption / scene description"),
        PortDef("extra", "TEXT", "Extra context appended verbatim", optional=True),
    ]
    output_ports = [
        PortDef("observation", "TEXT", "Formatted observation text"),
    ]
    ui_config = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        lines: list[str] = []

        pose = inputs.get("pose")
        if isinstance(pose, dict):
            pos = pose.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                lines.append(f"You are at position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}).")
            quat = pose.get("orientation")
            if isinstance(quat, (list, tuple)) and len(quat) >= 4:
                yaw = _quat_to_yaw_deg(list(quat))
                lines.append(f"You are facing heading {yaw:.0f} degrees.")

        caption = inputs.get("caption")
        if caption and str(caption).strip():
            lines.append(f"You see: {str(caption).strip()}")

        extra = inputs.get("extra")
        if extra and str(extra).strip():
            lines.append(str(extra).strip())

        return {"observation": "\n".join(lines) if lines else "(no observation)"}


class FrameSampleNode(BaseCanvasNode):
    """Subsample a frame list to at most k frames.

    Entries may be image arrays, base64 strings, or file paths — paths are
    loaded into RGB arrays after sampling (env nodesets like openeqa emit
    frame paths and defer loading to the sampler).
    """

    node_type = "basic_agent__frame_sample"
    display_name = "Frames: Sample"
    description = "Uniformly subsample a list of frames to at most k (loads file paths)"
    category = "tool"
    icon = "Film"
    input_ports = [
        PortDef("frames", "LIST[IMAGE]", "Frames to subsample"),
    ]
    output_ports = [
        PortDef("sampled", "LIST[IMAGE]", "At most k frames, original order"),
    ]
    ui_config = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("k", "slider", label="Max frames", default=8, min=1, max=64, step=1),
            ConfigField(
                "strategy",
                "select",
                label="Strategy",
                default="uniform",
                options=[
                    {"value": "uniform", "label": "Uniform spread"},
                    {"value": "first", "label": "First k"},
                    {"value": "last", "label": "Last k"},
                ],
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        frames = inputs.get("frames") or []
        if not isinstance(frames, list):
            frames = [frames]
        k = int(self.config.get("k", 8))
        strategy = (self.config.get("strategy") or "uniform").strip()

        if len(frames) <= k:
            sampled = list(frames)
        elif strategy == "first":
            sampled = frames[:k]
        elif strategy == "last":
            sampled = frames[-k:]
        else:
            if k == 1:
                indices = [0]
            else:
                indices = sorted({round(i * (len(frames) - 1) / (k - 1)) for i in range(k)})
            sampled = [frames[i] for i in indices]

        sampled = [self._load_if_path(f) for f in sampled]
        sampled = [f for f in sampled if f is not None]

        self._self_log("input_count", len(frames))
        self._self_log("sampled_count", len(sampled))
        return {"sampled": sampled}

    @staticmethod
    def _load_if_path(frame: Any) -> Any:
        """Load a file-path entry into an RGB uint8 array; pass others through."""
        if not isinstance(frame, str) or len(frame) > 4096 or "\n" in frame:
            return frame
        import os

        if not os.path.isfile(frame):
            return frame
        try:
            import numpy as np
            from PIL import Image

            with Image.open(frame) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class BasicAgentNodeSet(BaseNodeSet):
    """General-purpose tools for LLM-based VLN agents.

    13 canvas nodes across 6 categories:
    - Scratch Pad: note_write, note_read, note_list
    - Web Grounding: web_search, web_fetch
    - Vision: image_analyze
    - Spatial Math: measure_distance, compute_heading
    - Episode Context: get_instruction, get_step_count, get_history
    - Observation Encoding: obs_to_text, frame_sample
    """

    name = "basic_agent"
    description = "General-purpose tools for LLM-based VLN agents"

    def get_tools(self) -> list:
        return [
            # Scratch Pad
            NoteWriteNode(),
            NoteReadNode(),
            NoteListNode(),
            # Web Grounding
            WebSearchNode(),
            WebFetchNode(),
            # Vision
            ImageAnalyzeNode(),
            # Spatial Math
            MeasureDistanceNode(),
            ComputeHeadingNode(),
            # Episode Context
            GetInstructionNode(),
            GetStepCountNode(),
            GetHistoryNode(),
            # Observation Encoding
            ObsToTextNode(),
            FrameSampleNode(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("BasicAgentNodeSet initialized (13 nodes)")

    async def shutdown(self) -> None:
        _scratch_pads.clear()
        log.info("BasicAgentNodeSet shut down (scratch pads cleared)")
