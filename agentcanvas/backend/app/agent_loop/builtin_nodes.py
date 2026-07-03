"""Built-in node catalog — framework-owned ``BaseCanvasNode`` subclasses.

Handlers here ship with the framework and are registered into
``NODE_HANDLERS`` at import time. Workspace nodes are added later by
``WorkspaceComponentRegistry`` via :func:`register_node`. The actual execution
engine lives in ``graph_executor.py``; this file only defines the
handler classes.

The ``ctx`` argument passed to ``forward(inputs, ctx)`` is a
``_NodeStateProxy`` (see ``graph_executor.py``) — an attribute-bag
backed by a per-node state dict, plus ``ctx.graph_state`` / ``ctx.containers``
for shared state containers and ``ctx.session`` / ``ctx.step`` as invariants.

Node groupings in this file:

* **Processors** — have inputs and outputs, do substantive work.

  - ``LLMCallNode``

* **Iteration** — loop lifecycle (two-sided pivots, ADR-dataflow-008).

  - ``IterInNode`` (two-sided: run-start init inputs + per-iter source),
    ``IterOutNode`` (two-sided: loop-carry sink + ``stop`` halt input;
    ``final_*`` outputs emit once at scope termination — the standalone
    ``termination`` node was removed when stop moved onto the pivot).

* **Composite boundaries** — bridged by ``flatten_graph`` at execution time.

  - ``GraphInNode``, ``GraphOutNode``

* **Sinks** — no output ports; consume inputs for side effects
  (WebSocket events).

  - ``ImageViewerSink`` — configurable grid of RGB/DEPTH cells; ports
    derived from ``config.ports`` (see ``_resolve_ports``).
  - ``_SinkBase`` subclasses (``TextScrollSink``, ``ActionLogSink``,
    ``MetricsViewerSink``, ``TextViewerSink``) — emit per-node
    ``viewer_data`` WS events, one per viewer instance. ``TextViewerSink``
    shows the latest TEXT; ``TextScrollSink`` accumulates TEXT into a
    scrollable history.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from ..components.bases import BaseCanvasNode, ConfigField, DisplayField, NodeUIConfig, PortDef
from ..state import broadcast

log = logging.getLogger("agentcanvas.builtin-nodes")


# ── Helpers ──


def _get_input(inputs: dict, port_name: str, default: Any = None) -> Any:
    """Get input value by port name with legacy fallback."""
    if port_name in inputs:
        return inputs[port_name]
    for val in inputs.values():
        if isinstance(val, dict) and port_name in val:
            return val[port_name]
    return default


# ══════════════════════════════════════════════════════════════════════════════
# NODE HANDLER CLASSES — all inherit BaseCanvasNode
# ══════════════════════════════════════════════════════════════════════════════


class LLMCallNode(BaseCanvasNode):
    """Unified LLM/VLM node: template interpolation + model call.

    Combines the functionality of PromptTemplateNode + old LLMCallNode +
    VLMCallNode into a single node.  When the ``rgb`` port receives image
    data the node uses the VLM path; otherwise the text path.  Either
    path multi-samples through the ``*_n`` variant when ``config.n > 1``.

    Modes (``config.mode``):

    * ``"single_turn"`` (default) — one user message built from the template,
      no persisted history.
    * ``"conversation"`` — multi-turn chat. Requires a ``graph_state``
      container with a ``messages`` state entry and an access grant on this
      node. The per-node ``ctx.messages`` fallback was removed — conversation
      history must live in a tracked state container so lifetime / checkpoint
      semantics are explicit.

    If the legacy ``prompt`` port is wired (from a deprecated
    PromptTemplateNode), its value is used directly and template
    interpolation is skipped.
    """

    node_type = "llmCall"
    display_name = "LLM Call"
    description = "Interpolate template + call LLM/VLM (auto-detects vision)"
    category = "llm"
    icon = "MessageSquare"
    ports_mode = "input"
    input_ports = [
        PortDef("scene", "TEXT", "Scene description", optional=True),
        PortDef("history", "TEXT", "Formatted action history", optional=True),
        PortDef("plan", "TEXT", "Action plan", optional=True),
        PortDef("pose", "POSE", "Agent pose (position + orientation)", optional=True),
        PortDef("instruction", "TEXT", "Navigation instruction", optional=True),
        PortDef("context", "TEXT", "Generic context", optional=True),
        PortDef("system_prompt", "TEXT", "System prompt (port overrides config)", optional=True),
        PortDef("prompt", "TEXT", "Pre-rendered prompt (legacy, skips template)", optional=True),
        PortDef(
            "rgb",
            "LIST[IMAGE]",
            "Image input(s) — any attached images enable VLM mode (ADR-027)",
            optional=True,
        ),
        PortDef(
            "image_labels",
            "LIST[TEXT]",
            "Per-image captions interleaved before each image (1:1 with rgb)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("response", "TEXT", "LLM/VLM response text"),
        PortDef("responses", "LIST[TEXT]", "All sampled responses when n>1; len-1 list when n=1"),
    ]
    ui_config = NodeUIConfig(
        config_fields=[
            ConfigField(
                "ports",
                "port_list",
                label="Input Ports",
                default=[
                    {"name": "scene", "wire_type": "TEXT"},
                    {"name": "history", "wire_type": "TEXT"},
                    {"name": "plan", "wire_type": "TEXT"},
                    {"name": "pose", "wire_type": "POSE"},
                    {"name": "instruction", "wire_type": "TEXT"},
                    {"name": "context", "wire_type": "TEXT"},
                    {"name": "system_prompt", "wire_type": "TEXT"},
                    {"name": "prompt", "wire_type": "TEXT"},
                    {"name": "rgb", "wire_type": "LIST[IMAGE]"},
                    {"name": "image_labels", "wire_type": "LIST[TEXT]"},
                ],
            ),
            ConfigField(
                "template",
                "textarea",
                "Prompt template",
                placeholder="e.g. Given: {instruction}\\nScene: {scene}\\nDecide action.",
            ),
            ConfigField(
                "system_prompt",
                "textarea",
                "System prompt",
                placeholder="System prompt (overridden by port if wired)",
            ),
            ConfigField(
                "profile",
                "select",
                "Model",
                default="",
                options=[{"value": "__DYNAMIC_PROFILES__", "label": ""}],
            ),
            ConfigField(
                "temperature", "slider", "Temperature", default=0.7, min=0.0, max=2.0, step=0.1
            ),
            ConfigField(
                "max_tokens", "slider", "Max tokens", default=1024, min=64, max=4096, step=64
            ),
            ConfigField(
                "n",
                "slider",
                "Sample count (n)",
                default=1,
                min=1,
                max=10,
                step=1,
            ),
            ConfigField(
                "image_detail",
                "select",
                "Image detail (VLM)",
                default="low",
                options=[
                    {"value": "low", "label": "low (token-cheap)"},
                    {"value": "high", "label": "high (fine grounding)"},
                    {"value": "auto", "label": "auto"},
                ],
            ),
            ConfigField(
                "image_mime",
                "select",
                "Image MIME (b64 passthrough)",
                default="image/png",
                options=[
                    {"value": "image/png", "label": "image/png"},
                    {"value": "image/jpeg", "label": "image/jpeg"},
                ],
            ),
            ConfigField("write_to_state", "text", "Write response to state", default=""),
            ConfigField(
                "mode",
                "select",
                "Mode",
                default="single_turn",
                options=[
                    {"value": "single_turn", "label": "Single turn"},
                    {"value": "conversation", "label": "Conversation"},
                ],
            ),
        ],
    )

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        ports_cfg = config.get("ports")
        if ports_cfg and isinstance(ports_cfg, list):
            input_list = [
                PortDef(
                    p["name"],
                    p.get("wire_type", "TEXT"),
                    optional=not bool(p.get("required", False)),
                )
                for p in ports_cfg
            ]
            return (input_list, cls.output_ports)
        return (cls.input_ports, cls.output_ports)

    async def forward(self, inputs, ctx):
        from ..llm import (
            get_llm_config,
            llm_complete,
            llm_complete_n,
            vlm_complete,
            vlm_complete_n,
        )

        # ── 1. Build the prompt ──
        prompt_input = _get_input(inputs, "prompt")

        if prompt_input:
            # Legacy path: pre-rendered prompt from deprecated PromptTemplateNode
            rendered_prompt = str(prompt_input)
        else:
            # Unified path: interpolate {variables} into config.template
            template = self.config.get("template", "")
            variables: dict[str, str] = {"step": str(ctx.step)}

            # Collect ALL input port values as template variables
            # (excludes prompt, rgb, image_labels, system_prompt, pose — handled separately)
            _skip = {"prompt", "rgb", "image_labels", "system_prompt", "pose"}
            for key, val in inputs.items():
                if key not in _skip and val is not None:
                    variables[key] = str(val)

            # Unpack POSE dict into {pos} and {heading}
            pose = _get_input(inputs, "pose")
            if isinstance(pose, dict):
                pos = pose.get("position", [0, 0, 0])
                variables["pos"] = (
                    "{:.2f}, {:.2f}, {:.2f}".format(*tuple(pos[:3])) if len(pos) >= 3 else str(pos)
                )
                orient = pose.get("orientation", [0, 0, 0, 1])
                variables["heading"] = str(_quat_to_heading_deg(orient))
                variables["position"] = variables["pos"]

            # Render template — first pass with port variables
            rendered_prompt = template
            for key, value in variables.items():
                rendered_prompt = rendered_prompt.replace(f"{{{key}}}", value)

            # Fallback: read unresolved {variables} from graph_state
            if hasattr(ctx, "graph_state") and ctx.graph_state:
                for match in re.findall(r"\{(\w+)\}", rendered_prompt):
                    if match not in variables:
                        try:
                            val = ctx.graph_state.read(match)
                        except KeyError:
                            continue
                        if val is not None:
                            if isinstance(val, list):
                                variables[match] = "\n".join(str(v) for v in val)
                            else:
                                variables[match] = str(val)
                # Re-render with state-resolved variables
                for key, value in variables.items():
                    rendered_prompt = rendered_prompt.replace(f"{{{key}}}", value)

            # Replace remaining unresolved variables with defaults
            rendered_prompt = re.sub(
                r"\{(\w+)\}",
                lambda m: f"(no {m.group(1)})",
                rendered_prompt,
            )

        # Log the assembled prompt
        self._self_log("rendered_prompt", rendered_prompt)

        # ── 2. Resolve system prompt (port overrides config) ──
        system_prompt = _get_input(inputs, "system_prompt", "") or ""
        if not system_prompt:
            system_prompt = self.config.get("system_prompt", "") or ""

        if system_prompt:
            self._self_log("system_prompt", system_prompt)

        # ── 3. Get LLM config ──
        profile_name = self.config.get("profile", "")
        llm_config = get_llm_config(profile_name)
        self._self_log("model", getattr(llm_config, "model", "") if llm_config else "none")
        if not llm_config:
            return {"response": "(no LLM profile active)"}

        temp = self.config.get("temperature", 0.7)
        max_tokens = int(self.config.get("max_tokens", 1024))
        n = max(1, int(self.config.get("n", 1) or 1))
        image_detail = str(self.config.get("image_detail", "low") or "low")
        # data-URL MIME for image payloads. b64 strings arriving on the rgb
        # port are passed through verbatim, so a producer emitting JPEG
        # (e.g. Three-Step's build_images, mirroring upstream's JPEG
        # re-encodes) sets image_mime="image/jpeg" on this node's config.
        # numpy arrays are still encoded PNG — leave the default for those.
        image_mime = str(self.config.get("image_mime", "image/png") or "image/png")
        mode = self.config.get("mode", "single_turn")
        _has_gs = hasattr(ctx, "graph_state") and ctx.graph_state is not None

        # ── 4. Call LLM or VLM ──
        # rgb is declared as LIST[IMAGE] (ADR-027).  The executor auto-wraps
        # single-image producers to [img], so we always iterate here.  A
        # legacy scalar value is tolerated as a 1-item fallback in case a
        # caller bypassed the port-binding seam.
        rgb = _get_input(inputs, "rgb")
        images: list[str] = []
        if rgb is not None:
            import numpy as np

            from ..standard.wire_types import image_to_base64

            items = rgb if isinstance(rgb, list) else [rgb]
            for item in items:
                if isinstance(item, np.ndarray):
                    if item.size == 0 or item.ndim != 3 or item.shape[2] != 3:
                        continue
                    images.append(image_to_base64(item))
                elif isinstance(item, str) and len(item) > 100:
                    images.append(item)

        labels_raw = _get_input(inputs, "image_labels")
        image_labels: list[str] | None = None
        if labels_raw is not None:
            if isinstance(labels_raw, list):
                image_labels = [str(s) for s in labels_raw]
            elif isinstance(labels_raw, str):
                image_labels = [labels_raw]

        if images:
            self._self_log("image_count", len(images))
            self._self_log("label_count", len(image_labels) if image_labels else 0)
            # VLM path — vision model. n>1 multi-sampling goes through
            # vlm_complete_n (provider-native ``n`` with a concurrent
            # single-sample fallback for providers that ignore it).
            if n > 1:
                responses_list = await vlm_complete_n(
                    llm_config,
                    rendered_prompt,
                    images,
                    n=n,
                    image_labels=image_labels,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    detail=image_detail,
                    mime=image_mime,
                )
                response = responses_list[0] if responses_list else None
            else:
                response = await vlm_complete(
                    llm_config,
                    rendered_prompt,
                    images,
                    image_labels=image_labels,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    detail=image_detail,
                    mime=image_mime,
                )
                responses_list = [response] if response else []
        else:
            # LLM path — text-only model
            if mode == "single_turn":
                messages = [{"role": "user", "content": rendered_prompt}]
            elif mode == "conversation":
                # Conversation mode requires a graph_state access grant with
                # a ``messages`` container entry (ADR-014). Per-node state
                # (``ctx.messages``) was removed — it had no lifetime
                # management and was invisible to checkpoint/restore.
                if not _has_gs:
                    raise ValueError(
                        "LLMCallNode: mode='conversation' requires a "
                        "graph_state container with an access grant on this "
                        "node. Declare a 'messages' state entry and wire an "
                        "access grant, or use mode='single_turn'."
                    )
                ctx.graph_state.write("messages", {"role": "user", "content": rendered_prompt})
                messages = ctx.graph_state.read("messages")
            else:
                raise ValueError(
                    f"LLMCallNode: unknown mode {mode!r} "
                    "(expected 'single_turn' or 'conversation')."
                )

            stop_cfg = self.config.get("stop") or None
            if isinstance(stop_cfg, str):
                stop_cfg = [stop_cfg]
            if n > 1 and mode == "single_turn":
                responses_list = await llm_complete_n(
                    llm_config,
                    messages,
                    n=n,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    stop=stop_cfg,
                )
                response = responses_list[0] if responses_list else None
            else:
                response = await llm_complete(
                    llm_config,
                    messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    stop=stop_cfg,
                )
                responses_list = [response] if response else []

            if response and mode == "conversation":
                # _has_gs already verified above — this branch only reached
                # when conversation mode passed the guard.
                ctx.graph_state.write("messages", {"role": "assistant", "content": response})

        # Per-call token usage is auto-emitted by the executor's LLM-usage
        # hook — see ``_current_node_usage`` in ``app.llm.call``.
        self._self_log("response_length", len(response) if response else 0)
        self._self_log("response_count", len(responses_list))

        # ── 5. Optionally persist response to graph_state ──
        write_state = self.config.get("write_to_state", "")
        if write_state and response and _has_gs:
            ctx.graph_state.write(write_state, response)

        return {"response": response or "", "responses": responses_list}


class TextParseNode(BaseCanvasNode):
    """Generic structured parser for LLM response text.

    Three modes:
      - ``choice``: match one of N comma-separated keywords (case-insensitive).
        ``index`` is the keyword's position in the list — for an action space
        declared as ``"STOP,FORWARD,LEFT,RIGHT"`` the index doubles as the
        discrete env action id.
      - ``regex``: ``re.search`` with the configured pattern; ``value`` is the
        first capture group (or the whole match if no groups).
      - ``json_field``: parse the text as JSON (falling back to the first
        ``{...}`` block found) and read a dot-separated field path.

    Replaces per-method parse nodes for demo-level graphs — wire
    ``llmCall.response → textParse.text → env step / downstream``.
    """

    node_type = "textParse"
    display_name = "Text Parse"
    description = "Extract a keyword, regex capture, or JSON field from text"
    category = "processing"
    icon = "GitBranch"
    input_ports = [PortDef("text", "TEXT", "Text to parse")]
    output_ports = [
        PortDef("value", "TEXT", "Matched keyword / capture / field value"),
        PortDef("index", "ACTION", "Choice mode: index of matched keyword; -1 otherwise"),
        PortDef("rest", "TEXT", "Input text with the matched part removed"),
    ]
    ui_config = NodeUIConfig(
        color="pink",
        config_fields=[
            ConfigField(
                "mode",
                "select",
                label="Mode",
                default="choice",
                options=[
                    {"value": "choice", "label": "Choice (keyword list)"},
                    {"value": "regex", "label": "Regex (first capture)"},
                    {"value": "json_field", "label": "JSON field (dot path)"},
                ],
            ),
            ConfigField(
                "choices",
                "text",
                label="Choices (comma-separated)",
                default="STOP,FORWARD,LEFT,RIGHT",
                placeholder="e.g. STOP,FORWARD,LEFT,RIGHT or A,B,C,D",
            ),
            ConfigField(
                "pattern",
                "text",
                label="Regex pattern",
                default="",
                placeholder=r"e.g. Your mark:\s*(\d)",
            ),
            ConfigField(
                "json_key",
                "text",
                label="JSON field path",
                default="",
                placeholder="e.g. action or plan.next_step",
            ),
            ConfigField(
                "scan",
                "select",
                label="Choice scan strategy",
                default="last_line_first",
                options=[
                    {"value": "last_line_first", "label": "Last line first, then full text"},
                    {"value": "full_text", "label": "Full text only"},
                ],
            ),
            ConfigField(
                "default",
                "text",
                label="Default value (no match)",
                default="",
                placeholder="Fallback when nothing matches",
            ),
        ],
    )

    async def forward(self, inputs, ctx):
        text = str(inputs.get("text", "") or "")
        mode = (self.config.get("mode") or "choice").strip()
        default = str(self.config.get("default", "") or "")

        value, index, rest = "", -1, text

        if mode == "choice":
            choices = [
                c.strip() for c in str(self.config.get("choices", "")).split(",") if c.strip()
            ]
            scan = (self.config.get("scan") or "last_line_first").strip()
            regions: list[str] = []
            if scan == "last_line_first":
                lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
                if lines:
                    regions.append(lines[-1])
            regions.append(text)
            for region in regions:
                region_upper = region.upper()
                for i, choice in enumerate(choices):
                    if choice.upper() in region_upper:
                        value, index = choice, i
                        break
                if index >= 0:
                    break
            if index < 0 and default:
                value = default
                if default in choices:
                    index = choices.index(default)
            if value:
                pos = text.upper().find(value.upper())
                if pos >= 0:
                    rest = (text[:pos] + text[pos + len(value) :]).strip()

        elif mode == "regex":
            pattern = str(self.config.get("pattern", "") or "")
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE) if pattern else None
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                rest = (text[: match.start()] + text[match.end() :]).strip()
            else:
                value = default

        elif mode == "json_field":
            import json as _json

            payload = None
            try:
                payload = _json.loads(text)
            except (ValueError, TypeError):
                block = re.search(r"\{.*\}", text, re.DOTALL)
                if block:
                    try:
                        payload = _json.loads(block.group(0))
                    except (ValueError, TypeError):
                        payload = None
            if isinstance(payload, dict):
                node: Any = payload
                for key in str(self.config.get("json_key", "")).split("."):
                    if isinstance(node, dict) and key in node:
                        node = node[key]
                    else:
                        node = None
                        break
                value = str(node) if node is not None else default
            else:
                value = default

        self._self_log("mode", mode)
        self._self_log("value", value)
        self._self_log("index", index)
        return {"value": value, "index": index, "rest": rest}


class HistoryLogNode(BaseCanvasNode):
    """Accumulate one TEXT entry per firing and emit the formatted history.

    State lives on the node instance (persists across firings within a run;
    batch eval creates a fresh LoopRunner per episode, so history naturally
    resets per episode). Typical loop wiring: ``textParse.value → entry``,
    then carry ``history`` through ``iterOut`` back into the next iteration's
    prompt — no state container or graph_state grant needed.
    """

    node_type = "historyLog"
    display_name = "History Log"
    description = "Accumulate per-step entries into formatted history text"
    category = "processing"
    icon = "ClipboardList"
    input_ports = [PortDef("entry", "TEXT", "This step's entry", optional=True)]
    output_ports = [PortDef("history", "TEXT", "Formatted history including this step")]
    ui_config = NodeUIConfig(
        color="emerald",
        config_fields=[
            ConfigField(
                "max_entries", "slider", label="Max entries", default=20, min=5, max=200, step=5
            ),
            ConfigField(
                "template",
                "text",
                label="Line template",
                default="Step {step}: {entry}",
                placeholder="Placeholders: {step}, {entry}",
            ),
        ],
    )

    async def forward(self, inputs, ctx):
        if ctx.history_entries is None:
            ctx.history_entries = []

        entry = inputs.get("entry")
        if entry is not None and str(entry).strip():
            ctx.history_entries = [
                *ctx.history_entries,
                {"step": getattr(ctx, "step", 0), "entry": str(entry).strip()},
            ]

        entries = ctx.history_entries
        if not entries:
            return {"history": "(no history yet)"}

        max_entries = int(self.config.get("max_entries", 20))
        template = str(self.config.get("template") or "Step {step}: {entry}")
        lines: list[str] = []
        if len(entries) > max_entries:
            lines.append(f"(...{len(entries) - max_entries} earlier steps omitted)")
        for item in entries[-max_entries:]:
            try:
                lines.append(template.format(step=item["step"], entry=item["entry"]))
            except (KeyError, IndexError, ValueError):
                lines.append(f"Step {item['step']}: {item['entry']}")

        self._self_log("entry_count", len(entries))
        return {"history": "\n".join(lines)}


class NullSourceNode(BaseCanvasNode):
    """Emits ``None`` on a typed output port — once, at run-start.

    Bridges the gap when a graph needs to feed a typed-but-empty value into
    a downstream port (e.g. SIMPLER has no wrist camera, but
    ``policy_vla__adapt_env_to_canonical.wrist_image`` is a required IMAGE
    input). Wire ``value`` into an iterIn init port as a run-invariant;
    downstream nodes receive ``None`` every iteration. The output's
    ``wire_type`` is author-configurable so the edge type-checker stays
    meaningful.

    Has no inputs → seeded by GraphExecutor at run-start, fires once.
    """

    node_type = "nullSource"
    display_name = "Null Source"
    description = "Emits None on a typed output (seed; fires once at run-start)."
    category = "control"
    icon = "MinusCircle"
    ui_config = NodeUIConfig(
        color="slate",
        config_fields=[
            ConfigField(
                "wire_type",
                "select",
                label="Output wire type",
                default="ANY",
                options=[
                    {"value": "ANY", "label": "ANY"},
                    {"value": "TEXT", "label": "TEXT"},
                    {"value": "BOOL", "label": "BOOL"},
                    {"value": "IMAGE", "label": "IMAGE"},
                    {"value": "DEPTH", "label": "DEPTH"},
                    {"value": "ACTION", "label": "ACTION"},
                    {"value": "POSE", "label": "POSE"},
                    {"value": "METRICS", "label": "METRICS"},
                    {"value": "OBSERVATION", "label": "OBSERVATION"},
                    {"value": "STEP_RESULT", "label": "STEP_RESULT"},
                ],
            ),
        ],
    )
    input_ports: list[PortDef] = []
    # Static placeholder so the canvas renders the output handle. The actual
    # wire_type is per-instance — `_resolve_ports` overrides this at execute
    # time using config.wire_type so the edge type-checker sees the author's
    # choice. Without the placeholder the schema endpoint returns no output
    # ports and the canvas can't draw the handle, making valid edges look
    # disconnected even though the run executes them correctly.
    output_ports: list[PortDef] = [PortDef("value", "ANY", "Constant None")]

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list[PortDef], list[PortDef]]:
        wt = (config.get("wire_type") or "ANY").strip() or "ANY"
        return ([], [PortDef("value", wt, "Constant None")])

    async def forward(self, inputs, ctx):
        return {"value": None}


class NoteNode(BaseCanvasNode):
    """Markdown billboard — pure annotation, no inputs/outputs.

    Authors place a Note next to a cluster of nodes to explain what the
    cluster does, document a constraint (e.g. "this adapter is bound to
    the LIBERO env — switching env requires rewiring both env-adapter
    nodes"), or pin a TODO. Frontend renders ``markdown`` inline as
    GitHub-flavoured markdown; the executor sees a no-op leaf.

    Has no inputs → seeded by GraphExecutor at run-start; emits an
    empty dict so it can't satisfy any downstream dependency. Harmless.
    """

    node_type = "note"
    display_name = "Note"
    description = (
        "Markdown annotation displayed inline on the canvas (passive — no execution effect)."
    )
    category = "annotation"
    icon = "StickyNote"
    ui_config = NodeUIConfig(
        color="yellow",
        layout="note",
        min_width="200px",
        max_width="400px",
        min_height="80px",
        config_fields=[
            ConfigField(
                "markdown",
                "textarea",
                label="Markdown",
                default="**Note**\n\nWrite something here.",
                placeholder="Markdown content (headings, bold, lists, links, code, …).",
            ),
        ],
    )
    input_ports: list[PortDef] = []
    output_ports: list[PortDef] = []

    async def forward(self, inputs, ctx):
        return {}


class GraphInNode(BaseCanvasNode):
    """Boundary node — defines an input on the parent composite.

    At execution time, flatten_graph() rewires parent edges through this node.
    When executed standalone (pre-flatten), acts as a pass-through seed.
    """

    node_type = "graphIn"
    display_name = "Graph In"
    description = "Composite input boundary"
    category = "control"
    icon = "ArrowDownToLine"
    kind = "control"
    ui_config = NodeUIConfig(
        color="blue",
        layout="strip",
        width="40px",
        min_height="80px",
        config_fields=[
            ConfigField("portName", "text", label="Port Name", default="input"),
            ConfigField(
                "wireType",
                "select",
                label="Wire Type",
                default="ANY",
                # Full wire_types.py inner-type registry (ADR-026 + ADR-027).
                # This is UI metadata only — flatten.py and the executor do
                # not read graphIn/graphOut.config.wireType at runtime.
                options=[
                    {"value": "ANY", "label": "ANY"},
                    {"value": "TEXT", "label": "TEXT"},
                    {"value": "BOOL", "label": "BOOL"},
                    {"value": "IMAGE", "label": "IMAGE"},
                    {"value": "DEPTH", "label": "DEPTH"},
                    {"value": "ACTION", "label": "ACTION"},
                    {"value": "POSE", "label": "POSE"},
                    {"value": "METRICS", "label": "METRICS"},
                    {"value": "OBSERVATION", "label": "OBSERVATION"},
                    {"value": "STEP_RESULT", "label": "STEP_RESULT"},
                ],
            ),
        ],
    )
    input_ports: list[PortDef] = []
    output_ports = [PortDef("value", "ANY", "Data from parent graph", optional=True)]

    async def forward(self, inputs, ctx):
        # Pass through whatever was injected (by flatten or by graph executor)
        return dict(inputs)


class GraphOutNode(BaseCanvasNode):
    """Boundary node — defines an output on the parent composite.

    At execution time, flatten_graph() rewires this to parent edges.
    When executed standalone (pre-flatten), acts as a pass-through sink.
    """

    node_type = "graphOut"
    display_name = "Graph Out"
    description = "Composite output boundary"
    category = "control"
    icon = "ArrowUpFromLine"
    kind = "control"
    ui_config = NodeUIConfig(
        color="orange",
        layout="strip",
        width="40px",
        min_height="80px",
        config_fields=[
            ConfigField("portName", "text", label="Port Name", default="output"),
            ConfigField(
                "wireType",
                "select",
                label="Wire Type",
                default="ANY",
                # Full wire_types.py inner-type registry (ADR-026 + ADR-027).
                # UI metadata only — the executor does not read this field.
                options=[
                    {"value": "ANY", "label": "ANY"},
                    {"value": "TEXT", "label": "TEXT"},
                    {"value": "BOOL", "label": "BOOL"},
                    {"value": "IMAGE", "label": "IMAGE"},
                    {"value": "DEPTH", "label": "DEPTH"},
                    {"value": "ACTION", "label": "ACTION"},
                    {"value": "POSE", "label": "POSE"},
                    {"value": "METRICS", "label": "METRICS"},
                    {"value": "OBSERVATION", "label": "OBSERVATION"},
                    {"value": "STEP_RESULT", "label": "STEP_RESULT"},
                ],
            ),
        ],
    )
    input_ports = [PortDef("value", "ANY", "Data to parent graph", optional=True)]
    output_ports: list[PortDef] = []

    async def forward(self, inputs, ctx):
        return dict(inputs)


# IterIn is the iteration boundary pivot, two-sided since ADR-dataflow-008:
# the left/input side receives run-start seeds on its ``init_<X>`` handles
# (declared via ``config.initPorts``); per-iteration loop-carry arrives via
# the graph executor's transfer from the paired iterOut. The output side is
# derived from ``config.ports`` and exposes the loop-carry bundle to the
# body of the iteration.
class IterInNode(BaseCanvasNode):
    """Iteration boundary — exposes loop-carry data as one unified port list.

    Each entry in ``config.ports`` is ``{name, wire_type, persist}`` and
    produces exactly one output handle named ``<name>`` (no prefix). The
    runtime maintains one slot per port in ``NodeInstance.port_slots``:

      - Writers are auto-detected by name match. Ports authored in
        ``config.initPorts`` are the node's left/input side — canvas
        edges targeting ``iterIn.init_<name>`` write those slots at
        run-start. If the paired ``iterOut`` declares a matching port,
        it writes the slot at each iteration boundary. A canvas edge
        targeting ``iterIn.<name>`` directly is also a first-class
        init-writer.
      - At fire time, iterIn emits all populated slots.
      - After each fire, for every port with ``persist=false`` the slot
        is cleared. Ports with ``persist=true`` keep their last-written
        value until next write.

    Six legal configurations per port, indexed by
    ``(persist, init-writer present, iterOut-writer present)``:

      C1 run-constant       persist=T, init=T, loop=F  (static text / instruction)
      C2 step-0 one-shot    persist=F, init=T, loop=F  (emitted only iter-0)
      C3 loop-carried       persist=F, init=F, loop=T  (fresh each iteration)
      P1 seeded + refreshed persist=T, init=T, loop=T  (seed then per-iter refresh)
      P5 pure feedback      persist=T, init=F, loop=T  (late init, carried)
      -- no-op              persist=*,  init=F, loop=F  (rejected — nothing writes)

    Legacy configs with ``init_ports`` / ``loop_ports`` are rejected at load
    by ``validate_graph_connectivity`` (ADR-031 migration to v3 schema).
    """

    node_type = "iterIn"
    display_name = "Iter In"
    description = "Iteration boundary — exposes loop-carry data each iteration"
    category = "control"
    icon = "RefreshCw"
    kind = "control"
    ports_mode = "source"
    # iterIn has no user-authored config fields — its port surface is
    # synthesised from its own initPorts + the paired iterOut at graph load
    # (``_synthesize_iterin_ports``). The persist flag lives on each writer's
    # port entry; the frontend editor aggregates and writes back to them.
    ui_config = NodeUIConfig(
        color="emerald",
        layout="strip",
        width="44px",
        min_height="140px",
        rounding="rounded-r-lg",
        config_fields=[],
    )
    # Port surface is fully synthesised from authored initPorts + paired
    # iterOut at graph load (``graph_def._synthesize_iterin_ports``) and
    # written back to ``config.ports``. Class-level lists are intentionally
    # empty — ``_resolve_ports`` always reads from ``config.ports``.
    input_ports: list[PortDef] = []
    output_ports: list[PortDef] = []

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        ports_cfg = config.get("ports")
        if ports_cfg and isinstance(ports_cfg, list):
            output_list = [
                PortDef(p["name"], p.get("wire_type", "TEXT"), optional=True) for p in ports_cfg
            ]
            return ([], output_list)
        # No config.ports yet — return empty shape. ``validate_graph_connectivity``
        # rejects iterIn nodes that reach runtime without a synthesised port list.
        return ([], [])

    async def forward(self, inputs, ctx):
        result: dict[str, Any] = {"step": ctx.step}

        for key, val in inputs.items():
            if val is not None:
                result[key] = val

        self._self_log("step", ctx.step)
        self._self_log("ports_received", [k for k in inputs if inputs[k] is not None])
        return result


# IterOut is input-only. Its port surface is author-declared in
# ``config.ports`` (populated via the frontend port-list editor). Class-level
# ``input_ports`` is intentionally empty — ``_resolve_ports`` is the single
# source of truth; ``validate_graph_connectivity`` rejects empty configs at
# load time.
class IterOutNode(BaseCanvasNode):
    """Two-sided iteration gate.

    Left/input side: the author-declared loop-carry ports (``config.ports``)
    plus a class-level ``stop`` BOOL — the loop's halt signal, read by the
    executor's once-per-iteration Decide check at the iterOut boundary.
    Unwired ``stop`` = budget-only loop.

    Right/output side: one ``final_<name>`` handle per loop-carry port plus
    a constant ``final_stop`` (BOOL). These emit exactly once, when the
    scope terminates (stop=true, step-budget exhaust, or the error path),
    carrying the last collected iteration's values — the engine-guaranteed
    input contract for the after-loop stage (evaluate / graphOut). During
    the loop they never fire.

    ``stop`` is excluded from both the loop-carry transfer (slot matching
    is by ``config.ports`` names) and the final mirror (only the constant
    ``final_stop`` represents it).
    """

    node_type = "iterOut"
    display_name = "Iter Out"
    description = (
        "Collects end-of-step data; stop input halts the loop; final_* emit once at termination"
    )
    category = "control"
    icon = "CornerDownLeft"
    kind = "control"
    ports_mode = "sink"
    ui_config = NodeUIConfig(
        color="emerald",
        layout="strip",
        width="44px",
        min_height="140px",
        rounding="rounded-l-lg",
        config_fields=[
            ConfigField(
                "ports",
                "port_list",
                label="Iteration Ports",
                default=[
                    {"name": "rgb", "wire_type": "IMAGE", "persist": True},
                    {"name": "depth", "wire_type": "DEPTH", "persist": True},
                    {"name": "pose", "wire_type": "POSE", "persist": True},
                    {"name": "action", "wire_type": "ACTION", "persist": True},
                    {"name": "response", "wire_type": "TEXT", "persist": True},
                ],
            ),
        ],
    )
    # Class-level ports intentionally empty — see note above.
    input_ports: list[PortDef] = []
    output_ports: list[PortDef] = []

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        stop_port = PortDef("stop", "BOOL", "Halt signal — truthy ends this loop", optional=True)
        ports_cfg = config.get("ports")
        if ports_cfg and isinstance(ports_cfg, list):
            port_list = [
                PortDef(
                    p["name"],
                    p.get("wire_type", "TEXT"),
                    optional=not bool(p.get("required", False)),
                )
                for p in ports_cfg
            ]
            final_ports = [
                PortDef(
                    f"final_{p['name']}",
                    p.get("wire_type", "TEXT"),
                    "Terminal-iteration value (emits once at scope termination)",
                )
                for p in ports_cfg
            ]
            final_ports.append(
                PortDef("final_stop", "BOOL", "True once at scope termination — after-loop trigger")
            )
            return ([stop_port, *port_list], final_ports)
        # No config.ports — validator rejects this at graph load.
        return ([stop_port], [PortDef("final_stop", "BOOL", "True once at scope termination")])

    async def forward(self, inputs, ctx):
        self._self_log("ports_carried", [k for k in inputs if inputs[k] is not None])
        return dict(inputs)


# ── Helpers ──


def _quat_to_heading_deg(orientation: list) -> float:
    """Convert [x, y, z, w] quaternion to heading in degrees (0-360)."""
    if not orientation or len(orientation) < 4:
        return 0.0
    x, y, z, w = orientation[:4]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return round(math.degrees(yaw_rad) % 360, 1)


# ── Output Viewer Sinks ──
# Unified canvas: output viewers are graph participants that accept wired
# inputs and act as data sinks (no outputs).  Data display on the frontend
# still comes via WS routing; the backend handler simply consumes the data
# so the graph executor can fire these nodes when edges deliver inputs.


class _SinkBase(BaseCanvasNode):
    """Base for output viewer nodes — accepts inputs, produces no outputs.

    Each viewer instance serializes its own wired inputs and emits a
    per-node ``viewer_data`` WS event.  The executor does not need to
    know about viewer types — viewers are self-contained.
    """

    category = "output"
    output_ports: list[PortDef] = []

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        from ..standard.wire_types import serialize_for_display

        # Identify which fields accumulate (append to array across firings)
        acc_keys = {f.data_key for f in self.ui_config.display_fields if f.accumulate}

        # Build per-port display data using this node's port definitions
        port_map = {p.name: p.wire_type for p in self.input_ports}
        fields: dict[str, Any] = {}
        for key, val in inputs.items():
            if val is not None:
                serialized = serialize_for_display(port_map.get(key, "ANY"), val)
                if key in acc_keys:
                    # Accumulate in persistent node state — send full array
                    # Note: key must NOT start with "_" (blocked by _NodeStateProxy)
                    hist_key = f"vacc_{key}"
                    history = getattr(ctx, hist_key, None) or []
                    history.append(serialized)
                    setattr(ctx, hist_key, history)
                    fields[key] = list(history)  # copy so mutations don't leak
                else:
                    fields[key] = serialized

        # Emit per-node WS event (routed by node_id on frontend)
        if fields and hasattr(ctx, "session") and ctx.session:
            await broadcast(
                ctx.session._ws(
                    "viewer_data",
                    {
                        "node_id": self.node_id,
                        "step": getattr(ctx, "step", 0),
                        "fields": fields,
                    },
                )
            )

        self._self_log("received_ports", list(fields.keys()))
        return {}


class ImageViewerSink(BaseCanvasNode):
    """Configurable multi-image/depth viewer.

    Ports are instance-derived from ``config.ports`` (list of
    ``{name, wire_type}`` with ``wire_type`` restricted to ``IMAGE`` or
    ``DEPTH``). The frontend lays out ``rows x cols`` image cells keyed by
    port name. Each firing emits a per-node ``viewer_data`` WS event
    containing the serialized fields for whichever ports delivered data.

    Dynamic ports default to ``optional=True`` so the viewer fires on any
    field that arrives (matches ``_SinkBase`` semantics). Graphs can mark a
    specific port ``required: true`` via ``config.ports[*]`` — the dataflow
    executor honours it.
    """

    node_type = "imageViewer"
    display_name = "Image Viewer"
    description = "Configurable grid of RGB/DEPTH cells (ports from config)"
    category = "output"
    icon = "Images"
    kind = "block"
    ports_mode = "sink"
    input_ports: list[PortDef] = []
    output_ports: list[PortDef] = []
    default_config = {
        "rows": 1,
        "cols": 1,
        "ports": [{"name": "rgb", "wire_type": "IMAGE"}],
    }
    # ``default_config`` above is the single source of truth for freshly
    # dropped nodes (consumed by frontend ``unifiedCatalog.buildBuiltinEntries``
    # at drag-drop time). ``ConfigField`` widgets read the live ``data`` on
    # the node (``ConfigFieldRenderer.tsx``: ``data[field.name] ?? field.default``),
    # so we intentionally omit ``default=`` here to avoid double-maintained
    # defaults drifting apart.
    ui_config = NodeUIConfig(
        color="orange",
        layout="imageGrid",
        config_fields=[
            ConfigField("rows", "slider", label="Rows", min=1, max=4, step=1),
            ConfigField("cols", "slider", label="Cols", min=1, max=4, step=1),
            ConfigField("ports", "port_list", label="Image Ports"),
        ],
    )

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        from ..standard.wire_types import is_list_type, unwrap_list

        ports_cfg = config.get("ports")
        if ports_cfg and isinstance(ports_cfg, list):
            allowed = {"IMAGE", "DEPTH"}
            port_list: list[PortDef] = []
            for p in ports_cfg:
                wt = p.get("wire_type", "IMAGE")
                # LIST[IMAGE] / LIST[DEPTH] render as a tiled thumbnail strip
                # (e.g. SmartWay's per-candidate RGB set fed to the planner LLM).
                inner = unwrap_list(wt) if is_list_type(wt) else wt
                if inner not in allowed:
                    raise ValueError(
                        "imageViewer: wire_type must be IMAGE, DEPTH, "
                        f"LIST[IMAGE] or LIST[DEPTH], got {wt!r}"
                    )
                port_list.append(
                    PortDef(
                        p["name"],
                        wt,
                        "",
                        optional=not bool(p.get("required", False)),
                    )
                )
            return (port_list, [])
        return (cls.input_ports, cls.output_ports)

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        from ..standard.wire_types import (
            is_list_type,
            serialize_for_viewer,
            unwrap_list,
        )

        ports_cfg = self.config.get("ports") or []
        wt_map: dict[str, str] = {p["name"]: p.get("wire_type", "IMAGE") for p in ports_cfg}
        fields: dict[str, Any] = {}
        for name, val in inputs.items():
            if val is None or name not in wt_map:
                continue
            wt = wt_map[name]
            if is_list_type(wt):
                # Emit a list of base64 tiles; the frontend lays them out in a
                # thumbnail strip within the port's cell.
                inner = unwrap_list(wt)
                items = val if isinstance(val, list) else [val]
                fields[name] = [
                    serialize_for_viewer(inner, item) for item in items if item is not None
                ]
            else:
                fields[name] = serialize_for_viewer(wt, val)

        if fields and getattr(ctx, "session", None):
            await broadcast(
                ctx.session._ws(
                    "viewer_data",
                    {
                        "node_id": self.node_id,
                        "step": getattr(ctx, "step", 0),
                        "fields": fields,
                    },
                )
            )

        self._self_log("rendered_images", list(fields.keys()))
        return {}


class TextScrollSink(_SinkBase):
    """Accumulating TEXT viewer — scrollable history of strings across firings.

    Companion to :class:`TextViewerSink` (latest-only). Both share the same
    ``text_viewer`` display_type on the frontend; the renderer detects
    array-vs-scalar and renders accordingly. Backend distinction is purely
    the ``accumulate=True`` flag, which routes ``_SinkBase`` through the
    ``vacc_*`` history path.
    """

    node_type = "textScroll"
    display_name = "Scroll Text"
    description = "Accumulating TEXT history, newest-last"
    icon = "ScrollText"
    ui_config = NodeUIConfig(
        color="orange",
        layout="viewer",
        display_fields=[
            DisplayField(
                "history",
                "text_viewer",
                "",
                data_key="text",
                max_visible=20,
                accumulate=True,
            ),
        ],
    )
    input_ports = [
        PortDef("text", "TEXT", "Text to append to history", optional=True),
    ]


class ActionLogSink(_SinkBase):
    node_type = "actionLog"
    display_name = "Action Log"
    description = "Action history with step numbers"
    icon = "ListOrdered"
    ui_config = NodeUIConfig(
        color="orange",
        layout="viewer",
        display_fields=[
            DisplayField(
                "actions", "log_list", "Actions", data_key="action", max_visible=20, accumulate=True
            ),
        ],
    )
    input_ports = [
        # Display sink: accumulates whatever it is given into a scrollable log
        # (raw LLM ``response`` TEXT, a parsed ACTION, etc.). A log viewer does
        # not constrain its input shape, so the port is ANY — mirrors the other
        # ``_SinkBase`` viewers and avoids false wire-type rejections at load.
        PortDef("action", "ANY", "Action / log entry to display", optional=True),
    ]


class MetricsViewerSink(_SinkBase):
    node_type = "metrics"
    display_name = "Metrics"
    description = "SPL/SR/nDTW/SDTW metrics display"
    icon = "BarChart3"
    ui_config = NodeUIConfig(
        color="orange",
        layout="viewer",
        display_fields=[
            DisplayField("metrics", "metric_table", "Metrics", data_key="metrics"),
        ],
    )
    input_ports = [
        # Display sink: renders whatever it is given as a metric table. Envs
        # disagree on the payload shape — env_habitat emits a METRICS dict,
        # env_mp3d emits a JSON string (TEXT) — so the port is ANY rather than
        # falsely constraining to one shape (mirrors the ActionLogSink port).
        PortDef("metrics", "ANY", "Metrics to display (dict or JSON string)", optional=True),
    ]


class TextViewerSink(_SinkBase):
    node_type = "textViewer"
    display_name = "Text Viewer"
    description = "Text/markdown content display"
    icon = "FileText"
    ui_config = NodeUIConfig(
        color="orange",
        layout="viewer",
        display_fields=[
            DisplayField("content", "text_viewer", "Content", data_key="text"),
        ],
    )
    input_ports = [
        PortDef("text", "TEXT", "Text to display", optional=True),
    ]


# ── Handler Registry ──


_BUILTIN_NODES: list[type[BaseCanvasNode]] = [
    LLMCallNode,
    TextParseNode,
    HistoryLogNode,
    IterInNode,
    IterOutNode,
    NullSourceNode,
    NoteNode,
    GraphInNode,
    GraphOutNode,
    # Output viewer sinks (unified canvas — ADR-010)
    ImageViewerSink,
    TextViewerSink,
    TextScrollSink,
    ActionLogSink,
    MetricsViewerSink,
]

NODE_HANDLERS: dict[str, type[BaseCanvasNode]] = {cls.node_type: cls for cls in _BUILTIN_NODES}


def register_node(cls: type[BaseCanvasNode]) -> None:
    """Register a custom node type (called by WorkspaceComponentRegistry).

    ADR-028 PC-1: validate batched-opt-in shape at scan time. If
    ``batched=True``, the class must declare a ``batch_dim`` that names an
    actual input port — fail fast here so misconfigured nodes never reach
    the runtime rendezvous tier.
    """
    if getattr(cls, "batched", False):
        batch_dim = getattr(cls, "batch_dim", "")
        if not batch_dim:
            raise ValueError(f"{cls.__name__}: batched=True requires a non-empty batch_dim")
        port_names = {p.name for p in getattr(cls, "input_ports", [])}
        if batch_dim not in port_names:
            raise ValueError(
                f"{cls.__name__}: batch_dim={batch_dim!r} must name an "
                f"input port, but input_ports has {sorted(port_names)}"
            )
    NODE_HANDLERS[cls.node_type] = cls
