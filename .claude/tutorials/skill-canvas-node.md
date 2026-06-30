# Skill: Canvas Node

## When to Use

You need a single atomic function on the canvas — a tool, transformer,
data source, or any block that takes inputs, does work, returns outputs.
Every canvas element is a `BaseCanvasNode` subclass.

## File Location & Naming

- **File**: `workspace/nodesets/{nodeset_name}.py` (nodes live alongside their NodeSet)
- **`node_type`**: `"{nodeset}__{node}"` (double underscore, e.g. `"my_tools__process"`)
- **Icon**: pick a Lucide icon name from https://lucide.dev/icons

## Skeleton

```python
from __future__ import annotations

from typing import Any, ClassVar

from app.components import BaseCanvasNode, NodeUIConfig, PortDef
from app.components.bases import ConfigField

class MyNode(BaseCanvasNode):
    """One-line description of what this node does."""

    node_type = "my_set__my_node"           # TODO: {nodeset}__{node}
    display_name = "My Node"                # TODO: human label
    description = "Does something useful"   # TODO: tooltip text
    category = "tool"                       # TODO: sidebar group
    icon = "Sparkles"                       # TODO: Lucide icon name

    input_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "Input text"),
        # TODO: add ports — each becomes an input handle on canvas
    ]
    output_ports: ClassVar[list] = [
        PortDef("result", "TEXT", "Processed output"),
        # TODO: add ports — each becomes an output handle on canvas
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="sky",                        # TODO: Tailwind color key
        config_fields=[
            # TODO: add inline UI controls (see ConfigField Types below)
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Access inputs by port name
        text = inputs.get("text", "")

        # Access config by key name
        # option = self.config.get("key", default)

        # Optional: log internal details for debugging
        # self._self_log("key", value)

        # Return dict matching output port names exactly
        return {"result": text}
```

## Required Fields

| Field | Type | Notes |
|-------|------|-------|
| `node_type` | `ClassVar[str]` | Must be unique across all nodes. Format: `{nodeset}__{node}` |
| `display_name` | `ClassVar[str]` | Human-readable label for sidebar and canvas |
| `description` | `ClassVar[str]` | Tooltip / docs text |
| `category` | `ClassVar[str]` | Sidebar group: `"tool"`, `"environment"`, `"llm"`, `"policy"`, `"example"`, etc. |
| `forward()` | `async` method | Must return `{port_name: value}` matching `output_ports` |

## Optional Fields

| Field | Default | When to use |
|-------|---------|-------------|
| `icon` | `""` | Always recommended — Lucide icon name |
| `input_ports` | `[]` | Omit for entry nodes (no inputs, fires immediately) |
| `output_ports` | `[]` | Omit for sink nodes (no outputs) |
| `ui_config` | `NodeUIConfig()` | Custom color, config controls, display fields |
| `kind` | `"block"` | Change to `"composite"` or `"control"` only for structural nodes |
| `config_schema` | `{}` | JSON Schema for properties panel (rarely needed — use ConfigField instead) |
| `default_config` | `{}` | Default values when dropped onto canvas |
| `batched` | `False` | Set `True` to opt into the per-`AutoServerApp` `BatchedInferenceServer` rendezvous (ADR-eval-002 PC-1). Requires `batch_dim`. |
| `batch_dim` | `""` | Name of the input port carrying the per-sample slot. Validated at scan time — must match an actual `input_ports` entry. The server is **pure-functional**: any per-call state (RNN hidden, etc.) must travel as explicit input/output ports. |

## ConfigField Types

Use these in `NodeUIConfig(config_fields=[...])`. Access in forward via `self.config.get("name", default)`.

| `field_type` | Widget | Key params |
|-------------|--------|------------|
| `"slider"` | Range slider | `min`, `max`, `step`, `default` |
| `"text"` | Single-line input | `placeholder`, `default` |
| `"textarea"` | Multi-line input | `placeholder`, `default` |
| `"select"` | Dropdown | `options=[{"value": "x", "label": "X"}]`, `default` |
| `"toggle"` | Checkbox | `default` (bool) |
| `"label"` | Read-only text | `default` |
| `"port_list"` | Dynamic port editor | For `_resolve_ports()` pattern (advanced) |

## NodeUIConfig Layouts

`NodeUIConfig(layout=...)` chooses the renderer used by `GenericBlockRenderer`. **Layout is independent of `kind`.**

| `layout` | When to use |
|----------|-------------|
| `"block"` (default) | Standard rectangle — title, ports, config_fields, display_fields |
| `"strip"` | Narrow vertical gate (IterIn / IterOut / PortIn / PortOut). Set `width="44px"` and a `rounding` class. |
| `"viewer"` | Display-only sink (textViewer, textScroll, actionLog, metrics). Hides `config_fields`, renders `display_fields` full-width. |
| `"imageGrid"` | ImageViewerSink-style — grid of image panels derived from `config.rows` / `config.cols` / `config.ports` (ADR-components-007). |

## DisplayField Types

Use these in `NodeUIConfig(display_fields=[...])` for live runtime data from WS events.

| `display_type` | Widget | Use for |
|----------------|--------|---------|
| `"image_viewer"` | `<img>` tag | Live image output (RGB, depth) |
| `"log_list"` | Scrollable list | Action history, reasoning trace (`accumulate=True`) |
| `"metric_table"` | Key-value table | SPL, SR, step count |
| `"text_viewer"` | Text block / stack | Plain text. `accumulate=False` → single latest block (textViewer); `accumulate=True` → scrollable stack of writes (textScroll). ADR-components-008. |

## Key Patterns

### Entry node (no inputs)
Set `input_ports = []`. The node fires immediately when execution starts.

### Async operations (HTTP, VLM calls)
```python
import httpx
async with httpx.AsyncClient() as client:
    resp = await client.post(url, json=payload)
```

### Module-level shared state
```python
_shared_data: dict[str, dict] = {}  # keyed by config value

async def forward(self, inputs, ctx):
    pad_id = self.config.get("pad_id", "default")
    pad = _shared_data.setdefault(pad_id, {})
```

### Voluntary logging
```python
self._self_log("prompt", assembled_prompt)
self._self_log("token_count", 1234)
```

### Dynamic ports (advanced)
Override `_resolve_ports()` classmethod — see `example.py` FlexibleMergeNode.
For sink-only / source-only nodes (boundary pivots like `iterIn`, `iterOut`), set the `ports_mode` ClassVar so the resolver, validator, and frontend agree on which side `config.ports` populates: `"sink"` (input-only) · `"source"` (output-only) · `"input"` (inputs from config, outputs from class) · `"mirror"` (both sides mirror config). Most authors don't need this.

### LIST[T] consumer ports (fan-in / multi-value)
Any wire type `T` can be wrapped as `LIST[T]` **on a consumer port** (ADR-dataflow-005). Producers stay scalar; the executor wraps single inputs to `[value]` and concatenates fan-in in edge declaration order at the port-binding seam. Used by multi-image LLM calls (`LLMCallNode.rgb: LIST[IMAGE]`), multi-LLM debate, and search-population fan-in. Nested `LIST[LIST[T]]` is not supported in v1. Example: `PortDef("frames", "LIST[IMAGE]", "All RGB frames this step")`.

## Checklist

1. [ ] `node_type` follows `{nodeset}__{node}` naming
2. [ ] `display_name` and `description` are set
3. [ ] `forward()` returns dict with keys matching `output_ports` names exactly
4. [ ] Input port names used in `forward()` match `input_ports` definitions
5. [ ] `from __future__ import annotations` at file top
6. [ ] `category` matches an existing sidebar group or is a new meaningful group
7. [ ] Node is returned from a `BaseNodeSet.get_tools()` (see `skill-nodeset.md`)
8. [ ] Wire types are valid: IMAGE, DEPTH, ACTION, POSE, TEXT, BOOL, METRICS, OBSERVATION, STEP_RESULT, ANY (10 inner types — `STATE` was renamed to `POSE` in ADR-dataflow-004). Consumer ports may also declare `LIST[T]` for fan-in / multi-value (ADR-dataflow-005).
9. [ ] No blocking I/O in forward — use `await` or `run_in_executor()` for heavy work
