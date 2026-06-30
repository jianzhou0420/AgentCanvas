# Skill: NodeSet

## When to Use

You have one or more canvas nodes that belong together — shared initialization,
shared shutdown, or just logical grouping. Every node must belong to a NodeSet
to be discoverable on the canvas.

## File Location & Naming

Placement, role directories, prefixes, and hierarchy rules live in
`.claude/standard/nodeset-layout.md` — read it first when creating, renaming,
or moving a file. Summary of the parts that matter while writing:

- **Role directory first**: every nodeset lives under one of
  `env/ method/ policy/ model/ common/ other/` — pick via the standard's
  decision table. Deployment (own conda env) is NOT a directory concern; it is
  the `server_python` ClassVar.
- **Single file**: `workspace/nodesets/{role}/{name}.py` — default; use when there's no support code.
- **Folder package**: `workspace/nodesets/{role}/{name}/__init__.py` — use when the nodeset has sidecars (wrapper around a vendored simulator, vendored model code, presets, tests, etc.). Sidecars live inside the folder; intra-package relative imports (`from ._wrapper import …`, `from .adapters import …`) keep coupling visible.

Either way:

- **`name` ClassVar**: short identifier **exactly equal to the file/folder stem, full role prefix included** (e.g. `model/model_sam.py` → `name = "model_sam"`; `env/env_libero/__init__.py` → `name = "env_libero"`). Pre-migration files that violate this are TODO #40 backlog, not precedent.
- **All node classes** live in the entry module (single .py or `__init__.py`), above the NodeSet class.
- **Module docstring**: include `Load: POST /api/components/nodesets/{name}/load`.

**When to choose folder over single file**: when you'd otherwise create a sibling `_xxx.py` or `_xxx/` next to the nodeset (wrapper, vendored tree, future preset configs). Folder form lets all of that live under one owner with relative imports — e.g. `libero/{__init__.py, _wrapper.py}`, `policy_vla/{__init__.py, adapters/, models/, policies/, presets/}`.

## Skeleton

```python
"""My Tools NodeSet — brief description.

Load:  POST /api/components/nodesets/my_tools/load
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef
from app.components.bases import ConfigField


# ── Nodes ──


class ToolOneNode(BaseCanvasNode):
    """First tool in this set."""

    node_type = "my_tools__tool_one"        # TODO: {nodeset}__{node}
    display_name = "Tool One"
    description = "Does the first thing"
    category = "tool"                       # TODO: sidebar group
    icon = "Wrench"

    input_ports: ClassVar[list] = [
        PortDef("input", "TEXT", "Input data"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("output", "TEXT", "Processed result"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="sky")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        data = inputs.get("input", "")
        return {"output": data}


# Add more node classes here...


# ── NodeSet Registration ──


class MyToolsNodeSet(BaseNodeSet):
    """My tools — brief description."""

    name = "my_tools"                       # TODO: must match filename
    description = "A set of useful tools"   # TODO: one-line summary

    def get_tools(self) -> list:
        return [
            ToolOneNode(),
            # TODO: add all node instances
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Optional: setup heavyweight resources (GPU, models, connections)."""
        pass

    async def shutdown(self) -> None:
        """Optional: release resources."""
        pass
```

## Required Fields

| Field | Type | Notes |
|-------|------|-------|
| `name` | `ClassVar[str]` | Unique nodeset identifier. Must match filename. |
| `get_tools()` | method | Must return **instances** (not classes): `[NodeA(), NodeB()]` |

## Optional Fields

| Field | Default | When to use |
|-------|---------|-------------|
| `description` | `""` | Always recommended — shown in NodeSet Manager UI |
| `server_python` | `None` | Set when nodes need a different Python env (see `skill-env-nodeset.md`) |
| `parallelism` | `"shared"` | Eval-time parallelism contract (ADR-server-003). `"shared"` = 1 instance, K callers may rendezvous through `BatchedInferenceServer` (right default for stateless tools). `"replicated"` = N tagged subprocesses, one per worker — required for stateful nodesets like envs. Tool/method nodesets should leave the default; env nodesets opt into `"replicated"` (covered in `skill-env-nodeset.md`). |
| `env_panel` | `None` | Set to a `BaseEnvPanel` subclass to register a control-plane panel for this nodeset (right-side props panel that exposes runtime state, e.g. episode picker, dataset switcher). When set, `WorkspaceComponentRegistry` instantiates and registers it on load. |
| `default_per_step_budget_sec` | `5.0` | Per-nodeset eval timeout knob (ADR-028). The batch runner clamps each episode at `max_steps * per_step_budget_sec`. Override on nodesets whose step latency diverges from the default — Habitat ~2.0, LLM-heavy method nodesets (e.g. MapGPT) ~30.0. |
| `initialize()` | no-op | GPU/model loading, connection setup |
| `shutdown()` | no-op | Resource cleanup, `torch.cuda.empty_cache()` |
| `get_eval_metadata()` | `{}` | Env nodesets only (see `skill-env-nodeset.md`) |

## Key Patterns

### Module-level shared state

When multiple nodes in a nodeset need to share data (scratch pads, caches):

```python
_scratch_pads: dict[str, dict] = {}

class WriteNode(BaseCanvasNode):
    async def forward(self, inputs, ctx):
        pad_id = self.config.get("pad_id", "default")
        pad = _scratch_pads.setdefault(pad_id, {})
        pad[inputs["key"]] = inputs["value"]
        return {"ok": True}
```

### Lazy loading vs eager loading

| Approach | Use when | Pattern |
|----------|----------|---------|
| **Eager** (in `initialize()`) | Resource is always needed, fast to load | `await self.load_model()` |
| **Lazy** (in first `forward()`) | Resource is expensive, may not be used | `if ctx.model is None: ctx.model = load()` |

### Parallelism in batched eval

Most tool/method nodesets are stateless — leave `parallelism` at the default
`"shared"`. Under `worker_count > 1` the framework runs a single instance
and routes K callers through `BatchedInferenceServer`. If the nodeset holds
per-episode state that cannot be cleanly partitioned (env scene state,
simulator handles, RNN buffers tied to a specific episode), opt into
`"replicated"` and read `skill-env-nodeset.md` for the full contract — the
framework will then spawn N tagged subprocesses, one per worker.

### Loading a nodeset at runtime

```
# Local mode (same process):
POST /api/components/nodesets/{name}/load

# Server mode (separate process, different Python):
POST /api/components/nodesets/{name}/load?mode=server
```

## Checklist

1. [ ] Exactly one `BaseNodeSet` subclass per nodeset (single .py file, or `__init__.py` of a folder package)
2. [ ] `name` matches the file/folder stem (e.g. `sam.py` → `name = "sam"`; `libero/__init__.py` → `name = "env_libero"`)
3. [ ] All node `node_type` values start with `{nodeset_name}__`
4. [ ] `get_tools()` returns instances, not classes
5. [ ] `shutdown()` cleans up any module-level state or GPU memory
6. [ ] Module docstring includes `Load: POST /api/components/nodesets/{name}/load`
7. [ ] Nodeset lives under `workspace/nodesets/` (single file or folder), not elsewhere — sidecars (`_wrapper.py`, vendored subtrees, presets) go *inside* the folder, not as `_xxx.py` siblings
8. [ ] `from __future__ import annotations` at file top
9. [ ] `parallelism` left at default `"shared"` for stateless tool/method nodesets; set to `"replicated"` only for stateful env-style nodesets
10. [ ] `default_per_step_budget_sec` overridden if step latency genuinely diverges from the 5.0s framework default (e.g. LLM-heavy nodesets)
