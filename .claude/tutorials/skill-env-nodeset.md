# Skill: Environment NodeSet

## When to Use

You're wrapping a simulator or interactive environment (Habitat, AI2-THOR,
Matterport3D, a custom gym-like env) that provides reset/step semantics
and usually needs its own process or Python interpreter.

## File Location & Naming

Hierarchy and naming rules live in `.claude/standard/nodeset-layout.md` —
read it first. Env-specific summary:

- **Directory**: every env nodeset lives in `workspace/nodesets/env/`.
  Needing a non-default Python interpreter (e.g. `vlnce` for habitat-sim
  0.1.7) is expressed by the `server_python` ClassVar, not by the directory —
  almost every env nodeset sets it.
- **Single file**: `workspace/nodesets/env/env_{sim}.py`; **folder package**:
  `workspace/nodesets/env/env_{sim}/__init__.py` when the env has sidecars
  (vendored simulator wrapper, preset configs, renderers, tests, etc.).
  All sidecars live inside the folder; use intra-package relative imports
  (`from ._wrapper import …`).
- **Nodeset name**: `env_{simulator}` (e.g. `"env_habitat"`, `"env_mp3d"`), and
  the file/folder stem equals the name (`env/env_libero/`, `env/env_mp3d.py`).
  Pre-migration files without the prefix (`habitat.py`, `libero/`, …) are
  TODO #40 backlog, not precedent.
- **Node types**: `env_{simulator}__{verb}_{noun}` (e.g. `"env_habitat__step"`,
  `"env_mp3d__render_panorama"`).

## Three-Tier Contract

Every env nodeset ships functionality in three tiers. The tiers exist so
that agent graphs can be written against Tier 1 and still run on a second
env with only a `node_type` rename — no edge surgery.

| Tier | What lives here | Enforcement |
|---|---|---|
| **Required** | `reset` + `step` canvas nodes **and** a `BaseEnvPanel` subclass | Missing any of these fails the contract — agent graphs can't be portable |
| **Recommended** | `evaluate`, `get_observation` canvas nodes | Skip only if genuinely not applicable |
| **Optional** | env-specific capabilities (`render_panorama`, `get_gps`, …) | Author's call; agents using these ports accept env coupling |

**Episode management lives only on the env panel**, not as canvas nodes.
`BatchEvalRunner`, the episode dropdown, and manual resets all route
through the same `BaseEnvPanel.on_action("reset", ...)` path, which
calls `EnvManager.set_episode(split, idx)` internally.

---

### Tier 1a: Required canvas nodes (`reset` + `step`)

#### `{env}__reset` — fire-once at episode start, returns initial bundle

| Port | Direction | Wire type | Required? | Notes |
|---|---|---|---|---|
| `trigger` | in  | `ANY`        | optional | explicit firing from upstream; Initialize already fires once per run |
| `instruction` | out | `TEXT`       | ✅ | natural-language task string |
| `episode_id`  | out | `TEXT`       | ✅ | for eval / logging / replay |
| `observation` | out | `LIST[IMAGE]`| ✅ | continuous env emits `[rgb]`; discrete env emits multi-view list |
| `pose`        | out | `POSE`       | ✅ | may be `None` if the env has no continuous pose (graph envs like MP3D node-graph nav). **Don't fake it with a zeroed dict** — downstream would mistake the fake for real data and silently compute wrong distances / wrong maps. Downstream nodes consuming `pose` must explicitly handle `None`. |

#### `{env}__step` — executes an action, returns next bundle + `done`

| Port | Direction | Wire type | Required? | Notes |
|---|---|---|---|---|
| `action`      | in  | `TEXT`       | ✅ | env-native serialization (vp id, action index, JSON vector) — see TODO #46 for unified `action_manifest` contract |
| `instruction` | out | `TEXT`       | ✅ | bundle reuse — same port as reset |
| `episode_id`  | out | `TEXT`       | ✅ | bundle reuse |
| `observation` | out | `LIST[IMAGE]`| ✅ | bundle reuse |
| `pose`        | out | `POSE`       | ✅ | bundle reuse; same `None`-allowed rule as reset (graph envs emit `None`, never a fake zeroed dict) |
| `done`        | out | `BOOL`       | ✅ | `True` on terminal step (STOP / max-steps / goal-reached), `False` otherwise |

**Extras are allowed alongside required outputs.** A nodeset MAY add
env-specific extra output ports next to the required bundle (e.g. MP3D's
`viewpoint_id` / `heading` / `navigable_json` / `directions`). Agents
wanting cross-env portability connect only to the required ports;
agents that want env-specific detail connect to the extras. The contract
is a floor on what every env must offer, not a ceiling.

**Action port note (contract v2 coming)**: today we accept env-native shapes
as TEXT — MP3D sends vp ids, Habitat-continuous sends action indices or
JSON vectors. The unified `action_manifest` contract (roadmap TODO #46)
will standardize this once a third env nodeset lands so agents stop
depending on env-specific parsing. **Don't prematurely standardize.**

---

### Tier 1b: Required env panel (`BaseEnvPanel`)

Every env nodeset declares a `BaseEnvPanel` subclass in the same file
and wires it via `MyNodeSet.env_panel = MyEnvPanel`. The env panel
owns episode selection, splits, and run lifecycle buttons — there are
**no** `set_episode` / `list_episodes` canvas nodes.

**Required fields**:

| Name | Kind | Purpose |
|---|---|---|
| `split` | `select` | dataset split (`val_unseen`, `val_seen`, …); options via `get_options("split")` |
| `episode_index` | `select` | episode within split; options via `get_options("episode_index")` |

**Required actions**:

| Name | `side_effect` | Purpose |
|---|---|---|
| `play`  | `"run_start"` | start a run at the selected episode |
| `pause` | `"run_pause"` | pause the running executor |
| `stop`  | `"run_stop"`  | stop the running executor |
| `reset` | `"signal"` (or `"run_start"` if combined with play) | call `mgr.set_episode(split, idx)`, emit `episode_reset` |

**Required hooks** (from `BaseEnvPanel`):

- `on_load()` — return `{split, episode_index, episode_count, splits, current_episode, step_budget, ...}`. `step_budget` is the per-episode iteration cap and is read by the framework's eval-batch resolver after every episode reset; populate it per episode when the env's natural budget is scene-adaptive (e.g. HM-EQA's `int(sqrt(scene_size)*3)`), and as a static ceiling otherwise.
- `on_field_change(name, value)` — for `split` / `episode_index`, emit
  `side_effect="signal"` with `signal_name="episode_reset"` so
  `lifetime="episode"` state containers clear
- `on_action(name, params)` — route buttons to side effects; `reset`
  calls `mgr.set_episode(...)` then emits `episode_reset`
- `get_options(field)` — return `[{"value": ..., "label": ...}]` for
  `split` and `episode_index`, sourced from the manager

---

### Tier 1c: Parallelism contract (ADR-server-003)

Every env nodeset MUST declare its parallelism mode explicitly:

```python
class EnvNodeSet(BaseNodeSet):
    parallelism: ClassVar[str] = "replicated"   # for stateful sims
```

| Mode | Semantics under `worker_count > 1` | Use for |
|---|---|---|
| `"replicated"` | `WorkspaceComponentRegistry` spawns N tagged subprocesses (`{name}#0` … `{name}#N-1`); `EnvWorkerPool` hands each `LoopRunner` its own `env_panel_overrides` + `server_url_overrides`. Per-worker scene + agent pose are isolated. | **All env nodesets shipped today** (Habitat, MP3D, HM-EQA, OpenEQA). Any sim that holds mutable scene/episode/pose state. |
| `"shared"` (default) | One subprocess; K callers coalesce through `BatchedInferenceServer`. Pure-functional contract — no per-call state allowed. | LLM/policy/perception nodesets that are stateless across calls (e.g. `policy_cma`'s `forward`). |

**`worker_count = 1` is bit-identical in both modes** — the contract only kicks in under multi-worker batch eval. Forgetting `parallelism = "replicated"` on an env nodeset is silent at single-worker eval and at canvas Play, then explodes (random scene state, wrong SPL, episodes from the wrong scan) the moment someone runs `worker_count = 4`.

The mode follows the **wrapper**, not the upstream library: MatterSim supports `setBatchSize(K)` upstream, but our `env_mp3d` wrapper is single-batch + thread-affine, so it stays `replicated`. Decide based on what your `EnvManager` actually does, not what the underlying SDK could theoretically do.

---

### Tier 2: Recommended nodes

| Node | Inputs | Outputs | Purpose |
|---|---|---|---|
| `{env}__evaluate` | `trigger` (any) | `metrics` (`METRICS`) | SPL / NDTW / success at episode end |
| `{env}__get_observation` | `trigger` (any) | same as `reset` (minus `instruction`/`episode_id` optional) | query current obs without stepping — UI preview / debug |

Skip only if the env genuinely can't provide them (pure sandbox with no
metrics, non-idempotent observation).

---

### Tier 3: Optional nodes — env-specific capabilities

Two soft constraints (no schema enforcement):

1. **Naming**: `{env}__{verb}_{noun}` (e.g. `env_mp3d__render_panorama`, `env_habitat__get_gps_compass`).
2. **Category**: `category = "environment"` for sidebar grouping.

Everything else — port shapes, seed vs non-seed, whether it reads state —
is the author's call. Agents using these ports accept the env-specific
coupling.

Examples:

- `env_mp3d__render_panorama(n_views)` — discrete-env skybox stitching
- `env_mp3d__get_nav_graph` — graph-structured env only
- `env_habitat__get_gps_compass` — continuous env only
- `env_habitat__panorama_rgbd` — Habitat-specific viewport synthesis
- `env_ai2thor__pick_up_object` — manipulation env only

---

## Skeleton

```python
"""Env{Name}NodeSet — {Simulator} environment as a NodeSet.

Works in-process or as an auto-hosted server:
  Local:  POST /api/components/nodesets/env_{name}/load
  Server: POST /api/components/nodesets/env_{name}/load?mode=server
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import Any, ClassVar

from app.components import (
    BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef,
)
from app.components.env_panel import (
    BaseEnvPanel, EnvPanelAction, EnvPanelField,
)

log = logging.getLogger("agentcanvas.env_{name}")


# ── EnvManager — singleton simulator runtime ──────────────────────────

class EnvManager:
    """Single simulator instance. All public methods are blocking — call
    via ``asyncio.get_running_loop().run_in_executor(mgr.executor, fn)``.
    Single-thread executor enforces GL/physics thread affinity.
    """

    _instance: EnvManager | None = None

    def __init__(self) -> None:
        self._env = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="env",
        )

    @classmethod
    def get(cls) -> EnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self._executor

    # ── Lifecycle (called by NodeSet, not exposed as canvas nodes) ──

    def initialize(self, **kwargs: Any) -> None:
        """Build sim config, create env. Does NOT load an episode yet."""
        # TODO: import simulator; build env; store in self._env

    def shutdown(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── Required (backs reset + step canvas nodes) ──

    def reset(self, **kwargs: Any) -> dict:
        """Reset to current episode start. Returns the observation bundle.

        Returns: {instruction, episode_id, observation, pose}  (pose may be None)
        """
        # TODO: self._env.reset() and pack bundle

    def step(self, action: str) -> dict:
        """Execute action. Returns bundle + done flag.

        Returns: {instruction, episode_id, observation, pose, done}
        """
        # TODO: parse action; advance; pack bundle with done

    # ── Required (backs env panel, NOT canvas nodes) ──

    def list_splits(self) -> list[str]: ...
    def list_episodes(self, split: str) -> list[dict]: ...
    def set_episode(self, split: str, index: int) -> dict: ...

    # ── Recommended (backs evaluate / get_observation canvas nodes) ──

    def evaluate(self) -> dict: ...
    def get_observation(self) -> dict: ...


def _mgr() -> EnvManager:
    return EnvManager.get()


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(
        _mgr().executor, fn, *args,
    )


# ── Tier 1a: Required canvas nodes ────────────────────────────────────

class ResetNode(BaseCanvasNode):
    node_type = "env_{name}__reset"                # TODO
    display_name = "{Name}: Reset"
    description = "Reset to episode start; return initial observation bundle"
    category = "environment"
    icon = "RotateCcw"

    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Optional fire trigger"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("instruction", "TEXT", "NL task instruction"),
        PortDef("episode_id",  "TEXT", "Episode identifier"),
        PortDef("observation", "LIST[IMAGE]", "Initial visual observation"),
        PortDef("pose",        "POSE", "Initial agent pose (None if unavailable)"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        obs = await _run(_mgr().reset)
        return {
            "instruction": obs["instruction"],
            "episode_id":  obs["episode_id"],
            "observation": obs["observation"],      # list of images
            "pose":        obs.get("pose"),         # POSE or None
        }


class StepNode(BaseCanvasNode):
    node_type = "env_{name}__step"                 # TODO
    display_name = "{Name}: Step"
    description = "Execute action; return next observation bundle + done"
    category = "environment"
    icon = "Play"

    input_ports: ClassVar[list] = [
        PortDef("action", "TEXT", "Env-native action (vp id, index, or JSON)"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("instruction", "TEXT"),
        PortDef("episode_id",  "TEXT"),
        PortDef("observation", "LIST[IMAGE]"),
        PortDef("pose",        "POSE"),
        PortDef("done",        "BOOL", "True on terminal step"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        action = str(inputs["action"])
        obs = await _run(_mgr().step, action)
        self._self_log("action", action)
        self._self_log("done", obs.get("done", False))
        return obs


# ── Tier 1b: Required env panel ──────────────────────────────────────

class MyEnvPanel(BaseEnvPanel):
    name: ClassVar[str] = "env_{name}"             # TODO: match nodeset name
    display_name: ClassVar[str] = "{Name} Env"

    fields: ClassVar[list[EnvPanelField]] = [
        EnvPanelField("split", "select", "Split"),
        EnvPanelField("episode_index", "select", "Episode"),
    ]
    actions: ClassVar[list[EnvPanelAction]] = [
        EnvPanelAction("play",  "Play",  side_effect="run_start", enabled_when="idle"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop",  "Stop",  side_effect="run_stop",  enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none",      enabled_when="idle"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {"split": "", "episode_index": 0}

    async def on_load(self) -> dict[str, Any]:
        mgr = _mgr()
        splits = await _run(mgr.list_splits)
        split = self._state["split"] or (splits[0] if splits else "")
        episodes = await _run(mgr.list_episodes, split) if split else []
        return {
            "split": split,
            "episode_index": int(self._state.get("episode_index", 0)),
            "episode_count": len(episodes),
            "splits": splits,
            "step_budget": 30,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        self._state[name] = value if name == "split" else int(value)
        if name == "split":
            self._state["episode_index"] = 0
        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = {
            "split": self._state["split"],
            "episode_index": self._state["episode_index"],
        }
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name == "reset":
            result = await _run(
                _mgr().set_episode,
                self._state["split"],
                int(self._state["episode_index"]),
            )
            if result.get("ok"):
                return {
                    "ok": True,
                    "side_effect": "signal",
                    "signal_name": "episode_reset",
                    "signal_payload": {
                        "split": self._state["split"],
                        "episode_index": self._state["episode_index"],
                    },
                }
            return {"ok": False, "side_effect": "none", "error": result.get("error")}
        if name in ("play", "pause", "stop"):
            return {"ok": True, "side_effect": f"run_{name}"}
        return {"ok": False, "side_effect": "none", "error": f"Unknown action '{name}'"}

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        mgr = _mgr()
        if field == "split":
            splits = await _run(mgr.list_splits)
            return [{"value": s, "label": s} for s in splits]
        if field == "episode_index":
            episodes = await _run(mgr.list_episodes, self._state["split"])
            return [{"value": e["index"], "label": f'{e["index"]}: {e.get("scan", "")}'}
                    for e in episodes]
        return []


# ── Tier 2: Recommended nodes (evaluate, get_observation) ─────────────
# ── Tier 3: Optional nodes (env-specific) ─────────────────────────────
#   … define more BaseCanvasNode subclasses as needed.


# ── NodeSet Registration ──────────────────────────────────────────────

class EnvNodeSet(BaseNodeSet):
    name = "env_{name}"                            # TODO
    description = "{Simulator} environment"
    # TODO: pick env-var name (e.g. "VLNCE_PYTHON") + sensible default path.
    # All shipped env nodesets follow this pattern so CI / other machines
    # can override without editing source.
    server_python = os.environ.get(
        "{NAME}_PYTHON",                           # TODO: env var name
        "python",  # TODO: fallback
    )
    env_panel = MyEnvPanel                     # Required (Tier 1b)
    # ADR-server-003: stateful simulators MUST be "replicated" so each batch
    # eval worker gets its own tagged subprocess. Default is "shared" (wrong
    # for env nodesets — would coalesce K workers into one sim, corrupting
    # per-worker scene + agent pose).
    parallelism: ClassVar[str] = "replicated"      # Required (Tier 1c)
    # ADR-eval-002: BatchEvalRunner caps each episode at
    # ``max_steps * default_per_step_budget_sec``. Tune to actual step
    # latency: Habitat ~2.0, MP3D ~5.0 (default), HM-EQA ~5.0,
    # OpenEQA-LLM-judge ~90.0, framework default 5.0.
    default_per_step_budget_sec: ClassVar[float] = 5.0  # TODO: tune

    def get_tools(self) -> list:
        return [
            ResetNode(),        # Tier 1
            StepNode(),         # Tier 1
            # EvaluateNode(),   # Tier 2
            # GetObservationNode(),  # Tier 2
            # …optional env-specific tools
        ]

    async def initialize(self, **kwargs: Any) -> None:
        await _run(_mgr().initialize)

    async def shutdown(self) -> None:
        await _run(_mgr().shutdown)

    async def get_eval_metadata(self) -> dict:
        return {
            "env_name": "{name}",                  # TODO
            "splits": ["val_unseen", "val_seen"],  # TODO
            "episode_counts": {},
            "metrics": ["spl", "success", "ndtw"], # TODO
            "supports_set_episode": True,
            "step_budget": 30,
        }
```

## Key Patterns

### Singleton EnvManager

Simulators (Habitat, AI2-THOR, MatterSim) have GL/physics thread affinity —
they must run on the thread that created them. Single-thread executor:

```python
self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
```

### Blocking calls via `run_in_executor`

All simulator calls are blocking. Hop to the executor thread:

```python
result = await asyncio.get_running_loop().run_in_executor(
    mgr.executor, mgr.step, action,
)
```

### Server-mode auto-routing

When `server_python` is set and the nodeset loads with `?mode=server`,
`AutoServerApp` wraps it as a separate HTTP process. Env panels bridge
via `/env-panel/*` routes and `RemoteEnvPanelProxy` (ADR-server-002) — no
code changes needed; same nodeset works in both modes.

### `episode_reset` signal forwarding

When the user changes the split or episode in the panel, the env panel
returns `side_effect="signal"` with `signal_name="episode_reset"`. The
env panel router forwards that to the running executor, which fans it
to every state container; states declared with `lifetime="episode"`
clear automatically (see `LIFETIME_TO_SIGNALS` in `state_containers.py`).

---

## Checklist

### Manager

1. [ ] `EnvManager` singleton with single-thread `ThreadPoolExecutor(max_workers=1)`
2. [ ] `initialize()` / `shutdown()` for lifecycle (not exposed as nodes)
3. [ ] `reset()` returns bundle dict with `{instruction, episode_id, observation, pose}`
4. [ ] `step(action)` returns bundle dict + `done`
5. [ ] `list_splits()`, `list_episodes(split)`, `set_episode(split, idx)` on the manager (for env panel)

### Required canvas nodes (Tier 1a)

6. [ ] `{env}__reset` — 4 required outputs + optional `trigger` input
7. [ ] `{env}__step` — `action` input + 4 required outputs + `done`
8. [ ] `observation` port declared as `LIST[IMAGE]` (continuous env wraps with `[rgb]`)
9. [ ] `pose` port emits `None` if env has no continuous pose

### Required env panel (Tier 1b)

10. [ ] `BaseEnvPanel` subclass with `split` + `episode_index` fields
11. [ ] `play` / `pause` / `stop` / `reset` actions
12. [ ] `on_field_change` emits `episode_reset` signal on split/episode change
13. [ ] `on_action("reset")` calls `mgr.set_episode()` and emits `episode_reset`
14. [ ] `get_options(field)` returns dynamic split + episode lists
15. [ ] `NodeSet.env_panel = MyEnvPanel` declared at class level

### Parallelism contract (Tier 1c)

15a. [ ] `parallelism = "replicated"` declared on the NodeSet class (ADR-server-003) — required for any sim with mutable scene/episode/pose state. Skip only if the env is genuinely stateless.
15b. [ ] `default_per_step_budget_sec` tuned to actual step latency (ADR-eval-002) — controls batch eval per-episode timeout (`max_steps × budget`). Default 5.0 is wrong for slow sims (LLM-judge, RxR-CE).
15c. [ ] `server_python` reads from an env var with a fallback (`os.environ.get("XXX_PYTHON", "/fallback")`) — so CI / other machines can override without editing source.

### Recommended + Optional

16. [ ] `{env}__evaluate` node for metrics (Tier 2, skip only if impossible)
17. [ ] `{env}__get_observation` for UI preview (Tier 2, skip only if impossible)
18. [ ] Optional env-specific nodes follow `{env}__{verb}_{noun}` naming with `category="environment"`

### Lifecycle + server mode

19. [ ] `server_python` set if env needs a different Python interpreter
20. [ ] `get_eval_metadata()` returns dict with splits, metrics, counts
21. [ ] Blocking simulator calls wrapped in `run_in_executor`
22. [ ] Singleton manager pattern with class-level `_instance`

---

## Deep Dive

- Env panel contract: `agentcanvas/backend/app/components/env_panel.py` (`BaseEnvPanel`, `RemoteEnvPanelProxy`)
- Real examples (single-file): `workspace/nodesets/server/habitat.py`, `workspace/nodesets/server/matterport3d.py`, `workspace/nodesets/server/hmeqa.py`, `workspace/nodesets/server/openeqa.py`
- Real examples (folder): `workspace/nodesets/server/libero/`, `workspace/nodesets/server/simpler/` (each `__init__.py` + `_wrapper.py` sidecar), `workspace/nodesets/server/policy_vla/` (with vendored `adapters/`, `models/`, `policies/` subtrees)
- Signal system (`episode_reset`, `step_end`, `run_end`): `agentcanvas/backend/app/agent_loop/state_containers.py` (`LIFETIME_TO_SIGNALS`)
- ADR-server-002 (generic BaseEnvPanel contract): `docs/core/decisions/server/adr-server-002-base-env panel.md`
- ADR-server-003 (env parallelism contract — `replicated` vs `shared`): `docs/core/decisions/server/adr-server-003-env-parallelism-contract.md`
- ADR-eval-002 (worker pool + batched inference + `default_per_step_budget_sec`): `docs/core/decisions/eval/adr-eval-002-worker-pool-and-batched-inference.md`
- TODO #46 (unified `action_manifest` contract, deferred): `docs/core/roadmap.md`
