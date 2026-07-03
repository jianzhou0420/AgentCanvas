# Skill: Environment NodeSet

## When to Use

You're wrapping a simulator or interactive environment (Habitat, AI2-THOR,
Matterport3D, a custom gym-like env) that provides reset/step semantics
and usually needs its own process or Python interpreter.

> **Canonical contract**: `docs/pages/developer-guide/nodesets/env/template.html`
> (Developer Guide → Nodesets → Env → Template). This tutorial summarizes it;
> on any conflict the template wins. The pre-2026-06 obs-bundle contract
> (reset/step returning `observation`/`pose`/`done` directly) is **gone** —
> don't copy it from old graphs or old versions of this file.

## Classify First

`env/` holds three interface types (template §1). This tutorial covers only
the first:

| Type | Test | Verb surface |
|---|---|---|
| **Interactive MDP** (this tutorial) | has episode lifecycle AND an action loop | `reset / step_* / observe_* / evaluate` |
| Replay benchmark | lifecycle but no action loop (pre-recorded data) | `reset / episode_info / … / emit_metrics` — see `env_openeqa_em` |
| Stateless service | neither | plain request→response tools; belongs in `model/` or `method/` — don't add new ones to `env/` |

## File Location & Naming

Hierarchy and naming rules live in `.claude/standard/nodeset-layout.md` —
read it first. Env-specific summary:

- **Directory**: every env nodeset lives in `workspace/nodesets/env/`.
  Needing a non-default Python interpreter (e.g. `ac-vlnce` for habitat-sim
  0.1.7) is expressed by the `server_python` ClassVar, not by the directory —
  almost every env nodeset sets it.
- **Single file**: `workspace/nodesets/env/env_{sim}.py`; **folder package**:
  `workspace/nodesets/env/env_{sim}/__init__.py` when the env has sidecars
  (vendored simulator wrapper, preset configs, renderers, tests, etc.).
  All sidecars live inside the folder; use intra-package relative imports
  (`from ._wrapper import …`).
- **Nodeset name**: `env_{simulator}` (e.g. `"env_habitat"`, `"env_mp3d"`), and
  the file/folder stem equals the name (`env/env_libero/`, `env/env_mp3d/`).
- **Node types**: `env_{simulator}__{verb}_{space}` where the verb families
  and space suffixes come from the template's vocabulary (§4) — e.g.
  `env_habitat__step_discrete`, `env_libero__observe_objects`.

## The Contract: four verbs, pull perception

`step` is a pure transition that returns **no observation**; perception is
**agent-pulled** on demand via `observe_*`. The split is deliberate:
observations are parametric and sometimes expensive (a panorama must not be
rendered every step just because gym's `step` would return one).

| Verb | Kind | Returns |
|---|---|---|
| `reset` | lifecycle | **episode metadata only** — `instruction`/`question`, `episode_id`, env-specific ids. **No observation, no rgb/depth** — first frame comes from an `observe_*` call |
| `step_<actionspace>` | transition | `reward:ANY` · `terminated:BOOL` · `truncated:BOOL` · `info:ANY` — control signals only |
| `observe_<obsspace>` | perception (pull) | the obs-space payload from the template §4.2 (idempotent read; never advances the env) |
| `evaluate` | metric sink | `metrics:METRICS` (+ env-specific summary ports); fires once in the after-loop band |

plus suite-level `close` owned by the manager/env panel — **never a graph
node** (SIMPLER must never close per-episode: SAPIEN GC segfault).

Rules that trip people up:

- **reset is an idempotent ensure-live.** A done episode (canvas re-run —
  reset fires in the pre-loop band) is re-armed *in place*; a live one — the
  batch-eval path, where the runner has just placed a fresh episode via
  `set_episode` — is **read without disturbance** (no rebuild). Reset never
  *chooses* an episode; placement is env-panel-owned.
- **Spaces are naming axes, not config.** `step_discrete`, `step_waypoint`,
  `step_pose` (nav target:POSE), `step_hightolow`, `step_continuous`,
  `step_ee_pose` (manipulation) / `observe_egocentric`, `observe_panorama`,
  `observe_navigable`, `observe_objects`, `observe_frames`. **Reuse an
  existing suffix before inventing one** — the same suffix must carry the
  same port shape on every env (that's the whole point: a method written
  against one env finds the same verbs elsewhere). A new suffix is a
  vocabulary change every future env inherits — add it to template §4 in the
  same commit.
- **The four step ports are a floor, not a ceiling.** Env-specific extra
  outputs may sit alongside (mirrored inside `info`) — e.g. `success` +
  `step_index` on the manipulation envs. Agents wiring extras accept env
  coupling.
- **Rollover never lives in `observe_*`.** Observing a finished episode
  returns the terminal frame unchanged — an auto-reset inside observe can
  silently roll the env into a new episode under `evaluate` (removed
  2026-06-11 for exactly that bug).
- **Wire types**: no `FLOAT`/`DICT`/`VECTOR` in the registry — scalars and
  dicts ride `ANY`; continuous actions ride `TEXT` as JSON (single 7-vec or
  runtime-variable K-step chunk).

### Required env panel (`BaseEnvPanel`)

Every env nodeset declares a `BaseEnvPanel` subclass in the same file
and wires it via `MyNodeSet.env_panel = MyEnvPanel`. The env panel
owns episode selection, splits, and run lifecycle buttons — there are
**no** `set_episode` / `list_episodes` canvas nodes.

**Required fields**: a placement cascade ending in `episode_index` —
typically `split` + `episode_index`; hierarchical benchmarks may cascade
deeper (LIBERO: `suite → task_id → episode_index`). If the cascade head
isn't literally named `split`, also accept a `split` field-change as an
alias for the head so eval harnesses can target it (see
`LiberoEnvPanel.on_field_change`).

**Required actions**:

| Name | `side_effect` | Purpose |
|---|---|---|
| `play`  | `"run_start"` | re-seat the selected episode via `mgr.set_episode(...)`, then start the run |
| `pause` | `"run_pause"` | pause the running executor |
| `stop`  | `"run_stop"`  | stop the running executor |
| `reset` | `"signal"` | call `mgr.set_episode(...)`, emit `episode_reset` |

A panel response carries **one** side effect — play returns `run_start` and
does *not* also emit a signal; the `episode_reset` signal comes from field
changes and the reset action.

**Required hooks** (from `BaseEnvPanel`):

- `on_load()` — return `{split, episode_index, episode_count, splits, current_episode, step_budget, ...}`. `step_budget` is the per-episode iteration cap and is read by the framework's eval-batch resolver after every episode reset; populate it per episode when the env's natural budget is scene-adaptive (e.g. HM-EQA's `int(sqrt(scene_size)*3)`), and as a static ceiling otherwise.
- `on_field_change(name, value)` — update the cascade, push `set_episode`,
  emit `side_effect="signal"` with `signal_name="episode_reset"` so
  `lifetime="episode"` state containers clear
- `on_action(name, params)` — route buttons to side effects as in the table
- `get_options(field)` — return `[{"value": ..., "label": ...}]` per cascade
  field, sourced from the manager

### Parallelism contract (ADR-server-003)

Every env nodeset MUST declare its parallelism mode explicitly:

```python
class EnvNodeSet(BaseNodeSet):
    parallelism: ClassVar[str] = "replicated"   # for stateful sims
```

| Mode | Semantics under `worker_count > 1` | Use for |
|---|---|---|
| `"replicated"` | `WorkspaceComponentRegistry` spawns N tagged subprocesses (`{name}#0` … `{name}#N-1`); `EnvWorkerPool` hands each `LoopRunner` its own `env_panel_overrides` + `server_url_overrides`. Per-worker scene + agent pose are isolated. | **All env nodesets shipped today**. Any sim that holds mutable scene/episode/pose state. |
| `"shared"` (default) | One subprocess; K callers coalesce through `BatchedInferenceServer`. Pure-functional contract — no per-call state allowed. | LLM/policy/perception nodesets that are stateless across calls (e.g. `policy_cma`'s `forward`). |

**`worker_count = 1` is bit-identical in both modes** — the contract only kicks in under multi-worker batch eval. Forgetting `parallelism = "replicated"` on an env nodeset is silent at single-worker eval and at canvas Play, then explodes (random scene state, wrong SPL, episodes from the wrong scan) the moment someone runs `worker_count = 4`.

The mode follows the **wrapper**, not the upstream library: MatterSim supports `setBatchSize(K)` upstream, but our `env_mp3d` wrapper is single-batch + thread-affine, so it stays `replicated`. Decide based on what your `EnvManager` actually does, not what the underlying SDK could theoretically do.

### Optional nodes — env-specific extras

Env-specific capability nodes beyond the four verbs are the author's call
(naming `{env}__{verb}_{noun}`, `category = "environment"`). Prefer
expressing them inside the vocabulary (a new observe space beats a bespoke
getter). Sim-mutating helpers outside the `step_*` family (e.g.
`env_libero__reset_to_home`, `env_libero__close_gripper`) are debug-tier:
allowed, but no graph should need them — agents using any extra accept env
coupling.

---

## Skeleton

```python
"""Env{Name}NodeSet — {Simulator} environment as a NodeSet.

Gym-like interface (template.html): reset (metadata only) /
step_<actionspace> (control signals) / observe_<obsspace> (pull) / evaluate.

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
    conda_env_python,
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
        self._done = False
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
        """Build sim config. Cheap; the env itself opens on set_episode."""

    def shutdown(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── Episode control (backs env panel + reset) ──

    def list_splits(self) -> list[str]: ...
    def list_episodes(self, split: str) -> list[dict]: ...

    def set_episode(self, split: str, index: int) -> dict:
        """Place + arm an episode. Returns the metadata bundle."""
        # TODO: (re)build env for the episode; reset counters; return metadata

    def ensure_live(self) -> dict:
        """Template §5.1 reset semantics: live episode → read untouched;
        done episode → re-arm the SAME placement in place. Never chooses."""
        # if self._env is not None and not self._done: return metadata
        # else: return self.set_episode(<current placement>)

    # ── Transition + perception (back step_* / observe_* nodes) ──

    def step(self, action: str) -> dict:
        """Advance the env. Returns control signals only:
        {reward, terminated, truncated, info}."""

    def observe(self) -> dict:
        """Idempotent read of the current frame (obs-space payload)."""

    # ── Metric sink (backs evaluate) ──

    def evaluate(self) -> dict: ...


def _mgr() -> EnvManager:
    return EnvManager.get()


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(
        _mgr().executor, fn, *args,
    )


# ── Required canvas nodes: the four verbs ─────────────────────────────

class ResetNode(BaseCanvasNode):
    node_type = "env_{name}__reset"                # TODO
    display_name = "{Name}: Reset"
    description = "Ensure a live episode (re-arm if done) — metadata only"
    category = "environment"
    icon = "RotateCcw"

    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports: ClassVar[list] = [
        # env-specific metadata — VLN: instruction/episode_id/scene_id;
        # EQA: question/answer_gt/…; manipulation: suite/task_id/max_steps.
        # NO observation ports here.
        PortDef("instruction", "TEXT", "NL task instruction"),
        PortDef("episode_id",  "TEXT", "Episode identifier"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        meta = await _run(_mgr().ensure_live)
        return {
            "instruction": meta.get("instruction", ""),
            "episode_id":  str(meta.get("episode_id", "")),
        }


class StepNode(BaseCanvasNode):
    node_type = "env_{name}__step_{actionspace}"   # TODO: pick from template §4.1
    display_name = "{Name}: Step ({actionspace})"
    description = "Execute action; control signals only (pull obs via observe_*)"
    category = "environment"
    icon = "Play"

    input_ports: ClassVar[list] = [
        # shape per action space: discrete action:ACTION · waypoint
        # viewpoint_id:TEXT · pose target:POSE · continuous/ee_pose action:TEXT(JSON)
        PortDef("action", "TEXT", "Action, env-native JSON"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("reward",     "ANY",  "Per-step reward (scalar)"),
        PortDef("terminated", "BOOL", "MDP terminal"),
        PortDef("truncated",  "BOOL", "Budget / step-limit cutoff"),
        PortDef("info",       "ANY",  "Diagnostics + terminal metrics"),
        # optional env-specific extras (also mirrored inside info)
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().step, str(inputs["action"]))
        self._self_log("terminated", result.get("terminated"))
        return result


class ObserveEgocentricNode(BaseCanvasNode):
    node_type = "env_{name}__observe_egocentric"   # TODO: pick from template §4.2
    display_name = "{Name}: Observe (egocentric)"
    description = "Pull the current frame (read-only, no env step)"
    category = "environment"
    icon = "Eye"

    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports: ClassVar[list] = [
        # payload per obs space (template §4.2); egocentric:
        PortDef("rgb",        "IMAGE", "First-person RGB"),
        PortDef("depth",      "DEPTH", "Depth (None if the sim doesn't render it)"),
        PortDef("pose",       "POSE",  "Agent pose (None if env has no pose)"),
        PortDef("intrinsics", "ANY",   "Camera intrinsics (None if unavailable)"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return await _run(_mgr().observe)


class EvaluateNode(BaseCanvasNode):
    node_type = "env_{name}__evaluate"
    display_name = "{Name}: Evaluate"
    description = "Post-hoc metrics sink (fires in the after-loop band)"
    category = "environment"
    icon = "CheckCircle"

    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports: ClassVar[list] = [
        PortDef("metrics", "METRICS", "Episode metrics dict"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {"metrics": await _run(_mgr().evaluate)}


# ── Required env panel ────────────────────────────────────────────────

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
        if name in ("play", "reset"):
            result = await _run(
                _mgr().set_episode,
                self._state["split"],
                int(self._state["episode_index"]),
            )
            if name == "play":
                return {"ok": True, "side_effect": "run_start"}
            if result.get("ok", True):
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
        if name in ("pause", "stop"):
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


# ── NodeSet Registration ──────────────────────────────────────────────

class EnvNodeSet(BaseNodeSet):
    name = "env_{name}"                            # TODO
    description = "{Simulator} environment"
    # Dedicated conda env + env-var override, so CI / other machines can
    # repoint without editing source (all shipped env nodesets do this).
    server_python = conda_env_python("ac-{name}", "{NAME}_PYTHON")  # TODO
    env_panel = MyEnvPanel                         # Required
    # ADR-server-003: stateful simulators MUST be "replicated" so each batch
    # eval worker gets its own tagged subprocess. Default is "shared" (wrong
    # for env nodesets — would coalesce K workers into one sim, corrupting
    # per-worker scene + agent pose).
    parallelism: ClassVar[str] = "replicated"      # Required
    # ADR-eval-002: BatchEvalRunner caps each episode at
    # ``max_steps * default_per_step_budget_sec``. Tune to actual step
    # latency: Habitat ~2.0, MP3D ~5.0 (default), HM-EQA ~5.0, LIBERO 30.0,
    # OpenEQA-LLM-judge ~90.0, framework default 5.0.
    default_per_step_budget_sec: ClassVar[float] = 5.0  # TODO: tune

    def get_tools(self) -> list:
        return [
            ResetNode(),
            StepNode(),               # one per supported action space
            ObserveEgocentricNode(),  # one per supported obs space
            EvaluateNode(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        await _run(lambda: _mgr().initialize(**kwargs))

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

Simulators (Habitat, AI2-THOR, MatterSim, robosuite/MuJoCo) have GL/physics
thread affinity — they must run on the thread that created them.
Single-thread executor:

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
3. [ ] `set_episode(...)` places + arms an episode and resets counters
4. [ ] `ensure_live()` — live episode read untouched, done episode re-armed in place (never chooses)
5. [ ] `list_splits()`, `list_episodes(split)` for the env panel cascade

### Required canvas nodes (the four verbs)

6. [ ] `{env}__reset` — metadata-only outputs (no observation ports), forwards to `ensure_live()`
7. [ ] one `{env}__step_<x>` per supported action space — suffix + input shape from template §4.1; outputs `reward`/`terminated`/`truncated`/`info` (extras allowed alongside, mirrored in `info`)
8. [ ] one `{env}__observe_<y>` per supported obs space — suffix + payload from template §4.2; idempotent, no lifecycle action, no auto-reset
9. [ ] `{env}__evaluate` — thin metric sink for the after-loop band
10. [ ] new space suffixes (if truly unavoidable) added to template §4 in the same commit

### Required env panel

11. [ ] `BaseEnvPanel` subclass with a placement cascade ending in `episode_index` (accept `split` as an alias for a non-`split` cascade head)
12. [ ] `play` / `pause` / `stop` / `reset` actions; play re-seats via `set_episode` then `run_start`
13. [ ] `on_field_change` pushes `set_episode` and emits the `episode_reset` signal
14. [ ] `on_action("reset")` calls `mgr.set_episode()` and emits `episode_reset`
15. [ ] `get_options(field)` returns dynamic option lists per cascade field
16. [ ] `NodeSet.env_panel = MyEnvPanel` declared at class level

### Parallelism + scheduling ClassVars

17. [ ] `parallelism = "replicated"` declared on the NodeSet class (ADR-server-003) — required for any sim with mutable scene/episode/pose state
18. [ ] `default_per_step_budget_sec` tuned to actual step latency (ADR-eval-002)
19. [ ] `server_python = conda_env_python("ac-{name}", "{NAME}_PYTHON")` — dedicated conda env with env-var override

### Lifecycle + server mode

20. [ ] `get_eval_metadata()` returns dict with splits, metrics, counts, step_budget
21. [ ] Blocking simulator calls wrapped in `run_in_executor`
22. [ ] Singleton manager pattern with class-level `_instance`
23. [ ] suite-level `close` stays manager/panel-owned — no close node, no per-episode close (SIMPLER segfaults)

---

## Deep Dive

- **The contract itself**: `docs/pages/developer-guide/nodesets/env/template.html` (verbs, vocabulary, return contracts, migration record)
- Env panel contract: `agentcanvas/backend/app/components/env_panel.py` (`BaseEnvPanel`, `RemoteEnvPanelProxy`)
- Real examples (single-file): `workspace/nodesets/env/env_habitat.py`, `workspace/nodesets/env/env_openeqa_em.py`
- Real examples (folder): `workspace/nodesets/env/env_libero/`, `workspace/nodesets/env/env_simpler/` (each `__init__.py` + `_wrapper.py` sidecar), `workspace/nodesets/env/env_mp3d/`, `workspace/nodesets/env/env_hmeqa/`
- Signal system (`episode_reset`, `step_end`, `run_end`): `agentcanvas/backend/app/agent_loop/state_containers.py` (`LIFETIME_TO_SIGNALS`)
- ADR-server-002 (generic BaseEnvPanel contract): `docs/pages/developer-guide/core/decisions/server/adr-server-002-base-controller.html`
- ADR-server-003 (env parallelism contract — `replicated` vs `shared`): `docs/pages/developer-guide/core/decisions/server/adr-server-003-env-parallelism-contract.html`
- ADR-eval-002 (worker pool + batched inference + `default_per_step_budget_sec`): `docs/pages/developer-guide/core/decisions/eval/adr-eval-002-worker-pool-and-batched-inference.html`
