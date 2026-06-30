# Code Style

## Naming Conventions

### Python Files (backend)

**Pattern: `snake_case`, descriptive noun or noun phrase.**

Files should describe *what the module contains*, not what action it performs. Prefer `subject_role` compounds over bare verbs.

| Pattern | When to use | Examples |
|---|---|---|
| `subject_role.py` | Module defines a class/system with a clear role | `graph_executor.py`, `loop_runner.py`, `builtin_nodes.py`, `state_containers.py` |
| `bare_noun.py` | Module defines a single focused concept | `models.py`, `config.py`, `state.py`, `layout.py` |
| `domain_scope.py` | Module scoped to a specific domain area | `eval_batch.py`, `eval_storage.py`, `wire_types.py`, `node_io.py` |

**Avoid:**
- Bare verbs as filenames (`flatten.py` is grandfathered; prefer `graph_flattener.py` for new files)
- `_api` or `_handler` suffixes — the `api/` package provides that context
- `utils.py` or `helpers.py` — name by what the utilities *are*, not that they're utilities

**Package names:** short singular or compound nouns describing the domain: `agent_loop/`, `api/`, `llm/`, `server/`, `components/`, `standard/`.

**API sub-packages:** group by domain, bare noun filenames. The package provides the "API" context:
```
api/
  canvas/       → graphs.py, env.py
  execution/    → run.py, eval.py, logs.py, websocket.py
  platform/     → config.py, components.py, profiles.py
```

### Python Classes

| Type | Convention | Examples |
|---|---|---|
| Node classes | `PascalCase` ending in `Node` or `Sink` | `LLMCallNode`, `IterInNode`, `ObservationViewerSink` |
| Executor/runner classes | `PascalCase` describing role | `GraphExecutor`, `LoopRunner`, `BatchEvalRunner` |
| Data classes | `PascalCase` noun | `ExecutionContext`, `EvalConfig`, `EpisodeResult`, `NodeInstance` |
| Protocols/ABCs | `PascalCase`, prefixed with `Base` | `BaseCanvasNode`, `BaseNodeSet`, `BaseServer` |
| Enums | `PascalCase` noun | `EvalStatus`, `ExecutionMode` |

### Python Variables and Parameters

| Type | Convention | Examples |
|---|---|---|
| Runtime session handle | `session` | `GraphExecutor.run(session=...)` |
| Node handler registry | `NODE_HANDLERS` | Global `dict[str, type[BaseCanvasNode]]` |
| Config/settings | `snake_case` noun | `llm_config`, `eval_config`, `step_delay_ms` |
| Private/internal | `_prefix` | `_current_step`, `_flatten_map`, `_ws_connections` |
| Constants | `UPPER_SNAKE` | `ACTION_STOP`, `MEDIA_TYPES`, `GRAPHS_DIR` |

### TypeScript Files (frontend)

| Pattern | When to use | Examples |
|---|---|---|
| `camelCase.ts` | Utility modules, API clients, stores | `evalApi.ts`, `runPipeline.ts`, `graphConversion.ts`, `useFlowStore.ts` |
| `PascalCase.tsx` | React components | `EvalPage.tsx`, `CanvasPage.tsx`, `MetricCards.tsx` |
| `camelCase.ts` | Type definitions, config | `types.ts`, `edgeTypes.ts` |

### Terms to Keep Consistent

These terms have specific meanings in this project. Do not use synonyms:

| Term | Means | NOT |
|---|---|---|
| `session` | `LoopRunner` instance passed to executor at runtime | ~~orchestrator~~, ~~context~~ |
| `node` | A `BaseCanvasNode` subclass (canvas participant) | ~~handler~~, ~~processor~~ |
| `nodeset` | A `BaseNodeSet` (atomic group of tools/nodes) | ~~plugin~~, ~~module~~ |
| `graph` | A `GraphDefinition` (editable template, `kind="graph"`) | ~~pipeline~~, ~~workflow~~ |
| `graph node` | A frozen composite (`kind="node"`) | ~~subgraph~~, ~~template~~ |
| `flatten` | Recursive composite expansion before execution | ~~inline~~, ~~expand~~ |
| `wire type` | Port data type (TEXT, IMAGE, ACTION, STATE, etc.) | ~~data type~~, ~~port type~~ |
| `profile` | An LLM configuration (model + provider + params) | ~~config~~, ~~model~~ |
| `step` / `iteration` | One IterOut→IterIn cycle | ~~tick~~, ~~round~~ |
| `fire` | A node executing once (receives inputs, produces outputs) | ~~run~~, ~~invoke~~ |

---

## Python

- `from __future__ import annotations` at top of every file
- Type hints on all function signatures
- PEP 604 unions: `str | None` (not `Optional[str]`)
- Import order: `__future__` → stdlib → third-party → local
- Exception: `vlnworkspace/` files on Python 3.8 may use `typing.Dict` etc.

## TypeScript

- Strict TypeScript, functional components + hooks
- `type` keyword for type-only imports
- Tailwind for styling, Zustand for state
