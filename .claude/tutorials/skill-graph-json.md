# Skill: Graph JSON

## When to Use

You need to define a saved graph (full agent pipeline) or a reusable composite
(graph node) as a JSON file. Graphs are the primary artifact — one JSON file
= one agent architecture.

## Design Principles

1. **One JSON = one agent** — the graph JSON is the complete declarative spec
   of an agent's topology, wiring, and config. Node behaviour lives in Python
   `BaseCanvasNode` subclasses resolved from the `type` field via the registry;
   the JSON configures them. Reading the JSON should reveal the full
   architecture without needing to read the Python.

2. **Dataflow, not procedural** — nodes fire when their required inputs
   arrive, not in declaration order. Design the graph as a dataflow, not a
   script. Don't assume top-to-bottom or left-to-right execution order.

3. **Data wires vs access grants** — typed entries in `edges` are
   single-firing dataflow; entries in `access_grants` (dashed violet in the
   UI) are state read/write authorisations with no firing (ADR-dataflow-002/026).
   Don't smuggle shared state through data wires; put it in a
   `StateContainer` and grant access.

4. **Two-pivot iteration (ADR-dataflow-008)** — for loop graphs, use a
   two-sided `iterIn` (left/input side: run-start values wired into
   `init_<name>` handles declared via `config.initPorts`; right/output
   side: the loop-carry bundle) paired with `iterOut` (end-of-iter sink,
   transfers back via `pairedWith`, no canvas wire). `iterIn` never fires
   implicitly; its init-side edges are the explicit run-start handoff.
   (The former third pivot `Initialize` was removed 2026-06-10 — the
   validator rejects graphs that still carry one.)

5. **Composite snapshots, not references** — dragging a graph node
   (`kind="node"`) onto a canvas deep-copies it (ADR-canvas-003). Treat
   `workspace/graph_nodes/` as a frozen library; edits to the source file
   do **not** propagate into graphs that already embed it.

6. **Required ports must be wired** — `validate_graph_connectivity`
   (ADR-dataflow-006) rejects dangling required ports at load time, before execution
   starts. Mark a port `optional=True` only if the node genuinely handles
   absence; otherwise leave it required so the validator catches misses.

7. **Layout encodes phase** — left-to-right with ~280px horizontal spacing,
   and vertical grouping by role (env / perception / reasoning / action /
   state). Position is documentation — a reader should be able to trace the
   pipeline visually without opening every node.

8. **`step_budget` is intent, not a safety margin** — set `1` to declare a
   DAG, or the genuine per-episode step budget (e.g. `150` for VLN-CE,
   `500` for long-horizon nav) to declare a loop. Don't pad it "to be safe";
   the value communicates what kind of graph this is. The framework's
   resolver chain (eval batch path) lets the env override this per episode
   via the env panel's `on_load()` return — graphs that hand off to a
   scene-adaptive env can leave `step_budget` as a failsafe ceiling.

9. **LLM `profile` is user-level config, never graph-level** — every
   profile-bearing node in a saved graph (currently `llmCall`) **must**
   have `config.profile = ""`. Empty resolves to the user's active profile
   at run time. Profiles bind a name to `provider + model + api_key + base_url`
   — they are per-deployment / per-account state, not portable graph
   topology. Hardcoding a profile name (e.g. `"openeqa-judge"`) makes
   the graph fail for any user who hasn't created that exact profile,
   and can silently fail when a named profile points to an unfunded /
   wrong-account key. If the graph needs different LLMs for different
   roles (reasoner vs judge), document the requirement in a Note node
   and let the user pick per-node in the canvas UI; do **not** bake
   profile names into the JSON.

## File Location

| Type | Directory | `kind` field |
|------|-----------|-------------|
| Full graph (editable pipeline) | `workspace/graphs/{name}.json` | omitted (default) |
| Composite (reusable graph node) | `workspace/graph_nodes/{name}.json` | `"node"` |

## Graph Skeleton

```json
{
  "name": "My Agent",
  "description": "Brief description of this agent graph",
  "nodes": [
    {
      "id": "unique_node_id",
      "type": "nodeset__node_type",
      "label": "Human Label",
      "position": { "x": 0, "y": 0 },
      "config": {}
    }
  ],
  "edges": [
    {
      "id": "e_source_to_target",
      "source": "source_node_id",
      "target": "target_node_id",
      "sourceHandle": "output_port_name",
      "targetHandle": "input_port_name"
    }
  ],
  "step_budget": 1
}
```

## Node Entry Format

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | Unique within graph. Convention: descriptive name (e.g. `"env_step"`, `"planner_llm"`) |
| `type` | yes | Must match a registered `node_type` from a loaded nodeset |
| `label` | yes | Human-readable label shown on canvas |
| `position` | yes | `{"x": number, "y": number}` — canvas layout (left-to-right, ~280px spacing) |
| `config` | yes | Per-node config (can be `{}`). Keys match `ConfigField` names |

## Edge Entry Format

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | Unique edge ID. Convention: `"e_{source}_{port}_to_{target}"` |
| `source` | yes | Source node `id` |
| `target` | yes | Target node `id` |
| `sourceHandle` | yes | Must match an `output_ports` name on the source node |
| `targetHandle` | yes | Must match an `input_ports` name on the target node |

## Loop Pattern (IterIn / IterOut — ADR-dataflow-008 two-pivot model)

For cyclic agent loops, declare **two** paired pivots, both two-sided:
`iterIn` (left/input side captures run-start values once; right/output side
emits the loop-carry bundle each iteration) and `iterOut` (left/input side
collects end-of-step data plus the class-level `stop` BOOL halt input;
right/output side is the **final side** — `final_<name>` handles that emit
exactly once at loop termination, feeding the after-loop verdict stage).
Author writes
`iterIn.config.initPorts` and `iterOut.config.ports`; **iterIn's
`config.ports` is auto-synthesised at graph load** from those two writers
(with `init_*` / `iterout_*` prefixes — see "iterIn port synthesis" below)
and must not be hand-written. Run-start values reach iterIn as **ordinary
canvas edges** targeting the prefixed `init_<name>` handles; the
iterOut→iterIn transfer is executor-internal via `pairedWith` (no canvas
wire between the pivots).

```json
{
  "id": "iter_in",
  "type": "iterIn",
  "label": "Iteration Start",
  "position": { "x": 560, "y": 0 },
  "config": {
    "version": 3,
    "pairedWith": "iter_out",
    "initPorts": [
      {"name": "instruction", "wire_type": "TEXT", "persist": true},
      {"name": "init_observation", "wire_type": "TEXT", "persist": false}
    ]
  }
},
{
  "id": "iter_out",
  "type": "iterOut",
  "label": "Iteration End",
  "position": { "x": 1960, "y": 0 },
  "config": {
    "pairedWith": "iter_in",
    "ports": [
      {"name": "rgb",  "wire_type": "IMAGE", "persist": true,  "required": true},
      {"name": "pose", "wire_type": "POSE",  "persist": true,  "required": true}
    ]
  }
}
```

Run-start values arrive over plain edges into the prefixed init handles:

```json
{
  "id": "e_obs_rgb_to_init",
  "source": "env_observe",
  "target": "iter_in",
  "sourceHandle": "rgb",
  "targetHandle": "init_rgb"
}
```

Key rules:
- `iter_in.pairedWith` and `iter_out.pairedWith` must reference each other's `id`.
- `iterOut.config.ports` is author-declared and **must be non-empty** (validator rejects empties). `iterIn.config.initPorts` declares the init side; each entry's init edge targets `init_<name>`.
- `iter_in.config.ports` is **derived** at graph load by `_synthesize_iterin_ports`. Do not hand-author it; if you write `init_ports` / `loop_ports` (legacy v2 schema) the validator rejects the load.
- `iterIn` has no `seed: true` flag and is never queued at run-start — it fires when its init-side edges deliver (iter 0) or when the paired iterOut transfers (iter 1+).
- **`persist` on initPorts**: default `false` = one-shot (the slot empties after iter 0). Any run-invariant read by loop-body nodes on every iteration (`instruction`, intrinsics, bounds, runtime handles) **must set `persist: true`**, or consumers starve from iter 1 onwards — the failure is silent (loop fires once, then stalls). One-shot `false` is only right when the same name is also loop-carried by iterOut (init covers iter 0, iterOut takes over).
- **Stop**: wire the halt signal (BOOL — `env_step.done`, `parse.is_stop`, …)
  into `iter_out.stop`. The engine checks it once per iteration at the
  iterOut boundary; truthy ends the loop. Unwired `stop` = budget-only
  loop. There is no separate termination node (removed 2026-06-11; the
  validator rejects graphs that still carry one).
- **Final side**: edges *from* iterOut may only use `final_<port>` handles
  (one per `config.ports` entry) plus the constant `final_stop` (BOOL,
  emits True once at termination — the canonical after-loop trigger).
  These fire exactly once, with the terminal iteration's values.
- **After-loop band**: nodes downstream of `final_*` form the verdict
  stage (evaluate → graphOut chains). They may take inputs ONLY from the
  final side or from each other — "verdict inputs ride the pivot": any
  value the verdict needs must be an iterOut port, read via
  `final_<name>`. The validator rejects in-loop edges into the band, and
  (for eval graphs) `metrics`/`success` graphOuts fed from the loop body.
- Set `step_budget` at top level (e.g. `150` for VLN-CE navigation, `1` for DAG).

### iterIn port synthesis (read-only, for understanding)

Each `iterIn.config.initPorts[].name` becomes an iterIn output named
`init_<name>` (default `persist=false`). Each `iterOut.ports[].name` becomes
an iterIn output named `iterout_<name>` (default `persist=true`). Direct
canvas edges targeting an undeclared `iter_in.<handle>` create a third class
of init-writer surrogates (handle name used as-is, `persist=false`).
Downstream nodes wire from these synthesised handles — e.g.
`iter_in.init_instruction` → `planner.instruction`,
`iter_in.iterout_rgb` → `vlm.rgb`.

### Where does cross-step data live? (init / loop / container)

Loop graphs have **three** mechanisms for carrying data across iterations.
Picking the wrong one is the most common Loop Pattern mistake — usually
run-invariants get smuggled through `loop_ports`, forcing upstream nodes to
re-emit them every step or requiring explicit `iter_in.X → iter_out.X`
passthrough edges. Decision rule:

| Data shape | Lives in | iterIn handle |
|------------|----------|---------------|
| Changes every iteration (`rgb`, `pose`, `last_action`) | `iterOut.config.ports` (loop-carry) | `iterout_<X>` |
| Set once, never changes during the run (`instruction`, `episode_id`, `choices`, `cam_intr`, `tsdf_bnds`, `answer_gt`) | `iterIn.config.initPorts` only, with `persist: true` | `init_<X>` |
| Cross-step accumulating / shared memory (`action_history`, `visited` set, score history, topo map) | `StateContainer` + `access_grant` | (no handle — `ctx.containers["..."]`) |

Stop-and-ask rule: if you are about to draw `iter_in.iterout_X → ... →
iter_out.X`, ask "**Does X change inside the loop?**"

- **No** → move it out of `iterOut.ports`. Declare it on
  `iterIn.config.initPorts` with `persist: true` and wire its source into
  `init_X`. Consumers wire from `iter_in.init_X`; the persistent slot
  re-emits the captured value every iteration. The legacy
  `iter_in.X → iter_out.X` passthrough edge belongs exactly here.
- **Yes, and only one node uses it** → keep it on `iterOut.ports`.
- **Yes, and multiple nodes / post-loop nodes need it, or it accumulates**
  → put it in a `StateContainer` (next section). Don't carry it on the
  loop just so a sink at the end can read it 50 iterations later.

Symptoms that this rule was violated:

- An env / observation node has output ports for episode metadata
  (`question`, `episode_id`, `choices`, `answer_gt`) — those don't belong
  on the env contract; they're being threaded back through it to keep
  them on the loop rail. Violates the decoupled-nodesets principle.
- A consumer has **two** edges from `iter_in` to the same input port
  (one `init_X`, one `iterout_X`) carrying the same value — the
  `iterout_X` half is dead weight; loop-overrides-init resolves to the
  same value either way.
- `iterIn.config.initPorts` and `iterOut.config.ports` declare large
  overlapping sets — the overlap is exactly the run-invariants that
  should be init-only. (Worse: with `persist: true` on the init half,
  wiring both `init_X` **and** `iterout_X` into the same consumer input
  freezes that input at its step-0 value — the persistent init slot
  re-satisfies the port every iteration and shadows the loop-carry.)
- Explicit `iter_in.iterout_X → iter_out.X` passthrough edges with no
  producer in between — pure circulation.

## State Containers (Optional)

For graphs needing shared persistent state across iterations. The schema has
exactly two top-level fields: `containers` (list of ContainerDef) and
`access_grants` (list of node→container authorisations). There is **no**
separate `graph_state` field — `"graph_state"` is just the well-known **id**
of the optional graph-level blackboard container, which lives inside
`containers` like any other.

```json
{
  "containers": [
    {
      "id": "graph_state",
      "label": "Navigation State",
      "position": { "x": 1120, "y": 320 },
      "states": {
        "action_history": { "type": "accumulator", "value_type": "ACTION", "lifetime": "episode" },
        "step":           { "type": "counter",     "value_type": "ANY",    "config": {"initial_value": 0}, "lifetime": "episode" },
        "plan":           { "type": "lastWrite",   "value_type": "TEXT",   "lifetime": "run" }
      }
    },
    {
      "id": "nav_memory",
      "label": "Navigation Memory",
      "position": { "x": 1400, "y": 320 },
      "states": {
        "visited": { "type": "accumulator", "value_type": "TEXT", "lifetime": "episode" }
      }
    }
  ],
  "access_grants": [
    { "id": "ag_planner_gs",   "node_id": "planner",  "container_id": "graph_state" },
    { "id": "ag_env_step_nav", "node_id": "env_step", "container_id": "nav_memory"  }
  ]
}
```

**Reducer types** (controls *how* writes combine — `StateDef.type`):

| Reducer      | Behaviour |
|--------------|-----------|
| `accumulator` | Append each write to a list (optional `max_size` config) |
| `lastWrite`   | Replace prior value with the new write |
| `counter`     | Integer counter; writes increment by the supplied delta |

**Lifetime** (controls *when* state clears — `StateDef.lifetime`, orthogonal
to reducer; default `"forever"`):

| Lifetime    | When it clears |
|-------------|----------------|
| `"forever"` | Never auto-clears (default) |
| `"step"`    | Clears at every IterOut `step_end` signal — replaces the old "ephemeral" reducer pattern |
| `"episode"` | Clears on env-panel `episode_reset` signal |
| `"run"`     | Clears on `run_end` (after the loop finishes) |
| `"custom"`  | Explicit signal list via the `reset_on: ["sig_a", "sig_b"]` field |

`access_grants` are **not** wires — they carry no data, never trigger
firing, and are the only way a node may call `container.read()` /
`container.write()` at execution time. Rendered as dashed violet lines on
the canvas.

## Hooks (Optional)

Shell-command hooks fire at graph lifecycle events. The hook process
receives the event payload as JSON on stdin and may write a JSON action
response on stdout (e.g. to abort, retry, or mutate execution).

```json
{
  "hooks": [
    {
      "event": "PreNodeExecute",
      "command": "/usr/local/bin/log-node.sh",
      "match_node_type": "env_mp3d__*",
      "timeout_ms": 1000,
      "enabled": true
    }
  ]
}
```

| Field | Notes |
|-------|-------|
| `event` | One of `PreNodeExecute`, `PostNodeExecute`, `GraphStart`, `GraphComplete`, `GraphError` |
| `command` | Shell command run as a subprocess |
| `match_node_type` | `"*"` (all), exact `"env_mp3d__step"`, or prefix glob `"env_mp3d__*"` (matched via `startswith`) |
| `match_node_id` | Optional — exact node-instance match (overrides `match_node_type` granularity) |
| `timeout_ms` | Default 1000 |
| `enabled` | Default `true` |

## Edge Wire-Type Compatibility (ADR-027)

Edge validation runs at graph load (`validate_edge_wire_type`). Rules:

| Source → Target           | Result |
|---------------------------|--------|
| `T → T` (equal)           | OK |
| `ANY → *` or `* → ANY`    | OK (escape hatch) |
| `T → LIST[T]`             | OK — executor auto-wraps scalar as `[scalar]` (ADR-dataflow-005) |
| `LIST[T] → T`             | **Rejected** — would be lossy |
| `T → U` (different inner) | **Rejected** — type mismatch |

Use `LIST[T]` consumer ports for fan-in (multi-image VLM, multi-LLM debate).
Wire scalar producers to a `LIST[T]` consumer freely; the executor handles
wrapping and concatenation in edge declaration order.

## Load-Time Invariants (validate_graph_connectivity)

Caught at graph load, before execution starts. Authors hit these as
`ValueError: Graph connectivity validation failed:` with a bulleted list.

- **Required input ports must have an incoming edge.** A port is required
  iff `optional=False` on its `PortDef`. Use `_resolve_ports` per-instance
  schemas where applicable; otherwise class-level `input_ports`.
- **`type: "initialize"` is rejected outright.** The node was removed
  2026-06-10; the error carries a migration hint — declare the ports on the
  paired iterIn's `config.initPorts` and wire the seeds into `init_<name>`.
- **`iterOut` must declare non-empty `config.ports`.** Empty or missing
  port lists are rejected. Use the frontend port-list editor or hand-edit
  the JSON.
- **`pairedWith` must point to the right kind on the other end.** iterIn's
  `pairedWith` must reference an iterOut node id; iterOut's must reference
  an iterIn node id.
- **Legacy v2 iterIn schema is rejected.** If iterIn.config has
  `init_ports` or `loop_ports` keys, the validator refuses to load. Migrate
  by deleting those keys — iterIn ports are synthesised from its own
  `initPorts` + the paired iterOut now.
- **Edges from iterIn may only use synthesised handle names.** Valid
  `sourceHandle` values are the synthesised `init_*` / `iterout_*` names
  in `iter_in.config.ports` (auto-derived) plus the literal `"step"`. An
  edge referencing an unknown handle is rejected.

## Multi-scope graphs (ADR-dataflow-007)

A single graph may declare **N** (iterIn, iterOut) pairs — each one is a *scope*. Topology determines nesting: scope B is nested in scope A iff `B.iter_in` and `B.iter_out` both lie in A's BFS-reachable interior. Use this when an agent has two coexisting iteration cadences (outer reasoning loop + inner execution loop, HRL planner + low-level controller, etc.).

**Mandatory shape for an inner (non-outermost) scope:**

```
        outer-body wires
             │
             ▼
       [portIn_a]   [portIn_b]   ◀── parameter slots (≥0)
             │           │
             ▼           ▼
       ┌── inner scope body ──┐
       │  (init_* seed edges) │
       │       │               │
       │       ▼               │
       │   iterIn_inner ─→ ... │  ← two-pivot loop cadence
       │       ▲       ↓       │
       │   iterOut_inner       │
       └───────────────────────┘
             │           │
             ▼           ▼
       [portOut_x]  [portOut_y]   ◀── return slots (≥0)
             │
             ▼
       outer-body wires
```

**Rules** (enforced by `validate_graph_connectivity` → `analyze_scopes`):

1. **Cross-author-scope wires must go through a named exit (iterOut / graphOut) into a named entry (graphIn / iterIn's init side).** Direct outer-body → inner-body wires are rejected. Graph-scope ↔ author-scope wires are exempt (legacy seed/sink pattern).
2. **portIn/portOut node membership = innermost scope adjacent to it.** No author labels — topology decides.
3. **Each iter_in carries its own `step_budget`.** The outermost scope falls back to `graph.step_budget` for backward compat. The outer iter_in's config gets a derivative `nested_scope_ids: [...]` written by the analyzer (UI hint only — backend reads canonical from each iter_in).
4. **portOut latches.** A portOut inside a non-graph scope BUFFERS its incoming `value` per inner-iter and flushes the LAST one to outer-scope downstream when the inner scope terminates. Outer sees a function return value, not a stream.
5. **Inner scope re-entry is automatic.** When the outer iter brings the inner `iter_in` back, the executor resets the inner scope's `terminated` flag and `step_counter`. Inner-body node-instance ctx is NOT auto-reset — author-side: detect re-entry by comparing input identity (e.g. trajectory hash).
6. **An inner scope's stop ends only the inner scope** (its iterOut emits its final side, then outer continues). Only the root scope's stop — or its budget exhaust — halts the run.

**Canonical example**: `workspace/graphs/vla/verified/voxposer_libero_decomposed.json` — outer scope iterates per subtask (composer LMP + voxel maps + path planner runs ONCE per outer iter, emits `trajectory`); inner scope iterates per waypoint (`dispense_waypoint` cursor → `vp_plan_executor` → `env_libero__move_to_pose` → `episode_info` → `check_waypoint_done`); cross-scope IO via `port_in__trajectory` (outer→inner) and `port_out__{episode_success, step_index, waypoints_completed, inner_status}` (inner→outer with latched values).

**Authoring note — stop wiring**: the halt signal lives on each scope's own `iter_out.stop`, so it is structurally bound to the right scope — no scope-assignment workarounds needed. A check-done node wired to `iter_out_<scope>.stop` is automatically backward-reachable and lands in that scope.

## Composite Graph Node

For reusable sub-graphs saved as drag-and-drop nodes:

```json
{
  "kind": "node",
  "name": "My Reasoner",
  "description": "Reusable reasoning sub-graph",
  "group": "reasoning",
  "nodes": [
    { "id": "port_in", "type": "portIn", "label": "Input", "position": {"x": 0, "y": 0}, "config": {} },
    { "id": "port_out", "type": "portOut", "label": "Output", "position": {"x": 560, "y": 0}, "config": {} }
  ],
  "edges": []
}
```

Key: `portIn`/`portOut` nodes define the composite's external ports. Dragging onto canvas creates a deep copy (snapshot semantics, no reference).

## Checklist

1. [ ] `name` and `description` fields set
2. [ ] All node `id` values are unique within the graph
3. [ ] All `type` values match registered `node_type` from loaded nodesets
4. [ ] Edge `sourceHandle`/`targetHandle` match port names exactly (use POSE not the legacy STATE)
5. [ ] No dangling edges (source/target IDs all exist in nodes array)
6. [ ] `step_budget` set: `1` for DAG graphs, `100-500` for loop graphs (default in code is 500)
7. [ ] Loop graphs use the **two-pivot model**: two-sided `iterIn` + `iterOut`, paired via `pairedWith`; no `initialize` node (rejected by validator)
8. [ ] `iterOut.config.ports` is non-empty; init-side seeds declared in `iterIn.config.initPorts` with edges into `init_<name>`; `iterIn.config.ports` is **not** hand-written (auto-synthesised at load)
9. [ ] Multi-scope graphs: every cross-author-scope wire passes through portIn / portOut / iterOut-final / iterIn-init (rejected by validator otherwise); each inner scope has its own (iterIn, iterOut) pair with its own `step_budget` on the iter_in
9. [ ] Run-invariants (instruction, episode_id, intrinsics, choices, ground-truth) appear on `iterIn.config.initPorts` only, with `persist: true` — **not** in `iterOut.config.ports`, and not threaded through env-step → iter_out passthrough
10. [ ] No legacy `seed`, `init_ports`, or `loop_ports` keys on `iterIn`
11. [ ] Loop graphs wire a halt signal (BOOL) into `iter_out.stop` (or rely on `step_budget` alone); verdict nodes hang off `final_*` handles
12. [ ] Edges from `iter_in` only reference synthesised handles (`init_*` / `iterout_*` / `step`)
13. [ ] State containers live in `containers: list`; `"graph_state"` is just a well-known container id, not a top-level field
14. [ ] Each `StateDef.type` is one of `accumulator` / `lastWrite` / `counter`; ephemeral-style behaviour uses `lifetime: "step"` instead
15. [ ] State sharing uses `access_grants` (not the obsolete `state_edges`)
16. [ ] Edge wire types satisfy ADR-027 (equal, ANY, or `T → LIST[T]`); no `LIST[T] → T`
17. [ ] Composite graph nodes use `"kind": "node"` and have `portIn`/`portOut`
18. [ ] Positions use ~280px horizontal spacing for readability
19. [ ] Every `llmCall` (and any other profile-bearing node) has `config.profile = ""` — never bake a named profile into a saved graph (Design Principle #9)
