**English** | [中文](docs-md/README_zh.md) | [Espanol](docs-md/README_es.md) | [日本語](docs-md/README_ja.md) | [한국어](docs-md/README_ko.md)

<div align="center">

# AgentCanvas

### Automating the Design of Embodied Agent Architectures

**Jian Zhou · Sihao Lin · Jin Li · Shuai Fu · Gengze Zhou · Qi Wu**

Australian Institute for Machine Learning, University of Adelaide

<p>
  <a href="https://arxiv.org/abs/2606.30111"><img src="https://img.shields.io/badge/arXiv-2606.30111-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/index.html"><img src="https://img.shields.io/badge/Project%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/paper.html"><img src="https://img.shields.io/badge/Paper%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Paper Page"></a>
  <a href="https://jianzhou0420.github.io/AgentCanvas/"><img src="https://img.shields.io/badge/Docs-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="#9-citation"><img src="https://img.shields.io/badge/BibTeX-Cite-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white" alt="BibTeX"></a>
</p>

<img src="assets/readme/editor-hero.gif" alt="AgentCanvas editor: the MapGPT executor loads as a node-and-wire graph, then a live R2R episode runs end-to-end" width="760">

<sub><em>Recorded live in the editor — the MapGPT executor loads, then a real R2R episode runs end-to-end.</em></sub>

</div>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)](#7-project-status)
[![GitHub stars](https://img.shields.io/github/stars/jianzhou0420/AgentCanvas?style=social)](https://github.com/jianzhou0420/AgentCanvas/stargazers)

**A visual agent-design platform for embodied AI research.** One typed graph, two roles: a *harness* that runs embodied agents, and a *scaffold* that coding agents edit and verify.

AgentCanvas lets researchers prototype embodied agents — for VLN, EQA, VLA, and adjacent tasks — by drawing node graphs that execute in real time against simulators (Habitat-Sim, MatterSim, SAPIEN/ManiSkill2, MuJoCo/robosuite) or, in principle, real-world setups. *One JSON = one agent = one graph*: agent behaviour is a dataflow graph, not imperative code; the graph is the source of truth, saved as a single JSON file and loaded as a complete agent.

**Built for**: researchers who want to compose, compare, and share embodied agent architectures without rewriting the execution stack each time. The platform covers VLN (Vision-and-Language Navigation), EQA (Embodied Question Answering), VLA (Vision-Language-Action) policy benchmarks, and adapts to other embodied / agentic settings via the nodeset model.

> **Status**: Research preview, under active development · 46 ADRs · 40+ nodesets across four swappable palettes — **env** (simulators), **method** (reasoning loops), **model** (foundation models), **policy** (neural controllers) · canvas editor, graph executor with multi-scope iteration, state containers, auto-hosted server-mode nodesets, hook system, subprocess-per-run JobScheduler + worker-pool + batched inference, and a unified error bus — all in production.

> **Versioning**: pre-1.0 (v0.x). v1.0 ships when the public API is stable (open-sourced + frozen under SemVer) — independent of any paper. See the [Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html).

> **Contributing**: Two kinds, both welcome. **Content** — write a nodeset (tool or method) or compose a graph, by PR into `workspace/`; you're credited on the [Credits](#credits) board, with a citation link if it has a paper. **Core** — improve the framework (UI, backend, features, refactors); open a [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) first for anything big. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Contents

1. [Why AgentCanvas?](#1-why-agentcanvas) — a searchable substrate for embodied agents, and the pain points it solves
2. [Features](#2-features) — the *one JSON = one agent* (§2.2) / *one Python class = one node* (§2.6) principles, plus canvas editor, graph executor, isolated runtime envs, nested graphs, state containers, hooks
3. [Sim-to-Real Path](#3-sim-to-real-path) — same agent graph on simulator today, real robot tomorrow — via env-as-nodeset + server mode + ROS
4. [Getting Started](#4-getting-started) — prerequisites, run the web dashboard, run an evaluation, run architecture search, serve the docs
5. [Architecture](#5-architecture) — frontend · backend · workspace · simulators
6. [Project Structure](#6-project-structure) — top-level directory map
7. [Project Status](#7-project-status) — versions: v0.1 experiments → v0.2 preview → v1.0 → v2.0
8. [Contributing](#8-contributing) — where help is most wanted · credits
9. [Citation](#9-citation) — cite the AgentCanvas paper
10. [License](#10-license) — Apache 2.0

---

## 1. Why AgentCanvas?

Embodied agents — spanning VLN, EQA, and VLA — are increasingly built by composing foundation models with perception, mapping, memory, planning, and action. Unlike end-to-end policies, whose structure is absorbed into weights, this architecture is *explicit and editable*. That raises the question AgentCanvas is built around — **can agent design be searched rather than hand-built?** — alongside two stacks of pain it has to clear on the way.

<details>
<summary><b>Agent architecture is hand-built — and could be searched</b></summary>

<br>

Each agent fixes a choice at every join — sensor abstractions, map representations, memory state, prompt structure, planner topology, model placement, action interfaces — by hand, usually for a single benchmark. As foundation models and embodied tools multiply, the space grows faster than manual iteration can cover, so the natural move is to search it rather than hand-tune it.

Agent Architecture Search (AAS) already does this for text-domain agents, but embodied transfer is not free: stateful simulators, noisy multi-episode scoring, long perception/action traces, and no off-the-shelf palette of embodied primitives. AgentCanvas is our attempt to supply the missing substrate — a scaffold a coding agent can read, edit, run, and verify — so that searching agent design becomes possible for embodied agents too.

</details>

<details>
<summary><b>Embodied-specific pain points</b></summary>

<br>

- **Modern embodied stack is thick** — a working embodied agent needs LLM reasoning + tool use + simulator coupling + spatial tools, all wired together. Building this from scratch per project is prohibitively costly, and most of the effort goes into the execution layer rather than the idea being tested.
- **Engineering nightmare** — an embodied agent is not one model but a whole system — a stateful simulator plus a stack of heavy models and tools. Just running it, let alone at the scale benchmarking needs, is a hard engineering job in itself:
  - **Python env hell** — no single Python env satisfies every part; each simulator, VLM, detector, and policy pins its own clashing CUDA / torch / Python, so finding one runtime they all share is often impossible — you end up maintaining several incompatible environments just to load the agent.
  - **Batching** — each worker's simulator is a separate stateful process advancing at its own pace; you can batch the model but not the sims, so every step becomes an async gather-observations → batch-infer → scatter-actions dance.
  - **Other infra** — multimodal trajectories that must be logged and replayable, checkpoint/resume of multi-hour GPU runs that *will* crash, and debugging across process boundaries.

  Over a single paper's research cycle, the researcher pays far too much of this engineering cost instead of focusing on the algorithm itself.
- **Hidden ground-truth dependencies** — many methods quietly rely on simulator-provided ground truth (object poses, semantic labels, navigability) rather than real perception. Sometimes that's a legitimate way to control the experiment — but, oversight or not, it often goes unmentioned in the paper.

</details>

<details>
<summary><b>Common AI research pain points (amplified here)</b></summary>

<br>

- **Non-reproducible implementations** — every paper builds its agent from scratch with a different codebase; comparing methods fairly or reproducing results is painful — and many of them are **`Code coming SOON`** (**S**omeday, **O**r **O**bviously **N**ever).
- **Paper ≠ code** — papers show clean flow diagrams, but the actual code diverges in undocumented ways. Reproducing a paper means reverse-engineering its implementation.
- **Tightly coupled code** — domain logic (prompts, tools, policies) is tangled with infrastructure. Swapping one component means rewriting the pipeline.

</details>

---

## 2. Features

> **Full reference in the docs** — most features below have an implementation page (mechanism · key files · current status): **[The Nine Capabilities →](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/index.html)**

### 2.1 Visual Canvas Editor

A ComfyUI-style flat workspace where all node types coexist — environments, LLMs, reasoning chains, control gates, and output viewers. Drag nodes from the sidebar, wire them together, hit Play.

### 2.2 Graph Execution Engine

**One JSON = one agent.** An agent's entire behavior — nodes, wiring, configs, state containers, hooks — is a single JSON file: load it, run it, share it, diff it. No hidden pipeline code; what you see on the canvas is what executes.

```jsonc
// Simplified — real graphs include state containers, hooks, and more nodes
{
  "name": "NavGPT-CE",
  "description": "VLN reasoning graph with planner, VLM, and navigation memory",
  "kind": "graph",
  "nodes": [
    { "id": "observe", "type": "env_habitat__observe_egocentric", "config": {} },
    { "id": "planner", "type": "llmCall",                         "config": { "temperature": 0.0 } },
    { "id": "step",    "type": "env_habitat__step_discrete",      "config": {} }
  ],
  "edges": [
    { "source": "observe", "sourceHandle": "rgb", "target": "planner", "targetHandle": "image" },
    { "source": "planner", "sourceHandle": "action", "target": "step", "targetHandle": "action" }
  ]
}
```

The engine then runs that graph: nodes fire when their inputs arrive, not in a fixed order. The same engine handles every graph shape AgentCanvas v1 supports — see [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) for the full list of agent forms covered by v1's bounded-static-topology paradigm:

- **DAG workflows** — single forward pass for acyclic pipelines
- **Cyclic agent loops** — observe-think-act-repeat via a **two-pivot** model: a two-sided **`IterIn`** (run-start init inputs on its left, per-iteration loop-carry on its right) plus **`IterOut`**, keeping the graph visually acyclic while enabling runtime cycles (ADR-dataflow-008, which folded the earlier three-pivot `initialize`/IterIn/IterOut of ADR-dataflow-006 down to two)
- **Multi-scope iteration** — N coexisting `(IterIn, IterOut)` pairs in one flat graph (ADR-dataflow-007 / ADR-executor-003)
- **ReAct loops** — either hidden inside an `LLMCallNode` subclass or expressed explicitly as router + N predeclared tool branches
- **Bounded multi-agent** — fixed-N or `K_max`-bounded fan-out (e.g., DiscussNav-style debate, AutoGen-style fixed roles)
- **Plan-and-Execute** — over a bounded tool pool, dispatched by router

### 2.3 Isolated Runtime Environments

Research tools often need conflicting Python environments (Habitat needs Python 3.8, SLAM needs ROS). Any `BaseNodeSet` can run in **server mode** — the framework auto-generates an HTTP server from the nodeset's port definitions, running in its own interpreter. Zero extra code:

```
# Same nodeset code, two deployment modes:
POST /api/components/nodesets/env_habitat/load              # in-process
POST /api/components/nodesets/env_habitat/load?mode=server  # separate process
```

### 2.4 Nested Graph System

Save any canvas graph as a **graph node** and drag it onto another canvas as a reusable block. This enables hierarchical agent architectures — a high-level planner containing sub-agent graph nodes. Snapshot semantics: each instance is a deep copy.

### 2.5 State Container System

Shared persistent state across agent loop iterations via a dual-wire architecture:

- **Data edges** carry dataflow between nodes (IMAGE, TEXT, ACTION, POSE, …)
- **Access grants** let nodes read/write **StateContainers** — visible canvas elements with named entries, configurable reducers (Accumulator, LastWrite, Counter), and a **Lifetime** axis (`forever` / `step` / `episode` / `run` / `custom`) that auto-clears memory on the right signal boundary (ADR-dataflow-002, ADR-dataflow-004)

→ [State Containers design doc](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/graph/state-containers.html)

### 2.6 Python-Defined Nodes

**One Python class = one node.** Every canvas node — tools, environments, skills, policies — is a single Python class: declare ports, implement `forward()`, drop the file in `workspace/`, and the platform auto-discovers it. No framework changes, no TypeScript, no registration boilerplate.

```python
from app.components import BaseCanvasNode, PortDef

class MeasureDistanceNode(BaseCanvasNode):
    node_type    = "basic_agent__measure_distance"
    display_name = "Measure Distance"
    description  = "Euclidean distance between two 3D positions"
    category     = "tool"
    icon         = "Ruler"

    input_ports  = [
        PortDef("pos_a", "TEXT", "Position A as [x, y, z]"),
        PortDef("pos_b", "TEXT", "Position B as [x, y, z]"),
    ]
    output_ports = [
        PortDef("distance", "TEXT", "Euclidean distance (meters)"),
    ]

    async def forward(self, inputs, ctx):
        a, b = parse_vec3(inputs["pos_a"]), parse_vec3(inputs["pos_b"])
        dist = math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
        return {"distance": f"{dist:.2f}"}
```

The node then appears in the canvas sidebar and wires to any other node with matching port types. Its appearance is Python-driven too: `GenericBlockRenderer` renders any node automatically from `NodeUIConfig` — colors, layout, inline config controls (sliders, dropdowns, text fields), and display widgets — so no custom React component is needed.

### 2.7 Hook System

Shell commands fire before/after each node execution and at graph lifecycle boundaries. Hooks can log outputs, validate inputs, block nodes, or modify data — all without changing graph nodes. Hooks travel with saved graphs.

### 2.8 Batch Evaluation & Job Queue

The same graph that runs on the canvas can be submitted as an eval job that scores it over hundreds of episodes. A backend-owned `JobScheduler` gates admission against a VRAM budget shared across all sessions (ADR-eval-003); each admitted run is its own subprocess, so backend restarts don't kill in-flight evals. Per-episode logs land in a self-contained layout (ADR-eval-004) so a teammate can replay any single episode without re-running.

### 2.9 Real-Time Observability

Every step streams observations, reasoning, actions, and metrics via WebSocket, routed by `execution_id` so concurrent runs don't cross streams. Errors from any source — node exceptions, server-mode subprocess crashes, and HTTP failures — flow through a unified `ErrorBus` and surface as Report-tab entries + toasts (ADR-observability-004). (React render errors are caught by a client-side error boundary.)

---

## 3. Sim-to-Real Path

AgentCanvas is designed for portability: a single agent graph can execute against a simulator today and migrate to a real robot in the future without graph-level changes. This property follows from two architectural decisions — environments are themselves nodesets (ADR-components-002), and any nodeset can execute in an isolated runtime via *server mode* (ADR-server-001).

### Today: Simulator Nodesets

The shipped environments — Habitat (VLN-CE), MatterSim / MP3D, HM-EQA, OpenEQA, SIMPLER (real-to-sim VLA), and LIBERO (manipulation) — are each implemented as a `BaseNodeSet` that exposes observation and action ports. The agent graph connects to these ports and never imports the simulator directly, which keeps the graph independent of any specific environment implementation.

### Tomorrow: A ROS Nodeset with the Same Interface

Real-robot deployment is achieved by replacing the simulator nodeset with a **ROS nodeset** that exposes the same `observation` / `act` interface. Internally, this nodeset composes existing ROS components — `cv_bridge`, `Nav2`, `MoveIt`, and hardware driver packages — into a unified facade. Server mode launches the nodeset inside its own ROS Python environment and bridges it to the canvas over HTTP. The agent graph itself is unchanged.

This division of labor is favorable because the substantive engineering — perception, control, motion planning, and hardware interfacing — already exists as mature ROS packages. The ROS-side adapter is therefore a composition task rather than greenfield development, and the AgentCanvas-side env nodeset reduces to a thin HTTP client.

### Bidirectional Integration

The boundary between AgentCanvas and ROS is symmetric; either side may own the control loop:

- **ROS as a subsystem of AgentCanvas** *(native pattern; server mode is designed for this case)* — the ROS nodeset runs in server mode, AgentCanvas drives the agent loop, and ROS provides sensing and actuation.
- **AgentCanvas as a subsystem of ROS** *(also supported; no framework modifications required)* — when the broader project is ROS-led, the ROS-side control loop invokes AgentCanvas's `/run` endpoint at each step (treating the graph as a policy) and publishes the returned action. This requires only a thin ROS bridge node on the ROS side.

### Visibility of Ground-Truth Dependencies

The same nodeset abstraction directly addresses two pain points raised in §1. A node that queries simulator ground truth (e.g., `env_habitat__get_object_pose`) and a node that performs real perception (e.g., a SAM-based detector) appear as visibly distinct blocks on the canvas. Whether an agent depends on ground truth or on perception is therefore a property of the graph topology, not a hidden implementation detail. Substituting one for the other is a local edge change, not a code refactor.

### Status

All currently shipped environment nodesets are simulator-based. A real-robot **ROS nodeset remains a [call-for-contribution](#8-contributing) slot** — the architectural pathway is established and intentional, and the necessary ROS-side components are already available in the ecosystem.

---

## 4. Getting Started

There are two ways to use AgentCanvas, both over the same typed-graph substrate:

1. **Build & run a graph by hand** — compose nodes on the canvas, run an agent live against a simulator, and evaluate it at scale (the rest of this section).
2. **Agent Architecture Search (AAS)** — hand a seed graph to a coding agent and let it search architectures for you ([jump](#44-run-agent-architecture-search-aas)).

### 4.1 Prerequisites

- Python 3.10+ with Conda (the default `agentcanvas` env — ADR-platform-004)
- Node.js 18+
- *(Optional, for Habitat-Sim)* a separate Python 3.8 env — `habitat-sim 0.1.7` only runs here; AgentCanvas talks to it via server mode, see [INSTALL.md](docs-md/INSTALL.md)

### 4.2 Run the Web Dashboard

```bash
# Activate environment
conda activate agentcanvas

# Start backend (FastAPI :8000) + frontend (Vite :5173)
cd agentcanvas && bash run_dev.sh
```

Open [http://localhost:5173](http://localhost:5173) to access the canvas editor.

### 4.3 Run an Evaluation

The same eval pipeline is exposed through four interfaces — pick by what you're holding:

| # | Interface | Audience | Best for |
|---|-----------|----------|----------|
| 1 | **Frontend Eval page** | Human                | Click-driven, watch live progress in the UI |
| 2 | **`/experiment:run` slash command** | Coding agent (Claude Code) | Profile-gated GPU admission, auto-allocated port, no `:8000` clobber |
| 3 | **MCP server** | Coding agent              | Conversational, ad-hoc eval — no slash-command overhead |
| 4 | **HTTP API** | Scripts / CI                | Direct REST, no MCP required |

#### 1. Frontend Eval page — for humans

Open a saved graph on the **Eval** page, pick a split + episode range, hit **Start**. Progress streams live over WebSocket; results land as per-episode JSONL under `outputs/eval_runs/{run_id}/episodes/ep{NNNN}/` (ADR-eval-004) and are browsable in the Run Detail panel. Multi-worker env fan-out and batched inference are configurable from the form (ADR-eval-002).

→ [Batch Eval tutorial](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/batch-eval.html)

#### 2. `/experiment:run` — for coding agents on this repo

When using Claude Code, `/experiment:run <profile> -- <cmd>` wraps any eval invocation in the backend's `JobScheduler` admission gate (ADR-eval-003): the wrapper claims VRAM under the declared profile in `.claude/commands/experiment/profiles.yaml`, spawns the backend on an allocated port (`BACKEND_URL=http://127.0.0.1:<port>` is exported to the wrapped command), and releases the slot on exit. Companion commands: `/experiment:status` for run snapshots, `/experiment:teardown` for graceful cancellation.

→ [`.claude/commands/experiment/README.md`](.claude/commands/experiment/README.md)

For full architecture-search design loops (many iterations of propose → evaluate → keep-the-best over a seed graph), see [Run Agent Architecture Search](#44-run-agent-architecture-search-aas) below.

#### 3. MCP server — for coding agents

Register `agentcanvas-backend` with any MCP-aware client (Claude Code, Cursor, …) and call typed tools (`graph_list`, `eval_start`, `eval_status`, `eval_export`, `eval_stop`) conversationally. No iter-tree bookkeeping — just raw eval against a borrow-or-spawned backend.

→ [`agentcanvas/mcp_server/README.md`](agentcanvas/mcp_server/README.md)

#### 4. HTTP API — for scripts and CI

Direct REST for scripts, CI, or non-MCP environments:

```bash
curl -X POST http://localhost:8000/api/eval/v2/start \
  -H 'content-type: application/json' \
  -d '{"graph_name": "navgpt_ce", "split": "val_unseen", "worker_count": 4}'
# poll  GET /api/eval/v2/status
# fetch GET /api/eval/v2/export/{run_id}
```

→ [Driving the Backend from a Coding Agent](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/coding-agent-backend.html) — deep dive on all programmatic modes side-by-side

### 4.4 Run Agent Architecture Search (AAS)

Beyond evaluating a fixed graph, AgentCanvas is the substrate for **Agent Architecture Search** — a development-time loop where an LLM coding-agent *Optimizer* repeatedly proposes graph edits to a seed *Executor*, evaluates each candidate in the simulator, and keeps the improvements (§1 — [why a searchable substrate](#1-why-agentcanvas)). Because an agent is a typed graph, each candidate is a type-checked patch run before any expensive rollout, and per-node episode logs let the Optimizer attribute score changes to specific modules.

<p align="center">
  <img src="assets/readme/aas-search.gif" alt="The coding-agent optimizer searching over an embodied executor's graph — proposing edits, running them, keeping the gains" width="800">
  <br><sub><em>The coding-agent optimizer searching over an embodied executor's graph — propose an edit, run it, keep the gains.</em></sub>
</p>

Search is **method-seeded**: `iter_0` is a published embodied method and the loop searches graph-level edits around it. Three search variants ship as Claude Code skills under `.claude/commands/architect/`, sharing one coding-agent harness (proposer → implementer → evaluator) and differing only in proposer logic + persistent memory:

| Variant skill | Paper name | Search policy |
|---|---|---|
| `myloop` | **KDLoop** | Four-phase THINK → CRITIC → EXPERIMENT → DISTILL cycle, typed memory + REFLECT meta-phase |
| `adas-subagent` | **ADAS** (port) | Reflexion-style proposals over a flat append-only archive |
| `aflow` | **AFlow** (port) | Score-softmax parent selection + anti-replay memory |

```text
# In a Claude Code session on this repo — run KDLoop over the MapGPT executor
/architect:myloop:loop mapgpt_mp3d --goal "raise val_unseen SR"

# The ADAS / AFlow ports take the same  <graph> [<version>]  form
/architect:adas-subagent:loop smartway_ce
/architect:aflow:loop explore_eqa_hmeqa
```

Seed graphs currently wired for search: `mapgpt_mp3d`, `smartway_ce` (VLN), `explore_eqa_hmeqa` (EQA), `voxposer_libero_monolithic` (VLA). Each iteration writes its proposal, patch, eval scores, and logs under `outputs/design_runs/{variant}/{graph}/vN/iter_M/`.

→ [AAS pipelines reference](https://jianzhou0420.github.io/AgentCanvas/pages/aas/index.html)

### 4.5 Documentation

```bash
# Serve the doc-site locally on :8092 (live-reload via SSE)
bash docs/run_dev.sh
```

---

## 5. Architecture

```
Frontend (React 18 + React Flow + Zustand)
    |
    |  REST + WebSocket
    v
Backend (FastAPI + Python 3.10+)
    |
    |-- WorkspaceComponentRegistry  -->  workspace/  (auto-discovery)
    |-- GraphExecutor   -->  graph execution (DAG + cyclic + multi-scope)
    |-- AutoServerApp      -->  server-mode nodesets (isolated envs)
    |-- HookRunner         -->  pre/post interceptors
    |-- JobScheduler       -->  subprocess-per-run eval admission (ADR-eval-003)
    |-- ErrorBus           -->  unified error reporting (ADR-observability-004)
    v
Simulators (Habitat-Sim, MatterSim/MP3D, HM3D, SAPIEN/ManiSkill2, MuJoCo/robosuite, ...)
```

**Key design**: The framework has **zero domain knowledge** (ADR-platform-001). All domain-specific code — VLN policies, LLM prompts, navigation tools, environment wrappers — lives in `workspace/`. The framework discovers components at runtime via base class inheritance. It never imports domain code directly; the import boundary is enforced by `agentcanvas/backend/app/test_import_boundary.py`.

---

## 6. Project Structure

```
vlnworkspace/                  # repo root (legacy name; the platform is "AgentCanvas")
├── agentcanvas/               # Full-stack web application
│   ├── backend/app/         #   FastAPI backend (execution engine, APIs, services, errors)
│   ├── frontend/src/        #   React + TypeScript (canvas editor)
│   └── mcp_server/          #   MCP server for coding-agent integration
├── workspace/                 # User workspace — all domain components (auto-discovered)
│   ├── nodesets/            #   Nodesets by palette: env / method / model / policy (+ common, _upstream)
│   ├── graphs/              #   Saved agent graphs (kind="graph")
│   ├── graph_nodes/         #   Reusable composite nodes (kind="node")
│   ├── nodes/               #   Standalone BaseCanvasNode subclasses
│   ├── architect/           #   AAS search profiles + run scaffolding
│   └── hooks.json           #   Workspace-level hook definitions
├── data/                      # Datasets, model weights (gitignored)
├── outputs/                   # Eval + design-run outputs (eval_runs/, design_runs/, …)
├── docs/                      # Hand-authored HTML doc-site (run_dev.sh → :8092)
├── third_party/               # Git submodules (habitat-lab, VLN-CE, MatterSim, vla_workspace, …)
└── scripts/                   # Data setup + install scripts
```

---

## 7. Project Status

AgentCanvas is **pre-1.0 and under active development**. Status is tracked by version, not a running feature checklist — see the [Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html) and [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) for detail.

- **v0.1 — AAS experiments.** The snapshot the paper's Agent Architecture Search runs were executed on — a reproducibility anchor for those results, not a public release.
- **v0.2 — research preview (current).** The first open-sourced release: the canvas editor, graph executor (DAG + cyclic + multi-scope), state containers, auto-hosted server-mode nodesets, batch eval, and 40+ nodesets (env / method / model / policy) all run in production. The public API is not yet frozen, so minor releases may break it. Shipped inventory: [§2 Features](#2-features) and the [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vla-support-status.html) support-status pages.
- **v1.0 — in progress.** Ships when the public API is stable — open-sourced and frozen under SemVer, independent of any paper.
- **v2.0 — future.** Topology-mutating execution: unbounded subagent spawning, runtime fan-out over runtime lists, new tool types emerging at runtime, self-modifying graphs. See [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) §2 for the thesis and open questions.

---

## 8. Contributing

Two kinds of contribution, both welcome — see [CONTRIBUTING.md](CONTRIBUTING.md):

- **Content — nodesets & graphs.** Write a nodeset that wraps a tool / simulator / model (e.g. real-time 3D Gaussian Splatting, a voxel-based SLAM system) or encodes a method (e.g. NavGPT, MapGPT), or compose a graph that wires existing nodesets into a complete agent. Open a PR into `workspace/`; review is light.
- **Core — UI, backend, framework.** Bug fixes, new features, even refactors are welcome. The one ask: if a change is big enough to cost real time, open a [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) first so we can align before you build.

Every nodeset and graph is credited to its author/maintainer on the board below — with a citation link if it has an associated paper — so contributing here doesn't cost you authorship. The **AgentCanvas framework and all first-release components** are by **AC-Team**. The board is names-only by design: the **canonical inventory** with per-graph verification detail lives on the [doc-site Credits page](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html) and the [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vla-support-status.html) support-status pages.

### Credits

✅ verified — reproduces its paper / reference implementation · 🚧 runs end-to-end, verification in progress

<table>
  <thead align="center">
    <tr>
      <th>Environments</th>
      <th>Methods</th>
      <th>Models &amp; Policies</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/habitat.html">Habitat (VLN-CE)</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/matterport3d.html">MatterSim / MP3D</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/hmeqa.html">HM-EQA</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/openeqa.html">OpenEQA (EM-EQA)</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/simpler.html">SIMPLER</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/libero.html">LIBERO</a> ✅</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>VLN</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/navgpt.html">NavGPT</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/mapgpt.html">MapGPT</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/smartway.html">SmartWay</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/threestepnav.html">Three-Step Nav</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/aoplanner.html">AO-Planner</a> 🚧</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/discussnav.html">DiscussNav</a> 🚧</li>
              <li>Open-Nav 🚧</li>
              <li>SpatialNav 🚧</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/common/tools/basic-agent.html">Basic Agent toolkit</a> ✅</li>
            </ul>
          </li>
          <li><b>EQA</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/openeqa.html">EM-EQA baselines</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/explore-eqa.html">Explore-EQA</a> ✅</li>
              <li>ToolEQA 🚧</li>
            </ul>
          </li>
          <li><b>VLA (zero-shot)</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/voxposer.html">VoxPoser-LIBERO</a> ✅</li>
            </ul>
          </li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>Policies</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-cma.html">CMA</a> ✅</li>
              <li>Octo (SIMPLER baseline) ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-vla.html">VLA framework (Pi0 / SmolVLA / DP / DROID-DP)</a> 🚧</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-adapters.html">R2R-CE policy registry (12 variants)</a> 🚧</li>
            </ul>
          </li>
          <li><b>Perception &amp; foundation models</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/common/tools/sam.html">SAM</a> ✅</li>
              <li>BLIP-2 + Faster R-CNN ✅</li>
              <li>InstructBLIP ✅</li>
              <li>RAM ✅</li>
              <li>Grounding DINO ✅</li>
              <li>SpatialBot ✅</li>
              <li>Qwen2.5-VL ✅</li>
              <li>DetAny3D ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/common/foundation-models/vlm-prismatic.html">Prismatic VLM</a> ✅</li>
              <li>TSDF mapping ✅</li>
              <li>Semantic scene graph ✅</li>
            </ul>
          </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

**Call for contribution** — reserved slots, credited to whoever lands them ([how to contribute](CONTRIBUTING.md); IDs are roadmap slots on the [Credits page](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)):

<table>
  <thead align="center">
    <tr>
      <th>Benchmarks</th>
      <th>Methods</th>
      <th>Features &amp; Infra</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul>
          <li>AI2-THOR — ALFRED / TEACh <i>(E4)</i></li>
          <li>RxR-CE — multilingual VLN-CE <i>(E2)</i></li>
          <li>REVERIE — remote object grounding <i>(E3)</i></li>
          <li>OpenEQA A-EQA — active EQA <i>(E10)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>HAMT — hierarchical history transformer <i>(M5)</i></li>
          <li>DUET — dual-scale graph transformer <i>(M6)</i></li>
          <li>InstructNav — dynamic CoN + value maps <i>(M8)</i></li>
          <li>VLN-SIG — sub-instruction grounding <i>(M4)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Memory nodeset — episodic recall + semantic search <i>(F1)</i></li>
          <li>Parallel node execution — Pregel supersteps <i>(F3)</i></li>
          <li>Export graph as standalone Python <i>(F4)</i></li>
          <li>Docker server mode — Habitat / MP3D containers <i>(F7)</i></li>
          <li>ROS nodeset — real-robot deployment (<a href="#3-sim-to-real-path">§3</a>)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


---

## 9. Citation

If you use AgentCanvas in your research, please cite:

```bibtex
@misc{jian2026AgentCanvas,
  title         = {Automating the Design of Embodied Agent Architectures},
  author        = {Jian Zhou and Sihao Lin and Jin Li and Shuai Fu and Gengze Zhou and Qi Wu},
  year          = {2026},
  eprint        = {2606.30111},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.30111}
}
```

## 10. License

Apache License 2.0 — see [LICENSE](LICENSE).
