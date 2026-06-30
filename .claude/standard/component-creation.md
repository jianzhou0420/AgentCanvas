# Component Creation Guide

Quick-reference index for creating AgentCanvas components. Read the
skill file for the type you're building, then follow its skeleton and checklist.

## What Are You Creating?

| If you need to... | Read this skill | Target directory |
|--------------------|----------------|------------------|
| Build a single canvas node (tool, transformer, data source) | `../tutorials/skill-canvas-node.md` | `workspace/nodesets/{role}/{name}.py` (role per `nodeset-layout.md`) |
| Group multiple nodes with shared lifecycle | `../tutorials/skill-nodeset.md` | `workspace/nodesets/{role}/{name}.py` |
| Wrap a simulator or interactive environment | `../tutorials/skill-env-nodeset.md` | `workspace/nodesets/env/env_{sim}.py` |
| Define a saved graph or reusable composite node | `../tutorials/skill-graph-json.md` | `workspace/graphs/` or `workspace/graph_nodes/` |

## Universal Rules

These apply to every component type:

- **Import line**: `from app.components import BaseCanvasNode, BaseNodeSet, PortDef`
- **Future annotations**: `from __future__ import annotations` at top of every file
- **Node type naming**: `"{nodeset}__{node}"` (double underscore separator)
- **One NodeSet per file**: all its node classes live in the same file, above the NodeSet class
- **Wire types**: `IMAGE`, `DEPTH`, `ACTION`, `STATE`, `TEXT`, `BOOL`, `METRICS`, `OBSERVATION`, `STEP_RESULT`, `ANY`
- **File location & naming**: all components go in `workspace/nodesets/` (auto-discovered by WorkspaceComponentRegistry); the hierarchy, role prefixes, and folder-vs-file rules are in `nodeset-layout.md` — read it before placing or naming any file
- **LLM profiles are user-level, not graph-level**: when authoring a saved graph that includes `llmCall` (or any future profile-bearing node), set `config.profile = ""` so it resolves to the user's active profile at run time. Profiles bundle `api_key + provider + model` and are per-deployment state — hardcoding a name in graph JSON breaks portability and can silently fail when the named profile points to a different / unfunded account. See `../tutorials/skill-graph-json.md` Design Principle #9.

## Doc-Site References (Deep Dives)

Don't duplicate these — point to them when more detail is needed.

| Topic | Doc path |
|-------|----------|
| Full tutorial (14 chapters) | `docs/tutorials/component-cookbook.md` |
| ClassVars, ConfigField, auto-discovery | `docs/capabilities/customizable-node-system.md` |
| BaseCanvasNode — Concepts | `docs/ds-concepts/base-canvas-node.md` |
| BaseCanvasNode — Tutorial | `docs/ds-tutorial/base-canvas-node.md` |
| BaseCanvasNode — Recipes | `docs/ds-recipes/base-canvas-node.md` |
| BaseCanvasNode — API Reference | `docs/ds-api-reference/base-canvas-node.md` |
| NodeSet system (local vs server) | `docs/ds-concepts/nodesets.md` (concepts) · `docs/ds-tutorial/nodesets.md` (tutorial) · `docs/ds-recipes/nodesets.md` (recipes) · `docs/ds-api-reference/nodesets.md` (API reference) |
| Environment nodeset tutorial | `docs/tutorials/habitat-nodeset.md` |
| Wire type catalog | `docs/design-docs/wire-types.md` |
| Working example code | `workspace/nodesets/example.py` |
