# Update Feature Docs

You are updating the doc-site to reflect **non-architectural changes** made in the current
conversation — feature enhancements, bug fixes, new config options, API additions, node
changes, etc. These are changes that don't warrant an ADR but still need the docs to stay
accurate.

## When to use

Use this skill when the session made changes that affect documented features but didn't
introduce new abstractions, pattern shifts, or structural decisions. Examples:
- Added a config field to a node
- Fixed a bug in the executor
- Added a new tool to a nodeset
- Changed an API endpoint's response shape
- Updated wire types or port definitions
- Completed a TODO item

If the change is **architectural** (new abstraction, new pattern, structural change), use
`/docs/adr` instead.
If the change is **trivial** (typo fix, comment update, formatting), skip doc updates entirely.

## Steps

### Step 1: Get timestamp

Run `date "+%Y-%m-%d %H:%M"` to get the precise timestamp. Use this exact value for ALL
`last updated:` headers and changelog entries.

### Step 2: Analyze the conversation

Review what was changed in this session. Identify:
- Which **features** were modified (canvas, executor, nodesets, wire types, plugin servers,
  graph system, state containers, base-canvas-node, graph nodes, execution logs, llm config)
- Whether any **new terms** were introduced (→ glossary update)
- Whether any **TODO items** were completed (→ roadmap update)
- Whether any **capabilities** were affected in how they're explained (→ capability narrative)
- Whether any **codebase map** entries need updating (new file/role mapping)

### Step 3: Map changes to docs

Determine which doc-site files need updating. Use this mapping:

| Change area | Primary doc | Also check |
|-------------|-------------|------------|
| Canvas nodes, UI, catalog | `design-docs/canvas-system.md` | `capabilities/customizable-node-system.md` |
| Dataflow/DAG executor | `design-docs/graph-executor.md` | `capabilities/graph-execution-engine.md` |
| Wire types, ports | `design-docs/wire-types.md` | `core/glossary.md` |
| NodeSets — concept, twist, decision | `ds-concepts/nodesets.md` | — |
| NodeSets — tutorial (hello-world + level-ups) | `ds-tutorial/nodesets.md` | — |
| NodeSets — recipes (conda env, debugging, eval metadata) | `ds-recipes/nodesets.md` | — |
| NodeSets — API reference (BaseNodeSet, REST, file map) | `ds-api-reference/nodesets.md` | — |
| Plugin servers, auto-hosted mode | `design-docs/plugin-servers.md` | `capabilities/isolated-runtime-environments.md` |
| State containers, dual-wire | `design-docs/state-containers.md` | `core/glossary.md` |
| BaseCanvasNode — concept, NodeUIConfig | `ds-concepts/base-canvas-node.md` | `capabilities/customizable-node-system.md` |
| BaseCanvasNode — tutorial | `ds-tutorial/base-canvas-node.md` | — |
| BaseCanvasNode — recipes | `ds-recipes/base-canvas-node.md` | — |
| BaseCanvasNode — API reference | `ds-api-reference/base-canvas-node.md` | — |
| Graph save/load, kind field | `design-docs/graph-node-system.md` | `capabilities/nested-graph-system.md` |
| Graph model, GraphDefinition | `design-docs/graph-system.md` | — |
| Execution logs, `_self_log` | `design-docs/execution-logs.md` | `capabilities/real-time-observability.md` |
| LLM/VLM config, profiles | `design-docs/llm-config-system.md` | — |
| Hook system (lifecycle shell hooks) | `capabilities/hook-system.md` | — |
| Any-agent-form (DAG, cyclic, composition) | `capabilities/any-agent-form.md` | `design-docs/graph-executor.md` |
| Nodeset catalog pages | `nodesets/<name>.md` | `nodesets/index.md` |
| New terms introduced | `core/glossary.md` | — |
| TODO completed | `core/roadmap.md` | — |
| Architecture/data flow | `core/architecture.md` | — |
| Codebase map (file → role) | `core/codebase-map.md` | — |

**Rule for Diátaxis-split topics** (nodesets, base-canvas-node): update the *specific quadrant*
that changed. Don't touch all four. E.g. if a new REST endpoint was added, only
`ds-api-reference/nodesets.md` needs a row; if a new debugging tip was learned, only
`ds-recipes/nodesets.md`.

Only update docs that are actually affected. Do NOT touch unrelated docs.

### Step 4: Read affected docs

Read each doc file identified in Step 3 **in parallel**. Understand the current content so you
can make targeted updates (not rewrites).

### Step 5: Update each affected doc

For each doc:

1. **Modify the relevant content section** — add, update, or remove content to reflect the
   changes. Be precise and match the existing style.
2. **Update the `last updated:` timestamp** at the top to the value from Step 1.
3. **Add a changelog entry** at the bottom:
   ```
   - YYYY-MM-DD HH:MM: [concise description of what changed]
   ```

### Step 6: Summary

Print a summary:
```
Docs updated:
  - design-docs/canvas-system.md — [what changed]
  - capabilities/customizable-node-system.md — updated NodeUIConfig example
  - core/glossary.md — added 2 terms
  ...
```

## Important Rules

- Do NOT create new doc files — only update existing ones. If a new feature doc is needed,
  tell the user to run `/docs/add-doc` separately.
- Do NOT modify `core/decisions.md` — that's `/docs/adr`'s job.
- Do NOT rewrite entire sections — make surgical, targeted edits.
- Match the existing style and formatting of each doc.
- If a doc has skeleton/placeholder content, update it rather than noting "needs content."
- For Diátaxis-split topics, pick the *single correct quadrant* — don't propagate the same
  edit across all four files.
