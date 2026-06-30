# Fetch TODO list

Print the current open TODO list from `docs/core/roadmap.md`, with caching.

## Cache

- Cache file: `.claude/cache/todo-list.txt`
- First, run a **cache-check** via Bash (output only, no rendering):
  ```
  [ -f .claude/cache/todo-list.txt ] && [ .claude/cache/todo-list.txt -nt docs/core/roadmap.md ] && echo HIT || echo MISS
  ```
- On **HIT**: use the `Read` tool on `.claude/cache/todo-list.txt`, then **reprint the full contents verbatim as assistant message text** (inside a fenced code block). Do **not** use `cat` — its output gets folded in the conversation UI. Skip the regenerate steps.
- On **MISS**: regenerate (steps below), write to the cache, then reprint the rendered block as assistant message text (same no-`cat` rule).

## Regenerate (cache miss)

1. Read `docs/core/roadmap.md`
2. Collect all `[ ]` (open) items from these four sections:
   - `## TODO` — maintenance, refactors, docs (IDs: numeric)
   - `## Feature TODO` — new platform capabilities (IDs: F1, F2, …)
   - `## Env TODO` — environment/simulator integration (IDs: E1, E2, …)
   - `## Method TODO` — VLN research implementations (IDs: M1, M2, …)
   Skip `## Completed TODO` and any `[x]` items entirely.
3. Build each line showing ID, short **scope tag**, and brief description. Tags: `doc-site`, `frontend`, `backend`, `canvas`, `nodeset`, `executor`, `agent`, `infra`, `feature`, `env`, `method`. Format:
   ```
   [ ]   6  agent        NavGPT-CE completion ...
   [ ]  10  nodeset      Add ui_config to nodeset tools ...
   [ ]  F1  nodeset      Memory nodeset ...
   [ ]  E1  env          Matterport3D Simulator nodeset ...
   [ ]  M1  agent        NavGPT-CE canvas graph ...
   ```
4. Group by section with headers:
   ```
   --- TODO (N open) ---
   ...
   --- Feature TODO (N open) ---
   ...
   --- Env TODO (N open) ---
   ...
   --- Method TODO (N open) ---
   ...
   ```
5. Append a summary line: `N open total`
6. Write the full rendered block to `.claude/cache/todo-list.txt`, then print it.

## Invalidation

`/todo/add`, `/todo/complete`, `/todo/delete`, and any manual edit to `roadmap.md` update its mtime, which naturally invalidates the cache on the next fetch. No explicit invalidation needed.
