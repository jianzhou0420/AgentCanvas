# /memory:hygiene — after-run cleanup for auto-memory writes

After an AAS (or any) session that may have written to `.claude/memory/`, use this skill to review the new entries and remove pollution before it bleeds into the next session.

## Why

Claude Code's auto-memory system writes "surprise / non-obvious" facts to `.claude/memory/` during a session. When that session is an AAS (`/architect:*`) run, many of those writes are **graph-specific fitting noise** — they read like global truths but only hold on the one graph/profile the run was on, and they pollute the next run's proposer context.

See `.claude/commands/experiment/EXPERIMENT_INSTRUCTIONS.markdown` § 2.1 ("Pollution → Stale memory facts") and § 5 ("Post-run hygiene → Memory writes are surprise-only") for the threat model. This skill is the after-run hygiene step that doc references.

## Usage

```
/memory:hygiene                      # review all uncommitted memory writes (default)
/memory:hygiene --scope architect    # restrict to one subdir
/memory:hygiene --since HEAD~3       # also include the last 3 commits
```

## Steps

### Step 1 — Detect candidates

```bash
python3 .claude/commands/memory/data/memory_diff.py [--scope <subdir>] [--since <ref>]
```

That prints a numbered list of new/modified `.claude/memory/*.md` files with previews. If the output is `[memory_diff] no candidates — .claude/memory/ is clean`, stop here and tell the user.

### Step 2 — Read + classify each candidate

For every candidate, Read the full file. Classify into one of three buckets:

- **KEEP** — framework-level real fact that applies across graphs/methods (engine bug, port/wire/state semantics, lifecycle invariant, API contract). Belongs in global memory at its current path.
- **DELETE** — graph-specific or single-run finding ("smartway_ce iter_3 SR jumped to 0.42 after persist=true", "this nodeset on this profile needs worker_count=8"), or content that has been superseded / refers to removed code paths. Does NOT generalize or is already wrong. Remove from disk (no forensic archive — `stale/` was retired 2026-05-19).
- **RELOCATE** — content is correct and worth keeping, but lives in the wrong subdir per `CLAUDE.md § Memory Layout` (e.g. a paper-scoping memo found in `nodeset/` instead of `research/`). `git mv` to the right subdir.

**Classification heuristics:**

| Signal in the memory body | Likely class |
|---|---|
| Names a specific graph, run_id, iter_M, or quantitative metric | DELETE |
| Names a `workspace/architect/exp_profiles/{graph}.yaml`, an `outputs/design_runs/...` path, or per-episode artefact | DELETE |
| Refers to code paths / files / classes that no longer exist (verify with `ls` / `grep`) | DELETE |
| Near-duplicate of another existing entry | DELETE the less-detailed one |
| Talks about engine semantics, port/wire/state contracts, lifecycle, framework invariants | KEEP |
| Records the user's preference / workflow rule (`feedback_*`) | usually KEEP |
| Content is correct but subdir doesn't match CLAUDE.md's routing table | RELOCATE |

When in doubt: **DELETE > RELOCATE > KEEP** — easier to re-add a real fact than to clean a recurring pollution. Ask the user before deleting if there's a reasonable case for KEEP.

### Step 3 — Present proposal table

Output one markdown table to the user, plus a one-line `why` for each row pulled from your classification reasoning:

```
| # | path                                              | proposed | why                                       |
|---|---------------------------------------------------|----------|-------------------------------------------|
| 1 | .claude/memory/architect/feedback_iter3_sr_jump.md| DELETE   | mentions smartway_ce iter_3 number        |
| 2 | .claude/memory/platform/feedback_executor_bug.md  | KEEP     | engine-level lifecycle rule               |
| 3 | .claude/memory/nodeset/feedback_paper_scoping.md  | RELOCATE | paper-scoping decision → belongs in research/ |
```

Then ask the user in plain prose (no `AskUserQuestion`):

> Apply this plan? Or tell me which rows to override (e.g. "change #1 to KEEP, #3 to DELETE").

Wait for the user's reply. Apply their overrides to the plan.

### Step 4 — Execute approved plan

For each row in the final plan:

- **KEEP**: no-op (file stays where it is).
- **DELETE**:
  - If untracked: `rm <path>`
  - If tracked: `git rm <path>`
  - Also remove the index line from `.claude/memory/MEMORY.md` (look for `[Title](<subdir>/<basename>)` and delete the bullet).
- **RELOCATE**:
  - Move: `git mv <old-path> .claude/memory/<new-subdir>/<basename>` if tracked, else `mv ...`
  - Update `.claude/memory/MEMORY.md`: cut the index line from its current `## <old-subdir>/` section, paste under the `## <new-subdir>/` section, and rewrite the path prefix in the link target.
  - If the source subdir's shareability differs from the destination's (per CLAUDE.md § Memory Layout — `working-style/` and `user/` are local-only, everything else is tracked), `git mv` will surface the right add/remove; double-check `git status` shows the rename, not separate D/A pairs.

### Step 5 — Offer commit

After execution:

```bash
git status .claude/memory/
```

Show the user the diff summary, then ask:

> Commit this cleanup now? Suggested subject: `chore(memory): hygiene after <short-tag>`.

If yes: stage only the affected files (and `MEMORY.md`), commit with that subject and a body listing the DELETE / RELOCATE moves. **Do not use `git add -A`**. Follow `.claude/standard/git-conventions.md`.

If no: leave staged/unstaged state as is and print final `git status` so the user can finish manually.

## Notes

- This skill is **not** auto-invoked anywhere. Run it explicitly after an AAS session, or fold it into your own wrap-up routine.
- **DELETE on an untracked file is permanent** (no git history to recover from). When unsure, ask the user before deleting an untracked candidate.
- **KEEP means "this is genuine global memory"** — if an entry is actually variant- or run-scoped knowledge that's still valuable for one specific run, propose DELETE here and copy the content into that run's `lineage.md` before approving.
- `MEMORY.md` edits must be hand-applied via the Edit tool (not regex), since the index has per-subdir `##` sections — see `CLAUDE.md § Memory Layout`.
- `stale/` subdir was retired 2026-05-19. Do not recreate it; the forensic-archive role was deemed not worth the carrying cost.

## Limits

- Does not detect pollution **inside** a file (e.g. a KEEP file that has one polluted paragraph). You'd need to read and edit those by hand.
- Does not auto-classify; classification quality depends on the calling Claude reading each candidate carefully. If you skim, you'll mis-classify.
- `--since <ref>` only sees git-committed changes; uncommitted auto-writes are always included regardless.

## See also

- `.claude/commands/memory/data/memory_diff.py` — the detection helper this skill drives
- `.claude/commands/experiment/EXPERIMENT_INSTRUCTIONS.markdown` § 5 — when to run this skill in the AAS flow
- `CLAUDE.md § Memory Layout` — subdir routing rules (KEEP candidates land in the right subdir)
- `.claude/standard/git-conventions.md` — commit format for Step 5
