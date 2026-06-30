---
description: Log session changes, triage doc-site updates, check off completed TODOs, then commit to master with conventional commit format
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, Skill
---

# Commit Command

Log the current conversation's changes and commit them to `master`. Does NOT push.

## Instructions

### Step 1: Run /log

First, invoke the `/log` skill to update `.claude/log.md` and `.claude/project_overview.md` with all modifications from the current conversation.

### Step 2: Doc-site triage (ADR vs feature-docs vs skip)

Review the conversation and classify the changes into one of three categories:

#### Category A — Architectural decision → `/adr`
The session introduced **new abstractions, design patterns, structural changes, or major API redesigns** that affect how the system is organized. Examples: new base class hierarchy, new execution model, new wire system, new storage strategy.

→ Invoke the `/adr` skill for each architectural decision. Derive context, alternatives, and rationale from the conversation — do NOT ask the user.

#### Category B — Feature-level change → `/update-docs`
The session made **substantive changes to existing features** that are documented in the doc-site but don't introduce new architectural patterns. Examples: added a config field to a node, fixed executor behavior, added tools to a toolset, changed API response shape, completed a TODO item, added a new node type to an existing system.

→ Invoke the `/update-docs` skill to update the relevant feature docs, glossary, roadmap, etc.

#### Category C — Trivial change → skip
The session was purely a **bug fix with no behavior change**, docs-only update, formatting cleanup, dependency bump, or exploratory work with no code changes.

→ Skip doc updates silently.

**Guidelines for triage:**
- If unsure between A and B, choose B (feature-docs). ADRs are for decisions that future developers need to understand *why* the system is shaped this way.
- A session can trigger both A and B if it contains both an architectural decision and feature-level changes.
- Most sessions will be Category B or C. ADRs should be rare (~1-2 per major feature).

### Step 3: TODO check-off

Before committing, check whether the session's changes complete any open TODO items:

1. Invoke `/fetchtodo` to retrieve the current TODO list.
2. If the list is empty, skip to Step 4.
3. Compare each open TODO against the changes made in this conversation. A TODO is **done** if the conversation's code or doc changes fully satisfy its description.
4. For every TODO that is now done, invoke `/completetodo` with the TODO's ID.
5. If no TODOs match, skip silently — do NOT ask the user.

### Step 4: Stage and Commit

1. Run `git status` to see all modified files (never use `-uall` flag). **Pre-flight dirty-tree check:** compare the files this conversation actually touched against the full `git status`. If the tree carries a lot of dirty/untracked files *beyond* this conversation's scope (unrelated WIP, dirty submodules), note it — that is the leading signal that the git-hook's whole-tree stash (Tier-1 below) will conflict, and Tier-2 will be needed. Do not try to clean or stash the user's unrelated WIP yourself.
2. Run `git diff --cached --stat` and `git diff --stat` to review what will be committed.
3. Stage **all files that were modified in this conversation**, including `.claude/log.md`, `.claude/project_overview.md`, and any `docs/` files updated by the ADR or update-docs step. Stage files by name — do NOT use `git add -A` or `git add .`. Stage **whole files** (never partial-hunk staging) so the staged set is exactly what the hooks will verify.
4. Write the commit message in **Conventional Commits** format:

```
type(scope): short imperative description
```

- **Type:** `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `ci`, `perf`
- **Scope:** `backend`, `frontend`, `eval`, `orchestrator`, `data`, `infra`
- Lowercase, imperative mood, max 72 chars

Examples:
```
feat(orchestrator): add VLM scene captioning to NavGPT-CE
fix(backend): correct TURN_LEFT action mapping
chore: configure pre-commit hooks
```

5. If `$ARGUMENTS` is provided, use it as the commit message subject line.
6. **Tier-1 — normal commit (git-hook runs the hooks).** Commit using a HEREDOC:

```bash
git commit -m "$(cat <<'EOF'
type(scope): subject line

Body details here if needed.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

   If this succeeds, go to step 8. If it **fails**, read the failure and classify it:
   - **Real hook violation** — a hook reports an actual problem in the *staged* content (a lint error, a private key, a non-conventional subject). → Fix the staged files, re-stage by name, retry Tier-1. Never route a real violation to Tier-2.
   - **Stash/restore conflict** — the failure is `[ERROR] ... Stashed changes conflicted with hook auto-fixes`, or a rollback message, i.e. pre-commit could not park-and-restore the dirty working tree (see the pre-flight check). The staged content is not the problem; the whole-tree stash machinery is. → Go to Tier-2.

7. **Tier-2 — verified `--no-verify` (only for a stash/restore conflict, never a real violation).** Verify the hooks on exactly the staged set *without* the whole-tree stash, then commit skipping only the redundant git-hook re-run:

```bash
# Verify hooks against the precise staged file list (no whole-tree stash at this scope).
git diff --cached --name-only | xargs pre-commit run --files
```

   - If **every hook passes green** → the committed content is hook-verified out-of-band; commit with `--no-verify` (same HEREDOC message as Tier-1, plus `--no-verify`). The flag skips only the tree-conflicting re-run of hooks you just confirmed pass — it does **not** skip verification.
   - If **any hook fails or modifies a file** → it is a real violation after all. Fix the staged files, re-stage, and return to Tier-1. Do **not** `--no-verify` past a red `pre-commit run --files`.

   Record in the commit body (or your final summary) that Tier-2 was used and that `pre-commit run --files` passed — so the `--no-verify` is auditable, not silent.

8. Run `git status` after committing to verify success.

### Step 5: Conversation summary (final output)

After the commit succeeds, invoke the `/summary` skill. Its output becomes the last thing in the response — a readable record of the session's narrative arc (origin / process including detours / outcome) that the commit message intentionally omits.

This is in addition to, not a replacement for, the commit message. Do not paraphrase or pre-empt `/summary`'s output here; just invoke it.

### Important Rules

- Do NOT push to remote — the user will do that
- Do NOT use `git add -A` or `git add .` — stage specific files only
- Do NOT amend previous commits — always create a new commit
- **Never commit hook-unverified content.** The mandate is that the hooks *verify the staged content*, not that the git-hook stash machinery must run. `--no-verify` is permitted **only** via the Tier-2 path — after `pre-commit run --files <staged set>` has independently passed green — and **only** when Tier-1 failed on a stash/restore conflict, never to bypass a real hook violation. A bare `--no-verify` with no prior `pre-commit run --files` green is forbidden.
- Do NOT try to clean, stash, or "tidy" the user's unrelated WIP to make Tier-1 pass — a dirty working tree is the user's normal state. Use Tier-2 instead of touching their tree.
- If there are no changes to commit after logging, say so and stop
