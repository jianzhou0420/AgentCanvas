# /architect:cleanup — wrap up after an AAS run

After `/architect:<variant>:loop` finishes (or you stop one), invoke this skill to wrap up. Two actions, both dry-run-first / human-approved before any destructive change:

1. **Archive `outputs/design_runs/`** so the live tree stays small and the next run starts on a clean slate.
2. **Scrub auto-memory writes** via `/memory:hygiene` so this run's graph-specific findings don't pollute the next session's proposer context.

## Why

Each AAS cycle leaves two kinds of debris:

- `outputs/design_runs/<method>/<graph>/v{N}/` directories accumulate fast (one per cycle); without periodic archival the working tree gets cluttered and disk fills.
- Claude's auto-memory may have written "surprise / non-obvious" facts mid-run; some are universal, but most are graph-specific fitting noise. See `.claude/commands/experiment/EXPERIMENT_INSTRUCTIONS.markdown` § 1.5.1 (Pollution → Stale memory facts).

This skill is the load-bearing step that prevents an iter's pollution from contaminating the next.

## Usage

```
/architect:cleanup
```

No arguments. The skill always runs interactively (dry-run → user approves → live).

## Steps

### Step 1 — Archive `outputs/design_runs/` (dry-run)

```bash
python3 .claude/commands/architect/data/archive_outputs.py --dry-run --prune
```

Prints the full three-phase plan:
- **Phase 1** — which finished `v{N}/` dirs would be renamed under their `_archive/` sibling (a `v{N}/` is skipped only if its `.staging/` is non-empty, i.e. an iter is in flight).
- **Phase 2** — which `outputs/` subdirs would be rsynced to the sibling archive root (default `<repo>/../_outputs_archive/`).
- **Phase 3 (`--prune`)** — which in-repo `_archive/` dirs (and the `outputs/archive/` mirror) would be verified against their external copy and then `rm`'d so the working tree actually shrinks.

No files are touched.

Show the plan to the user, then ask:

> Apply this archive plan? Or skip archiving this round?

Wait for reply. If the user wants to filter (e.g. "only archive method X"), pass `--methods <name>` to the live invocation in Step 2.

### Step 2 — Archive `outputs/design_runs/` (live)

If the user confirmed:

```bash
python3 .claude/commands/architect/data/archive_outputs.py --prune [--methods <names>] [--reason <tag>]
```

(Drop `--dry-run`.) Executes the same plan for real. Phase 3 only `rm`'s an in-repo `_archive/` dir after a checksum dry-run rsync confirms its external copy is byte-complete — an unverified dir is kept and reported on stderr. Print the helper's stdout — the final stat lines name how many version dirs were moved and how many archived dirs were pruned.

If the user skipped archive: print `Skipped archive.` and proceed. The cycle's output stays in `outputs/design_runs/` until next time.

### Step 3 — Scrub auto-memory writes

Invoke `/memory:hygiene`. That skill runs its own propose-and-confirm flow:

- Detects new/modified `.claude/memory/*.md` entries via `.claude/commands/memory/data/memory_diff.py`.
- Reads each candidate, classifies KEEP / DELETE / RELOCATE.
- Asks the user to approve or override the proposal table.
- Executes the approved plan (`rm` / `git rm` / `git mv` + MEMORY.md index update).
- Offers a cleanup commit.

Run this step **even if the user skipped Step 2**. Archive and memory hygiene are independent — memory pollution will recur regardless of where the run output lives.

### Step 4 — Print final status

After Step 3 returns, print a one-line summary:

```
Cleanup done: archived <N> version dirs, pruned <P> in-repo _archive dirs, <M> memory entries reviewed (<K> deleted, <J> relocated).
```

Pull the archive + prune counts from Step 2's output (`[phase1] … moved` and `[phase3] … pruned` lines) and the memory counts from `/memory:hygiene`'s execution log. If a step was skipped, say so explicitly.

## Notes

- This skill is **not** auto-invoked anywhere. Call it manually after each `/architect:<variant>:loop` run.
- **Don't invoke mid-cycle.** If `/architect:<variant>:loop` is still running, `archive_outputs.py` skips any `v{N}/` whose `.staging/` is non-empty (a committed iter empties `.staging/` into `iteration/`, so a non-empty one means an iter is in flight) — but the memory hygiene step would still catch incomplete writes. Wait for the loop to exit. Note: a `.loop_state/` dir is *not* a liveness signal — the loop creates it at start and never deletes it on termination, so it persists (often empty) after every finished run.
- Archive Step 1+2 only touches `outputs/`. Memory Step 3 only touches `.claude/memory/`. Independent — skipping one does not affect the other.
- For unusual situations (a crashed loop that left `.staging/iter_n/` debris, archive target full, etc.), invoke the underlying helpers directly with their own flags — e.g. `--abandon-graphs method/graph` force-archives a `v{N}/` despite a non-empty `.staging/`. This skill is the common path, not the only path.

## See also

- `.claude/commands/architect/data/archive_outputs.py` — the archive helper this skill drives
- `/memory:hygiene` — the memory scrubber this skill chains to
- `.claude/commands/experiment/EXPERIMENT_INSTRUCTIONS.markdown` § 2.2 — when in the AAS flow to run this
