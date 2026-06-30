---
description: Summarize the current work streak from .claude/log.md — a daily wrap-up
allowed-tools: Read, Bash, Grep
---

# Call It A Day

Summarize all work done in the **current work streak** by reading `.claude/log.md`. A work streak is a continuous run of sessions without a long break between them.

## Steps

### Step 1: Get current time

```bash
date '+%Y-%m-%d %H:%M'
```

### Step 2: Read .claude/log.md

Read the log file. If it's too large, read from the top (most recent entries are at the top).

### Step 3: Identify the current work streak

Walk the session entries from **newest to oldest** (they're in reverse chronological order in the file). For each consecutive pair of sessions, compute the gap between them.

**A streak breaks when the gap between two consecutive sessions exceeds 6 hours.**

- Start from the most recent session
- Include each prior session as long as the gap to the next-newer session is ≤ 6 hours
- Stop when a gap > 6 hours is found — that's the streak boundary

Example: sessions at 23:13, 23:11, 23:01, 20:04, 15:51, 15:08, 15:02, 14:17, 14:17, 00:52 — gap between 14:17 and 00:52 is ~13 hours, so the streak is the 9 sessions from 14:17 to 23:13. The 00:52 session belongs to the previous streak.

If `$ARGUMENTS` contains a date (e.g. `2026-03-30`), summarize all sessions from that calendar date instead of using the streak logic.

### Step 4: Print the daily summary

Format the output as:

```
═══════════════════════════════════════════════════
  Daily Summary — YYYY-MM-DD [HH:MM – HH:MM]
═══════════════════════════════════════════════════

Sessions: N  |  Span: Xh Ym

1. [HH:MM] Session Title
   → bullet summary of what was done (1-2 lines from the Summary section)

2. [HH:MM] Session Title
   → bullet summary

...

───────────────────────────────────────────────────
Files touched: X created, Y modified, Z deleted
───────────────────────────────────────────────────
```

- **Date range**: show the date of the first session. If the streak spans midnight, show both dates (e.g. `2026-03-30 – 2026-03-31`)
- **Time range**: `[earliest HH:MM – latest HH:MM]`
- **Span**: total duration from first to last session in hours and minutes

### Rules for the summary

- List sessions in **chronological order** (earliest first)
- For each session, extract the **Summary** subsection and condense it to 1-2 lines
- At the bottom, aggregate the **Files Modified** tables from all sessions:
  - Count total files created, modified, deleted (deduplicate — if the same file was modified in two sessions, count it once)
- If no entries exist in the log, print: `No log entries found. Nothing to summarize.`
- If `$ARGUMENTS` is provided and is not a date, append it as a "Notes" line at the end
