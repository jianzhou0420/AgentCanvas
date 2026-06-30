#!/usr/bin/env python3
"""memory_diff.py — list .claude/memory/ entries that need hygiene review.

After an AAS run, the auto-memory system may have written new entries to
`.claude/memory/*/`. This script lists candidates for review: each
new/modified file with its path, status, mtime, and a short preview.

Consumed by `/memory:hygiene` (the slash command) or by a human running
it directly. The Python side is intentionally dumb — it just detects;
classification (KEEP / DELETE / STALE) requires reading the content and
is done by the slash command (Claude) or the human.

Defaults to scanning uncommitted changes (working tree + untracked),
which is the common case right after an AAS run. Pass `--since <ref>`
to also include recently-committed writes.

Usage (run from repo root):
  python .claude/commands/memory/data/memory_diff.py                   # uncommitted only (default)
  python .claude/commands/memory/data/memory_diff.py --since HEAD~3    # also last 3 commits
  python .claude/commands/memory/data/memory_diff.py --scope architect # restrict to one subdir
  python .claude/commands/memory/data/memory_diff.py --json            # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # repo root: .claude/commands/memory/data/<this>
MEM_DIR = REPO / ".claude" / "memory"
MEM_REL = ".claude/memory"
PREVIEW_LINES = 8


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


def _collect_uncommitted() -> list[tuple[Path, str]]:
    """Parse `git status --porcelain` for .claude/memory/*.md changes."""
    out: list[tuple[Path, str]] = []
    raw = _git("status", "--porcelain", "--", MEM_REL)
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        flags, rel = line[:2], line[3:]
        # Handle rename: "R  old -> new" — take the new path
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        if not rel.endswith(".md"):
            continue
        if flags == "??":
            status = "NEW (untracked)"
        elif flags.strip() == "A":
            status = "NEW (staged)"
        elif "M" in flags:
            status = "MODIFIED"
        elif "D" in flags:
            continue  # deletions don't need hygiene
        elif flags.startswith("R"):
            status = "RENAMED"
        else:
            status = flags.strip() or "CHANGED"
        out.append((REPO / rel, status))
    return out


def _collect_since(since: str) -> list[tuple[Path, str]]:
    """Files changed in commits since `since`."""
    out: list[tuple[Path, str]] = []
    raw = _git("diff", "--name-status", f"{since}..HEAD", "--", MEM_REL)
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, rel = parts[0], parts[-1]
        if not rel.endswith(".md"):
            continue
        if code.startswith("A"):
            status = "NEW (committed)"
        elif code.startswith("M"):
            status = "MODIFIED (committed)"
        elif code.startswith("R"):
            status = "RENAMED (committed)"
        elif code.startswith("D"):
            continue
        else:
            status = f"{code} (committed)"
        out.append((REPO / rel, status))
    return out


def _scoped(candidates: list[tuple[Path, str]], scope: str | None) -> list[tuple[Path, str]]:
    if not scope:
        return candidates
    prefix = MEM_DIR / scope
    return [(p, s) for (p, s) in candidates if str(p).startswith(str(prefix) + "/")]


def _dedup(*lists: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Merge, keeping the uncommitted status if a path appears in both."""
    seen: dict[Path, str] = {}
    for lst in lists:
        for path, status in lst:
            if path not in seen:
                seen[path] = status
    return sorted(seen.items(), key=lambda kv: str(kv[0]))


def _preview(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"(read error: {e})"]
    return text.splitlines()[:PREVIEW_LINES]


def _build_records(pairs: list[tuple[Path, str]]) -> list[dict]:
    records: list[dict] = []
    for path, status in pairs:
        if not path.exists():
            # Renamed-away source — skip silently
            continue
        records.append(
            {
                "path": str(path.relative_to(REPO)),
                "status": status,
                "mtime": path.stat().st_mtime,
                "preview": _preview(path),
            }
        )
    return records


def _print_human(records: list[dict]) -> None:
    if not records:
        print("[memory_diff] no candidates — .claude/memory/ is clean")
        return
    print(f"[memory_diff] {len(records)} candidate(s) for hygiene review:")
    print()
    for i, c in enumerate(records, 1):
        print(f"  [{i}] {c['status']:18}  {c['path']}")
        for line in c["preview"]:
            print(f"         {line}")
        print()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--since", default=None, help="also include files changed since this git ref")
    p.add_argument(
        "--scope", default=None, help="restrict to one .claude/memory/ subdir (e.g. 'architect')"
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of human-readable text")
    args = p.parse_args()

    if not MEM_DIR.exists():
        print(f"[memory_diff] {MEM_REL}/ does not exist", file=sys.stderr)
        return 2

    uncommitted = _collect_uncommitted()
    committed = _collect_since(args.since) if args.since else []
    pairs = _dedup(uncommitted, committed)
    pairs = _scoped(pairs, args.scope)
    records = _build_records(pairs)

    if args.json:
        json.dump(records, sys.stdout, indent=2)
        print()
    else:
        _print_human(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
