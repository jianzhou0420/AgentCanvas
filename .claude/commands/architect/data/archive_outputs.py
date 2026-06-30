#!/usr/bin/env python3
"""archive_outputs.py — three-phase archival of outputs/design_runs/.

Phase 1: rename each live `v{N}/` under outputs/design_runs/{method}/{graph}/
         into the sibling `_archive/v{N}_<YYYY-MM-DD>_<HHMMSS>[_<reason>]/`.
         Same-FS atomic rename. Skips a v{N}/ only if it has an iter still
         in flight — i.e. its `.staging/` directory is non-empty. A finished
         loop has committed every iter from `.staging/iter_{n}/` into
         `iteration/`, leaving `.staging/` empty. A bare `.loop_state/` dir
         does NOT count as live: the loop creates it at start and never
         deletes it on termination, so its presence (even empty) means a
         loop *ran here*, not that one is *running now*. --abandon-graphs
         forces a move regardless. Skips already-`_`-prefixed dirs (the
         existing _archive/, _legacy/ trees).

Phase 2: rsync the in-tree `outputs/design_runs/` to a sibling archive root
         (default: <repo>/../_outputs_archive/outputs/design_runs/).
         Incremental, source preserved (rsync without --delete, no
         --remove-source-files).

Phase 3 (opt-in, --prune): after the phase 2 rsync, verify each in-repo
         `_archive/` dir (the phase 1 renames) and the `outputs/archive/`
         mirror against its external copy with a checksum dry-run rsync,
         then `rm -rf` each verified in-repo copy so the working tree
         actually shrinks. A dir whose copy does not verify is left in
         place. Live `v{N}/` dirs are never touched.

Defaults are chosen so a bare
`python .claude/commands/architect/data/archive_outputs.py --dry-run`
prints the full plan without touching anything.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # repo root: .claude/commands/architect/data/<this>
OUTPUTS_DIR = REPO / "outputs"
DESIGN_RUNS_DIR = OUTPUTS_DIR / "design_runs"
DEFAULT_TARGET = REPO.parent / "_outputs_archive"
DEFAULT_MIRROR_SUBDIRS = ("archive", "design_runs")  # eval_runs (177G) + runs (2.5G) opt-in


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _is_version_dir(d: Path) -> bool:
    """A live v{N}/ candidate: name starts with 'v' followed by an int."""
    if not d.is_dir() or d.name.startswith("_"):
        return False
    if not d.name.startswith("v"):
        return False
    suffix = d.name[1:]
    head = suffix.split("_", 1)[0]
    return head.isdigit()


def _loop_is_live(vdir: Path) -> bool:
    """True if v{N}/ has an iter still in flight.

    A loop commits each finished iter from `.staging/iter_{n}/` into
    `iteration/iter_{n}/` (mv on success, rm -rf on SKIP), so a settled loop
    leaves `.staging/` empty. A non-empty `.staging/` means an iter is
    mid-flight — or a crashed loop left debris; either way, don't archive
    without an explicit --abandon-graphs.

    A bare `.loop_state/` dir is NOT a liveness signal: the loop creates it
    at start and never deletes it on termination, so its presence (even
    empty) only means a loop *ran here*, not that one is *running now*.
    """
    staging = vdir / ".staging"
    if not staging.is_dir():
        return False
    return any(staging.iterdir())


def _graphs_with_live_loop(method_dir: Path) -> list[str]:
    """Return graph names whose live v{N}/ has an iter still in flight."""
    flagged: list[str] = []
    for graph_dir in method_dir.iterdir():
        if not graph_dir.is_dir() or graph_dir.name.startswith("_"):
            continue
        for vdir in graph_dir.iterdir():
            if not _is_version_dir(vdir):
                continue
            if _loop_is_live(vdir):
                flagged.append(f"{method_dir.name}/{graph_dir.name}")
                break
    return flagged


def _plan_phase1(
    abandon_graphs: set[str],
    reason: str | None,
    methods: set[str] | None,
) -> list[tuple[Path, Path, str]]:
    """Return list of (src, dst, status_tag) for every v{N}/ rename action."""
    if not DESIGN_RUNS_DIR.exists():
        return []

    ts = _ts()
    actions: list[tuple[Path, Path, str]] = []

    for method_dir in sorted(DESIGN_RUNS_DIR.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith("_"):
            continue
        if methods is not None and method_dir.name not in methods:
            continue

        for graph_dir in sorted(method_dir.iterdir()):
            if not graph_dir.is_dir() or graph_dir.name.startswith("_"):
                continue

            graph_key = f"{method_dir.name}/{graph_dir.name}"
            archive_dir = graph_dir / "_archive"

            for vdir in sorted(graph_dir.iterdir()):
                if not _is_version_dir(vdir):
                    continue

                if _loop_is_live(vdir) and graph_key not in abandon_graphs:
                    actions.append((vdir, vdir, "SKIP_LOOP_LIVE"))
                    continue

                new_name = f"{vdir.name}_{ts}"
                if reason:
                    new_name += f"_{reason}"
                dst = archive_dir / new_name
                actions.append((vdir, dst, "MOVE"))

    return actions


def _execute_phase1(actions: list[tuple[Path, Path, str]], dry_run: bool) -> int:
    n_moved = 0
    for src, dst, status in actions:
        if status == "SKIP_LOOP_LIVE":
            print(
                f"[phase1] SKIP {src.relative_to(REPO)}  "
                f"(.staging/ non-empty — iter in flight; pass --abandon-graphs "
                f"to override)"
            )
            continue
        if status == "MOVE":
            arrow = "→ (dry)" if dry_run else "→"
            print(f"[phase1] mv  {src.relative_to(REPO)}  {arrow}  {dst.relative_to(REPO)}")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    print(f"[phase1] ERR dst already exists: {dst}", file=sys.stderr)
                    return -1
                shutil.move(str(src), str(dst))
            n_moved += 1
    return n_moved


def _execute_phase2(target_root: Path, subdirs: tuple[str, ...], dry_run: bool) -> int:
    target_root.mkdir(parents=True, exist_ok=True)
    rc_total = 0
    for sub in subdirs:
        src = OUTPUTS_DIR / sub
        if not src.exists():
            print(f"[phase2] WARN {src} does not exist; skipping")
            continue
        dst = target_root / "outputs" / sub
        dst.parent.mkdir(parents=True, exist_ok=True)
        # rsync without --delete: incremental, additive only.
        # dry-run gets stats-only; live gets progress so user sees the 41G crawl.
        info_flag = "--info=stats2" if dry_run else "--info=progress2,stats2"
        cmd = [
            "rsync",
            "-a",
            info_flag,
            f"{src}/",
            f"{dst}/",
        ]
        if dry_run:
            cmd.insert(1, "--dry-run")
        print(f"[phase2] {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[phase2] ERR rsync exit {rc} for {sub}", file=sys.stderr)
            rc_total = rc
    return rc_total


def _plan_phase3(
    target_root: Path,
    methods: set[str] | None,
    extra_archive_dirs: set[Path] | None = None,
) -> list[tuple[Path, Path]]:
    """Return (in_repo_src, archive_dst) pairs for every prune candidate.

    Candidates: each in-repo `<method>/<graph>/_archive/` dir under
    design_runs/, plus the `outputs/archive/` mirror dir. The `outputs/archive/`
    mirror is included only on a full run (no --methods filter), since it is
    not method-scoped.

    `extra_archive_dirs` lets the dry-run preview include `_archive/` dirs that
    a same-invocation phase 1 *would* create but that do not exist on disk yet.
    """
    srcs: set[Path] = set(extra_archive_dirs or set())
    if DESIGN_RUNS_DIR.exists():
        for method_dir in sorted(DESIGN_RUNS_DIR.iterdir()):
            if not method_dir.is_dir() or method_dir.name.startswith("_"):
                continue
            if methods is not None and method_dir.name not in methods:
                continue
            for graph_dir in sorted(method_dir.iterdir()):
                if not graph_dir.is_dir() or graph_dir.name.startswith("_"):
                    continue
                arch = graph_dir / "_archive"
                if arch.is_dir() and any(arch.iterdir()):
                    srcs.add(arch)

    targets: list[tuple[Path, Path]] = []
    for arch in sorted(srcs):
        rel = arch.relative_to(OUTPUTS_DIR)
        targets.append((arch, target_root / "outputs" / rel))
    if methods is None:
        archive_mirror = OUTPUTS_DIR / "archive"
        if archive_mirror.is_dir() and any(archive_mirror.iterdir()):
            targets.append((archive_mirror, target_root / "outputs" / "archive"))
    return targets


def _verify_synced(src: Path, dst: Path) -> tuple[bool, str]:
    """True iff every file under `src` exists byte-identical under `dst`.

    Runs a checksum dry-run rsync; any file rsync would still transfer means
    the archive copy is incomplete and the source must NOT be deleted.
    """
    if not dst.is_dir():
        return False, f"archive copy missing: {dst}"
    res = subprocess.run(
        ["rsync", "-rcn", "--out-format=%n", f"{src}/", f"{dst}/"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return False, f"verify rsync exit {res.returncode}: {res.stderr.strip()}"
    pending = [ln for ln in res.stdout.splitlines() if ln.strip() and not ln.endswith("/")]
    if pending:
        return False, f"{len(pending)} file(s) not in archive (e.g. {pending[0]})"
    return True, "verified"


def _execute_phase3(
    target_root: Path,
    methods: set[str] | None,
    dry_run: bool,
    extra_archive_dirs: set[Path] | None = None,
) -> int:
    """Verify each archived dir against its external copy, then rm the source.

    Returns the count of dirs pruned. A dir whose verification fails is left
    in place and reported — never deleted on an unverified copy.
    """
    targets = _plan_phase3(target_root, methods, extra_archive_dirs)
    if not targets:
        print("[phase3] nothing to prune")
        return 0

    n_pruned = 0
    for src, dst in targets:
        rel = src.relative_to(REPO)
        if dry_run:
            print(f"[phase3] would verify {rel} against {dst}, then rm if verified")
            continue
        ok, detail = _verify_synced(src, dst)
        if not ok:
            print(f"[phase3] KEEP {rel}  (NOT verified: {detail})", file=sys.stderr)
            continue
        print(f"[phase3] rm  {rel}  ({detail} against {dst})")
        shutil.rmtree(src)
        n_pruned += 1
    return n_pruned


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without touching files (recommended first)",
    )
    p.add_argument("--phase", choices=["1", "2", "both"], default="both")
    p.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"phase 2 destination root (default: {DEFAULT_TARGET})",
    )
    p.add_argument(
        "--mirror",
        default=",".join(DEFAULT_MIRROR_SUBDIRS),
        help=f"comma-separated outputs/ subdirs to mirror in phase 2 "
        f"(default: {','.join(DEFAULT_MIRROR_SUBDIRS)})",
    )
    p.add_argument(
        "--methods",
        default="",
        help="comma-separated method names to include in phase 1 + phase 3 "
        "(empty = all; a method filter also excludes the non-scoped "
        "outputs/archive/ mirror from phase 3 pruning)",
    )
    p.add_argument("--reason", default="", help="optional reason tag appended to renamed v{N} dirs")
    p.add_argument(
        "--abandon-graphs",
        default="",
        help="comma-separated 'method/graph' entries to move even if an "
        "iter looks in flight (non-empty .staging/)",
    )
    p.add_argument(
        "--prune",
        action="store_true",
        help="phase 3: after phase 2 rsync, verify each in-repo _archive/ dir "
        "(and outputs/archive/) against its external copy, then rm the verified "
        "in-repo copy so the working tree shrinks",
    )
    args = p.parse_args()

    methods = set(filter(None, args.methods.split(","))) or None
    abandon = set(filter(None, args.abandon_graphs.split(",")))
    mirror_subdirs = tuple(filter(None, args.mirror.split(",")))
    reason = args.reason.strip() or None

    print(f"[arch] repo            = {REPO}")
    print(f"[arch] outputs         = {OUTPUTS_DIR}")
    print(f"[arch] target root     = {args.target}")
    print(f"[arch] phase           = {args.phase}")
    print(f"[arch] dry_run         = {args.dry_run}")
    print(f"[arch] methods filter  = {methods or 'ALL'}")
    print(f"[arch] mirror subdirs  = {mirror_subdirs}")
    print(f"[arch] abandon_graphs  = {abandon or 'NONE'}")
    print(f"[arch] reason tag      = {reason or '(none)'}")
    print(f"[arch] prune           = {args.prune}")
    print(flush=True)

    planned_archive_dirs: set[Path] = set()
    if args.phase in ("1", "both"):
        # Show what graphs still have an iter in flight so the user can
        # re-run with --abandon-graphs if they wish.
        flagged: list[str] = []
        if DESIGN_RUNS_DIR.exists():
            for method_dir in sorted(DESIGN_RUNS_DIR.iterdir()):
                if not method_dir.is_dir() or method_dir.name.startswith("_"):
                    continue
                flagged.extend(_graphs_with_live_loop(method_dir))
        if flagged:
            print(f"[phase1] graphs with an iter in flight (.staging/ non-empty): {flagged}")
        actions = _plan_phase1(abandon, reason, methods)
        planned_archive_dirs = {dst.parent for src, dst, status in actions if status == "MOVE"}
        if not actions:
            print("[phase1] nothing to do")
        else:
            rc = _execute_phase1(actions, args.dry_run)
            if rc < 0:
                return 2
            print(f"[phase1] {rc} version dir(s) {'would be ' if args.dry_run else ''}moved")
        print()

    if args.phase in ("2", "both"):
        rc = _execute_phase2(args.target, mirror_subdirs, args.dry_run)
        if rc != 0:
            return rc

        if args.prune:
            print()
            n = _execute_phase3(args.target, methods, args.dry_run, planned_archive_dirs)
            verb = "would be pruned" if args.dry_run else "pruned"
            print(f"[phase3] {n} archived dir(s) {verb}")
    elif args.prune:
        print("[phase3] --prune requires phase 2 (rsync) to run; skipped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
