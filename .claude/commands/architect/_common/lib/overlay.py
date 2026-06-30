"""Architect active_workspace overlay helper — files-contract.md §3, §7.

Shared by every variant's implementer. Two jobs, and deliberately NO
editing logic:

  * seed()        — copy-on-first-touch a frozen ``workspace/`` file (or
                    its whole enclosing Python package) into an iter's
                    ``active_workspace/`` overlay, so the implementer
                    sub-agent can then edit it with native Edit/Write.
  * check_target()— the §7 edit whitelist: a hard wall (block) plus a
                    soft off-scope warn.

The implementer sub-agent performs the actual editing natively once a
target is seeded. There is no patch DSL here — the typed ``graph_edits``
op-applier was retired 2026-05-20 (an agentic implementer edits files
directly; serialising intent into an op enum for a deterministic
replayer was an ADAS-era vestige). See files-contract.md §7 and each
variant README.

Path convention: ``rel`` arguments are workspace-prefixed, e.g.
``workspace/graphs/foo.json`` / ``workspace/nodesets/bar/__init__.py``
— the same form the proposer writes in a spec's ``targets`` list. The
``active_workspace/`` tree itself is rooted one level in
(``active_workspace/{graphs,nodesets}/...``), so the prefix is stripped
when computing the overlay destination.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys

_WS = "workspace/"


def _resolve_graph(ws_root: str, graph_name: str) -> str | None:
    """Locate ``graphs/{graph_name}.json`` under a workspace root.

    Accepts a relative-path id (``vln/verified/mapgpt_mp3d``) directly, and
    falls back to a recursive ``graphs/**/{stem}.json`` search for bare stems
    that live in a subfolder after the graphs-hierarchy reorg. Returns None
    if nothing matches.
    """
    flat = os.path.join(ws_root, "graphs", f"{graph_name}.json")
    if os.path.exists(flat):
        return flat
    if "/" not in graph_name and "\\" not in graph_name:
        matches = sorted(
            glob.glob(os.path.join(ws_root, "graphs", "**", f"{graph_name}.json"), recursive=True)
        )
        if matches:
            return matches[0]
    return None


# ── §7 edit whitelist ────────────────────────────────────────────────


def graph_prefixes(graph_json_path: str) -> set:
    """Distinct ``<nodeset>`` prefixes of every node's
    ``<nodeset>__<node>`` ``type`` in a graph JSON — i.e. the nodesets
    that graph uses."""
    with open(graph_json_path) as f:
        g = json.load(f)
    out = set()
    for n in g.get("nodes", []):
        t = n.get("type", "")
        if "__" in t:
            out.add(t.split("__", 1)[0])
    return out


def check_target(rel_path: str, graph_name: str, prefixes: set) -> bool:
    """Edit-whitelist check — files-contract.md §7's two boundaries.

    HARD WALL (blocking). Only paths under ``workspace/`` are editable.
    Everything else — framework ``agentcanvas/backend/app/**``, vendored
    ``third_party/**`` — has no ``active_workspace/`` overlay, so a write
    there is a real, global, cross-session mutation. Returns False → the
    caller must refuse the target.

    SOFT SCOPE (non-blocking warn). Within ``workspace/`` the expected
    scope of one iter is its own graph plus the nodesets that graph uses
    (``prefixes``). A patch touching a graph or nodeset outside that
    scope gets a printed warning — usually a proposer path mistake — but
    is NOT blocked: a nodeset imported transitively by another (never
    named in any node ``type``) is legitimately off-prefix and must stay
    editable. Returns True.
    """
    if not rel_path.startswith(_WS):
        return False  # hard wall — outside workspace/, no overlay exists

    rel = rel_path[len(_WS) :]
    if rel.startswith("graphs/"):
        leaf = os.path.basename(rel)
        if leaf != f"{graph_name}.json":
            print(
                f"  [off-scope WARN] patch edits graph '{leaf}' but the iter's "
                f"graph is '{graph_name}.json' — check the proposer target"
            )
    elif rel.startswith("nodesets/"):
        # Drop the "nodesets/" head and any "server/" bucket dir; the
        # first remaining component is the nodeset (file or package).
        parts = [p for p in rel.split("/")[1:] if p and p != "server"]
        if parts:
            token = parts[0][:-3] if parts[0].endswith(".py") else parts[0]
            # Loose match — a node `type` prefix (e.g. ``env_libero``) and
            # the nodeset dir name (e.g. ``server/libero/``) need not be
            # equal, so substring either way counts as in-scope.
            if not any(token in p or p in token for p in prefixes):
                print(
                    f"  [off-scope WARN] patch edits nodeset '{token}', not among "
                    f"the iter graph's nodesets {sorted(prefixes)} — OK if it is a "
                    f"transitive dependency, else check the proposer target"
                )
    return True


# ── active_workspace seeding ─────────────────────────────────────────


def _enclosing_package_rel(frozen_root: str, rel: str) -> str | None:
    """Workspace-prefixed rel-path of the outermost Python package under
    ``workspace/nodesets/`` that contains ``rel``, or None if ``rel`` is
    not inside a package. ``rel`` is workspace-prefixed.

    A package is a dir with ``__init__.py``. ``nodesets/`` and the
    ``nodesets/server/`` bucket are not packages, so the shallowest
    ``__init__.py``-bearing dir below them is the package root.
    """
    prefix = _WS + "nodesets/"
    if not rel.startswith(prefix):
        return None
    parts = rel.split("/")  # workspace / nodesets / ... / file
    for i in range(3, len(parts)):  # parts[:i] is a candidate dir
        cand_rel = "/".join(parts[:i])
        cand_abs = os.path.join(frozen_root, cand_rel)
        if os.path.isdir(cand_abs) and os.path.isfile(os.path.join(cand_abs, "__init__.py")):
            return cand_rel
    return None


def seed(active_ws: str, frozen_root: str, rel: str) -> str:
    """Copy a frozen ``workspace/`` file into ``active_ws`` on first
    touch so it can be edited natively. ``rel`` is workspace-prefixed.

    If the file lives in a Python package under ``nodesets/``, the whole
    package tree is mirrored — a partial package overlay breaks imports
    when ``auto_host`` runs against the overlay's PYTHONPATH. Idempotent:
    if the destination already exists, nothing is copied.

    Returns the ``active_workspace/`` path of the seeded file.
    """
    if not rel.startswith(_WS):
        raise ValueError(f"target must be workspace-prefixed: {rel}")
    inner = rel[len(_WS) :]  # graphs/x.json | nodesets/...
    dest = os.path.join(active_ws, inner)
    if os.path.exists(dest):
        return dest

    pkg_rel = _enclosing_package_rel(frozen_root, rel)
    if pkg_rel is None:
        src = os.path.join(frozen_root, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"frozen source not found: {src}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)
    else:
        pkg_dest = os.path.join(active_ws, pkg_rel[len(_WS) :])
        if not os.path.exists(pkg_dest):
            # Exclude what hash_nodeset_tree skips so the overlay's
            # content hash stays stable (files-contract.md §8, TODO #60).
            shutil.copytree(
                os.path.join(frozen_root, pkg_rel),
                pkg_dest,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    return dest


# ── CLI ──────────────────────────────────────────────────────────────


def _prepare(active_ws: str, frozen_root: str, graph_name: str, targets: list) -> int:
    """check_target + seed every target. Print one line each. Exit
    non-zero iff any target hits the §7 hard wall."""
    # Graph prefixes drive the off-scope warn; read from the overlay
    # graph if already seeded, else from frozen — without forcing a seed.
    gpath = _resolve_graph(active_ws, graph_name) or _resolve_graph(
        os.path.join(frozen_root, _WS), graph_name
    )
    prefixes = graph_prefixes(gpath) if gpath and os.path.exists(gpath) else set()

    blocked = []
    for t in targets:
        if not check_target(t, graph_name, prefixes):
            blocked.append(t)
            print(f"  [BLOCKED] {t} — outside workspace/, no overlay exists")
            continue
        path = seed(active_ws, frozen_root, t)
        print(f"  [seeded]  {t} -> {path}")
    if blocked:
        print(
            f"ERROR: {len(blocked)} target(s) violate the §7 hard wall "
            f"(framework / third_party are never editable)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Seed frozen workspace/ files into an iter's active_workspace/ overlay."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="check_target + seed a list of targets")
    p.add_argument("--active-ws", required=True, help="path to iter's active_workspace/")
    p.add_argument("--frozen-root", required=True, help="repo root (contains workspace/)")
    p.add_argument("--graph", required=True, help="graph name (no .json)")
    p.add_argument("targets", nargs="+", help="workspace-prefixed target paths")
    args = ap.parse_args()
    if args.cmd == "prepare":
        sys.exit(_prepare(args.active_ws, args.frozen_root, args.graph, args.targets))
