"""Standalone graph wire-type validator (dev tool).

Loads a graph from disk and runs the SAME wire-type core the backend uses
(``app.graph_def.wire_type_report``), then prints a report and exits non-zero
on any mismatch. Run it at dev time without a backend:

    # from agentcanvas/backend/
    python -m app.tools.validate_graph mapgpt_mp3d
    python -m app.tools.validate_graph workspace/graphs/vln/verified/smartway_ce.json
    python -m app.tools.validate_graph --all          # every saved graph
    python -m app.tools.validate_graph mapgpt_mp3d --json

Nodeset coverage (the "dry run"): by default the tool statically imports each
nodeset module and reads its nodes' ``input_ports`` / ``output_ports`` — the
port specs are plain class declarations, so this needs NO simulator, NO GPU,
NO running backend (heavy runtimes like habitat-sim load lazily only when a
node actually runs). So env↔method edges are checked too. Pass ``--builtins``
to skip introspection (builtin + iterIn/iterOut edges only).

Exit code: 0 = no mismatches, 1 = mismatches found, 2 = usage / load error.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import sys
from pathlib import Path

# .../agentcanvas/backend/app/tools/validate_graph.py
_BACKEND = Path(__file__).resolve().parents[2]  # .../agentcanvas/backend
_ROOT = _BACKEND.parent.parent  # repo root (.../vlnworkspace)
# ``app.*`` (backend) and ``workspace.nodesets.*`` (nodesets, incl. their own
# absolute intra-package imports) must both resolve.
for _p in (str(_BACKEND), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.graph_def import GraphDefinition, _resolve_port_type, wire_type_report  # noqa: E402

_WORKSPACE = _ROOT / "workspace"
_GRAPH_ROOTS = (_WORKSPACE / "graphs", _WORKSPACE / "graph_nodes")
_NS_ROOT = _WORKSPACE / "nodesets"
_NS_CATEGORIES = ("env", "method", "policy", "model", "common", "other")


# ── Static nodeset port introspection (the dry-run) ────────────────────────


def introspect_nodesets() -> tuple[dict, list[str], list[tuple[str, str]]]:
    """Import every nodeset module and read its nodes' declared port types.

    Returns ``(schema, ok, failed)`` where ``schema`` is
    ``{node_type: {"in": {handle: wire_type}, "out": {handle: wire_type}}}``,
    ``ok`` lists nodeset modules imported, and ``failed`` lists
    ``(name, error)`` for modules that could not be imported (their edges
    stay unresolved → reported as skipped, never a false mismatch).
    """
    from app.components.bases import BaseNodeSet

    schema: dict[str, dict] = {}
    ok: list[str] = []
    failed: list[tuple[str, str]] = []

    def collect(mod) -> None:
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if isinstance(obj, type) and issubclass(obj, BaseNodeSet) and obj is not BaseNodeSet:
                try:
                    for tool in obj().get_tools():
                        schema[tool.node_type] = {
                            "in": {p.name: p.wire_type for p in getattr(tool, "input_ports", [])},
                            "out": {p.name: p.wire_type for p in getattr(tool, "output_ports", [])},
                        }
                except Exception:
                    pass

    for cat in _NS_CATEGORIES:
        cat_dir = _NS_ROOT / cat
        if not cat_dir.is_dir():
            continue
        for entry in sorted(cat_dir.iterdir()):
            stem = entry.stem
            if stem.startswith("_") or stem.startswith("test_") or entry.name == "__pycache__":
                continue
            if entry.is_file() and entry.suffix != ".py":
                continue
            mod_name = f"workspace.nodesets.{cat}.{stem}"
            try:
                collect(importlib.import_module(mod_name))
                ok.append(stem)
            except Exception as ex:
                failed.append((stem, f"{type(ex).__name__}: {ex}"))
    return schema, ok, failed


def make_resolver(schema: dict):
    """Resolver that consults the core (builtins + iterIn/iterOut) first, then
    the statically-introspected nodeset ``schema``."""

    def resolve(node, handle, direction):
        t = _resolve_port_type(node, handle, direction)
        if t is not None:
            return t
        entry = schema.get(node.type)
        if entry is not None:
            return entry["out" if direction == "out" else "in"].get(handle)
        return None

    return resolve


# ── Graph resolution + per-graph check ─────────────────────────────────────


def _resolve_path(name_or_path: str) -> Path | None:
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    hits: list[str] = []
    for root in _GRAPH_ROOTS:
        hits += glob.glob(str(root / "**" / f"{name_or_path}.json"), recursive=True)
    if not hits:
        return None
    if len(hits) > 1:
        print(f"ambiguous name '{name_or_path}' — matches:", file=sys.stderr)
        for h in sorted(hits):
            print(f"  {h}", file=sys.stderr)
        print("pass an explicit path instead.", file=sys.stderr)
        return None
    return Path(hits[0])


def _check_one(path: Path, resolver) -> dict:
    graph = GraphDefinition.from_dict(json.loads(path.read_text()))
    rep = wire_type_report(graph, resolver)
    rep["path"] = str(path)
    return rep


def _print_human(rep: dict) -> None:
    cov = f"{rep['checked']} checked / {rep['skipped']} skipped / {rep['total_edges']} edges"
    mark = "✗" if rep["errors"] else "✓"
    print(f"{mark} {rep['path']}  [{cov}]")
    for e in rep["errors"]:
        print(f"    {e}")
    if rep["unresolved_node_types"]:
        print(f"    note: unresolved node types: {', '.join(rep['unresolved_node_types'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="validate_graph", description=__doc__.split("\n")[0])
    ap.add_argument("graph", nargs="?", help="graph name (stem) or path to a .json")
    ap.add_argument("--all", action="store_true", help="validate every saved graph + graph_node")
    ap.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    ap.add_argument(
        "--builtins",
        action="store_true",
        help="skip nodeset introspection (check builtin + iterIn/iterOut edges only)",
    )
    args = ap.parse_args(argv)

    if not args.graph and not args.all:
        ap.error("pass a graph name/path or --all")

    if args.builtins:
        resolver = None  # wire_type_report falls back to the builtin core resolver
        if not args.json:
            print("(builtins-only mode — nodeset edges will be skipped)\n")
    else:
        schema, ok, failed = introspect_nodesets()
        resolver = make_resolver(schema)
        if not args.json:
            print(
                f"introspected nodesets: {len(ok)} ok, {len(failed)} failed, {len(schema)} node types"
            )
            for name, err in failed:
                print(f"  [skip] {name}: {err[:90]}")
            print()

    if args.all:
        paths = sorted(
            Path(h)
            for root in _GRAPH_ROOTS
            for h in glob.glob(str(root / "**" / "*.json"), recursive=True)
        )
    else:
        p = _resolve_path(args.graph)
        if p is None:
            print(f"graph not found: {args.graph}", file=sys.stderr)
            return 2
        paths = [p]

    reports: list[dict] = []
    load_errors = 0
    for path in paths:
        try:
            reports.append(_check_one(path, resolver))
        except Exception as ex:
            load_errors += 1
            reports.append({"path": str(path), "load_error": f"{type(ex).__name__}: {ex}"})

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        bad = 0
        for rep in reports:
            if rep.get("load_error"):
                print(f"! {rep['path']}  load-error: {rep['load_error']}")
                continue
            _print_human(rep)
            if rep["errors"]:
                bad += 1
        if args.all:
            print(f"\n{len(paths)} graphs — {bad} with mismatches, {load_errors} load errors")

    has_mismatch = any(r.get("errors") for r in reports)
    if load_errors:
        return 2
    return 1 if has_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
