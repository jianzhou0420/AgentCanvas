"""Regression test: reload must evict deleted standalone nodes from NODE_HANDLERS.

Bug (fixed): ``workspace/nodes/`` + ``workspace/policies/`` nodes register into
the global ``NODE_HANDLERS`` at *scan* time, but ``unregister_all()`` never
popped them — so a node deleted from disk lingered in the live registry (and in
``GET /node-schemas`` / the sidebar) until a full backend restart. ``scan_all``
calls ``unregister_all`` first, so a rescan must now drop the stale entry.
"""

from __future__ import annotations

from pathlib import Path

from ..agent_loop.builtin_nodes import NODE_HANDLERS
from .registry import WorkspaceComponentRegistry

_NODE_SRC = """
from app.components import BaseCanvasNode, PortDef


class _TmpThrowawayNode(BaseCanvasNode):
    node_type = "tmp_throwaway_node"
    display_name = "Tmp Throwaway"
    category = "tool"
    input_ports = []
    output_ports = [PortDef("out", "TEXT", "x")]

    async def forward(self, inputs, ctx=None):
        return {"out": "x"}
"""


def test_reload_evicts_deleted_standalone_node(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    nodes_dir = workspace_root / "nodes"
    nodes_dir.mkdir(parents=True)
    node_file = nodes_dir / "tmp_throwaway.py"
    node_file.write_text(_NODE_SRC)

    reg = WorkspaceComponentRegistry(scan_dir=workspace_root)

    # First scan: standalone node registered into the global registry + tracked.
    reg.scan_all()
    assert "tmp_throwaway_node" in NODE_HANDLERS
    assert "tmp_throwaway_node" in reg._standalone_node_types
    assert "llmCall" in NODE_HANDLERS  # built-in baseline present

    # Delete from disk, rescan — the stale entry must be gone.
    node_file.unlink()
    reg.scan_all()
    assert "tmp_throwaway_node" not in NODE_HANDLERS, (
        "deleted standalone node lingered in NODE_HANDLERS after reload"
    )
    assert "tmp_throwaway_node" not in reg._standalone_node_types
    # Built-ins must survive the eviction (only standalone types are popped).
    assert "llmCall" in NODE_HANDLERS
