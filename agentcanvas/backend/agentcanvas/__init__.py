"""AgentCanvas — Graph SDK (public SDK surface).

Build and run AgentCanvas VLN-agent graphs in Python, LangGraph-style, without
authoring graph JSON or standing up the canvas GUI::

    from agentcanvas import Graph

    g = Graph(name="demo")
    src = g.add("const_source", value=7)
    inc = g.add("increment")
    out = g.graph_out("result")
    g.connect(src.out("value"), inc.in_("x"))
    g.connect(inc.out("y"), out.in_("value"))
    print(g.run()["result"])          # 8

This is a thin namespace over :mod:`app.graph_sdk`; every symbol below is
re-exported from there so the two import paths are interchangeable.  See that
module for the full API, and :func:`graph_to_code` for the inverse direction
(compile an existing graph back into a builder script).
"""

from __future__ import annotations

from app.graph_sdk import (
    DefaultSession,
    EvalResult,
    Graph,
    GraphValidationError,
    Loop,
    NodeHandle,
    NodeProxy,
    PortRef,
    RunResult,
    catalog,
    describe,
    generate_node_stubs,
    nodesets,
)
from app.graph_sdk_codegen import graph_to_code

__version__ = "0.1.0"

__all__ = [
    "DefaultSession",
    "EvalResult",
    "Graph",
    "GraphValidationError",
    "Loop",
    "NodeHandle",
    "NodeProxy",
    "PortRef",
    "RunResult",
    "__version__",
    "catalog",
    "describe",
    "generate_node_stubs",
    "graph_to_code",
    "nodesets",
]
