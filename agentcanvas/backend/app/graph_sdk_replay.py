"""Record / replay for server-node graphs — run a graph whose env / model
nodes are server-mode (GPU + a conda env + weights) inside a hermetic unit
test, with none of that present.

The server-node proxy's ``forward()`` (:mod:`app.server.proxy`) is the single
seam: every server call is one ``(node_type, result)`` pair. **Record** wraps
that seam during a *real* run to tee each call into a "cassette" — msgpack then
zlib, so ndarray / torch / PIL results ride the same lossless codec the proxy
uses on the wire (:func:`app.server.serialization.pack_body`) — alongside a
manifest synthesised from the live proxy classes. **Replay** skips nodeset
loading entirely, rebuilds the proxy classes from the cassette's manifests, and
returns the recorded results in call order, so the graph runs offline. Local
(pure-Python) nodes still run for real, so replay exercises the actual wiring +
port interfaces: it fails loudly when a port or interface drifts away from the
recording (the drift a hand-rerun would otherwise have to catch).

Driven via :meth:`app.graph_sdk.Graph.run` ``record=`` / ``replay=``. Both
mutate the global ``NODE_HANDLERS`` for the duration of one run and restore it
afterwards; they are for a single in-process run at a time, not concurrent use.
"""

from __future__ import annotations

import zlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .server import serialization

CASSETTE_VERSION = 1


class ReplayExhausted(RuntimeError):
    """Replay was asked for a server call with no recorded result left — the
    graph fired a different sequence of server calls than was recorded, i.e. a
    topology / interface drift from the cassette. Raised inside the replayed
    node's ``forward``; the run then surfaces it as a node error."""


# ── cassette persistence (msgpack + zlib) ─────────────────────────────────


def dump_cassette(path: str | Path, manifests: dict, calls: list) -> Path:
    """Write a cassette: ``{version, manifests, calls}`` → msgpack → zlib."""
    body = {"version": CASSETTE_VERSION, "manifests": manifests, "calls": calls}
    raw = zlib.compress(serialization.pack_body(body), level=9)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return p


def load_cassette(path: str | Path) -> dict:
    """Inverse of :func:`dump_cassette`."""
    body = serialization.unpack_body(zlib.decompress(Path(path).read_bytes()))
    if body.get("version") != CASSETTE_VERSION:
        raise ValueError(
            f"cassette version {body.get('version')!r} != {CASSETTE_VERSION} (regenerate it)"
        )
    return body


# ── manifest synthesis (from the live proxy classes) ──────────────────────


def _portschema(p: Any) -> Any:
    from .server.manifest import PortSchema

    return PortSchema(
        name=p.name,
        wire_type=getattr(p, "wire_type", "ANY"),
        description=getattr(p, "description", "") or "",
        optional=bool(getattr(p, "optional", False)),
    )


def _synthesize_manifests(node_types: set[str]) -> dict:
    """Build one ``ServerManifest`` dict per nodeset from the registered proxy
    classes, so replay can rebuild the proxy classes without a live server.

    Only the ports (+ config schema) matter for replay — ``forward`` is replaced
    with the cassette lookup — so this reads exactly what
    :func:`app.server.proxy.create_proxy_node` bakes onto each class.
    """
    from .agent_loop.builtin_nodes import NODE_HANDLERS
    from .server.manifest import FunctionSchema, ServerManifest

    by_ns: dict[str, list] = defaultdict(list)
    for nt in node_types:
        cls = NODE_HANDLERS.get(nt)
        if cls is None or "__" not in nt:
            continue
        ns = nt.split("__", 1)[0]
        by_ns[ns].append(
            FunctionSchema(
                name=nt,
                description=getattr(cls, "description", "") or "",
                input_ports=[_portschema(p) for p in getattr(cls, "input_ports", [])],
                output_ports=[_portschema(p) for p in getattr(cls, "output_ports", [])],
                config_schema=getattr(cls, "config_schema", {}) or {},
            )
        )
    return {ns: ServerManifest(name=ns, functions=fns).to_dict() for ns, fns in by_ns.items()}


# ── record: tee server-proxy calls during a real run ──────────────────────


class Recorder:
    """Wraps each target proxy class's ``forward`` to tee ``(node_type, result)``
    in call order; :meth:`dump` writes the cassette. Restore with :meth:`unwrap`
    before dumping (so the manifest is read off the un-mutated classes)."""

    def __init__(self, node_types: set[str]) -> None:
        self._targets = {nt for nt in node_types if "__" in nt}
        self.calls: list[dict] = []
        self._originals: dict[str, Any] = {}

    def wrap(self) -> None:
        from .agent_loop.builtin_nodes import NODE_HANDLERS

        for nt in self._targets:
            cls = NODE_HANDLERS.get(nt)
            if cls is None:
                continue
            orig = cls.forward
            self._originals[nt] = orig
            cls.forward = self._tee(nt, orig)

    def _tee(self, nt: str, orig: Any) -> Any:
        calls = self.calls

        async def forward(self_node: Any, inputs: dict, ctx: Any) -> Any:
            result = await orig(self_node, inputs, ctx)
            calls.append({"node_type": nt, "result": result})
            return result

        return forward

    def unwrap(self) -> None:
        from .agent_loop.builtin_nodes import NODE_HANDLERS

        for nt, orig in self._originals.items():
            cls = NODE_HANDLERS.get(nt)
            if cls is not None:
                cls.forward = orig
        self._originals.clear()

    def write(self, path: str | Path) -> Path:
        """Synthesise manifests off the (restored) proxy classes and write the
        cassette. Call :meth:`unwrap` first; only write on a clean run so a
        crash mid-record never leaves a truncated cassette."""
        return dump_cassette(path, _synthesize_manifests(self._targets), self.calls)


# ── replay: serve recorded results, no server ─────────────────────────────


class Player:
    """Rebuilds the cassette's proxy classes (from its saved manifests) and
    overrides every replayed node type's ``forward`` to pop the next recorded
    result. :meth:`install` before the run, :meth:`uninstall` after."""

    def __init__(self, cassette: dict) -> None:
        self.queues: dict[str, deque] = defaultdict(deque)
        for c in cassette.get("calls", []):
            self.queues[c["node_type"]].append(c["result"])
        self._manifests: dict = cassette.get("manifests", {})
        self._registered: list[str] = []  # node types we added to NODE_HANDLERS
        self._overridden: dict[str, Any] = {}  # node type -> original forward

    def install(self) -> None:
        from .agent_loop.builtin_nodes import NODE_HANDLERS, register_node
        from .server.manifest import ServerManifest
        from .server.proxy import generate_proxy_nodes

        needed = set(self.queues)
        # Rebuild any proxy class the cassette needs that isn't registered
        # (the real case: no nodeset was loaded, so no server subprocess).
        for mdict in self._manifests.values():
            manifest = ServerManifest.from_dict(mdict)
            for cls in generate_proxy_nodes("http://replay.invalid", manifest):
                if cls.node_type in needed and cls.node_type not in NODE_HANDLERS:
                    register_node(cls)
                    self._registered.append(cls.node_type)
        # Override forward on every replayed node type (rebuilt or pre-existing).
        for nt in needed:
            cls = NODE_HANDLERS.get(nt)
            if cls is None:
                continue
            if nt not in self._registered:
                self._overridden[nt] = cls.forward
            cls.forward = self._replay(nt)

    def _replay(self, nt: str) -> Any:
        queue = self.queues[nt]

        async def forward(self_node: Any, inputs: dict, ctx: Any) -> Any:
            if not queue:
                raise ReplayExhausted(
                    f"no recorded result left for {nt!r} — the replayed graph fired "
                    f"more {nt!r} calls than the cassette holds (topology/interface drift)"
                )
            return queue.popleft()

        return forward

    def uninstall(self) -> None:
        from .agent_loop.builtin_nodes import NODE_HANDLERS

        for nt, orig in self._overridden.items():
            cls = NODE_HANDLERS.get(nt)
            if cls is not None:
                cls.forward = orig
        for nt in self._registered:
            NODE_HANDLERS.pop(nt, None)
        self._overridden.clear()
        self._registered.clear()
