"""The loop pivots — iterIn, iterOut and graphOut runtime semantics in one place.

Three builtin node types pivot the dataflow instead of transforming it:

* ``iterIn`` opens an iteration. Its outputs are not computed from inputs;
  they are *slots*, written by two writers at disjoint times (run-start
  init edges, and the paired iterOut at the iteration boundary) and
  re-emitted each iteration under the per-port ``persist`` rule.
  :class:`LoopCarry` owns the slots and the rule.
* ``iterOut`` closes an iteration. Its fired result is the loop-carry
  bundle: :meth:`LoopCarry.handoff` writes it back into the paired iterIn,
  and when the scope terminates the same bundle is mirrored once onto the
  ``final_*`` handles (:func:`final_payload`). The fallback emission (loop
  ended without a boundary) rebuilds it from the slots
  (:meth:`LoopCarry.last_carried`).
* ``graphOut`` is the run-output channel, two keys on the node's state:
  ``_last_inputs`` — the harvest snapshot that ``graph_run`` and the
  ``nav_step`` broadcast read — and ``latched_value`` — an inner-scope
  graphOut's buffered "return value", flushed outward when its scope
  terminates. The ``graphout_*`` functions are the only writers.

The scheduling loop (``GraphExecutor.run``) calls into this module at fixed
points and holds no slot / latch / snapshot logic of its own. Nothing here
schedules, fires or routes: ADR-executor-003 keeps one executor instance,
and these are its pivots, not a second engine. The authoring side of the
same contract — load-time synthesis of iterIn's port list
(``graph_def._synthesize_iterin_ports``) and the SDK's ``Graph.loop`` —
stays where it is.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_executor import NodeInstance


# ── iterIn ──────────────────────────────────────────────────────────────


class LoopCarry:
    """One iterIn's port slots — the loop-carried state of one scope.

    Built from the iterIn's ``config.ports`` (``[{name, wire_type,
    persist}]``, synthesised at graph load: ``init_<x>`` for the iterIn's
    own init ports, ``iterout_<x>`` for the paired iterOut's ports). One
    slot per port; a slot is either populated or absent.

    Writers — last write wins, never LIST[T] fan-in (``persist=True`` on a
    LIST[T] port means "carry the last written list", not "accumulate"):

    * :meth:`write` — a run-start init edge landing on the iterIn.
    * :meth:`handoff` — the paired iterOut's fired bundle at the boundary.

    Readers:

    * :meth:`ready` — the iterIn readiness rule.
    * :meth:`fire_inputs` — what the iterIn fires with.
    * :meth:`after_fire` — ``persist=False`` slots are cleared (emit once on
      the step of the write, then gone); ``persist=True`` slots keep their
      value until the next write.
    * :meth:`last_carried` — the last handed-off bundle, for the final-side
      fallback when the loop ended without a boundary emission.

    The six legal per-port configurations, indexed by ``(persist,
    init-writer, iterOut-writer)``, are listed on ``IterInNode``.
    """

    def __init__(self, config: dict | None = None) -> None:
        ports = (config or {}).get("ports") or []
        self._persist: dict[str, bool] = {
            p["name"]: bool(p.get("persist", True))
            for p in ports
            if isinstance(p, dict) and "name" in p
        }
        self.slots: dict[str, Any] = {}

    @property
    def port_names(self) -> set[str]:
        return set(self._persist)

    def write(self, name: str, value: Any) -> None:
        """A writer lands a value in one slot (run-start init edges)."""
        self.slots[name] = value

    def handoff(self, carried: dict) -> list[str]:
        """The paired iterOut's fired bundle, written into the
        ``iterout_<x>`` slots this iterIn declares. Keys without a slot are
        dropped — ``stop`` is never loop-carried. Returns the slots written."""
        written: list[str] = []
        for key, value in carried.items():
            slot = f"iterout_{key}"
            if slot in self._persist:
                self.slots[slot] = value
                written.append(slot)
        return written

    def ready(self, required: set[str]) -> bool:
        """Required ports (``required: true`` on the port entry) must all be
        populated; with none required, any populated slot makes it ready."""
        if required:
            return all(name in self.slots for name in required)
        return bool(self.slots)

    def fire_inputs(self) -> dict:
        """The populated slots, as the iterIn's inputs for this fire."""
        return dict(self.slots)

    def after_fire(self) -> None:
        """Clear every ``persist=False`` slot; the rest carry over."""
        for name, keep in self._persist.items():
            if not keep:
                self.slots.pop(name, None)

    def last_carried(self, port_names: Iterable[str]) -> dict:
        """``{port: slot value}`` of the last handoff for the paired
        iterOut's ports (``None`` for a port never handed off)."""
        return {name: self.slots.get(f"iterout_{name}") for name in port_names}


# ── iterOut ─────────────────────────────────────────────────────────────


def iterout_port_names(config: dict | None) -> list[str]:
    """The loop-carry port names an iterOut declares in ``config.ports``."""
    return [
        p["name"]
        for p in (config or {}).get("ports") or []
        if isinstance(p, dict) and p.get("name")
    ]


def stop_requested(result: Any) -> bool:
    """The Decide check: the just-fired iterOut's own ``stop`` input."""
    return isinstance(result, dict) and bool(result.get("stop"))


def final_payload(values: dict | None) -> dict:
    """The final side of a terminating scope: ``final_<name>`` for every
    carried value (``stop`` is not mirrored) plus the constant
    ``final_stop=True``, the after-loop trigger."""
    payload: dict[str, Any] = {f"final_{k}": v for k, v in (values or {}).items() if k != "stop"}
    payload["final_stop"] = True
    return payload


# ── graphOut ────────────────────────────────────────────────────────────


def graphout_snapshot(node: NodeInstance) -> None:
    """Record the node's pending inputs as the harvest snapshot — the
    canonical run-output channel."""
    node.state["_last_inputs"] = dict(node.pending_inputs)


def graphout_last(node: NodeInstance) -> dict:
    """The harvest snapshot (empty when the graphOut never received a value)."""
    return node.state.get("_last_inputs") or {}


def graphout_latch(node: NodeInstance, value: Any) -> None:
    """Buffer an inner-scope graphOut's value until its scope terminates."""
    node.state["latched_value"] = value


def graphout_latched(node: NodeInstance) -> Any:
    """The buffered value, or ``None`` when nothing was latched."""
    return node.state.get("latched_value")
