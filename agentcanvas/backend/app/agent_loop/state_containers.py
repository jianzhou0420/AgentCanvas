"""State container system — agent memory visible on the canvas.

A StateContainer is a dict of named BaseState entries. Each state has a
reducer behaviour (accumulator, lastWrite, counter) and a value
type from the STATE_VALUE_TYPES registry.

State containers are the **memory** subsystem — multi-writer, survives
across iterations, read does not trigger firing. Distinct from the **wire**
subsystem (single-firing dataflow on typed ports).

Nodes access containers via **access grants** (``AccessGrantDef``), not
wires.  The GraphExecutor injects granted containers into
``ctx._containers`` before each node fires.

last updated: 2026-04-15
"""

from __future__ import annotations

import contextlib
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from ..graph_def import ContainerDef, StateDef

log = logging.getLogger("agentcanvas.state-containers")


# ---------------------------------------------------------------------------
# Value type registry — memory is structured/aggregated data, not a raw frame
# ---------------------------------------------------------------------------
#
# State containers hold **agent memory**, not wire payloads.  Raw wire
# shapes (IMAGE / DEPTH / ACTION / OBSERVATION / STEP_RESULT) are therefore
# not valid memory value types — if you want to remember an image, store
# a caption / embedding / scene graph instead.  This restriction is what
# keeps the wire system and the state container system from blurring into
# one undifferentiated blackboard (see ADR-026).

STATE_VALUE_TYPES = {
    # Scalars / structured primitives
    "TEXT",
    "BOOL",
    "METRICS",
    "POSE",  # {position, orientation} — small and stable
    # Domain-specific structured memory
    "POINTCLOUD",  # SLAM point cloud map
    "OCCUPANCY_MAP",  # 2D grid map
    "EMBEDDING",  # vector embedding
    "EPISODE_CONTEXT",  # full episode metadata
    # Escape hatch for prototyping
    "ANY",
}

# Raw wire payload types that were once accepted but are now rejected.
# Kept as a named set so ``build_container`` can give a clear error.
_REMOVED_VALUE_TYPES = frozenset(
    {"STATE", "IMAGE", "DEPTH", "ACTION", "OBSERVATION", "STEP_RESULT"}
)


# ---------------------------------------------------------------------------
# Lifetime → signal expansion
# ---------------------------------------------------------------------------
#
# ``lifetime`` is the friendly, user-selectable axis declaring *when* a state
# clears — orthogonal to the reducer (which controls *how* writes combine).
# At container build time, each lifetime expands to an internal list of
# signal names that the state subscribes to. The rest of the runtime only
# knows about ``reset_on`` — a single mechanism.
#
# Adding a new lifetime = one line here. Adding a custom signal boundary =
# author a node/env panel that calls ``executor.broadcast_signal(name, ...)``
# and declare your state with ``lifetime="custom", reset_on=["<name>"]``.

LIFETIME_TO_SIGNALS: dict[str, list[str]] = {
    "forever": [],
    "step": ["step_end"],
    "episode": ["episode_reset"],
    "run": ["run_end"],
    "custom": [],  # honoured explicitly at build time via sdef.reset_on
}


# ---------------------------------------------------------------------------
# BaseState — standalone dataclass, one per named entry
# ---------------------------------------------------------------------------


@dataclass
class BaseState(ABC):
    """A single named state with reducer behaviour and typed value."""

    name: str
    value_type: str = "ANY"
    initial_value: Any = None
    value: Any = None
    reset_on: list[str] = None  # type: ignore[assignment]  # set by __post_init__

    state_type: ClassVar[str]

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = copy.deepcopy(self.initial_value)
        if self.reset_on is None:
            self.reset_on = []

    @abstractmethod
    def write(self, data: Any) -> None:
        """Write data — merged via the state's reducer."""

    def read(self) -> Any:
        """Get current value.  Returns initial_value if never written."""
        return self.value

    def clear(self) -> None:
        """Reset to initial_value."""
        self.value = copy.deepcopy(self.initial_value)

    def checkpoint(self) -> Any:
        """Serialize current value for persistence."""
        return copy.deepcopy(self.value)

    def from_checkpoint(self, data: Any) -> None:
        """Restore value from a checkpoint."""
        self.value = data

    def on_signal(self, name: str, payload: dict[str, Any] | None = None) -> None:
        """Handle a framework or user-emitted signal.

        Default behaviour: if ``name`` appears in this state's ``reset_on``
        list, reset to ``initial_value``.  Custom reducers may override to
        snapshot before clearing or rehydrate from ``payload``.
        """
        if name in self.reset_on:
            self.clear()


# ---------------------------------------------------------------------------
# Reducer implementations
# ---------------------------------------------------------------------------


@dataclass
class AccumulatorState(BaseState):
    """Appends each write to a list.  Config: max_size trims oldest."""

    state_type: ClassVar[str] = "accumulator"
    max_size: int | None = None

    def __post_init__(self) -> None:
        if self.initial_value is None:
            self.initial_value = []
        if self.value is None:
            self.value = copy.deepcopy(self.initial_value)

    def write(self, data: Any) -> None:
        self.value.append(data)
        if self.max_size is not None and len(self.value) > self.max_size:
            self.value = self.value[-self.max_size :]


@dataclass
class LastWriteState(BaseState):
    """Keeps only the most recent value."""

    state_type: ClassVar[str] = "lastWrite"

    def write(self, data: Any) -> None:
        self.value = data


@dataclass
class CounterState(BaseState):
    """Sums numeric writes."""

    state_type: ClassVar[str] = "counter"

    def __post_init__(self) -> None:
        if self.initial_value is None:
            self.initial_value = 0
        if self.value is None:
            self.value = copy.deepcopy(self.initial_value)

    def write(self, data: Any) -> None:
        self.value = self.value + data


# ---------------------------------------------------------------------------
# State type registry
# ---------------------------------------------------------------------------

STATE_TYPE_REGISTRY: dict[str, type] = {
    "accumulator": AccumulatorState,
    "lastWrite": LastWriteState,
    "counter": CounterState,
}


# ---------------------------------------------------------------------------
# StateContainer — dict of named states
# ---------------------------------------------------------------------------


class StateContainer:
    """A dict of named states — visible on the canvas.

    Nodes with an access grant to this container can call ``read(name)``
    and ``write(name, data)`` to access individual states.
    """

    def __init__(self, container_id: str, label: str, states: dict[str, BaseState]) -> None:
        self.container_id = container_id
        self.label = label
        # Declared template states (one per name). For non-keyed access these
        # ARE the live slots; for keyed access they are cloned per key.
        self.states = states
        # Per-key partitions: ``{name: {key: BaseState}}`` — lazily cloned from
        # the template on first keyed write. Empty for non-keyed containers.
        # Used by nodeset-owned containers under worker parallelism: one
        # container serves N concurrent workers, partitioned by an explicit
        # key (e.g. ``episode_id``) the caller passes at read/write time.
        self._keyed: dict[str, dict[str, BaseState]] = {}

    def _template(self, name: str, action: str) -> BaseState:
        state = self.states.get(name)
        if state is None:
            raise KeyError(
                f"Container '{self.container_id}': cannot {action} undeclared state "
                f"'{name}'. Declared states: {list(self.states.keys())}. Add '{name}' "
                "to the container definition."
            )
        return state

    def _keyed_state(self, name: str, key: str, *, create: bool) -> BaseState | None:
        """Per-key sub-state for ``(name, key)``; lazily cloned from the
        template at its ``initial_value``. Returns None if absent and not
        creating (caller substitutes the template default)."""
        template = self._template(name, "access")
        bucket = self._keyed.setdefault(name, {})
        state = bucket.get(key)
        if state is None and create:
            state = copy.deepcopy(template)
            state.clear()  # fresh initial_value, isolated from sibling keys
            bucket[key] = state
        return state

    def read(self, name: str, key: str | None = None) -> Any:
        """Read a named state's value.  Raises KeyError if not declared.

        With ``key`` set, reads that key's partition; a never-written key
        yields a fresh copy of the slot's ``initial_value``."""
        if key is None:
            return self._template(name, "read").read()
        state = self._keyed_state(name, key, create=False)
        if state is None:
            return copy.deepcopy(self._template(name, "read").initial_value)
        return state.read()

    def write(self, name: str, data: Any, key: str | None = None) -> None:
        """Write to a named state — merged via its reducer.  Raises KeyError
        if not declared.  With ``key`` set, writes into that key's partition
        (lazily created)."""
        if key is None:
            self._template(name, "write").write(data)
            return
        self._keyed_state(name, key, create=True).write(data)  # type: ignore[union-attr]

    def evict(self, key: str) -> None:
        """Drop every per-key sub-state for ``key`` — worker-safe cleanup at
        episode/lease end. Touches only this key; concurrent siblings under
        other keys are untouched (this is what replaces the old module-global
        'never clear' race fix)."""
        for bucket in self._keyed.values():
            bucket.pop(key, None)

    def on_signal(self, name: str, payload: dict[str, Any] | None = None) -> None:
        """Fan a framework or user-emitted signal to every state (incl. per-key)."""
        for state in self.states.values():
            state.on_signal(name, payload)
        for bucket in self._keyed.values():
            for state in bucket.values():
                state.on_signal(name, payload)

    def checkpoint(self) -> dict[str, Any]:
        """Snapshot all states for persistence.  Non-checkpointable opaque
        slots (e.g. a numba object held by reference) are skipped, not raised
        — by design (an opaque keyed slot is non-checkpointable)."""
        out: dict[str, Any] = {}
        for name, s in self.states.items():
            try:
                out[name] = s.checkpoint()
            except Exception:
                log.debug("checkpoint: skipping non-checkpointable state '%s'", name)
        if self._keyed:
            keyed: dict[str, dict[str, Any]] = {}
            for name, bucket in self._keyed.items():
                kk: dict[str, Any] = {}
                for key, s in bucket.items():
                    with contextlib.suppress(Exception):  # opaque slot — skip
                        kk[key] = s.checkpoint()
                if kk:
                    keyed[name] = kk
            if keyed:
                out["__keyed__"] = keyed
        return out

    def from_checkpoint(self, data: dict[str, Any]) -> None:
        """Restore all states from a checkpoint produced by :meth:`checkpoint`."""
        keyed = data.get("__keyed__") if isinstance(data, dict) else None
        for name, s in self.states.items():
            if name in data:
                s.from_checkpoint(data[name])
        if keyed:
            for name, bucket in keyed.items():
                if name not in self.states:
                    continue
                for key, val in bucket.items():
                    st = self._keyed_state(name, key, create=True)
                    if st is not None:
                        st.from_checkpoint(val)

    def get_preview(self) -> dict[str, dict[str, Any]]:
        """Build a preview dict for WebSocket broadcast.

        For keyed containers, summarize across live keys (key count + a bounded
        sample of one key) — never serialize an opaque per-key value, only a
        truncated repr."""

        def _fill(info: dict[str, Any], val: Any, prefix: str = "") -> None:
            if isinstance(val, list):
                info["size"] = len(val)
                info["preview"] = prefix + str(val[:3]) + ("..." if len(val) > 3 else "")
            elif isinstance(val, (int, float)):
                info["value"] = val
                if prefix:
                    info["preview"] = prefix + str(val)
            elif isinstance(val, str):
                info["preview"] = prefix + val[:100] + ("..." if len(val) > 100 else "")
            elif val is None:
                info["preview"] = prefix + "null"
            else:
                info["preview"] = prefix + str(val)[:100]

        result: dict[str, dict[str, Any]] = {}
        for name, s in self.states.items():
            info: dict[str, Any] = {"type": s.state_type, "value_type": s.value_type}
            bucket = self._keyed.get(name)
            if bucket:
                info["keys"] = len(bucket)
                sample_key = next(iter(bucket))
                _fill(info, bucket[sample_key].read(), prefix=f"[{sample_key}] ")
            else:
                _fill(info, s.read())
            result[name] = info
        return result


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _resolve_reset_on(sdef: StateDef) -> list[str]:
    """Expand ``sdef.lifetime`` into the list of signal names the state
    should reset on.

    * ``"custom"`` reads the explicit list from ``sdef.reset_on``
    * Any other lifetime looks up ``LIFETIME_TO_SIGNALS``
    * Unknown lifetimes fall back to ``"forever"`` with a warning
    """
    lifetime = sdef.lifetime or "forever"
    if lifetime == "custom":
        return list(sdef.reset_on)
    if lifetime not in LIFETIME_TO_SIGNALS:
        log.warning(
            "Unknown lifetime '%s' (valid: %s), treating as 'forever'",
            lifetime,
            sorted(LIFETIME_TO_SIGNALS),
        )
        return []
    return list(LIFETIME_TO_SIGNALS[lifetime])


def _build_state(name: str, sdef: StateDef) -> BaseState:
    """Create a live BaseState instance from a StateDef.

    Expands ``lifetime`` into the internal ``reset_on`` signal list.
    """
    reducer_type = sdef.type
    reset_on = _resolve_reset_on(sdef)

    cls = STATE_TYPE_REGISTRY.get(reducer_type)
    if cls is None:
        log.warning(
            "Unknown state type '%s' for '%s', falling back to LastWrite",
            reducer_type,
            name,
        )
        cls = LastWriteState
    kwargs: dict[str, Any] = {
        "name": name,
        "value_type": sdef.value_type,
        "reset_on": reset_on,
    }
    if "initial_value" in sdef.config:
        kwargs["initial_value"] = sdef.config["initial_value"]
    if "max_size" in sdef.config and cls is AccumulatorState:
        kwargs["max_size"] = sdef.config["max_size"]
    return cls(**kwargs)


def build_container(cdef: ContainerDef, allow_opaque: bool = False) -> StateContainer:
    """Create a live StateContainer from a ContainerDef.

    Rejects any state slot whose ``value_type`` is a raw wire payload
    (IMAGE / DEPTH / ACTION / OBSERVATION / STEP_RESULT / the legacy
    STATE) — memory must be a structured/aggregated form.  Use ``ANY``
    as an escape hatch for prototyping.

    ``allow_opaque=True`` relaxes that check for **nodeset-level** containers:
    these live in the owning nodeset's process and are accessed by reference
    only — they are never serialized across the boundary — so they may hold
    opaque/raw values (e.g. a numba planner object). Graph-level (home)
    containers keep the strict check (default), which is what ADR-026 guards.
    """
    for sname, sdef in cdef.states.items():
        vt = sdef.value_type
        if vt in _REMOVED_VALUE_TYPES and not allow_opaque:
            raise ValueError(
                f"Container '{cdef.id}' state '{sname}': value_type '{vt}' "
                f"is no longer allowed — state containers hold agent memory, "
                f"not raw wire payloads. Use 'ANY' (escape hatch) or a "
                f"structured type ({sorted(STATE_VALUE_TYPES)})."
            )
        if vt not in STATE_VALUE_TYPES and not allow_opaque:
            log.warning(
                "Container '%s' state '%s': unknown value_type '%s' "
                "(falling back to ANY at runtime). Valid: %s",
                cdef.id,
                sname,
                vt,
                sorted(STATE_VALUE_TYPES),
            )
    states = {name: _build_state(name, sdef) for name, sdef in cdef.states.items()}
    return StateContainer(container_id=cdef.id, label=cdef.label, states=states)


def build_containers(
    cdefs: list[ContainerDef], allow_opaque: bool = False
) -> dict[str, StateContainer]:
    """Create all live containers from a list of ContainerDefs.

    ``allow_opaque`` is forwarded to :func:`build_container` — set it for
    nodeset-level containers (never serialized; may hold opaque values).
    """
    result: dict[str, StateContainer] = {}
    for cdef in cdefs:
        result[cdef.id] = build_container(cdef, allow_opaque=allow_opaque)
        log.info(
            "Built container '%s' with %d states: %s",
            cdef.id,
            len(cdef.states),
            list(cdef.states.keys()),
        )
    return result
