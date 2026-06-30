"""Data models for the execution log system."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NodeLogEntry(BaseModel):
    """One node firing: captured inputs, outputs, timing, and voluntary inner log."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_id: str
    source: str  # "canvas" | "eval"
    step: int  # GraphExecutor.step_counter at time of firing
    node_id: str
    node_type: str
    node_label: str
    duration_ms: float
    # System Log perf breakdown — all backward-compatible, default None so
    # pre-existing readers and old JSONL lines are unaffected.
    queue_wait_ms: float | None = None  # time waited in the ready-queue before firing
    compute_ms: float | None = (
        None  # forward() time (== duration_ms until transport split out in P2)
    )
    transport_ms: float | None = None  # server-mode HTTP round-trip (P2)
    transfer_bytes: int | None = None  # server-mode payload bytes (P2)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    inner_log: list[dict] = Field(default_factory=list)
    port_wire_types: dict[str, str] = Field(default_factory=dict)
    # Maps port names to wire types, e.g. {"rgb": "IMAGE", "depth": "DEPTH", "action": "ACTION"}
    error: str | None = None

    # C.5 Dynamic Fire-List provenance — set on log entries for ephemeral
    # children of a ``DynamicFireListNode`` spawner; ``parent_node_id`` is
    # the spawner's static node id (e.g. ``voxposer_composer_dyn``), and
    # ``dynamic_index`` is the child's position in the spawner's emitted
    # ``FireList.specs``. Both default ``None`` for entries from regular
    # (static-topology) node firings. Backward-compatible — readers that
    # don't know about these fields just ignore them.
    parent_node_id: str | None = None
    dynamic_index: int | None = None


class ExecutionSummary(BaseModel):
    """Aggregate stats for a completed (or in-progress) execution."""

    execution_id: str
    source: str
    started_at: datetime
    ended_at: datetime | None = None
    total_steps: int = 0
    total_firings: int = 0
    error_count: int = 0
    node_types_fired: list[str] = Field(default_factory=list)
