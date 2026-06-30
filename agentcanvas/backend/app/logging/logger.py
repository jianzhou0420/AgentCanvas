"""ExecutionLogger — ring buffer + JSONL persistence + WS broadcast.

Two-layer capture:
- **Automatic** (exterior): the executor calls ``log_node()`` after each
  ``_fire_node()``, passing port inputs/outputs and timing.
- **Voluntary** (interior): nodes call ``self._self_log()`` inside
  ``execute()``; the executor collects via ``instance.log()`` and passes
  as ``inner_log``.

Large array values (numpy / torch tensors) are summarized by
``log_serialize()`` (shape + dtype). Images go through ``save_asset()``
to disk. Strings are persisted in full.
"""

from __future__ import annotations

import logging
import os
import re
from collections import deque
from datetime import datetime
from typing import Any

from .models import ExecutionSummary, NodeLogEntry

log = logging.getLogger("agentcanvas.exec-log")


# ── Serialization ──


def log_serialize(
    obj: Any,
    max_str_len: int = 1024,
    max_depth: int = 5,
    _depth: int = 0,
) -> Any:
    """Recursively serialize ``obj`` for JSONL, summarizing binary/array values.

    - Strings persisted in full (logs are for debugging — truncating prompts
      defeats the purpose). The ``max_str_len`` parameter is retained for API
      back-compat but no longer triggers truncation.
    - ``bytes`` → length summary
    - numpy arrays / tensors → shape + dtype summary
    - Recursion capped at *max_depth*
    """
    if _depth > max_depth:
        return "<max_depth>"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return {"__type": "bytes", "length": len(obj)}
    if isinstance(obj, dict):
        return {
            str(k): log_serialize(v, max_str_len, max_depth, _depth + 1) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [log_serialize(v, max_str_len, max_depth, _depth + 1) for v in obj]
    # numpy arrays, torch tensors, etc.
    type_name = type(obj).__name__
    if hasattr(obj, "shape"):
        dtype = str(getattr(obj, "dtype", ""))
        return {"__type": type_name, "shape": list(obj.shape), "dtype": dtype}
    return f"<{type_name}>"


def _sanitize_filename(s: str) -> str:
    """Replace non-alphanumeric chars (except underscore/dash/dot) with underscore."""
    return re.sub(r"[^\w\-.]", "_", s)


def save_asset(
    persist_dir: str,
    step: int,
    node_label: str,
    port_name: str,
    wire_type: str,
    value: Any,
    firing_idx: int,
) -> dict | None:
    """Save a numpy array as an image file, return asset reference dict or None."""
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    if not isinstance(value, np.ndarray):
        return None

    assets_dir = os.path.join(persist_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    safe_label = _sanitize_filename(node_label)
    safe_port = _sanitize_filename(port_name)
    shape = list(value.shape)
    dtype = str(value.dtype)

    if wire_type in ("IMAGE",) and value.ndim == 3 and value.shape[2] == 3:
        # RGB image -> JPEG
        fname = f"s{step}_f{firing_idx}_{safe_label}__{safe_port}.jpg"
        fpath = os.path.join(assets_dir, fname)
        img = Image.fromarray(value.astype(np.uint8))
        img.save(fpath, format="JPEG", quality=85)
    elif wire_type in ("DEPTH",) and value.ndim in (2, 3):
        # Depth -> uint8 normalized PNG
        fname = f"s{step}_f{firing_idx}_{safe_label}__{safe_port}.png"
        fpath = os.path.join(assets_dir, fname)
        d = value.squeeze() if value.ndim == 3 else value
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 0:
            d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d, dtype=np.uint8)
        img = Image.fromarray(d_norm, mode="L")
        img.save(fpath, format="PNG")
    else:
        return None

    return {
        "__type": "asset",
        "path": f"assets/{fname}",
        "wire_type": wire_type,
        "shape": shape,
        "dtype": dtype,
    }


def log_serialize_with_assets(
    obj: dict,
    persist_dir: str | None,
    step: int,
    node_label: str,
    port_wire_types: dict[str, str],
    firing_idx: int,
    max_str_len: int = 1024,
) -> dict:
    """Serialize a node's inputs/outputs dict, saving images as asset files.

    For IMAGE/DEPTH ports: save as files, return asset references.
    For OBSERVATION ports (dict with rgb/depth): recurse and save sub-images.
    For STEP_RESULT ports: recurse into 'observation' sub-key.
    For everything else: fall through to log_serialize().
    """
    if not persist_dir or not isinstance(obj, dict):
        return log_serialize(obj, max_str_len)

    result = {}
    for key, value in obj.items():
        wt = port_wire_types.get(key, "")

        # LIST[T] (ADR-027).  For LIST[IMAGE]/LIST[DEPTH] persist each tile as an
        # asset so the Logs viewer can render the panorama (image_list marker with
        # per-tile asset refs).  Other element types stay a compact count marker.
        if wt.startswith("LIST[") and wt.endswith("]"):
            inner_wt = wt[len("LIST[") : -1]
            if inner_wt in ("IMAGE", "DEPTH") and isinstance(value, list):
                MAX_TILES = 64  # panorama is 36; cap pathological lists
                items: list[Any] = []
                for i, item in enumerate(value[:MAX_TILES]):
                    asset = save_asset(
                        persist_dir, step, node_label, f"{key}_{i}", inner_wt, item, firing_idx
                    )
                    items.append(asset if asset is not None else log_serialize(item, max_str_len))
                result[key] = {
                    "__type": "image_list",
                    "wire_type": wt,
                    "count": len(value),
                    "items": items,
                }
                continue
            count = len(value) if isinstance(value, list) else 0
            result[key] = {
                "__type": "list",
                "wire_type": wt,
                "count": count,
            }
            continue

        if wt in ("IMAGE", "DEPTH") and value is not None:
            asset = save_asset(persist_dir, step, node_label, key, wt, value, firing_idx)
            if asset is not None:
                result[key] = asset
                continue

        elif wt == "OBSERVATION" and isinstance(value, dict):
            # Recurse: save rgb and depth separately
            group: dict[str, Any] = {"__type": "asset_group", "wire_type": "OBSERVATION"}
            for sub_key in ("rgb", "depth"):
                sub_val = value.get(sub_key)
                sub_wt = "IMAGE" if sub_key == "rgb" else "DEPTH"
                if sub_val is not None:
                    sub_port = f"{key}_{sub_key}"
                    asset = save_asset(
                        persist_dir, step, node_label, sub_port, sub_wt, sub_val, firing_idx
                    )
                    if asset is not None:
                        group[sub_key] = asset
                    else:
                        group[sub_key] = log_serialize(sub_val, max_str_len)
                else:
                    group[sub_key] = None
            result[key] = group
            continue

        elif wt == "STEP_RESULT" and isinstance(value, dict):
            # Recurse into observation sub-key
            sr: dict[str, Any] = {}
            for sr_key, sr_val in value.items():
                if sr_key == "observation" and isinstance(sr_val, dict):
                    group = {"__type": "asset_group", "wire_type": "OBSERVATION"}
                    for sub_key in ("rgb", "depth"):
                        sub_val = sr_val.get(sub_key)
                        sub_wt = "IMAGE" if sub_key == "rgb" else "DEPTH"
                        if sub_val is not None:
                            sub_port = f"{sr_key}_{sub_key}"
                            asset = save_asset(
                                persist_dir, step, node_label, sub_port, sub_wt, sub_val, firing_idx
                            )
                            if asset is not None:
                                group[sub_key] = asset
                            else:
                                group[sub_key] = log_serialize(sub_val, max_str_len)
                        else:
                            group[sub_key] = None
                    sr[sr_key] = group
                else:
                    sr[sr_key] = log_serialize(sr_val, max_str_len)
            result[key] = sr
            continue

        # Default: regular serialization
        result[key] = log_serialize(value, max_str_len)

    return result


# ── Logger ──


class ExecutionLogger:
    """Captures per-node log entries during a graph execution.

    * **Ring buffer** (``collections.deque``, thread-safe append) for real-time
      in-memory queries.
    * **Write queue** accumulated during an iteration; call :meth:`flush` at
      ``iterOut`` boundaries to batch-write to JSONL.
    * **WS broadcast** is handled externally by the caller (executor passes
      entries to the broadcast function) to keep this class sync-friendly.
    """

    def __init__(
        self,
        execution_id: str,
        source: str,
        persist_dir: str | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.source = source

        self._buffer: deque[NodeLogEntry] = deque(maxlen=2000)
        self._write_queue: list[NodeLogEntry] = []
        self._started_at = datetime.utcnow()
        self._total_firings: int = 0
        self._error_count: int = 0
        self._node_types: set[str] = set()

        self._persist_dir = persist_dir
        self._jsonl_path: str | None = None
        if persist_dir:
            try:
                os.makedirs(persist_dir, exist_ok=True)
                self._jsonl_path = os.path.join(persist_dir, "log.jsonl")
            except OSError as e:
                log.warning("Cannot create log dir %s: %s", persist_dir, e)

    # ── Recording ──

    def log_node(
        self,
        step: int,
        node_id: str,
        node_type: str,
        node_label: str,
        duration_ms: float,
        inputs: dict,
        outputs: dict,
        inner_log: list[dict] | None = None,
        port_wire_types: dict[str, str] | None = None,
        error: str | None = None,
        parent_node_id: str | None = None,
        dynamic_index: int | None = None,
        queue_wait_ms: float | None = None,
        compute_ms: float | None = None,
        transport_ms: float | None = None,
        transfer_bytes: int | None = None,
    ) -> NodeLogEntry:
        """Record one node firing.  Returns the entry for optional WS broadcast.

        ``parent_node_id`` / ``dynamic_index`` carry C.5 dynamic-firelist
        provenance: when set, this entry is for an ephemeral child fired by
        a ``DynamicFireListNode`` spawner. Default ``None`` for regular
        static-topology firings.
        """
        entry = NodeLogEntry(
            execution_id=self.execution_id,
            source=self.source,
            step=step,
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
            duration_ms=round(duration_ms, 2),
            queue_wait_ms=round(queue_wait_ms, 2) if queue_wait_ms is not None else None,
            compute_ms=round(compute_ms, 2) if compute_ms is not None else None,
            transport_ms=round(transport_ms, 2) if transport_ms is not None else None,
            transfer_bytes=transfer_bytes,
            inputs=log_serialize_with_assets(
                inputs,
                self._persist_dir,
                step,
                node_label,
                port_wire_types or {},
                self._total_firings,
            ),
            outputs=log_serialize_with_assets(
                outputs,
                self._persist_dir,
                step,
                node_label,
                port_wire_types or {},
                self._total_firings,
            ),
            inner_log=log_serialize(inner_log or [], max_str_len=50000),
            port_wire_types=port_wire_types or {},
            error=error,
            parent_node_id=parent_node_id,
            dynamic_index=dynamic_index,
        )

        self._buffer.append(entry)
        self._write_queue.append(entry)
        self._total_firings += 1
        self._node_types.add(node_type)
        if error:
            self._error_count += 1

        return entry

    # ── Persistence ──

    def flush(self) -> None:
        """Batch-write queued entries to JSONL.  Non-blocking on failure."""
        if not self._jsonl_path or not self._write_queue:
            return
        try:
            with open(self._jsonl_path, "a") as f:
                for entry in self._write_queue:
                    f.write(entry.json() + "\n")
            self._write_queue.clear()
        except OSError as e:
            log.warning("Failed to write log JSONL %s: %s", self._jsonl_path, e)
            self._write_queue.clear()

    # ── Queries ──

    def get_entries(
        self,
        node_id: str | None = None,
        node_type: str | None = None,
        step: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[NodeLogEntry]:
        """Query the in-memory ring buffer with optional filters."""
        entries: list[NodeLogEntry] = list(self._buffer)
        if node_id:
            entries = [e for e in entries if e.node_id == node_id]
        if node_type:
            entries = [e for e in entries if e.node_type == node_type]
        if step is not None:
            entries = [e for e in entries if e.step == step]
        return entries[offset : offset + limit]

    def get_summary(self) -> ExecutionSummary:
        """Return aggregate stats for this execution."""
        max_step = max((e.step for e in self._buffer), default=0)
        return ExecutionSummary(
            execution_id=self.execution_id,
            source=self.source,
            started_at=self._started_at,
            ended_at=datetime.utcnow(),
            total_steps=max_step,
            total_firings=self._total_firings,
            error_count=self._error_count,
            node_types_fired=sorted(self._node_types),
        )
