"""Wire type serialization for HTTP transport.

Self-contained — no AgentCanvas imports — so servers can use it standalone.
Depends on numpy, Pillow, and msgpack (msgpack is imported lazily so this
module still loads — and the legacy JSON path still works — in an env that
has not installed it yet).

Two transport encodings live here:

- **JSON path (legacy / migration window)** — ``serialize_value`` /
  ``deserialize_value`` (per-port, wire-type aware) plus ``_make_json_safe`` /
  ``_restore_ndarrays`` (the ``__ndarray__`` base64 marker). Still accepted by
  ``/call`` so an un-upgraded caller keeps working.
- **msgpack path (Move 1, default)** — ``pack_body`` / ``unpack_body`` encode
  the whole ``{"inputs", "config"}`` body in one shot. Binary types that JSON
  can't hold (ndarray / torch.Tensor / PIL.Image) ride a single ExtType
  (code 1, "blob") as raw bytes — no base64, no text parse; ``bytes`` ride
  msgpack's native bin type. Decoding is *type-driven* (not wire-type-driven)
  and **degrades** when the receiving env lacks a type (a torch/PIL blob comes
  back as ndarray rather than crashing — see ``_decode_blob``).
"""

from __future__ import annotations

import base64
import io
import struct
from contextvars import ContextVar
from typing import Any

import numpy as np

try:  # capability probe — actual use is lazy (see pack_body / unpack_body)
    import msgpack as _msgpack  # noqa: F401

    MSGPACK_OK = True
except ImportError:  # pragma: no cover - env without msgpack falls back to JSON
    MSGPACK_OK = False

# ── transport accounting (System Log P2) ──
# Per-node-firing bucket, set by GraphExecutor around each forward() and read
# back into the node's log entry. The proxy forward() accumulates one entry per
# server round-trip. ContextVar so concurrent firings don't cross-contaminate.
_current_node_transport: ContextVar["dict | None"] = ContextVar(
    "_current_node_transport", default=None
)


def accumulate_transport(
    rtt_ms: float = 0.0,
    req_bytes: int = 0,
    resp_bytes: int = 0,
    serialize_ms: float = 0.0,
    deserialize_ms: float = 0.0,
) -> None:
    """Add one server round-trip's transport metrics to the active bucket.
    No-op when none is set (a call outside the executor's firing path)."""
    b = _current_node_transport.get()
    if b is None:
        return
    b["calls"] += 1
    b["rtt_ms"] += rtt_ms
    b["req_bytes"] += req_bytes
    b["resp_bytes"] += resp_bytes
    b["serialize_ms"] += serialize_ms
    b["deserialize_ms"] += deserialize_ms


# ── IMAGE (np.ndarray uint8 H,W,3) ↔ base64 PNG ──


def _image_to_base64(image: np.ndarray) -> str:
    from PIL import Image

    img = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _base64_to_image(b64: str) -> np.ndarray:
    from PIL import Image

    buf = io.BytesIO(base64.b64decode(b64))
    img = Image.open(buf).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ── DEPTH transport ──
#
# DEPTH is float32 metres. PNG is 8/16-bit integer and cannot hold float32, so
# there is NO depth PNG codec on this (data) path — DEPTH rides the lossless
# ``__ndarray__`` marker like any other ndarray, preserving exact metric scale
# across the server-mode HTTP boundary. The lossy normalise-to-heatmap PNG used
# by the WebSocket viewer is display-only and lives in ``standard.wire_types``.


# ── Public API ──


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types (numpy arrays, etc.).

    Numpy arrays are encoded as compact base64 blobs with shape/dtype
    metadata (``__ndarray__`` marker), preserving exact values and dtype.
    """
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": base64.b64encode(obj.tobytes()).decode("ascii"),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    return obj


def _restore_ndarrays(obj: Any) -> Any:
    """Recursively restore numpy arrays from ``__ndarray__`` markers."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            raw = base64.b64decode(obj["__ndarray__"])
            dtype = np.dtype(obj["dtype"])
            shape = obj["shape"]
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        return {k: _restore_ndarrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_ndarrays(v) for v in obj]
    return obj


def serialize_value(value: Any, wire_type: str) -> Any:
    """Convert a native Python value to its JSON-safe representation.

    IMAGE is encoded as a base64 PNG string — lossless for uint8 RGB and
    compact for natural images. DEPTH is float32 metres, which PNG cannot hold,
    so it rides the lossless ``__ndarray__`` marker via ``_make_json_safe``
    (the metric scale must survive the server-mode HTTP boundary). All other
    types are made JSON-safe (numpy arrays → ``__ndarray__``, etc.).
    """
    if value is None:
        return None
    if wire_type == "IMAGE" and isinstance(value, np.ndarray):
        return _image_to_base64(value)
    # DEPTH and every other ndarray fall through to the lossless __ndarray__ path.
    return _make_json_safe(value)


def deserialize_value(value: Any, wire_type: str) -> Any:
    """Convert a JSON value back to its native Python representation.

    IMAGE base64 PNG strings are decoded to np.ndarray. DEPTH and all other
    types arrive as ``__ndarray__`` markers (or plain JSON) and are restored by
    ``_restore_ndarrays`` — DEPTH comes back as exact float32 metres.
    """
    if value is None:
        return None
    if wire_type == "IMAGE" and isinstance(value, str):
        return _base64_to_image(value)
    return _restore_ndarrays(value)


# ── msgpack wire codec (Move 1) ──
#
# Whole-body transport for the server-mode HTTP boundary. A single ExtType
# (code 1, "blob") carries raw bytes for the binary types JSON can't represent;
# everything else rides msgpack natively. Framing inside the ext payload:
#
#     [tag:u8][meta_len:u32 big-endian][meta(msgpack)][raw bytes]
#
# The msgpack ``default`` hook encodes on the way out and ``ext_hook`` decodes
# on the way in. Decoding is type-driven and degrades when a type is missing
# from the receiving env (constraint A: cross-boundary = cross-env).

MSGPACK_CONTENT_TYPE = "application/msgpack"

_EXT_BLOB = 1

_TAG_NDARRAY = 0
_TAG_TORCH = 1
_TAG_PIL = 2


def _encode_blob(tag: int, meta: dict, raw: bytes) -> Any:
    import msgpack

    meta_bytes = msgpack.packb(meta, use_bin_type=True)
    framed = bytes([tag]) + struct.pack(">I", len(meta_bytes)) + meta_bytes + raw
    return msgpack.ExtType(_EXT_BLOB, framed)


def _decode_blob(data: bytes) -> Any:
    import msgpack

    tag = data[0]
    (meta_len,) = struct.unpack(">I", data[1:5])
    meta = msgpack.unpackb(data[5 : 5 + meta_len], raw=False)
    raw = bytes(data[5 + meta_len :])
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"]).copy()
    if tag == _TAG_NDARRAY:
        return arr
    if tag == _TAG_TORCH:
        try:
            import torch

            return torch.from_numpy(arr)
        except ImportError:
            # Receiver-side degrade: this env has no torch → hand back the
            # ndarray rather than crash. The data is intact; only the wrapper
            # type is dropped (constraint A — rebuilding a type is not free
            # across envs).
            return arr
    if tag == _TAG_PIL:
        try:
            from PIL import Image

            img = Image.fromarray(arr)
            mode = meta.get("mode")
            if mode and img.mode != mode:
                img = img.convert(mode)
            return img
        except ImportError:
            return arr
    return arr


def _ext_default(obj: Any) -> Any:
    """msgpack ``default`` hook — encode types msgpack can't natively pack."""
    if isinstance(obj, np.ndarray):
        return _encode_blob(
            _TAG_NDARRAY, {"dtype": str(obj.dtype), "shape": list(obj.shape)}, obj.tobytes()
        )
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            # device / requires_grad are lost across the boundary by contract
            # (constraint A) — only the values + dtype + shape travel.
            arr = obj.detach().cpu().numpy()
            return _encode_blob(
                _TAG_TORCH, {"dtype": str(arr.dtype), "shape": list(arr.shape)}, arr.tobytes()
            )
    except ImportError:
        pass
    try:
        from PIL import Image

        if isinstance(obj, Image.Image):
            arr = np.array(obj)
            return _encode_blob(
                _TAG_PIL,
                {"dtype": str(arr.dtype), "shape": list(arr.shape), "mode": obj.mode},
                arr.tobytes(),
            )
    except ImportError:
        pass
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Cannot serialize {type(obj)!r} over the msgpack wire")


def _ext_hook(code: int, data: bytes) -> Any:
    """msgpack ``ext_hook`` — decode our blob ExtType, passthrough others."""
    import msgpack

    if code == _EXT_BLOB:
        return _decode_blob(data)
    return msgpack.ExtType(code, data)


def pack_body(obj: Any) -> bytes:
    """Encode a request/response body to msgpack bytes (binary types → blobs)."""
    import msgpack

    return msgpack.packb(obj, default=_ext_default, use_bin_type=True)


def unpack_body(data: bytes) -> Any:
    """Decode a msgpack body back to native objects (blobs → ndarray/torch/PIL)."""
    import msgpack

    # strict_map_key=False: pack_body() emits int-keyed maps (e.g. SmartWay's
    # candidates dict keyed by 0..K-1), so unpack must accept them to stay
    # symmetric with pack. The strict default is DoS-hardening for untrusted
    # input; these are trusted internal auto-server responses.
    return msgpack.unpackb(data, ext_hook=_ext_hook, raw=False, strict_map_key=False)
