"""Standard wire data types — what flows between nodes on edges.

Each type has ONE canonical on-wire format (the single source of truth is
``WIRE_FORMAT_SPEC`` below). A *producer* must emit exactly that format; a
*consumer* may assume it and never defensively normalise. So a node author
never reasons about whether an image is uint8 0-255 vs float 0-1 vs -1~1, or
whether depth is metres vs normalised — the wire type pins it. Formats are a
documented contract, not runtime-enforced (the load-time validator only checks
type-name compatibility); see ``design-docs/graph/wire-types.html``.

The catalog spans embodied task families (VLN / EQA / VLA / manipulation), not
just VLN. The action surface is split by *kind* rather than one VLN-shaped int:
``DISCRETE_ACTION`` (pick from an env-declared set), ``CONTROL`` (continuous
motor command), with ``POSE`` doubling as a goal-pose/teleport action and
``TEXT`` as a free-text answer. No raw-ndarray escape type — structured payloads
carry named fields.

Mirrors ComfyUI's typed IO system but for embodied agent loops.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore  — numpy not available in this env


# ── Wire Type Names (used in node input_ports / output_ports) ──
#
# The canonical format of each is the single source of truth in
# ``WIRE_FORMAT_SPEC`` (below). The trailing comment here is the short form.

# Perception
IMAGE = "IMAGE"  # np.ndarray (H, W, 3) uint8, RGB, 0-255
DEPTH = "DEPTH"  # np.ndarray (H, W) float32, metres (invalid = 0.0)

# Spatial / state
POSE = "POSE"  # dict {"position": [x,y,z] metres world, "orientation": [x,y,z,w] unit quaternion}

# Action — split by kind so the catalog spans VLN / VLA / manipulation
DISCRETE_ACTION = "DISCRETE_ACTION"  # int | str — index/id into the env's declared valid set
CONTROL = "CONTROL"  # dict {"pos":[3], "rot":[3] axis-angle rad, "gripper": float∈[-1,1] (+1=open), "joint_position"?:[N]}
# POSE doubles as a goal-pose / teleport action; TEXT doubles as a free-text answer (EQA).

# Language / scalar / episode
TEXT = "TEXT"  # str
BOOL = "BOOL"  # bool
METRICS = "METRICS"  # dict[str, float]
ANY = "ANY"  # escape hatch — unstructured or research data (plain dicts, lists, numbers)

# ── Deprecated names (kept registered for backward-compat until the migration
# sweep flips PortDefs over; see ``WIRE_TYPE_ALIASES`` + roadmap) ──
ACTION = "ACTION"  # DEPRECATED alias of DISCRETE_ACTION (was VLN-only int 0-3)
OBSERVATION = "OBSERVATION"  # DEPRECATED — dead; flat rgb/depth ports won. Drop in sweep
STEP_RESULT = "STEP_RESULT"  # DEPRECATED — dead; gym tuple (reward/terminated/truncated/info) won

# Registry of inner wire types.  A full wire type may also be ``LIST[<inner>]``
# (see ``is_list_type``/``unwrap_list``) — a consumer-side modifier that lets a
# port carry ``list[<inner>]``.  ``WIRE_TYPES`` always refers to the *inner*
# type registry; use ``is_valid_wire_type`` to validate full type strings.
WIRE_TYPES = {
    # current
    IMAGE,
    DEPTH,
    POSE,
    DISCRETE_ACTION,
    CONTROL,
    TEXT,
    BOOL,
    METRICS,
    ANY,
    # deprecated — still valid so legacy graphs/nodesets load until migration
    ACTION,
    OBSERVATION,
    STEP_RESULT,
}

# ── Canonical format spec — the single source of truth a node author trusts ──
#
# Maps each *current* wire type to a one-line description of its ONE on-wire
# format (dtype / shape / range / units / layout). Surfaced in the design-doc
# and the ``/api/graphs/validate`` report. Not runtime-enforced; producers are
# contractually responsible for emitting exactly this.
WIRE_FORMAT_SPEC: dict[str, str] = {
    IMAGE: "np.ndarray (H, W, 3) uint8, RGB, range 0-255",
    DEPTH: "np.ndarray (H, W) float32, metres; invalid = 0.0; never normalised on-wire",
    POSE: "dict{position:[x,y,z] float metres (world), orientation:[x,y,z,w] unit quaternion}",
    DISCRETE_ACTION: "int | str — index or id into the env's declared valid action set",
    CONTROL: (
        "dict{pos:[3] float, rot:[3] float axis-angle rad, "
        "gripper: float in [-1,1] (+1=open), joint_position?:[N] float}"
    ),
    TEXT: "str",
    BOOL: "bool",
    METRICS: "dict[str, float]",
    ANY: "any python/JSON-safe value — escape hatch, no format contract",
}


# ── LIST[T] modifier helpers (ADR-027) ──
#
# ``LIST[T]`` is a wire-type modifier, not a new wire type.  It wraps any
# existing inner type ``T`` to mean "this port carries ``list[T]``".  The
# executor coerces scalar inputs to length-1 lists at the consumer port
# binding seam, and concatenates fan-in values in edge declaration order —
# producers stay scalar unless they genuinely emit a list.

_LIST_PREFIX = "LIST["
_LIST_SUFFIX = "]"


def is_list_type(wire_type: str) -> bool:
    """Return True if ``wire_type`` is of the form ``LIST[<inner>]``."""
    return (
        isinstance(wire_type, str)
        and wire_type.startswith(_LIST_PREFIX)
        and wire_type.endswith(_LIST_SUFFIX)
    )


def unwrap_list(wire_type: str) -> str:
    """Strip the ``LIST[...]`` wrapper.  Returns ``wire_type`` unchanged if
    it is not a list type."""
    if is_list_type(wire_type):
        return wire_type[len(_LIST_PREFIX) : -len(_LIST_SUFFIX)]
    return wire_type


def wrap_list(inner: str) -> str:
    """Wrap ``inner`` as ``LIST[inner]``.  Idempotent on already-wrapped types."""
    if is_list_type(inner):
        return inner
    return f"{_LIST_PREFIX}{inner}{_LIST_SUFFIX}"


def is_valid_wire_type(wire_type: str) -> bool:
    """Return True if ``wire_type`` is a known inner type or a valid
    ``LIST[<inner>]`` wrapper around one."""
    if not isinstance(wire_type, str):
        return False
    if wire_type in WIRE_TYPES:
        return True
    if is_list_type(wire_type):
        return unwrap_list(wire_type) in WIRE_TYPES
    return False


# ── Deprecated-name aliases ──
#
# Maps a deprecated wire-type name to its current canonical name.  Callers
# canonicalise BOTH sides of an edge before a compatibility check, so a legacy
# ``ACTION`` port and a migrated ``DISCRETE_ACTION`` port still connect cleanly
# while the migration sweep is in flight.  Remove an entry once every PortDef
# for the old name has been flipped.

WIRE_TYPE_ALIASES = {
    ACTION: DISCRETE_ACTION,
}


def canonical_wire_type(wire_type: str) -> str:
    """Resolve a deprecated wire-type name to its current canonical name.

    ``LIST[<inner>]`` is canonicalised on its inner type.  Non-deprecated and
    unknown types pass through unchanged.
    """
    if not isinstance(wire_type, str):
        return wire_type
    if is_list_type(wire_type):
        return wrap_list(canonical_wire_type(unwrap_list(wire_type)))
    return WIRE_TYPE_ALIASES.get(wire_type, wire_type)


# ── Type Validation Helpers ──


def is_valid_image(obj: Any) -> bool:
    """Check if obj is a valid IMAGE wire value."""
    return isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[2] == 3


def is_valid_depth(obj: Any) -> bool:
    """Check if obj is a valid DEPTH wire value."""
    return isinstance(obj, np.ndarray) and obj.ndim == 2


def is_valid_discrete_action(obj: Any) -> bool:
    """Check if obj is a valid DISCRETE_ACTION wire value.

    An int index or a str id into the env's declared valid set.  No fixed
    range — the env owns the valid set (VLN 0-3, RxR 0-5, MP3D viewpoint id …).
    ``bool`` is excluded (it is an ``int`` subclass but a different wire type).
    """
    return (isinstance(obj, int) and not isinstance(obj, bool)) or isinstance(obj, str)


# DEPRECATED alias — ``ACTION`` is now ``DISCRETE_ACTION`` (range no longer 0-3).
def is_valid_action(obj: Any) -> bool:
    """Deprecated alias of :func:`is_valid_discrete_action`."""
    return is_valid_discrete_action(obj)


def is_valid_control(obj: Any) -> bool:
    """Check if obj is a valid CONTROL wire value.

    A named-field continuous motor command — at minimum a ``gripper`` or a
    motion component (``pos`` / ``rot`` / ``joint_position``).  Field shapes are
    a documented contract, not asserted here.
    """
    return isinstance(obj, dict) and any(
        k in obj for k in ("pos", "rot", "gripper", "joint_position")
    )


def is_valid_pose(obj: Any) -> bool:
    """Check if obj is a valid POSE wire value."""
    return isinstance(obj, dict) and "position" in obj and "orientation" in obj


def is_valid_list_of(inner_type: str, value: Any) -> bool:
    """Check if ``value`` is a list whose elements are all valid ``inner_type``.

    Used to validate values arriving at a ``LIST[T]`` consumer port.  Unknown
    inner types pass the element check (we only enforce shape for the known
    array-shaped types).
    """
    if not isinstance(value, list):
        return False
    inner_type = canonical_wire_type(inner_type)
    if inner_type == IMAGE:
        return all(is_valid_image(v) for v in value)
    if inner_type == DEPTH:
        return all(is_valid_depth(v) for v in value)
    if inner_type == DISCRETE_ACTION:
        return all(is_valid_discrete_action(v) for v in value)
    if inner_type == CONTROL:
        return all(is_valid_control(v) for v in value)
    if inner_type == POSE:
        return all(is_valid_pose(v) for v in value)
    # TEXT/BOOL/METRICS/ANY: loose check — any list OK
    return True


# ── Conversion Helpers (nodes call these internally) ──


def image_to_base64(image: np.ndarray) -> str:
    """Convert IMAGE wire type to base64 PNG string. For broadcast/LLM API."""
    import base64
    import io

    from PIL import Image

    img = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def depth_to_base64(depth: np.ndarray) -> str:
    """Convert DEPTH wire type to base64 PNG string. For broadcast/frontend."""
    import base64
    import io

    from PIL import Image

    d = np.squeeze(depth)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    img = Image.fromarray(d_norm, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_thumb_jpeg_base64(image: np.ndarray, max_px: int = 256, quality: int = 70) -> str:
    """Downscaled JPEG base64 of an IMAGE — for viewer display ONLY.

    Viewer cells render at ~80-140px, so a full-resolution PNG wastes ~30x
    bandwidth and can back up the WebSocket send buffer: a single multi-MB
    drain parks long enough for the keepalive ping's drain to collide
    (``assert waiter is None`` → connection drop, seen with 36-view panorama
    over a tunnel). Capping the longest side at ``max_px`` and using JPEG keeps
    each tile a few KB. Deliberately distinct from :func:`image_to_base64`,
    which stays lossless PNG because it also feeds the LLM vision API.
    """
    import base64
    import io

    from PIL import Image

    img = Image.fromarray(image.astype(np.uint8))
    img.thumbnail((max_px, max_px))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def depth_to_thumb_jpeg_base64(depth: np.ndarray, max_px: int = 256, quality: int = 70) -> str:
    """Downscaled JPEG base64 of a DEPTH map — for viewer display ONLY.

    Same rationale as :func:`image_to_thumb_jpeg_base64`; mirrors the min-max
    normalisation in :func:`depth_to_base64` before downscaling.
    """
    import base64
    import io

    from PIL import Image

    d = np.squeeze(depth)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    img = Image.fromarray(d_norm, mode="L")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def serialize_for_display(wire_type: str, value: Any) -> Any:
    """Convert wire data to JSON-safe format for frontend viewer display.

    Each wire type has a canonical serialization:
    - IMAGE/DEPTH → base64 PNG string
    - DISCRETE_ACTION → {"action": int|str, "action_name": str}
    - CONTROL → named-field dict with arrays coerced to lists
    - TEXT/BOOL/METRICS/POSE → passthrough (already JSON-safe)
    - OBSERVATION (deprecated) → {"rgb": base64, "depth": base64}
    """
    from .actions import ACTION_NAMES

    # LIST[T] — render as {"count": N} in v1 (no per-tile thumbnails).
    # The log panel detects this shape and formats as "[N <inner>(s)]".
    if is_list_type(wire_type):
        if isinstance(value, list):
            return {"count": len(value)}
        return {"count": 0}

    # Canonicalise so legacy ``ACTION`` renders via the DISCRETE_ACTION branch.
    wire_type = canonical_wire_type(wire_type)

    if wire_type in (IMAGE, DEPTH):
        if isinstance(value, str) and len(value) > 100:
            return value  # already base64
        if np is not None and isinstance(value, np.ndarray):
            return image_to_base64(value) if wire_type == IMAGE else depth_to_base64(value)
        return None

    if wire_type == DISCRETE_ACTION:
        if isinstance(value, str):
            return {"action": value, "action_name": value}
        try:
            a = int(value)
            return {"action": a, "action_name": ACTION_NAMES.get(a, f"ACTION_{a}")}
        except (ValueError, TypeError):
            return {"action": value, "action_name": str(value)}

    if wire_type == CONTROL and isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if np is not None and isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    if wire_type == OBSERVATION and isinstance(value, dict):
        result: dict[str, Any] = {}
        rgb = value.get("rgb")
        if np is not None and isinstance(rgb, np.ndarray):
            result["rgb"] = image_to_base64(rgb)
        depth = value.get("depth")
        if np is not None and isinstance(depth, np.ndarray):
            result["depth"] = depth_to_base64(depth)
        return result or value

    # TEXT, BOOL, METRICS, POSE, STEP_RESULT, ANY — passthrough
    return value


def serialize_for_viewer(wire_type: str, value: Any, max_px: int = 256) -> Any:
    """Like :func:`serialize_for_display`, but emits downscaled JPEG thumbnails
    for IMAGE/DEPTH so viewer broadcasts stay a few KB per tile instead of a
    full-resolution PNG. Accepts the same input shapes as
    :func:`serialize_for_display` (numpy array OR already-base64 string), and
    always returns JPEG base64 for IMAGE/DEPTH (frontend renders these as
    ``data:image/jpeg``). Non-image types delegate unchanged.
    """
    wt = canonical_wire_type(wire_type)
    if wt == IMAGE:
        if np is not None and isinstance(value, np.ndarray):
            return image_to_thumb_jpeg_base64(value, max_px)
        if isinstance(value, str) and len(value) > 100:
            return image_to_thumb_jpeg_base64(base64_to_image(value), max_px)
        return None
    if wt == DEPTH:
        if np is not None and isinstance(value, np.ndarray):
            return depth_to_thumb_jpeg_base64(value, max_px)
        if isinstance(value, str) and len(value) > 100:
            return depth_to_thumb_jpeg_base64(base64_to_depth(value), max_px)
        return None
    return serialize_for_display(wire_type, value)


def base64_to_image(b64: str) -> np.ndarray:
    """Convert base64 PNG string back to IMAGE wire type."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO(base64.b64decode(b64))
    img = Image.open(buf).convert("RGB")
    return np.array(img, dtype=np.uint8)


def base64_to_depth(b64: str) -> np.ndarray:
    """Convert base64 PNG string back to DEPTH wire type.

    Returns a float32 ndarray with values in [0, 1] (normalized).
    Note: ``depth_to_base64`` normalises to 0-255 during encoding,
    so the original depth scale is lost.  This is acceptable for
    transport — server nodes produce fresh depth arrays.
    """
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO(base64.b64decode(b64))
    img = Image.open(buf).convert("L")
    return np.array(img, dtype=np.float32) / 255.0
