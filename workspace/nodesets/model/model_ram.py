from __future__ import annotations

"""RAM (Recognize Anything Model) tagging — dedicated server-mode nodeset.

Hosts RAM Swin-L 14M tag inference in its own conda env (`ac-ram`)
to avoid version coupling with `opennav` (which carries a transformers
release that broke the RAM library's `apply_chunking_to_forward` import).
Per the per-model env discipline (memory: feedback_dedicated_env_per_model),
each new model that needs server mode gets a clean conda env.

Source: DiscussNav.py:135 + 165-178 — RAM init + per-direction
``ram_img_tagging`` inside ``Vision_Perception_Experts.observe_environment``.

Two tools:

    model_ram__tag_panorama  (views: list[{dir_id, rgb_base64}])
                             → tags_per_dir: LIST[TEXT]
    model_ram__tag_views     (views: {key: {rgb_base64}})
                             → tags: {key: tag string}

``tag_panorama`` tags every direction in order; output aligned 1:1 with input
``views``; each entry is the space-joined RAM tag string after stripping RAM's
`" |"` separators (DiscussNav convention, unchanged since 2026-05-10).

``tag_views`` (added 2026-07-04, TODO #56 extraction) is the generic keyed-dict
primitive that replaces the method-embedded RAM copies in ``opennav_perception``
and ``smartway_perception``. Per-node config picks the variant:

    variant         ram | ram_plus       (factory ``ram.models.ram`` / ``ram_plus``)
    image_size      384 (default) | 224  (Open-Nav runs 224, upstream-faithful)
    keep_separators false (default)      (Open-Nav keeps RAM's raw " |" — an
                                          LLM-visible upstream fidelity detail)

One resident model per (variant, ckpt, image_size) — Open-Nav's ram@224 and
SmartWay's ram_plus@384 can co-reside in this server (~2 models' VRAM).

Runs in `ac-ram` env (Python 3.10, torch 2.4.1, transformers
<4.40, recognize-anything from upstream git — ships both ``ram`` and
``ram_plus`` factories; import verified 2026-07-04).

last updated: 2026-07-04
"""

import asyncio
import base64
import io
import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.ram_perception")

# Per-server content-hash cache: (variant|image_size|keep|sha1(rgb_base64)) →
# tag string. Reuses tags for byte-identical views recurring across
# steps/workers (upstream view_record). The key MUST carry the variant /
# image_size / separator mode or one config's tags would cross-serve another's.
_TAG_CACHE: dict[str, str] = {}


def _find_repo_root() -> str:
    """Walk upward from this file until a dir containing ``data/`` is found.

    A fixed ``../../..`` breaks when this file is served from a workspace
    OVERLAY copy (e.g. ``<overlay>/nodesets/model/…`` sits at a different
    depth than frozen ``workspace/nodesets/model/…``) — the 2026-07-04 smoke
    failures traced to exactly that. Fall back to the auto_host server CWD
    (``<repo>/agentcanvas/backend``) if no anchor is found.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 7):
        cand = os.path.normpath(os.path.join(here, *[".."] * up))
        if os.path.isdir(os.path.join(cand, "data")):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), "..", ".."))


_REPO_ROOT = _find_repo_root()
_RAM_CKPT_DEFAULT = os.environ.get(
    "RAM_CKPT",
    os.path.join(_REPO_ROOT, "data", "opennav", "ram_swin_large_14m.pth"),
)
_RAM_PLUS_CKPT_DEFAULT = os.environ.get(
    "RAM_PLUS_CKPT",
    os.path.join(_REPO_ROOT, "data", "smartway", "ram_plus", "ram_plus_swin_large_14m.pth"),
)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per (variant, ckpt, image_size)
# ══════════════════════════════════════════════════════════════════════


class _RAMEngine:
    """Lazy singleton registry: one resident model per (variant, ckpt, image_size).

    ``variant`` picks the factory (``ram.models.ram`` vs ``ram.models.ram_plus``
    — same package, same transform/inference API). Both RAM@224 (Open-Nav) and
    RAM++@384 (SmartWay) can co-reside in one server subprocess.
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()
    _default_ckpt: ClassVar[dict] = {
        "ram": _RAM_CKPT_DEFAULT,
        "ram_plus": _RAM_PLUS_CKPT_DEFAULT,
    }

    def __init__(self, variant: str, ckpt: str, image_size: int) -> None:
        self.variant = variant
        self.ckpt = ckpt
        self.image_size = image_size
        self.device = None
        self.model = None
        self.transform = None
        self._loaded = False

    @classmethod
    def get(
        cls,
        variant: str = "ram",
        ckpt: str | None = None,
        image_size: int = 384,
    ) -> _RAMEngine:
        if variant not in cls._default_ckpt:
            raise ValueError(f"unknown RAM variant {variant!r} (ram | ram_plus)")
        resolved = ckpt or cls._default_ckpt[variant]
        key = (variant, resolved, image_size)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(variant, resolved, image_size)
            return cls._instances[key]

    def _ensure(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            log.info(
                "Loading %s Swin-L from %s (image_size=%d)",
                self.variant, self.ckpt, self.image_size,
            )
            import torch
            from ram import get_transform  # type: ignore

            if self.variant == "ram_plus":
                from ram.models import ram_plus as factory  # type: ignore
            else:
                from ram.models import ram as factory  # type: ignore

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = factory(pretrained=self.ckpt, image_size=self.image_size, vit="swin_l")
            model.eval().to(self.device)
            self.model = model
            self.transform = get_transform(image_size=self.image_size)
            self._loaded = True
            log.info("%s ready (device=%s)", self.variant, self.device)

    def tag(self, rgb: np.ndarray, keep_separators: bool = False) -> str:
        self._ensure()
        from PIL import Image
        from ram import inference_ram  # type: ignore

        pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        image = self.transform(pil).unsqueeze(0).to(self.device)
        result = inference_ram(image, self.model)
        if isinstance(result, tuple):
            tags_en = result[0]
        else:
            tags_en = str(result)
        if keep_separators:
            # Open-Nav passes the RAM output RAW — "tag | tag" separators
            # included (api.py:109-110); an LLM-visible fidelity detail.
            return tags_en
        return tags_en.replace(" |", "").strip()

    def cache_key(self, keep_separators: bool, b64: str) -> str:
        import hashlib

        digest = hashlib.sha1(b64.encode("ascii")).hexdigest()
        return f"{self.variant}|{self.image_size}|{int(keep_separators)}|{digest}"


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class TagPanoramaTool(BaseCanvasNode):
    """RAM-tag every direction in a panorama, output ordered.

    Source: DiscussNav.py:165-178 — ``ram_img_tagging`` per heading inside
    ``observe_environment`` (a 12-direction sweep).
    """

    node_type: ClassVar[str] = "model_ram__tag_panorama"
    display_name: ClassVar[str] = "RAM: Tag Panorama"
    description: ClassVar[str] = (
        "Recognize Anything (Swin-L 14M tags) per direction; ordered LIST[TEXT] output"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Tag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64} dicts (e.g. 12 directions)"),
    ]
    output_ports = [
        PortDef(
            "tags_per_dir",
            "LIST[TEXT]",
            "Ordered list of space-joined RAM tag strings, aligned 1:1 with `views`",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        if not views:
            return {"tags_per_dir": []}

        loop = asyncio.get_running_loop()
        engine = _RAMEngine.get()

        def _tag_all() -> list[str]:
            out: list[str] = []
            for v in views:
                if not isinstance(v, dict):
                    out.append("")
                    continue
                b64 = v.get("rgb_base64")
                if not b64:
                    out.append("")
                    continue
                # Content-hash cache: identical view bytes (same vp+heading,
                # recurring across steps on backtrack) reuse the tag — mirrors
                # upstream DiscussNav view_record caching, cheap and transparent.
                key = engine.cache_key(False, b64)
                if key in _TAG_CACHE:
                    out.append(_TAG_CACHE[key])
                    continue
                try:
                    tag = engine.tag(_decode_rgb(b64))
                except Exception as exc:
                    log.warning("RAM tag failed for %s: %s", v.get("dir_id"), exc)
                    tag = ""
                _TAG_CACHE[key] = tag
                out.append(tag)
            return out

        tags = await loop.run_in_executor(None, _tag_all)
        self._self_log("num_directions", len(tags))
        for i, t in enumerate(tags):
            self._self_log(f"tags_{i}", t[:200])
        return {"tags_per_dir": tags}


class TagViewsTool(BaseCanvasNode):
    """RAM/RAM++ tag every view in a keyed dict, keys preserved.

    Generic keyed-batch primitive (TODO #56 extraction of the RAM copies
    embedded in ``opennav_perception`` / ``smartway_perception``):
    ``{key: view}`` in → ``{key: tags}`` out, insertion order preserved.
    Keys are opaque (dir_id, idx, …); the caller owns selection/filtering.
    """

    node_type: ClassVar[str] = "model_ram__tag_views"
    display_name: ClassVar[str] = "RAM: Tag Views"
    description: ClassVar[str] = "RAM / RAM++ tags per keyed view; {key: tags} output"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Tag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "variant", "select", "Model variant",
                options=[
                    {"value": "ram", "label": "RAM"},
                    {"value": "ram_plus", "label": "RAM++"},
                ],
                default="ram",
            ),
            ConfigField("image_size", "text", "Input resolution (224 | 384)", default=384),
            ConfigField(
                "keep_separators", "toggle",
                "Keep RAM's raw ' |' separators (Open-Nav upstream fidelity)",
                default=False,
            ),
            ConfigField("ckpt", "text", "Checkpoint override (blank = per-variant default)", default=""),
        ],
    )
    input_ports = [
        PortDef("views", "ANY", "{key: {rgb_base64}} keyed view dict"),
    ]
    output_ports = [PortDef("tags", "ANY", "{key: tag string}")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or {}
        if not views:
            return {"tags": {}}

        cfg = getattr(self, "config", None) or {}
        variant = cfg.get("variant", "ram")
        image_size = int(cfg.get("image_size", 384))
        keep_separators = bool(cfg.get("keep_separators", False))
        ckpt = str(cfg.get("ckpt", "") or "").strip() or None

        loop = asyncio.get_running_loop()
        engine = _RAMEngine.get(variant, ckpt, image_size)

        def _tag_all() -> dict[str, str]:
            out: dict[str, str] = {}
            for key, v in views.items():
                key = str(key)
                b64 = v.get("rgb_base64") if isinstance(v, dict) else None
                if not b64:
                    out[key] = ""
                    continue
                cache_key = engine.cache_key(keep_separators, b64)
                if cache_key in _TAG_CACHE:
                    out[key] = _TAG_CACHE[cache_key]
                    continue
                try:
                    tag = engine.tag(_decode_rgb(b64), keep_separators=keep_separators)
                except Exception as exc:
                    log.warning("RAM tag failed for %s: %s", key, exc)
                    tag = ""
                _TAG_CACHE[cache_key] = tag
                out[key] = tag
            return out

        tags = await loop.run_in_executor(None, _tag_all)
        self._self_log("variant", variant)
        self._self_log("num_views", len(tags))
        return {"tags": tags}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class RAMPerceptionNodeSet(BaseNodeSet):
    """RAM / RAM++ Swin-L 14M tag inference, dedicated env."""

    name = "model_ram"
    description = "Recognize Anything Model (RAM / RAM++, Swin-L 14M tags) — dedicated server-mode nodeset"
    server_python = os.environ.get(
        "RAM_PERCEPTION_PYTHON",
        os.path.expanduser("~/miniforge3/envs/ac-ram/bin/python"),
    )

    def get_tools(self) -> list:
        return [TagPanoramaTool(), TagViewsTool()]

    async def initialize(self, **kwargs: Any) -> None:
        if "ram_ckpt" in kwargs:
            _RAMEngine._default_ckpt["ram"] = str(kwargs["ram_ckpt"])
        if "ram_plus_ckpt" in kwargs:
            _RAMEngine._default_ckpt["ram_plus"] = str(kwargs["ram_plus_ckpt"])
        log.info(
            "RAMPerceptionNodeSet ready (ram=%s, ram_plus=%s)",
            _RAMEngine._default_ckpt["ram"],
            _RAMEngine._default_ckpt["ram_plus"],
        )

    async def shutdown(self) -> None:
        pass
