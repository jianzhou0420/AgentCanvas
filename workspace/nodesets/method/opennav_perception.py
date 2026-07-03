from __future__ import annotations

"""Open-Nav scene perception nodeset (server mode).

Wraps the two vision models used by Open-Nav (ICRA 2025) for fine-grained
scene understanding:

    Recognize Anything (RAM, Swin-L 14M tags)
        https://arxiv.org/abs/2306.03514
        Repo: https://github.com/xinyu1205/recognize-anything

    SpatialBot-3B (Phi-3 derivative VLM with depth-aware captioning)
        https://arxiv.org/abs/2406.13642

Source call sites:

    Open-Nav/vlnce_baselines/common/navigator/api.py :: spatialClient
    (RAM init :: line ~80, SpatialBot init :: ~95, generate :: ~120)

Per-direction usage in the rollout loop:

    tags  = RAM(rgb)
    cap   = SpatialBot([rgb, depth_packed], "What objects ... in meter.")
    obs   = "Scene Description: {cap}\\nScene Objects: {tags};"

Depth packing (verbatim from api.py) — required because SpatialBot expects
a 3-channel image as its second image slot:

    img_packed[..., 0] = (depth // 1024) * 4
    img_packed[..., 1] = (depth // 32)   * 8
    img_packed[..., 2] = (depth %  32)   * 8

Runs in the dedicated ``opennav`` conda env (Python 3.10, modern
transformers, no habitat dependency). Loaded as a server-mode nodeset.

Two nodes:

    opennav_perception__tag      (rgb_b64) → tags string
    opennav_perception__caption  (rgb_b64, depth_b64) → caption string

Both expect a single view; fan out from the panorama_rgbd output across
candidate directions in the graph.

last updated: 2026-04-15
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

log = logging.getLogger("agentcanvas.opennav_perception")


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
)

_RAM_CKPT_DEFAULT = os.environ.get(
    "OPENNAV_RAM_CKPT",
    os.path.join(_REPO_ROOT, "data", "opennav", "ram_swin_large_14m.pth"),
)
_SPATIALBOT_PATH_DEFAULT = os.environ.get(
    "OPENNAV_SPATIALBOT_PATH",
    os.path.join(_REPO_ROOT, "data", "opennav", "SpatialBot-3B"),
)
_SPATIAL_PROMPT = (
    "What objects are in the image, and how far are these objects from the camera, "
    "calculate the result in meter."
)
_SPATIAL_MAX_NEW_TOKENS = 200  # api.py:133


# ══════════════════════════════════════════════════════════════════════
# PerceptionEngine — singleton model loader
# ══════════════════════════════════════════════════════════════════════


class PerceptionEngine:
    """Lazy loader for RAM + SpatialBot. Models load on first use."""

    _instance: "PerceptionEngine | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.ram_ckpt = _RAM_CKPT_DEFAULT
        self.spatialbot_path = _SPATIALBOT_PATH_DEFAULT
        self.device = None
        self.ram_model = None
        self.ram_transform = None
        self.spatial_model = None
        self.spatial_tokenizer = None
        self._ram_loaded = False
        self._spatial_loaded = False

    @classmethod
    def get(cls) -> "PerceptionEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_ram(self) -> None:
        if self._ram_loaded:
            return
        with self._lock:
            if self._ram_loaded:
                return
            log.info("Loading RAM Swin-L from %s", self.ram_ckpt)
            import torch  # noqa: WPS433

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            from ram.models import ram  # type: ignore
            from ram import inference_ram  # noqa: F401  # type: ignore
            from ram import get_transform  # type: ignore

            model = ram(
                pretrained=self.ram_ckpt,
                image_size=224,
                vit="swin_l",
            )
            model.eval()
            model = model.to(self.device)
            self.ram_model = model
            self.ram_transform = get_transform(image_size=224)
            self._ram_loaded = True
            log.info("RAM ready (device=%s)", self.device)

    def _ensure_spatialbot(self) -> None:
        if self._spatial_loaded:
            return
        with self._lock:
            if self._spatial_loaded:
                return
            log.info("Loading SpatialBot-3B from %s", self.spatialbot_path)
            import torch  # noqa: WPS433
            from transformers import (  # noqa: WPS433
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            self.device = self.device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.spatial_tokenizer = AutoTokenizer.from_pretrained(
                self.spatialbot_path, trust_remote_code=True
            )
            self.spatial_model = (
                AutoModelForCausalLM.from_pretrained(
                    self.spatialbot_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.device)
            )
            # The SigLIP vision tower is lazily constructed (is_loaded=False, no
            # params), so the model-level .to(device) misses it — it would then
            # load onto CPU during generate and raise a device-mismatch 500.
            # Force-load and move it (cf. api.py:270 get_vision_tower().to(cuda)).
            vt = self.spatial_model.get_vision_tower()
            if hasattr(vt, "load_model") and not getattr(vt, "is_loaded", False):
                vt.load_model()
            vt.to(device=self.device, dtype=self.spatial_model.dtype)
            self._spatial_loaded = True
            log.info("SpatialBot ready")

    def tag(self, rgb: np.ndarray) -> str:
        self._ensure_ram()
        from PIL import Image  # noqa: WPS433
        from ram import inference_ram  # type: ignore

        pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        image = self.ram_transform(pil).unsqueeze(0).to(self.device)
        result = inference_ram(image, self.ram_model)
        if isinstance(result, tuple):
            tags_en = result[0]
        else:
            tags_en = str(result)
        # Upstream passes the RAM output RAW — "tag | tag | tag" separators
        # included (api.py:109-110, both Open-Nav parent and Three-Step fork);
        # the old ``.replace(" |", "").strip()`` was an LLM-visible divergence
        # (fixed 2026-07-03).
        return tags_en

    @staticmethod
    def _pack_depth(depth_u8: np.ndarray) -> np.ndarray:
        """SpatialBot 3-channel depth packing — byte-faithful to the upstream
        RUNTIME behaviour (api.py:260-268 on the generate_input depth chain).

        Upstream packs the per-tile min-max-normalised **uint8** depth image
        (generate_input, base_il_trainer_llm.py:186-191) — NOT raw 16-bit
        depth — so ``img // 1024`` is always 0 and the R channel is
        degenerate-zero. The previous port packed uint16 millimetres
        ("improved", but not what upstream ever ran); fixed 2026-07-03.
        """
        # NumPy 1.x (upstream env) silently promotes ``uint8 // 1024`` to
        # int16; NumPy ≥2 (ac-ram env, NEP 50) raises OverflowError on the
        # out-of-range literal instead. Promote explicitly — identical bytes
        # to the upstream promotion (R = 0, G ≤ 56, B ≤ 248 all fit uint8).
        img = depth_u8.astype(np.int16)
        height, width = img.shape[-2:]
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8
        return three_channel_array

    def caption(self, rgb: np.ndarray, depth: np.ndarray | None) -> str:
        self._ensure_spatialbot()
        import torch  # noqa: WPS433
        from PIL import Image  # noqa: WPS433

        rgb_pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        images = [rgb_pil]
        if depth is not None:
            packed = self._pack_depth(depth)
            # Image.fromarray on a 3-channel uint8 array = mode RGB, exactly
            # upstream's Image.fromarray(three_channel_array, 'RGB').
            images.append(Image.fromarray(packed))
        else:
            images.append(rgb_pil)

        prompt = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's "
            f"questions. USER: <image 1>\n<image 2>\n{_SPATIAL_PROMPT} ASSISTANT:"
        )

        # Bunny-Phi expects the two image placeholders spliced into the input
        # ids as special token ids -201 / -202 (api.py:255-257), NOT the literal
        # "<image 1>" text — otherwise the model never attends to the images
        # ("I cannot see any images provided").
        chunks = [
            self.spatial_tokenizer(c).input_ids
            for c in prompt.split("<image 1>\n<image 2>\n")
        ]
        text_ids = (
            torch.tensor(chunks[0] + [-201, -202] + chunks[1], dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )
        try:
            image_tensors = self.spatial_model.process_images(
                images, self.spatial_model.config
            ).to(dtype=self.spatial_model.dtype, device=self.device)
        except Exception:
            image_tensors = None

        # Generate args mirror api.py:271-277 EXACTLY (max_new_tokens=200,
        # use_cache=True, repetition_penalty=1.0 — nothing else; upstream
        # relies on the checkpoint's own generation defaults, so injecting
        # do_sample/temperature would diverge). Fixed 2026-07-03.
        with torch.no_grad():
            output_ids = self.spatial_model.generate(
                text_ids,
                images=image_tensors,
                max_new_tokens=_SPATIAL_MAX_NEW_TOKENS,
                use_cache=True,
                repetition_penalty=1.0,
            )
        out = self.spatial_tokenizer.decode(
            output_ids[0][text_ids.shape[1]:], skip_special_tokens=True
        )
        return out.strip()


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image  # noqa: WPS433

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _decode_depth_norm_u8(b64: str) -> np.ndarray:
    # Per-tile min-max-normalised uint8 depth (env encode_depth_base64) —
    # byte-identical to upstream generate_input's ``depth_img``
    # (base_il_trainer_llm.py:186-191), the image SpatialBot's packer
    # actually receives at upstream runtime.
    from PIL import Image  # noqa: WPS433

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)), dtype=np.uint8)


def _depth_norm_u8_from_raw(b64: str) -> np.ndarray:
    # Fallback: reconstruct the upstream normalisation from the 16-bit
    # absolute-depth payload when a view lacks ``depth_base64``.
    from PIL import Image  # noqa: WPS433

    raw = base64.b64decode(b64)
    d = np.asarray(Image.open(io.BytesIO(raw)), dtype=np.float32)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        return (255 * (d - d_min) / (d_max - d_min)).astype(np.uint8)
    return np.zeros_like(d, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# Canvas tools
# ══════════════════════════════════════════════════════════════════════


def _select_views_for_candidates(views: list, candidates: dict) -> list:
    """Return only the view dicts whose ``dir_id`` is in ``candidates``."""
    if not candidates:
        return list(views or [])
    keys = {str(k) for k in candidates.keys()}
    return [v for v in (views or []) if isinstance(v, dict) and str(v.get("dir_id")) in keys]


class TagCandidatesTool(BaseCanvasNode):
    """RAM tag every candidate direction in a panorama.

    Returns ``{dir_id: 'tag tag tag'}``. The reference implementation runs
    one RAM forward pass per candidate inside ``observe_environment``;
    this node folds that loop inside its ``execute()`` so the graph stays
    static and matches the reference call pattern 1:1.
    """

    node_type: ClassVar[str] = "opennav_perception__tag"
    display_name: ClassVar[str] = "Open-Nav: RAM Tag (per candidate)"
    description: ClassVar[str] = "Recognize Anything Swin-L tags, one pass per candidate dir_id"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Tag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64} from panorama_rgbd"),
        PortDef("candidates", "ANY", "{dir_id: ...} from waypoint predictor"),
    ]
    output_ports = [PortDef("tags", "ANY", "{dir_id: tag string}")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        candidates = inputs.get("candidates") or {}
        selected = _select_views_for_candidates(views, candidates)
        if not selected:
            return {"tags": {}}

        loop = asyncio.get_running_loop()
        engine = PerceptionEngine.get()

        def _tag_all() -> dict[str, str]:
            out: dict[str, str] = {}
            for v in selected:
                b64 = v.get("rgb_base64")
                dir_id = str(v.get("dir_id"))
                if not b64:
                    out[dir_id] = ""
                    continue
                out[dir_id] = engine.tag(_decode_rgb(b64))
            return out

        tags = await loop.run_in_executor(None, _tag_all)
        self._self_log("num_directions", len(tags))
        return {"tags": tags}


class CaptionCandidatesTool(BaseCanvasNode):
    """SpatialBot caption every candidate direction in a panorama.

    Returns ``{dir_id: 'caption text'}`` — same loop-fold pattern as
    :class:`TagCandidatesTool`.
    """

    node_type: ClassVar[str] = "opennav_perception__caption"
    display_name: ClassVar[str] = "Open-Nav: SpatialBot Caption (per candidate)"
    description: ClassVar[str] = "Depth-aware spatial caption, one pass per candidate dir_id"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "MessageSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("max_new_tokens", "text", label="max_new_tokens", default=200),
        ],
    )
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64, depth_base64}"),
        PortDef("candidates", "ANY", "{dir_id: ...} from waypoint predictor"),
    ]
    output_ports = [PortDef("captions", "ANY", "{dir_id: caption string}")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        candidates = inputs.get("candidates") or {}
        selected = _select_views_for_candidates(views, candidates)
        if not selected:
            return {"captions": {}}

        loop = asyncio.get_running_loop()
        engine = PerceptionEngine.get()

        def _caption_all() -> dict[str, str]:
            out: dict[str, str] = {}
            for v in selected:
                rgb_b64 = v.get("rgb_base64")
                depth_raw_b64 = v.get("depth_raw_base64")
                depth_b64 = v.get("depth_base64")
                dir_id = str(v.get("dir_id"))
                if not rgb_b64:
                    out[dir_id] = ""
                    continue
                rgb = _decode_rgb(rgb_b64)
                # Upstream packs the per-tile NORMALISED uint8 depth image
                # (generate_input chain) — prefer depth_base64, which is
                # exactly that; reconstruct it from the raw payload otherwise.
                if depth_b64:
                    depth = _decode_depth_norm_u8(depth_b64)
                elif depth_raw_b64:
                    depth = _depth_norm_u8_from_raw(depth_raw_b64)
                else:
                    depth = None
                out[dir_id] = engine.caption(rgb, depth)
            return out

        captions = await loop.run_in_executor(None, _caption_all)
        self._self_log("num_directions", len(captions))
        return {"captions": captions}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class OpenNavPerceptionNodeSet(BaseNodeSet):
    """Open-Nav scene perception (RAM + SpatialBot)."""

    name = "opennav_perception"
    description = "RAM Swin-L tags + SpatialBot-3B depth-aware captions (Open-Nav)"
    # The ``opennav`` env carries transformers 5.5.4, where RAM (recognize_anything)
    # fails to import (apply_chunking_to_forward / find_pruneable_heads_and_indices /
    # transformers.file_utils all moved/removed). The curated ``ac-ram`` env
    # (transformers 4.39.3 + torch 2.4.1) imports RAM cleanly and is also in
    # Bunny-Phi's compatible range, so it hosts both RAM tags and SpatialBot-3B
    # captions. Default there; override via OPENNAV_PERCEPTION_PYTHON.
    server_python = os.environ.get(
        "OPENNAV_PERCEPTION_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-ram/bin/python")
    )

    def get_tools(self) -> list:
        return [TagCandidatesTool(), CaptionCandidatesTool()]

    async def initialize(self, **kwargs: Any) -> None:
        engine = PerceptionEngine.get()
        if "ram_ckpt" in kwargs:
            engine.ram_ckpt = str(kwargs["ram_ckpt"])
        if "spatialbot_path" in kwargs:
            engine.spatialbot_path = str(kwargs["spatialbot_path"])
        log.info(
            "OpenNavPerceptionNodeSet ready (ram=%s, spatialbot=%s)",
            engine.ram_ckpt,
            engine.spatialbot_path,
        )

    async def shutdown(self) -> None:
        pass
