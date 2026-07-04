from __future__ import annotations

"""SpatialBot-3B — generic depth-aware VLM, server-mode foundation-model nodeset.

Extracted from ``opennav_perception`` (where SpatialBot lived co-hosted with RAM
in one method-owned PerceptionEngine) per the method / foundation-model boundary
principle (roadmap TODO #56). The model-side semantics move here **byte-faithful
to upstream** (Open-Nav api.py; deviations fixed 2026-07-03 are preserved):

  - 3-channel depth packing of the per-tile min-max-normalised uint8 depth
    (NOT raw 16-bit), with explicit int16 promotion for NumPy≥2 / NEP-50;
  - Bunny-Phi chat template with the ``<image 1>/<image 2>`` placeholders
    spliced into the input ids as special token ids ``-201 / -202``;
  - ``generate(max_new_tokens=200, use_cache=True, repetition_penalty=1.0)``
    and nothing else (checkpoint's own generation defaults);
  - SigLIP vision-tower force-load fix (lazily-built tower misses the
    model-level ``.to(device)`` → device-mismatch 500 at generate).

Method glue (candidate filtering) stays with the methods:
``opennav__select_candidate_views`` / ``threestepnav__select_candidate_views``
produce the keyed views dict this nodeset consumes.

SpatialBot-3B: https://arxiv.org/abs/2406.13642

Two tools::

    vlm_spatialbot__caption_views  (views: {key: {rgb_base64, depth_base64?,
                                    depth_raw_base64?}}) → captions: {key: str}
    vlm_spatialbot__generate       (rgb_b64, depth_b64?, prompt) → text

Runs **server mode** in the ``ac-ram`` env (transformers 4.39.3 + torch 2.4.1
— in Bunny-Phi's compatible range; the same env that hosted this exact forward
inside ``opennav_perception``, so outputs are unchanged). Override with
$SPATIALBOT_PYTHON if a dedicated env is built (the dedicated env remains the
ideal per memory feedback_dedicated_env_per_model).

Load: POST /api/components/nodesets/vlm_spatialbot/load?mode=server

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

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

log = logging.getLogger("agentcanvas.vlm_spatialbot")

def _find_repo_root() -> str:
    """Walk upward from this file until a dir containing ``data/`` is found.

    A fixed ``../../..`` breaks when this file is served from a workspace
    OVERLAY copy (different depth than the frozen tree) — see the identical
    helper in ``model_ram.py``. Falls back to the auto_host server CWD.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 7):
        cand = os.path.normpath(os.path.join(here, *[".."] * up))
        if os.path.isdir(os.path.join(cand, "data")):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), "..", ".."))


_REPO_ROOT = _find_repo_root()
_SPATIALBOT_PATH_DEFAULT = os.environ.get(
    "SPATIALBOT_PATH",
    os.environ.get(  # legacy name from the opennav_perception era
        "OPENNAV_SPATIALBOT_PATH",
        os.path.join(_REPO_ROOT, "data", "opennav", "SpatialBot-3B"),
    ),
)
_SPATIAL_PROMPT = (
    "What objects are in the image, and how far are these objects from the camera, "
    "calculate the result in meter."
)
_SPATIAL_MAX_NEW_TOKENS = 200  # api.py:133


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton
# ══════════════════════════════════════════════════════════════════════


class _SpatialBotEngine:
    """Lazy singleton: load SpatialBot-3B on first caption() call, stay resident."""

    _instance: _SpatialBotEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model_path = _SPATIALBOT_PATH_DEFAULT
        self.device = None
        self.model = None
        self.tokenizer = None
        self._loaded = False

    @classmethod
    def get(cls) -> _SpatialBotEngine:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            log.info("Loading SpatialBot-3B from %s", self.model_path)
            import torch  # noqa: WPS433
            from transformers import (  # noqa: WPS433
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_path,
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
            vt = self.model.get_vision_tower()
            if hasattr(vt, "load_model") and not getattr(vt, "is_loaded", False):
                vt.load_model()
            vt.to(device=self.device, dtype=self.model.dtype)
            self._loaded = True
            log.info("SpatialBot ready")

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

    def generate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        prompt: str,
        max_new_tokens: int = _SPATIAL_MAX_NEW_TOKENS,
    ) -> str:
        self._ensure()
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

        chat = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's "
            f"questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        )

        # Bunny-Phi expects the two image placeholders spliced into the input
        # ids as special token ids -201 / -202 (api.py:255-257), NOT the literal
        # "<image 1>" text — otherwise the model never attends to the images
        # ("I cannot see any images provided").
        chunks = [
            self.tokenizer(c).input_ids
            for c in chat.split("<image 1>\n<image 2>\n")
        ]
        text_ids = (
            torch.tensor(chunks[0] + [-201, -202] + chunks[1], dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )
        try:
            image_tensors = self.model.process_images(
                images, self.model.config
            ).to(dtype=self.model.dtype, device=self.device)
        except Exception:
            image_tensors = None

        # Generate args mirror api.py:271-277 EXACTLY (max_new_tokens=200,
        # use_cache=True, repetition_penalty=1.0 — nothing else; upstream
        # relies on the checkpoint's own generation defaults, so injecting
        # do_sample/temperature would diverge). Fixed 2026-07-03.
        with torch.no_grad():
            output_ids = self.model.generate(
                text_ids,
                images=image_tensors,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                repetition_penalty=1.0,
            )
        out = self.tokenizer.decode(
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


def _view_depth(v: dict) -> np.ndarray | None:
    """Pick the depth image the upstream packer receives (prefer the
    normalised u8 payload; reconstruct it from raw otherwise)."""
    depth_b64 = v.get("depth_base64")
    depth_raw_b64 = v.get("depth_raw_base64")
    if depth_b64:
        return _decode_depth_norm_u8(depth_b64)
    if depth_raw_b64:
        return _depth_norm_u8_from_raw(depth_raw_b64)
    return None


# ══════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════


class CaptionViewsTool(BaseCanvasNode):
    """SpatialBot caption for every view in a keyed dict, keys preserved.

    Generic keyed-batch primitive: ``{key: view}`` in → ``{key: caption}``
    out, insertion order preserved. Keys are opaque (dir_id, idx, …); the
    caller owns any selection/filtering (e.g. candidate dirs).
    """

    node_type: ClassVar[str] = "vlm_spatialbot__caption_views"
    display_name: ClassVar[str] = "SpatialBot: Caption Views"
    description: ClassVar[str] = "Depth-aware spatial caption per keyed view; {key: caption} output"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "MessageSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("prompt", "text", label="Prompt", default=_SPATIAL_PROMPT),
            ConfigField("max_new_tokens", "text", label="max_new_tokens", default=200),
        ],
    )
    input_ports = [
        PortDef(
            "views", "ANY",
            "{key: {rgb_base64, depth_base64?, depth_raw_base64?}} keyed view dict",
        ),
    ]
    output_ports = [PortDef("captions", "ANY", "{key: caption string}")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or {}
        if not views:
            return {"captions": {}}

        cfg = getattr(self, "config", None) or {}
        prompt = cfg.get("prompt", _SPATIAL_PROMPT)
        max_new_tokens = int(cfg.get("max_new_tokens", _SPATIAL_MAX_NEW_TOKENS))

        loop = asyncio.get_running_loop()
        engine = _SpatialBotEngine.get()

        def _caption_all() -> dict[str, str]:
            out: dict[str, str] = {}
            for key, v in views.items():
                key = str(key)
                if not isinstance(v, dict) or not v.get("rgb_base64"):
                    out[key] = ""
                    continue
                rgb = _decode_rgb(v["rgb_base64"])
                out[key] = engine.generate(rgb, _view_depth(v), prompt, max_new_tokens)
            return out

        captions = await loop.run_in_executor(None, _caption_all)
        self._self_log("num_views", len(captions))
        return {"captions": captions}


class GenerateTool(BaseCanvasNode):
    """Single-view SpatialBot generate — generic (rgb, depth?, prompt) → text."""

    node_type: ClassVar[str] = "vlm_spatialbot__generate"
    display_name: ClassVar[str] = "SpatialBot: Generate"
    description: ClassVar[str] = "Depth-aware VLM generate on one rgb(+depth) view"
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "MessageSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("max_new_tokens", "text", label="max_new_tokens", default=200),
        ],
    )
    input_ports = [
        PortDef("rgb_b64", "TEXT", "Base64 RGB image"),
        PortDef("depth_b64", "TEXT", "Base64 normalised-u8 depth image (optional)", optional=True),
        PortDef("prompt", "TEXT", "User prompt"),
    ]
    output_ports = [PortDef("text", "TEXT", "Generated answer")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        rgb_b64 = inputs.get("rgb_b64")
        if not rgb_b64:
            return {"text": ""}
        prompt = inputs.get("prompt") or _SPATIAL_PROMPT
        cfg = getattr(self, "config", None) or {}
        max_new_tokens = int(cfg.get("max_new_tokens", _SPATIAL_MAX_NEW_TOKENS))

        depth_b64 = inputs.get("depth_b64")
        depth = _decode_depth_norm_u8(depth_b64) if depth_b64 else None

        loop = asyncio.get_running_loop()
        engine = _SpatialBotEngine.get()
        text = await loop.run_in_executor(
            None, engine.generate, _decode_rgb(rgb_b64), depth, prompt, max_new_tokens
        )
        self._self_log("text", text[:200])
        return {"text": text}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class SpatialBotNodeSet(BaseNodeSet):
    """SpatialBot-3B depth-aware VLM — server-mode FM nodeset."""

    name = "vlm_spatialbot"
    description = "SpatialBot-3B depth-aware VLM (caption/generate) — server-mode FM nodeset"
    # Stateless VLM — one shared server, K eval workers coalesce onto it.
    parallelism = "shared"
    # Default env: ac-ram (transformers 4.39.3 + torch 2.4.1) — in Bunny-Phi's
    # compatible range and the env that already hosted this exact forward
    # inside opennav_perception. Override with $SPATIALBOT_PYTHON.
    server_python = conda_env_python("ac-ram", "SPATIALBOT_PYTHON")

    def get_tools(self) -> list:
        return [CaptionViewsTool(), GenerateTool()]

    async def initialize(self, **kwargs: Any) -> None:
        engine = _SpatialBotEngine.get()
        if "spatialbot_path" in kwargs:
            engine.model_path = str(kwargs["spatialbot_path"])
        log.info("SpatialBotNodeSet ready (path=%s)", engine.model_path)

    async def shutdown(self) -> None:
        pass
