"""GroundingDINO open-vocabulary text->box detector — server-mode FM nodeset.

A thin, generic perception primitive: given a base64 RGB image and a text
prompt (e.g. ``"ground"``), return the detected bounding boxes. Built for the
AO-Planner port's Visual Affordances Prompting layer — AO-Planner uses
``GroundingDINO('ground', box_threshold=0.4, text_threshold=0.4)`` to detect
the navigable floor, then feeds those boxes to SAM (``model_sam__segment_box``)
for the ground mask (DiscussNav-style: detector → segmenter). Generic enough
that any other open-vocab-detection method can reuse it unchanged, which is why
it lives in ``model/`` (TODO #56 method/foundation-model boundary), like
``model_sam`` / ``model_ram``.

Primary tool::

    model_grounding_dino__detect  (image_b64: TEXT, [text_prompt: TEXT])
        -> result: TEXT  (JSON {boxes:[{xyxy,score,phrase}], count, image_w, image_h, text_prompt})

Variant matrix (FM-template alignment 2026-07-05 — model identity is node
config, engines in lazy registries so several checkpoints co-host):

    variant   engine                                   env
    native    groundingdino-py + local .pth ckpt       ac-detany3d (compiled ops;
              (Swin-T OGC default; Swin-B cogcoor       torch 2.1.2, transformers
              via ckpt — config file inferred from      4.39 has NO grounding_dino)
              the ckpt filename)
    hf_tiny   HF transformers zero-shot detection      ac-fm (transformers ≥4.40)
              (IDEA-Research/grounding-dino-tiny)

The two variants genuinely need different interpreter envs, and one server is
one env — so the *server env* stays a deployment choice
(``$GROUNDING_DINO_BACKEND`` picks which env boots and the variant default
follows it; ``$GROUNDING_DINO_PYTHON`` overrides the interpreter), while the
*variant* is per-node config. Asking a server for a variant its env cannot run
**raises a clear node error** (honest failure beats a silently ignored knob).

Inference recipes are verbatim ports: native from ``model_detany3d/
__init__.py:287-331`` (DetAny3D ``app_mp.py:94-105, 171-187`` — RandomResize
800 / ImageNet norm, ``groundingdino.util.inference.predict``, cxcywh→xyxy);
hf_tiny from the retired navgpt ``OpenVocabDetectNode``. Thresholds default to
AO-Planner's 0.4 / 0.4, caption ``"ground"`` (C-2 fidelity alignment
2026-06-17). Both variants emit the same ``result`` JSON schema, so graphs are
variant-agnostic; NavGPT-style post-processing lives in the pure
``navgpt_mp3d_tools__format_detections`` node.

TRANSITIONAL second tool ``model_grounding_dino__ground_mask`` (GroundingDINO
boxes → embedded SAM ViT-H → union mask): a cross-model composition scheduled
for graph-level replacement (detect → ``model_sam__segment_box`` → union);
kept byte-identical until ``aoplanner_ce`` migrates, then deleted.

Runs **server mode** (own subprocess + CUDA context) so the parent eval holds
no GroundingDINO VRAM and worker pools coalesce onto one shared server.

Load:  POST /api/components/nodesets/model_grounding_dino/load

last updated: 2026-07-05
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import json
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

log = logging.getLogger("agentcanvas.model_grounding_dino")


def _find_repo_root() -> str:
    """Walk upward from this file until a dir containing ``data/`` is found
    (overlay-safe — see the identical helper in ``model_ram.py``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 7):
        cand = os.path.normpath(os.path.join(here, *[".."] * up))
        if os.path.isdir(os.path.join(cand, "data")):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), "..", ".."))


_REPO_ROOT = _find_repo_root()
# ── Default weights (reuse the detany3d data dir); config overrides per node ─
_DEFAULT_WEIGHTS = os.environ.get(
    "GROUNDING_DINO_WEIGHTS",
    os.path.join(_REPO_ROOT, "data", "detany3d", "weights", "groundingdino_swint_ogc.pth"),
)
_DEFAULT_HF_MODEL = os.environ.get("GROUNDING_DINO_HF_MODEL", "IDEA-Research/grounding-dino-tiny")
# SAM ViT-H (AO-Planner's exact SAM variant) — used only by the transitional
# ground_mask tool; dies with it.
_SAM_VIT_H_WEIGHTS = os.environ.get(
    "GROUNDING_DINO_SAM_WEIGHTS",
    os.path.join(_REPO_ROOT, "data", "detany3d", "weights", "sam_vit_h_4b8939.pth"),
)

# AO-Planner defaults (llm/run_grounded_sam.sh: box/text_threshold 0.4).
_DEFAULT_TEXT_PROMPT = "ground"
_DEFAULT_BOX_THRESHOLD = 0.4
_DEFAULT_TEXT_THRESHOLD = 0.4

# Deployment default: which env boots this server (and the variant default).
_BACKEND = os.environ.get("GROUNDING_DINO_BACKEND", "native")

# Single-thread executor: GPU affinity + single-flight inference across all
# engines in this server (mirrors DetAny3DEnvManager's executor pattern).
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gdino")

# Transitional ground_mask SAM (module-global; removed with the tool).
_sam_predictor = None
_sam_lock = threading.Lock()


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _anchor_ckpt(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def _infer_native_config(ckpt_path: str) -> str:
    """Locate the GroundingDINO config in the pip-installed package, inferring
    the backbone from the checkpoint filename (``swinb`` → SwinB cfg, else
    Swin-T OGC). ``$GROUNDING_DINO_CONFIG`` overrides the basename."""
    base = os.path.basename(ckpt_path).lower()
    inferred = "GroundingDINO_SwinB_cfg.py" if "swinb" in base else "GroundingDINO_SwinT_OGC.py"
    cfg_name = os.environ.get("GROUNDING_DINO_CONFIG", inferred)
    try:
        import groundingdino  # type: ignore[import-not-found]

        gd_pkg = os.path.dirname(os.path.abspath(groundingdino.__file__))
        candidate = os.path.join(gd_pkg, "config", cfg_name)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError(
        f"{cfg_name} not found in installed groundingdino package. "
        "Run scripts/install/install_ac_detany3d.sh to install groundingdino-py, or set "
        "$GROUNDING_DINO_PYTHON to an env that has it."
    )


# ══════════════════════════════════════════════════════════════════════
# Engines — lazy registries per checkpoint / model id
# ══════════════════════════════════════════════════════════════════════


class _GDinoNativeEngine:
    """groundingdino-py engine; one resident model per checkpoint path.

    Swin-T OGC (AO-Planner) and Swin-B cogcoor (DetAny3D's 2D stage) co-host
    in one server. Raises a clear error if groundingdino-py is missing (wrong
    env for the ``native`` variant).
    """

    _instances: ClassVar[dict] = {}
    _registry_lock = threading.Lock()

    def __init__(self, ckpt: str) -> None:
        self.ckpt = ckpt
        self.model = None
        self.device = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    @classmethod
    def get(cls, ckpt: str = "") -> "_GDinoNativeEngine":
        resolved = _anchor_ckpt(ckpt or _DEFAULT_WEIGHTS)
        key = (resolved,)
        with cls._registry_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(resolved)
            return cls._instances[key]

    def ensure(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                import groundingdino  # noqa: F401
            except ImportError:
                raise ValueError(
                    "variant=native needs groundingdino-py (ac-detany3d env); this "
                    "server env lacks it — boot with GROUNDING_DINO_BACKEND=native "
                    "(default) or point $GROUNDING_DINO_PYTHON at ac-detany3d"
                )
            try:
                import torch
                from groundingdino.util.inference import load_model

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                log.info(
                    "Loading GroundingDINO on %s (ckpt=%s) …",
                    device, os.path.basename(self.ckpt),
                )
                model = load_model(_infer_native_config(self.ckpt), self.ckpt)
                model.to(device)
                model.eval()
                self.model, self.device = model, device
                self._loaded = True
                log.info("GroundingDINO loaded (%s)", device)
                return True
            except Exception as exc:
                log.warning("GroundingDINO load failed (%s): %s", self.ckpt, exc)
                self._load_failed = True
                return False

    def detect(self, b64: str, text: str, box_threshold: float, text_threshold: float) -> dict:
        """Native recipe from ``model_detany3d/__init__.py:287-331``
        (DetAny3D ``app_mp.py:94-105, 171-187``)."""
        import groundingdino.datasets.transforms as T  # type: ignore[import-not-found]
        import torch
        from groundingdino.util.inference import predict as dino_predict
        from PIL import Image
        from torchvision.ops import box_convert

        img = _decode_rgb(b64)  # HxWx3 uint8
        src = Image.fromarray(img, "RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_t, _ = transform(src, None)
        with torch.no_grad():
            # remove_combined omitted — not in the groundingdino-py 0.4.0 signature.
            boxes, logits, phrases = dino_predict(
                model=self.model,
                image=image_t,
                caption=text,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )
        h, w = img.shape[:2]
        out_boxes: list = []
        if len(boxes) > 0:
            xyxy = box_convert(boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy")
            for i in range(len(xyxy)):
                out_boxes.append(
                    {
                        "xyxy": xyxy[i].to(torch.int).cpu().numpy().tolist(),
                        "score": float(logits[i]),
                        "phrase": phrases[i],
                    }
                )
        return {
            "boxes": out_boxes,
            "count": len(out_boxes),
            "image_w": int(w),
            "image_h": int(h),
            "text_prompt": text,
        }


class _GDinoHfEngine:
    """HF transformers zero-shot-detection engine; one model per HF id.

    Verbatim from the retired navgpt ``_get_gdino``; requires
    transformers>=4.40 (the grounding_dino model family) — absent in
    ac-detany3d, present in ac-fm.
    """

    _instances: ClassVar[dict] = {}
    _registry_lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    @classmethod
    def get(cls, model_id: str = "") -> "_GDinoHfEngine":
        resolved = model_id or _DEFAULT_HF_MODEL
        key = (resolved,)
        with cls._registry_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(resolved)
            return cls._instances[key]

    def ensure(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                import transformers.models.grounding_dino  # noqa: F401
            except Exception:
                raise ValueError(
                    "variant=hf_tiny needs transformers>=4.40 (ac-fm env); this "
                    "server env lacks the grounding_dino model family — boot with "
                    "GROUNDING_DINO_BACKEND=hf_tiny or point "
                    "$GROUNDING_DINO_PYTHON at ac-fm"
                )
            try:
                import torch
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                log.info("Loading GroundingDINO (hf backend) %s on %s …", self.model_id, device)
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device)
                self.model.eval()
                self.device = device
                self._loaded = True
                log.info("GroundingDINO hf backend loaded (%s)", device)
                return True
            except Exception as exc:
                log.warning("GroundingDINO hf load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False

    def detect(self, b64: str, text: str, box_threshold: float, text_threshold: float) -> dict:
        import torch
        from PIL import Image

        img = _decode_rgb(b64)
        pil = Image.fromarray(img, "RGB")
        proc_inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**proc_inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            proc_inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        h, w = img.shape[:2]
        out_boxes: list = []
        for box, score, label in zip(
            results["boxes"].cpu().numpy(), results["scores"].cpu().numpy(), results["labels"]
        ):
            out_boxes.append(
                {
                    "xyxy": [int(c) for c in box],
                    "score": float(score),
                    "phrase": str(label).strip(),
                }
            )
        return {
            "boxes": out_boxes,
            "count": len(out_boxes),
            "image_w": int(w),
            "image_h": int(h),
            "text_prompt": text,
        }


def _engine_from_config(cfg: dict):
    variant = cfg.get("variant", _BACKEND)
    ckpt = str(cfg.get("ckpt", "") or "").strip()
    if variant == "hf_tiny":
        return _GDinoHfEngine.get(ckpt)
    if variant == "native":
        return _GDinoNativeEngine.get(ckpt)
    raise ValueError(f"unknown grounding_dino variant {variant!r} (native | hf_tiny)")


# ══════════════════════════════════════════════════════════════════════
# Transitional ground_mask internals (deleted with the tool)
# ══════════════════════════════════════════════════════════════════════


def _ensure_sam():
    """Lazy-load SAM ViT-H (the AO-Planner SAM variant) once per server."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    with _sam_lock:
        if _sam_predictor is not None:
            return _sam_predictor
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("Loading SAM ViT-H on %s (ckpt=%s) …", device, _SAM_VIT_H_WEIGHTS)
        sam = sam_model_registry["vit_h"](checkpoint=_SAM_VIT_H_WEIGHTS)
        sam.to(device)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
        log.info("SAM ViT-H loaded (%s)", device)
        return _sam_predictor


def _ground_mask(b64: str, text: str, box_threshold: float, text_threshold: float) -> dict:
    """GroundingDINO('ground') boxes -> SAM masks (per box) -> union ground mask.

    Faithful AO-Planner Grounded-SAM: detect ground boxes, segment each with SAM
    ViT-H (multimask_output=False), union the masks. Returns a base64 PNG mask.
    Always the native detector (its historical serving env).
    """
    import numpy as np
    from PIL import Image

    engine = _GDinoNativeEngine.get("")
    if not engine.ensure():
        return {"mask_b64": "", "n_boxes": 0, "image_w": 0, "image_h": 0}
    det = engine.detect(b64, text, box_threshold, text_threshold)
    boxes = det.get("boxes") or []
    img = _decode_rgb(b64)
    h, w = img.shape[:2]
    if not boxes:
        return {"mask_b64": "", "n_boxes": 0, "image_w": int(w), "image_h": int(h)}
    predictor = _ensure_sam()
    predictor.set_image(img)
    union = None
    for b in boxes:
        xy = b.get("xyxy")
        if not xy or len(xy) != 4:
            continue
        masks, _scores, _ = predictor.predict(box=np.asarray(xy, dtype=float), multimask_output=False)
        m = np.asarray(masks[0]).astype(bool)
        union = m if union is None else (union | m)
    if union is None:
        return {"mask_b64": "", "n_boxes": 0, "image_w": int(w), "image_h": int(h)}
    buf = io.BytesIO()
    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(buf, format="PNG")
    return {
        "mask_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "n_boxes": len(boxes),
        "image_w": int(w),
        "image_h": int(h),
    }


# ══════════════════════════════════════════════════════════════════════
# Canvas nodes
# ══════════════════════════════════════════════════════════════════════


class GroundingDinoDetectTool(BaseCanvasNode):
    """Open-vocabulary text→box detection with GroundingDINO.

    Returns every box matching ``text_prompt`` (default ``"ground"``) as pixel
    ``xyxy`` + score + matched phrase, aligned to the input image's resolution.
    """

    node_type: ClassVar[str] = "model_grounding_dino__detect"
    display_name: ClassVar[str] = "GroundingDINO: Detect (open-vocab)"
    description: ClassVar[str] = (
        "Open-vocabulary text→box detection; returns boxes matching a text prompt (e.g. 'ground')"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "ScanSearch"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "variant", "select", "Engine variant (must match the server env)",
                options=[
                    {"value": "native", "label": "native (groundingdino-py, local .pth)"},
                    {"value": "hf_tiny", "label": "hf_tiny (HF transformers)"},
                ],
                default=_BACKEND,
            ),
            ConfigField(
                "ckpt", "text",
                "Checkpoint override (native: .pth path — swinb filename picks the "
                "Swin-B config; hf_tiny: HF repo id). Blank = variant default",
                default="",
            ),
            ConfigField(
                "text_prompt", "text", "Object/region to detect (open-vocab)",
                default=_DEFAULT_TEXT_PROMPT,
            ),
            ConfigField(
                "box_threshold", "text", "Box confidence threshold (AO-Planner 0.4)",
                default=str(_DEFAULT_BOX_THRESHOLD),
            ),
            ConfigField(
                "text_threshold", "text", "Text-match threshold (AO-Planner 0.4)",
                default=str(_DEFAULT_TEXT_THRESHOLD),
            ),
        ],
    )
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64 PNG/JPEG RGB image"),
        PortDef(
            "text_prompt", "TEXT", "Optional: override the configured text prompt",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "result", "TEXT",
            "JSON {boxes:[{xyxy:[x1,y1,x2,y2], score, phrase}], count, image_w, image_h, text_prompt}",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        b64 = inputs.get("image_b64") or ""
        if not b64:
            return {"result": json.dumps({"boxes": [], "count": 0, "error": "no image_b64"})}

        config = getattr(self, "config", None) or {}
        text = (inputs.get("text_prompt") or config.get("text_prompt") or _DEFAULT_TEXT_PROMPT).strip()
        box_threshold = float(config.get("box_threshold", _DEFAULT_BOX_THRESHOLD) or _DEFAULT_BOX_THRESHOLD)
        text_threshold = float(config.get("text_threshold", _DEFAULT_TEXT_THRESHOLD) or _DEFAULT_TEXT_THRESHOLD)
        engine = _engine_from_config(config)

        def _run() -> "dict | None":
            if not engine.ensure():
                return None
            return engine.detect(b64, text, box_threshold, text_threshold)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, _run)
        if result is None:
            self._self_log("degraded", "GroundingDINO engine failed to load")
            return {"result": ""}
        self._self_log("n_boxes", result.get("count", 0))
        self._self_log("text_prompt", text)
        return {"result": json.dumps(result)}


class GroundingDinoGroundMaskTool(BaseCanvasNode):
    """GroundingDINO('ground') + SAM ViT-H → union navigable-ground mask.

    TRANSITIONAL (cross-model composition): scheduled for graph-level
    replacement by ``__detect`` → ``model_sam__segment_box`` → union in
    ``aoplanner_ce``; kept byte-identical until that migration lands.
    Output ``mask_b64`` feeds ``aoplanner__sample_waypoints``.
    """

    node_type: ClassVar[str] = "model_grounding_dino__ground_mask"
    display_name: ClassVar[str] = "Grounded-SAM: Ground Mask"
    description: ClassVar[str] = "GroundingDINO('ground') + SAM ViT-H → union navigable-ground mask (base64 PNG)"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Layers"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("text_prompt", "text", "Ground prompt (AO-Planner 'ground')", default="ground"),
            ConfigField("box_threshold", "text", "Box confidence threshold (AO-Planner 0.4)", default="0.4"),
            ConfigField("text_threshold", "text", "Text-match threshold (AO-Planner 0.4)", default="0.4"),
        ],
    )
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64 PNG/JPEG RGB image"),
    ]
    output_ports = [
        PortDef("mask_b64", "TEXT", "Base64 PNG of the union ground mask (empty if no ground)"),
        PortDef("n_boxes", "ANY", "Number of ground boxes segmented"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        b64 = inputs.get("image_b64") or ""
        if not b64:
            return {"mask_b64": "", "n_boxes": 0}
        config = getattr(self, "config", None) or {}
        text = (config.get("text_prompt") or _DEFAULT_TEXT_PROMPT).strip()
        box_threshold = float(config.get("box_threshold", _DEFAULT_BOX_THRESHOLD) or _DEFAULT_BOX_THRESHOLD)
        text_threshold = float(config.get("text_threshold", _DEFAULT_TEXT_THRESHOLD) or _DEFAULT_TEXT_THRESHOLD)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, _ground_mask, b64, text, box_threshold, text_threshold)
        self._self_log("n_boxes", result.get("n_boxes", 0))
        return {"mask_b64": result["mask_b64"], "n_boxes": result.get("n_boxes", 0)}


class GroundingDinoNodeSet(BaseNodeSet):
    """GroundingDINO open-vocab detector — server-mode FM nodeset."""

    name = "model_grounding_dino"
    description = "GroundingDINO open-vocabulary text→box detector — server-mode FM nodeset"
    # Stateless detector — one shared server, K eval workers coalesce onto it.
    parallelism = "shared"
    # Env follows the deployment backend default: native needs ac-detany3d
    # (compiled groundingdino-py 0.4.0, frozen); hf_tiny lives in the shared
    # ac-fm env (transformers ≥4.40 — where the grounding_dino model family
    # exists; parity gate 2026-07-05). $GROUNDING_DINO_PYTHON overrides both.
    server_python = conda_env_python(
        "ac-fm" if _BACKEND == "hf_tiny" else "ac-detany3d",
        "GROUNDING_DINO_PYTHON",
    )

    def get_tools(self) -> list:
        return [GroundingDinoDetectTool(), GroundingDinoGroundMaskTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("GroundingDinoNodeSet ready (server_python=%s)", self.server_python)

    async def shutdown(self) -> None:
        global _sam_predictor
        _GDinoNativeEngine._instances.clear()
        _GDinoHfEngine._instances.clear()
        _sam_predictor = None
