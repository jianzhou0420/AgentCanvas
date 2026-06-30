from __future__ import annotations

"""EnvDetAny3DNodeSet — DetAny3D 3D detection as a server-mode NodeSet.

Wraps DetAny3D (Zhai 2025; ToolEQA dependency) as a canvas server-mode
nodeset. Exposes 2D detection, 3D detection, and SAM-prompted
segmentation as canvas tool nodes — the `tooleqa` method-side nodeset
calls these via the standard server-mode HTTP route (NOT via DetAny3D's
posix_ipc IPC, which is replaced here).

Workspace-standalone: the DetAny3D source we depend on is **copied into
``_vendor/``** under this folder; we never import from
``third_party/``. GroundingDINO + UniDepth + SAM are pip-installed at
env-create time (`scripts/install/install_ac_detany3d.sh`).

Architecture — mirrors `hmeqa.py`:

1. `DetAny3DEnvManager` (singleton)
     Holds the loaded model triple — DetAny3D's WrapModel (3D head),
     GroundingDINO (open-vocab 2D detector), and SAM ViT-H predictor.
     Models load lazily on first call (~10 GB of weights). Pinned to
     a single ThreadPoolExecutor for GPU thread affinity.

2. Canvas tool nodes (`BaseCanvasNode` adapters)
     env_detany3d__locate_2d   — text → 2D bboxes via GroundingDINO
     env_detany3d__locate_3d   — text → 3D centers + sizes (DetAny3D)
     env_detany3d__segment     — text → masks (SAM prompted by DINO bboxes)

3. `EnvDetAny3DNodeSet` (collection + lifecycle)
     server_python defaults to `$DETANY3D_PYTHON` so the framework
     auto-hosts this in the dedicated `detany3d` conda env subprocess.

Why we re-load models in this nodeset rather than IPC into upstream's
`app_mp.py`:

  Upstream uses POSIX shared memory (`utils/shared_memory.py`) to ferry
  images + prompts across process boundaries. AgentCanvas already has
  HTTP server-mode (auto_host) for cross-subprocess calls. Replacing
  posix_ipc with our HTTP route keeps one IPC mechanism. Model load is
  identical to upstream's `app_mp.py:init_models`.

Status: SCAFFOLD — the model-loading + predict bodies mirror upstream
verbatim but require:
  1. `$DETANY3D_PYTHON` pointing at the `detany3d` conda env
     (see `scripts/install/install_ac_detany3d.sh`).
  2. Model weights under `data/detany3d/weights/` (or `DETANY3D_DATA_ROOT`):
       - GroundingDINO Swin-B  (`groundingdino_swinb_cogcoor.pth`)
       - SAM ViT-H             (`sam_vit_h_4b8939.pth`)
       - DetAny3D checkpoint   (see `_vendor/UPSTREAM_README.md`)
  3. DetAny3D demo config at `_vendor/detect_anything/configs/demo.yaml`
     (shipped in the vendored copy; verify `cfg.resume` +
     `cfg.model.checkpoint` resolve correctly — adjust paths to point at
     `data/detany3d/weights/...` in this conda env).

last updated: 2026-05-10
"""

import asyncio
import concurrent.futures
import logging
import os
import sys
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
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)

log = logging.getLogger("agentcanvas.detany3d")


# ══════════════════════════════════════════════════════════════════════
# Local-vendor path setup (workspace-standalone — no third_party reference)
# ══════════════════════════════════════════════════════════════════════

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_ROOT = os.path.join(_THIS_DIR, "_vendor")
_DETANY3D_CONFIG = os.path.join(_VENDOR_ROOT, "detect_anything", "configs", "demo.yaml")

# Project root for resolving the data dir (e.g. `data/detany3d/weights/`).
# `__file__` lives at workspace/nodesets/env/env_detany3d/__init__.py
# → ../../../../ resolves to repo root.
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_DATA_ROOT = os.environ.get("DETANY3D_DATA_ROOT", os.path.join(_REPO_ROOT, "data", "detany3d"))
_WEIGHTS_DIR = os.path.join(_DATA_ROOT, "weights")
_GROUNDINGDINO_WEIGHTS = os.path.join(_WEIGHTS_DIR, "groundingdino_swinb_cogcoor.pth")


def _ensure_vendor_on_path() -> None:
    """Make ``_vendor/`` importable so ``train_utils`` / ``wrap_model`` /
    ``detect_anything`` resolve as top-level packages (matching upstream).

    GroundingDINO is pip-installed in the `detany3d` conda env; its config
    is resolved via the installed package, not from a local file.
    """
    if _VENDOR_ROOT not in sys.path:
        sys.path.insert(0, _VENDOR_ROOT)


def _resolve_groundingdino_config() -> str:
    """Locate ``GroundingDINO_SwinB_cfg.py`` in the pip-installed groundingdino package."""
    try:
        import groundingdino  # type: ignore[import-not-found]

        gd_pkg = os.path.dirname(os.path.abspath(groundingdino.__file__))
        candidate = os.path.join(gd_pkg, "config", "GroundingDINO_SwinB_cfg.py")
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError(
        "GroundingDINO_SwinB_cfg.py not found in installed groundingdino package. "
        "Run scripts/install/install_ac_detany3d.sh to install groundingdino-py."
    )


# ══════════════════════════════════════════════════════════════════════
# Defaults — verbatim from DetAny3D/app_mp.py:75-76
# ══════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "box_threshold": 0.37,
    "text_threshold": 0.25,
}


# ══════════════════════════════════════════════════════════════════════
# DetAny3DEnvManager — singleton model holder
# ══════════════════════════════════════════════════════════════════════


class DetAny3DEnvManager:
    """Singleton manager for DetAny3D + GroundingDINO + SAM models.

    Lazy-loads on first `initialize()`. Holds all three model handles +
    the SAM transform helper. Single-thread executor gives GPU affinity.
    """

    _instance: DetAny3DEnvManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="detany3d",
        )
        self._gpu_id: int = 0
        self._cfg: Any = None  # OmegaConf / Box object

        # Models — populated by initialize()
        self._sam_model: Any = None  # DetAny3D's WrapModel
        self._dino_model: Any = None  # GroundingDINO
        self._sam_predictor: Any = None  # segment-anything SamPredictor
        self._sam_trans: Any = None  # ResizeLongestSide

        self._initialized: bool = False

    @classmethod
    def get(cls) -> DetAny3DEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── Lifecycle ──

    def initialize(self, gpu_id: int = 0) -> None:
        """Load DetAny3D + DINO + SAM models. Idempotent.

        Direct port of DetAny3D/app_mp.py:36-78 (`init_models`).
        """
        with self._lock:
            if self._initialized:
                return

            _ensure_vendor_on_path()

            import torch
            import yaml
            from box import Box  # python-box, used by DetAny3D
            from groundingdino.util.inference import load_model as load_dino
            from segment_anything import SamPredictor, sam_model_registry
            from train_utils import ResizeLongestSide  # type: ignore[import-not-found]
            from wrap_model import WrapModel  # type: ignore[import-not-found]

            self._gpu_id = int(gpu_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_id)
            torch.cuda.set_device(self._gpu_id)

            # Disable distributed init (upstream app_mp.py:46-48 — single-process model load).
            torch.distributed.is_available = lambda: False  # type: ignore[assignment]
            torch.distributed.is_initialized = lambda: False  # type: ignore[assignment]
            torch.distributed.get_world_size = lambda group=None: 1  # type: ignore[assignment]
            torch.distributed.get_rank = lambda group=None: 0  # type: ignore[assignment]

            # Load DetAny3D config
            with open(_DETANY3D_CONFIG, encoding="utf-8") as f:
                cfg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

            # Upstream demo.yaml uses CWD-relative paths like
            # "./checkpoints/detany3d_ckpts/other_exp_ckpt.pth". We resolve
            # those to absolute paths under _WEIGHTS_DIR so the server
            # works regardless of CWD.
            def _abs_ckpt(p: str) -> str:
                if isinstance(p, str) and p.startswith("./checkpoints/"):
                    return os.path.join(_WEIGHTS_DIR, p[2:])
                return p

            if "resume" in cfg_dict:
                cfg_dict["resume"] = _abs_ckpt(cfg_dict["resume"])
            if "dino_path" in cfg_dict:
                cfg_dict["dino_path"] = _abs_ckpt(cfg_dict["dino_path"])
                # If the local DINOv2 ckpt isn't present, set to "" so
                # _make_dinov2_model falls back to the fbaipublicfiles
                # CDN URL (see backbones/dinov2.py: pretrained == "" branch).
                if not os.path.isfile(cfg_dict["dino_path"]):
                    cfg_dict["dino_path"] = ""
            if isinstance(cfg_dict.get("model"), dict) and "checkpoint" in cfg_dict["model"]:
                cfg_dict["model"]["checkpoint"] = _abs_ckpt(cfg_dict["model"]["checkpoint"])
            if "unidepth_path" in cfg_dict:
                cfg_dict["unidepth_path"] = _abs_ckpt(cfg_dict["unidepth_path"])
            self._cfg = Box(cfg_dict)

            log.info("DetAny3D: loading WrapModel from %s", self._cfg.resume)
            sam_model = WrapModel(self._cfg)
            checkpoint = torch.load(self._cfg.resume, map_location=f"cuda:{self._gpu_id}")
            new_state = sam_model.state_dict()
            for k, v in new_state.items():
                if (
                    k in checkpoint["state_dict"]
                    and checkpoint["state_dict"][k].size() == new_state[k].size()
                ):
                    new_state[k] = checkpoint["state_dict"][k].detach()
            sam_model.load_state_dict(new_state)
            sam_model.to(f"cuda:{self._gpu_id}")
            sam_model.setup()
            sam_model.eval()
            self._sam_model = sam_model

            log.info("DetAny3D: loading GroundingDINO Swin-B")
            dino = load_dino(_resolve_groundingdino_config(), _GROUNDINGDINO_WEIGHTS)
            dino.to(f"cuda:{self._gpu_id}")
            dino.eval()
            self._dino_model = dino

            log.info("DetAny3D: loading SAM ViT-H predictor")
            sam = sam_model_registry["vit_h"](checkpoint=self._cfg.model.checkpoint)
            sam.to(f"cuda:{self._gpu_id}")
            self._sam_predictor = SamPredictor(sam)

            self._sam_trans = ResizeLongestSide(self._cfg.model.pad)

            self._initialized = True
            log.info("DetAny3DEnvManager initialized on cuda:%d", self._gpu_id)

    def shutdown(self) -> None:
        with self._lock:
            self._sam_model = None
            self._dino_model = None
            self._sam_predictor = None
            self._sam_trans = None
            self._cfg = None
            self._initialized = False

    # ── Inference helpers (verbatim port from app_mp.py predict_*) ──

    def _convert_dino_image(self, img: np.ndarray) -> tuple[np.ndarray, Any]:
        """Mirror of app_mp.py:94-105 (`convert_image`)."""
        import groundingdino.datasets.transforms as T  # type: ignore[import-not-found]
        from PIL import Image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(img, "RGB")
        image_np = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image_np, image_transformed

    def _dino_predict(self, img: np.ndarray, text: str) -> tuple[list, list]:
        """Run GroundingDINO and return ``(bboxes_xyxy, labels)``.

        Verbatim from app_mp.py:171-187 (the open-vocab 2D detection
        branch shared by all three predict_* functions).
        """
        import torch
        from groundingdino.util.inference import predict as dino_predict
        from torchvision.ops import box_convert

        image_source_dino, image_dino = self._convert_dino_image(img)
        boxes, _logits, phrases = dino_predict(
            model=self._dino_model,
            image=image_dino,
            caption=text,
            box_threshold=_DEFAULTS["box_threshold"],
            text_threshold=_DEFAULTS["text_threshold"],
            remove_combined=False,
        )
        h, w, _ = image_source_dino.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_2d_list: list = []
        label_list: list = []
        for i, box in enumerate(xyxy):
            bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
            label_list.append(phrases[i])
        return bbox_2d_list, label_list

    def predict_2d(self, image: np.ndarray, text: str) -> dict[str, Any]:
        """Open-vocab 2D detection — port of app_mp.py:161-188."""
        with self._lock:
            if not self._initialized:
                return {"error": "DetAny3D not initialized — call initialize() first"}
            try:
                bbox_2d_list, label_list = self._dino_predict(image, text)
            except Exception as e:
                return {"error": str(e)}
            return {"bboxes_2d": bbox_2d_list, "labels": label_list, "text": text}

    def predict_3d(self, image: np.ndarray, text: str) -> dict[str, Any]:
        """3D detection — port of app_mp.py:190-295.

        Returns ``{bboxes_3d, rot_mat, text}``. Bboxes_3d is a list of
        ``[cx, cy, cz, sx, sy, sz, yaw]`` rows; centers are first 3 cols,
        sizes are next 3 cols. Caller (ToolEQA's location_3d.py:63-66)
        rounds to 2 decimal places.
        """
        with self._lock:
            if not self._initialized:
                return {"error": "DetAny3D not initialized — call initialize() first"}

            try:
                import torch
                import torch.nn.functional as F
                from train_utils import (  # type: ignore[import-not-found]
                    decode_bboxes,
                    rotation_6d_to_matrix,
                )

                bbox_2d_list, _ = self._dino_predict(image, text)
                if not bbox_2d_list:
                    return {"error": f"No objects matching '{text}' found in the image."}

                # Pre-process for SAM (app_mp.py:248-289)
                original_size = tuple(image.shape[:-1])
                img_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
                img_t = img_t.unsqueeze(0)
                img_t = self._sam_trans.apply_image_torch(img_t)
                img_t = self._crop_hw(img_t)
                before_pad_size = tuple(img_t.shape[2:])
                img_for_sam = self._preprocess_sam(img_t).to(f"cuda:{self._gpu_id}")
                img_for_dino = self._preprocess_dino(img_t).to(f"cuda:{self._gpu_id}")

                if self._cfg.model.vit_pad_mask:
                    vit_pad_size = (
                        before_pad_size[0] // self._cfg.model.image_encoder.patch_size,
                        before_pad_size[1] // self._cfg.model.image_encoder.patch_size,
                    )
                else:
                    vit_pad_size = (
                        self._cfg.model.pad // self._cfg.model.image_encoder.patch_size,
                        self._cfg.model.pad // self._cfg.model.image_encoder.patch_size,
                    )

                bbox_t = torch.tensor(bbox_2d_list)
                bbox_t = (
                    self._sam_trans.apply_boxes_torch(bbox_t, original_size)
                    .to(torch.int)
                    .to(f"cuda:{self._gpu_id}")
                )

                input_dict = {
                    "images": img_for_sam,
                    "vit_pad_size": torch.tensor(vit_pad_size)
                    .to(f"cuda:{self._gpu_id}")
                    .unsqueeze(0),
                    "images_shape": torch.Tensor(before_pad_size)
                    .to(f"cuda:{self._gpu_id}")
                    .unsqueeze(0),
                    "image_for_dino": img_for_dino,
                    "boxes_coords": bbox_t,
                }

                with torch.no_grad():
                    ret = self._sam_model(input_dict)

                K = ret["pred_K"]
                _, decoded_3d = decode_bboxes(ret, self._cfg, K)
                rot_mat = rotation_6d_to_matrix(ret["pred_pose_6d"])

                return {
                    "bboxes_3d": decoded_3d.cpu().tolist()
                    if hasattr(decoded_3d, "cpu")
                    else decoded_3d,
                    "rot_mat": rot_mat.cpu().tolist() if hasattr(rot_mat, "cpu") else rot_mat,
                    "text": text,
                }
            except Exception as e:
                import traceback

                traceback.print_exc()
                return {"error": str(e)}

    def predict_seg(self, image: np.ndarray, text: str) -> dict[str, Any]:
        """SAM-prompted segmentation — port of app_mp.py:297-337."""
        with self._lock:
            if not self._initialized:
                return {"error": "DetAny3D not initialized — call initialize() first"}
            try:
                bbox_2d_list, _ = self._dino_predict(image, text)
                if not bbox_2d_list:
                    return {"error": f"No objects matching '{text}' found."}

                image_source_dino, _ = self._convert_dino_image(image)
                self._sam_predictor.set_image(image_source_dino)
                masks_result: list = []
                for bbox in bbox_2d_list:
                    masks, _, _ = self._sam_predictor.predict(box=np.array(bbox))
                    mask = np.zeros_like(masks[0])
                    for i in range(masks.shape[0]):
                        mask = mask | masks[i]
                    masks_result.append((mask.astype(np.uint8) * 255).tolist())
                return {"masks": masks_result, "text": text}
            except Exception as e:
                return {"error": str(e)}

    # ── Internal preprocessing helpers (verbatim from app_mp.py) ──

    def _crop_hw(self, img: Any) -> Any:
        """Mirror of app_mp.py:107-123."""
        import torch

        if img.dim() == 4:
            img = img.squeeze(0)
        h, w = img.shape[1:3]
        assert max(h, w) % 112 == 0, "target_size must be divisible by 112"
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        center_h, center_w = h // 2, w // 2
        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2
        return img[:, start_h : start_h + new_h, start_w : start_w + new_w].unsqueeze(0)

    def _preprocess_sam(self, x: Any) -> Any:
        """Mirror of app_mp.py:125-135."""
        import torch
        import torch.nn.functional as F

        sam_pixel_mean = torch.Tensor(self._cfg.dataset.pixel_mean).view(-1, 1, 1)
        sam_pixel_std = torch.Tensor(self._cfg.dataset.pixel_std).view(-1, 1, 1)
        x = (x - sam_pixel_mean) / sam_pixel_std
        h, w = x.shape[-2:]
        padh = self._cfg.model.pad - h
        padw = self._cfg.model.pad - w
        return F.pad(x, (0, padw, 0, padh))

    def _preprocess_dino(self, x: Any) -> Any:
        """Mirror of app_mp.py:137-143."""
        import torch

        x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (x - mean) / std


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════


def _get_mgr() -> DetAny3DEnvManager:
    return DetAny3DEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    mgr = _get_mgr()
    return await asyncio.get_running_loop().run_in_executor(mgr.executor, fn, *args)


def _decode_image_input(value: Any) -> np.ndarray:
    """Accept np.ndarray (already decoded) or path string. Returns RGB ndarray."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, str):
        from PIL import Image

        if not os.path.exists(value):
            raise FileNotFoundError(f"image_path not found: {value}")
        return np.array(Image.open(value).convert("RGB"))
    raise TypeError(f"unsupported image input type: {type(value)}")


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes
# ══════════════════════════════════════════════════════════════════════


class Locate2DTool(BaseCanvasNode):
    node_type = "env_detany3d__locate_2d"
    display_name = "DetAny3D: Locate 2D"
    description = (
        "Open-vocab 2D detection via GroundingDINO. Input image + text prompt; output 2D bboxes."
    )
    category = "tool"
    icon = "Square"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("image", "ANY", "RGB image (np.ndarray) or path string"),
        PortDef("text", "TEXT", "Object name(s) to localize, dot-separated for multi-class"),
    ]
    output_ports = [
        PortDef("bboxes_2d", "ANY", "List of [x1, y1, x2, y2] integer bboxes"),
        PortDef("labels", "ANY", "List of label strings, 1:1 with bboxes_2d"),
        PortDef("error", "TEXT", "Empty on success; error message otherwise"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        try:
            image = _decode_image_input(inputs["image"])
        except Exception as e:
            return {"bboxes_2d": [], "labels": [], "error": str(e)}
        text = str(inputs.get("text", ""))
        result = await _run_sync(_get_mgr().predict_2d, image, text)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"bboxes_2d": [], "labels": [], "error": result["error"]}
        return {
            "bboxes_2d": result.get("bboxes_2d", []),
            "labels": result.get("labels", []),
            "error": "",
        }


class Locate3DTool(BaseCanvasNode):
    node_type = "env_detany3d__locate_3d"
    display_name = "DetAny3D: Locate 3D"
    description = "3D detection — returns 3D bboxes (centers + sizes) and rotation matrix."
    category = "tool"
    icon = "Box"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("image", "ANY", "RGB image (np.ndarray) or path string"),
        PortDef("text", "TEXT", "Object name to localize"),
    ]
    output_ports = [
        PortDef("centers_3d", "ANY", "List of [cx, cy, cz] per detected object"),
        PortDef("sizes_3d", "ANY", "List of [sx, sy, sz] per detected object"),
        PortDef("rot_mat", "ANY", "Rotation matrix (3x3 list) per detected object"),
        PortDef("error", "TEXT", "Empty on success; error message otherwise"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        try:
            image = _decode_image_input(inputs["image"])
        except Exception as e:
            return {"centers_3d": [], "sizes_3d": [], "rot_mat": [], "error": str(e)}
        text = str(inputs.get("text", ""))
        result = await _run_sync(_get_mgr().predict_3d, image, text)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"centers_3d": [], "sizes_3d": [], "rot_mat": [], "error": result["error"]}
        bboxes_3d = result.get("bboxes_3d", [])
        # Mirror location_3d.py:63-66 — first 3 cols = center, next 3 = size
        centers = [[round(b[0], 2), round(b[1], 2), round(b[2], 2)] for b in bboxes_3d]
        sizes = [[round(b[3], 2), round(b[4], 2), round(b[5], 2)] for b in bboxes_3d]
        return {
            "centers_3d": centers,
            "sizes_3d": sizes,
            "rot_mat": result.get("rot_mat", []),
            "error": "",
        }


class SegmentTool(BaseCanvasNode):
    node_type = "env_detany3d__segment"
    display_name = "DetAny3D: Segment"
    description = "GroundingDINO + SAM open-vocab instance segmentation."
    category = "tool"
    icon = "Scissors"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("image", "ANY", "RGB image (np.ndarray) or path string"),
        PortDef("text", "TEXT", "Object name(s) to segment, dot-separated for multi-class"),
    ]
    output_ports = [
        PortDef("masks", "ANY", "List of HxW uint8 masks (255 = object, 0 = background)"),
        PortDef("error", "TEXT", "Empty on success; error message otherwise"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        try:
            image = _decode_image_input(inputs["image"])
        except Exception as e:
            return {"masks": [], "error": str(e)}
        text = str(inputs.get("text", ""))
        result = await _run_sync(_get_mgr().predict_seg, image, text)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"masks": [], "error": result["error"]}
        return {"masks": result.get("masks", []), "error": ""}


# ══════════════════════════════════════════════════════════════════════
# Env panel
# ══════════════════════════════════════════════════════════════════════


class DetAny3DEnvPanel(BaseEnvPanel):
    """Minimal canvas panel env panel — initialization status + GPU id."""

    name = "env_detany3d"
    display_name = "DetAny3D"
    fields = [
        EnvPanelField("gpu_id", "select", "GPU"),
    ]
    actions = [
        EnvPanelAction("init", "Init Models", side_effect="none"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {"gpu_id": 0}

    def _mgr(self) -> DetAny3DEnvManager:
        return DetAny3DEnvManager.get()

    async def on_load(self) -> dict[str, Any]:
        mgr = self._mgr()
        return {
            "available": True,
            "gpu_id": int(self._state.get("gpu_id", 0)),
            "initialized": mgr.initialized,
            "vendor_root": _VENDOR_ROOT,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        if name == "gpu_id":
            try:
                self._state["gpu_id"] = int(value)
            except (TypeError, ValueError):
                self._state["gpu_id"] = 0
        else:
            self._state[name] = value
        return await self.on_load()

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name == "init":
            mgr = self._mgr()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                mgr.executor,
                lambda: mgr.initialize(gpu_id=int(self._state.get("gpu_id", 0))),
            )
            return {"ok": True, "side_effect": "none"}
        return {"ok": False, "side_effect": "none", "error": f"Unknown action {name!r}"}

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        if field == "gpu_id":
            try:
                import torch

                return [
                    {"value": i, "label": f"cuda:{i}"} for i in range(torch.cuda.device_count())
                ] or [{"value": 0, "label": "cuda:0 (no GPU detected)"}]
            except Exception:
                return [{"value": 0, "label": "cuda:0"}]
        return []


# ══════════════════════════════════════════════════════════════════════
# NodeSet binding
# ══════════════════════════════════════════════════════════════════════


class EnvDetAny3DNodeSet(BaseNodeSet):
    """DetAny3D 3D detection as a server-mode NodeSet.

    Loads in server mode against the `detany3d` conda env by default
    (Python 3.9 + torch + flash-attn + GroundingDINO + UniDepth + SAM
    + DetAny3D model weights).
    """

    name = "env_detany3d"
    description = "DetAny3D — open-vocab 2D + 3D detection + SAM segmentation"
    server_python = conda_env_python("ac-detany3d", "DETANY3D_PYTHON")
    env_panel = DetAny3DEnvPanel
    # DetAny3D is stateless inference (no per-worker session) — one server
    # serves all eval workers, which keeps the 12 GB model load singleton.
    # "shared" also makes the URL eligible for spec._shared_urls so the
    # eval subprocess can reach it. ADR-server-003.
    parallelism = "shared"
    default_per_step_budget_sec = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = DetAny3DEnvManager.get()

    def get_tools(self) -> list:
        return [
            Locate2DTool(),
            Locate3DTool(),
            SegmentTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        gpu_id = int(kwargs.get("gpu_id", 0))
        if self._mgr.initialized:
            log.info("DetAny3D already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(gpu_id=gpu_id),
        )
        log.info("EnvDetAny3DNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        return {
            "env_name": "detany3d",
            "datasets": [],
            "splits": [],
            "metrics": [],
            "supports_set_episode": False,
            "vendor_root": _VENDOR_ROOT,
        }

    async def shutdown(self) -> None:
        self._mgr.shutdown()
