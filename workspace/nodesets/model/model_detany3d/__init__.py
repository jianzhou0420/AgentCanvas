"""ModelDetAny3DNodeSet — DetAny3D promptable 3D detection, server-mode.

Wraps DetAny3D (Zhai 2025; ToolEQA dependency) as a canvas server-mode
nodeset. DetAny3D is a **promptable** 3D detector — like SAM, its native
input is 2D box prompts — so the nodeset exposes exactly that native
capability (FM-boundary reshape 2026-07-05):

    model_detany3d__locate_3d   — image + boxes_2d prompts → 3D centers/
                                  sizes + rotation (DetAny3D WrapModel)

Text→box generation is a *composition* and lives outside: the upstream
demo bolts a GroundingDINO Swin-B in front, which callers now reach via
``model_grounding_dino__detect`` (variant native, Swin-B ckpt @
0.37/0.25 — DetAny3D's exact first stage, same env, same int rounding);
ToolEQA's toolbox does exactly that. The former ``locate_2d`` (pure
GroundingDINO — GDINO's capability, not DetAny3D's) and ``segment``
(GDINO+SAM union; zero consumers) nodes were removed in the same
reshape, dropping the standalone GroundingDINO and SAM ViT-H loads
(~4 GB resident) — the manager now holds only WrapModel. The GDINO
image *features* inside WrapModel are part of DetAny3D's architecture
and stay, of course.

The `tooleqa` method-side nodeset calls this via the standard
server-mode HTTP route (NOT via DetAny3D's posix_ipc IPC, which is
replaced here). Workspace-standalone: the DetAny3D source we depend on
is **copied into ``_vendor/``** under this folder; we never import from
``third_party/``. UniDepth etc. are pip-installed at env-create time
(`scripts/install/install_ac_detany3d.sh`).

Requirements:
  1. `$DETANY3D_PYTHON` pointing at the `ac-detany3d` conda env.
  2. Model weights under `data/detany3d/weights/` (or `DETANY3D_DATA_ROOT`):
       - DetAny3D checkpoint   (see `_vendor/UPSTREAM_README.md`)
       - SAM ViT-H             (`sam_vit_h_4b8939.pth` — WrapModel's
                                backbone init via cfg.model.checkpoint)
  3. DetAny3D demo config at `_vendor/detect_anything/configs/demo.yaml`.

last updated: 2026-07-05 (promptable reshape: locate_3d takes boxes_2d
prompts; locate_2d + segment removed; standalone GDINO/SAM loads dropped)
"""

from __future__ import annotations

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
# `__file__` lives at workspace/nodesets/env/model_detany3d/__init__.py
# → ../../../../ resolves to repo root.
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_DATA_ROOT = os.environ.get("DETANY3D_DATA_ROOT", os.path.join(_REPO_ROOT, "data", "detany3d"))
_WEIGHTS_DIR = os.path.join(_DATA_ROOT, "weights")


def _ensure_vendor_on_path() -> None:
    """Make ``_vendor/`` importable so ``train_utils`` / ``wrap_model`` /
    ``detect_anything`` resolve as top-level packages (matching upstream).

    GroundingDINO is pip-installed in the `detany3d` conda env; its config
    is resolved via the installed package, not from a local file.
    """
    if _VENDOR_ROOT not in sys.path:
        sys.path.insert(0, _VENDOR_ROOT)




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

            # The standalone GroundingDINO detector + SAM ViT-H predictor
            # loads are gone (2026-07-05 promptable reshape): text→box is a
            # composition served by model_grounding_dino; segmentation was
            # dead code. WrapModel is the whole resident surface.
            self._sam_trans = ResizeLongestSide(self._cfg.model.pad)

            self._initialized = True
            log.info("DetAny3DEnvManager initialized on cuda:%d", self._gpu_id)

    def shutdown(self) -> None:
        with self._lock:
            self._sam_model = None
            self._sam_trans = None
            self._cfg = None
            self._initialized = False

    # ── Inference helpers (verbatim port from app_mp.py predict_*) ──

    def predict_3d(self, image: np.ndarray, bbox_2d_list: list) -> dict[str, Any]:
        """Promptable 3D detection — port of app_mp.py:190-295, with the 2D
        box prompts supplied by the caller instead of an internal detector.

        ``bbox_2d_list`` is a list of integer ``[x1, y1, x2, y2]`` rows —
        exactly what the upstream GroundingDINO stage produced (int-rounded);
        ``model_grounding_dino__detect`` (native, Swin-B @ 0.37/0.25) emits
        the identical format. Returns ``{bboxes_3d, rot_mat}``: bboxes_3d is
        a list of ``[cx, cy, cz, sx, sy, sz, yaw]`` rows; centers are first
        3 cols, sizes next 3. Caller (ToolEQA's location_3d.py:63-66) rounds
        to 2 decimal places.
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

                if not bbox_2d_list:
                    return {"error": "no boxes_2d prompts provided"}

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
                }
            except Exception as e:
                import traceback

                traceback.print_exc()
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


class Locate3DTool(BaseCanvasNode):
    """Promptable 3D detection — image + 2D box prompts → 3D boxes.

    DetAny3D's native surface (like SAM, it consumes box prompts). Feed it
    boxes from ``model_grounding_dino__detect`` (or any box source); the
    text→box stage is a graph/caller-level composition since 2026-07-05.
    """

    node_type = "model_detany3d__locate_3d"
    display_name = "DetAny3D: Locate 3D"
    description = (
        "Promptable 3D detection — image + [x1,y1,x2,y2] box prompts → "
        "3D bboxes (centers + sizes) and rotation matrix."
    )
    category = "tool"
    icon = "Box"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("image", "ANY", "RGB image (np.ndarray) or path string"),
        PortDef("boxes_2d", "ANY", "List of [x1, y1, x2, y2] integer box prompts (or JSON string)"),
    ]
    output_ports = [
        PortDef("centers_3d", "ANY", "List of [cx, cy, cz] per prompted box"),
        PortDef("sizes_3d", "ANY", "List of [sx, sy, sz] per prompted box"),
        PortDef("rot_mat", "ANY", "Rotation matrix (3x3 list) per prompted box"),
        PortDef("error", "TEXT", "Empty on success; error message otherwise"),
    ]

    @staticmethod
    def _coerce_boxes(val: Any) -> list:
        if val is None:
            return []
        if isinstance(val, str):
            import json

            s = val.strip()
            if not s:
                return []
            val = json.loads(s)
        boxes = [list(b) for b in np.asarray(val, dtype=int).reshape(-1, 4).tolist()]
        return boxes

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        try:
            image = _decode_image_input(inputs["image"])
            boxes_2d = self._coerce_boxes(inputs.get("boxes_2d"))
        except Exception as e:
            return {"centers_3d": [], "sizes_3d": [], "rot_mat": [], "error": str(e)}
        result = await _run_sync(_get_mgr().predict_3d, image, boxes_2d)
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


# ══════════════════════════════════════════════════════════════════════
# Env panel
# ══════════════════════════════════════════════════════════════════════


class DetAny3DEnvPanel(BaseEnvPanel):
    """Minimal canvas panel env panel — initialization status + GPU id."""

    name = "model_detany3d"
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


class ModelDetAny3DNodeSet(BaseNodeSet):
    """DetAny3D 3D detection as a server-mode NodeSet.

    Loads in server mode against the `detany3d` conda env by default
    (Python 3.9 + torch + flash-attn + GroundingDINO + UniDepth + SAM
    + DetAny3D model weights).
    """

    name = "model_detany3d"
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
        return [Locate3DTool()]

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
