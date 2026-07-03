from __future__ import annotations

"""Open-Nav waypoint predictor nodeset (server mode).

Wraps the frozen ``BinaryDistPredictor_TRM`` + ResNet50 RGB encoder +
DDPPO depth encoder used by Open-Nav (ICRA 2025) to score 120 angle bins
× 12 distance bins from a 12-view RGB-D panorama and emit ≤ 5 candidate
``(angle, distance)`` waypoints via NMS.

Source modules:

    ./_vendored/waypoint_prediction/TRM_net.py    BinaryDistPredictor_TRM
    ./_vendored/waypoint_prediction/utils.py      nms helper
    vlnce_baselines/models/encoders/...           ResNet50, DDPPO ResNet50
                                                  (from VLN-CE submodule)

Upstream: Open-Nav @ 3a8dcef — see
``workspace/nodesets/_upstream/open-nav/fetch_upstream.sh`` to re-fetch.

Runs in the ``vlnce`` conda env (Python 3.8 + habitat-sim 0.1.7 + torch 2.4)
because the encoder weights and habitat dependency live there. Loaded as
a server-mode nodeset; the agentcanvas backend talks to it over HTTP via
the standard ``AutoServerApp`` plumbing (ADR-009).

Two nodes:

    opennav_waypoint__predict           (rgb_views, depth_views) → candidates dict
    opennav_waypoint__bin_to_directions (heatmap)                → candidates dict

Currently only ``predict`` is exposed — ``bin_to_directions`` is fused
inside it because the angle-to-slot mapping is small and tightly coupled
to the heatmap shape.

Checkpoints (set via env vars or config fields):

    OPENNAV_WAYPOINT_CKPT  default data/opennav/check_val_best_avg_wayscore
    OPENNAV_DDPPO_CKPT     default data/opennav/ddppo-models/gibson-2plus-resnet50.pth

last updated: 2026-05-18
"""

import asyncio
import base64
import io
import logging
import os
import sys
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.opennav_waypoint")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# This file lives at workspace/nodesets/method/opennav_waypoint/__init__.py —
# repo root is 4 levels up.
_REPO_ROOT = os.path.normpath(os.path.join(_PKG_DIR, "..", "..", "..", ".."))

# Default points at the vendored ``_vendored/`` sub-dir which contains
# the ``waypoint_prediction/`` sub-tree (verbatim copy of upstream).
# Override OPENNAV_REPO_PATH to a real upstream clone for local edits.
_OPENNAV_REPO_DEFAULT = os.environ.get(
    "OPENNAV_REPO_PATH",
    os.path.join(_PKG_DIR, "_vendored"),
)
_WAYPOINT_CKPT_DEFAULT = os.environ.get(
    "OPENNAV_WAYPOINT_CKPT",
    os.path.join(_REPO_ROOT, "data", "opennav", "check_val_best_avg_wayscore"),
)


# ══════════════════════════════════════════════════════════════════════
# WaypointEngine — singleton model loader (server-side)
# ══════════════════════════════════════════════════════════════════════


class WaypointEngine:
    """Lazy loader for ``BinaryDistPredictor_TRM`` + image encoders.

    Loaded once per server subprocess. All heavy ML imports are deferred
    until the first ``predict`` call so import-time stays cheap and the
    nodeset is registerable without GPU/torch present.
    """

    _instance: WaypointEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.repo_path = _OPENNAV_REPO_DEFAULT
        self.ckpt_path = _WAYPOINT_CKPT_DEFAULT
        self.device = None
        self.predictor = None
        self.rgb_encoder = None
        self.rgb_transform = None
        self.depth_encoder = None
        self._loaded = False

    @classmethod
    def get(cls) -> WaypointEngine:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            log.info("Loading Open-Nav waypoint predictor from %s", self.repo_path)

            if self.repo_path not in sys.path:
                sys.path.insert(0, self.repo_path)

            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM  # type: ignore

            predictor = BinaryDistPredictor_TRM(
                hidden_dim=768, n_classes=12, device=self.device
            ).to(self.device)
            if os.path.exists(self.ckpt_path):
                state = torch.load(self.ckpt_path, map_location=self.device)
                # Unwrap the nested training-checkpoint structure. The shipped
                # ``check_val_best_avg_wayscore`` is
                # ``{"predictor": {"epoch", "state_dict", "optimizer"}}`` — the
                # real weights live at state["predictor"]["state_dict"]. Peel
                # both wrappers (also tolerates {"predictor": weights} /
                # {"state_dict": weights} / a bare state_dict).
                sd = state
                if isinstance(sd, dict) and "predictor" in sd:
                    sd = sd["predictor"]
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                predictor.load_state_dict(sd)
                log.info("Loaded waypoint predictor checkpoint")
            else:
                log.warning(
                    "Waypoint predictor checkpoint not found at %s — running uninitialised",
                    self.ckpt_path,
                )
            for p in predictor.parameters():
                p.requires_grad = False
            predictor.eval()
            self.predictor = predictor

            # ── RGB encoder — TorchVisionResNet50 (resnet_encoders.py:109-240)
            # torchvision resnet50 pretrained, truncated [:-2] → (B,2048,7,7);
            # preprocessing = ConvertImageDtype(float) + ImageNet Normalize
            # (resnet_encoders.py:180-184), applied in predict() exactly as
            # TorchVisionResNet50.forward does. The previous port fed /255
            # WITHOUT the ImageNet Normalize — fixed 2026-07-03.
            from torchvision import transforms  # type: ignore
            from torchvision.models import resnet50  # type: ignore

            try:
                from torchvision.models import ResNet50_Weights  # type: ignore

                rgb_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            except Exception:
                rgb_net = resnet50(pretrained=True)  # older torchvision
            rgb_net = torch.nn.Sequential(*list(rgb_net.children())[:-2]).to(self.device)
            rgb_net.eval()
            for p in rgb_net.parameters():
                p.requires_grad = False
            self.rgb_encoder = rgb_net
            self.rgb_transform = torch.nn.Sequential(
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ).to(self.device)

            # ── DDPPO ResNet50 depth encoder — VlnResnetDepthEncoder,
            # file-level import of the vendored upstream module (bypasses the
            # unimportable ``vlnce_baselines`` package — the old package-level
            # import raised ModuleNotFoundError, was swallowed, and every run
            # silently used ZERO depth features; found 2026-07-03). Load
            # failure is now FATAL: upstream cannot run without this encoder,
            # so neither may we.
            import importlib.util

            enc_path = os.path.join(
                _PKG_DIR, "_vendored", "vlnce_encoders", "resnet_encoders.py"
            )
            spec = importlib.util.spec_from_file_location("_opennav_resnet_encoders", enc_path)
            enc_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(enc_mod)  # type: ignore[union-attr]

            from gym import spaces  # type: ignore

            obs_space = spaces.Dict(
                {
                    "depth": spaces.Box(
                        low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32
                    ),
                }
            )
            # Ctor args mirror Policy_ViewSelection.py:97-103 + run_OpenNav.yaml
            # (output_size=256 — metadata only at spatial use; ckpt; resnet50;
            # spatial_output=False → forward returns raw (B,128,4,4)).
            depth_net = enc_mod.VlnResnetDepthEncoder(
                observation_space=obs_space,
                output_size=256,
                checkpoint=os.environ.get(
                    "OPENNAV_DDPPO_CKPT",
                    os.path.join(
                        _REPO_ROOT,
                        "data",
                        "opennav",
                        "ddppo-models",
                        "gibson-2plus-resnet50.pth",
                    ),
                ),
                backbone="resnet50",
                trainable=False,
                spatial_output=False,
            ).to(self.device)
            depth_net.eval()
            self.depth_encoder = depth_net
            log.info("DDPPO depth encoder loaded")

            self._loaded = True
            log.info("Waypoint engine ready (device=%s)", self.device)

    def predict(
        self, rgb_views: list[np.ndarray], depth_views: list[np.ndarray]
    ) -> dict[str, list[float]]:
        """Return ``{slot_id: [angle_rad, distance_m]}`` keyed by 30° slot.

        Faithful transcription of the upstream waypoint-mode forward
        (``Policy_ViewSelection.py:232-380`` @ Three-Step fork 5cdbdcf ==
        Open-Nav parent) + ``construct_image_dicts`` binning
        (``base_il_trainer_llm.py:202-259``), rewritten 2026-07-03:

        1. clockwise slot arrangement ``ra = (12 - a) % 12`` (:253-258) —
           the previous port fed the panorama counter-clockwise;
        2. RGB 1024→224 via ``F.interpolate(mode="area")`` + dtype cast —
           the fork's ACTIVE resize path (ResizerPerSensor,
           ``habitat_extensions/obs_transformers.py:160-162``; the bilinear
           shim at Policy_ViewSelection.py:259-268 is dead belt-and-braces);
        3. TorchVisionResNet50 preprocessing = ConvertImageDtype + ImageNet
           Normalize (was: bare /255);
        4. depth = absolute-normalised [0,1] (was: per-tile min-max);
        5. softmax over the flat heatmap + circular first/last-row wrap
           BEFORE nms(5, (7.0, 5.0)), then unwrap (:300-315) (was: raw
           logits, no wrap);
        6. candidates iterated in ascending-angle-index ``nonzero()`` order,
           slot collisions LAST-wins (dict overwrite in
           construct_image_dicts) (was: strength-sorted, first-wins).
        """
        self._ensure_loaded()

        import torch
        import torch.nn.functional as F

        if not rgb_views:
            return {}

        # ── clockwise arrangement (Policy_ViewSelection.py:243-268) ──
        # env views arrive counter-clockwise (dir i = i·30° left);
        # slot 0 keeps dir 0, slots 11..1 take dirs 1..11.
        n_views = min(12, len(rgb_views))
        rgb_cw: list = [None] * 12
        depth_cw: list = [None] * 12
        for a in range(n_views):
            ra = (12 - a) % 12
            rgb_cw[ra] = rgb_views[a]
            if a < len(depth_views):
                depth_cw[ra] = depth_views[a]
        rgb_fill = next(v for v in rgb_cw if v is not None)
        rgb_cw = [v if v is not None else np.zeros_like(rgb_fill) for v in rgb_cw]
        d_fill = next((v for v in depth_cw if v is not None), None)
        depth_cw = [
            v
            if v is not None
            else (np.zeros_like(d_fill) if d_fill is not None else np.zeros((256, 256), np.float32))
            for v in depth_cw
        ]

        with torch.no_grad():
            # ── RGB: (12,H,W,3) uint8 → ResizerPerSensor area-resize to 224
            # (obs_transformers.py:160-162) → TorchVisionResNet50.forward
            # (permute NHWC→NCHW, ConvertImageDtype+Normalize, resnet
            # :189-217) → (12,2048,7,7).
            rgb_batch = torch.from_numpy(np.stack(rgb_cw).astype(np.uint8)).to(self.device)
            x = rgb_batch.permute(0, 3, 1, 2)  # NCHW uint8
            if x.shape[-2:] != (224, 224):
                x = F.interpolate(x.float(), size=(224, 224), mode="area").to(dtype=x.dtype)
            rgb_feats = self.rgb_encoder(self.rgb_transform(x).contiguous())

            # ── Depth: (12,256,256,1) float32 in [0,1] (absolute habitat
            # normalisation) → VlnResnetDepthEncoder → (12,128,4,4).
            depth_np = np.stack(
                [np.asarray(d, dtype=np.float32).reshape(256, 256, 1) for d in depth_cw]
            )
            depth_tensor = torch.from_numpy(depth_np).to(self.device)
            depth_feats = self.depth_encoder({"depth": depth_tensor})

            heatmap = self.predictor(rgb_feats, depth_feats)  # (1, 120, 12)

            # ── softmax + circular wrap + NMS (Policy_ViewSelection.py:300-315)
            batch_x_norm = torch.softmax(
                heatmap.reshape(1, 120 * 12), dim=1
            ).reshape(1, 120, 12)
            batch_x_norm_wrap = torch.cat(
                (batch_x_norm[:, -1:, :], batch_x_norm, batch_x_norm[:, :1, :]), dim=1
            )

            from waypoint_prediction.utils import nms  # type: ignore

            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=5, sigma=(7.0, 5.0)
            )
            batch_output_map = batch_output_map.squeeze(1)[:, 1:-1, :]

        # ── candidates: ascending nonzero order + 2π− angle mapping
        # (Policy_ViewSelection.py:368-378) → construct_image_dicts binning
        # with LAST-wins dict assignment (base_il_trainer_llm.py:202-259).
        # Angle math in float32 TENSOR arithmetic exactly as upstream —
        # python float64 here drifts at the 7th decimal (equiv-test proven).
        import math

        nz = batch_output_map[0].nonzero(as_tuple=False)
        angle_idxes = nz[:, 0]
        distance_idxes = nz[:, 1]
        angle_rad_t = 2 * math.pi - angle_idxes.float() / 120 * 2 * math.pi
        dist_t = ((distance_idxes + 1) * 0.25).float()
        out: dict[str, list[float]] = {}
        for angle_rad, distance_m in zip(angle_rad_t.tolist(), dist_t.tolist()):
            out[_bin_angle_to_slot(angle_rad)] = [angle_rad, distance_m]
        return out


def _bin_angle_to_slot(angle_rad: float) -> str:
    """Bin an angle (radians) into one of 12 30° slot ids ``'0'..'11'``.

    Mirrors ``base_il_trainer_llm.py :: construct_image_dicts`` half-open-
    right buckets:

        0 < deg <=  30 → '1'     (30° left of heading)
        30 < deg <=  60 → '2'
        ...
        150 < deg <= 180 → '6'   (behind)
        180 < deg <= 210 → '7'
        ...
        300 < deg <= 330 → '11'  (30° right of heading)
        else (330..360 or 0)     → '0'  (current orientation)
    """
    angle_deg = float(np.rad2deg(angle_rad)) % 360.0
    if 0 < angle_deg <= 30:
        return "1"
    elif 30 < angle_deg <= 60:
        return "2"
    elif 60 < angle_deg <= 90:
        return "3"
    elif 90 < angle_deg <= 120:
        return "4"
    elif 120 < angle_deg <= 150:
        return "5"
    elif 150 < angle_deg <= 180:
        return "6"
    elif 180 < angle_deg <= 210:
        return "7"
    elif 210 < angle_deg <= 240:
        return "8"
    elif 240 < angle_deg <= 270:
        return "9"
    elif 270 < angle_deg <= 300:
        return "10"
    elif 300 < angle_deg <= 330:
        return "11"
    else:
        return "0"


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _decode_rgb_b64(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _decode_depth_b64(b64: str) -> np.ndarray:
    # LEGACY per-tile min-max 8-bit path — loses absolute depth. Kept only
    # as a fallback for old payloads without ``depth_raw_base64``.
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return np.asarray(img, dtype=np.float32) / 255.0


def _decode_depth_raw_b64(b64: str) -> np.ndarray:
    # 16-bit habitat-normalised depth ×1000 (env encode_depth_raw_base64) →
    # back to the absolute [0,1] float the DDPPO encoder was trained on
    # (habitat NORMALIZE_DEPTH over 0–10 m). ~1e-3 quantisation from the
    # uint16 roundtrip; upstream feeds the raw float straight through.
    from PIL import Image

    raw = base64.b64decode(b64)
    arr = np.asarray(Image.open(io.BytesIO(raw)), dtype=np.float32)
    return arr / 1000.0


# ══════════════════════════════════════════════════════════════════════
# Canvas tools
# ══════════════════════════════════════════════════════════════════════


class PredictWaypointsTool(BaseCanvasNode):
    """Predict candidate waypoints from a 12-view RGB-D panorama."""

    node_type: ClassVar[str] = "opennav_waypoint__predict"
    display_name: ClassVar[str] = "Open-Nav: Predict Waypoints"
    description: ClassVar[str] = "Frozen TRM_net heatmap → ≤5 (angle, distance) candidates"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Target"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64, depth_base64} from panorama_rgbd"),
    ]
    output_ports = [
        PortDef("candidates", "ANY", "{dir_id: [angle_rad, distance_m]}"),
        PortDef("num_candidates", "ANY", "Count of returned candidates"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        rgb_arrays: list[np.ndarray] = []
        depth_arrays: list[np.ndarray] = []
        for v in views:
            if not isinstance(v, dict):
                continue
            rgb_b64 = v.get("rgb_base64")
            depth_raw_b64 = v.get("depth_raw_base64")
            depth_b64 = v.get("depth_base64")
            if rgb_b64:
                rgb_arrays.append(_decode_rgb_b64(rgb_b64))
            # Absolute-normalised depth (upstream DDPPO contract) — the
            # legacy per-tile min-max 8-bit path is fallback only.
            if depth_raw_b64:
                depth_arrays.append(_decode_depth_raw_b64(depth_raw_b64))
            elif depth_b64:
                depth_arrays.append(_decode_depth_b64(depth_b64))

        if not rgb_arrays:
            self._self_log("predict_skipped", "no_rgb_views")
            return {"candidates": {}, "num_candidates": 0}

        loop = asyncio.get_running_loop()
        engine = WaypointEngine.get()
        candidates = await loop.run_in_executor(None, engine.predict, rgb_arrays, depth_arrays)
        self._self_log("num_candidates", len(candidates))
        return {"candidates": candidates, "num_candidates": len(candidates)}


class BinToDirectionsTool(BaseCanvasNode):
    """Pass-through helper: candidates are already keyed by dir_id from predict.

    Kept as a separate node for graph clarity — mirrors the
    ``construct_image_dicts`` step in the reference, which folds the
    heatmap argmax indices into 12 evenly-spaced direction slots.
    """

    node_type: ClassVar[str] = "opennav_waypoint__bin_to_directions"
    display_name: ClassVar[str] = "Open-Nav: Bin To Directions"
    description: ClassVar[str] = "Pass-through; candidates already keyed by direction id"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Grid"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [PortDef("candidates", "ANY", "{dir_id: [angle, distance]}")]
    output_ports = [
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]}"),
        PortDef("dir_ids", "ANY", "Sorted list of direction id strings"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        cands = inputs.get("candidates") or {}
        ids = sorted(cands.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        return {"candidates": cands, "dir_ids": ids}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class OpenNavWaypointNodeSet(BaseNodeSet):
    """Open-Nav frozen waypoint predictor."""

    name = "opennav_waypoint"
    description = "Frozen TRM_net waypoint predictor + image encoders (Open-Nav)"
    # The vendored TRM_net imports ``pytorch_transformers`` (legacy HF), which
    # the ``vlnce`` env lacks → import 500 (root cause of Open-Nav SR=0; the
    # predictor never loaded). The ``smartway`` env (Py 3.8 + torch 2.1.1 +
    # pytorch_transformers 1.2.0 + habitat-sim 0.1.7) is the purpose-built
    # VLN-CE waypoint env where the identical TRM already runs for
    # smartway_waypoint. Default there; override via OPENNAV_WAYPOINT_PYTHON.
    server_python = os.environ.get(
        "OPENNAV_WAYPOINT_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-smartway/bin/python")
    )

    def get_tools(self) -> list:
        return [PredictWaypointsTool(), BinToDirectionsTool()]

    async def initialize(self, **kwargs: Any) -> None:
        repo = kwargs.get("repo_path")
        ckpt = kwargs.get("ckpt_path")
        engine = WaypointEngine.get()
        if repo:
            engine.repo_path = str(repo)
        if ckpt:
            engine.ckpt_path = str(ckpt)
        log.info(
            "OpenNavWaypointNodeSet ready (repo=%s, ckpt=%s)",
            engine.repo_path,
            engine.ckpt_path,
        )

    async def shutdown(self) -> None:
        pass
