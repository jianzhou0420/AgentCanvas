"""WaypointEngine — DINOv2 + DDPPO depth + ID-cross-attn TRM predictor.

All heavy ML imports are deferred to first ``predict`` so the server can
register the nodeset cheaply (the agentcanvas backend will spawn this
under the ``smartway`` conda env via ``auto_host`` — see ``__init__.py``).

Mirrors upstream ``Policy_ViewSelection_VLNBERT.forward(mode='waypoint',
...)`` in SmartWay-Code @ daa2dd8: ``vlnce_baselines/models/
Policy_ViewSelection_VLNBERT.py:187-298``. See
workspace/nodesets/_upstream/smartway-code/fetch_upstream.sh to re-fetch.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import sys
import threading
from typing import Any

import numpy as np

from . import (
    SMARTWAY_DDPPO_CKPT_DEFAULT,
    SMARTWAY_REPO_DEFAULT,
    SMARTWAY_WAYPOINT_CKPT_DEFAULT,
)

log = logging.getLogger("agentcanvas.smartway_waypoint.engine")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _decode_rgb_b64(b64: str) -> np.ndarray:
    """Decode a base64 RGB image string → (H, W, 3) uint8."""
    from PIL import Image  # noqa: WPS433

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _decode_depth_b64(b64: str) -> np.ndarray:
    """Decode a base64 depth image string → (H, W) float in [0, 1]."""
    from PIL import Image  # noqa: WPS433

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _encode_rgb_b64(arr: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 ndarray → base64 PNG string."""
    from PIL import Image  # noqa: WPS433

    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_views(views: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Decode a list of ``{rgb_base64, depth_base64}`` dicts into ndarrays."""
    rgb_arrays: list[np.ndarray] = []
    depth_arrays: list[np.ndarray] = []
    for v in views:
        if not isinstance(v, dict):
            continue
        rgb_b64 = v.get("rgb_base64")
        depth_b64 = v.get("depth_base64")
        if rgb_b64:
            try:
                rgb_arrays.append(_decode_rgb_b64(rgb_b64))
            except Exception as exc:
                log.warning("Failed to decode RGB view: %s", exc)
        if depth_b64:
            try:
                depth_arrays.append(_decode_depth_b64(depth_b64))
            except Exception as exc:
                log.warning("Failed to decode depth view: %s", exc)
    return rgb_arrays, depth_arrays


# ──────────────────────────────────────────────────────────────────────
# WaypointEngine
# ──────────────────────────────────────────────────────────────────────


class WaypointEngine:
    """Lazy loader for the SmartWay waypoint stack (DINOv2 + DDPPO + TRM)."""

    _instance: "WaypointEngine | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.repo_path = SMARTWAY_REPO_DEFAULT
        self.waypoint_ckpt_path = SMARTWAY_WAYPOINT_CKPT_DEFAULT
        self.ddppo_ckpt_path = SMARTWAY_DDPPO_CKPT_DEFAULT
        self.device = None
        self.predictor = None
        self.rgb_encoder_dino = None
        self.dino_processor = None
        self.depth_encoder = None
        self._loaded = False

    @classmethod
    def get(cls) -> "WaypointEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            log.info("Loading SmartWay waypoint predictor from %s", self.repo_path)

            if self.repo_path not in sys.path:
                sys.path.insert(0, self.repo_path)
            # waypoint_predictor.TRM_net imports `utils` from a sibling
            # path (``waypoint_predictor/utils.py``) via ``import utils``;
            # mirror upstream by adding the waypoint_predictor dir too.
            wp_dir = os.path.join(self.repo_path, "waypoint_predictor")
            if wp_dir not in sys.path:
                sys.path.insert(0, wp_dir)

            import torch  # noqa: WPS433

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # ──── TRM predictor (BinaryDistPredictor_TRM with cross-attn) ────
            from waypoint_predictor.TRM_net import (  # type: ignore
                BinaryDistPredictor_TRM,
            )

            predictor = BinaryDistPredictor_TRM(
                hidden_dim=768, n_classes=12
            ).to(self.device)
            if os.path.exists(self.waypoint_ckpt_path):
                state = torch.load(
                    self.waypoint_ckpt_path, map_location=self.device
                )
                # Upstream best.pth wraps the model weights two levels deep:
                # ckpt['predictor']['state_dict']. Unwrap recursively until we
                # hit a dict whose values are tensors (the real state_dict).
                while isinstance(state, dict) and not any(
                    isinstance(v, torch.Tensor) for v in state.values()
                ):
                    if "state_dict" in state:
                        state = state["state_dict"]
                    elif "predictor" in state:
                        state = state["predictor"]
                    else:
                        break
                missing, unexpected = predictor.load_state_dict(state, strict=False)
                if missing or unexpected:
                    log.warning(
                        "Waypoint ckpt load: %d missing, %d unexpected keys",
                        len(missing), len(unexpected),
                    )
                log.info("Loaded waypoint predictor checkpoint")
            else:
                log.warning(
                    "Waypoint ckpt not found at %s — running uninitialised",
                    self.waypoint_ckpt_path,
                )
            for p in predictor.parameters():
                p.requires_grad = False
            predictor.eval()
            self.predictor = predictor

            # ──── DINOv2-small (with registers) — RGB feature backbone ────
            # Upstream Policy_ViewSelection_VLNBERT.py:111
            # ``torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')``
            try:
                self.rgb_encoder_dino = (
                    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
                    .to(self.device)
                )
                self.rgb_encoder_dino.eval()
                for p in self.rgb_encoder_dino.parameters():
                    p.requires_grad = False
            except Exception as exc:
                log.warning("DINOv2 load failed: %s — running with random init", exc)
                self.rgb_encoder_dino = None

            # ──── DINO image processor (preprocess to ViT-14 input) ────
            # Upstream base_il_trainer.py:356.
            try:
                from transformers import AutoImageProcessor  # type: ignore

                self.dino_processor = AutoImageProcessor.from_pretrained(
                    "facebook/dinov2-small"
                )
            except Exception as exc:
                log.warning("DINO processor load failed: %s", exc)
                self.dino_processor = None

            # ──── DDPPO depth encoder — mirrors Open-Nav ────
            try:
                # Bypass vlnce_baselines/__init__.py (which pulls in
                # ss_trainer_VLNBERT → tensorflow). Import the submodule
                # file directly via spec_from_file_location.
                import importlib.util
                enc_path = os.path.join(
                    self.repo_path, "vlnce_baselines", "models",
                    "encoders", "resnet_encoders.py",
                )
                spec = importlib.util.spec_from_file_location(
                    "_smartway_resnet_encoders", enc_path,
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                VlnResnetDepthEncoder = mod.VlnResnetDepthEncoder  # type: ignore

                # Construct the depth observation_space the encoder expects.
                # SmartWay upstream uses 256x256 depth at 1 channel — see
                # vlnce_task.yaml SIMULATOR.DEPTH_SENSOR config.
                from gym import spaces  # noqa: WPS433
                import numpy as np  # noqa: WPS433
                obs_space = spaces.Dict({
                    "depth": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(256, 256, 1),
                        dtype=np.float32,
                    ),
                })

                depth_net = VlnResnetDepthEncoder(
                    observation_space=obs_space,
                    output_size=128,
                    checkpoint=self.ddppo_ckpt_path,
                    backbone="resnet50",
                    trainable=False,
                ).to(self.device)
                depth_net.eval()
                self.depth_encoder = depth_net
            except Exception:
                log.warning(
                    "DDPPO depth encoder unavailable — depth features zeroed",
                    exc_info=True,
                )
                self.depth_encoder = None

            self._loaded = True
            log.info("Waypoint engine ready (device=%s)", self.device)

    def predict(
        self,
        rgb_views: list[np.ndarray],
        depth_views: list[np.ndarray],
    ) -> dict[int, dict[str, Any]]:
        """Return ``{idx: {angle, distance, rgb_base64}}`` keyed by 0..K-1.

        ``rgb_views`` / ``depth_views`` carry 12 views, ordered as emitted
        by ``env_habitat__panorama_rgbd``. SmartWay's upstream reverses
        the order to clockwise internally (``Policy_ViewSelection_VLNBERT.py``
        lines 198-205). The angle output is in **counter-clockwise**
        radians (``2π - idx/120 * 2π``) — the same convention as the
        ``env_habitat__step_hightolow`` action arg.
        """
        self._ensure_loaded()

        import torch  # noqa: WPS433

        if not rgb_views:
            return {}

        # Match upstream clockwise reversal: a_count = (12 - a) % 12.
        # rgb_views[0] stays at slot 0; rgb_views[1..11] reverse into
        # slots 11..1.
        n_views = min(12, len(rgb_views))
        rgb_cw = [None] * 12  # type: list[np.ndarray | None]
        depth_cw = [None] * 12  # type: list[np.ndarray | None]
        for a in range(n_views):
            ra = (12 - a) % 12
            rgb_cw[ra] = rgb_views[a]
            if a < len(depth_views):
                depth_cw[ra] = depth_views[a]

        # Pad missing views with the first valid view.
        first_rgb = next((v for v in rgb_cw if v is not None), rgb_views[0])
        for i in range(12):
            if rgb_cw[i] is None:
                rgb_cw[i] = first_rgb
        rgb_batch = np.stack(rgb_cw, axis=0)  # (12, H, W, 3) uint8

        first_depth = next(
            (v for v in depth_cw if v is not None),
            np.zeros_like(rgb_views[0][:, :, 0], dtype=np.float32) if rgb_views else None,
        )
        if first_depth is not None:
            for i in range(12):
                if depth_cw[i] is None:
                    depth_cw[i] = first_depth
            depth_batch = np.stack(depth_cw, axis=0)  # (12, H, W) float32
        else:
            depth_batch = None

        rgb_tensor_hwc = torch.from_numpy(rgb_batch).to(self.device)
        # ─── Depth encoding ───
        if self.depth_encoder is not None and depth_batch is not None:
            depth_tensor = (
                torch.from_numpy(depth_batch).unsqueeze(-1).to(self.device).float()
            )
            with torch.no_grad():
                depth_embedding = self.depth_encoder({"depth": depth_tensor})
        else:
            depth_embedding = torch.zeros(12, 128, 4, 4, device=self.device)

        # ─── RGB encoding (DINOv2) ───
        # Upstream feeds the processor a CHW int tensor; AutoImageProcessor
        # also accepts HWC numpy. We pass the numpy (12, H, W, 3) directly.
        if self.rgb_encoder_dino is not None and self.dino_processor is not None:
            pp = self.dino_processor(
                images=[rgb_batch[i] for i in range(12)], return_tensors="pt"
            )
            dino_img = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                rgb_embedding = self.rgb_encoder_dino(dino_img)  # (12, 384)
        else:
            rgb_embedding = torch.zeros(12, 384, device=self.device)

        # ─── TRM predictor with cross-attention ───
        # Upstream calls ``waypoint_predictor(rgb_embedding, depth_embedding,
        # sem_map, pre_fuse=False, cross_attn=True)``. sem_map is unused
        # in eval — pass None.
        with torch.no_grad():
            waypoint_logits = self.predictor(
                rgb_embedding, depth_embedding, None,
                pre_fuse=False, cross_attn=True,
            )  # (1, 120, 12)

        # ─── Softmax + NMS (with wrap-around padding) ───
        batch_size = 1
        NUM_ANGLES = 120
        NUM_CLASSES = 12

        batch_x_norm = torch.softmax(
            waypoint_logits.reshape(batch_size, NUM_ANGLES * NUM_CLASSES),
            dim=1,
        ).reshape(batch_size, NUM_ANGLES, NUM_CLASSES)
        batch_x_norm_wrap = torch.cat(
            (batch_x_norm[:, -1:, :], batch_x_norm, batch_x_norm[:, :1, :]),
            dim=1,
        )

        from waypoint_predictor.utils import nms  # type: ignore

        batch_output_map = nms(
            batch_x_norm_wrap.unsqueeze(1),
            max_predictions=5,
            sigma=(7.0, 5.0),
        )
        batch_output_map = batch_output_map.squeeze(1)[:, 1:-1, :]  # (1, 120, 12)

        # ─── Extract candidates ───
        out: dict[int, dict[str, Any]] = {}
        nonzero = batch_output_map[0].nonzero()
        if nonzero.numel() == 0:
            return out
        angle_idxes = nonzero[:, 0].cpu().numpy()
        distance_idxes = nonzero[:, 1].cpu().numpy()
        # 2π - idx/120 * 2π   (counter-clockwise radians)
        angles_rad = (2 * math.pi) - angle_idxes.astype(np.float32) / 120.0 * (
            2 * math.pi
        )
        distances_m = (distance_idxes.astype(np.float32) + 1) * 0.25
        # img_idxes = (angle + 5) // 10; map clockwise view index 0..11
        img_idxes = (angle_idxes + 5) // 10
        img_idxes = np.where(img_idxes == 12, 0, img_idxes)

        for k in range(len(angle_idxes)):
            tile = rgb_batch[int(img_idxes[k])]
            try:
                rgb_b64 = _encode_rgb_b64(tile)
            except Exception:
                rgb_b64 = ""
            out[k] = {
                "angle": float(angles_rad[k]),
                "distance": float(distances_m[k]),
                "rgb_base64": rgb_b64,
                "type": "waypoint",
            }
        return out
