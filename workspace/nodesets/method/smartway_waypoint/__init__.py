"""SmartWay waypoint predictor nodeset (server mode).

Source modules (sys.path-inserted at engine load time, mirroring the
existing ``opennav_waypoint`` pattern for vendored upstream-repo imports):

    ./_vendored/waypoint_predictor/TRM_net.py                 BinaryDistPredictor_TRM
    ./_vendored/waypoint_predictor/img_depth_corss_attention.py ID_CrossAttention
    ./_vendored/waypoint_predictor/transformer/waypoint_bert.py WaypointBert
    ./_vendored/waypoint_predictor/utils.py                   nms
    vlnce_baselines/models/encoders/resnet_encoders.py        VlnResnetDepthEncoder
                                                              (from VLN-CE submodule)

Upstream: SmartWay-Code @ daa2dd8 — see
``workspace/nodesets/_upstream/smartway-code/fetch_upstream.sh`` to re-fetch.

The DINOv2 RGB backbone was extracted to the ``model_dinov2`` FM nodeset
(TODO #56, 2026-07-04): graphs wire ``model_dinov2__extract_features``
(env-order per-view embeddings) into this node's ``rgb_features`` input; the
engine applies its own clockwise remap, byte-identical to the old in-engine
forward. Unwired ``rgb_features`` → zeros (degraded mode, mirrors the old
DINOv2 load-failure fallback).

Runs in the new ``smartway`` conda env (Python 3.8.20 + torch 2.1.1 + cu121
+ recognize-anything). ``SMARTWAY_PYTHON`` env var overrides the
interpreter path.

One tool:

    smartway_waypoint__predict   views + rgb_features → candidates
                                 (incl. per-cand RGB), num_candidates

The output ``candidates`` is **keyed by integer index 0..K-1** (not by
12-slot dir_id like Open-Nav) because SmartWay's prompt assembly reads
candidates in dict-iteration order. K is variable per step.

Checkpoints (env-var override → default path under ``data/smartway/``):

    SMARTWAY_REPO_PATH        ./_vendored (override for real upstream clone)
    SMARTWAY_WAYPOINT_CKPT    data/smartway/waypoint_ckpt/best.pth
    SMARTWAY_DDPPO_CKPT       data/smartway/ddppo/gibson-2plus-resnet50.pth
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.smartway_waypoint")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_PKG_DIR, "..", "..", "..", ".."))

# Default points at the vendored ``_vendored/`` sub-dir which contains
# the ``waypoint_predictor/`` sub-tree (verbatim copy of upstream).
# Override SMARTWAY_REPO_PATH to a real upstream clone for local edits.
SMARTWAY_REPO_DEFAULT = os.environ.get(
    "SMARTWAY_REPO_PATH",
    os.path.join(_PKG_DIR, "_vendored"),
)
SMARTWAY_WAYPOINT_CKPT_DEFAULT = os.environ.get(
    "SMARTWAY_WAYPOINT_CKPT",
    os.path.join(_REPO_ROOT, "data", "smartway", "waypoint_ckpt", "best.pth"),
)
SMARTWAY_DDPPO_CKPT_DEFAULT = os.environ.get(
    "SMARTWAY_DDPPO_CKPT",
    os.path.join(_REPO_ROOT, "data", "smartway", "ddppo", "gibson-2plus-resnet50.pth"),
)


# ══════════════════════════════════════════════════════════════════════
# Canvas tool
# ══════════════════════════════════════════════════════════════════════


class SmartwayWaypointPredictTool(BaseCanvasNode):
    """Predict candidate waypoints from a 12-view RGB-D panorama (SmartWay)."""

    node_type: ClassVar[str] = "smartway_waypoint__predict"
    display_name: ClassVar[str] = "SmartWay: Predict Waypoints"
    description: ClassVar[str] = (
        "ID-cross-attn + TRM heatmap over wired DINOv2 features → ≤5 candidates with RGB tiles"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Target"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef(
            "views",
            "ANY",
            "List of {dir_id, rgb_base64, depth_base64} (12 views, clockwise convention)",
        ),
        PortDef(
            "rgb_features",
            "TEXT",
            'DINOv2 feature envelope {"shape":[N,384],"dtype":"float32","b64":…} from'
            " model_dinov2__extract_features, env order. REQUIRED — an optional port"
            " does not gate firing, so the executor raced ahead of the DINOv2 node"
            " and silently ran zeros-degraded (caught 2026-07-04). Load-failure"
            " degradation still works: an empty envelope decodes to None → zeros.",
        ),
    ]
    output_ports = [
        PortDef(
            "candidates",
            "ANY",
            "{idx: {angle, distance, rgb_base64}} keyed by 0..K-1",
        ),
        PortDef("num_candidates", "ANY", "K"),
    ]

    @staticmethod
    def _decode_features(envelope: Any):
        """base64-npy envelope → (N, D) float32 ndarray, or None."""
        if not envelope or not isinstance(envelope, str):
            return None
        import base64
        import json

        import numpy as np

        try:
            meta = json.loads(envelope)
            buf = base64.b64decode(meta["b64"])
            arr = np.frombuffer(buf, dtype=meta.get("dtype", "float32"))
            return arr.reshape(meta["shape"]).astype(np.float32)
        except Exception:
            log.warning("rgb_features envelope decode failed — features zeroed", exc_info=True)
            return None

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from ._engine import WaypointEngine, decode_views

        views = inputs.get("views") or []
        rgb_arrays, depth_arrays = decode_views(views)
        rgb_features = self._decode_features(inputs.get("rgb_features"))

        if not rgb_arrays:
            self._self_log("skipped", "no_rgb_views")
            return {"candidates": {}, "num_candidates": 0}

        loop = asyncio.get_running_loop()
        engine = WaypointEngine.get()
        candidates = await loop.run_in_executor(
            None, engine.predict, rgb_arrays, depth_arrays, rgb_features
        )
        self._self_log("num_candidates", len(candidates))
        self._self_log("rgb_features", "wired" if rgb_features is not None else "MISSING(zeros)")
        return {"candidates": candidates, "num_candidates": len(candidates)}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class SmartWayWaypointNodeSet(BaseNodeSet):
    """SmartWay enhanced waypoint predictor (server mode, smartway env)."""

    name = "smartway_waypoint"
    description = "DINOv2 + ID-cross-attn waypoint predictor (SmartWay, IROS 2025)"
    # Smartway env: Py 3.8.20 + torch 2.1.1 + dinov2 + habitat-sim 0.1.7.
    # The depth encoder import chain needs vlnce_baselines on sys.path,
    # which the engine handles at load time.
    server_python = os.environ.get(
        "SMARTWAY_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-smartway/bin/python")
    )
    # Pure-functional inference: K waypoint predictions per call, no
    # caller-scoped state — safe to share across batched-eval workers.
    parallelism: ClassVar[str] = "shared"

    def get_tools(self) -> list:
        return [SmartwayWaypointPredictTool()]

    async def initialize(self, **kwargs: Any) -> None:
        # Defer all heavy ML imports to first ``predict`` call so the
        # backend can register the nodeset cheaply.
        repo = kwargs.get("repo_path")
        wayp_ckpt = kwargs.get("waypoint_ckpt")
        ddppo_ckpt = kwargs.get("ddppo_ckpt")
        from ._engine import WaypointEngine

        engine = WaypointEngine.get()
        if repo:
            engine.repo_path = str(repo)
        if wayp_ckpt:
            engine.waypoint_ckpt_path = str(wayp_ckpt)
        if ddppo_ckpt:
            engine.ddppo_ckpt_path = str(ddppo_ckpt)
        log.info(
            "SmartWayWaypointNodeSet ready (repo=%s, wp=%s, ddppo=%s)",
            engine.repo_path,
            engine.waypoint_ckpt_path,
            engine.ddppo_ckpt_path,
        )

    async def shutdown(self) -> None:
        from ._engine import WaypointEngine

        WaypointEngine.reset()
