"""SmartWay perception nodeset (server mode) — RAM+ tag-only.

Wraps Recognize Anything Plus (RAM+, swin_large_14m) for per-candidate
object tagging. Mirrors ``workspace/nodesets/method/opennav_perception.py``
but swaps:

    from ram.models import ram        →  from ram.models import ram_plus
    image_size=224                    →  image_size=384  (upstream _eval_checkpoint:369)
    weights: ram_swin_large_14m.pth   →  ram_plus_swin_large_14m.pth

Upstream call site (SmartWay-Code @ daa2dd8; see
workspace/nodesets/_upstream/smartway-code/fetch_upstream.sh):
    vlnce_baselines/common/base_il_trainer.py
        line 368-371  ram_plus(pretrained=..., image_size=384, vit='swin_l')
        line 440-442  inference_ram(img, ram_model) per candidate RGB

Runs in the same ``smartway`` conda env as ``smartway_waypoint``
(SMARTWAY_PYTHON env var). Single tool:

    smartway_perception__tag    candidates → tags  (per-candidate tag string)

The input ``candidates`` is the SmartWay waypoint predictor's output —
``{idx: {angle, distance, rgb_base64}}`` — so this nodeset is a direct
downstream consumer with no decoding mismatch.

Author-relationship note: SmartWay is Adelaide Qi Wu group; this is a
side-experiment port, not PortBench v1 (vln-methods.md § 3.2).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.smartway_perception")

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
)

SMARTWAY_RAM_PLUS_CKPT_DEFAULT = os.environ.get(
    "SMARTWAY_RAM_PLUS_CKPT",
    os.path.join(_REPO_ROOT, "data", "smartway", "ram_plus", "ram_plus_swin_large_14m.pth"),
)


class SmartwayPerceptionTagTool(BaseCanvasNode):
    """Per-candidate RAM+ object tagging.

    Input ``candidates`` shape ``{idx: {angle, distance, rgb_base64, ...}}``
    (output of ``smartway_waypoint__predict``). For each candidate with a
    non-empty rgb_base64, decode and tag; output ``tags`` is keyed by the
    same idx so the downstream prompt_assembly can look up tag per option.
    """

    node_type: ClassVar[str] = "smartway_perception__tag"
    display_name: ClassVar[str] = "SmartWay: Perception Tag"
    description: ClassVar[str] = "RAM+ per-candidate object tags"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Tag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef(
            "candidates",
            "ANY",
            "{idx: {rgb_base64, ...}} from smartway_waypoint__predict",
        ),
    ]
    output_ports = [
        PortDef("tags", "ANY", "{idx: tag_string} aligned with candidates"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from ._engine import PerceptionEngine, decode_rgb_b64

        cands = inputs.get("candidates") or {}
        if not isinstance(cands, dict) or not cands:
            self._self_log("skipped", "no_candidates")
            return {"tags": {}}

        # Materialise idx → ndarray
        loop = asyncio.get_running_loop()
        engine = PerceptionEngine.get()

        out: dict[int, str] = {}
        for k, v in cands.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(v, dict):
                continue
            rgb_b64 = v.get("rgb_base64", "")
            if not rgb_b64:
                out[idx] = ""
                continue
            try:
                arr = decode_rgb_b64(rgb_b64)
            except Exception as exc:
                log.warning("Failed to decode RGB for idx=%s: %s", idx, exc)
                out[idx] = ""
                continue
            tag = await loop.run_in_executor(None, engine.tag, arr)
            out[idx] = tag

        self._self_log("num_tagged", len(out))
        return {"tags": out}


class SmartWayPerceptionNodeSet(BaseNodeSet):
    """SmartWay RAM+ perception (server mode, smartway env)."""

    name = "smartway_perception"
    description = "RAM+ swin_large_14m per-candidate object tagging (SmartWay)"
    server_python = os.environ.get(
        "SMARTWAY_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-smartway/bin/python")
    )
    # Pure-functional: tag(image) → tag_string, no caller-scoped state.
    parallelism: ClassVar[str] = "shared"

    def get_tools(self) -> list:
        return [SmartwayPerceptionTagTool()]

    async def initialize(self, **kwargs: Any) -> None:
        ckpt = kwargs.get("ram_plus_ckpt")
        from ._engine import PerceptionEngine

        engine = PerceptionEngine.get()
        if ckpt:
            engine.ram_ckpt = str(ckpt)
        log.info(
            "SmartWayPerceptionNodeSet ready (ram_plus_ckpt=%s)",
            engine.ram_ckpt,
        )

    async def shutdown(self) -> None:
        from ._engine import PerceptionEngine

        PerceptionEngine.reset()
