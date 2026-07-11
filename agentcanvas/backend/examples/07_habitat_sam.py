"""Example 07 — Habitat → SAM: segment the first frame.  [EXPERIMENT]

    PYTHONPATH=. python examples/07_habitat_sam.py

One graph, three conda envs, wired by the Graph SDK and run in-process::

    env_habitat__reset ──episode_id──▶ env_habitat__observe_egocentric
       (server · ac-vlnce)                 (server · ac-vlnce)
                                              │  rgb (np.uint8 H,W,3)
                        ┌─────────────────────┴───────────────────┐
                        ▼                                          ▼
                   RgbToPng  (LOCAL)                          OverlayMasks (LOCAL)
                        │  image_b64 (base64 PNG)                  ▲  masks
                        ▼                                          │
              model_sam__segment_auto ──────────masks────────────┘
                 (server · ac-fm)                                  │ annotated
                                                                   ▼
                                                              graphOut

Two of the node types run in FOREIGN conda envs this process must never import:

  * ``env_habitat__*``  — habitat-sim under ``ac-vlnce`` (Python 3.8)
  * ``model_sam__*``    — SAM under ``ac-fm``

``g.run(load_nodesets="auto")`` scans the workspace registry, spawns one
``auto_host`` subprocess per env, and reaches each over an HTTP proxy — your
process never imports habitat or torch. The two glue nodes (``RgbToPng``,
``OverlayMasks``) are the OTHER kind: tiny ``BaseCanvasNode`` subclasses
defined and ``register_node()``'d right here, running in-process. So a single
graph shows both worlds — and the real seam between them: Habitat emits ``rgb``
as an ``np.ndarray`` (msgpack restores it as one across the proxy), while SAM
wants a base64 PNG string, so ``RgbToPng`` bridges it.

⚠ This is a real env + GPU run and belongs behind ``/experiment:run``. It needs
the ``ac-vlnce`` env (Habitat R2R-CE data), the ``ac-fm`` env (SAM ``sam_vit_b``
checkpoint at ``data/habitat/checkpoints/sam/sam_vit_b.pth``), and a GPU.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agentcanvas import Graph
from app.agent_loop.builtin_nodes import register_node
from app.components.bases import BaseCanvasNode, PortDef


# ── LOCAL node #1 — IMAGE (np.uint8 H,W,3) → base64 PNG, which is what SAM wants.
#    This is the seam: Habitat's `rgb` arrives here as an np.ndarray; SAM's
#    `segment_auto` decodes a base64 PNG string. A three-line node bridges it.
class RgbToPng(BaseCanvasNode):
    node_type = "ex_rgb_to_png"
    input_ports = [PortDef("rgb", "IMAGE", "RGB frame (np.uint8 H,W,3)")]
    output_ports = [PortDef("image_b64", "TEXT", "Base64-encoded PNG")]

    async def forward(self, inputs: dict, ctx) -> dict:
        rgb = np.asarray(inputs["rgb"], dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buf, format="PNG")
        return {"image_b64": base64.b64encode(buf.getvalue()).decode()}


# ── LOCAL node #2 — paint SAM's masks onto the frame → an annotated IMAGE.
#    The masks envelope carries, per instance, a PNG mask (`mask_b64`) and a
#    bounding box (`bbox_xyxy`); we tint each mask and outline its box.
_PALETTE = np.array(
    [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
     [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128]],
    dtype=np.uint8,
)


class OverlayMasks(BaseCanvasNode):
    node_type = "ex_overlay_masks"
    input_ports = [
        PortDef("rgb", "IMAGE", "RGB frame (np.uint8 H,W,3)"),
        PortDef("masks", "TEXT", "SAM masks envelope JSON"),
    ]
    output_ports = [
        PortDef("annotated", "IMAGE", "Frame with the mask overlay drawn on"),
        PortDef("count", "ANY", "Number of masks drawn"),
    ]

    async def forward(self, inputs: dict, ctx) -> dict:
        rgb = np.asarray(inputs["rgb"], dtype=np.uint8)
        envelope = json.loads(inputs["masks"]) if inputs.get("masks") else {"masks": []}
        masks = envelope.get("masks", [])

        canvas = rgb.astype(np.float32)
        for i, m in enumerate(masks):
            color = _PALETTE[i % len(_PALETTE)].astype(np.float32)
            mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(m["mask_b64"]))).convert("L")) > 127
            canvas[mask] = 0.55 * canvas[mask] + 0.45 * color  # 45% translucent tint
        annotated = Image.fromarray(canvas.clip(0, 255).astype(np.uint8), "RGB")

        draw = ImageDraw.Draw(annotated)
        for i, m in enumerate(masks):
            outline = tuple(int(c) for c in _PALETTE[i % len(_PALETTE)])
            draw.rectangle(list(m["bbox_xyxy"]), outline=outline, width=2)
        return {"annotated": np.asarray(annotated, dtype=np.uint8), "count": len(masks)}


for _cls in (RgbToPng, OverlayMasks):
    register_node(_cls)


def build() -> Graph:
    g = Graph(name="habitat-sam-first-frame")

    reset = g.add("env_habitat__reset", id="reset")
    observe = g.add("env_habitat__observe_egocentric", id="observe")
    to_png = g.add("ex_rgb_to_png", id="to_png")
    sam = g.add("model_sam__segment_auto", id="sam")  # defaults: variant=sam1, sam_vit_b.pth
    overlay = g.add("ex_overlay_masks", id="overlay")
    annotated = g.graph_out("annotated")
    num_masks = g.graph_out("num_masks")

    g.connect(reset.out("episode_id"), observe.in_("trigger"))   # observe only after reset
    g.connect(observe.out("rgb"), to_png.in_("rgb"))             # frame → base64 for SAM
    g.connect(to_png.out("image_b64"), sam.in_("image_b64"))
    g.connect(observe.out("rgb"), overlay.in_("rgb"))            # fan-out: raw frame reused
    g.connect(sam.out("masks"), overlay.in_("masks"))
    g.connect(overlay.out("annotated"), annotated.in_("value"))
    g.connect(overlay.out("count"), num_masks.in_("value"))
    return g


if __name__ == "__main__":
    g = build()
    print("Habitat → SAM, one frame, in-process (load_nodesets='auto') — "
          "spawning ac-vlnce + ac-fm servers…")
    r = g.run(load_nodesets="auto", validate=True)

    frame = r.outputs.get("annotated")
    n = r.outputs.get("num_masks")
    if frame is None:
        raise SystemExit("no frame produced — env/SAM server failed to load (see logs above)")

    dst = Path("outputs") / "habitat_sam_first_frame.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB").save(dst)
    print(f"SAM found {n} masks — annotated frame saved to {dst}")
    sys.exit(0)
