"""ToolEQA toolbox — 6 callable tools wired into evaluate_python_code.

Each tool is a `transformers.Tool` subclass, mirroring upstream's
`src/tools/{vqa,location_2d,location_3d,go_next_point,crop,final_answer}.py`.
The differences from upstream are limited to the IPC backend:

  * Upstream's `ObjectLocation2D` / `ObjectLocation3D` use posix_ipc shared
    memory to talk to a separate DetAny3D worker (`app_mp.py`). Here they
    POST to the agentcanvas backend's `env_detany3d` server-mode subprocess
    over HTTP — same DetAny3D models, different IPC.
  * Upstream's `GoNextPointTool` calls `eqa_modeling.go_next_point(command)`
    directly (in-process simulator). Here it dispatches over HTTP to the
    hmeqa-side `tooleqa_explore__go_next` node (full TSDF frontier step +
    teleport), which returns the next observation; the new frame is stashed
    in `pending_frame` for the canvas node to thread to the next iter, and
    the new RGB path is returned to the agent.
  * Upstream's `FinalAnswerTool` returns the answer; here it sets
    `_pending_answer` + `_pending_done` on the toolbox.
  * `SegmentInstance` is intentionally absent — upstream's
    `src/tools/tool_box.py:19,34` comments it out of the active toolbox.

Tool descriptions + input schemas are verbatim from upstream so
`transformers.agents.tools.get_tool_description_with_args` produces the
same prompt content (R3 + R4 faithfulness).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# `transformers.Tool` is the upstream base class; we lazy-import to keep
# this module loadable without transformers in the import path during
# tests / linting.

ToolBaseT: Any = None  # populated lazily inside `_get_tool_base()`


def _get_tool_base() -> Any:
    """Lazy-load `transformers.Tool` (upstream's pre-deprecation API)."""
    global ToolBaseT
    if ToolBaseT is not None:
        return ToolBaseT
    try:
        from transformers import Tool  # type: ignore[attr-defined]

        ToolBaseT = Tool
    except ImportError as e:
        raise ImportError(
            "transformers.Tool not importable. ToolEQA pins `transformers` "
            "to a version that still ships transformers.agents (~=4.40, "
            "deprecated thereafter — moved to `smolagents`). Pin in the "
            "agentcanvas backend env: pip install 'transformers>=4.40,<4.46'."
        ) from e
    return ToolBaseT


# ══════════════════════════════════════════════════════════════════════
# Toolbox state container — buffers for action / answer that the canvas
# node reads post-evaluate_python_code
# ══════════════════════════════════════════════════════════════════════


class ToolEQAToolbox:
    """Per-step toolbox bundle.

    Lifetime: constructed fresh each canvas iter from the current
    observation snapshot. Read-only tools (vqa, locate_2d, locate_3d,
    crop) execute inline against the snapshot; the GoNextPointTool steps
    the env in-tool (HTTP) and stashes the new frame in `pending_frame`;
    FinalAnswerTool writes `pending_answer` / `pending_done`. The canvas
    node drains these after exec.
    """

    def __init__(
        self,
        *,
        rgb_path: str,
        output_dir: str,
        sample_id: str,
        vlm_call: Callable[[str, list[str]], str],
        detany3d_locate_2d: Callable[[Any, str], dict],
        detany3d_locate_3d: Callable[[Any, str], dict],
        go_next_call: Callable[[str], dict],
    ) -> None:
        self.rgb_path = rgb_path
        self.output_dir = output_dir
        self.sample_id = sample_id
        self.vlm_call = vlm_call
        self.detany3d_locate_2d = detany3d_locate_2d
        self.detany3d_locate_3d = detany3d_locate_3d
        # go_next_call(direction) -> new-frame dict from tooleqa_explore__go_next
        self.go_next_call = go_next_call

        # Output buffers — read by canvas node post-exec
        self.pending_frame: dict | None = None  # new observation from go_next
        self.pending_answer: str = ""  # final answer text
        self.pending_done: bool = False

    def build_tools(self) -> list:
        """Return the 6 tool instances bound to this toolbox."""
        return [
            VisualQATool(self),
            ObjectLocation2D(self),
            ObjectLocation3D(self),
            ObjectCrop(self),
            GoNextPointTool(self),
            FinalAnswerTool(self),
        ]

    def tools_dict(self) -> dict[str, Any]:
        """Return ``{tool_name: tool_callable}`` for evaluate_python_code's `static_tools`."""
        return {t.name: t for t in self.build_tools()}


# ══════════════════════════════════════════════════════════════════════
# Tool subclasses — verbatim signatures from upstream tool files
# ══════════════════════════════════════════════════════════════════════


def _resolve_path(path: str, output_dir: str) -> str:
    """Mirror upstream's path-resolution dance (e.g. location_3d.py:43-48)."""
    if os.path.exists(path):
        return path
    if os.path.exists("/" + path):
        return "/" + path
    candidate = os.path.join(output_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path  # let the caller raise


def _make_visual_qa_tool() -> type:
    Tool = _get_tool_base()

    class VisualQATool(Tool):  # type: ignore[misc, valid-type]
        # Verbatim from upstream src/tools/vqa.py
        name = "VisualQATool"
        description = "A tool that can answer questions about attached images."
        inputs = {
            "question": {"description": "the question to answer", "type": "string"},
            "image_paths": {
                "description": "The path to the image on which to answer the question",
                "type": "string",
            },
        }
        output_type = "string"

        def __init__(self, toolbox: ToolEQAToolbox) -> None:
            super().__init__()
            self._tb = toolbox

        def forward(self, question: str, image_paths: Any = "") -> str:
            paths = image_paths
            if isinstance(paths, str):
                paths = [paths] if paths else []
            elif not isinstance(paths, list):
                raise TypeError("image_paths must be string or list of strings")
            if not paths:
                paths = [self._tb.rgb_path]
            paths = [_resolve_path(p, self._tb.output_dir) for p in paths]
            if not question:
                question = "Please write a detailed caption for this image."
            result = self._tb.vlm_call(question, paths)
            return result if isinstance(result, str) else ""

    return VisualQATool


def _make_object_location_2d_tool() -> type:
    Tool = _get_tool_base()

    class ObjectLocation2D(Tool):  # type: ignore[misc, valid-type]
        # Verbatim from upstream src/tools/location_2d.py
        name = "ObjectLocation2D"
        description = (
            "A tool that can localize objects in given images, outputing the "
            "bounding boxes of the objects."
        )
        inputs = {
            "object": {"description": "the object that need to be localized", "type": "string"},
            "image_path": {
                "description": "The path to the image on which to localize objects.",
                "type": "string",
            },
        }
        output_type = "any"

        def __init__(self, toolbox: ToolEQAToolbox) -> None:
            super().__init__()
            self._tb = toolbox

        def forward(self, object: str, image_path: str) -> Any:
            import numpy as np
            from PIL import Image

            path = _resolve_path(image_path, self._tb.output_dir)
            image = np.array(Image.open(path).convert("RGB"))
            res = self._tb.detany3d_locate_2d(image, object)
            if res.get("error"):
                raise Exception(f"Error: {res['error']}")
            return {
                "bboxes_2d": res.get("bboxes_2d", []),
                "labels": res.get("labels", []),
                "text": object,
            }

    return ObjectLocation2D


def _make_object_location_3d_tool() -> type:
    Tool = _get_tool_base()

    class ObjectLocation3D(Tool):  # type: ignore[misc, valid-type]
        # Verbatim from upstream src/tools/location_3d.py
        name = "ObjectLocation3D"
        description = (
            "Localize 3D objects in the scene and return their 3D bounding boxes "
            "and center coordinates."
        )
        inputs = {
            "object": {"description": "the object that need to be localized", "type": "string"},
            "image_path": {
                "description": "List of the pathes to the images on which to localize 3D objects.",
                "type": "string",
            },
        }
        output_type = "any"

        def __init__(self, toolbox: ToolEQAToolbox) -> None:
            super().__init__()
            self._tb = toolbox

        def forward(self, object: str, image_path: str) -> tuple:
            import numpy as np
            from PIL import Image

            path = _resolve_path(image_path, self._tb.output_dir)
            image = np.array(Image.open(path).convert("RGB"))
            res = self._tb.detany3d_locate_3d(image, object)
            if res.get("error"):
                print(f"Error: {object}, {image_path}, {res['error']}")
                return None, None
            centers = res.get("centers_3d", [])
            sizes = res.get("sizes_3d", [])
            return centers, sizes

    return ObjectLocation3D


def _make_object_crop_tool() -> type:
    Tool = _get_tool_base()

    class ObjectCrop(Tool):  # type: ignore[misc, valid-type]
        # Verbatim from upstream src/tools/crop.py
        name = "ObjectCrop"
        description = (
            "Given the bounding boxes of objects, crop and save the relevant "
            "objects from the image."
        )
        inputs = {
            "bound_boxes": {"description": "the bounding boxes of objects", "type": "number"},
            "image_path": {
                "description": "The path to the image on which to crop objects.",
                "type": "string",
            },
        }
        output_type = "string"

        def __init__(self, toolbox: ToolEQAToolbox) -> None:
            super().__init__()
            self._tb = toolbox

        @staticmethod
        def _list_dim(lst: Any) -> int:
            if not isinstance(lst, list):
                return 0
            if not lst:
                return 1
            return 1 + ObjectCrop._list_dim(lst[0])

        def forward(self, bound_boxes: list, image_path: str) -> list:
            bounding_box = bound_boxes
            from PIL import Image

            path = _resolve_path(image_path, self._tb.output_dir)
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}") from e

            if self._list_dim(bounding_box) == 1:
                bounding_box = [bounding_box]
            if not isinstance(bounding_box, list) or not all(len(b) == 4 for b in bounding_box):
                raise ValueError("Bounding boxes must be a list of [x1, y1, x2, y2]")

            base_name = os.path.splitext(os.path.basename(path))[0]
            folder = os.path.dirname(path)
            output_paths: list[str] = []
            for idx, bbox in enumerate(bounding_box):
                cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                out = os.path.join(folder, f"{base_name}_crop_obj_{idx}.jpg")
                cropped.save(out)
                output_paths.append(out)
            return output_paths

    return ObjectCrop


def _make_go_next_point_tool() -> type:
    Tool = _get_tool_base()

    class GoNextPointTool(Tool):  # type: ignore[misc, valid-type]
        # Verbatim signature from upstream src/tools/go_next_point.py.
        # IMPLEMENTATION DIVERGES from in-process simulation: dispatches the
        # full TSDF frontier step + teleport over HTTP to the hmeqa-side
        # tooleqa_explore__go_next node (faithful to go_next_point's
        # semantics — the command is a direction hint, not a discrete move).
        name = "GoNextPointTool"
        description = (
            "the agent conitnue explore next point and obtain next observation (rgb image)."
        )
        inputs = {
            "direction": {
                "description": (
                    "Next exploration direction, ONLY [`move_forward`, `turn_left`, `turn_right`, "
                    "`turn_around`] are supported. `move_forward` means moving forward by 0.5 "
                    "meters. `turn_left` is a 45 degree left turn. `turn_right` is a 45 degree "
                    "right turn. `turn_around` is a 180 degree turn."
                ),
                "type": "string",
            },
        }
        output_type = "string"

        def __init__(self, toolbox: ToolEQAToolbox) -> None:
            super().__init__()
            self._tb = toolbox

        def forward(self, direction: Any) -> str:
            command = direction
            if isinstance(command, dict):
                command = command.get("direction", "")
            from ._actions import VALID_DIRECTIONS

            cmd = (str(command) or "").strip().lower().replace(" ", "_").replace("-", "_")
            if cmd not in VALID_DIRECTIONS:
                raise ValueError(
                    f"GoNextPointTool: direction must be one of {VALID_DIRECTIONS}, got {command!r}"
                )

            # Full TSDF frontier step + teleport over HTTP. Returns the next
            # observation; stash the new frame for the canvas node to thread
            # to the next iter, and hand the new RGB path back to the agent.
            frame = self._tb.go_next_call(cmd)
            self._tb.pending_frame = frame
            rgb_path = str(frame.get("rgb_path", "") or "")
            if rgb_path:
                # Subsequent tools in this same code block see the new view.
                self._tb.rgb_path = rgb_path
            return rgb_path

    return GoNextPointTool


def _make_final_answer_tool() -> type:
    Tool = _get_tool_base()

    class FinalAnswerTool(Tool):  # type: ignore[misc, valid-type]
        # Verbatim signature from upstream src/tools/final_answer.py
        name = "final_answer"
        description = "Provides a final answer to the given problem."
        inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
        output_type = "any"

        def __init__(self, toolbox: ToolEQAToolbox) -> None:
            super().__init__()
            self._tb = toolbox

        def forward(self, answer: Any = None) -> Any:
            if isinstance(answer, dict) and "answer" in answer:
                answer = answer["answer"]
            answer_str = str(answer) if answer is not None else ""
            self._tb.pending_answer = answer_str
            self._tb.pending_done = True
            return answer_str

    return FinalAnswerTool


# Bind the tool classes to module-level names so build_tools() can construct them.
# Lazy resolution — only instantiated when the canvas backend has transformers.

VisualQATool: Any = None
ObjectLocation2D: Any = None
ObjectLocation3D: Any = None
ObjectCrop: Any = None
GoNextPointTool: Any = None
FinalAnswerTool: Any = None


def _resolve_tool_classes() -> None:
    """Bind module-level names. Called from `ToolEQAToolbox.build_tools()`."""
    global VisualQATool, ObjectLocation2D, ObjectLocation3D
    global ObjectCrop, GoNextPointTool, FinalAnswerTool
    if VisualQATool is None:
        VisualQATool = _make_visual_qa_tool()
        ObjectLocation2D = _make_object_location_2d_tool()
        ObjectLocation3D = _make_object_location_3d_tool()
        ObjectCrop = _make_object_crop_tool()
        GoNextPointTool = _make_go_next_point_tool()
        FinalAnswerTool = _make_final_answer_tool()


# Patch build_tools to lazy-resolve before constructing.
_orig_build_tools = ToolEQAToolbox.build_tools


def _build_tools_lazy(self: ToolEQAToolbox) -> list:
    _resolve_tool_classes()
    return [
        VisualQATool(self),
        ObjectLocation2D(self),
        ObjectLocation3D(self),
        ObjectCrop(self),
        GoNextPointTool(self),
        FinalAnswerTool(self),
    ]


ToolEQAToolbox.build_tools = _build_tools_lazy  # type: ignore[method-assign]
