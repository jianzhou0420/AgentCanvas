"""ToolEQA exploration nodeset — the fused ``go_next`` frontier step.

One node:

  tooleqa_explore__go_next  — full Explore-EQA TSDF frontier step + teleport

This is the faithful port of upstream ToolEQA's ``GoNextPointTool`` →
``EQA_Modeling.go_next_point(command)`` (``src/runs/eqa_modeling.py``):
integrate the current RGB-D into the per-episode TSDF, find frontier
candidate points within view, score them with the VLM (LSV = pick a
labelled direction, GSV = is-any-direction-worth-exploring Yes/No),
integrate the semantic value into the map, pick a frontier-weighted next
pose, and teleport the agent there to obtain the next observation. The
upstream ``command`` ("move_forward" / "turn_left" / …) contributes ONLY a
direction *hint word* to the LSV prompt — it is NOT a discrete motion
(that misreading was the bug in the first, deleted, port).

Why a separate hmeqa-side nodeset (not inside the backend ``tooleqa``
reasoner): the TSDF frontier planner (``method/_explore_eqa_tsdf.py``'s ``TSDFPlanner``,
numba) and habitat live in the ``hmeqa`` env, while the ReAct engine
(``transformers.agents``) lives in the agentcanvas backend env. ToolEQA's
``go_next`` tool, called inside the ReAct loop, dispatches over HTTP to
this node; this node in turn calls the Qwen VLM and ``env_hmeqa__step``
over HTTP (their URLs are resolved by the backend and passed in as inputs,
since an auto_host subprocess has no component registry of its own).

Metric depth: ``env_hmeqa__step``/``__reset`` emit depth on the ANY wire
(lossless ndarray) — the DEPTH wire normalizes to [0,1] over HTTP, which
would corrupt the TSDF. Keep every depth port in this pipeline on ANY.

Per-episode TSDFPlanner state is subprocess-local, keyed by ``episode_id``,
and retained across episodes (cleared only at subprocess teardown) —
mirrors ``explore_eqa``'s race-fix discipline.

Server mode: dedicated ``hmeqa`` env (Python 3.9 + numba). Reuses the
vendored ``method/_explore_eqa_tsdf.py``'s ``TSDFPlanner``.

last updated: 2026-06-08
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, ClassVar

import numpy as np

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)
from app.server.serialization import deserialize_value, serialize_value

log = logging.getLogger("agentcanvas.tooleqa_explore")


# ── Per-episode live TSDFPlanner (subprocess-local, never globally cleared) ──
_TSDF_PLANNERS: dict[str, Any] = {}

# ── Defaults (mirror explore-eqa/cfg/vlm_exp.yaml + react-eqa.yaml) ──
_DEFAULT_TSDF_VOXEL_SIZE = 0.1
_DEFAULT_INIT_CLEARANCE = 0.5
_DEFAULT_MARGIN_H_RATIO = 0.6
_DEFAULT_MARGIN_W_RATIO = 0.25
_GSV_T = 0.5
_GSV_F = 3.0

_DRAW_LETTERS = ["A", "B", "C", "D"]
_CIRCLE_RADIUS = 18
_FONT_SIZE = 30

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_DEFAULT_FONT_PATH = os.path.join(
    _REPO_ROOT, "data", "hm3d", "hmeqa", "Open_Sans", "static", "OpenSans-Regular.ttf"
)
_DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs", "tooleqa_runs")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _to_pil_rgb(rgb):
    from PIL import Image

    if rgb is None:
        return None
    if isinstance(rgb, Image.Image):
        return rgb.convert("RGB")
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def _get_or_build_planner(
    episode_id: str,
    tsdf_bnds,
    pts_init_normal,
    voxel_size: float,
    init_clearance: float,
):
    if episode_id in _TSDF_PLANNERS:
        return _TSDF_PLANNERS[episode_id]
    from workspace.nodesets.method._explore_eqa_tsdf import TSDFPlanner

    bnds = np.asarray(tsdf_bnds, dtype=np.float64)
    planner = TSDFPlanner(
        vol_bnds=bnds,
        voxel_size=float(voxel_size),
        floor_height_offset=0,
        pts_init=np.asarray(pts_init_normal, dtype=np.float64),
        init_clearance=float(init_clearance) * 2,
    )
    _TSDF_PLANNERS[episode_id] = planner
    log.info("TSDFPlanner built episode_id=%s vol_dim=%s", episode_id, planner._vol_dim.tolist())
    return planner


def _http_call(url: str, function_name: str, inputs_payload: dict, config: dict, timeout: float):
    """POST to ``{url}/call/{function_name}`` (server-mode proxy route).

    Returns the raw ``outputs`` dict (still wire-serialized) or raises.
    ``trust_env=False`` mirrors the loopback-proxy policy so a shell
    ``HTTP_PROXY`` doesn't swallow loopback traffic.
    """
    import requests

    call_url = "{}/call/{}".format(url.rstrip("/"), function_name)
    sess = requests.Session()
    sess.trust_env = False
    resp = sess.post(call_url, json={"inputs": inputs_payload, "config": config}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("outputs", data)


def _qwen_generate(qwen_url: str, prompt: str, image_path: str, max_new_tokens: int = 8) -> str:
    """One Qwen2.5-VL generate call over HTTP. Greedy (temperature 0) for
    deterministic LSV/GSV scoring."""
    payload = {
        "prompt": serialize_value(prompt, "TEXT"),
        "image_paths": serialize_value([image_path], "ANY"),
    }
    out = _http_call(
        qwen_url,
        "vlm_qwen2_5_vl__generate",
        payload,
        {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        timeout=120.0,
    )
    return str(deserialize_value(out.get("text"), "TEXT") or "")


def _first_letter(text: str, letters: list[str]) -> int:
    """Return the index in ``letters`` of the first matching upper-case letter
    in ``text``; -1 if none. Mirrors upstream's ``response == letters[i]`` but
    tolerant of trailing punctuation/words."""
    up = (text or "").strip().upper()
    for i, lt in enumerate(letters):
        if (
            up == lt
            or up.startswith(lt + ".")
            or up.startswith(lt + " ")
            or up.startswith(lt + ",")
        ):
            return i
    for i, lt in enumerate(letters):
        if lt in up:
            return i
    return -1


# ══════════════════════════════════════════════════════════════════════
# Node: GoNext — the fused frontier step
# ══════════════════════════════════════════════════════════════════════


class GoNextNode(BaseCanvasNode):
    """Full Explore-EQA frontier step + teleport (upstream go_next_point)."""

    node_type: ClassVar[str] = "tooleqa_explore__go_next"
    display_name: ClassVar[str] = "ToolEQA: Go Next Point"
    description: ClassVar[str] = (
        "Integrate RGB-D into TSDF, score frontier candidates with the VLM "
        "(LSV/GSV), pick a frontier-weighted next pose, teleport via "
        "env_hmeqa__step, return the next observation."
    )
    category: ClassVar[str] = "planner"
    icon: ClassVar[str] = "Navigation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="blue")

    config_schema: ClassVar[list[ConfigField]] = [
        ConfigField("voxel_size", "number", default=_DEFAULT_TSDF_VOXEL_SIZE),
        ConfigField("init_clearance", "number", default=_DEFAULT_INIT_CLEARANCE),
        ConfigField("margin_h_ratio", "number", default=_DEFAULT_MARGIN_H_RATIO),
        ConfigField("margin_w_ratio", "number", default=_DEFAULT_MARGIN_W_RATIO),
        ConfigField("num_prompt_points", "integer", default=3),
        ConfigField("min_num_prompt_points", "integer", default=2),
        ConfigField("use_lsv", "boolean", default=True),
        ConfigField("use_gsv", "boolean", default=True),
        ConfigField("min_random_init_steps", "integer", default=2),
        ConfigField("img_width", "integer", default=640),
        ConfigField("img_height", "integer", default=480),
        ConfigField("output_dir", "text", default=_DEFAULT_OUTPUT_DIR),
    ]

    input_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "Current RGB observation"),
        PortDef("depth", "ANY", "Current metric depth (HxW float, lossless ANY wire)"),
        PortDef("cam_pose", "ANY", "4x4 TSDF-frame camera extrinsic (cam_pose_matrix)"),
        PortDef("cam_intr", "ANY", "3x3 camera intrinsics"),
        PortDef("pose_normal", "ANY", "Current 3-vector normal-frame position"),
        PortDef("angle", "ANY", "Current yaw (radians)"),
        PortDef("floor_height", "ANY", "Floor z (normal frame, episode constant)"),
        PortDef("tsdf_bnds", "ANY", "3x2 TSDF bounds (used on first call only)"),
        PortDef("episode_id", "TEXT", "Episode id (planner lookup key)"),
        PortDef("step_index", "ANY", "Current step index"),
        PortDef("vlm_question", "TEXT", "Formatted question for LSV/GSV prompts"),
        PortDef("direction", "TEXT", "Hint word from the LLM command (e.g. move_forward)"),
        PortDef("qwen_url", "TEXT", "Resolved vlm_qwen2_5_vl server URL"),
        PortDef("env_hmeqa_url", "TEXT", "Resolved env_hmeqa server URL"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "Post-teleport RGB"),
        PortDef("depth", "ANY", "Post-teleport metric depth"),
        PortDef("pose_normal", "ANY", "New 3-vector normal-frame position"),
        PortDef("angle", "ANY", "New yaw (radians)"),
        PortDef("cam_pose", "ANY", "New 4x4 TSDF-frame extrinsic"),
        PortDef("cam_intr", "ANY", "Pass-through intrinsics"),
        PortDef("step_index", "ANY", "New step index (from env)"),
        PortDef("done", "BOOL", "Env-side done flag"),
        PortDef("rgb_path", "TEXT", "Absolute path of the saved post-teleport RGB"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        return await asyncio.to_thread(self._run, inputs)

    def _run(self, inputs: dict) -> dict:
        cfg = self.config or {}
        voxel_size = float(cfg.get("voxel_size", _DEFAULT_TSDF_VOXEL_SIZE))
        init_clearance = float(cfg.get("init_clearance", _DEFAULT_INIT_CLEARANCE))
        margin_h_ratio = float(cfg.get("margin_h_ratio", _DEFAULT_MARGIN_H_RATIO))
        margin_w_ratio = float(cfg.get("margin_w_ratio", _DEFAULT_MARGIN_W_RATIO))
        num_prompt = int(cfg.get("num_prompt_points", 3))
        min_prompt = int(cfg.get("min_num_prompt_points", 2))
        use_lsv = bool(cfg.get("use_lsv", True))
        use_gsv = bool(cfg.get("use_gsv", True))
        min_random_init_steps = int(cfg.get("min_random_init_steps", 2))
        img_w = int(cfg.get("img_width", 640))
        img_h = int(cfg.get("img_height", 480))
        output_dir = str(cfg.get("output_dir") or _DEFAULT_OUTPUT_DIR)

        episode_id = str(inputs.get("episode_id", "") or "0")
        rgb = np.asarray(inputs.get("rgb"))
        depth = np.asarray(inputs.get("depth"))
        cam_pose = np.asarray(inputs.get("cam_pose"), dtype=np.float64)
        cam_intr = np.asarray(inputs.get("cam_intr"), dtype=np.float64)
        pose_normal = np.asarray(inputs.get("pose_normal"), dtype=np.float64)
        angle = float(inputs.get("angle", 0.0) or 0.0)
        float(inputs.get("floor_height", 0.0) or 0.0)
        tsdf_bnds = inputs.get("tsdf_bnds")
        step_index = int(inputs.get("step_index", 0) or 0)
        vlm_question = str(inputs.get("vlm_question", "") or "")
        command = str(inputs.get("direction", "") or "")
        qwen_url = str(inputs.get("qwen_url", "") or "")
        env_hmeqa_url = str(inputs.get("env_hmeqa_url", "") or "")

        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if depth.ndim == 3:
            depth = depth.squeeze()

        if not env_hmeqa_url:
            self._self_log("error", "env_hmeqa_url not provided")
            return self._passthrough(rgb, depth, pose_normal, angle, cam_pose, cam_intr, step_index)

        ep_dir = os.path.join(output_dir, episode_id)
        os.makedirs(ep_dir, exist_ok=True)

        # ── 1. Build/lookup planner; integrate current frame ──
        planner = _get_or_build_planner(
            episode_id, tsdf_bnds, pose_normal, voxel_size, init_clearance
        )
        margin_h = int(margin_h_ratio * img_h)
        margin_w = int(margin_w_ratio * img_w)
        try:
            planner.integrate(
                color_im=rgb.astype(np.uint8),
                depth_im=depth.astype(np.float32),
                cam_intr=cam_intr,
                cam_pose=cam_pose,
                obs_weight=1.0,
                margin_h=margin_h,
                margin_w=margin_w,
            )
        except Exception as exc:
            log.exception("TSDF integrate failed")
            self._self_log("error", f"integrate: {exc}")

        # ── 2. Frontier candidates + LSV/GSV VLM scoring ──
        try:
            candidates_pix = planner.find_prompt_points_within_view(
                pose_normal, img_w, img_h, cam_intr, cam_pose, num_prompt_points=num_prompt
            )
        except Exception as exc:
            log.exception("find_prompt_points_within_view failed")
            self._self_log("error", f"frontier: {exc}")
            candidates_pix = []

        actual_n = len(candidates_pix)
        self._self_log("num_candidates", actual_n)
        if actual_n >= min_prompt and qwen_url:
            self._score_and_integrate(
                planner=planner,
                rgb=rgb,
                candidates_pix=candidates_pix,
                actual_n=actual_n,
                vlm_question=vlm_question,
                command=command,
                use_lsv=use_lsv,
                use_gsv=use_gsv,
                qwen_url=qwen_url,
                ep_dir=ep_dir,
            )

        # ── 3. Frontier-weighted next pose ──
        try:
            next_point_normal, next_yaw, _ = planner.find_next_pose(
                pts=pose_normal,
                angle=angle,
                flag_no_val_weight=step_index < min_random_init_steps,
            )
            nx, ny = float(next_point_normal[0]), float(next_point_normal[1])
            nyaw = float(next_yaw)
        except Exception as exc:
            log.exception("find_next_pose failed")
            self._self_log("error", f"find_next_pose: {exc}")
            nx, ny, nyaw = float(pose_normal[0]), float(pose_normal[1]), angle + 0.1

        # ── 4. Teleport via env_hmeqa__step ──
        action = json.dumps({"position_normal": [nx, ny], "angle": nyaw})
        try:
            out = _http_call(
                env_hmeqa_url,
                "env_hmeqa__step",
                {"action": serialize_value(action, "TEXT")},
                {},
                timeout=120.0,
            )
        except Exception as exc:
            log.exception("env_hmeqa__step failed")
            self._self_log("error", f"env_step: {exc}")
            return self._passthrough(rgb, depth, pose_normal, angle, cam_pose, cam_intr, step_index)

        new_rgb = deserialize_value(out.get("rgb"), "IMAGE")
        new_depth = deserialize_value(out.get("depth"), "ANY")
        new_pose_normal = deserialize_value(out.get("pose_normal"), "ANY")
        new_angle = deserialize_value(out.get("angle"), "ANY")
        new_cam_pose = deserialize_value(out.get("cam_pose_matrix"), "ANY")
        new_step_index = deserialize_value(out.get("step_index"), "ANY")
        done = bool(deserialize_value(out.get("done"), "BOOL"))

        # ── 5. Persist the new RGB to disk for the backend's file-path tools ──
        new_step = int(new_step_index) if new_step_index is not None else step_index + 1
        rgb_path = os.path.join(ep_dir, f"next_point_{new_step}.jpg")
        try:
            # PIL, not cv2 — the hmeqa env ships PIL but not opencv. PIL takes
            # RGB directly (no BGR swap).
            from PIL import Image

            Image.fromarray(np.asarray(new_rgb).astype(np.uint8)).convert("RGB").save(rgb_path)
        except Exception as exc:
            self._self_log("error", f"rgb save: {exc}")
            rgb_path = ""

        self._self_log("step_index", new_step)
        self._self_log("done", done)
        return {
            "rgb": new_rgb,
            "depth": new_depth,
            "pose_normal": new_pose_normal,
            "angle": new_angle,
            "cam_pose": new_cam_pose,
            "cam_intr": cam_intr,
            "step_index": new_step,
            "done": done,
            "rgb_path": os.path.abspath(rgb_path) if rgb_path else "",
        }

    def _score_and_integrate(
        self,
        *,
        planner,
        rgb,
        candidates_pix,
        actual_n,
        vlm_question,
        command,
        use_lsv,
        use_gsv,
        qwen_url,
        ep_dir,
    ) -> None:
        """LSV + GSV VLM scoring → integrate_sem. Verbatim logic from upstream
        go_next_point (Qwen-generate one-hot, not score_tokens softmax)."""
        from PIL import ImageDraw, ImageFont

        pil = _to_pil_rgb(rgb)
        if pil is None:
            return

        # Draw A/B/C/D labels on the frontier candidates.
        pil_draw = pil.copy()
        draw = ImageDraw.Draw(pil_draw)
        try:
            font = ImageFont.truetype(_DEFAULT_FONT_PATH, _FONT_SIZE)
        except Exception:
            font = ImageFont.load_default()
        for i, pt in enumerate(candidates_pix[: len(_DRAW_LETTERS)]):
            px, py = int(pt[0]), int(pt[1])
            draw.ellipse(
                (
                    px - _CIRCLE_RADIUS,
                    py - _CIRCLE_RADIUS,
                    px + _CIRCLE_RADIUS,
                    py + _CIRCLE_RADIUS,
                ),
                fill=(200, 200, 200, 255),
                outline=(0, 0, 0, 255),
                width=3,
            )
            draw.text((px, py), _DRAW_LETTERS[i], font=font, fill=(0, 0, 0, 255), anchor="mm")
        draw_path = os.path.join(ep_dir, "frontier_draw.png")
        base_path = os.path.join(ep_dir, "frontier_base.png")
        pil_draw.save(draw_path)
        pil.save(base_path)

        # LSV — pick a labelled direction.
        if use_lsv:
            proposal = _DRAW_LETTERS[:actual_n]
            direction = command.split("_")[-1] if command else ""
            prompt_lsv = (
                f"\nConsider the question: '{vlm_question}', and you will explore {direction} "
                f"the environment for answering it.\nWhich direction (black letters on the "
                f"image {proposal}) would you explore then? Answer with a single letter."
            )
            resp = _qwen_generate(qwen_url, prompt_lsv, draw_path)
            lsv = np.zeros(actual_n)
            idx = _first_letter(resp, proposal)
            if idx >= 0:
                lsv[idx] = 1
            lsv = lsv * (actual_n / 3.0)
        else:
            lsv = np.ones(actual_n) / actual_n

        # GSV — is any direction worth exploring (global value).
        if use_gsv:
            prompt_gsv = (
                f"\nConsider the question: '{vlm_question}', and you will explore the "
                f"environment for answering it. Is there any direction shown in the image "
                f"worth exploring? Answer with Yes or No."
            )
            resp = _qwen_generate(qwen_url, prompt_gsv, base_path)
            yes = 1.0 if resp.strip().strip(".").lower().startswith("yes") else 0.0
            gsv = float(np.exp(yes / _GSV_T) / _GSV_F)
        else:
            gsv = 1.0

        sv = (np.asarray(lsv) * gsv).astype(np.float64)
        self._self_log("lsv", [float(x) for x in lsv.tolist()])
        self._self_log("gsv", gsv)
        try:
            planner.integrate_sem(sem_pix=sv, radius=1.0, obs_weight=1.0)
        except Exception as exc:
            log.exception("integrate_sem failed")
            self._self_log("error", f"integrate_sem: {exc}")

    @staticmethod
    def _passthrough(rgb, depth, pose_normal, angle, cam_pose, cam_intr, step_index) -> dict:
        """Failure fallback: return the unchanged frame, done=False so the
        loop can recover next iter."""
        return {
            "rgb": rgb,
            "depth": depth,
            "pose_normal": pose_normal,
            "angle": angle,
            "cam_pose": cam_pose,
            "cam_intr": cam_intr,
            "step_index": step_index,
            "done": False,
            "rgb_path": "",
        }


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class ToolEQAExploreNodeSet(BaseNodeSet):
    """ToolEQA exploration nodeset — the fused go_next frontier step.

    Server mode under the dedicated ``hmeqa`` env (numba TSDF). Calls the
    Qwen VLM and ``env_hmeqa__step`` over HTTP (URLs passed in as inputs by
    the backend ``tooleqa`` reasoner).
    """

    name: ClassVar[str] = "tooleqa_explore"
    description: ClassVar[str] = (
        "ToolEQA go_next — TSDF frontier step + VLM (LSV/GSV) scoring + teleport"
    )
    server_python: ClassVar[str] = os.environ.get(
        "HMEQA_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-hmeqa/bin/python")
    )

    def get_tools(self) -> list:
        return [GoNextNode()]

    async def shutdown(self) -> None:
        _TSDF_PLANNERS.clear()
