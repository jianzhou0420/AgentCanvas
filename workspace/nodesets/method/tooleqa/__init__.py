from __future__ import annotations

"""ToolEQA reasoner nodeset (method-side, backend env).

One canvas iter = one ReAct step. The `tooleqa__step` node runs the
upstream ReAct loop body in-process (the backend env ships
`transformers.agents` 4.45.2) and dispatches every heavy operation over
HTTP to a sibling server nodeset:

    1. Saves the current rgb to disk under `<output_dir>/<episode_id>/`
       so the file-path-based tools (matching upstream's contract) resolve.
    2. Builds a `ToolEQAToolbox` snapshot bound to that image path + closures
       for the VLM (Qwen2.5-VL), DetAny3D 2D/3D detection, and go_next.
    3. Renders the prompt — system_prompt (verbatim) + tool descriptions +
       the running Thought/Code/Observation scratchpad — and calls Qwen.
    4. Parses the code blob via `transformers.agents.agents.parse_code_blob`.
    5. Runs it through `transformers.agents.python_interpreter.evaluate_python_code`
       with the toolbox tools as `static_tools` and `LIST_SAFE_MODULES`
       as `authorized_imports` — the verbatim faithfulness path.
    6. Drains the toolbox buffers. GoNextPointTool has, in-tool, already
       stepped the env (full TSDF frontier step + teleport via the hmeqa-side
       `tooleqa_explore__go_next`) and stashed the next observation in
       `pending_frame`; FinalAnswerTool stashed `pending_answer`.
    7. Appends a Thought/Code/Observation entry to the scratchpad and emits
       the next working frame (rgb/depth/pose/...) for the next iter.

Cross-nodeset wiring (resolved by THIS backend node, which has the
component registry, and passed into the closures / go_next inputs):

    vlm_qwen2_5_vl   — ReAct LLM + VisualQATool   (/call/vlm_qwen2_5_vl__generate)
    model_detany3d     — ObjectLocation2D / 3D       (/call/model_detany3d__locate_2d|3d)
    tooleqa_explore  — GoNextPointTool             (/call/tooleqa_explore__go_next)
    env_hmeqa        — stepped *inside* go_next     (URL forwarded to tooleqa_explore)

Unlike the first (deleted) port, go_next is NOT discrete dead-reckoning to
a downstream `env_hmeqa__step` canvas node — it is the faithful Explore-EQA
frontier step, so `env_hmeqa__step` no longer appears in the graph loop.

Vendor: tools/prompt verbatim from the ToolEQA release. SegmentInstance is
intentionally absent (commented out upstream). Requires `transformers`
with `transformers.agents` (4.40-4.45; removed thereafter -> `smolagents`).

last updated: 2026-06-08
"""

import json
import logging
import os
import re
import time
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

from ._prompts import load_system_prompt
from ._toolbox import ToolEQAToolbox

log = logging.getLogger("agentcanvas.tooleqa")


# Per upstream tool_eqa_agent.py — agent max_iterations = min(60, env.max_step).
_MAX_ITER_HARD_CAP = 60

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "tooleqa_runs")


# ══════════════════════════════════════════════════════════════════════
# HTTP helpers (backend → sibling server nodesets)
# ══════════════════════════════════════════════════════════════════════


def _resolve_server_url(ctx: Any, name: str) -> str:
    """Resolve a sibling nodeset's server URL via the executor override map
    (per-worker) then the global registry. Empty string if not loaded."""
    url = None
    executor = getattr(ctx, "_executor", None)
    if executor is not None:
        try:
            url = executor.get_server_url(name)
        except Exception:
            url = None
    if not url:
        try:
            from app.state import get_services

            url = get_services().workspace_component_registry.get_server_url(name)
        except Exception:
            url = None
    return url or ""


async def _call_server(url: str, function_name: str, inputs_payload: dict, config: dict) -> dict:
    """POST to ``{url}/call/{function_name}``; return the raw outputs dict."""
    import httpx

    from app.server._loopback_proxy import loopback_httpx_kwargs

    call_url = "{}/call/{}".format(url.rstrip("/"), function_name)
    async with httpx.AsyncClient(timeout=180.0, **loopback_httpx_kwargs()) as client:
        resp = await client.post(call_url, json={"inputs": inputs_payload, "config": config})
        resp.raise_for_status()
        data = resp.json()
    return data.get("outputs", data)


def _save_rgb_to_disk(rgb: np.ndarray, output_dir: str, sample_id: str, step_idx: int) -> str:
    """Save the current rgb so file-path tools can resolve it. Mirrors
    upstream GoNextPointTool's per-step naming."""
    import cv2

    sub = os.path.join(output_dir, sample_id)
    os.makedirs(sub, exist_ok=True)
    name = "init_rgb.jpg" if step_idx == 0 else f"next_point_{step_idx}.jpg"
    path = os.path.join(sub, name)
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    cv2.imwrite(path, cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return os.path.abspath(path)


# ══════════════════════════════════════════════════════════════════════
# Step node
# ══════════════════════════════════════════════════════════════════════


class ToolEQAStepNode(BaseCanvasNode):
    """One ReAct step — Qwen call + Python eval + toolbox buffer drain."""

    node_type: ClassVar[str] = "tooleqa__step"
    display_name: ClassVar[str] = "ToolEQA Step"
    description: ClassVar[str] = (
        "One ReAct step: build prompt → Qwen → parse code → eval with toolbox "
        "(VQA / DetAny3D / go_next over HTTP) → drain frame/answer buffer."
    )
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Brain"

    input_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "Current RGB observation"),
        PortDef("depth", "ANY", "Current metric depth (lossless ANY wire)"),
        PortDef("pose_normal", "ANY", "Current 3-vector normal-frame position"),
        PortDef("angle", "ANY", "Current yaw (radians)"),
        PortDef("cam_pose", "ANY", "4x4 TSDF-frame camera extrinsic"),
        PortDef("cam_intr", "ANY", "3x3 camera intrinsics"),
        PortDef("floor_height", "ANY", "Floor z (episode constant)", optional=True),
        PortDef("tsdf_bnds", "ANY", "3x2 TSDF bounds (first go_next only)", optional=True),
        PortDef("question", "TEXT", "EQA question (raw, no A/B/C/D tail)"),
        PortDef("choices", "ANY", "Multi-choice options [A,B,C,D]", optional=True),
        PortDef("scratchpad", "TEXT", "Running ReAct transcript (empty at iter 0)", optional=True),
        PortDef("agent_state", "ANY", "Agent state dict (print_outputs)", optional=True),
        PortDef("step_index", "ANY", "Current step index", optional=True),
        PortDef("max_step", "ANY", "Per-episode step cap (from env on_load)", optional=True),
        PortDef("episode_id", "TEXT", "Episode id", optional=True),
        PortDef("answer_gt", "TEXT", "Ground-truth letter (for self-evaluation)", optional=True),
    ]
    output_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "Working RGB for next iter (post-go_next or unchanged)"),
        PortDef("depth", "ANY", "Working depth for next iter"),
        PortDef("pose_normal", "ANY", "Working pose for next iter"),
        PortDef("angle", "ANY", "Working yaw for next iter"),
        PortDef("cam_pose", "ANY", "Working cam extrinsic for next iter"),
        PortDef("cam_intr", "ANY", "Pass-through intrinsics"),
        PortDef("step_index", "ANY", "Updated step index"),
        PortDef("final_answer", "TEXT", "Final answer letter (fires only at episode end)"),
        PortDef("success", "BOOL", "final_answer == answer_gt (fires only at episode end)"),
        PortDef("metrics", "METRICS", "{success, num_steps} (fires only at episode end)"),
        PortDef("done", "BOOL", "True on final_answer or max_iter"),
        PortDef("next_scratchpad", "TEXT", "Updated Thought/Code/Observation transcript"),
        PortDef("next_agent_state", "ANY", "Updated agent state"),
        PortDef("rgb_saved_path", "TEXT", "Absolute path of the saved working rgb"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("output_dir", "text", label="Output dir", default=_DEFAULT_OUTPUT_DIR),
            ConfigField(
                "max_iterations",
                "slider",
                label="Max iterations",
                default=_MAX_ITER_HARD_CAP,
                min=10,
                max=120,
                step=5,
            ),
            ConfigField(
                "qwen_temperature",
                "slider",
                label="Qwen temperature",
                default=0.7,
                min=0.0,
                max=1.5,
                step=0.1,
            ),
            ConfigField(
                "qwen_max_tokens",
                "slider",
                label="Qwen max tokens",
                default=2048,
                min=512,
                max=8192,
                step=256,
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        cfg = self.config or {}
        output_dir = str(cfg.get("output_dir") or _DEFAULT_OUTPUT_DIR)
        max_iter_cfg = int(cfg.get("max_iterations") or _MAX_ITER_HARD_CAP)
        temperature = float(cfg.get("qwen_temperature", 0.7))
        max_tokens = int(cfg.get("qwen_max_tokens", 2048))

        # ── 1. Resolve the working frame + episode meta ──
        rgb = inputs.get("rgb")
        depth = inputs.get("depth")
        pose_normal = inputs.get("pose_normal")
        if pose_normal is None:
            pose_normal = [0.0, 0.0, 0.0]
        angle = float(inputs.get("angle") or 0.0)
        cam_pose = inputs.get("cam_pose")
        cam_intr = inputs.get("cam_intr")
        floor_height = float(inputs.get("floor_height") or 0.0)
        tsdf_bnds = inputs.get("tsdf_bnds")
        question = str(inputs.get("question", "")).strip()
        choices_raw = inputs.get("choices")
        choices = list(choices_raw) if isinstance(choices_raw, (list, tuple)) else []
        scratchpad = str(inputs.get("scratchpad", "") or "")
        agent_state = inputs.get("agent_state") or {"print_outputs": ""}
        if not isinstance(agent_state, dict):
            agent_state = {"print_outputs": ""}
        step_index = int(inputs.get("step_index") or 0)
        max_step = int(inputs.get("max_step") or max_iter_cfg) or max_iter_cfg
        episode_id = str(inputs.get("episode_id", "") or f"ep_{int(time.time())}")
        answer_gt = str(inputs.get("answer_gt", "") or "").strip().upper()
        vlm_question = _build_vlm_question(question, choices)

        if rgb is None or not isinstance(rgb, np.ndarray):
            self._self_log("error", "no rgb input — emitting no-op")
            return _empty_step_output(inputs, scratchpad, agent_state)

        # ── 2. Resolve sibling server URLs ──
        qwen_url = _resolve_server_url(ctx, "vlm_qwen2_5_vl")
        detany3d_url = _resolve_server_url(ctx, "model_detany3d")
        go_next_url = _resolve_server_url(ctx, "tooleqa_explore")
        env_hmeqa_url = _resolve_server_url(ctx, "env_hmeqa")
        if not qwen_url:
            self._self_log("error", "vlm_qwen2_5_vl not loaded")
            return _empty_step_output(inputs, scratchpad, agent_state, error="no Qwen server")

        # ── 3. Save rgb so file-path tools resolve it ──
        rgb_path = _save_rgb_to_disk(rgb, output_dir, episode_id, step_index)

        # ── 4. Closures: Qwen VLM + DetAny3D + go_next (all HTTP) ──
        async def _qwen_async(prompt: str, image_paths: list[str], max_new: int) -> str:
            payload = {
                "prompt": serialize_value(prompt, "TEXT"),
                "image_paths": serialize_value(list(image_paths), "ANY"),
            }
            out = await _call_server(
                qwen_url,
                "vlm_qwen2_5_vl__generate",
                payload,
                {"max_new_tokens": max_new, "temperature": temperature},
            )
            return str(deserialize_value(out.get("text"), "TEXT") or "")

        import asyncio

        loop = asyncio.get_event_loop()

        def _vlm_call_sync(question_text: str, image_paths: list[str]) -> str:
            # Called synchronously inside evaluate_python_code (which runs in a
            # thread executor); drive the coroutine on the main loop.
            fut = asyncio.run_coroutine_threadsafe(
                _qwen_async(question_text, image_paths, max_tokens), loop
            )
            return fut.result(timeout=180.0)

        async def _detany3d_async(endpoint: str, image: np.ndarray, text: str) -> dict:
            if not detany3d_url:
                return {"error": "model_detany3d not loaded"}
            payload = {
                "image": serialize_value(image, "ANY"),
                "text": serialize_value(text, "TEXT"),
            }
            try:
                out = await _call_server(detany3d_url, endpoint, payload, {})
            except Exception as e:
                return {"error": f"detany3d HTTP error: {e}"}
            return {k: deserialize_value(v, "ANY") for k, v in out.items()}

        def _detany3d_sync(endpoint: str, image: np.ndarray, text: str) -> dict:
            fut = asyncio.run_coroutine_threadsafe(_detany3d_async(endpoint, image, text), loop)
            return fut.result(timeout=180.0)

        def _locate_2d(image: np.ndarray, text: str) -> dict:
            return _detany3d_sync("model_detany3d__locate_2d", image, text)

        def _locate_3d(image: np.ndarray, text: str) -> dict:
            return _detany3d_sync("model_detany3d__locate_3d", image, text)

        # go_next: the toolbox passes the *latest* working frame each call. We
        # capture the live frame via a mutable holder so a second go_next in
        # one code block integrates from the post-teleport view.
        frame_state = {
            "rgb": rgb,
            "depth": depth,
            "pose_normal": pose_normal,
            "angle": angle,
            "cam_pose": cam_pose,
            "cam_intr": cam_intr,
            "step_index": step_index,
        }

        async def _go_next_async(direction: str) -> dict:
            if not go_next_url:
                return {"rgb_path": "", "error": "tooleqa_explore not loaded"}
            f = frame_state
            payload = {
                "rgb": serialize_value(f["rgb"], "IMAGE"),
                "depth": serialize_value(f["depth"], "ANY"),
                "cam_pose": serialize_value(f["cam_pose"], "ANY"),
                "cam_intr": serialize_value(f["cam_intr"], "ANY"),
                "pose_normal": serialize_value(f["pose_normal"], "ANY"),
                "angle": serialize_value(f["angle"], "ANY"),
                "floor_height": serialize_value(floor_height, "ANY"),
                "tsdf_bnds": serialize_value(tsdf_bnds, "ANY"),
                "episode_id": serialize_value(episode_id, "TEXT"),
                "step_index": serialize_value(f["step_index"], "ANY"),
                "vlm_question": serialize_value(vlm_question, "TEXT"),
                "direction": serialize_value(direction, "TEXT"),
                "qwen_url": serialize_value(qwen_url, "TEXT"),
                "env_hmeqa_url": serialize_value(env_hmeqa_url, "TEXT"),
            }
            out = await _call_server(go_next_url, "tooleqa_explore__go_next", payload, dict(cfg))
            frame = {
                "rgb": deserialize_value(out.get("rgb"), "IMAGE"),
                "depth": deserialize_value(out.get("depth"), "ANY"),
                "pose_normal": deserialize_value(out.get("pose_normal"), "ANY"),
                "angle": deserialize_value(out.get("angle"), "ANY"),
                "cam_pose": deserialize_value(out.get("cam_pose"), "ANY"),
                "cam_intr": deserialize_value(out.get("cam_intr"), "ANY"),
                "step_index": deserialize_value(out.get("step_index"), "ANY"),
                "done": bool(deserialize_value(out.get("done"), "BOOL")),
                "rgb_path": str(deserialize_value(out.get("rgb_path"), "TEXT") or ""),
            }
            # Advance the live frame so a later go_next in the same block sees it.
            for k in ("rgb", "depth", "pose_normal", "angle", "cam_pose", "cam_intr", "step_index"):
                if frame.get(k) is not None:
                    frame_state[k] = frame[k]
            return frame

        def _go_next_sync(direction: str) -> dict:
            fut = asyncio.run_coroutine_threadsafe(_go_next_async(direction), loop)
            return fut.result(timeout=300.0)

        # ── 5. Build toolbox snapshot ──
        toolbox = ToolEQAToolbox(
            rgb_path=rgb_path,
            output_dir=output_dir,
            sample_id=episode_id,
            vlm_call=_vlm_call_sync,
            detany3d_locate_2d=_locate_2d,
            detany3d_locate_3d=_locate_3d,
            go_next_call=_go_next_sync,
        )

        # ── 6. Build prompt + Qwen call (the ReAct LLM) ──
        prompt_text = _build_prompt(
            scratchpad=scratchpad,
            question=question,
            choices=choices,
            toolbox=toolbox,
            authorized_imports=_get_authorized_imports(),
        )
        try:
            llm_output = await _qwen_async(prompt_text, [rgb_path], max_tokens)
        except Exception as e:
            self._self_log("error", f"Qwen call failed: {e}")
            return _empty_step_output(inputs, scratchpad, agent_state, error=str(e))
        if not llm_output:
            self._self_log("error", "Qwen returned empty output")
            return _empty_step_output(inputs, scratchpad, agent_state, error="Qwen empty output")
        for stop in ("<end_action>", "Observation:"):
            idx = llm_output.find(stop)
            if idx != -1:
                llm_output = llm_output[:idx]
        self._self_log("llm_output_len", len(llm_output))

        # ── 7. Parse + eval ──
        try:
            from transformers.agents.agents import parse_code_blob  # type: ignore[import-not-found]
            from transformers.agents.python_interpreter import (  # type: ignore[import-not-found]
                LIST_SAFE_MODULES,
                evaluate_python_code,
            )

            try:
                from transformers.agents.default_tools import (
                    BASE_PYTHON_TOOLS,  # type: ignore[import-not-found]
                )
            except ImportError:
                from transformers.agents.agents import (
                    BASE_PYTHON_TOOLS,  # type: ignore[import-not-found]
                )
        except ImportError as e:
            self._self_log("error", f"transformers.agents unavailable: {e}")
            return _empty_step_output(inputs, scratchpad, agent_state, error=str(e))

        rationale, code_str = _split_thought_code(llm_output)
        try:
            code_action = parse_code_blob(code_str)
        except Exception as e:
            self._self_log("error", f"parse_code_blob failed: {e}")
            next_scratchpad = _append_to_scratchpad(
                scratchpad,
                rationale,
                code_str,
                f"[parse error] {e}\nRemember to wrap code in ```py ... ``` markers.",
            )
            return _passthrough_step_output(
                inputs, frame_state, next_scratchpad, agent_state, done=False
            )

        try:
            tools_dict = toolbox.tools_dict()
            await loop.run_in_executor(
                None,
                lambda: evaluate_python_code(
                    code_action,
                    static_tools={**dict(BASE_PYTHON_TOOLS), **tools_dict},
                    custom_tools={},
                    state=agent_state,
                    authorized_imports=list(LIST_SAFE_MODULES),
                ),
            )
            print_outputs = agent_state.get("print_outputs", "") or ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            self._self_log("error", f"evaluate_python_code failed: {err}")
            print_outputs = f"[exec error] {err}"

        # ── 8. Drain buffers ──
        raw_answer = toolbox.pending_answer or ""
        final_answer = _extract_letter(raw_answer) if raw_answer else ""
        if raw_answer and not final_answer and choices:
            try:
                final_answer = await _letterize_via_qwen(_qwen_async, raw_answer, question, choices)
            except Exception as e:
                self._self_log("error", f"letterize failed: {e}")
        done = bool(toolbox.pending_done)

        # New working frame: go_next's pending_frame if it fired, else unchanged.
        pf = toolbox.pending_frame
        out_frame = dict(frame_state)
        if isinstance(pf, dict):
            for k in ("rgb", "depth", "pose_normal", "angle", "cam_pose", "cam_intr", "step_index"):
                if pf.get(k) is not None:
                    out_frame[k] = pf[k]
            done = done or bool(pf.get("done"))

        next_scratchpad = _append_to_scratchpad(scratchpad, rationale, code_action, print_outputs)

        new_step_index = int(out_frame.get("step_index") or step_index)
        if new_step_index >= min(max_iter_cfg, max_step):
            done = True
            if not final_answer:
                final_answer = "A"
                self._self_log("warn", "max_iter without final_answer; defaulting to A")

        new_rgb_path = (
            _save_rgb_to_disk(np.asarray(out_frame["rgb"]), output_dir, episode_id, new_step_index)
            if isinstance(out_frame.get("rgb"), np.ndarray)
            else rgb_path
        )

        self._self_log("step_index", new_step_index)
        self._self_log("final_answer", final_answer)
        self._self_log("done", done)
        result = {
            "rgb": out_frame.get("rgb"),
            "depth": out_frame.get("depth"),
            "pose_normal": out_frame.get("pose_normal"),
            "angle": out_frame.get("angle"),
            "cam_pose": out_frame.get("cam_pose"),
            "cam_intr": out_frame.get("cam_intr"),
            "step_index": new_step_index,
            "done": done,
            "next_scratchpad": next_scratchpad,
            "next_agent_state": agent_state,
            "rgb_saved_path": new_rgb_path,
        }
        # Fire `final_answer` / `success` / `metrics` only at episode end. We
        # self-evaluate here (like explore_eqa's aggregate_answer) rather than
        # wiring a post_loop env_hmeqa__evaluate fed by this node's output —
        # that node's input would resolve to None on the post-loop re-fire
        # (project_final_fire_upstream_dep). graphOut fed by this node fires
        # fine because it captures the last in-loop value.
        if done and final_answer:
            success = bool(answer_gt) and final_answer == answer_gt
            result["final_answer"] = final_answer
            result["success"] = success
            result["metrics"] = {
                "success": float(success),
                "num_steps": float(new_step_index),
            }
        return result


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _empty_step_output(inputs: dict, scratchpad: str, agent_state: Any, error: str = "") -> dict:
    """No-op step on input errors / missing config — terminate the episode."""
    return {
        "rgb": inputs.get("rgb"),
        "depth": inputs.get("depth"),
        "pose_normal": inputs.get("pose_normal"),
        "angle": inputs.get("angle"),
        "cam_pose": inputs.get("cam_pose"),
        "cam_intr": inputs.get("cam_intr"),
        "step_index": inputs.get("step_index") or 0,
        "final_answer": "",
        "done": True,
        "next_scratchpad": scratchpad + (f"\n[Error] {error}" if error else ""),
        "next_agent_state": agent_state,
        "rgb_saved_path": "",
    }


def _passthrough_step_output(
    inputs: dict, frame_state: dict, scratchpad: str, agent_state: Any, done: bool
) -> dict:
    """Continue the loop with the unchanged frame (e.g. after a parse error)."""
    return {
        "rgb": frame_state.get("rgb"),
        "depth": frame_state.get("depth"),
        "pose_normal": frame_state.get("pose_normal"),
        "angle": frame_state.get("angle"),
        "cam_pose": frame_state.get("cam_pose"),
        "cam_intr": frame_state.get("cam_intr"),
        "step_index": frame_state.get("step_index") or 0,
        "final_answer": "",
        "done": done,
        "next_scratchpad": scratchpad,
        "next_agent_state": agent_state,
        "rgb_saved_path": "",
    }


def _build_vlm_question(question: str, choices: list[str]) -> str:
    """Format Q + A/B/C/D tail for the frontier LSV/GSV prompts."""
    vlm_q = str(question)
    for letter, choice in zip(["A", "B", "C", "D"], choices):  # noqa: B905
        vlm_q += f"\n{letter}. {choice}"
    return vlm_q


async def _letterize_via_qwen(
    qwen_async, raw_answer: str, question: str, choices: list[str]
) -> str:
    """Map a free-text answer to A/B/C/D via a single text-only Qwen call.
    Mirrors upstream's vlm_pred_candidates post-processing."""
    letters = ["A", "B", "C", "D"]
    options = "\n".join(f"{lt}. {c}" for lt, c in zip(letters, choices))  # noqa: B905
    prompt = (
        f"Question: {question}\n\nOptions:\n{options}\n\n"
        f'An agent gave this free-text answer: "{raw_answer}"\n\n'
        f"Map the agent's answer to one of the options above. Respond with "
        f"exactly one letter (A, B, C, or D) and nothing else."
    )
    resp = await qwen_async(prompt, [], 8)
    return _extract_letter(resp or "")


def _extract_letter(answer: str) -> str:
    """Extract a single A/B/C/D letter from a final_answer string."""
    if not answer:
        return ""
    up = answer.strip().upper()
    if up in ("A", "B", "C", "D"):
        return up
    m = re.match(r"^\s*([A-D])\b", up)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", up)
    if m:
        return m.group(1)
    return ""


def _split_thought_code(llm_output: str) -> tuple[str, str]:
    """Split on the 'Code:' marker; return (rationale, raw_code)."""
    if "Code:" in llm_output:
        rationale, _, code = llm_output.partition("Code:")
        return rationale.strip(), code.strip()
    return llm_output, llm_output


def _append_to_scratchpad(scratchpad: str, rationale: str, code: str, observation: str) -> str:
    """Append a Thought/Code/Observation triple to the running transcript."""
    block = (
        "Thought: " + rationale.strip() + "\n"
        "Code: \n```py\n" + code.strip() + "\n```<end_action>\n"
        "Observation: " + observation.strip()
    )
    return scratchpad + ("\n" if scratchpad else "") + block


def _build_prompt(
    *,
    scratchpad: str,
    question: str,
    choices: list[str],
    toolbox: ToolEQAToolbox,
    authorized_imports: list[str],
) -> str:
    """Assemble system_prompt + tool descriptions + scratchpad + task."""
    system_prompt = load_system_prompt()
    tool_descs = _format_tool_descriptions(toolbox)
    imports_str = ", ".join(repr(m) for m in authorized_imports)
    rendered = system_prompt.replace("<<tool_descriptions>>", tool_descs)
    rendered = rendered.replace("<<authorized_imports>>", imports_str)
    task = question
    if choices:
        for letter, choice in zip(["A", "B", "C", "D"], choices):  # noqa: B905
            task += f"\n{letter}. {choice}"
        task += (
            "\n\nWhen you have enough evidence, respond by calling "
            "`final_answer(answer='X')` where X is exactly one of A/B/C/D."
        )
    if scratchpad:
        return f"{rendered}\n\nTask: {task}\n\n{scratchpad}\nThought:"
    return f"{rendered}\n\nTask: {task}\n\nThought:"


def _format_tool_descriptions(toolbox: ToolEQAToolbox) -> str:
    """transformers.agents.tools.get_tool_description_with_args, verbatim."""
    try:
        from transformers.agents.tools import (
            get_tool_description_with_args,  # type: ignore[import-not-found]
        )
    except ImportError:
        return "(tool descriptions unavailable — transformers.agents not importable)"
    return "\n".join(get_tool_description_with_args(t) for t in toolbox.build_tools())


def _get_authorized_imports() -> list[str]:
    """LIST_SAFE_MODULES from transformers.agents — verbatim."""
    try:
        from transformers.agents.python_interpreter import (
            LIST_SAFE_MODULES,  # type: ignore[import-not-found]
        )

        return list(LIST_SAFE_MODULES)
    except ImportError:
        return []


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class ToolEQANodeSet(BaseNodeSet):
    name = "tooleqa"
    description = "ToolEQA — tool-augmented active explorer (Zhai et al. 2025) for HM-EQA"

    def get_tools(self) -> list:
        return [ToolEQAStepNode()]
