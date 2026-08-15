from __future__ import annotations

"""Habitat-free inference session for the JanusVLN Qwen3-VL-Omega current-3-view VLA.

Upstream ships its inference glue inside ``src/evaluation*.py``, but those modules
import ``habitat`` / ``habitat_sim`` / ``gym`` / ``quaternion`` at module scope even
though **the model itself never touches habitat** (verified: nothing under
``src/qwen_vl/`` or ``third_party/vggt_omega/`` imports it). Importing their eval
classes would therefore drag a whole simulator stack into this policy env — and
habitat-sim 0.3.3 only publishes py3.9 conda builds, which is exactly what makes
upstream's ``environment.yml`` unsolvable next to ``python=3.10``.

So this module re-implements the ~120 lines of glue against ``qwen_vl.model.*``
only. Every preprocessing step is a semantic copy of upstream so the model stays
in-distribution; the provenance of each is noted at its definition. AgentCanvas
already owns the simulator (``env_habitat``), so the policy has no business
carrying a second one.

Upstream reference (pinned):
    repo   XinyuYan/qwen3vl-omega-current3-vlm
    glue   src/evaluation_qwen3_vl_omega_current3_actionformer.py
           src/evaluation_qwen3_vl_omega_actionformer.py  (Qwen3VLOmegaActionFormerInference)
           src/evaluation_qwen3_vl_omega.py               (prompt / image helpers)
    class  Qwen3VLOmegaFramewiseDeepStackGatedCurrentForConditionalGeneration
    run    run_eval.sh --controller discrete --vlm_only
"""

import copy
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

log = logging.getLogger("agentcanvas.policy_vla_nav")

# Upstream's four primitives, in upstream's order. AgentCanvas' env_habitat
# discrete ids: 0=STOP 1=FORWARD 2=LEFT 3=RIGHT (mcp_bridge.py STEP_DESC_BASE).
ACTION_NAMES = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
ACTION_TO_ID = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}

DEFAULT_REPO = os.path.expanduser(
    "~/Desktop/Projects/DeepStack/qwen3vl-omega-current3-vlm"
)


def repo_root() -> str:
    return os.environ.get("VLA_NAV_REPO", DEFAULT_REPO)


def _ensure_sys_path(root: str) -> None:
    """Upstream's run_eval.sh exports these two on PYTHONPATH; mirror it."""
    for entry in (
        os.path.join(root, "src"),
        os.path.join(root, "third_party", "vggt_omega"),
    ):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ.setdefault(
        "VGGT_OMEGA_ROOT", os.path.join(root, "third_party", "vggt_omega")
    )
    # Read at model-construction time by the current-view mixin — must be set
    # BEFORE the class is instantiated (upstream sets it in __init__ before
    # super().__init__()).
    os.environ["OMEGA_CURRENT_VIEWS"] = "3"


# ══════════════════════════════════════════════════════════════════════
# Preprocessing — semantic copies of upstream (do not "improve" these:
# any drift here silently pushes the policy out of its training distribution)
# ══════════════════════════════════════════════════════════════════════


def _crop_to_supported_aspect_ratio(image, min_aspect_ratio=0.5, max_aspect_ratio=2.0):
    """evaluation_qwen3_vl_omega.py::_crop_to_supported_aspect_ratio"""
    width, height = image.size
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height))
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height))
    return image


def _balanced_target_shape(aspect_ratio, image_resolution, patch_size):
    """evaluation_qwen3_vl_omega.py::_balanced_target_shape"""
    token_number = (image_resolution // patch_size) ** 2
    w_patches = (token_number / aspect_ratio) ** 0.5
    h_patches = token_number / w_patches
    w_patches = max(1, int(round(w_patches)))
    h_patches = max(1, int(round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _encode_chat_text(tokenizer, role, content):
    """evaluation_qwen3_vl_omega.py::_encode_chat_text"""
    return tokenizer.encode(
        f"<|im_start|>{role}\n{content}<|im_end|>\n", add_special_tokens=False
    )


def _build_qwen3_prompt_ids(tokenizer, prompt, grid_tokens):
    """evaluation_qwen3_vl_omega.py::_build_qwen3_prompt_ids"""
    import torch

    input_ids = []
    input_ids.extend(
        _encode_chat_text(tokenizer, "system", "You are a helpful assistant.")
    )

    visual_replicate_index = 0
    parts = prompt.split("<image>")
    expanded = []
    for idx in range(len(parts) - 1):
        expanded.append(parts[idx])
        if visual_replicate_index >= len(grid_tokens):
            raise ValueError(
                "More <image> placeholders than visual inputs: "
                f"{visual_replicate_index + 1} > {len(grid_tokens)}"
            )
        expanded.append(
            "<|vision_start|>"
            + "<|image_pad|>" * int(grid_tokens[visual_replicate_index])
            + "<|vision_end|>"
        )
        visual_replicate_index += 1
    expanded.append(parts[-1])
    if visual_replicate_index != len(grid_tokens):
        raise ValueError(
            "Visual inputs and <image> placeholders do not match: "
            f"{len(grid_tokens)} vs {visual_replicate_index}"
        )

    input_ids.extend(_encode_chat_text(tokenizer, "user", "".join(expanded)))
    input_ids.extend(
        tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    )
    return torch.tensor([input_ids], dtype=torch.long)


def _astranav_prompt(instruction: str, history_count: int) -> str:
    """evaluation_qwen3_vl_omega_current3_actionformer.py::_astranav_prompt

    Byte-identical. The ``<image>`` order the prompt declares is
    ``history… , left, right, front`` — the pixel batch must match it exactly.
    """
    history_tags = "<image>" * int(history_count)
    return (
        "You are an autonomous navigation robot. You will get a task with "
        "historical pictures and current pictures you see.\n"
        "Based on this information, decide your next navigation action. "
        "The action must be one of MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.\n"
        f"# Your historical pictures are: {history_tags}\n"
        "# Your current observations is leftside: <image>, "
        "rightside: <image>, frontside: <image>\n"
        f"# Your mission is: {instruction}\n"
        "Output the next action."
    )


def _normalize_action(text: str) -> str:
    """evaluation_qwen3_vl_omega.py::_normalize_action"""
    text = text.strip().upper()
    for action in ACTION_NAMES:
        if action in text:
            return action
    return text


def _to_pil(value: Any):
    """Accept what AgentCanvas puts on the wire.

    ndarray (msgpack wire) · base64-encoded PNG/JPEG str (env_habitat's
    ``rgb_base64``, and the only image form that survives a plain-JSON POST) ·
    raw bytes · a file path · a nested list · PIL.
    """
    import base64
    import binascii
    import io

    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value.astype(np.uint8)).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        if os.path.isfile(value):
            return Image.open(value).convert("RGB")
        # data: URL or bare base64 — env_habitat emits bare base64 PNG.
        payload = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"string image input is neither a path nor base64 ({exc})"
            ) from exc
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(value, list):
        return Image.fromarray(np.asarray(value, dtype=np.uint8)).convert("RGB")
    raise ValueError(f"unsupported image input type: {type(value)!r}")


# ══════════════════════════════════════════════════════════════════════
# Session
# ══════════════════════════════════════════════════════════════════════


class VlaNavSession:
    """One policy on GPU + one episode's rolling front-view history.

    ``parallelism = "replicated"`` on the nodeset means one of these per eval
    worker, so the history needs no cross-episode locking beyond the executor.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="policy_vla_nav"
        )
        self._model: Any = None
        self._tokenizer: Any = None
        self._image_processor: Any = None
        self._device: Any = None
        self._cfg_cache: tuple | None = None
        self._loaded_meta: dict[str, Any] = {}

        # Per-episode state
        self._history: list = []          # PIL front views, oldest first
        self._instruction: str = ""
        self._step_id: int = 0

        # Knobs (upstream run_eval.sh defaults)
        self.num_history = 20
        self.omega_image_resolution = 512
        self.omega_preprocess_mode = "balanced_pad"
        self.history_resize_ratio = 0.25
        self.llm_max_new_tokens = 12

    # ── model lifecycle ───────────────────────────────────────────────

    def ensure_model(
        self,
        *,
        checkpoint_path: str,
        base_model_path: str,
        vggt_omega_model_path: str | None,
        device: str,
        attn_implementation: str,
    ) -> dict[str, Any]:
        """Idempotent load — short-circuits when the config is unchanged."""
        import torch
        from safetensors.torch import load_file
        from transformers import AutoImageProcessor, AutoTokenizer

        key = (
            checkpoint_path,
            base_model_path,
            vggt_omega_model_path,
            device,
            attn_implementation,
        )
        with self._lock:
            if key == self._cfg_cache and self._model is not None:
                return {"unchanged": True, **self._loaded_meta}

            _ensure_sys_path(repo_root())
            from qwen_vl.model.modeling_qwen3_vl_omega_wrapper_variants import (
                Qwen3VLOmegaFramewiseDeepStackGatedCurrentForConditionalGeneration as ModelCls,
            )

            state_path = os.path.join(checkpoint_path, "model.safetensors")
            if not os.path.isfile(state_path):
                raise FileNotFoundError(f"missing checkpoint weights: {state_path}")

            self._device = torch.device(device)
            log.info("loading VLA policy: %s (attn=%s)", checkpoint_path, attn_implementation)

            model = ModelCls.from_pretrained(
                pretrained_model_name_or_path=base_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                vggt_omega_model_path=vggt_omega_model_path or None,
                reference_frame="first",
                trust_remote_code=True,
            )
            # The published checkpoint carries ALL trained Qwen3-VL + VGGT-Omega
            # weights, so vggt_omega_model_path stays empty and VGGT arrives here.
            state_dict = load_file(state_path, device="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            del state_dict
            model.to(self._device).eval()
            model.base_model.eval()

            self._model = model
            self._tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path, padding_side="left", use_fast=False, trust_remote_code=True
            )
            self._image_processor = AutoImageProcessor.from_pretrained(
                checkpoint_path, trust_remote_code=True
            )
            self._cfg_cache = key
            self._loaded_meta = {
                "checkpoint": checkpoint_path,
                "base_model": base_model_path,
                "device": str(self._device),
                "attn_implementation": attn_implementation,
                "missing_keys": len(load_result.missing_keys),
                "unexpected_keys": len(load_result.unexpected_keys),
            }
            log.info("VLA policy loaded: %s", self._loaded_meta)
            return dict(self._loaded_meta)

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._image_processor = None
            self._cfg_cache = None
            self._history = []
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.executor.shutdown(wait=False)

    # ── episode state ─────────────────────────────────────────────────

    def reset(self, instruction: str = "") -> dict[str, Any]:
        self._history = []
        self._instruction = instruction or ""
        self._step_id = 0
        if self._model is not None:
            for attr in ("reset_timing_records",):
                fn = getattr(self._model, attr, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        log.debug("%s failed", attr, exc_info=True)
            inner = getattr(self._model, "model", None)
            fn = getattr(inner, "reset_gate_diagnostic_records", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    log.debug("reset_gate_diagnostic_records failed", exc_info=True)
        return {"ok": True, "instruction": self._instruction}

    # ── preprocessing ─────────────────────────────────────────────────

    def _history_resolution(self) -> int:
        """Qwen3VLOmegaCurrent3ActionFormerInference::_history_resolution"""
        patch_size = int(getattr(self._image_processor, "patch_size", 16))
        merge_size = int(
            getattr(
                self._image_processor,
                "merge_size",
                getattr(self._image_processor, "spatial_merge_size", 2),
            )
        )
        minimum = max(patch_size * merge_size, patch_size)
        return max(
            minimum,
            int(round(self.omega_image_resolution * self.history_resize_ratio)),
        )

    def _sample_history(self, images: list) -> list:
        """Current3ActionFormerVLNEvaluator::_sample_history — uniform over the
        whole episode (NOT a last-K window), so early decision points survive."""
        if len(images) <= self.num_history:
            return list(images)
        if self.num_history == 1:
            return [images[-1]]
        last = len(images) - 1
        indices = sorted(
            {
                int(round(idx * last / (self.num_history - 1)))
                for idx in range(self.num_history)
            }
        )
        return [images[idx] for idx in indices]

    def _process_image(self, image, image_resolution=None):
        """Qwen3VLOmegaInference::_process_image"""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms as TF

        image = _to_pil(image)
        image = _crop_to_supported_aspect_ratio(image)

        image_processor = copy.deepcopy(self._image_processor)
        patch_size = int(getattr(image_processor, "patch_size", 16))
        merge_size = int(
            getattr(
                image_processor,
                "merge_size",
                getattr(image_processor, "spatial_merge_size", 2),
            )
        )
        width, height = image.size
        image_resolution = (
            self.omega_image_resolution
            if image_resolution is None
            else int(image_resolution)
        )
        target_h, target_w = _balanced_target_shape(
            height / max(width, 1), image_resolution, patch_size
        )
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        images = TF.ToTensor()(image).unsqueeze(0)

        _, _, height, width = images.shape
        h_grid = height // patch_size
        w_grid = width // patch_size
        if self.omega_preprocess_mode == "balanced_pad":
            target_h = ((h_grid + merge_size - 1) // merge_size) * merge_size * patch_size
            target_w = ((w_grid + merge_size - 1) // merge_size) * merge_size * patch_size
            pad_h = target_h - height
            pad_w = target_w - width
            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        else:
            height = (h_grid // merge_size) * merge_size * patch_size
            width = (w_grid // merge_size) * merge_size * patch_size
            images = images[:, :, :height, :width].contiguous()
        image_vggt = images[0].clone()

        visual_processed = image_processor(images, return_tensors="pt", do_rescale=False)
        pixel_values = visual_processed["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = pixel_values[0]
        image_grid_thw = visual_processed["image_grid_thw"][0]
        grid_tokens = image_grid_thw.prod() // (merge_size**2)
        return pixel_values, image_grid_thw, image_vggt, grid_tokens

    # ── the one that matters ──────────────────────────────────────────

    def act(
        self,
        rgb_front: Any,
        rgb_left: Any,
        rgb_right: Any,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        """One policy step: 3 current views + rolling history → one primitive.

        Mirrors ``Qwen3VLOmegaCurrent3ActionFormerInference._build_inputs`` +
        ``call_model`` under ``--vlm_only --controller discrete``.

        ``instruction`` overrides the episode instruction for THIS call only and
        does not disturb the visual history — that is the seam a sub-instruction
        outer agent drives (swap the text, keep the frames).
        """
        import torch

        if self._model is None:
            raise RuntimeError("policy not loaded — fire policy_vla_nav__reset first")

        task = instruction if instruction is not None else self._instruction
        front = _to_pil(rgb_front)
        left = _to_pil(rgb_left)
        right = _to_pil(rgb_right)

        history = self._sample_history(self._history)
        # Upstream order is [left, right, front] and the prompt declares it so.
        current = [left, right, front]

        processed_history = [
            self._process_image(img, image_resolution=self._history_resolution())
            for img in history
        ]
        processed_current = [
            self._process_image(img, image_resolution=self.omega_image_resolution)
            for img in current
        ]
        processed = processed_history + processed_current
        pixel_values, image_grid_thw, _, grid_tokens = zip(*processed)

        prompt = _astranav_prompt(task, len(history))
        input_ids = _build_qwen3_prompt_ids(self._tokenizer, prompt, grid_tokens)
        attention_mask = torch.ones_like(input_ids)
        image_token_id = self._tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_token_id is None or image_token_id == self._tokenizer.unk_token_id:
            image_token_id = 151655
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        mm_token_type_ids[input_ids == image_token_id] = 1

        inputs = {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention_mask.to(self._device),
            "mm_token_type_ids": mm_token_type_ids.to(self._device),
            "pixel_values": torch.cat(pixel_values, dim=0).to(
                self._device, dtype=self._model._module_dtype()
            ),
            "image_grid_thw": torch.stack(image_grid_thw).to(self._device),
        }
        # VGGT-Omega runs on the CURRENT three views only (never on history) —
        # that is what "current-3-view" names.
        images_vggt = [torch.stack([item[2] for item in processed_current])]

        with torch.no_grad():
            omega_embeds = self._model._build_omega_embeds(
                images_vggt=images_vggt,
                image_grid_thw=inputs["image_grid_thw"],
                multiview_history_count=[len(history)],
                multiview_current_count=[len(current)],
            )
            self._model._pending_omega_embeds = omega_embeds
            self._model._timing_current = {}
            try:
                generation = self._model.base_model.generate(
                    **inputs,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.pad_token_id,
                    do_sample=False,
                    max_new_tokens=self.llm_max_new_tokens,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            finally:
                self._model._pending_omega_embeds = None

        generated = generation.sequences[:, inputs["input_ids"].shape[1] :]
        raw_text = self._tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        action = _normalize_action(raw_text)
        recognized = action in ACTION_TO_ID

        # History grows AFTER the prediction (upstream appends post-call_model),
        # so step 0 sees an empty history and step N sees the previous N frames.
        self._history.append(front)
        self._step_id += 1

        return {
            "action": action if recognized else "STOP",
            "action_id": ACTION_TO_ID.get(action, ACTION_TO_ID["STOP"]),
            "stop": (not recognized) or action == "STOP",
            "raw_text": raw_text,
            "recognized": recognized,
            "step_id": self._step_id,
            "history_len": len(self._history),
            "instruction": task,
        }
