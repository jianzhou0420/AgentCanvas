"""ExploreEQA nodeset — the *reasoning core* for HM-EQA exploration.

Reasoning side of the HM-EQA stack. The Ren et al. 2024 "Explore until
Confident" mechanic is split across three wired nodesets:

  * ``env/env_hmeqa``           — the habitat simulator (RGB-D, pose, step)
  * ``model/vlm_prismatic``     — the VLM (``score_tokens``), swappable
  * ``method/explore_eqa_tsdf`` — the TSDF voxel *world model* (replicated)
  * ``method/explore_eqa``      — THIS nodeset: stateless reasoning glue

This nodeset owns only the reasoning glue (prompt construction, frontier
visual-prompt *labelling*, post-hoc voting) + a pure-data score history.
The TSDF voxel map was split out to ``explore_eqa_tsdf`` on 2026-06-14 (it
is a stateful per-worker world model, not method reasoning); the frontier
geometry now arrives as pixel coords over a wire from that server.

Six canvas nodes:

  explore_eqa__build_question      — format Q + A/B/C/D for the VLM
  explore_eqa__vlm_score_pre       — emit (image, prompt, tokens) bundles
  explore_eqa__vlm_score_post      — record per-step probs into history
  explore_eqa__frontier_pre        — draw A/B/C/D labels on the frontier
                                     pixel coords (from explore_eqa_tsdf),
                                     build LSV/GSV prompts
  explore_eqa__frontier_post       — combine LSV+GSV probs into the
                                     per-frontier semantic value sv
  explore_eqa__aggregate_answer    — post-hoc weighted-vote → final letter

State — a **graph-level (home) container** ``explore_eqa_mem`` declared in the
graph JSON (NOT nodeset-owned), reached via ``access_grants``:

  preds  (accumulator) — per-step answer-prob history (pure data)
  rels   (accumulator) — per-step relevancy-prob history (pure data)

This nodeset is fully ``local``/stateless (owns no container) and runs
in-process on the backend's ``agentcanvas`` interpreter, off the ``hmeqa``
env. Home containers are per-worker + in-process-reachable by local nodes, so
no ``episode_id`` key is needed (``lifetime="episode"`` resets per episode).

last updated: 2026-06-14
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

log = logging.getLogger("agentcanvas.explore_eqa")


# ══════════════════════════════════════════════════════════════════════
# Owned state container
# ══════════════════════════════════════════════════════════════════════
#
# explore_eqa is a server-mode ``shared`` singleton: under worker_count>1, N
# episodes run concurrently in this one subprocess. State lives in a
# nodeset-owned StateContainer (``explore_eqa_mem``), partitioned by the
# on-wire ``episode_id`` key so concurrent episodes never collide. Nodes are
# stateless — they only read/write the container. Per-episode cleanup is the
# framework's worker-safe ``container.evict(episode_id)`` at episode end (eval
# worker loop, ``POST /containers/evict``), NOT a module-global ``.clear()``:
# the 2026-05-03 race fix lives on in this new home — only one key is ever
# dropped, so concurrent siblings are untouched.

_CONTAINER_ID = "explore_eqa_mem"


def _container(ctx):
    """Return the nodeset-owned container, injected by-reference in server
    mode (``ctx.containers``). explore_eqa is server-only — raise a clear
    error otherwise so a misconfigured local-mode load fails loudly."""
    containers = getattr(ctx, "containers", None) or {}
    c = containers.get(_CONTAINER_ID)
    if c is None:
        raise RuntimeError(
            f"explore_eqa: home container '{_CONTAINER_ID}' not reachable — "
            "declare it in the graph's `containers` and grant this node access "
            "via `access_grants` (see the nodeset docstring)."
        )
    return c


# ── Defaults (mirror explore-eqa/cfg/vlm_exp.yaml) ──
# TSDF/integration defaults moved to the explore_eqa_tsdf verb nodes.

# Frontier visual-prompt letters (upstream uses exactly 4).
_DRAW_LETTERS = ["A", "B", "C", "D"]
_CIRCLE_RADIUS = 18
_FONT_SIZE = 30

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_DEFAULT_FONT_PATH = os.path.join(
    _REPO_ROOT, "data", "hm3d", "hmeqa", "Open_Sans", "static", "OpenSans-Regular.ttf"
)

# GSV scaling constants (from explore-eqa paper)
_GSV_T = 0.5
_GSV_F = 3.0


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _to_pil_rgb(rgb):
    """Accept numpy array, list, or PIL; return PIL.Image in RGB."""
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


# ══════════════════════════════════════════════════════════════════════
# Node 1: BuildVLMQuestion
# ══════════════════════════════════════════════════════════════════════


class BuildVLMQuestionNode(BaseCanvasNode):
    """Format the multi-choice question for VLM consumption."""

    node_type: ClassVar[str] = "explore_eqa__build_question"
    display_name: ClassVar[str] = "ExploreEQA: Build VLM Question"
    description: ClassVar[str] = (
        "Append A/B/C/D choices to the raw question to form the VLM prompt body"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "Type"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports: ClassVar[list] = [
        PortDef("question", "TEXT", "Raw question (no A/B/C/D tail)"),
        PortDef("choices", "ANY", "List of 4 choice strings"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("vlm_question", "TEXT", "Question with A/B/C/D tail for VLM prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        question = inputs.get("question", "") or ""
        choices = inputs.get("choices") or []
        if isinstance(choices, str):
            try:
                choices = json.loads(choices.replace("'", '"'))
            except Exception:
                choices = [s.strip() for s in choices.split(",")]
        vlm_q = str(question)
        for letter, choice in zip(["A", "B", "C", "D"], choices):  # noqa: B905
            vlm_q += f"\n{letter}. {choice}"
        self._self_log("vlm_question_len", len(vlm_q))
        return {"vlm_question": vlm_q}


# ══════════════════════════════════════════════════════════════════════
# Node 2a: VLMScorePre — emit prompts for downstream score_tokens
# ══════════════════════════════════════════════════════════════════════


class VLMScorePreNode(BaseCanvasNode):
    """Build the (image, prompt, tokens) bundles for the per-step
    answer-prob and relevancy VLM calls.

    Emits two parallel bundles:
      - pred  : answer letter A/B/C/D (vlm_question + 'answer with letter')
      - rel   : relevancy Yes/No ('are you confident answering with current view?')

    Pair this node with two ``vlm_prismatic__score_tokens`` instances
    and a ``explore_eqa__vlm_score_post`` to record the resulting probs.
    """

    node_type: ClassVar[str] = "explore_eqa__vlm_score_pre"
    display_name: ClassVar[str] = "ExploreEQA: VLM Score (Pre)"
    description: ClassVar[str] = (
        "Build prompts + tokens for the per-step answer-prob (A/B/C/D) "
        "and relevancy (Yes/No) VLM scoring calls"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "Sparkles"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "Current RGB view"),
        PortDef("vlm_question", "TEXT", "Formatted A/B/C/D question"),
        PortDef("question", "TEXT", "Raw question (for the relevancy prompt)"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("image_pred", "IMAGE", "Image for answer-prob scoring"),
        PortDef("prompt_pred", "TEXT", "Prompt for answer-prob scoring"),
        PortDef("tokens_pred", "ANY", "['A','B','C','D']"),
        PortDef("image_rel", "IMAGE", "Image for relevancy scoring"),
        PortDef("prompt_rel", "TEXT", "Prompt for relevancy scoring"),
        PortDef("tokens_rel", "ANY", "['Yes','No']"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        rgb = inputs.get("rgb")
        vlm_q = inputs.get("vlm_question", "") or ""
        question = inputs.get("question", "") or ""

        prompt_pred = vlm_q + "\nAnswer with the option's letter from the given choices directly."
        prompt_rel = (
            f"\nConsider the question: '{question}'. Are you confident about"
            " answering the question with the current view? Answer with Yes or No."
        )

        return {
            "image_pred": rgb,
            "prompt_pred": prompt_pred,
            "tokens_pred": ["A", "B", "C", "D"],
            "image_rel": rgb,
            "prompt_rel": prompt_rel,
            "tokens_rel": ["Yes", "No"],
        }


# ══════════════════════════════════════════════════════════════════════
# Node 2b: VLMScorePost — record per-step probs into history
# ══════════════════════════════════════════════════════════════════════


class VLMScorePostNode(BaseCanvasNode):
    """Record per-step (pred_probs, rel_probs) into the episode history.

    Reads probs from upstream ``vlm_prismatic__score_tokens`` nodes.
    Appends to the owned container's ``preds``/``rels`` accumulators under
    the ``episode_id`` key so AggregateAnswer can do post-hoc weighted
    voting without re-running the VLM.
    """

    node_type: ClassVar[str] = "explore_eqa__vlm_score_post"
    display_name: ClassVar[str] = "ExploreEQA: VLM Score (Post)"
    description: ClassVar[str] = "Record per-step pred/rel probs into the episode score history"
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "BookOpen"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports: ClassVar[list] = [
        PortDef("pred_probs", "ANY", "Softmax over A/B/C/D — shape (4,)"),
        PortDef("rel_probs", "ANY", "Softmax over Yes/No — shape (2,)"),
        PortDef("episode_id", "TEXT", "Episode id for history bookkeeping"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("pred_probs", "ANY", "Pass-through pred probs"),
        PortDef("rel_probs", "ANY", "Pass-through rel probs"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        episode_id = str(inputs.get("episode_id", "") or "0")
        pred = inputs.get("pred_probs") or []
        rel = inputs.get("rel_probs") or []

        pred_list = (
            [float(x) for x in np.asarray(pred).ravel().tolist()]
            if len(pred)
            else [
                0.25,
                0.25,
                0.25,
                0.25,
            ]
        )
        rel_list = (
            [float(x) for x in np.asarray(rel).ravel().tolist()]
            if len(rel)
            else [
                0.01,
                0.99,
            ]
        )

        container = _container(ctx)
        container.write("preds", pred_list)
        container.write("rels", rel_list)

        self._self_log("pred_probs", pred_list)
        self._self_log("rel_probs", rel_list)
        self._self_log("step_history_len", len(container.read("preds")))
        return {"pred_probs": pred_list, "rel_probs": rel_list}


# ══════════════════════════════════════════════════════════════════════
# Node 3a: FrontierPre — find candidates, draw labels, emit prompts
# ══════════════════════════════════════════════════════════════════════


class FrontierPreNode(BaseCanvasNode):
    """Find frontier candidates, draw A/B/C/D labels, emit the LSV+GSV
    (image, prompt, tokens) bundles for downstream score_tokens nodes.

    Bypass mode: when fewer than ``min_num_prompt_points`` candidates
    are returned, emit empty token lists so the downstream score_tokens
    short-circuits to []. The post node then no-ops the integration.
    """

    node_type: ClassVar[str] = "explore_eqa__frontier_pre"
    display_name: ClassVar[str] = "ExploreEQA: Frontier (Pre)"
    description: ClassVar[str] = (
        "Find candidate frontier points, draw labels, build LSV+GSV prompts"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "MapPin"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    config_schema: ClassVar[list[ConfigField]] = [
        ConfigField("min_num_prompt_points", "integer", default=2),
        ConfigField("use_lsv", "boolean", default=True),
        ConfigField("use_gsv", "boolean", default=True),
    ]

    input_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "Current RGB view"),
        PortDef("question", "TEXT", "Raw question"),
        PortDef("candidates_pix", "ANY", "Frontier pixel coords (from explore_eqa_tsdf__find_frontiers)"),
        PortDef("num_candidates", "ANY", "Number of frontier candidates"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("annotated_image", "IMAGE", "RGB with A/B/C/D labels (or rgb if skipped)"),
        PortDef("image_lsv", "IMAGE", "Annotated image for LSV scoring"),
        PortDef("prompt_lsv", "TEXT", "LSV prompt"),
        PortDef("tokens_lsv", "ANY", "Letters [A,B,C,D][:n] (empty if skipped)"),
        PortDef("image_gsv", "IMAGE", "Base image for GSV scoring"),
        PortDef("prompt_gsv", "TEXT", "GSV prompt"),
        PortDef("tokens_gsv", "ANY", "[Yes,No] (empty if skipped)"),
        PortDef("num_candidates", "ANY", "Number of frontier candidates"),
        PortDef("skip", "BOOL", "True iff below min_num_prompt_points"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        cfg = self.config or {}
        min_prompt = int(cfg.get("min_num_prompt_points", 2))
        use_lsv = bool(cfg.get("use_lsv", True))
        use_gsv = bool(cfg.get("use_gsv", True))

        rgb = inputs.get("rgb")
        question = inputs.get("question", "") or ""
        cp = inputs.get("candidates_pix")
        candidates_pix = np.asarray(cp) if cp is not None else np.empty((0, 2))
        actual_n = int(inputs.get("num_candidates", len(candidates_pix)) or 0)

        empty_skip = {
            "annotated_image": rgb,
            "image_lsv": rgb,
            "prompt_lsv": "",
            "tokens_lsv": [],
            "image_gsv": rgb,
            "prompt_gsv": "",
            "tokens_gsv": [],
            "num_candidates": 0,
            "skip": True,
        }

        self._self_log("num_candidates", actual_n)
        pil = _to_pil_rgb(rgb)
        if actual_n < min_prompt or pil is None or len(candidates_pix) == 0:
            self._self_log("skipped", True)
            return empty_skip

        from PIL import ImageDraw, ImageFont

        pil_draw = pil.copy()
        draw = ImageDraw.Draw(pil_draw)
        try:
            font = ImageFont.truetype(_DEFAULT_FONT_PATH, _FONT_SIZE)
        except Exception:
            font = ImageFont.load_default()

        for i, pt in enumerate(candidates_pix[: len(_DRAW_LETTERS)]):
            px = int(pt[0])
            py = int(pt[1])
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
            draw.text(
                (px, py),
                _DRAW_LETTERS[i],
                font=font,
                fill=(0, 0, 0, 255),
                anchor="mm",
                font_size=12,
            )

        annotated_arr = np.asarray(pil_draw)
        prompt_lsv = (
            f"\nConsider the question: '{question}', and you will explore"
            " the environment for answering it.\nWhich direction (black"
            " letters on the image) would you explore then? Answer with"
            " a single letter."
        )
        prompt_gsv = (
            f"\nConsider the question: '{question}', and you will explore"
            " the environment for answering it. Is there any direction"
            " shown in the image worth exploring? Answer with Yes or No."
        )

        return {
            "annotated_image": annotated_arr,
            "image_lsv": annotated_arr if use_lsv else rgb,
            "prompt_lsv": prompt_lsv,
            "tokens_lsv": list(_DRAW_LETTERS[:actual_n]) if use_lsv else [],
            "image_gsv": np.asarray(pil),
            "prompt_gsv": prompt_gsv,
            "tokens_gsv": ["Yes", "No"] if use_gsv else [],
            "num_candidates": int(actual_n),
            "skip": False,
        }


# ══════════════════════════════════════════════════════════════════════
# Node 3b: FrontierPost — combine LSV+GSV, integrate into TSDF
# ══════════════════════════════════════════════════════════════════════


class FrontierPostNode(BaseCanvasNode):
    """Combine LSV+GSV probs into the per-frontier semantic value (sv = lsv*gsv).

    The integration into the TSDF semantic map is now done by the
    ``explore_eqa_tsdf__integrate_sem`` verb — this node only does the pure
    arithmetic and wires ``frontier_scores`` (+ skip/num_candidates) to it.
    """

    node_type: ClassVar[str] = "explore_eqa__frontier_post"
    display_name: ClassVar[str] = "ExploreEQA: Frontier (Post)"
    description: ClassVar[str] = "Combine LSV+GSV probs into per-frontier semantic value sv"
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "MapPin"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    input_ports: ClassVar[list] = [
        PortDef("lsv_probs", "ANY", "Probs over labelled frontier letters"),
        PortDef("gsv_probs", "ANY", "Probs over [Yes, No] for global value"),
        PortDef("num_candidates", "ANY", "Number of frontier candidates"),
        PortDef("skip", "BOOL", "True iff frontier_pre skipped"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("frontier_scores", "ANY", "sv = lsv*gsv (→ explore_eqa_tsdf__integrate_sem.sem_pix)"),
        PortDef("num_candidates", "ANY", "Pass-through for integrate_sem"),
        PortDef("skip", "BOOL", "Pass-through for integrate_sem"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        lsv_in = inputs.get("lsv_probs") or []
        gsv_in = inputs.get("gsv_probs") or []
        n = int(inputs.get("num_candidates", 0) or 0)
        skip = bool(inputs.get("skip", False))

        if skip or n == 0:
            self._self_log("skipped", True)
            return {"frontier_scores": [], "num_candidates": n, "skip": True}

        # LSV: scale by candidate count (matches upstream).
        if len(lsv_in) >= n:
            lsv = np.asarray(lsv_in[:n]) * (n / 3.0)
        else:
            lsv = np.ones(n) / n

        # GSV: scalar derived from P(Yes) (upstream cfg/vlm_exp.yaml).
        if len(gsv_in) >= 1:
            gsv_yes = float(gsv_in[0])
            gsv = float(np.exp(gsv_yes / _GSV_T) / _GSV_F)
        else:
            gsv = 1.0

        sv = (np.asarray(lsv) * gsv).astype(np.float64)
        sv_list = [float(x) for x in sv.tolist()]
        self._self_log("frontier_scores", sv_list)
        self._self_log("gsv", gsv)
        return {"frontier_scores": sv_list, "num_candidates": n, "skip": False}


# ══════════════════════════════════════════════════════════════════════
# Node 6: AggregateAnswer
# ══════════════════════════════════════════════════════════════════════


class AggregateAnswerNode(BaseCanvasNode):
    """Post-hoc weighted aggregation over the trajectory."""

    node_type: ClassVar[str] = "explore_eqa__aggregate_answer"
    display_name: ClassVar[str] = "ExploreEQA: Aggregate Answer"
    description: ClassVar[str] = (
        "Weighted-vote over per-step VLM scores to emit the final answer letter"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "CheckCircle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    input_ports: ClassVar[list] = [
        PortDef("episode_id", "TEXT", "Episode id (score-history key)"),
        PortDef("choices", "ANY", "List of 4 choice strings"),
        PortDef("answer_gt", "TEXT", "Ground-truth letter (optional)", optional=True),
    ]
    output_ports: ClassVar[list] = [
        PortDef("pred_letter", "TEXT", "Predicted letter A/B/C/D (weighted variant)"),
        PortDef("pred_text", "TEXT", "Full text of the predicted choice"),
        PortDef("success_weighted", "BOOL", "pred_letter == answer_gt (weighted)"),
        PortDef("success_max", "BOOL", "pred_letter == answer_gt (max-relevancy)"),
        PortDef("num_steps_scored", "ANY", "How many VLM-scored steps were aggregated"),
        PortDef(
            "metrics",
            "METRICS",
            "{success_weighted, success_max, num_steps_scored} surfaced to eval-API",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        episode_id = str(inputs.get("episode_id", "") or "0")
        choices = inputs.get("choices") or []
        answer_gt = (inputs.get("answer_gt") or "").strip()

        container = _container(ctx)
        preds = container.read("preds") or []
        rels = container.read("rels") or []
        n = len(preds)
        candidates = ["A", "B", "C", "D"]

        if n == 0:
            self._self_log("error", "no score history")
            return {
                "pred_letter": "",
                "pred_text": "",
                "success_weighted": False,
                "success_max": False,
                "num_steps_scored": 0,
                "metrics": {
                    "success_weighted": 0.0,
                    "success_max": 0.0,
                    "num_steps_scored": 0.0,
                },
            }

        pred_arr = np.asarray(preds, dtype=np.float64)
        rel_arr = np.asarray(rels, dtype=np.float64)
        rel_yes = rel_arr[:, 0]
        weighted = rel_yes[:, None] * pred_arr

        smx_max = np.max(weighted, axis=0)
        letter_weighted = candidates[int(np.argmax(smx_max))]

        best_step = int(np.argmax(rel_yes))
        letter_max = candidates[int(np.argmax(weighted[best_step]))]

        success_weighted = bool(answer_gt and letter_weighted == answer_gt)
        success_max = bool(answer_gt and letter_max == answer_gt)

        try:
            idx = candidates.index(letter_weighted)
            pred_text = str(choices[idx]) if idx < len(choices) else letter_weighted
        except (ValueError, IndexError):
            pred_text = letter_weighted

        self._self_log("pred_letter", letter_weighted)
        self._self_log("success_weighted", success_weighted)
        self._self_log("success_max", success_max)
        self._self_log("num_steps_scored", n)
        return {
            "pred_letter": letter_weighted,
            "pred_text": pred_text,
            "success_weighted": success_weighted,
            "success_max": success_max,
            "num_steps_scored": n,
            "metrics": {
                "success_weighted": float(success_weighted),
                "success_max": float(success_max),
                "num_steps_scored": float(n),
            },
        }


# ══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ══════════════════════════════════════════════════════════════════════


class ExploreEQANodeSet(BaseNodeSet):
    """ExploreEQA reasoning core — fully stateless ``local`` nodeset.

    The TSDF voxel world-model was split out to the ``replicated``
    ``explore_eqa_tsdf`` server (2026-06-14); VLM scoring is the separate
    ``vlm_prismatic`` server. All three are composed on the canvas via wires.

    This nodeset owns **no** container — it runs ``local`` (in-process, on the
    backend's ``agentcanvas`` interpreter, off the ``hmeqa`` env). The
    ``preds``/``rels`` score history is a **graph-level (home) container**
    (``explore_eqa_mem``, declared in the graph's ``containers`` + reached by
    ``vlm_score_post`` / ``aggregate_answer`` via ``access_grants``). Home
    containers are per-worker and in-process-reachable by local nodes, so no
    ``episode_id`` key is needed (lifetime="episode" resets them per episode).
    """

    name: ClassVar[str] = "explore_eqa"
    description: ClassVar[str] = (
        "ExploreEQA reasoning core — VLM prompt prep, frontier visual-prompt"
        " labelling, post-hoc vote (TSDF via explore_eqa_tsdf, VLM via vlm_prismatic)"
    )

    def get_tools(self) -> list:
        return [
            BuildVLMQuestionNode(),
            VLMScorePreNode(),
            VLMScorePostNode(),
            FrontierPreNode(),
            FrontierPostNode(),
            AggregateAnswerNode(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        # No VLM warm-up here — Prismatic lives in vlm_prismatic now.
        pass

    # No get_containers(): preds/rels is a graph-level home container, declared
    # in the graph JSON and reached via access_grants (see module docstring).

    async def shutdown(self) -> None:
        pass
