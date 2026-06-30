from __future__ import annotations

"""EnvOpenEQAEMNodeSet — OpenEQA (EM-EQA mode) as a NodeSet.

OpenEQA (Majumdar et al. CVPR 2024) is a free-form Embodied QA benchmark
with two modes:

* **EM-EQA** (Episodic Memory): the agent is given a fixed set of
  pre-recorded frames + a question, and must answer in natural
  language. Scoring is by an LLM-as-judge (1–5 scale → LLM-Match).
  This file implements EM-EQA only — v1 scope.
* **A-EQA** (Active EQA): the agent actively explores the scene before
  answering. Deferred (roadmap E10) — would reuse the HM-EQA Habitat
  manager + an explore-eqa-style termination policy.

Architecture mirrors ``hmeqa.py``:

1. ``OpenEQAEMManager`` (subprocess-singleton)
     Loads the question JSON once at ``initialize()``, then lazily
     loads per-episode frame lists on ``set_episode``. No simulator —
     pure asset I/O — so we keep a small ThreadPoolExecutor purely
     for IO offloading parity with sim-backed nodesets.

2. Canvas tool nodes (``BaseCanvasNode`` adapters):
     env_openeqa_em__reset           — load episode frames + question
     env_openeqa_em__episode_info    — Q, GT answer, category, scene
     env_openeqa_em__sample_frames   — keyframe stride / first-mid-last
     env_openeqa_em__llm_judge       — litellm-backed 1–5 score

3. ``EnvOpenEQAEMNodeSet`` (lifecycle + env panel binding)
     Reuses the ``hmeqa`` conda env in server mode (no habitat
     dependency in v1 — the env has PIL/numpy/litellm already and
     A-EQA later will need habitat-sim from the same env).

Action contract:
    None — EM-EQA is not interactive. The graph is a single-pass DAG
    (frames → VLM → judge → metric). There is **no** ``step`` node.

Dataset layout (env override: ``OPENEQA_DATA_DIR``):

    data/openeqa/
        open-eqa-v0.json             — list[dict] of question records
        episodes/<history>/          — per-episode pre-recorded media
            *.{jpg,png}              — RGB frames (lex-sorted by name)

The loader is tolerant to the upstream JSON's field-naming variation —
it tries several common keys (``question`` / ``q``, ``answer`` /
``gt_answer``, ``category`` / ``question_type``, ``episode_history`` /
``episode_id``). Where the dataset's exact field names differ from
this list, edit ``_QUESTION_FIELD_ALIASES`` below.

last updated: 2026-04-27
"""


import asyncio
import concurrent.futures
import glob
import io
import json
import logging
import os
import re
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)

log = logging.getLogger("agentcanvas.openeqa_em")


# ══════════════════════════════════════════════════════════════════════
# Paths & defaults
# ══════════════════════════════════════════════════════════════════════

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_DATA_ROOT = os.environ.get(
    "OPENEQA_DATA_DIR", os.path.join(_REPO_ROOT, "data", "openeqa")
)

_QUESTIONS_JSON = "open-eqa-v0.json"
_EPISODES_DIR = "episodes"

# ── Question-record field-name aliases (tolerant to upstream variation) ──
_QUESTION_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "question_id":      ("question_id", "id", "episode_id", "qid"),
    "question":         ("question", "q", "prompt"),
    "answer":           ("answer", "gt_answer", "gt", "ground_truth"),
    "category":         ("category", "question_type", "type", "tag"),
    "episode_history":  ("episode_history", "history", "episode_history_path",
                         "episode_id", "scene_id"),
    "extra_answers":    ("extra_answers", "alt_answers", "additional_answers"),
}

_DEFAULTS = {
    "frame_glob_patterns": ("*.png", "*.jpg", "*.jpeg"),
    "max_frames_per_episode": 0,   # 0 = no cap on raw frames; sample_frames node trims
    "judge_profile": "openeqa-judge",
    "judge_temperature": 0.0,
    "judge_max_tokens": 512,
}


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _first_present(d: dict, keys: tuple[str, ...], default: Any = "") -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _normalize_question(raw: dict, index: int) -> dict[str, Any]:
    """Project a raw upstream question dict onto our canonical schema."""
    A = _QUESTION_FIELD_ALIASES
    qid = str(_first_present(raw, A["question_id"], default=str(index)))
    history = str(_first_present(raw, A["episode_history"], default=qid))
    return {
        "index":           index,
        "question_id":     qid,
        "question":        str(_first_present(raw, A["question"], default="")),
        "answer":          str(_first_present(raw, A["answer"], default="")),
        "category":        str(_first_present(raw, A["category"], default="")),
        "episode_history": history,
        "extra_answers":   list(_first_present(raw, A["extra_answers"], default=[])),
        "_raw":            raw,
    }


_AUX_BASENAME_RE = re.compile(r"-(?:depth|seg|sem|semantic|normal|instance)\.[a-zA-Z0-9]+$")


def _list_frame_files(history_dir: str, patterns: tuple[str, ...]) -> list[str]:
    """Return lex-sorted RGB frame paths under ``history_dir``.

    Searches one level deep — handles flat layouts (``episodes/<h>/0001.jpg``),
    nested ones (``episodes/<h>/frames/0001.jpg``), and the AIGeeksGroup
    ``<NNNNN>-rgb.png`` / ``<NNNNN>-depth.png`` schema (depth/aux modalities
    are filtered out so they aren't loaded as RGB).
    """
    if not os.path.isdir(history_dir):
        return []
    candidate_roots: list[str] = [history_dir]
    for sub in ("frames", "rgb", "color", "images"):
        sub_path = os.path.join(history_dir, sub)
        if os.path.isdir(sub_path):
            candidate_roots.append(sub_path)
    found: list[str] = []
    for root in candidate_roots:
        for pat in patterns:
            found.extend(glob.glob(os.path.join(root, pat)))
        if found:
            break
    return sorted(p for p in found if not _AUX_BASENAME_RE.search(os.path.basename(p)))


def _load_frame_as_rgb(path: str) -> np.ndarray | None:
    """Load an image file as H×W×3 uint8 RGB. Returns None on failure."""
    try:
        from PIL import Image
    except ImportError:
        log.error("PIL not installed in subprocess env — cannot load OpenEQA frames")
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.asarray(im, dtype=np.uint8)
    except Exception as exc:
        log.warning("Failed to load frame %s: %s", path, exc)
        return None


# ══════════════════════════════════════════════════════════════════════
# OpenEQAEMManager — subprocess-singleton dataset runtime
# ══════════════════════════════════════════════════════════════════════


class OpenEQAEMManager:
    """Per-subprocess OpenEQA EM dataset manager.

    State held:
        - ``_questions``: full normalized question list (loaded once).
        - ``_current_episode_idx``: cursor used by reset/episode_info.
        - ``_frames_cache_paths``: most recently loaded episode's frame
          paths (decoded by ``sample_frames`` after K is known).
    """

    _instance: OpenEQAEMManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="openeqa",
        )
        self._questions: list[dict[str, Any]] = []
        self._config: dict[str, Any] = dict(_DEFAULTS)
        self._current_episode_idx: int = -1
        self._frames_cache_idx: int = -1
        self._frames_cache_paths: list[str] = []

    @classmethod
    def get(cls) -> OpenEQAEMManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return bool(self._questions)

    # ── Lifecycle ──

    def initialize(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if k in self._config:
                    self._config[k] = v
            json_path = os.path.join(_DATA_ROOT, _QUESTIONS_JSON)
            if not os.path.isfile(json_path):
                log.warning(
                    "OpenEQA questions JSON missing at %s — run "
                    "scripts/data/fetch_dataset_openeqa.sh", json_path,
                )
                self._questions = []
                return
            try:
                with open(json_path) as f:
                    raw = json.load(f)
            except Exception as exc:
                log.error("Failed to parse %s: %s", json_path, exc)
                self._questions = []
                return
            if not isinstance(raw, list):
                log.error(
                    "OpenEQA JSON expected list, got %s — wrong file?",
                    type(raw).__name__,
                )
                self._questions = []
                return
            self._questions = [
                _normalize_question(r, i) for i, r in enumerate(raw)
                if isinstance(r, dict)
            ]
            log.info(
                "OpenEQAEMManager: loaded %d questions from %s",
                len(self._questions), json_path,
            )

    def shutdown(self) -> None:
        with self._lock:
            self._questions = []
            self._frames_cache_paths = []
            self._current_episode_idx = -1
            self._frames_cache_idx = -1

    # ── Episode control ──

    def get_total_episodes(self) -> int:
        return len(self._questions)

    def list_episodes(self, start: int = 0, count: int = 10000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(start, min(start + count, len(self._questions))):
            q = self._questions[i]
            out.append({
                "index":          i,
                "episode_id":     q["question_id"],
                "category":       q["category"],
                "question":       q["question"][:80],
            })
        return out

    def get_episode_info(self, index: int) -> dict[str, Any]:
        if not self._questions:
            return {"error": "OpenEQA not initialized"}
        if index < 0 or index >= len(self._questions):
            return {"error": f"index {index} out of range (0, {len(self._questions)})"}
        q = self._questions[index]
        return {
            "index":            index,
            "episode_id":       q["question_id"],
            "question":         q["question"],
            "answer_gt":        q["answer"],
            "extra_answers":    list(q["extra_answers"]),
            "category":         q["category"],
            "episode_history":  q["episode_history"],
        }

    def set_episode_by_index(self, index: int) -> dict[str, Any]:
        with self._lock:
            if not self._questions:
                return {"error": "OpenEQA not initialized — call initialize() first"}
            if index < 0 or index >= len(self._questions):
                return {"error": f"index {index} out of range"}
            self._current_episode_idx = index
            self._frames_cache_paths = []
            self._frames_cache_idx = -1
            return {"ok": True, "index": index}

    def _load_frames_unlocked(self, index: int) -> list[str]:
        # Returns the episode's frame paths only — decoding happens in
        # ``sample_frames`` (after K is known). Avoids the eager-decode
        # RSS spike on 600-frame ScanNet episodes AND keeps the wire
        # payload between reset → sample_frames JSON-serialisable
        # (custom container classes would not survive pydantic serialisation).
        if self._frames_cache_idx == index and self._frames_cache_paths:
            return list(self._frames_cache_paths)
        q = self._questions[index]
        history_dir = os.path.join(_DATA_ROOT, _EPISODES_DIR, q["episode_history"])
        paths = _list_frame_files(history_dir, self._config["frame_glob_patterns"])
        cap = int(self._config.get("max_frames_per_episode") or 0)
        if cap > 0 and len(paths) > cap:
            # Stride sample (preserve temporal coverage) rather than truncate.
            step = len(paths) / cap
            paths = [paths[int(i * step)] for i in range(cap)]
        self._frames_cache_paths = paths
        self._frames_cache_idx = index
        log.info(
            "OpenEQA: indexed %d frame paths for episode %d (history=%s)",
            len(paths), index, q["episode_history"],
        )
        return list(paths)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if not self._questions:
                return {"error": "OpenEQA not initialized"}
            idx = self._current_episode_idx if self._current_episode_idx >= 0 else 0
            if idx >= len(self._questions):
                idx = 0
            self._current_episode_idx = idx
            q = self._questions[idx]
            frames = self._load_frames_unlocked(idx)
            return {
                "frames":         frames,
                "question":       q["question"],
                "answer_gt":      q["answer"],
                "category":       q["category"],
                "episode_id":     q["question_id"],
                "num_frames":     len(frames),
                "episode_history": q["episode_history"],
            }

    def current_episode(self) -> dict[str, Any]:
        with self._lock:
            if self._current_episode_idx < 0:
                # Lazy default: episode 0 if available (matches reset behavior)
                if not self._questions:
                    return {"error": "no active episode"}
                self._current_episode_idx = 0
            return self.get_episode_info(self._current_episode_idx)


def _get_mgr() -> OpenEQAEMManager:
    return OpenEQAEMManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    mgr = _get_mgr()
    return await asyncio.get_running_loop().run_in_executor(mgr.executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes
# ══════════════════════════════════════════════════════════════════════


def _resize_long_side(arr: Any, target: int) -> Any:
    """Downscale long side to ``target`` px (preserve aspect, no upscale).

    Mirrors the OpenEQA paper baseline (``openai_utils.py:42-44``), which
    uses 512 px to keep multi-frame VLM payloads under provider size caps.
    """
    import numpy as np
    from PIL import Image as _PILImage

    if not isinstance(arr, np.ndarray) or arr.ndim < 2:
        return arr
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= target:
        return arr
    factor = target / float(long_side)
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    pil = _PILImage.fromarray(arr.astype(np.uint8))
    pil = pil.resize((new_w, new_h), _PILImage.LANCZOS)
    return np.asarray(pil)


class ResetOpenEQAEMTool(BaseCanvasNode):
    node_type = "env_openeqa_em__reset"
    display_name = "OpenEQA EM: Reset"
    description = "Load pre-recorded frames + question for the current episode"
    category = "environment"
    icon = "Play"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("trigger", "ANY", "Optional trigger; Initialize fires this once per run",
                optional=True),
    ]
    output_ports = [
        # ``frames`` is a JSON-serialisable list of frame file paths
        # (not decoded ndarrays) — paired with ``sample_frames`` which
        # decodes only the K sampled paths. ANY rather than LIST[IMAGE]
        # because the payload is path strings, not images.
        PortDef("frames",      "ANY",         "Frame file paths (decoded by sample_frames)"),
        PortDef("question",    "TEXT",        "Free-form question text"),
        PortDef("answer_gt",   "TEXT",        "Ground-truth answer (for judge node)"),
        PortDef("category",    "TEXT",        "Question category / type"),
        PortDef("episode_id",  "TEXT",        "Episode identifier"),
        PortDef("num_frames",  "ANY",         "Number of frames returned"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run_sync(_get_mgr().reset)
        if "error" in result:
            self._self_log("error", result["error"])
            return {
                "frames": [], "question": "", "answer_gt": "",
                "category": "", "episode_id": "", "num_frames": 0,
            }
        self._self_log("episode_id", result["episode_id"])
        self._self_log("num_frames", result["num_frames"])
        self._self_log("category", result["category"])
        self._self_log("question", result["question"][:200])
        return {
            "frames":     result["frames"],
            "question":   result["question"],
            "answer_gt":  result["answer_gt"],
            "category":   result["category"],
            "episode_id": result["episode_id"],
            "num_frames": result["num_frames"],
        }


class EpisodeInfoOpenEQAEMTool(BaseCanvasNode):
    node_type = "env_openeqa_em__episode_info"
    display_name = "OpenEQA EM: Episode Info"
    description = "Current episode metadata — question, GT answer, category"
    category = "environment"
    icon = "Info"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports: list = []
    output_ports = [
        PortDef("question",       "TEXT", "Free-form question"),
        PortDef("answer_gt",      "TEXT", "Ground-truth answer"),
        PortDef("extra_answers",  "ANY",  "Optional list of alternative reference answers"),
        PortDef("category",       "TEXT", "Question category"),
        PortDef("episode_id",     "TEXT", "Episode identifier"),
        PortDef("episode_history", "TEXT", "Episode history folder name (debug)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        info = await _run_sync(_get_mgr().current_episode)
        if "error" in info:
            self._self_log("error", info["error"])
            return {
                "question": "", "answer_gt": "", "extra_answers": [],
                "category": "", "episode_id": "", "episode_history": "",
            }
        self._self_log("episode_id", info.get("episode_id"))
        self._self_log("category", info.get("category"))
        self._self_log("answer_gt", info.get("answer_gt"))
        return {
            "question":         info.get("question", ""),
            "answer_gt":        info.get("answer_gt", ""),
            "extra_answers":    info.get("extra_answers", []),
            "category":         info.get("category", ""),
            "episode_id":       info.get("episode_id", ""),
            "episode_history":  info.get("episode_history", ""),
        }


class SampleFramesOpenEQAEMTool(BaseCanvasNode):
    node_type = "env_openeqa_em__sample_frames"
    display_name = "OpenEQA EM: Sample Frames"
    description = "Down-select pre-recorded frames to fit a VLM context window"
    category = "environment"
    icon = "Filter"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "k", "slider", "K (target frame count)",
                default=15, min=1, max=64, step=1,
            ),
            ConfigField(
                "strategy", "select", "Sampling strategy",
                default="uniform",
                options=[
                    {"value": "uniform",            "label": "Uniform stride"},
                    {"value": "first_last_middle",  "label": "First / last / middle (legacy)"},
                ],
            ),
            ConfigField(
                "image_size", "slider", "Long-side resize (px, 0 = no resize)",
                default=512, min=0, max=2048, step=64,
            ),
        ],
    )
    default_config: ClassVar[dict] = {"k": 15, "strategy": "uniform", "image_size": 512}
    input_ports = [
        # ANY because reset emits a list of file path strings (lazy decode
        # contract). We also still accept a real list[ndarray] for callers
        # that pre-decode (e.g. unit tests).
        PortDef("frames", "ANY", "Frame paths or decoded images (typically from reset)"),
    ]
    output_ports = [
        PortDef("sampled", "LIST[IMAGE]", "Down-selected decoded frames (length ≤ K)"),
        PortDef("indices", "ANY",         "Indices into the input list (for debug)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        frames = inputs.get("frames") or []
        # Accept either list[str] (paths from reset — decode-on-sample) or
        # list[ndarray] (pre-decoded for tests / alt sources).
        try:
            n = len(frames)
        except TypeError:
            self._self_log("error", f"frames must be sequence-like, got {type(frames).__name__}")
            return {"sampled": [], "indices": []}
        k = max(1, int(self.config.get("k", self.default_config["k"])))
        strategy = str(self.config.get("strategy", self.default_config["strategy"]))
        image_size = int(self.config.get("image_size", self.default_config["image_size"]))
        if n == 0:
            return {"sampled": [], "indices": []}
        if k >= n:
            indices = list(range(n))
        elif strategy == "first_last_middle":
            if k == 1:
                indices = [n // 2]
            elif k == 2:
                indices = [0, n - 1]
            else:
                middle_count = k - 2
                step = (n - 1) / (middle_count + 1)
                indices = [0] + [int(round(step * (i + 1))) for i in range(middle_count)] + [n - 1]
                # Dedup while preserving order
                seen: set[int] = set()
                indices = [i for i in indices if not (i in seen or seen.add(i))]
        else:  # uniform
            step = n / k
            indices = [int(min(n - 1, round(step * i))) for i in range(k)]
            seen = set()
            indices = [i for i in indices if not (i in seen or seen.add(i))]
        # Materialise the K decoded ndarrays here — this is where lazy decode
        # actually fires (one PNG decode per index). Path strings from reset
        # are loaded via _load_frame_as_rgb; pre-decoded ndarrays pass through.
        # Drop any decode failures so downstream consumers see only valid arrays.
        picked = [frames[i] for i in indices]
        sampled: list[np.ndarray] = []
        for item in picked:
            if isinstance(item, np.ndarray):
                sampled.append(item)
            elif isinstance(item, str):
                arr = _load_frame_as_rgb(item)
                if arr is not None:
                    sampled.append(arr)
        if image_size > 0:
            sampled = [_resize_long_side(f, image_size) for f in sampled]
        self._self_log("input_count", n)
        self._self_log("k", k)
        self._self_log("strategy", strategy)
        self._self_log("image_size", image_size)
        self._self_log("output_count", len(sampled))
        return {"sampled": sampled, "indices": indices}


# ══════════════════════════════════════════════════════════════════════
# Composable judge: builtin llmCall → ParseScore → EmitMetrics
# ══════════════════════════════════════════════════════════════════════


class ParseScoreOpenEQAEMTool(BaseCanvasNode):
    """Extract OpenEQA's 1-5 LLM-Match score from a judge LLM response.

    Stateless transform — splits the score-parsing concern out of the
    monolithic ``env_openeqa_em__llm_judge`` so the LLM call itself can be
    a vanilla builtin ``llmCall`` node and the OpenEQA-specific 1-5
    parsing stays in the OpenEQA nodeset where the convention belongs.

    Pattern: defaults to OpenEQA's ``mmbench.txt`` example format
    ``Your mark: <int>``; falls back to the first ``\\b[1-5]\\b`` digit
    if the LLM omits the prefix. On any parse failure returns
    ``score_1to5 = -1`` (sentinel) and ``parsed = False``.
    """

    node_type = "env_openeqa_em__parse_score"
    display_name = "OpenEQA EM: Parse Score"
    description = "Extract 1-5 score from a judge LLM response (mmbench format)"
    category = "environment"
    icon = "Hash"
    input_ports = [PortDef("text", "TEXT", "Judge LLM response text")]
    output_ports = [
        PortDef("score_1to5", "ANY", "Integer 1-5 (or -1 on parse failure)"),
        PortDef("parsed",     "ANY", "True if a valid 1-5 was extracted, else False"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    default_config: ClassVar[dict] = {}

    _PRIMARY_PATTERN = re.compile(r"Your mark:\s*([1-5])")
    _FALLBACK_PATTERN = re.compile(r"\b([1-5])\b")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        text = str(inputs.get("text", "") or "").strip()
        if not text:
            return {"score_1to5": -1, "parsed": False}

        # Pure-digit response (paper's parse_score also handles this case).
        if text.isdigit() and 1 <= int(text) <= 5:
            score = int(text)
            self._self_log("matched", "pure_digit")
            self._self_log("score", score)
            return {"score_1to5": score, "parsed": True}

        match = self._PRIMARY_PATTERN.search(text)
        which = "primary"
        if not match:
            match = self._FALLBACK_PATTERN.search(text)
            which = "fallback"
        if not match:
            self._self_log("no_match", text[:200])
            return {"score_1to5": -1, "parsed": False}

        score = int(match.group(1))
        self._self_log("matched", which)
        self._self_log("score", score)
        return {"score_1to5": score, "parsed": True}


class EmitMetricsOpenEQAEMTool(BaseCanvasNode):
    """Wrap a 1-5 OpenEQA score into the canonical metrics dict.

    Emits both the raw integer score and its [0,1]-normalized form
    (``llm_match = (score - 1) / 4``) per the OpenEQA paper's reporting
    convention. Failure sentinel (``score < 1``) → ``openeqa_score = -1``,
    ``openeqa_llm_match = 0.0`` so downstream aggregation flags the
    episode as un-judged.
    """

    node_type = "env_openeqa_em__emit_metrics"
    display_name = "OpenEQA EM: Emit Metrics"
    description = "Pack a 1-5 score into {openeqa_score, openeqa_llm_match}"
    category = "environment"
    icon = "BarChart3"
    input_ports = [PortDef("score_1to5", "ANY", "Integer score from parse_score node")]
    output_ports = [
        PortDef("metrics",   "METRICS", "{openeqa_score, openeqa_llm_match}"),
        PortDef("llm_match", "ANY",     "(score - 1) / 4, in [0,1]; 0.0 on sentinel"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    default_config: ClassVar[dict] = {}

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        try:
            score = int(inputs.get("score_1to5", -1))
        except (TypeError, ValueError):
            score = -1

        if score < 1 or score > 5:
            metrics = {"openeqa_score": -1.0, "openeqa_llm_match": 0.0}
            self._self_log("sentinel", score)
            return {"metrics": metrics, "llm_match": 0.0}

        llm_match = (score - 1) / 4.0
        metrics = {"openeqa_score": float(score), "openeqa_llm_match": llm_match}
        self._self_log("score", score)
        self._self_log("llm_match", llm_match)
        return {"metrics": metrics, "llm_match": llm_match}


class OpenEQAEMEnvPanel(BaseEnvPanel):
    """Two-field cascade ``split → episode_index``.

    OpenEQA's public release is val-only (no train/test labels), so the
    split selector has a single option today; the field is retained
    for future-proofing without an env panel-schema migration.
    """

    name = "env_openeqa_em"
    display_name = "OpenEQA (EM)"
    fields = [
        EnvPanelField("split",         "select", "Split"),
        EnvPanelField("episode_index", "select", "Episode"),
    ]
    actions = [
        EnvPanelAction("play",  "Play",  side_effect="run_start"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop",  "Stop",  side_effect="run_stop",  enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {"split": "val", "episode_index": 0}

    def _mgr(self) -> OpenEQAEMManager:
        return OpenEQAEMManager.get()

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mgr().executor, fn, *args)

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "split": self._state.get("split", "val"),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    async def on_load(self) -> dict[str, Any]:
        mgr = self._mgr()
        if not mgr.initialized:
            return {
                "available": False,
                "split": "val",
                "episode_index": 0,
                "episode_count": 0,
                "splits": ["val"],
                "step_budget": 1,
                "message": (
                    "OpenEQA not initialized. Load env_openeqa_em from the "
                    "NodeSet Manager (and run fetch_dataset_openeqa.sh first)."
                ),
            }
        total = mgr.get_total_episodes()
        current_idx = (
            mgr._current_episode_idx if mgr._current_episode_idx >= 0 else 0
        )
        self._state["episode_index"] = current_idx
        ep_info = mgr.get_episode_info(current_idx)
        return {
            "available": True,
            "split": self._state.get("split", "val"),
            "episode_index": current_idx,
            "episode_count": total,
            "splits": ["val"],
            "step_budget": 1,  # EM-EQA is a single-pass DAG
            "current_episode": ep_info if "error" not in ep_info else None,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        mgr = self._mgr()
        if name == "split":
            self._state["split"] = str(value)
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                idx = int(value)
            except (TypeError, ValueError):
                idx = 0
            self._state["episode_index"] = idx
            if mgr.initialized:
                await self._run(mgr.set_episode_by_index, idx)
        else:
            self._state[name] = value

        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = self._episode_reset_payload()
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        mgr = self._mgr()
        if name in ("play", "reset"):
            if not mgr.initialized:
                return {"ok": False, "side_effect": "none",
                        "error": "OpenEQA not initialized"}
            await self._run(mgr.set_episode_by_index, int(self._state["episode_index"]))
            if name == "play":
                return {"ok": True, "side_effect": "run_start"}
            return {
                "ok": True,
                "side_effect": "signal",
                "signal_name": "episode_reset",
                "signal_payload": self._episode_reset_payload(),
            }
        if name in ("pause", "stop"):
            return {"ok": True, "side_effect": f"run_{name}"}
        return {"ok": False, "side_effect": "none", "error": f"Unknown action '{name}'"}

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        if field == "split":
            mgr = self._mgr()
            count = mgr.get_total_episodes() if mgr.initialized else 0
            return [{"value": "val", "label": f"val ({count} questions)"}]
        if field == "episode_index":
            mgr = self._mgr()
            if not mgr.initialized:
                return []
            episodes = await self._run(mgr.list_episodes, 0, 10000)
            return [
                {
                    "value": ep["index"],
                    "label": "{}: [{}] {}".format(
                        ep["index"],
                        ep.get("category", "")[:12],
                        ep.get("question", "")[:60],
                    ),
                }
                for ep in episodes
            ]
        return []


# ══════════════════════════════════════════════════════════════════════
# EnvOpenEQAEMNodeSet — the nodeset binding
# ══════════════════════════════════════════════════════════════════════


def _resolve_server_python() -> str | None:
    """Pick an interpreter for server-mode hosting, or None to stay local.

    Order: explicit ``$HMEQA_PYTHON`` → known ``hmeqa`` conda env if present
    → ``None``. Returning None lets ``ComponentRegistry.load_nodeset`` skip
    the auto-route and keep the nodeset in local mode — fine here because
    OpenEQA EM-EQA only needs PIL + numpy + litellm, which the default
    ``agentcanvas`` env already ships. A-EQA (E10) will require habitat-sim
    and should re-pin to the hmeqa env at that time.
    """
    return conda_env_python("ac-hmeqa", "HMEQA_PYTHON")


class EnvOpenEQAEMNodeSet(BaseNodeSet):
    """OpenEQA EM-EQA mode as a NodeSet.

    Server-hosted in the ``hmeqa`` conda env when available (set
    ``$HMEQA_PYTHON`` or install the env). Falls back to local mode in the
    backend's own interpreter when no hmeqa env is present — OpenEQA
    EM-EQA's runtime needs (PIL, numpy, litellm, the app) are all in the
    default ``agentcanvas`` env. A future A-EQA mode (E10) needs
    habitat-sim and will re-require server mode.
    """

    name = "env_openeqa_em"
    description = "OpenEQA — Embodied QA (Episodic Memory mode), free-form + LLM-judge"
    server_python = _resolve_server_python()
    env_panel = OpenEQAEMEnvPanel
    parallelism = "replicated"  # Stateful per-episode frame cache
    # LLM judge call dominates per-episode time (10–60 s); give a roomy budget.
    default_per_step_budget_sec = 90.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = OpenEQAEMManager.get()

    def get_tools(self) -> list:
        return [
            ResetOpenEQAEMTool(),
            EpisodeInfoOpenEQAEMTool(),
            SampleFramesOpenEQAEMTool(),
            ParseScoreOpenEQAEMTool(),
            EmitMetricsOpenEQAEMTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        if self._mgr.initialized:
            log.info("OpenEQA already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(**kwargs),
        )
        log.info("EnvOpenEQAEMNodeSet initialized")

    async def shutdown(self) -> None:
        self._mgr.shutdown()

    async def get_eval_metadata(self) -> dict:
        count = self._mgr.get_total_episodes() if self._mgr.initialized else 0
        return {
            "env_name": "openeqa_em",
            "datasets": ["OpenEQA"],
            "splits": ["val"],
            "episode_counts": {"val": count},
            "metrics": ["openeqa_score", "openeqa_llm_match"],
            "supports_set_episode": self._mgr.initialized,
            "step_budget": 1,  # single-pass DAG
        }
