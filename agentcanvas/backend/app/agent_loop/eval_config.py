"""``EvalConfig`` — what one batch-eval run evaluates, and the one place its
defaults live.

Built once per request (``api/execution/eval.py``), carried to the eval
subprocess inside ``JobSpec`` (``services/run_state_io``), recorded in the
run's ``summary.json`` (``RunSummary.config``), and read back by every
consumer through :meth:`EvalConfig.from_dict` — so a field's default is
stated here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class EvalConfig:
    """Configuration for a batch eval run."""

    graph_name: str
    env_nodeset: str = ""
    # Generic cascade — dict insertion order is the order in which the
    # eval batch pushes fields through ``BaseEnvPanel.on_field_change``
    # before each episode. Each env nodeset declares its own cascade
    # (Habitat: dataset → split → episode_index; SIMPLER: split →
    # task_id → episode_index; HM-EQA: split → episode_index). Callers
    # can populate any subset; ``episode_index`` is always pushed last
    # by the runner and must not appear here.
    selectors: dict[str, Any] = field(default_factory=dict)
    # Legacy convenience fields — kept for serialization back-compat.
    # ``start_eval_v2`` merges these into ``selectors`` at request
    # handling time, so internal eval code reads ``selectors`` only.
    # ``dataset`` is empty for envs that don't need it (Habitat / VLN-CE);
    # populated to "R2R" / "R4R" / "RxR" / "REVERIE" / "CVDN" / "NDH" for
    # the multi-dataset MP3D nodeset (ADR-030 step-2).
    dataset: str = ""
    split: str = "val_unseen"
    episode_count: int = -1  # -1 = all
    start_episode_index: int = 0  # ADR-023: start from specific episode
    # ADR-028: worker-pool batch eval. ``worker_count`` is the number of
    # concurrent env subprocesses driving episodes in parallel. 1 = today's
    # sequential single-tenant behaviour. ``per_step_budget_sec`` bounds the
    # wall-clock budget for a single step; the per-episode timeout is
    # ``resolved_step_budget * per_step_budget_sec`` (resolved per episode
    # via the env hook + graph + API resolver chain). None falls back to
    # the env nodeset's ``default_per_step_budget_sec`` ClassVar (BaseNodeSet
    # default 5.0; Habitat 2.0; MapGPT 30.0).
    worker_count: int = 1
    # Per-episode iteration cap. When None, the framework's resolver chain
    # picks: env-supplied dynamic value (per episode) → graph's authored
    # ``step_budget`` → ``DEFAULT_STEP_BUDGET``. When set, this is the
    # explicit user override and wins over everything except the loop's
    # own stop signal (the iterOut ``stop`` input).
    step_budget: int | None = None
    per_step_budget_sec: float | None = None
    # Optional explicit list of episode indices to evaluate. When set,
    # overrides ``start_episode_index`` + ``episode_count`` and dispatches
    # workers across exactly these indices (preserves order in the queue).
    episode_indices: list[int] | None = None
    # Optional per-episode selector overrides, parallel to the resolved
    # index list. When set, ``episode_selectors[i]`` is merged on top of
    # run-level ``selectors`` before cascade push for episode ``i``. Use
    # this for cross-task sweeps (e.g. SIMPLER 25 tasks x N episodes in a
    # single run). When None, every episode shares run-level selectors —
    # pre-existing behaviour, bit-identical. Length must match the
    # resolved index list (validated at start_eval_v2).
    episode_selectors: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe copy, every field present."""
        return {
            "graph_name": self.graph_name,
            "env_nodeset": self.env_nodeset,
            "selectors": dict(self.selectors),
            "dataset": self.dataset,
            "split": self.split,
            "episode_count": self.episode_count,
            "start_episode_index": self.start_episode_index,
            "worker_count": self.worker_count,
            "step_budget": self.step_budget,
            "per_step_budget_sec": self.per_step_budget_sec,
            "episode_indices": list(self.episode_indices) if self.episode_indices else None,
            "episode_selectors": (
                [dict(s) for s in self.episode_selectors]
                if self.episode_selectors is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> EvalConfig:
        """Tolerant read: a missing field takes its default above, an unknown
        key is ignored, ``None`` where a value is expected means "default"."""
        d = d or {}
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in d.items() if k in known and v is not None}
        kwargs.setdefault("graph_name", "")
        if "selectors" in kwargs:
            kwargs["selectors"] = dict(kwargs["selectors"])
        if d.get("episode_indices") is not None:
            kwargs["episode_indices"] = [int(i) for i in d["episode_indices"]]
        if d.get("episode_selectors") is not None:
            kwargs["episode_selectors"] = [dict(s) for s in d["episode_selectors"]]
        return cls(**kwargs)
