"""VLNVerse episode metrics — nDTW + per-episode accumulator.

Vendored from:
    NavHarness navharness/model/basemodel/mllm/mllm_utils.py  (``calc_ndtw``)
    NavHarness navharness/model/basemodel/mllm/agentic/env_bridge.py
    (``EnvBridge`` metric bookkeeping + ``final_metrics``, lines ~182-532)

``calc_ndtw`` is a verbatim numpy port (upstream's module drags in
cv2/torch/matplotlib at import time; this file needs numpy only — parity is
pinned in ``test_metrics.py`` against the extracted upstream source).

``EpisodeMetricsAccumulator`` reproduces EnvBridge's accumulation contract:

- ``initial_distance`` captured at episode start;
- before EVERY env step the *pre-action* pose and dist-to-goal are recorded
  (``traj`` / ``dists``), and after it the step's collision flag (``cols``);
- ``path_length`` accumulates the *intended* move distance (upstream adds
  the commanded ``distance``, not the occupancy-snapped actual — quirk kept);
- ``final_metrics`` computes the exact upstream metric dict, including its
  edge-case quirks (documented inline).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def calc_ndtw(pred_traj, gt_traj, threshold: float = 3.0) -> float:
    """Normalized Dynamic Time Warping between two [x, y, z] trajectories.

    Verbatim numpy port of NavHarness ``mllm_utils.calc_ndtw`` — xy-only,
    plain DTW accumulation on Euclidean cell cost, normalized as
    ``exp(-DTW / (threshold * len(gt)))``.
    """
    if len(pred_traj) == 0 or len(gt_traj) == 0:
        return 0.0

    pred_xy = np.array(pred_traj)[:, :2]
    gt_xy = np.array(gt_traj)[:, :2]

    n, m = len(pred_xy), len(gt_xy)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = np.linalg.norm(pred_xy[i - 1] - gt_xy[j - 1])
            # (upstream computes a Gaussian similarity here too, then never
            # uses it — dropped)
            dtw[i, j] = dist + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    dtw_distance = dtw[n, m]
    ndtw = np.exp(-dtw_distance / (threshold * len(gt_xy)))
    return float(ndtw)


class EpisodeMetricsAccumulator:
    """Per-episode metric bookkeeping, EnvBridge-faithful.

    Usage (manager drives it):
        acc = EpisodeMetricsAccumulator(initial_distance=kin.current_dist_to_goal())
        # per env step — pre-action pose/dist, post-action collision:
        acc.record_step(position, dist_to_goal, collided, moved_distance=d, is_move=True)
        acc.mark_stop(); acc.set_end_reason("stop")
        metrics = acc.final_metrics(ep_id, kin.current_dist_to_goal(), ref_path)
    """

    def __init__(self, initial_distance: float) -> None:
        self.initial_distance = float(initial_distance)
        self.path_length = 0.0
        self.traj: List[Any] = []
        self.dists: List[float] = []
        self.cols: List[bool] = []
        self.steps = 0
        self.moves = 0
        self.called_stop = False
        self.end_reason: Optional[str] = None

    def record_step(
        self,
        position: Any,
        dist_to_goal: float,
        collided: bool,
        moved_distance: float = 0.0,
        is_move: Optional[bool] = None,
    ) -> None:
        """Record one env step.

        ``position`` / ``dist_to_goal`` are the PRE-action values (upstream
        ``_record_pre``); ``collided`` is the step's outcome. ``moved_distance``
        is the commanded distance (0 for STOP). ``is_move`` defaults to
        ``moved_distance > 0`` — mirrors upstream counting forward/goto moves
        but not STOPs.
        """
        self.dists.append(float(dist_to_goal))
        self.traj.append(position.copy() if hasattr(position, "copy") else position)
        self.cols.append(bool(collided))
        self.path_length += float(moved_distance)
        self.steps += 1
        if is_move if is_move is not None else moved_distance > 0:
            self.moves += 1

    def mark_stop(self) -> None:
        self.called_stop = True

    def set_end_reason(self, reason: str) -> None:
        self.end_reason = reason

    def final_metrics(
        self,
        episode_id: Any,
        current_dist_to_goal: float,
        reference_path: Any,
        success_distance: float = 3.0,
    ) -> dict:
        """The upstream ``EnvBridge.final_metrics`` dict, key-for-key."""
        succ_d = float(success_distance)
        final_dist = float(current_dist_to_goal)
        dists = self.dists if self.dists else [self.initial_distance]

        success = 1 if final_dist <= succ_d else 0
        oracle = 1 if any(d <= succ_d for d in dists) else 0
        spl = 0.0
        stop_success = 0
        success_efficiency = 0
        if success:
            if self.path_length > 0:
                spl = self.initial_distance / max(self.path_length, self.initial_distance)
            stop_success = 1 if self.called_stop else 0
            total = len(dists) - 1
            first = next((k for k, d in enumerate(dists) if d <= succ_d), None)
            # Upstream quirk kept: `first and total` is falsy when first == 0
            # (within success radius before the first step) — efficiency 0.
            success_efficiency = first / total if (first and total) else 0
        try:
            ndtw = calc_ndtw(self.traj, reference_path)
        except Exception:
            ndtw = 0.0
        colrate = (sum(1 for c in self.cols if c) / len(self.cols)) if self.cols else 0.0

        return {
            "episode_id": episode_id,
            "steps_taken": self.steps,
            "path_length": self.path_length,
            "distance_to_goal": final_dist,
            "success": success,
            "stop_success": stop_success,
            "oracle_success": oracle,
            "success_efficiency": success_efficiency,
            "spl": spl,
            "ndtw": ndtw,
            "colrate": colrate,
            "initial_distance": self.initial_distance,
            "end_reason": self.end_reason,
            "moves": self.moves,
            "called_stop": self.called_stop,
        }
