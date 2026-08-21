"""Async SAM3 semantic layer over the SLAM occupancy map (phase 2, 2026-08-17).

OUR addition on top of the verbatim jian port (everything else in this
package mirrors /data/ws_vln/vlnworkspace): landmark phrases from the
episode's keyword table are segmented by the SAM3 auto_host and stamped —
ASYNCHRONOUSLY — into per-phrase score grids aligned cell-for-cell with the
port's ``OccupancyMap``. The eharness P1-1 contract is kept:

- submit early: one background worker, jobs submitted at coarse-action
  boundaries with the CAPTURE pose (4x4 world-from-camera, cv axes — the
  exact pose ``OccupancyMap.integrate`` used for that frame) frozen in;
- keep-latest: a queued-not-started job is superseded by a newer frame;
- merge at the consumption point: ``get_map`` drains every FINISHED job
  before rendering, waits at most ``WAIT_NEWEST_S`` on the newest pending
  one (an instrument query must stay snappy), and reports what is pending.

Projection reuses the port's own math: mask pixels x depth through the
capture pose, keep points in the obstacle/furniture height band plus a
low band for floor-marked things (rugs, pool edges), take world (x, z),
``world_to_cell``. No new geometry conventions are introduced.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import math
import os
import threading
import time
from typing import Any

import numpy as np

WAIT_NEWEST_S = 2.0
_ANCHOR_EVERY = 6         # §35: room-anchor vocab rides every Nth submit
SCORE_MIN = 0.35          # SAM instance score floor
CELL_HIT_CAP = 6.0        # per-cell accumulation cap (score-weighted votes)
CELL_SHOW_MIN = 0.9       # a cell needs ~2 solid votes to render
MAX_PHRASES = 6
PX_STRIDE = 3

# distinct, occupancy-palette-safe tints (map is gray/white/black)
_TINTS = ((246, 130, 210), (130, 200, 255), (250, 176, 90),
          (150, 246, 150), (230, 230, 120), (190, 150, 255))


def phrases_for_episode(keywords_path: str, episode_id: str,
                        split_key: str = "r2r_rand100") -> list:
    """Landmark phrases for one episode from the frozen keyword table."""
    try:
        with open(keywords_path) as fh:
            table = json.load(fh)
        for row in table.get("splits", {}).get(split_key, []):
            if str(row.get("episode_id")) == str(episode_id):
                return list(row.get("landmarks") or [])[:MAX_PHRASES]
    except Exception:  # noqa: BLE001 — semantics stay advisory
        pass
    return []


class SemanticSamLayer:
    """Per-phrase score grids the size of the occupancy grid, fed by an
    async SAM worker; overlay data consumed by the annotated renderer."""

    def __init__(self, sam_url: str, phrases: list, n: int, origin: int,
                 cell_size: float, camera_height: float) -> None:
        self.sam_url = sam_url.rstrip("/")
        self.phrases = list(phrases)
        self.n, self.origin, self.cell_size = n, origin, cell_size
        self.camera_height = camera_height
        self.grids: dict[str, np.ndarray] = {
            p: np.zeros((n, n), np.float32) for p in self.phrases}
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slam_sam")
        self._jobs: list[dict] = []
        self._lock = threading.Lock()
        self.stats = {"submitted": 0, "merged": 0, "merged_late": 0,
                      "superseded": 0, "errors": 0, "wait_ms": 0.0}

    # ── submit (coarse-action boundary; pose frozen at capture) ──────────

    def submit(self, rgb_b64: str, depth_m: np.ndarray, pose_wc: np.ndarray,
               intrinsics: dict, tick: int) -> None:
        if not self.phrases or pose_wc is None:
            return
        with self._lock:
            kept = []
            for job in self._jobs:
                if not job["future"].done() and job["future"].cancel():
                    self.stats["superseded"] += 1
                else:
                    kept.append(job)
            self._jobs = kept
            # §35 region naming: every ANCHOR_EVERY-th submit carries the
            # fixed room-anchor vocabulary too — same worker, same
            # keep-latest contract, zero extra blocking
            phrases = list(self.phrases)
            # anchor vocab feeds ONLY the region layer — when regions are
            # off (user 2026-08-17 night default) don't spend SAM on it
            if (os.environ.get("SLAMENV_REGIONS", "0") == "1"
                    and self.stats["submitted"] % _ANCHOR_EVERY == 0):
                from ._regions import ANCHOR_VOCAB
                phrases += [p for p in ANCHOR_VOCAB if p not in phrases]
            for p in phrases:
                if p not in self.grids:
                    self.grids[p] = np.zeros((self.n, self.n), np.float32)
            self._jobs.append({
                "future": self._pool.submit(self._segment, rgb_b64, phrases),
                "depth": np.asarray(depth_m, np.float32).copy(),
                "pose": np.asarray(pose_wc, np.float64).copy(),
                "intrinsics": dict(intrinsics),
                "tick": int(tick)})
            self.stats["submitted"] += 1

    def _segment(self, rgb_b64: str, phrases: list | None = None) -> dict:
        """Worker thread: one SAM call per phrase (model_sam__segment_text —
        the same auto_host verb the eharness organ uses)."""
        import requests
        out: dict[str, list] = {}
        for phrase in (phrases if phrases is not None else self.phrases):
            body = {"inputs": {"image_b64": rgb_b64, "text": phrase}}
            try:
                r = requests.post(
                    f"{self.sam_url}/call/model_sam__segment_text",
                    json=body, timeout=60)
                if r.status_code != 200:
                    continue
                envelope = json.loads(r.json()["outputs"]["masks"] or "{}")
                out[phrase] = list(envelope.get("masks") or [])[:3]
            except Exception:  # noqa: BLE001 — semantics stay advisory
                continue
        return out

    # ── merge (consumption point) ────────────────────────────────────────

    def drain(self, *, wait_newest_s: float = WAIT_NEWEST_S) -> int:
        """Merge every finished job (each at ITS OWN capture pose); block at
        most ``wait_newest_s`` on the newest still-running one. Returns the
        number of jobs still pending after the sweep."""
        with self._lock:
            jobs = list(self._jobs)
        if jobs and not jobs[-1]["future"].done() and wait_newest_s > 0:
            t0 = time.time()
            try:
                jobs[-1]["future"].result(timeout=wait_newest_s)
            except Exception:  # noqa: BLE001 — pending is a legal state
                pass
            self.stats["wait_ms"] = round(
                self.stats["wait_ms"] + (time.time() - t0) * 1000.0, 1)
        remaining: list[dict] = []
        newest_tick = jobs[-1]["tick"] if jobs else 0
        for job in jobs:
            if not job["future"].done():
                remaining.append(job)
                continue
            try:
                masks_by_phrase = job["future"].result()
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
                continue
            self._stamp(masks_by_phrase, job["depth"], job["pose"],
                        job["intrinsics"])
            self.stats["merged"] += 1
            if job["tick"] != newest_tick:
                self.stats["merged_late"] += 1
        with self._lock:
            self._jobs = remaining
        return len(remaining)

    def _stamp(self, masks_by_phrase: dict, depth_m: np.ndarray,
               pose_wc: np.ndarray, intr: dict) -> None:
        """Project mask pixels through the CAPTURE pose into grid cells —
        the port's own integrate() math, phrase-scored instead of ternary."""
        h, w = depth_m.shape
        fx = intr["fx"] * w / intr["width"]
        fy = intr["fy"] * h / intr["height"]
        cx = intr["cx"] * w / intr["width"]
        cy = intr["cy"] * h / intr["height"]
        for phrase, masks in masks_by_phrase.items():
            grid = self.grids.get(phrase)
            if grid is None:
                continue
            for m in masks:
                # the SAM host's wire schema scores as iou_score (probed
                # 2026-08-17: keys = mask_index/mask_b64/bbox_xyxy/
                # iou_score/area); "score" kept as fallback
                score = float(m.get("iou_score") or m.get("score") or 0.0)
                if score < SCORE_MIN:
                    continue
                seg = m.get("mask_b64") or m.get("counts") or m.get("mask")
                mask = _decode_mask(seg, h, w)
                if mask is None:
                    continue
                vs, us = np.nonzero(mask[::PX_STRIDE, ::PX_STRIDE])
                vs, us = vs * PX_STRIDE, us * PX_STRIDE
                z = depth_m[vs, us].astype(np.float64)
                ok = (z > 0.3) & (z < 6.0)
                us, vs, z = us[ok], vs[ok], z[ok]
                if z.size == 0:
                    continue
                x_cam = (us - cx) / fx * z
                y_cam = (vs - cy) / fy * z
                pts = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
                pw = pose_wc @ pts
                height = self.camera_height - pw[1]
                band = (height > -0.2) & (height < 2.0)
                xs = np.floor(pw[0][band] / self.cell_size).astype(int) + self.origin
                zs = np.floor(pw[2][band] / self.cell_size).astype(int) + self.origin
                ok = (xs >= 0) & (xs < self.n) & (zs >= 0) & (zs < self.n)
                np.add.at(grid, (zs[ok], xs[ok]),
                          score * (PX_STRIDE ** 2) / 25.0)
            np.clip(self.grids[phrase], 0.0, CELL_HIT_CAP,
                    out=self.grids[phrase])

    # ── consumption views ────────────────────────────────────────────────

    def overlay_cells(self) -> list:
        """[(phrase, bool grid, tint)] for the annotated renderer.
        Episode phrases only — the §35 anchor vocabulary feeds the region
        VOTE, not the patch overlay (seven extra named patches would bury
        the instruction's own landmarks)."""
        out = []
        for k, (phrase, grid) in enumerate(sorted(self.grids.items())):
            if phrase not in self.phrases:
                continue
            hit = grid >= CELL_SHOW_MIN
            if hit.any():
                out.append((phrase, hit, _TINTS[k % len(_TINTS)]))
        return out

    def landmarks_json(self, agent_xz, yaw_rad) -> list:
        """[{name, dir_deg (right +, relative heading), dist_m}] PER
        CLUSTER (validation 2026-08-17 §34: one centroid over a
        multi-instance category aliased different chair groups 3.35 m
        apart — per-cluster geometry agreed to 0.05 m). At most two
        clusters per phrase, largest first; a second one is suffixed
        " (another)" so the model reads them as separate instances."""
        from scipy import ndimage
        out = []
        for phrase, grid in sorted(self.grids.items()):
            if phrase not in self.phrases:
                continue   # anchor vocab feeds region votes, not landmarks
            hit = grid >= CELL_SHOW_MIN
            if not hit.any():
                continue
            # dilate before labelling (the eharness _sem_components move):
            # stride-sampled stamps leave 1-cell gaps that are one object
            lab, n = ndimage.label(
                ndimage.binary_dilation(hit, iterations=1))
            comps = []
            for k in range(1, n + 1):
                sel = hit & (lab == k)
                zi, xi = np.nonzero(sel)
                if zi.size < 3:
                    continue
                comps.append((zi, xi))
            comps.sort(key=lambda c: -c[0].size)
            for j, (zi, xi) in enumerate(comps[:2]):
                wgt = grid[zi, xi]
                cxw = float(np.average(
                    (xi.astype(float) - self.origin + 0.5) * self.cell_size,
                    weights=wgt))
                czw = float(np.average(
                    (zi.astype(float) - self.origin + 0.5) * self.cell_size,
                    weights=wgt))
                entry = {"name": phrase + (" (another)" if j else ""),
                         "cells": int(zi.size)}
                if agent_xz is not None and yaw_rad is not None:
                    bearing = math.atan2(cxw - agent_xz[0],
                                         czw - agent_xz[1])
                    d = ((bearing - yaw_rad) + math.pi) % (2 * math.pi) \
                        - math.pi
                    entry["dir_deg"] = round(math.degrees(d), 1)
                    entry["dist_m"] = round(
                        math.hypot(cxw - agent_xz[0], czw - agent_xz[1]), 2)
                out.append(entry)
        return out


def _decode_mask(seg: Any, h: int, w: int):
    """Mask decode tolerant of the SAM host's wire formats: b64 PNG or a
    nested list. Returns bool (h, w) or None."""
    try:
        if isinstance(seg, str):
            from PIL import Image
            arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(seg))))
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.shape != (h, w):
                from PIL import Image as _I
                arr = np.asarray(_I.fromarray(arr).resize((w, h)))
            return arr > 127
        if isinstance(seg, list):
            arr = np.asarray(seg)
            if arr.shape == (h, w):
                return arr.astype(bool)
    except Exception:  # noqa: BLE001
        return None
    return None


SAM_URL_ENV = "SLAMENV_SAM_URL"
KEYWORDS_ENV = "SLAMENV_SAM_KEYWORDS"


def build_layer(episode_id: str, occ: Any) -> "SemanticSamLayer | None":
    """Factory the manager calls at seat time when the sam_semantic arm is
    on. None when the server was launched without the SAM env vars."""
    url = os.environ.get(SAM_URL_ENV, "")
    kw = os.environ.get(KEYWORDS_ENV, "")
    if not url or not kw:
        return None
    phrases = phrases_for_episode(kw, episode_id)
    if not phrases:
        return None
    return SemanticSamLayer(url, phrases, occ.n, occ.origin, occ.cell_size,
                            occ.camera_height)
