from __future__ import annotations

"""TUM RGB-D sequence reader — pure numpy/PIL, no simulator, no GPL.

Parses a TUM RGB-D benchmark sequence directory (Sturm et al., IROS 2012):
``rgb/`` + ``depth/`` PNGs plus the ``rgb.txt`` / ``depth.txt`` /
``groundtruth.txt`` index files. RGB, depth, and mocap ground-truth are each
sampled on their own clock, so the reader time-associates them by nearest
timestamp (the classic ``associate.py``, done here with a bisect nearest
lookup instead of the O(N·M) greedy so a 2500-frame sequence indexes in
milliseconds).

What each frame yields (see :meth:`TumSequence.load_frame`):
  - ``rgb``   : HxWx3 uint8
  - ``depth`` : HxW float32 in **metres** (PNG uint16 ÷ 5000, the TUM factor)
  - ``pose``  : ground-truth camera pose ``{position:[x,y,z],
                orientation:[qx,qy,qz,qw]}`` (world-from-camera), or None when
                no mocap sample falls within tolerance
  - ``timestamp`` : the RGB frame timestamp (seconds)

Camera intrinsics are the published per-``freiburg`` ROS defaults, keyed off
the sequence directory name.
"""

import bisect
import os
from typing import Any

import numpy as np

# ── Per-camera pinhole intrinsics (TUM published ROS defaults) ─────────────
# https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
# All TUM RGB-D sequences are 640x480; depth PNG is uint16 with a 5000 factor
# (depth_metres = png / 5000).
_DEPTH_FACTOR = 5000.0
_WIDTH, _HEIGHT = 640, 480

_INTRINSICS: dict[str, dict[str, float]] = {
    "freiburg1": {"fx": 517.306408, "fy": 516.469215, "cx": 318.643040, "cy": 255.313989},
    "freiburg2": {"fx": 520.908620, "fy": 521.007327, "cx": 325.141442, "cy": 249.701764},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
    # ROS default — used when the sequence name carries no freiburg tag.
    "default": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
}


def freiburg_group(seq_name: str) -> str:
    """Map a sequence dir name to its intrinsics group (freiburg1/2/3)."""
    low = seq_name.lower()
    for grp in ("freiburg1", "freiburg2", "freiburg3"):
        if grp in low:
            return grp
    return "default"


def intrinsics_for(seq_name: str) -> dict[str, Any]:
    """Full pinhole intrinsics dict {fx,fy,cx,cy,width,height} for a sequence."""
    k = _INTRINSICS.get(freiburg_group(seq_name), _INTRINSICS["default"])
    return {**k, "width": _WIDTH, "height": _HEIGHT}


# ── Index-file parsing + timestamp association ─────────────────────────────


def _read_index(path: str) -> list[tuple[float, str]]:
    """Parse a TUM index file → sorted [(timestamp, rest_of_line)].

    Lines are ``timestamp token[ token...]`` (rgb.txt/depth.txt: one filename;
    groundtruth.txt: ``tx ty tz qx qy qz qw``). Comments (``#``) skipped.
    """
    out: list[tuple[float, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                continue
            out.append((ts, parts[1]))
    out.sort(key=lambda x: x[0])
    return out


def _nearest(query_ts: float, ref_ts: list[float], max_diff: float) -> int | None:
    """Index of the ref timestamp nearest ``query_ts`` within ``max_diff`` (s)."""
    if not ref_ts:
        return None
    pos = bisect.bisect_left(ref_ts, query_ts)
    best: int | None = None
    best_d = max_diff
    for k in (pos - 1, pos):
        if 0 <= k < len(ref_ts):
            d = abs(ref_ts[k] - query_ts)
            if d <= best_d:
                best_d = d
                best = k
    return best


class TumSequence:
    """A time-associated TUM RGB-D sequence — RGB↔depth↔ground-truth frames.

    Frame paths are indexed eagerly (cheap); the actual PNGs decode lazily in
    :meth:`load_frame` so a long sequence never sits fully in RAM.
    """

    def __init__(self, seq_dir: str, max_diff: float = 0.02) -> None:
        self.seq_dir = os.path.abspath(seq_dir)
        self.name = os.path.basename(self.seq_dir.rstrip("/"))
        self.intrinsics = intrinsics_for(self.name)

        rgb = _read_index(os.path.join(self.seq_dir, "rgb.txt"))
        depth = _read_index(os.path.join(self.seq_dir, "depth.txt"))
        gt_path = os.path.join(self.seq_dir, "groundtruth.txt")
        gt = _read_index(gt_path) if os.path.exists(gt_path) else []

        depth_ts = [t for t, _ in depth]
        gt_ts = [t for t, _ in gt]

        # Each RGB frame → nearest depth (required) + nearest GT pose (optional).
        self._frames: list[dict[str, Any]] = []
        for ts, rgb_name in rgb:
            di = _nearest(ts, depth_ts, max_diff)
            if di is None:
                continue  # no depth within tolerance — RGB-D needs both
            gi = _nearest(ts, gt_ts, max_diff)
            self._frames.append(
                {
                    "timestamp": ts,
                    "rgb": os.path.join(self.seq_dir, rgb_name),
                    "depth": os.path.join(self.seq_dir, depth[di][1]),
                    "gt": gt[gi][1] if gi is not None else None,
                }
            )

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    def num_frames(self, cap: int = 0) -> int:
        """Effective frame count — ``min(cap, total)`` when ``cap>0``."""
        return min(cap, len(self._frames)) if cap and cap > 0 else len(self._frames)

    @staticmethod
    def _parse_gt(rest: str) -> dict[str, Any] | None:
        """``tx ty tz qx qy qz qw`` → {position, orientation} pose dict."""
        toks = rest.split()
        if len(toks) < 7:
            return None
        v = [float(x) for x in toks[:7]]
        return {"position": v[0:3], "orientation": v[3:7]}

    def load_frame(self, i: int) -> dict[str, Any]:
        """Decode frame ``i`` → {rgb uint8, depth float32 m, pose|None, timestamp}."""
        from PIL import Image  # Pillow ships in the agentcanvas env

        if i < 0:
            i = 0
        if i >= len(self._frames):
            i = len(self._frames) - 1
        fr = self._frames[i]
        rgb = np.asarray(Image.open(fr["rgb"]).convert("RGB"), dtype=np.uint8)
        # TUM depth PNG is uint16 in 0.2 mm units (factor 5000) → metres.
        depth_png = np.asarray(Image.open(fr["depth"]), dtype=np.float32)
        depth = depth_png / _DEPTH_FACTOR
        return {
            "rgb": rgb,
            "depth": depth,
            "pose": self._parse_gt(fr["gt"]) if fr["gt"] else None,
            "timestamp": fr["timestamp"],
        }

    def gt_path_length(self, cap: int = 0) -> float:
        """Total ground-truth translation path length (m) over the used frames."""
        n = self.num_frames(cap)
        pts = []
        for k in range(n):
            g = self._frames[k]["gt"]
            if g:
                toks = g.split()
                if len(toks) >= 3:
                    pts.append([float(toks[0]), float(toks[1]), float(toks[2])])
        if len(pts) < 2:
            return 0.0
        arr = np.asarray(pts, dtype=np.float64)
        return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
