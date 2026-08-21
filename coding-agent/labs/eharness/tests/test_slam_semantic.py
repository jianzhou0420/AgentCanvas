"""Fusion unit test for the slamr2r phase-2 SAM semantic layer (§33).

Runs under the agentcanvas env (numpy/PIL only — no habitat import): the
stamping math must put a centre-of-frame mask at depth d into the occupancy
cell d metres forward of the start, and the landmark payload must report it
dead ahead at that range. Guards the OpenCV-pose/world convention the port
inherited from jian's OccupancyMap.integrate.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np

# load _semantic.py standalone — the package __init__ needs the backend's
# `app` framework, which the test env rightly does not have
import importlib.util as _ilu  # noqa: E402

_p = (Path(__file__).resolve().parents[3] / "workspace" / "nodesets" / "env"
      / "env_slam_vlnce" / "_semantic.py")
_spec = _ilu.spec_from_file_location("_semantic", _p)
_semantic = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_semantic)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def _mask_b64(h: int, w: int, box: tuple) -> str:
    from PIL import Image
    m = np.zeros((h, w), np.uint8)
    m[box[0]:box[1], box[2]:box[3]] = 255
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


lay = _semantic.SemanticSamLayer("http://unused", ["sofa"], n=480, origin=240,
                                 cell_size=0.10, camera_height=1.25)
h = w = 64
f = (w / 2) / np.tan(np.radians(45))
intr = {"fx": f, "fy": f, "cx": w / 2, "cy": h / 2, "width": w, "height": h}
depth = np.full((h, w), 2.0, np.float32)
lay._stamp({"sofa": [{"score": 0.9,
                      "mask_b64": _mask_b64(h, w, (24, 40, 24, 40))}]},
           depth, np.eye(4), intr)
g = lay.grids["sofa"]
zi, xi = np.nonzero(g > 0)
check("centre mask at 2 m stamps ~20 cells forward of origin",
      zi.size > 0 and set(zi) == {260}, f"zi={sorted(set(zi))}")
check("…centred on the start column", 240 in set(xi), f"xi={sorted(set(xi))}")
lm = lay.landmarks_json((0.0, 0.0), 0.0)
check("landmark payload: dead ahead at ~2 m",
      len(lm) == 1 and abs(lm[0]["dir_deg"]) < 5 and 1.8 < lm[0]["dist_m"] < 2.3,
      str(lm))

# rotated capture pose: same mask through a 90°-right-yaw pose lands right
c, s = 0.0, 1.0   # cos90, sin90 — camera forward (+z_cam) maps to world +x
pose_r = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
                  float)
lay2 = _semantic.SemanticSamLayer("http://unused", ["sofa"], n=480, origin=240,
                                  cell_size=0.10, camera_height=1.25)
lay2._stamp({"sofa": [{"score": 0.9,
                       "mask_b64": _mask_b64(h, w, (24, 40, 24, 40))}]},
            depth, pose_r, intr)
zi2, xi2 = np.nonzero(lay2.grids["sofa"] > 0)
check("capture pose honoured: 90°-right pose stamps +x, not +z",
      xi2.size > 0 and set(xi2) == {260}
      and min(zi2) < 240 < max(zi2),
      f"zi={sorted(set(zi2))} xi={sorted(set(xi2))}")

phr = _semantic.phrases_for_episode(
    str(Path(__file__).resolve().parents[2] / "bridges" / "keywords"
        / "rand100_keywords.json"), "7")
check("keyword lookup by episode_id", phr == ["pool", "bar", "chairs"],
      str(phr))


# ── §35 region segmentation + naming (synthetic two rooms + corridor) ──
_rp = (Path(__file__).resolve().parents[3] / "workspace" / "nodesets" / "env"
       / "env_slam_vlnce" / "_regions.py")
_rspec = _ilu.spec_from_file_location("_regions", _rp)
_regions = _ilu.module_from_spec(_rspec)
_rspec.loader.exec_module(_regions)

# 60x60 grid, 0.1 m cells: room A cells [5:25]x[5:25], room B [5:25]x[35:55],
# joined by a 3-cell-wide corridor at rows 14-16 — the neck's EDT < 0.65 m
# so cores never bridge
_g = np.zeros((60, 60), np.uint8)
_g[5:25, 5:25] = 1
_g[5:25, 35:55] = 1
_g[14:17, 25:35] = 1
_seg = _regions.segment_rooms(_g, 0.10)
_ids = sorted(set(_seg[_seg > 0].tolist()))
check("two rooms + corridor segment into exactly two rooms",
      len(_ids) == 2, f"ids={_ids}")
check("…and the two room interiors get DIFFERENT ids",
      _seg[15, 15] != _seg[15, 45] and _seg[15, 15] > 0 and _seg[15, 45] > 0,
      f"A={_seg[15, 15]} B={_seg[15, 45]}")

# naming: bed mass in room A, sofa mass in room B, below-floor junk in corridor
_bed = np.zeros((60, 60), np.float32); _bed[10:13, 10:13] = 1.0
_sofa = np.zeros((60, 60), np.float32); _sofa[10:13, 45:48] = 1.0
_names = _regions.vote_names(_seg, {"bed": _bed, "sofa": _sofa})
_by_room = {_seg[11, 11]: "bedroom", _seg[11, 46]: "living room"}
check("lexicon vote names each room from its own furniture",
      all(_names.get(k, ("", 0))[0] == v for k, v in _by_room.items()),
      str(_names))
_rj = _regions.regions_json(_seg, _names, origin=30, cell_size=0.10,
                            agent_xz=(0.0, 0.0), yaw_rad=0.0)
check("regions payload carries name/cells/confidence/dir/dist",
      len(_rj) == 2 and all(set(r) >= {"name", "cells", "confidence",
                                       "dir_deg", "dist_m"} for r in _rj),
      str(_rj)[:120])
_weak = _regions.vote_names(_seg, {"bed": _bed * 0.1})
check("insufficient evidence names NOTHING (no guessing)",
      _weak == {}, str(_weak))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("slam semantic fusion: all passed")
