"""slamsam_objnav_02 bridge — frozen per-experiment copy (exp_workspace rule).

Arm: ObjectNav (HM3D / MP3D via env_objnav, HM3D-OVON via env_ovon) on
jian's SLAM-instrument surface (auto-observe: step carries the view;
get_pose / get_map / get_trajectory) with SAM 3 as an EXTERNAL TOOL:
detect_target() runs the detector for the exact goal word on the view the
model is looking at and returns the overlay + per-instance direction /
distance / score; only matches scoring ≥ SAM_SCORE_THRESH (0.85; the user's
band was "0.8 或者 0.85") are returned, and ONLY those high-score matches are
stamped on the map (user 2026-08-18: "只有在它的 score 很高 (比如 0.85 以上) 的
情况下再标"). No detector watches the frames while walking — pure ReAct: the
model asks when it wants to know. Strict detection: the goal word only (no
synonyms — a cushion is not a pillow). FINAL PUSH on STOP (user: "人为地把这个
往前推, 离目标足够近"): with the target detected in view the harness faces it
and walks it down to ~PUSH_STOP_M (or blocked / out of sight / cap) before
the STOP primitive goes out. Map v1 (jian slam_r2r_01: single-floor
OccupancyMap + numbered stable-id frontiers). Every knob below is BAKED — no
runtime switch in this folder; fork the folder to change one.

Env contract — the driver's ObjectNav-family bridge env (bridge_env for
benchmark hm3d / mp3d / ovon-*, jian efe0397):
  OBJNAV_SERVER_URL   env auto_host base URL
  OBJNAV_VERB_PREFIX  env_objnav (hm3d / mp3d) | env_ovon (ovon-*)
  OBJNAV_STEP_BUDGET  movement budget reported to the model
  OBJNAV_LIVE_DIR     frames + map_latest.png land here (+ bootstrap.json)
The goal word is read from the env itself (<verb>__reset — a pure read of
the SEATED episode: the driver seats before this bridge spawns; env_objnav
categories carry dataset-internal underscores: "tv_monitor" -> "tv monitor").
Deployment (not experiment) config:
  LEAN_SAM_URL        SAM 3 auto_host (default http://127.0.0.1:9220)
"""

from __future__ import annotations

import base64
import importlib.util as _ilu
import os
import sys
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP, Image

_HERE = Path(__file__).resolve().parent
_CODING_AGENT = _HERE.parents[1]
for _p in (str(_CODING_AGENT), str(_CODING_AGENT / "harnesses" / "mini"),
           str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = _ilu.spec_from_file_location(f"_toolset_{_HERE.name}",
                                     _HERE / "toolset.py")
_lt = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lt)

# ── frozen knobs of THIS experiment ─────────────────────────────────────
MAP_MODE = "v1"
TURN_DEG = 30.0                 # ObjectNav / OVON agent
TASK = "objnav"
SAM_SYNONYMS = 0                # the goal word itself, nothing else
SAM_SCORE_THRESH = 0.85         # strict gate (user: 0.8 or 0.85 — the
                                # stricter end); shared organ default 0.5;
                                # the same gate decides what is stamped
CAM_HEIGHT_FALLBACK_M = 0.88    # ObjectNav agent camera height
FINAL_PUSH = True               # STOP → face the detected target, walk it down
PUSH_STOP_M = 0.5               # …until about this close
PUSH_MAX_STEPS = 8              # …at most this many forwards (2 m)

# ── the driver's ObjectNav bridge env ────────────────────────────────────
SERVER_URL = os.environ.get("OBJNAV_SERVER_URL", "http://127.0.0.1:9241")
VERB = os.environ.get("OBJNAV_VERB_PREFIX", "env_ovon")
STEP_BUDGET = int(os.environ.get("OBJNAV_STEP_BUDGET", "500"))
LIVE_DIR = (Path(os.environ["OBJNAV_LIVE_DIR"])
            if os.environ.get("OBJNAV_LIVE_DIR") else None)
SAM_URL = os.environ.get("LEAN_SAM_URL", "http://127.0.0.1:9220")


def _goal_word() -> str:
    """The seated episode's goal category (reset = ensure-live read; the
    driver placed the episode before spawning this process)."""
    try:
        r = requests.post(f"{SERVER_URL}/call/{VERB}__reset",
                          json={"inputs": {"trigger": "bridge"}}, timeout=120)
        r.raise_for_status()
        return str((r.json().get("outputs") or {}).get("object_category")
                   or "").strip().replace("_", " ")
    except Exception:  # noqa: BLE001 — no goal → no detector, never a crash
        return ""


GOAL = _goal_word()

mcp = FastMCP("habitat-env")
_ts: Any = None


def _ensure() -> Any:
    global _ts
    if _ts is None:
        _ts = _lt.LeanSlamToolSet(
            SERVER_URL, verb=VERB, phrases=[GOAL] if GOAL else [],
            step_budget=STEP_BUDGET, live_dir=LIVE_DIR, sam_url=SAM_URL,
            turn_deg=TURN_DEG, task=TASK, sam_synonyms=SAM_SYNONYMS,
            sam_score_thresh=SAM_SCORE_THRESH, map_mode=MAP_MODE,
            cam_height_fallback_m=CAM_HEIGHT_FALLBACK_M,
            final_push=FINAL_PUSH, push_stop_m=PUSH_STOP_M,
            push_max_steps=PUSH_MAX_STEPS)
    return _ts


def _to_mcp(result: Any) -> list:
    out: list[Any] = []
    texts: list[str] = []
    for part in (result.content or []):
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if "," in url:
                if texts:
                    out.append("\n".join(texts))
                    texts = []
                out.append(Image(data=base64.b64decode(url.split(",", 1)[1]),
                                 format="png"))
        elif isinstance(part, dict) and part.get("type") == "text":
            texts.append(str(part.get("text", "")))
    if texts:
        out.append("\n".join(texts))
    return out or ["(empty result)"]


@mcp.tool(description=_lt.step_desc(TURN_DEG, TASK, PUSH_STOP_M))
def step(actions: list[int]) -> list:
    return _to_mcp(_ensure().execute("step", {"actions": actions}))


@mcp.tool(description=_lt.detect_desc(GOAL, SAM_SCORE_THRESH))
def detect_target() -> list:
    return _to_mcp(_ensure().execute("detect_target", {}))


@mcp.tool(description=_lt.GET_POSE_DESC)
def get_pose() -> list:
    return _to_mcp(_ensure().execute("get_pose", {}))


@mcp.tool(description=_lt.get_map_desc(MAP_MODE))
def get_map() -> list:
    return _to_mcp(_ensure().execute("get_map", {}))


@mcp.tool(description=_lt.GET_TRAJECTORY_DESC)
def get_trajectory() -> list:
    return _to_mcp(_ensure().execute("get_trajectory", {}))


if __name__ == "__main__":
    # frame 0 rides the first message (bootstrap.json is the adapter's
    # contract); a stale artifact from an earlier attempt must never be
    # attached as if it were this episode's opening look
    if LIVE_DIR is not None:
        (LIVE_DIR / "bootstrap.json").unlink(missing_ok=True)
    try:
        _ensure().bootstrap()
    except Exception:  # noqa: BLE001 — a failed opening look must not kill
        # the server; the model's first step() shows it the world
        pass
    mcp.run()
