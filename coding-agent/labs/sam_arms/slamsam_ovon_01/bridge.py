"""slamsam_ovon_01 bridge — frozen per-experiment copy (exp_workspace rule).

Arm: HM3D-OVON ObjectNav on jian's SLAM-instrument surface (auto-observe:
step carries the view; get_pose / get_map / get_trajectory) over THIS
folder's env_ovon nodeset copy, with the SAM 3 landmark layer keyed to the
ONE goal word and the TARGET PUSH (a confirmed detection cuts the running
step short and hands the model direction + distance + the map; every later
result carries the target's dir/dist from the current pose so the model can
close in under 1 m). Map v1 (jian slam_r2r_01: single-floor OccupancyMap +
numbered stable-id frontiers). Every knob below is BAKED — no runtime switch
in this folder; fork the folder to change one.

Env contract — the driver's ObjectNav-family bridge env (bridge_env for
benchmark ovon-*, jian efe0397):
  OBJNAV_SERVER_URL   env auto_host base URL
  OBJNAV_VERB_PREFIX  env_ovon
  OBJNAV_STEP_BUDGET  movement budget reported to the model
  OBJNAV_LIVE_DIR     frames + map_latest.png land here (+ bootstrap.json)
The goal word is read from the env itself (env_ovon__reset — a pure read
of the SEATED episode: the driver seats before this bridge spawns).
Deployment (not experiment) config:
  LEAN_SAM_URL        SAM 3 auto_host (default http://127.0.0.1:9220)
"""

from __future__ import annotations

import base64
import importlib.util as _ilu
import json
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
CAM_HEIGHT_FALLBACK_M = 0.88    # ObjectNav agent camera height

# ── the driver's ObjectNav bridge env ────────────────────────────────────
SERVER_URL = os.environ.get("OBJNAV_SERVER_URL", "http://127.0.0.1:9241")
VERB = os.environ.get("OBJNAV_VERB_PREFIX", "env_ovon")
STEP_BUDGET = int(os.environ.get("OBJNAV_STEP_BUDGET", "500"))
LIVE_DIR = (Path(os.environ["OBJNAV_LIVE_DIR"])
            if os.environ.get("OBJNAV_LIVE_DIR") else None)
SAM_URL = os.environ.get("LEAN_SAM_URL", "http://127.0.0.1:9220")


def _goal_word() -> str:
    """The seated episode's open-vocabulary goal (env_ovon reset = ensure-live
    read; the driver placed the episode before spawning this process)."""
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
            map_mode=MAP_MODE, cam_height_fallback_m=CAM_HEIGHT_FALLBACK_M)
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


@mcp.tool(description=_lt.step_desc(TURN_DEG, TASK))
def step(actions: list[int]) -> list:
    return _to_mcp(_ensure().execute("step", {"actions": actions}))


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
