"""slamsam_r2r_01 bridge — frozen per-experiment copy (exp_workspace rule).

Arm: jian's SLAM-map shape (step + get_map) + our SAM landmark layer + recall
of the seen frames, over THIS folder's env_habitat nodeset copy, R2R-CE
rand100. Map = SLAM side-car in map v1 (jian slam_r2r_01: single-floor
OccupancyMap + the frontier renderer with stable ids) with the async SAM
landmark layer; 15° turns; the SAM phrases are the episode's FIXED
keywords (bridges/keywords/rand100_keywords.json, r2r_rand100 split, keyed
by instruction text — no model call), synonyms 0. Every knob below is BAKED
— there is no runtime switch in this folder; fork the folder to change one.

Env contract — the driver's habitat bridge env (bridge_env default family):
  HABITAT_SERVER_URL   env auto_host base URL
  HABITAT_VERB_PREFIX  env_habitat
  HABITAT_STEP_BUDGET  movement budget reported to the model
  HABITAT_LIVE_DIR     frames + map_latest.png land here (+ bootstrap.json)
The episode instruction (for the SAM phrase lookup) is read from the env
itself (env_habitat__reset — a pure ensure-live read of the SEATED episode).
Deployment (not experiment) config:
  LEAN_SAM_URL     SAM 3 auto_host (default http://127.0.0.1:9220)
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

_HERE = Path(__file__).resolve().parent
_CODING_AGENT = _HERE.parents[1]
for _p in (str(_CODING_AGENT), str(_CODING_AGENT / "harnesses" / "mini"),
           str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(f"_toolset_{_HERE.name}",
                                     _HERE / "toolset.py")
_lt = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lt)

# ── frozen knobs of THIS experiment ─────────────────────────────────────
MAP_MODE = "v1"
TURN_DEG = 15.0
TASK = "vln"
SAM_SYNONYMS = 0
KEYWORDS_PATH = _CODING_AGENT / "bridges" / "keywords" / "rand100_keywords.json"
CAM_HEIGHT_FALLBACK_M = 1.25

# ── generic exp env ─────────────────────────────────────────────────────
SERVER_URL = os.environ.get("HABITAT_SERVER_URL", "http://127.0.0.1:9203")
VERB = os.environ.get("HABITAT_VERB_PREFIX", "env_habitat")
STEP_BUDGET = int(os.environ.get("HABITAT_STEP_BUDGET", "500"))
LIVE_DIR = (Path(os.environ["HABITAT_LIVE_DIR"])
            if os.environ.get("HABITAT_LIVE_DIR") else None)
SAM_URL = os.environ.get("LEAN_SAM_URL", "http://127.0.0.1:9220")


def _episode_instruction() -> str:
    try:
        import requests
        r = requests.post(f"{SERVER_URL}/call/{VERB}__reset",
                          json={"inputs": {"trigger": "bridge"}}, timeout=120)
        r.raise_for_status()
        return str((r.json().get("outputs") or {}).get("instruction") or "")
    except Exception:  # noqa: BLE001 — no instruction → no phrases, no crash
        return ""


INSTRUCTION = _episode_instruction()


def phrases_for(instruction: str) -> list[str]:
    """The episode's fixed detector words (rand100_keywords.json, keyed by
    instruction text); falls back to the noun heuristic. Never a model call."""
    instr = instruction.strip()
    if not instr:
        return []
    try:
        data = json.loads(KEYWORDS_PATH.read_text())
        for rows in (data.get("splits") or {}).values():
            for r in rows:
                cand = str(r.get("instruction") or "")
                if cand == instr or " ".join(cand.split()) == " ".join(instr.split()):
                    marks = [str(x) for x in (r.get("landmarks") or []) if x]
                    if marks:
                        return marks
    except Exception:  # noqa: BLE001 — no table, fall through
        pass
    try:
        from eharness.landmarks import landmark_phrases
        return list(landmark_phrases([instr]))
    except Exception:  # noqa: BLE001
        return []


PHRASES = phrases_for(INSTRUCTION)

mcp = FastMCP("habitat-env")
_ts: Any = None


def _ensure() -> Any:
    global _ts
    if _ts is None:
        _ts = _lt.LeanSlamToolSet(
            SERVER_URL, verb=VERB, phrases=PHRASES, step_budget=STEP_BUDGET,
            live_dir=LIVE_DIR, sam_url=SAM_URL, turn_deg=TURN_DEG, task=TASK,
            sam_synonyms=SAM_SYNONYMS, map_mode=MAP_MODE,
            cam_height_fallback_m=CAM_HEIGHT_FALLBACK_M)
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


@mcp.tool(description=_lt.get_map_desc(MAP_MODE))
def get_map() -> list:
    return _to_mcp(_ensure().execute("get_map", {}))


@mcp.tool(description=_lt.RECALL_DESC)
def recall(kind: str, id: int | None = None, top_k: int | None = None,
           start: int | None = None, end: int | None = None) -> list:
    args: dict[str, Any] = {"kind": kind}
    for k, v in (("id", id), ("top_k", top_k), ("start", start), ("end", end)):
        if v is not None:
            args[k] = v
    return _to_mcp(_ensure().execute("recall", args))


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
