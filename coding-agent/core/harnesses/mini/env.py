"""Habitat environment for mini-swe-agent — thin session owner over the toolset.

mini's Environment role, reduced to what a stateful episode actually needs:
route parsed tool calls into the toolset, and when a call ends the episode
(STOP executed, step budget exhausted) raise ``Submitted`` so the agent loop
exits with the end reason in the trajectory. Episode placement, reset, and
metric collection stay driver-side (run_episodes.py), exactly like the
claude-SDK path — the agent never sees SR/SPL, reward, pose, or panoramas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minisweagent.exceptions import Submitted
from pydantic import BaseModel

from toolset import HabitatToolSet, HybridToolSet, WaypointToolSet


class HabitatEnvironmentConfig(BaseModel):
    server_url: str = "http://127.0.0.1:9200"
    bare: bool = False
    step_budget: int = 500
    turn_budget: int = 0
    pano_view_px: int = 0  # 0 = native render resolution, same as observe()
    live_dir: str = ""
    # waypoint action space (wp condition): a second auto_host (the predictor)
    # and its own decision-step budget replace the step()/clearance surface.
    wp: bool = False
    wp_server_url: str = ""
    wp_max_moves: int = 40
    wp_predict_fn: str = "smartway_waypoint__predict"
    # ImagineVLN (imagine condition): the wp surface plus, on every look, one
    # world-model rollout sheet per numbered candidate. Also turns goto() into
    # goto+auto-observe. imagine=False keeps the rollouts off but KEEPS the
    # auto-observe, so the two arms differ only by the extra images.
    imagine: bool = False
    imagine_rollouts: bool = True
    mw_url: str = "http://127.0.0.1:9270"
    # Agent-selected hybrid interface (hybrid condition): primitive step() AND
    # waypoint goto() in one surface, two observe lenses, look-then-move gate.
    # Uses the same predictor (wp_server_url) as wp; no wp_max_moves cap (a goto
    # just spends more of the shared 500 step budget). Mutually exclusive with wp.
    hybrid: bool = False
    # Depth-waypoint action space (dwp condition): the SAME numbered-candidate
    # surface as wp, but the candidates are derived geometrically from one 90°
    # depth frame instead of predicted by a network — no second auto_host, no
    # 3 m heatmap ceiling, and a straight-line reachability guarantee. Optional
    # SAM 3 landmark grounding rides along on `dwp_sam_url`.
    dwp: bool = False
    dwp_max_moves: int = 60
    dwp_sam_url: str = ""
    dwp_landmark_every: int = 3
    # 0 = follow the organ's own RANGE_CAP_M. A literal here is a second
    # place to remember, and it went stale the moment the grid moved to 12 m:
    # the agent would have kept building a 6 m map while the human tab built a
    # 12 m one, off the same code.
    dwp_range_cap_m: float = 0.0
    dwp_places: int = 0        # collapse operator: place memory replaces metric recall
    # §14.8: 0 = event-driven SAM (post-turn first frame, scene transitions,
    # low-frequency backstop every 4th sensor frame) — the run default.
    # N>0 pins a fixed every-Nth-SENSOR-frame cadence (1 = the correctness
    # upper bound, ~2.5 s/phrase/frame of latency).
    dwp_sam_intermediate_every: int = 0
    dwp_traj_archive: int = 1             # §4.3: per-primitive RGB-D archive on disk
    instruction: str = ""
    sub_instructions: tuple = ()
    terminate: str = ""


class HabitatEnvironment:
    def __init__(self, *, config_class: type = HabitatEnvironmentConfig, **kwargs: Any) -> None:
        self.config = config_class(**kwargs)
        live_dir = Path(self.config.live_dir) if self.config.live_dir else None
        if self.config.hybrid:
            self.toolset = HybridToolSet(
                self.config.server_url,
                wp_server_url=self.config.wp_server_url,
                predict_fn=self.config.wp_predict_fn,
                step_budget=self.config.step_budget,
                turn_budget=self.config.turn_budget,
                pano_view_px=self.config.pano_view_px,
                live_dir=live_dir,
            )
        elif self.config.dwp:
            from depth_toolset import DepthWaypointToolSet

            self.toolset = DepthWaypointToolSet(
                self.config.server_url,
                instruction=self.config.instruction,
                segments=list(self.config.sub_instructions),
                terminate=self.config.terminate,
                bare=self.config.bare,
                step_budget=self.config.step_budget,
                turn_budget=self.config.turn_budget,
                pano_view_px=self.config.pano_view_px,
                live_dir=live_dir,
                sam_url=self.config.dwp_sam_url,
                landmark_every=self.config.dwp_landmark_every,
                **({"range_cap_m": self.config.dwp_range_cap_m}
                   if self.config.dwp_range_cap_m else {}),
                places=self.config.dwp_places,
                sam_intermediate_every=self.config.dwp_sam_intermediate_every,
                traj_archive=bool(self.config.dwp_traj_archive),
                executor="mini",
            )
        elif self.config.imagine:
            import os as _os
            import sys as _sys
            _iv = _os.environ.get(
                "IMAGINEVLN_AGENT",
                _os.path.expanduser("~/Desktop/Projects/ImagineVLN/agent"))
            if _iv not in _sys.path:
                _sys.path.insert(0, _iv)
            from imagine_toolset import ImagineWaypointToolSet

            self.toolset = ImagineWaypointToolSet(
                self.config.server_url,
                wp_server_url=self.config.wp_server_url,
                wp_max_moves=self.config.wp_max_moves,
                predict_fn=self.config.wp_predict_fn,
                turn_budget=self.config.turn_budget,
                pano_view_px=self.config.pano_view_px,
                live_dir=live_dir,
                imagine=self.config.imagine_rollouts,
                mw_url=self.config.mw_url,
            )
        elif self.config.wp:
            self.toolset = WaypointToolSet(
                self.config.server_url,
                wp_server_url=self.config.wp_server_url,
                wp_max_moves=self.config.wp_max_moves,
                predict_fn=self.config.wp_predict_fn,
                turn_budget=self.config.turn_budget,
                pano_view_px=self.config.pano_view_px,
                live_dir=live_dir,
            )
        else:
            self.toolset = HabitatToolSet(
                self.config.server_url,
                bare=self.config.bare,
                step_budget=self.config.step_budget,
                turn_budget=self.config.turn_budget,
                pano_view_px=self.config.pano_view_px,
                live_dir=live_dir,
            )

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        """Run one parsed tool call; raise Submitted when the episode ends."""
        result = self.toolset.execute(action.get("tool", ""), action.get("args") or {})
        output = {"content": result.content, "info": result.info}
        if result.info.get("episode_over"):
            end_reason = result.info.get("end_reason") or "episode_over"
            raise Submitted(
                {
                    "role": "exit",
                    "content": json.dumps({"end_reason": end_reason, **result.info}),
                    "extra": {
                        "exit_status": end_reason,
                        "submission": "",
                        "final_info": result.info,
                    },
                }
            )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {**self.config.model_dump(), **kwargs}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
