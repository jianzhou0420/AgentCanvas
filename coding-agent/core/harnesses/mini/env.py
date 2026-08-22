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
    # eharness-evo T1 candidate (evo9b_blocked01): surface per-call realized
    # motion — moved_m + forward_blocked — in the step result. Default OFF
    # keeps every std cell byte-identical.
    blocked_signal: bool = False
    turn_macros: bool = False   # evo: 4/5/6 = L90/R90/turn-around + turned_deg
    memo: bool = False          # evo: remember() tool, notes echoed in results
    revisit: bool = False       # evo: ahash revisit hint on observe()
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
                blocked_signal=self.config.blocked_signal,
                turn_macros=self.config.turn_macros,
                memo=self.config.memo,
                revisit=self.config.revisit,
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
