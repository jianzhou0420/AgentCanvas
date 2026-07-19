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

from toolset import HabitatToolSet, WaypointToolSet


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


class HabitatEnvironment:
    def __init__(self, *, config_class: type = HabitatEnvironmentConfig, **kwargs: Any) -> None:
        self.config = config_class(**kwargs)
        live_dir = Path(self.config.live_dir) if self.config.live_dir else None
        if self.config.wp:
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
