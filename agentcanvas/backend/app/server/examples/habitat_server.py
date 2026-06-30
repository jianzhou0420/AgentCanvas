"""Habitat environment — both the server app and the server manager.

This file contains two classes:

1. **HabitatApp** (ServerApp) — the actual FastAPI service that wraps
   HabitatEnvManager methods.  Run directly to start the service::

       python -m app.server.examples.habitat_server --port 9100

2. **HabitatServer** (BaseServer) — the framework-side manager that
   launches and monitors the HabitatApp process.  Used by WorkspaceComponentRegistry
   when the YAML config has ``command`` set.

Tutorial: see ``docs/design-docs/creating-a-server.html``
"""

from __future__ import annotations

import argparse
import logging
import os

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Server-side: the actual FastAPI service
# ══════════════════════════════════════════════════════════════════════


from ..manifest import PortSchema
from ..server_app import ServerApp, ServerFunction


class HabitatApp(ServerApp):
    """Habitat-Sim VLN-CE environment as a FastAPI service."""

    name = "habitat"
    description = "Habitat-Sim VLN-CE environment"
    version = "1.0"
    port = 9100

    def __init__(self) -> None:
        super().__init__()
        self._env_mgr = None

    async def on_startup(self) -> None:
        from workspace.nodesets.server.habitat import HabitatEnvManager

        self._env_mgr = HabitatEnvManager.get()
        log.info("HabitatEnvManager ready")

    def get_functions(self) -> list[ServerFunction]:
        return [
            ServerFunction(
                name="observe",
                description="Get current RGB + depth without stepping",
                input_ports=[],
                output_ports=[
                    PortSchema("rgb", "IMAGE", "Current RGB observation"),
                    PortSchema("depth", "DEPTH", "Current depth map"),
                ],
                handler=self._observe,
            ),
            ServerFunction(
                name="step",
                description="Execute a navigation action and return new observation",
                input_ports=[
                    PortSchema("action", "ACTION", "Action to execute (0-3)"),
                ],
                output_ports=[
                    PortSchema("rgb", "IMAGE", "Post-step RGB"),
                    PortSchema("depth", "DEPTH", "Post-step depth"),
                    PortSchema("pose", "POSE", "Post-step agent pose"),
                    PortSchema("action", "ACTION", "Echo of executed action"),
                    PortSchema("done", "BOOL", "Whether episode ended"),
                    PortSchema("metrics", "METRICS", "Final metrics (when done)"),
                ],
                handler=self._step,
            ),
            ServerFunction(
                name="get_state",
                description="Get agent position and orientation",
                input_ports=[],
                output_ports=[
                    PortSchema("pose", "POSE", "Agent position + heading"),
                ],
                handler=self._get_state,
            ),
            ServerFunction(
                name="episode_info",
                description="Get current episode instruction and ID",
                input_ports=[],
                output_ports=[
                    PortSchema("instruction", "TEXT", "Navigation instruction"),
                    PortSchema("episode_id", "TEXT", "Episode ID"),
                ],
                handler=self._episode_info,
            ),
            ServerFunction(
                name="panorama",
                description="Render 360-degree panoramic observation",
                input_ports=[],
                output_ports=[
                    PortSchema("composite", "IMAGE", "Composite panorama image"),
                    PortSchema("scene", "TEXT", "View direction descriptions"),
                ],
                config_schema={"n_views": {"type": "int", "default": 4}},
                handler=self._panorama,
            ),
        ]

    # ── Handlers ──

    async def _observe(self, inputs: dict, config: dict) -> dict:
        raw = await self.run_blocking(self._env_mgr.get_raw_obs)
        rgb, depth = _extract_rgb_depth(raw)
        return {"rgb": rgb, "depth": depth}

    async def _step(self, inputs: dict, config: dict) -> dict:
        action = int(inputs.get("action", 1))
        result = await self.run_blocking(self._env_mgr.step, action)
        raw = await self.run_blocking(self._env_mgr.get_raw_obs)
        rgb, depth = _extract_rgb_depth(raw)
        return {
            "rgb": rgb,
            "depth": depth,
            "state": {
                "position": result.get("position", [0, 0, 0]),
                "orientation": result.get("orientation", [0, 0, 0, 1]),
            },
            "action": action,
            "done": result.get("done", False),
            "metrics": result.get("metrics"),
        }

    async def _get_state(self, inputs: dict, config: dict) -> dict:
        state = await self.run_blocking(self._env_mgr.get_state)
        return {"state": state}

    async def _episode_info(self, inputs: dict, config: dict) -> dict:
        info = await self.run_blocking(self._env_mgr.get_episode_info)
        return {
            "instruction": info.get("instruction", ""),
            "episode_id": str(info.get("episode_id", "")),
        }

    async def _panorama(self, inputs: dict, config: dict) -> dict:
        import numpy as np

        n_views = config.get("n_views", 4)
        pano = await self.run_blocking(self._env_mgr.render_panorama, n_views)
        composite = None
        composite_b64 = pano.get("composite_base64", "")
        if composite_b64:
            import base64 as b64mod
            import io

            from PIL import Image

            buf = io.BytesIO(b64mod.b64decode(composite_b64))
            composite = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
        scene = "\n".join(
            "{}: heading {}\u00b0".format(v["direction"], v["heading_deg"])
            for v in pano.get("views", [])
        )
        return {"composite": composite, "scene": scene}


def _extract_rgb_depth(raw: dict | None):
    """Extract rgb and depth arrays from raw Habitat observations."""
    import numpy as np

    rgb = depth = None
    if raw:
        for k in ("rgb", "RGB"):
            if k in raw:
                rgb = np.asarray(raw[k], dtype=np.uint8)
                break
        for k in ("depth", "DEPTH"):
            if k in raw:
                depth = np.asarray(raw[k], dtype=np.float32).squeeze()
                break
    return rgb, depth


# ══════════════════════════════════════════════════════════════════════
# Framework-side: the process manager
# ══════════════════════════════════════════════════════════════════════


from ..base_server import BaseServer

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_VLNCE_PYTHON = os.path.expanduser("~/miniforge3/envs/ac-vlnce/bin/python")


class HabitatServer(BaseServer):
    """Manages the Habitat server process.

    Used by WorkspaceComponentRegistry when ``workspace/servers/habitat.yaml``
    has ``managed: true``.
    """

    name = "habitat"
    description = "Habitat-Sim VLN-CE environment"
    port = 9100
    startup_timeout = 60  # Habitat init is slow (loading scenes)
    auto_restart = False

    command = "%s -m app.server.examples.habitat_server --port 9100" % _VLNCE_PYTHON
    working_dir = os.path.join(_REPO_ROOT, "backend")


# ══════════════════════════════════════════════════════════════════════
# CLI entry point: run the server app directly
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Habitat environment server")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = HabitatApp()
    app.port = args.port
    app.serve(host=args.host)


if __name__ == "__main__":
    main()
