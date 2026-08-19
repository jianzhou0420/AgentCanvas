from __future__ import annotations

"""HM-EQA replay renderer — habitat-sim subprocess for smooth-mode playback.

Standalone renderer (not a BaseNodeSet) launched by
``app.replay.renderer_host``. Holds one habitat_sim.Simulator instance
keyed by ``scene_id``; re-init on scene change.

POST /render
    body: {"scene": str, "position": [x, y, z], "angle": float}
    response: 200 image/jpeg (RGB, default 480×640, JPEG quality 85)

GET /health
    response: 200 {"status": "ok"}

Runs under the ``hmeqa`` conda env (habitat-sim 0.3.x) — same env
EnvHMEQANodeSet uses for live eval. Isolated subprocess, so concurrent
eval and replay don't collide.
"""


import io
import logging
import os
import threading
from typing import Any

import numpy as np

log = logging.getLogger("agentcanvas.hmeqa_renderer")


# ══════════════════════════════════════════════════════════════════════
# Paths & defaults — kept in sync with hmeqa.py
# ══════════════════════════════════════════════════════════════════════

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
_SCENE_ROOT = os.environ.get(
    "HMEQA_SCENE_ROOT", os.path.join(_REPO_ROOT, "data", "hm3d", "hm3dsem")
)

_DEFAULTS = {
    "img_height": 480,
    "img_width": 640,
    "hfov": 120,
    "camera_height": 1.5,
    "camera_tilt_deg": -30.0,
    "seed": 42,
    "jpeg_quality": 85,
}


# ══════════════════════════════════════════════════════════════════════
# Habitat-sim helpers — vendored from hmeqa.py to avoid cross-file import
# (workspace/nodesets/env/ has no __init__.py — bucket-mode loading)
# ══════════════════════════════════════════════════════════════════════


def _make_sim_cfg(
    scene_path: str, img_height: int, img_width: int, hfov: float, camera_height: float
) -> Any:
    import habitat_sim  # lazy — only works in hmeqa env

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [img_height, img_width]
    rgb_spec.position = [0.0, camera_height, 0.0]
    rgb_spec.hfov = hfov

    agent_cfg.sensor_specifications = [rgb_spec]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def _scene_paths(scene: str) -> tuple[str, str]:
    """Return (mesh, navmesh) paths for an HM3D scene id."""
    scene_short = scene[6:] if len(scene) > 6 else scene
    mesh = os.path.join(_SCENE_ROOT, scene, scene_short + ".basis.glb")
    navmesh = os.path.join(_SCENE_ROOT, scene, scene_short + ".basis.navmesh")
    return mesh, navmesh


# ══════════════════════════════════════════════════════════════════════
# HMEQARendererServer
# ══════════════════════════════════════════════════════════════════════


class HMEQARendererServer:
    """Single-simulator renderer for HM-EQA replay smooth mode.

    Construct, then call :meth:`build_app` to get a FastAPI instance
    suitable for ``uvicorn.run``. ``app.replay.renderer_host`` does
    exactly that.
    """

    def __init__(self, **overrides: Any) -> None:
        self._config = dict(_DEFAULTS)
        self._config.update(overrides)
        self._lock = threading.Lock()
        self._simulator: Any | None = None
        self._agent: Any | None = None
        self._loaded_scene: str = ""

    # ── HTTP ──────────────────────────────────────────────────────────

    def build_app(self):
        # NOTE: ``from __future__ import annotations`` makes annotations
        # strings, and FastAPI/Pydantic resolve them via ``get_type_hints``.
        # Closure-defined Pydantic models aren't visible in the function's
        # globals, so RenderRequest below is defined at module scope; the
        # endpoint signature is also explicitly typed via ``Body(...)`` to
        # short-circuit any remaining query-vs-body ambiguity.
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import Response

        app = FastAPI(title="HM-EQA replay renderer")
        renderer = self

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "scene": renderer._loaded_scene,
            }

        @app.post("/render")
        async def render(req=Body(...)):
            scene = req.get("scene") if isinstance(req, dict) else None
            position = req.get("position") if isinstance(req, dict) else None
            angle = req.get("angle") if isinstance(req, dict) else None
            if not isinstance(scene, str) or not scene:
                raise HTTPException(400, "missing/invalid 'scene'")
            if not isinstance(position, list) or len(position) != 3:
                raise HTTPException(400, "'position' must be [x, y, z]")
            if not isinstance(angle, (int, float)):
                raise HTTPException(400, "'angle' must be number")
            try:
                jpeg = renderer._render(
                    scene,
                    [float(v) for v in position],
                    float(angle),
                )
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            except Exception as exc:
                log.exception("Render failed")
                raise HTTPException(500, f"render failed: {exc}") from exc
            return Response(content=jpeg, media_type="image/jpeg")

        return app

    # ── Rendering ─────────────────────────────────────────────────────

    def _render(self, scene: str, position: list, angle: float) -> bytes:
        with self._lock:
            self._ensure_scene_unlocked(scene)
            self._set_pose_unlocked(np.asarray(position, dtype=np.float64), angle)
            obs = self._simulator.get_sensor_observations()
        rgb = np.asarray(obs["color_sensor"], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        return _encode_jpeg(rgb, quality=int(self._config["jpeg_quality"]))

    def _ensure_scene_unlocked(self, scene: str) -> None:
        if self._loaded_scene == scene and self._simulator is not None:
            return
        self._close_unlocked()

        mesh, navmesh = _scene_paths(scene)
        if not os.path.isfile(mesh):
            raise FileNotFoundError(f"scene mesh missing: {mesh}")

        import habitat_sim

        sim_cfg = _make_sim_cfg(
            scene_path=mesh,
            img_height=self._config["img_height"],
            img_width=self._config["img_width"],
            hfov=self._config["hfov"],
            camera_height=self._config["camera_height"],
        )
        self._simulator = habitat_sim.Simulator(sim_cfg)
        if os.path.isfile(navmesh):
            try:
                self._simulator.pathfinder.load_nav_mesh(navmesh)
            except Exception:
                log.exception("Failed to load navmesh %s — continuing without", navmesh)
        self._simulator.pathfinder.seed(int(self._config["seed"]))
        self._agent = self._simulator.initialize_agent(0)
        self._loaded_scene = scene
        log.info("HM-EQA renderer: loaded scene %s", scene)

    def _set_pose_unlocked(self, pts: np.ndarray, angle: float) -> None:
        import habitat_sim
        from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs

        camera_tilt = self._config["camera_tilt_deg"] * np.pi / 180.0
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.asarray(pts, dtype=np.float64)
        agent_state.rotation = rotation
        self._agent.set_state(agent_state)

    def _close_unlocked(self) -> None:
        if self._simulator is not None:
            try:
                self._simulator.close()
            except Exception:
                log.exception("Failed to close simulator")
        self._simulator = None
        self._agent = None
        self._loaded_scene = ""


def _encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    from PIL import Image

    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
