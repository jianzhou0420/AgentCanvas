from __future__ import annotations

"""pySLAM container shim — the *inside-the-container* half of the bridge.

Runs a tiny FastAPI app inside the ``agentcanvas/pyslam`` container (the only
place ``import pyslam`` is legal — GPL stays behind the process boundary). It
holds one :class:`_backend.PySlamSession` and exposes it over HTTP so the
framework-side :class:`_client.PySlamContainerClient` can drive it without ever
importing pyslam.

Deployment (see ``_client.py`` for the matching ``docker run``):
    the nodeset directory is bind-mounted read-only at ``/opt/bridge`` and this
    module is launched with the container's pyslam venv python::

        /home/slam/.python/venvs/pyslam/bin/python -m uvicorn _server:app \\
            --host 0.0.0.0 --port 8000

    ``import _backend`` resolves as a sibling top-level module (``PYTHONPATH`` /
    cwd = ``/opt/bridge``); the package ``__init__`` (which imports the
    agentcanvas framework) is never touched here.

Wire protocol (all POST unless noted; localhost only):
    GET  /health           → {"ok", "built"}                 readiness probe
    POST /configure_camera  {intrinsics kwargs}  → {"ok"}    pins the pinhole cam
    POST /start            → {"ok", "built"}                 builds the Slam instance
    POST /reset            → {"ok"}                          clears the map (new episode)
    POST /track   body=npz(rgb[,depth][,timestamp]) → JSON pose/state/num_map_points
    POST /get_trajectory   → JSON {poses, states, num_frames}
    POST /get_map          → octet-stream npz(points) + X-Num-Points/X-Num-Keyframes
    POST /close            → {"ok"}                          quits Slam (joins bg threads)

Heavy geometry (frames in, map out) is marshalled as raw ``.npz`` bytes, not
base64/JSON, so the map never rides the graph wire — the client writes the
returned points to a host-side handle file (design §5a).
"""

import io
import logging
import os

import _backend  # sibling module, bind-mounted at /opt/bridge
import numpy as np
from fastapi import FastAPI, Request, Response

# ── wire protocol (Tier-2 additions) ─────────────────────────────────────────
#   POST /extract_features ?detector&descriptor&num_features  body=npz(rgb)
#          → octet-stream npz(keypoints, descriptors) + X-Num-Keypoints
#   POST /match_features   ?matcher_type&ratio_test&cross_check[&norm_type]
#          body=npz(des_a, des_b) → octet-stream npz(idxs_a, idxs_b) + X-Num-Matches
#   POST /eval_trajectory  JSON {poses_est, poses_gt, is_monocular} → JSON {ate, rpe, ...}
# These are stateless (no Slam session) — see _backend.py "stateless perception".

logging.basicConfig(level=logging.INFO, format="%(asctime)s pyslam-shim %(levelname)s %(message)s")
log = logging.getLogger("agentcanvas.pyslam.shim")

app = FastAPI(title="pyslam-bridge")

_session: _backend.PySlamSession | None = None


def _get_session() -> _backend.PySlamSession:
    """Lazily build the (empty) session from the container's env config.

    Mirrors the nodeset's former ``initialize()``: construct the session object
    (cheap, no ``Slam`` yet — that waits for ``/start`` once intrinsics are in).
    """
    global _session
    if _session is None:
        _session = _backend.PySlamSession(
            sensor_type=os.environ.get("PYSLAM_SENSOR", "rgbd"),
            feature_preset=os.environ.get("PYSLAM_FEATURE", "ORB2"),
            loop_preset=os.environ.get("PYSLAM_LOOP", "DBOW3"),
            headless=True,
        )
        log.info(
            "session created: sensor=%s feature=%s loop=%s",
            _session.sensor_type, _session.feature_preset, _session.loop_preset,
        )
    return _session


@app.get("/health")
def health() -> dict:
    s = _get_session()
    return {"ok": True, "built": s.is_built}


@app.post("/configure_camera")
async def configure_camera(req: Request) -> dict:
    cfg = await req.json()
    _get_session().configure_camera(**cfg)
    log.info("camera configured: %s", cfg)
    return {"ok": True}


@app.post("/start")
def start() -> dict:
    s = _get_session()
    s.start()
    log.info("Slam built (is_built=%s)", s.is_built)
    return {"ok": True, "built": s.is_built}


@app.post("/reset")
def reset() -> dict:
    _get_session().reset()
    return {"ok": True}


@app.post("/track")
async def track(req: Request) -> dict:
    body = await req.body()
    npz = np.load(io.BytesIO(body), allow_pickle=False)
    rgb = npz["rgb"]
    depth = npz["depth"] if "depth" in npz.files else None
    ts = float(npz["timestamp"]) if "timestamp" in npz.files else None
    return _get_session().track(rgb, depth, ts)


@app.post("/get_trajectory")
def get_trajectory() -> dict:
    return _get_session().get_trajectory()


@app.post("/get_map")
def get_map() -> Response:
    arrs = _get_session().get_map_arrays()
    buf = io.BytesIO()
    np.savez_compressed(buf, points=arrs["points"])
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Num-Points": str(arrs["num_points"]),
            "X-Num-Keyframes": str(arrs["num_keyframes"]),
        },
    )


@app.post("/extract_features")
async def extract_features(req: Request) -> Response:
    body = await req.body()
    npz = np.load(io.BytesIO(body), allow_pickle=False)
    q = req.query_params
    out = _backend.extract_features(
        npz["rgb"],
        detector=q.get("detector", "ORB"),
        descriptor=q.get("descriptor", "ORB"),
        num_features=int(q.get("num_features", 2000)),
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, keypoints=out["keypoints"], descriptors=out["descriptors"])
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Num-Keypoints": str(out["num_keypoints"]),
            "X-Detector": out["detector"],
            "X-Descriptor": out["descriptor"],
        },
    )


@app.post("/match_features")
async def match_features(req: Request) -> Response:
    body = await req.body()
    npz = np.load(io.BytesIO(body), allow_pickle=False)
    q = req.query_params
    norm = q.get("norm_type")
    out = _backend.match_features(
        npz["des_a"],
        npz["des_b"],
        matcher_type=q.get("matcher_type", "BF"),
        ratio_test=float(q.get("ratio_test", 0.7)),
        cross_check=q.get("cross_check", "0") in ("1", "true", "True"),
        norm_type=int(norm) if norm is not None else None,
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, idxs_a=out["idxs_a"], idxs_b=out["idxs_b"])
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"X-Num-Matches": str(out["num_matches"])},
    )


@app.post("/eval_trajectory")
async def eval_trajectory(req: Request) -> dict:
    payload = await req.json()
    return _backend.eval_trajectory(
        payload["poses_est"],
        payload["poses_gt"],
        is_monocular=bool(payload.get("is_monocular", False)),
    )


@app.post("/close")
def close() -> dict:
    global _session
    if _session is not None:
        _session.close()
        _session = None
        log.info("session closed")
    return {"ok": True}
