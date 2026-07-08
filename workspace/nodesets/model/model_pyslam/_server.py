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
    POST /get_dense_map    → octet-stream npz(points,colors,vertices,triangles) + X-Num-*
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
#   POST /predict_depth    ?estimator&min_depth&max_depth&environment
#          body=npz(rgb[,image_right]) → octet-stream npz(depth) + X-Estimator/X-Depth-*
#   POST /reconstruct_multiview ?backend&as_pointcloud  body=npz(img_0,img_1,…)
#          → octet-stream npz(points,colors,vertices,faces,camera_poses,intrinsics)
#            + X-Backend/X-Num-Points/X-Num-Vertices/X-Num-Faces/X-Num-Views
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
            volumetric=os.environ.get("PYSLAM_VOLUMETRIC", "0").lower() in ("1", "true", "yes"),
            volumetric_type=os.environ.get("PYSLAM_VOLUMETRIC_TYPE", "VOXEL_GRID"),
            environment=os.environ.get("PYSLAM_ENV", "INDOOR"),
        )
        log.info(
            "session created: sensor=%s feature=%s loop=%s volumetric=%s(%s) env=%s",
            _session.sensor_type, _session.feature_preset, _session.loop_preset,
            _session.volumetric, _session.volumetric_type, _session.environment,
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


@app.post("/predict_depth")
async def predict_depth(req: Request) -> Response:
    body = await req.body()
    npz = np.load(io.BytesIO(body), allow_pickle=False)
    q = req.query_params
    out = _backend.predict_depth(
        npz["rgb"],
        npz["image_right"] if "image_right" in npz.files else None,
        estimator=q.get("estimator", "DEPTH_ANYTHING_V2"),
        min_depth=float(q.get("min_depth", 0.0)),
        max_depth=float(q.get("max_depth", 50.0)),
        environment=q.get("environment", "INDOOR"),
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, depth=out["depth"])
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Estimator": out["estimator"],
            "X-Depth-Shape": ",".join(str(d) for d in out["shape"]),
            "X-Depth-Min": f"{out['min']:.6g}",
            "X-Depth-Max": f"{out['max']:.6g}",
        },
    )


@app.post("/segment_semantic")
async def segment_semantic(req: Request) -> Response:
    body = await req.body()
    npz = np.load(io.BytesIO(body), allow_pickle=False)
    q = req.query_params
    isz = (q.get("image_size", "512,512")).split(",")
    out = _backend.segment_semantic(
        npz["rgb"],
        model=q.get("model", "DEEPLABV3"),
        feature_type=q.get("feature_type", "LABEL"),
        dataset=q.get("dataset", "CITYSCAPES"),
        image_size=(int(isz[0]), int(isz[1])),
    )
    buf = io.BytesIO()
    arrays: dict = {}
    if out["semantics"] is not None:
        arrays["semantics"] = out["semantics"]
    if out["instances"] is not None:
        arrays["instances"] = out["instances"]
    np.savez_compressed(buf, **arrays)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Model": out["model"],
            "X-Feature-Type": out["feature_type"],
            "X-Num-Classes": str(out["num_classes"]),
            "X-Has-Instances": "1" if out["instances"] is not None else "0",
        },
    )


@app.post("/reconstruct_multiview")
async def reconstruct_multiview(req: Request) -> Response:
    body = await req.body()
    npz = np.load(io.BytesIO(body), allow_pickle=False)
    q = req.query_params
    # Images arrive as img_0, img_1, … (variable view count) in index order.
    keys = sorted((k for k in npz.files if k.startswith("img_")),
                  key=lambda k: int(k.split("_")[1]))
    images = [npz[k] for k in keys]
    out = _backend.reconstruct_multiview(
        images,
        backend=q.get("backend", "MAST3R"),
        as_pointcloud=q.get("as_pointcloud", "1") in ("1", "true", "True"),
    )
    # Stack the per-view poses/intrinsics so the heavy geometry rides one npz.
    poses = np.stack(out["camera_poses"]) if out["camera_poses"] else np.empty((0, 4, 4), np.float32)
    intr = np.stack(out["intrinsics"]) if out["intrinsics"] else np.empty((0, 3, 3), np.float32)
    buf = io.BytesIO()
    np.savez_compressed(
        buf, points=out["points"], colors=out["colors"],
        vertices=out["vertices"], faces=out["faces"],
        camera_poses=poses, intrinsics=intr,
    )
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Backend": out["backend"],
            "X-Num-Points": str(out["num_points"]),
            "X-Num-Vertices": str(out["num_vertices"]),
            "X-Num-Faces": str(out["num_faces"]),
            "X-Num-Views": str(out["num_views"]),
        },
    )


@app.post("/eval_trajectory")
async def eval_trajectory(req: Request) -> dict:
    payload = await req.json()
    return _backend.eval_trajectory(
        payload["poses_est"],
        payload["poses_gt"],
        is_monocular=bool(payload.get("is_monocular", False)),
    )


@app.post("/get_dense_map")
def get_dense_map() -> Response:
    # Ship the arrays back so the framework writes a host-owned handle (rootless
    # uid remap), exactly like /get_map.
    d = _get_session().get_dense_map_arrays()
    buf = io.BytesIO()
    np.savez_compressed(buf, points=d["points"], colors=d["colors"],
                        vertices=d["vertices"], triangles=d["triangles"])
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Num-Points": str(d["num_points"]),
            "X-Num-Vertices": str(d["num_vertices"]),
            "X-Num-Triangles": str(d["num_triangles"]),
            "X-Dense-Type": str(d["type"]),
        },
    )


@app.post("/close")
def close() -> dict:
    global _session
    if _session is not None:
        _session.close()
        _session = None
        log.info("session closed")
    return {"ok": True}
