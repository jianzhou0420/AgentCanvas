from __future__ import annotations

"""VggtSlamSession — drives upstream ``vggt_slam`` inside the ac-vggt-slam env.

This module is imported ONLY in the AutoServerApp subprocess (server mode,
``server_python = ac-vggt-slam``) — never in the framework env. It holds the
long-lived SLAM session: upstream's ``Solver`` plus the injected VGGT model,
and a faithful mirror of the ``main.py`` driver loop (which upstream keeps
OUTSIDE the ``vggt_slam`` package — ``setup.py`` ships only the library, so the
keyframe gate / submap trigger / optimize cadence must live here).

Upstream: MIT-SPARK/VGGT-SLAM @ 35327ac (reference clone:
``third_party/zz_just_for_refer/vggt_slam/``). Citations ``main.py:NN`` below
point into that pin.

Deviations from upstream (enumerated for /grill-implement):
  D1. ``vggt_slam.solver.Viewer`` is swapped for ``_NoopViewer`` at import
      time. Upstream ``Solver.__init__`` unconditionally opens a viser server
      on port 8080 (``viewer.py:13``) — a headless server session cannot own a
      fixed listening port per Solver. No mainline math touches the viewer;
      the only Solver methods we call (``run_predictions`` / ``add_points``)
      never reference it. Audit surface: ``grep -n "self.viewer" solver.py``.
  D2. SAM 3 loads lazily on the first ``query_object`` call (upstream builds
      it eagerly under ``--run_os``, ``main.py:63-64``). VRAM prudence only —
      no numeric effect; the query math is identical.
  D3. Keyframes are re-encoded to lossless PNGs named by TUM timestamp
      (mirroring upstream's own streaming path, ``main_realtime.py
      save_keyframe``) because ``run_predictions`` consumes file paths.
      Pixel-identical to feeding the original dataset files; the numeric
      filename is load-bearing (``Submap.set_frame_ids`` parses it into the
      TUM pose-file timestamp column that evo associates on).
  D4. SAM3 inference wrapped in ``torch.autocast("cuda", bf16)`` — sam3 @
      5dd401d on torch 2.3.1 clashes (BFloat16/Float matmul) when called
      bare as upstream ``main.py:177`` does; autocast is SAM3's documented
      usage. Verified standalone 2026-07-14; no semantic effect.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import torch

# ── D1: headless viewer patch — MUST precede any Solver construction ────────
import vggt_slam.solver as _solver_mod


class _NoopViewer:
    """Headless stand-in for vggt_slam.viewer.Viewer (signature per viewer.py:10)."""

    def __init__(self, port: int = 8080) -> None:
        self.server = None


_solver_mod.Viewer = _NoopViewer

from vggt.models.vggt import VGGT  # noqa: E402
import vggt_slam.slam_utils as utils  # noqa: E402
from vggt_slam.solver import Solver  # noqa: E402

log = logging.getLogger("agentcanvas.vggt_slam.backend")

# main.py:75 — VGGT-1B weights, auto-cached by torch.hub on first load.
_VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

_DEFAULT_CONFIG: dict[str, Any] = {
    # Defaults mirror upstream argparse (main.py:29-34).
    "submap_size": 16,
    "max_loops": 1,
    "min_disparity": 50.0,
    "conf_threshold": 25.0,
    "lc_thres": 0.95,
    "run_os": False,
}


def _find_repo_root() -> str | None:
    """Walk upward until a dir containing data/ (same helper family as model_sam)."""
    d = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "data")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _pose_dict_to_tum_fields(pose: Any) -> list[float] | None:
    """{position:[x,y,z], orientation:[qx,qy,qz,qw]} → [x y z qx qy qz qw]."""
    if pose is None:
        return None
    if isinstance(pose, dict) and "position" in pose and "orientation" in pose:
        p = [float(v) for v in pose["position"]]
        q = [float(v) for v in pose["orientation"]]
        if len(p) == 3 and len(q) == 4:
            return p + q
    return None


class VggtSlamSession:
    """One SLAM session: Solver + injected VGGT model + main.py driver mirror.

    All methods are synchronous and MUST be called through :attr:`executor`
    (a 1-worker pool — thread affinity for the CUDA context and gtsam Values,
    same discipline as model_pyslam's backend). The nodeset wraps every call
    in ``run_in_executor(session.executor, ...)``; the pool being 1-wide is
    what serializes concurrent node fires.
    """

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vggt_slam")
        self._lock = threading.Lock()  # belt over the 1-worker braces

        # Loaded once per process, injected per call (Solver never owns the
        # VGGT model — solver.py stores no reference to it).
        self._model: Any = None
        # Open-set stack (run_os): PE-CLIP eager at reset, SAM3 lazy (D2).
        self._clip_model: Any = None
        self._clip_tokenizer: Any = None
        self._clip_preprocess: Any = None
        self._sam3_processor: Any = None

        self.cfg: dict[str, Any] = dict(_DEFAULT_CONFIG)
        self.solver: Solver | None = None

        self._workdir: str | None = None
        self._frames_dir: str | None = None
        self._out_dir: str | None = None

        self._buffer: list[str] = []          # keyframe PNG paths (main.py image_names_subset)
        self._gt_rows: list[str] = []         # TUM-format gt rows, keyframe-aligned
        self._frame_count = 0
        self._keyframe_count = 0
        self._submaps_processed = 0
        self._last_call_processed = False     # did the LAST track() fire a submap?
        self._finalized = False
        self._last_pose: list[list[float]] | None = None
        self._streamed_submaps = 0            # get_map(on_new_submap) high-water mark

    # ── model loading ────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"  # main.py:44
        log.info("loading VGGT-1B (SPARK fork) onto %s ...", device)
        model = VGGT()                                            # main.py:74
        model.load_state_dict(torch.hub.load_state_dict_from_url(_VGGT_URL))  # main.py:76
        model.eval()                                              # main.py:78
        model = model.to(torch.bfloat16)                          # main.py:79
        model = model.to(device)                                  # main.py:80
        self._model = model
        log.info("VGGT-1B ready")

    def _ensure_clip(self) -> None:
        if self._clip_model is not None:
            return
        # main.py:60-69 — PE-CLIP must be live DURING tracking (embeddings are
        # computed per submap inside run_predictions, solver.py:326-329).
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        log.info("loading Perception Encoder PE-Core-L14-336 ...")
        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # main.py:66
        clip_model = clip_model.cuda()                                        # main.py:67
        self._clip_tokenizer = pe_transforms.get_text_tokenizer(clip_model.context_length)  # main.py:68
        self._clip_preprocess = pe_transforms.get_image_transform(clip_model.image_size)    # main.py:69
        self._clip_model = clip_model
        log.info("PE-CLIP ready")

    def _ensure_sam3(self, confidence_threshold: float) -> None:
        if self._sam3_processor is not None:
            return
        # main.py:58-64 (built lazily here — D2).
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        log.info("loading SAM 3 ...")
        sam3_model = build_sam3_image_model()                     # main.py:63
        self._sam3_processor = Sam3Processor(sam3_model, confidence_threshold=confidence_threshold)  # main.py:64
        log.info("SAM 3 ready")

    # ── session lifecycle ────────────────────────────────────────────────

    def reset(self, config: dict[str, Any]) -> dict[str, Any]:
        """(Re)start the session for a new episode — fresh Solver, fresh map."""
        with self._lock:
            cfg = dict(_DEFAULT_CONFIG)
            cfg.update({k: v for k, v in (config or {}).items() if v is not None})
            self.cfg = cfg

            self._ensure_model()
            if cfg["run_os"]:
                self._ensure_clip()

            # Fresh workdir per episode; drop the previous episode's frames.
            if self._workdir:
                shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = tempfile.mkdtemp(prefix="vggt_slam_")
            self._frames_dir = os.path.join(self._workdir, "frames")
            self._out_dir = os.path.join(self._workdir, "out")
            os.makedirs(self._frames_dir, exist_ok=True)
            os.makedirs(self._out_dir, exist_ok=True)

            # main.py:47-52 — a new Solver is a new map + graph + retrieval DB
            # (Salad reloads inside ImageRetrieval; viewer is _NoopViewer, D1).
            self.solver = Solver(
                init_conf_threshold=float(cfg["conf_threshold"]),
                lc_thres=float(cfg["lc_thres"]),
                vis_voxel_size=None,
                vis_imgs=False,
            )

            self._buffer = []
            self._gt_rows = []
            self._frame_count = 0
            self._keyframe_count = 0
            self._submaps_processed = 0
            self._last_call_processed = False
            self._finalized = False
            self._last_pose = None
            self._streamed_submaps = 0

            info = {
                "submap_size": cfg["submap_size"],
                "max_loops": cfg["max_loops"],
                "min_disparity": cfg["min_disparity"],
                "conf_threshold": cfg["conf_threshold"],
                "lc_thres": cfg["lc_thres"],
                "run_os": bool(cfg["run_os"]),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            log.info("session reset: %s", info)
            return info

    # ── driver mirror (main.py:99-132) ───────────────────────────────────

    def _process_submap(self) -> int:
        """run_predictions → add_points → optimize; keep the overlap tail.

        Returns the number of loop closures detected in this submap.
        """
        assert self.solver is not None
        predictions = self.solver.run_predictions(                  # main.py:115
            self._buffer, self._model, int(self.cfg["max_loops"]),
            self._clip_model, self._clip_preprocess,
        )
        self.solver.add_points(predictions)                         # main.py:119
        self.solver.graph.optimize()                                # main.py:122
        self._submaps_processed += 1
        # main.py:132 — overlapping_window_size is hard-pinned to 1 upstream.
        self._buffer = self._buffer[-1:]

        # Latest optimized SE(3) pose of the newest non-LC submap's last frame.
        sm = self.solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
        poses_world = sm.get_all_poses_world(self.solver.graph, give_camera_mat=False)
        self._last_pose = poses_world[-1].tolist()
        return len(predictions["detected_loops"])

    def track(self, rgb: np.ndarray, timestamp: float | None, gt_pose: Any) -> dict[str, Any]:
        """Feed one RGB frame — keyframe gate, buffer, maybe run a submap."""
        with self._lock:
            if self.solver is None:
                raise RuntimeError("session not reset — fire model_vggt_slam2__reset first")
            self._frame_count += 1

            # Upstream reads frames with cv2.imread → BGR (main.py:102); our
            # wire carries RGB. Flip to BGR so the LK gate sees the same pixels.
            rgb = np.asarray(rgb, dtype=np.uint8)
            bgr = np.ascontiguousarray(rgb[..., ::-1])

            is_keyframe = self.solver.flow_tracker.compute_disparity(  # main.py:103
                bgr, float(self.cfg["min_disparity"]), False,
            )
            submap_loops = 0
            self._last_call_processed = False
            if is_keyframe:
                # main_realtime.py:44-47 save_keyframe — run_predictions consumes
                # paths, so persist the keyframe (D3). Numeric stem is load-bearing.
                stem = f"{timestamp:.6f}" if timestamp is not None else f"{self._frame_count:010d}"
                if not re.search(r"\d", stem):
                    raise ValueError(f"keyframe stem must be numeric, got {stem!r}")
                path = os.path.join(self._frames_dir or ".", f"{stem}.png")
                cv2.imwrite(path, bgr)
                self._buffer.append(path)                            # main.py:105
                self._keyframe_count += 1

                gt_fields = _pose_dict_to_tum_fields(gt_pose)
                if gt_fields is not None:
                    self._gt_rows.append(" ".join(f"{v:.8f}" for v in [float(stem)] + gt_fields))

                # main.py:111 — submap_size new frames + 1 overlap frame.
                if len(self._buffer) == int(self.cfg["submap_size"]) + 1:
                    submap_loops = self._process_submap()
                    self._last_call_processed = True

            return {
                "pose": self._last_pose,
                "is_keyframe": bool(is_keyframe),
                "submap_processed": bool(self._last_call_processed),
                "num_keyframes": self._keyframe_count,
                "num_submaps": len(self.solver.map.non_lc_submap_ids),
                "num_loops": self.solver.graph.get_num_loops(),
            }

    def finalize(self) -> dict[str, Any]:
        """Process the trailing buffer — mirrors main.py:111's last-image branch.

        Upstream triggers processing on the final iteration regardless of
        buffer fill, INCLUDING an overlap-only single-frame buffer — unless
        that same final frame already fired the size trigger. The
        ``_last_call_processed`` flag reproduces both cases exactly.
        """
        with self._lock:
            if self.solver is None:
                raise RuntimeError("session not reset")
            if not self._finalized and self._buffer and not self._last_call_processed:
                self._process_submap()
            self._finalized = True
            return {
                "submaps_processed": self._submaps_processed,
                "num_loops": self.solver.graph.get_num_loops(),
                "num_keyframes": self._keyframe_count,
                "num_frames_seen": self._frame_count,
            }

    # ── read-out surface ─────────────────────────────────────────────────

    def get_trajectory(self) -> dict[str, Any]:
        with self._lock:
            if self.solver is None or self.solver.map.get_num_submaps() == 0:
                return {"poses": [], "traj_tum": "", "gt_tum": "", "num_poses": 0}
            path = os.path.join(self._out_dir or ".", "poses.txt")
            # main.py:206 — the exact upstream writer (TUM rows via decompose_camera).
            self.solver.map.write_poses_to_file(path, self.solver.graph, kitti_format=False)
            with open(path, "r", encoding="utf-8") as f:
                traj_tum = f.read()

            poses: list[list[list[float]]] = []
            try:
                from scipy.spatial.transform import Rotation as R

                for line in traj_tum.splitlines():
                    vals = [float(v) for v in line.split()]
                    if len(vals) != 8:
                        continue
                    m = np.eye(4)
                    m[:3, :3] = R.from_quat(vals[4:8]).as_matrix()
                    m[:3, 3] = vals[1:4]
                    poses.append(m.tolist())
            except Exception:
                log.exception("pose-matrix parse failed (traj_tum still valid)")

            return {
                "poses": poses,
                "traj_tum": traj_tum,
                "gt_tum": "\n".join(self._gt_rows),
                "num_poses": len(poses),
            }

    def get_map(self, on_new_submap: bool = False) -> dict[str, Any]:
        with self._lock:
            if self.solver is None or self.solver.map.get_num_submaps() == 0:
                return {"map_handle": "", "pcd_path": "", "num_points": 0, "num_submaps": 0}
            n_submaps = len(self.solver.map.non_lc_submap_ids)
            # Live-streaming gate (demo graphs): the map only changes when a
            # submap lands, so an in-loop caller exports at submap boundaries
            # and returns a cheap empty handle otherwise (pointCloudViewer
            # no-ops on an empty cloud, keeping the last render).
            if on_new_submap and n_submaps <= self._streamed_submaps:
                return {"map_handle": "", "pcd_path": "", "num_points": 0, "num_submaps": n_submaps}
            import open3d as o3d

            pcd_path = os.path.join(self._out_dir or ".", "map_points.pcd")
            # main.py:210 — the exact upstream writer (merged colored .pcd).
            self.solver.map.write_points_to_file(self.solver.graph, pcd_path)
            # Canvas handle: pointCloudViewer + downstream consumers np.load an
            # .npz with points/colors keys (pyslam convention) — re-export the
            # SAME cloud the upstream writer just produced (zero drift risk).
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
            if on_new_submap:
                self._streamed_submaps = n_submaps
                # Unique name per export — a repeated path would let the viewer
                # cache a stale cloud (and a reader could catch a half-write).
                npz_path = os.path.join(self._out_dir or ".", f"map_stream_{n_submaps:04d}.npz")
            else:
                npz_path = os.path.join(self._out_dir or ".", "map_points.npz")
            np.savez_compressed(npz_path, points=points, colors=colors)
            return {
                "map_handle": npz_path,
                "pcd_path": pcd_path,
                "num_points": int(points.shape[0]),
                "num_submaps": n_submaps,
            }

    # ── eval (upstream ruler: evals/eval_tum.sh → evo_ape tum gt est -as) ─

    def eval_trajectory(
        self,
        traj_tum: str,
        gt_tum: str | None,
        sequence: str | None,
        align_scale: bool,
    ) -> dict[str, Any]:
        with self._lock:
            if not traj_tum or not traj_tum.strip():
                return {"ate_rmse": None, "metrics": json.dumps({"error": "empty trajectory"})}

            workdir = self._out_dir or tempfile.mkdtemp(prefix="vggt_slam_eval_")
            est_path = os.path.join(workdir, "eval_est.txt")
            with open(est_path, "w", encoding="utf-8") as f:
                f.write(traj_tum if traj_tum.endswith("\n") else traj_tum + "\n")

            gt_path: str | None = None
            if sequence:
                root = _find_repo_root()
                if root:
                    cand = os.path.join(root, "data", "tum", str(sequence), "groundtruth.txt")
                    if os.path.isfile(cand):
                        gt_path = cand
            if gt_path is None and gt_tum and gt_tum.strip():
                gt_path = os.path.join(workdir, "eval_gt.txt")
                with open(gt_path, "w", encoding="utf-8") as f:
                    f.write(gt_tum if gt_tum.endswith("\n") else gt_tum + "\n")
            if gt_path is None:
                return {"ate_rmse": None,
                        "metrics": json.dumps({"error": "no ground truth (wire gt_tum or sequence)"})}

            evo_ape = os.path.join(os.path.dirname(sys.executable), "evo_ape")
            # evals/eval_tum.sh — `evo_ape tum <gt> <est> -as` (-a align, -s Sim3 scale;
            # scale correction is mandatory: SL(4) poses are up-to-scale).
            cmd = [evo_ape, "tum", gt_path, est_path, "-as" if align_scale else "-a"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            out = proc.stdout + ("\n" + proc.stderr if proc.returncode != 0 else "")

            metrics: dict[str, Any] = {"cmd": " ".join(cmd), "returncode": proc.returncode}
            for key in ("rmse", "mean", "median", "std", "min", "max", "sse"):
                m = re.search(rf"^\s*{key}\s+([0-9.eE+-]+)\s*$", proc.stdout, re.MULTILINE)
                if m:
                    metrics[key] = float(m.group(1))
            log.info("evo_ape: rc=%s rmse=%s", proc.returncode, metrics.get("rmse"))
            return {
                "ate_rmse": metrics.get("rmse"),
                "metrics": json.dumps({**metrics, "raw": out.strip()}, ensure_ascii=False),
            }

    # ── open-set query (main.py:158-199, post-hoc) ───────────────────────

    def query_object(self, text: str, sam3_conf: float) -> dict[str, Any]:
        with self._lock:
            if self.solver is None or self.solver.map.get_num_submaps() == 0:
                return {"error": "no map — run tracking first"}
            if not self.cfg.get("run_os") or self._clip_model is None:
                return {"error": "run_os was not enabled at reset — no semantic vectors stored"}
            from torchvision.transforms.functional import to_pil_image  # main.py:8

            self._ensure_sam3(sam3_conf)

            text_emb = utils.compute_text_embeddings(                # main.py:169
                self._clip_model, self._clip_tokenizer, text)
            best_score, best_submap_id, best_frame_index = (
                self.solver.map.retrieve_best_semantic_frame(text_emb))  # main.py:170
            found_submap = self.solver.map.get_submap(best_submap_id)    # main.py:172

            best_img = found_submap.get_frame_at_index(best_frame_index)  # main.py:175
            # D4: SAM3 runs under explicit bf16 autocast — its float32 weights
            # expect autocast-managed matmuls (SAM3's documented usage); on
            # torch 2.3.1 a bare call dies with a BFloat16/Float matmul clash
            # (verified standalone 2026-07-14). No semantic effect: masks are
            # thresholded downstream, scores cast to float.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # main.py:177-182
                pil_img = to_pil_image(best_img)
                inference_state = self._sam3_processor.set_image(pil_img)
                output = self._sam3_processor.set_text_prompt(state=inference_state, prompt=text)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

            out_dir = self._out_dir or tempfile.mkdtemp(prefix="vggt_slam_os_")
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", text)[:40] or "query"

            frame_path = os.path.join(out_dir, f"query_{slug}_frame.png")
            pil_img.save(frame_path)
            overlay_path = os.path.join(out_dir, f"query_{slug}_overlay.png")
            utils.overlay_masks(pil_img, masks).save(overlay_path)   # main.py:187

            obbs: list[dict[str, Any]] = []
            points_by_instance: dict[str, np.ndarray] = {}
            for i in range(masks.shape[0]):                          # main.py:190-192
                mask = masks[i].cpu().numpy()
                pts = found_submap.get_points_in_mask(best_frame_index, mask, self.solver.graph)
                try:
                    center, extent, rotation = utils.compute_obb_from_points(pts)
                except ValueError:
                    continue  # empty/invalid point set for this mask
                obbs.append({
                    "center": center.tolist(),
                    "extent": extent.tolist(),
                    "rotation": rotation.tolist(),
                    "sam3_score": float(scores[i]),
                })
                points_by_instance[f"instance_{i}"] = pts.astype(np.float32)

            points_path = os.path.join(out_dir, f"query_{slug}_points.npz")
            np.savez_compressed(points_path, **points_by_instance)

            return {
                "best_submap_id": int(best_submap_id),
                "best_frame_id": int(best_frame_index),
                "score": float(best_score),
                "num_instances": len(obbs),
                "obb": obbs,
                "points_path": points_path,
                "overlay_path": overlay_path,
                "best_frame_image_path": frame_path,
            }

    # ── teardown ─────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self.solver = None
            self._model = None
            self._clip_model = None
            self._clip_tokenizer = None
            self._clip_preprocess = None
            self._sam3_processor = None
            if self._workdir:
                shutil.rmtree(self._workdir, ignore_errors=True)
                self._workdir = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.executor.shutdown(wait=False, cancel_futures=True)
