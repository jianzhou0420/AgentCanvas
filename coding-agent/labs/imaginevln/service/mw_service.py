#!/usr/bin/env python
"""Resident MemoryWorld world-model service (port 9270).

One process, model loaded once, T5 dropped (see mw_notext.py). Requests carry
an initial RGB-D frame plus a 25-entry camera trajectory and get back the
generated RGB (and optionally depth) frames.

    POST /imagine        one rollout
    POST /imagine_batch  N rollouts sharing one forward pass
    GET  /health         liveness + VRAM

The tensor path here mirrors BaseTrainer.prepare_batch + CogVTrainer.eval
exactly -- same normalisation, same conditioning, same seed -- minus the
dataset/mp4 round-trip. verify_service.py checks it against demo_run.py.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import types

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))          # coding-agent/imaginevln/service
PKG = os.path.dirname(HERE)                                  # coding-agent/imaginevln
AC_ROOT = os.path.dirname(os.path.dirname(PKG))              # 仓根
# 世界模型权重+代码（CogVideoX-2b / ckpt / code）——不进 git；默认放 <AC>/data/mw_export，
# 或用 MW_EXPORT 指到别处（四号机：/mnt/6t/mw_export）
EXPORT = os.environ.get("MW_EXPORT", os.path.join(AC_ROOT, "data", "mw_export"))
for p in (HERE, os.path.join(EXPORT, "code"), os.path.join(EXPORT, "code", "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mw_notext  # noqa: E402


class ImagineService:
    def __init__(self, ckpt=os.path.join(EXPORT, "ckpt"), base=os.path.join(EXPORT, "CogVideoX-2b")):
        from accelerate import Accelerator

        mw_notext.install()
        from video_sim.trainer.cogv import CogVTrainer

        self.args = types.SimpleNamespace(**json.load(open(os.path.join(ckpt, "args.json"))))
        self.args.pretrained_model_name_or_path = base
        self.acc = Accelerator(mixed_precision="bf16")
        self.dev = self.acc.device
        self.dtype = torch.bfloat16

        t0 = time.time()
        tr = CogVTrainer(self.args, self.acc)
        tr.to(self.dev, dtype=self.dtype)
        tr.build_pipe()
        for fn in ("enable_tiling", "enable_slicing"):
            if hasattr(tr.vae, fn):
                getattr(tr.vae, fn)()
        tr.set_grad()
        tr.load_from_scratch(ckpt)
        tr.to(self.dev, dtype=self.dtype)
        tr.transformer.eval()
        self.tr = tr
        self.lock = threading.Lock()
        self.load_s = round(time.time() - t0, 1)
        self.weights_gib = round(torch.cuda.memory_allocated() / 2**30, 2)
        self.n_calls = 0
        print(f"[mw] ready in {self.load_s}s, weights on card {self.weights_gib} GiB "
              f"(T5 dropped)", flush=True)

    # ---------------------------------------------------------------- helpers
    def _plucker(self, poses):
        """poses: (T, 18) -> local Plucker (T, 6, 48, 48), padded to num_frames
        by repeating the last entry -- the same rule dataset.py:442-445 uses."""
        from camera import Camera, get_plucker_embedding

        n = self.args.num_frames
        h = w = self.args.height // 8
        cams = [Camera(p, type="habitat") for p in poses]
        local, _ = get_plucker_embedding(cams, h, w, h, w, type="habitat")
        if len(local) < n:
            local = torch.cat([local] + [local[-1:]] * (n - len(local)), dim=0)
        return local[:n]

    @torch.no_grad()
    def imagine(self, items, want_depth=True):
        """items: list of {"rgb": (384,384,3) uint8, "depth": (384,384) float32
        in [0,1] = metres/10, "poses": (<=25, 18)}. Returns per-item dicts of
        uint8 frames."""
        from training.utils import randn_tensor, retrieve_timesteps
        from inference import decode_latents, inference

        args, tr, dev, dt = self.args, self.tr, self.dev, self.dtype
        n_f, S = args.num_frames, args.height
        B = len(items)

        def _fit(a, mode):
            if a.shape[0] == S and a.shape[1] == S:
                return a
            return np.asarray(Image.fromarray(a).resize((S, S), mode))

        rgb = np.stack([_fit(it["rgb"], Image.BILINEAR) for it in items])    # B,S,S,3
        dep = np.stack([_fit(it["depth"], Image.NEAREST) for it in items])   # B,S,S

        # Reproducibility: two RNGs feed a call. tr.generator seeds the initial
        # noise; the GLOBAL torch RNG seeds vae.encode's latent_dist.sample()
        # (encode_videos calls it without a generator). Reseed both, or request
        # N inherits request N-1's stream and no call can be reproduced.
        if tr.generator is not None:
            tr.generator.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # dataset.py: uint8 -> /255 -> Normalize(0.5, 0.5); the init frame is
        # repeated for all T because only latent frame 0 conditions anything.
        v = torch.from_numpy(rgb).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        v = (v - 0.5) / 0.5
        videos = v.unsqueeze(1).repeat(1, n_f, 1, 1, 1).to(dt)              # B,T,3,S,S

        d = torch.from_numpy(dep).to(dev).float().unsqueeze(1)              # B,1,S,S
        d = d.unsqueeze(1).repeat(1, n_f, 3, 1, 1).to(dt)                   # B,T,3,S,S

        cam = torch.stack([self._plucker(it["poses"]) for it in items]).to(dev, dt)

        lat = tr.encode_videos(videos)
        lat = torch.cat([lat, tr.encode_videos(tr.normalize_depth(d))], dim=2)
        image_latents = lat.clone()
        image_latents[:, args.cond_token:, :] = 0
        cam = tr.encode_camera(cam)

        shape = lat.shape
        timesteps, n_steps = retrieve_timesteps(tr.scheduler, args.inference_step, dev, None)
        noise = randn_tensor(shape, tr.generator, dev, dt) * tr.scheduler.init_noise_sigma
        rope = tr.prepare_position_embed(shape[3], shape[4], shape[1])

        out = inference(
            tr.transformer, tr.scheduler, noise,
            tr.encode_text([mw_notext.CONST_PROMPT] * B), timesteps, n_steps,
            rope, self.acc, len(timesteps), args, tr.pipe, tr.generator,
            camera_embeds=cam if args.camera_flag != "no" else None,
            image_latents=image_latents,
            map_feature=None, global_camera_embeds=None,
            mask_image=False, map_tmp_flag=False,
        ).detach()

        def _dec(chans):
            vid = torch.cat([decode_latents(tr.vae, s, tr.VAE_SCALING_FACTOR)
                             for s in out[:, :, chans].split(4)])
            vid = vid[:, :, -n_f:]
            return tr.pipe.video_processor.postprocess_video(video=vid, output_type="np")

        rgb_out = _dec(slice(0, 16))
        dep_out = _dec(slice(16, 32)) if (want_depth and out.shape[2] >= 32) else None

        res = []
        for i in range(B):
            r = {"rgb": (np.clip(rgb_out[i], 0, 1) * 255).astype(np.uint8)}
            if dep_out is not None:
                r["depth"] = np.clip(dep_out[i][..., 0], 0, 1).astype(np.float32)
            res.append(r)
        self.n_calls += B
        return res


# -------------------------------------------------------------------- codecs
def b64_png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def unb64_png(s):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


def b64_f32(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def unb64_f32(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).reshape(shape).copy()


def make_app(svc: ImagineService):
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(ok=True, calls=svc.n_calls, load_s=svc.load_s,
                       weights_gib=svc.weights_gib,
                       alloc_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
                       peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 2))

    def _run(payload):
        items = []
        for it in payload["items"]:
            hw = it.get("depth_shape", [384, 384])
            items.append({"rgb": unb64_png(it["rgb"]),
                          "depth": unb64_f32(it["depth"], tuple(hw)),
                          "poses": it["poses"]})
        want_depth = bool(payload.get("want_depth", False))
        t0 = time.time()
        with svc.lock:
            torch.cuda.reset_peak_memory_stats()
            out = svc.imagine(items, want_depth=want_depth)
            peak = torch.cuda.max_memory_allocated() / 2**30
        ms = int((time.time() - t0) * 1000)
        return jsonify(
            ms=ms, peak_gib=round(peak, 2), n=len(out),
            results=[{"rgb": [b64_png(f) for f in r["rgb"]],
                      **({"depth": [b64_f32(f) for f in r["depth"]]} if "depth" in r else {})}
                     for r in out])

    @app.post("/imagine")
    def imagine_one():
        p = request.get_json()
        return _run({"items": [p], "want_depth": p.get("want_depth", False)})

    @app.post("/imagine_batch")
    def imagine_many():
        return _run(request.get_json())

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9270)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    svc = ImagineService()
    make_app(svc).run(host=a.host, port=a.port, threaded=False)
