"""P1 服务侧验证（要 :9200/:9210/:9270 都在）：

1. 深度交叉验证 —— pano 视图 dir 0 的 depth_raw_base64 经 decode_depth_raw
   还原后，应与同一姿态 observe_egocentric 的深度（按 depth_units 换算）一致。
2. A/B rollout —— 找一个背后/侧面的候选点，各生成一次：
   一期口径（Front 起帧 + 全量转向） vs 二期口径（对应视图起帧 + 残差）。
   两张 sheet 存到 scratchpad 供肉眼对比，顺便打 k_eff 账。

跑法：python test_p1_services.py [--episode 0] [--outdir /tmp/...]
"""
from __future__ import annotations

import argparse
import base64
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagine_tools import (ImagineToolset, N_FRAMES, actions_to_poses,  # noqa: E402
                           as_ndarray, build_rollout_sheet, decode_depth_raw,
                           legacy_actions, metres_to_wm, norm_pi, np_to_png_b64,
                           png_to_np, residual_actions)

ENV_URL = os.environ.get("ENV_URL", "http://127.0.0.1:9200")
WP_URL = os.environ.get("WP_URL", "http://127.0.0.1:9210")
MW_URL = os.environ.get("MW_URL", "http://127.0.0.1:9270")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--split", default="rand100")
    ap.add_argument("--outdir", default=os.environ.get("P1_OUT", "/tmp/p1_ab"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ts = ImagineToolset(ENV_URL, WP_URL, MW_URL, out / "live")
    ts.reset_episode(args.split, args.episode)
    obs = ts.look()
    print(f"episode {args.episode}: {obs['n_cands']} candidates: {obs['waypoints']}")

    # ── 1. 深度交叉验证 ──
    ego = ts._call("env_habitat__observe_egocentric", {"trigger": "p1"})
    ego_dep = np.squeeze(as_ndarray(ego["depth"]).astype(np.float32))
    units = ego.get("depth_units") or {}
    ego_m = ego_dep * float(units.get("scale_m", 1.0)) if units.get("normalized") \
        else ego_dep
    v0 = next(v for v in ts.views if v.get("dir_id") == 0)
    pano_m = decode_depth_raw(v0["depth_raw_base64"], ts.depth_units)
    if pano_m.shape != ego_m.shape:
        print(f"NOTE: shapes differ ego {ego_m.shape} vs pano {pano_m.shape}")
    a, b = np.squeeze(pano_m), np.squeeze(ego_m)
    if a.shape == b.shape:
        diff = np.abs(a - b)
        print(f"depth cross-check: mean|Δ| {diff.mean():.4f} m, "
              f"p95 {np.percentile(diff, 95):.4f} m, "
              f"ego median {np.median(b):.2f} m, pano median {np.median(a):.2f} m")
        ok = diff.mean() < 0.05
    else:
        med_ratio = float(np.median(b[b > 0.1]) / max(np.median(a[a > 0.1]), 1e-6))
        print(f"depth cross-check (shape mismatch, median ratio ego/pano): {med_ratio:.3f}")
        ok = 0.8 < med_ratio < 1.25
    print("depth conversion:", "OK" if ok else "FAIL — check units handling")

    # ── 2. A/B：挑残差收益最大的候选点（|角度| 最大者） ──
    import requests
    cands = ts.cands
    ci = max(range(len(cands)), key=lambda i: abs(norm_pi(cands[i]["angle"])))
    c = cands[ci]
    ang_deg = math.degrees(norm_pi(c["angle"]))
    acts_new, dir_id, view, res = residual_actions(c["angle"], c["distance"])
    acts_old = legacy_actions(c["angle"], c["distance"])
    print(f"\nA/B candidate #{ci + 1}: {ang_deg:+.0f}°, {c['distance']:.2f} m -> "
          f"{view} view (residual {math.degrees(res):+.0f}°)\n"
          f"  legacy: {len(acts_old)} acts, k_eff {min(N_FRAMES, 1 + len(acts_old))}\n"
          f"  aligned: {len(acts_new)} acts, k_eff {min(N_FRAMES, 1 + len(acts_new))}")

    by_dir = {v.get("dir_id"): v for v in ts.views}

    def item(dir_key: int, actions: list[int]) -> dict:
        v = by_dir[dir_key]
        dep = metres_to_wm(decode_depth_raw(v["depth_raw_base64"], ts.depth_units))
        return {"rgb": np_to_png_b64(png_to_np(v["rgb_base64"])),
                "depth": base64.b64encode(
                    np.ascontiguousarray(dep, dtype=np.float32).tobytes()).decode(),
                "depth_shape": list(dep.shape),
                "poses": actions_to_poses(actions)}

    r = requests.post(f"{MW_URL}/imagine_batch",
                      json={"want_depth": False,
                            "items": [item(0, acts_old), item(dir_id, acts_new)]},
                      timeout=1800)
    r.raise_for_status()
    res_old, res_new = r.json()["results"]

    for tag, acts, rr, vname in (("legacy_front", acts_old, res_old, "Front"),
                                 ("aligned", acts_new, res_new, view)):
        k = min(N_FRAMES, 1 + len(acts))
        frames = [png_to_np(s) for s in rr["rgb"]][:k]
        sheet = build_rollout_sheet({
            "id": ci + 1, "view": vname, "angle_deg": ang_deg,
            "distance_m": c["distance"], "n_steps": len(acts), "frames": frames})
        (out / f"ab_{tag}.png").write_bytes(sheet)
        print(f"  wrote {out / f'ab_{tag}.png'} ({k} frames)")
    print("\nA/B sheets ready — eyeball them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
