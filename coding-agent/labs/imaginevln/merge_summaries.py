#!/usr/bin/env python
"""并行分片 summary 合并器：把 run 目录下 summary_w*.json 的 episodes 汇成
board 契约的 summary.json。循环运行（10s 间隔），目录里出现 MERGE_STOP 时
做最后一次合并后退出。"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path


def merge(out_dir: Path) -> dict:
    rows: list[dict] = []
    meta: dict = {}
    for p in sorted(glob.glob(str(out_dir / "summary_w*.json"))):
        try:
            s = json.loads(Path(p).read_text())
        except Exception:
            continue                     # worker 正在写，下轮再收
        meta = meta or {k: s.get(k) for k in ("run", "arm", "model", "split", "config")}
        rows += s.get("episodes") or []
    rows.sort(key=lambda e: (e.get("index") is None, e.get("index")))
    scored = [e for e in rows if (e.get("metrics") or {}).get("success") is not None]
    agg: dict = {"episode_count": len(scored)}
    for k in ("success", "spl", "ndtw", "oracle_success", "distance_to_goal"):
        vals = [float(e["metrics"][k]) for e in scored
                if isinstance((e.get("metrics") or {}).get(k), (int, float))]
        if vals:
            agg[k] = round(sum(vals) / len(vals), 4)
    (out_dir / "summary.json").write_text(json.dumps(
        {**meta, "aggregate": agg, "episodes": rows}, indent=2, default=str))
    return agg


if __name__ == "__main__":
    out = Path(sys.argv[1])
    while True:
        agg = merge(out)
        if (out / "MERGE_STOP").exists():
            print("final:", json.dumps(agg))
            break
        time.sleep(10)
