"""ImagineVLN 二期 toolset — 每集一个实例，被 in-process SDK MCP server 包一层。

角色分工（和一期 mini 版的差别）：

* 观察不是工具。runner 在每个决策周期开始时调 ``look()``，把全景条放进
  该周期的 user message —— “自动观察”。goto 的 tool result 只有文字。
* ``imagine(waypoints)`` 是按需工具：模型点名哪些候选点，就只为那几个生成。
* 预演是视图对齐的：候选点落在哪个视图（Left/Front/Right/Back），就用那个
  视图的 384×384 原图当第 0 帧，动作序列只含 ≤3 步残差转向 + 前进 ——
  背后的点不再烧 12 帧转身。合法性：世界模型的相机条件是相对 Plücker
  （第 0 帧归零），起始朝向对它透明。
* 事件（tool_use / tool_result + frames）由 toolset 自己写进 episode jsonl
  —— in-process 的好处：工具执行的当下就知道自己写了哪些帧文件。

服务依赖：:9200 env_habitat auto_host、:9210 smartway_waypoint、:9270 世界模型。
几何、深度、strip 画法全部沿用一期验证过的实现（legacy/run_mapgpt.py /
imagine_toolset.py），此处只改“第 0 帧选谁 + 残差动作序列”。
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

FWD_STEP = 0.25
TURN_STEP = math.radians(15.0)
N_FRAMES = 25
N_PANO_VIEWS = 12
MW_BATCH_MAX = 4                     # /imagine_batch 的服务端 assert
FILM_COLS = 6
FILM_TILE = 232
# dir_id i = 朝向 + i*30° CCW（TURN_LEFT 为 +yaw），所以 3=Left、9=Right。
STRIP_DIRS = ((3, "Left"), (0, "Front"), (9, "Right"), (6, "Back"))
# 视图对齐用：四个视图的中心角（CCW 弧度）与 dir_id。
VIEW_TABLE = (
    (0.0, 0, "Front"),
    (math.pi / 2, 3, "Left"),
    (math.pi, 6, "Back"),
    (-math.pi / 2, 9, "Right"),
)


def norm_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def align_view(angle_ccw: float) -> tuple[float, int, str, float]:
    """候选点角度 -> (视图中心角, dir_id, 视图名, 残差角)。

    残差 ∈ [-45°, +45°]；净转角 round(残差/15°)*15° + 中心角 与一期
    round(θ/15°)*15° 完全一致（中心角都是 15° 的倍数），改变的只是
    喂给世界模型的起始帧和序列长度，不是运动学。"""
    center, dir_id, label = min(
        VIEW_TABLE, key=lambda v: abs(norm_pi(angle_ccw - v[0])))
    return center, dir_id, label, norm_pi(angle_ccw - center)


def residual_actions(angle_ccw: float, distance: float) -> tuple[list[int], int, str, float]:
    """视图对齐的动作序列：(actions, dir_id, 视图名, 残差角)。
    n_fwd 与 env_habitat.step_hightolow 同一个 floor。"""
    _center, dir_id, label, res = align_view(angle_ccw)
    n_turn = int(round(res / TURN_STEP))
    n_fwd = int(distance // FWD_STEP)
    acts = [2 if n_turn > 0 else 3] * abs(n_turn) + [1] * n_fwd
    return acts, dir_id, label, res


def legacy_actions(angle_ccw: float, distance: float) -> list[int]:
    """一期的序列（Front 起帧，全量转向）—— 只用于 A/B 验证。"""
    n_turn = int(round(norm_pi(angle_ccw) / TURN_STEP))
    n_fwd = int(distance // FWD_STEP)
    return [2 if n_turn > 0 else 3] * abs(n_turn) + [1] * n_fwd


def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def actions_to_poses(actions: list[int], n_frames: int = N_FRAMES) -> list[list[float]]:
    """25 条 c2w（habitat 约定 -Z forward / +Y up），从单位阵积分。
    模型吃相对位姿（第 0 帧归零），短轨迹重复末位姿 —— dataset.py:442-445。"""
    f = 1.0 / math.tan(math.radians(90.0) / 2.0)
    R, t = np.eye(3), np.zeros(3)

    def pack():
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, t
        return [f, f, 0.5, 0.5, 0.0, 0.0] + c2w[:3, :4].reshape(-1).tolist()

    out = [pack()]
    for a in actions[: n_frames - 1]:
        if a == 1:
            t = t + R @ np.array([0.0, 0.0, -FWD_STEP])
        elif a == 2:
            R = R @ _Ry(+TURN_STEP)
        elif a == 3:
            R = R @ _Ry(-TURN_STEP)
        out.append(pack())
    while len(out) < n_frames:
        out.append(out[-1])
    return out[:n_frames]


# ── 编解码 ──

def png_to_np(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def np_to_png_b64(a: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def as_ndarray(o):
    if isinstance(o, dict) and "__ndarray__" in o:
        return np.frombuffer(base64.b64decode(o["__ndarray__"]),
                             dtype=np.dtype(o["dtype"])).reshape(o["shape"])
    return np.asarray(o)


def decode_depth_raw(b64: str, units: dict) -> np.ndarray:
    """depth_raw_base64（16-bit PNG，encoder 做了 raw*1000）-> 米。

    坑：encoder 假定输入是米，但 VLN-CE 的深度传感器通常 NORMALIZE_DEPTH=True，
    输入其实是 [0,1] 归一值 —— 所以 /1000 还原出 raw 后要按 depth_units 换算：
    normalized -> ×scale_m（=max_depth_m），否则本身就是米。"""
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    raw = np.asarray(img, dtype=np.float32) / 1000.0
    if units.get("known") and units.get("normalized"):
        raw = raw * float(units.get("scale_m", 10.0))
    return raw                                        # 米


def metres_to_wm(depth_m: np.ndarray) -> np.ndarray:
    """世界模型的深度约定：米/10，clip 到 [0,1]。"""
    return np.clip(depth_m / 10.0, 0.0, 1.0).astype(np.float32)


def heading_from_quat(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return (2.0 * math.atan2(y, w)) % (2.0 * math.pi)


def _font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate_strip(views: list[dict], cands: list[dict]) -> Image.Image:
    """[Left|Front|Right|Back] 条 + 编号绿圈 —— 一期原样。"""
    by_dir = {v.get("dir_id"): v for v in views}
    tiles = [Image.open(io.BytesIO(base64.b64decode(by_dir[d]["rgb_base64"]))).convert("RGB")
             for d, _ in STRIP_DIRS]
    w1, h = tiles[0].size
    wf = w1 * 4
    lab_h = max(20, h // 14)
    canvas = Image.new("RGB", (wf, h + lab_h), (255, 255, 255))
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * w1, lab_h))
    d = ImageDraw.Draw(canvas)
    lf = _font(int(lab_h * 0.7))
    for i, (_, label) in enumerate(STRIP_DIRS):
        bb = d.textbbox((0, 0), label, font=lf)
        d.text(((i + 0.5) * w1 - (bb[2] - bb[0]) / 2,
                (lab_h - (bb[3] - bb[1])) / 2 - bb[1]),
               label, fill=(0, 0, 0), font=lf)
    r = max(12, h // 20)
    nf = _font(int(r * 1.15))
    y = lab_h + h // 2
    for i, c in enumerate(cands):
        x = w1 * (1.5 - 2 * norm_pi(c["angle"]) / math.pi)
        if x < 0:
            x += wf
        x = min(max(x, r), wf - r)
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=(40, 200, 60), outline=(0, 0, 0), width=2)
        txt = str(i + 1)
        bb = d.textbbox((0, 0), txt, font=nf)
        d.text((x - (bb[2] - bb[0]) / 2 - bb[0], y - (bb[3] - bb[1]) / 2 - bb[1]),
               txt, fill=(0, 0, 0), font=nf)
    return canvas


def build_rollout_sheet(plan: dict) -> bytes:
    """一个候选点的全部预测帧铺成网格；表头标明起始视图。"""
    frames = plan["frames"]
    n = len(frames)
    cols = min(FILM_COLS, n)
    rows = math.ceil(n / cols)
    pad, hdr = 5, 34
    canvas = Image.new("RGB", (cols * (FILM_TILE + pad) + pad,
                               hdr + rows * (FILM_TILE + pad) + pad), (248, 248, 248))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8),
           f"Waypoint #{plan['id']}  ({plan['view']} view, {plan['angle_deg']:+.0f}°, "
           f"{plan['distance_m']:.2f} m, {plan['n_steps']} actions) — predicted "
           f"walk-through; frame 0 = the {plan['view']} view of your panorama, "
           f"frame {n - 1} = arrival",
           fill=(0, 0, 0), font=_font(16))
    sf = _font(13)
    for i, fr in enumerate(frames):
        x = pad + (i % cols) * (FILM_TILE + pad)
        y = hdr + (i // cols) * (FILM_TILE + pad)
        canvas.paste(Image.fromarray(fr).resize((FILM_TILE, FILM_TILE), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + FILM_TILE, y + FILM_TILE], outline=(170, 170, 170))
        tag = str(i)
        d.rectangle([x + 2, y + 2, x + 2 + 9 * len(tag) + 6, y + 20], fill=(0, 0, 0))
        d.text((x + 6, y + 4), tag, fill=(255, 240, 0), font=sf)
    b = io.BytesIO()
    canvas.save(b, format="PNG")
    return b.getvalue()


class ImagineToolset:
    """一集的全部环境状态 + 三个工具的实现（imagine / goto / stop）。"""

    def __init__(self, env_url: str, wp_url: str, mw_url: str, live_dir: Path,
                 max_moves: int = 30,
                 log_event: Callable[..., None] | None = None) -> None:
        self.env_url, self.wp_url, self.mw_url = env_url, wp_url, mw_url
        self.live_dir = live_dir
        self.max_moves = max_moves
        self.log_event = log_event or (lambda kind, **kw: None)

        self.cycle = 0                       # 观察轮次（look() 递增）
        self.moves = 0
        self.steps_taken = 0
        self.episode_over = False
        self.called_stop = False
        self.end_reason: str | None = None
        # 本会话内 imagine 调用数：runner 在开新会话时清零；goto 落地时若 >0，
        # 说明会话上下文里躺着已过时的预演图 → runner 触发会话重建剔除
        self.session_imagine_calls = 0

        self.views: list[dict] | None = None   # 12 视图（含 depth_raw）
        self.cands: list[dict] | None = None
        self.pano_png: bytes | None = None
        self.pano_name: str | None = None
        self.pano_history: list[tuple[int, bytes]] = []   # (cycle, png) 全保留
        self.last_obs: dict | None = None
        self.depth_units: dict = {}
        self.positions: list[np.ndarray] = []
        self.journey: list[dict] = []          # 每周期一条：pos / choice / reason
        self.imagine_calls = 0
        self.imagine_waypoints = 0
        self.imagine_ms = 0
        self.cands_offered = 0                 # RQ2 分母：见过的候选总数

    # ── HTTP ──

    def _call(self, fn: str, inputs: dict, config: dict | None = None,
              base: str | None = None) -> dict:
        body: dict = {"inputs": inputs}
        if config:
            body["config"] = config
        r = requests.post(f"{base or self.env_url}/call/{fn}", json=body, timeout=900)
        r.raise_for_status()
        return r.json()["outputs"]

    # ── episode 生命周期（runner 调） ──

    def reset_episode(self, split: str, index: int) -> dict:
        for name, value in (("dataset", "R2R-CE"), ("split", split),
                            ("episode_index", index)):
            requests.post(f"{self.env_url}/env-panel/field/{name}",
                          json={"value": value}, timeout=600).raise_for_status()
        requests.post(f"{self.env_url}/env-panel/action/play",
                      json={"params": {}}, timeout=600).raise_for_status()
        ep = self._call("env_habitat__reset", {"trigger": "imaginevln"})
        ego = self._call("env_habitat__observe_egocentric", {"trigger": "units"})
        self.depth_units = ego.get("depth_units") or {}
        self.live_dir.mkdir(parents=True, exist_ok=True)
        return ep

    def look(self) -> dict:
        """渲染全景 + 预测候选点。episode 开场由 runner 调一次；此后由
        tool_goto 在移动落地时内部调（自动观察，AgentCanvas 语义）。"""
        self.cycle += 1
        st = self._call("env_habitat__observe_camera_pose", {"trigger": "obs"})
        pos = np.asarray(st.get("position") or [0, 0, 0], dtype=float)
        self.positions.append(pos)

        pano = self._call("env_habitat__observe_panorama", {"trigger": "wp"},
                          config={"representation": "views_rgbd",
                                  "n_views": N_PANO_VIEWS})
        self.views = pano.get("views") or []
        slim = [{"dir_id": v.get("dir_id"), "rgb_base64": v.get("rgb_base64"),
                 "depth_base64": v.get("depth_base64")} for v in self.views]
        pred = self._call("smartway_waypoint__predict", {"views": slim},
                          base=self.wp_url)
        self.cands = [{"angle": float(v["angle"]), "distance": float(v["distance"])}
                      for v in (pred.get("candidates") or {}).values()
                      if isinstance(v, dict) and "angle" in v]
        self.cands_offered += len(self.cands)

        strip = annotate_strip(self.views, self.cands)
        b = io.BytesIO()
        strip.save(b, format="PNG")
        self.pano_png = b.getvalue()
        self.pano_name = f"obs_{self.cycle:03d}_pano.png"
        (self.live_dir / self.pano_name).write_bytes(self.pano_png)
        (self.live_dir / "latest.png").write_bytes(self.pano_png)

        wp_json = {
            str(i + 1): {"direction": self._dir_of(c["angle"]),
                         "angle_deg": round(math.degrees(norm_pi(c["angle"])), 1),
                         "distance_m": round(c["distance"], 2)}
            for i, c in enumerate(self.cands)}
        self.pano_history.append((self.cycle, self.pano_png))
        self.last_obs = {"cycle": self.cycle, "position": pos.round(2).tolist(),
                         "waypoints": wp_json, "n_cands": len(self.cands),
                         "pano_png": self.pano_png, "pano_name": self.pano_name}
        return self.last_obs

    def obs_summary(self) -> str:
        """监控日志用的一行摘要（前端只显示 300 字符，别倒 JSON）。"""
        o = self.last_obs or {}
        wps = o.get("waypoints") or {}
        parts = [f"#{k} {v['direction']} {v['angle_deg']:+.0f}° {v['distance_m']}m"
                 for k, v in wps.items()]
        pos = o.get("position") or [0, 0, 0]
        return (f"round {o.get('cycle')} · ({pos[0]}, {pos[2]}) · "
                f"{len(wps)} waypoints: " + ", ".join(parts)
                if wps else f"round {o.get('cycle')} · no waypoints here")

    def obs_text(self) -> str:
        """当前观察的文字部分（候选点 JSON + 预算），goto 结果和开场消息共用。"""
        o = self.last_obs or {}
        s = (f"New observation (round {o.get('cycle')}), waypoints:\n"
             + json.dumps(o.get("waypoints") or {}, indent=1)
             + f"\nPosition: ({o['position'][0]}, {o['position'][2]})"
             + f"\nMoves used: {self.moves}/{self.max_moves}")
        if not o.get("n_cands"):
            s += ("\nNo reachable waypoints predicted here — stop() if you "
                  "are at the goal.")
        return s

    def evaluate(self) -> dict:
        out = self._call("env_habitat__evaluate", {"trigger": "final"})
        return out.get("metrics") or {}

    # ── 工具实现（被 MCP 包装调用） ──

    def tool_imagine(self, waypoints: Any) -> list[dict]:
        """按需预演：只为点名的候选点生成，视图对齐。返回 MCP content parts。"""
        t_id = f"imagine[{waypoints}]"
        if self.episode_over:
            return self._text_only(t_id, {"error": "episode already over"})
        if not self.cands:
            return self._text_only(t_id, {"error": "no waypoint candidates here"})
        try:
            ids = sorted({int(w) for w in waypoints})
        except (TypeError, ValueError):
            return self._text_only(t_id, {"error":
                f"invalid waypoints {waypoints!r}; expected a list of integers"})
        bad = [i for i in ids if not 1 <= i <= len(self.cands)]
        if bad:
            return self._text_only(t_id, {"error":
                f"invalid waypoint ids {bad}; valid: 1-{len(self.cands)}"})

        self.log_event("tool_use", name="env__imagine",
                       input={"waypoints": ids})
        plans = []
        for i in ids:
            c = self.cands[i - 1]
            acts, dir_id, view, res = residual_actions(c["angle"], c["distance"])
            plans.append({
                "id": i, "view": view, "dir_id": dir_id,
                "residual_deg": round(math.degrees(res), 1),
                "angle_deg": round(math.degrees(norm_pi(c["angle"])), 1),
                "distance_m": c["distance"], "actions": acts,
                "n_steps": len(acts), "k_eff": min(N_FRAMES, 1 + len(acts))})

        by_dir = {v.get("dir_id"): v for v in self.views}
        items = []
        for p in plans:
            v = by_dir[p["dir_id"]]
            rgb = png_to_np(v["rgb_base64"])
            dep = metres_to_wm(decode_depth_raw(v["depth_raw_base64"],
                                                self.depth_units))
            items.append({
                "rgb": np_to_png_b64(rgb),
                "depth": base64.b64encode(
                    np.ascontiguousarray(dep, dtype=np.float32).tobytes()).decode(),
                "depth_shape": list(dep.shape),
                "poses": actions_to_poses(p["actions"])})

        t0 = time.time()
        results = []
        for lo in range(0, len(items), MW_BATCH_MAX):
            r = requests.post(f"{self.mw_url}/imagine_batch",
                              json={"want_depth": False,
                                    "items": items[lo:lo + MW_BATCH_MAX]},
                              timeout=1800)
            r.raise_for_status()
            results += r.json()["results"]
        ms = int((time.time() - t0) * 1000)
        self.imagine_calls += 1
        self.session_imagine_calls += 1
        self.imagine_waypoints += len(plans)
        self.imagine_ms += ms

        frame_names = []
        for p, out in zip(plans, results):
            p["frames"] = [png_to_np(s) for s in out["rgb"]][: p["k_eff"]]
            p["sheet"] = build_rollout_sheet(p)
            name = f"obs_{self.cycle:03d}_wp{p['id']}.png"
            (self.live_dir / name).write_bytes(p["sheet"])
            frame_names.append(name)

        note_lines = [
            "Predicted walk-throughs (world model), one image per requested "
            "waypoint, in order:"]
        for p in plans:
            note_lines.append(
                f"  - Waypoint {p['id']}: starts from the {p['view']} view of your "
                f"panorama, {p['n_steps']} actions "
                f"({abs(int(round(p['residual_deg'] / 15)))} turn + "
                f"{int(p['distance_m'] // FWD_STEP)} forward), "
                f"{len(p['frames'])} frames")
        note_lines.append(
            "Frames read left-to-right, top-to-bottom; frame 0 is the labeled "
            "panorama view you already saw, the last frame is predicted arrival. "
            "These are PREDICTIONS — trust layout, geometry and whether the path "
            "stays open more than fine detail or colour.")
        note = "\n".join(note_lines)

        self.log_event("tool_result", texts=[note], frames=frame_names,
                       cycle=self.cycle, imagine_ms=ms,
                       plans=[{k: v for k, v in p.items()
                               if k not in ("frames", "sheet")} for p in plans])
        parts: list[dict] = [{"type": "text", "text": note}]
        for p in plans:
            parts.append({"type": "image",
                          "data": base64.b64encode(p["sheet"]).decode(),
                          "mimeType": "image/png"})
        return parts

    def tool_goto(self, waypoint: Any, reason: str = "") -> list[dict]:
        """执行移动 + 自动观察（AgentCanvas 语义）：tool result 直接携带
        落地后的新全景 + 新候选点，模型无需也无法调 observe。"""
        if self.episode_over:
            return self._text_only("goto", {"error": "episode already over"})
        if not self.cands:
            return self._text_only("goto", {"error": "no waypoint candidates here"})
        try:
            waypoint = int(waypoint)
        except (TypeError, ValueError):
            return self._text_only("goto", {"error":
                f"invalid waypoint {waypoint!r}; expected an integer"})
        if not 1 <= waypoint <= len(self.cands):
            return self._text_only("goto", {"error":
                f"invalid waypoint {waypoint}; valid: 1-{len(self.cands)}"})

        self.log_event("tool_use", name="env__goto",
                       input={"waypoint": waypoint,
                              "reason": str(reason)[:300]})
        c = self.cands[waypoint - 1]
        out = self._call("env_habitat__step_hightolow",
                         {"angle": c["angle"], "distance": c["distance"]})
        info = out.get("info") or {}
        if isinstance(info.get("step_count"), (int, float)):
            self.steps_taken = int(info["step_count"])
        self.moves += 1
        self.journey.append({
            "cycle": self.cycle,
            "position": self.positions[-1].round(2).tolist(),
            "choice": waypoint,
            "direction": self._dir_of(c["angle"]),
            "distance_m": round(c["distance"], 2),
            "collided": bool(info.get("collided")),
            "reason": str(reason)[:200]})
        if out.get("terminated") or out.get("truncated"):
            self.episode_over = True
            self.end_reason = ("terminated" if out.get("terminated")
                               else "step_budget_exhausted")
        elif self.moves >= self.max_moves:
            self.episode_over = True
            self.end_reason = "move_budget_exhausted"
        self.cands = None                       # 位置变了，旧编号作废

        result = {"moved_to": waypoint, "distance_m": round(c["distance"], 2),
                  "collided": bool(info.get("collided")),
                  "moves_used": self.moves,
                  "moves_remaining": max(0, self.max_moves - self.moves),
                  "episode_over": self.episode_over}
        if self.episode_over:
            self.log_event("tool_result", texts=[json.dumps(result)],
                           cycle=self.cycle)
            return [{"type": "text", "text": json.dumps(result)}]

        # 自动观察：落地即看，新全景直接进 goto 的 tool result
        self.look()
        obs_txt = self.obs_text()
        self.log_event("tool_result",
                       texts=[f"moved to #{result['moved_to']} "
                              f"({result['distance_m']}m"
                              + (", collided" if result["collided"] else "")
                              + f", {result['moves_remaining']} moves left) → "
                              + self.obs_summary()],
                       frames=[self.pano_name], cycle=self.cycle)
        return [{"type": "text", "text": json.dumps(result)},
                {"type": "image",
                 "data": base64.b64encode(self.pano_png).decode(),
                 "mimeType": "image/png"},
                {"type": "text", "text": obs_txt}]

    def tool_stop(self, reason: str = "") -> list[dict]:
        if self.episode_over:
            return self._text_only("stop", {"error": "episode already over"})
        self.log_event("tool_use", name="env__stop",
                       input={"reason": str(reason)[:300]})
        self._call("env_habitat__step_discrete", {"action": 0})
        self.steps_taken += 1
        self.episode_over = True
        self.called_stop = True
        self.end_reason = "stop_called"
        self.journey.append({"cycle": self.cycle,
                             "position": (self.positions[-1].round(2).tolist()
                                          if self.positions else None),
                             "choice": "STOP", "reason": str(reason)[:200]})
        result = {"stopped": True, "episode_over": True}
        self.log_event("tool_result", texts=[json.dumps(result)], cycle=self.cycle)
        return [{"type": "text", "text": json.dumps(result)}]

    # ── helpers ──

    def _text_only(self, tool: str, result: dict) -> list[dict]:
        self.log_event("tool_result", texts=[json.dumps(result)],
                       cycle=self.cycle, tool=tool)
        return [{"type": "text", "text": json.dumps(result)}]

    @staticmethod
    def _dir_of(angle: float) -> str:
        a = angle % (2 * math.pi)
        if a <= math.pi / 4 or a >= 7 * math.pi / 4:
            return "Front"
        if a <= 3 * math.pi / 4:
            return "Left"
        if a <= 5 * math.pi / 4:
            return "Back"
        return "Right"

    def journey_text(self) -> str:
        if not self.journey:
            return "(none yet — this is your starting position)"
        lines = []
        for j in self.journey:
            if j["choice"] == "STOP":
                lines.append(f"  round {j['cycle']}: STOP — {j['reason']}")
            else:
                lines.append(
                    f"  round {j['cycle']}: at ({j['position'][0]}, {j['position'][2]}) "
                    f"chose waypoint {j['choice']} ({j['direction']}, "
                    f"{j['distance_m']} m)"
                    + (" [collided]" if j.get("collided") else "")
                    + (f" — {j['reason']}" if j.get("reason") else ""))
        return "\n".join(lines)
