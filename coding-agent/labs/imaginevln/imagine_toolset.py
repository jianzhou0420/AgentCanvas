"""WaypointToolSet + world-model rollouts.

Subclasses mini's wp toolset instead of forking it, so the `base` arm IS
mini's wp condition byte for byte (same observe/goto/stop, same panorama, same
budgets) and the `imagine` arm is that plus N extra images on every look.

Result layout, which the pruning in imagine_model.py depends on:

    content[0]      the panorama strip        <- kept in every turn, forever
    content[1]      the waypoint JSON
    content[2]      how to read the rollouts
    content[3..]    one rollout sheet per candidate, in waypoint order
                                              <- kept only for the newest look
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

_AC = os.environ.get("AGENTCANVAS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # coding-agent/imaginevln -> repo root
for p in (os.path.join(_AC, "coding-agent", "harnesses", "mini"),
          os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from toolset import HybridToolSet, ToolResult, WaypointToolSet, png_part, text_part  # noqa: E402
try:  # MapGPT's history formatter, loaded live from the VLNVerse source tree
    from vlnverse_mapgpt import format_mapgpt_history  # noqa: E402  — the real one
except ImportError:  # no VLNVerse checkout on this host (e.g. the H100 line):
    format_mapgpt_history = None  # the wp arm runs without the position-history
                                  # text; the hybrid arm never uses it

MW_URL = os.environ.get("MW_URL", "http://127.0.0.1:9270")
N_FRAMES = 25
FWD_STEP = 0.25
TURN_STEP = math.radians(15.0)
FILM_COLS = 6
FILM_TILE = 232


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _norm_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def waypoint_to_actions(angle_ccw: float, distance: float) -> list[int]:
    """(CCW radians, metres) -> habitat discrete actions 1=FWD 2=LEFT 3=RIGHT.
    Same floor as env_habitat.step_hightolow, and the yaw is quantised to 15°
    because every world-model training trajectory moved in 0.25 m / 15° steps."""
    n_turn = int(round(_norm_pi(angle_ccw) / TURN_STEP))
    n_fwd = int(distance // FWD_STEP)
    return [2 if n_turn > 0 else 3] * abs(n_turn) + [1] * n_fwd


def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def actions_to_poses(actions: list[int], n_frames: int = N_FRAMES) -> list[list[float]]:
    """25 c2w entries, habitat convention (-Z forward, +Y up), from the identity.
    The model conditions on poses RELATIVE to frame 0, so no env pose is needed;
    short trajectories repeat the last pose, as dataset.py:442-445 does."""
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


def _as_ndarray(o):
    if isinstance(o, dict) and "__ndarray__" in o:
        return np.frombuffer(base64.b64decode(o["__ndarray__"]),
                             dtype=np.dtype(o["dtype"])).reshape(o["shape"])
    return np.asarray(o)


def _png_np(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _np_b64(a: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def build_rollout_sheet(plan: dict) -> bytes:
    """Every predicted frame for one candidate, wrapped into a grid. One sheet
    per waypoint, so the image-pruning granularity matches the semantic unit the
    model has to map onto an action number."""
    frames = plan["frames"]
    n = len(frames)
    cols = min(FILM_COLS, n)
    rows = math.ceil(n / cols)
    pad, hdr = 5, 34
    canvas = Image.new("RGB", (cols * (FILM_TILE + pad) + pad,
                               hdr + rows * (FILM_TILE + pad) + pad), (248, 248, 248))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), f"Waypoint #{plan['id']}  ({plan['direction']}, "
                     f"{plan['angle_deg']:+.0f}°, {plan['distance_m']:.2f} m, "
                     f"{plan['n_steps']} actions) — predicted walk-through, "
                     f"frame 0 = now, frame {n - 1} = arrival",
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


class ImagineWaypointToolSet(WaypointToolSet):
    """wp toolset whose every look also carries the world model's prediction of
    walking to each numbered candidate."""

    def __init__(self, *args: Any, imagine: bool = True, mw_url: str = MW_URL,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.imagine = imagine
        self.mw_url = mw_url
        self.imagine_ms = 0
        self.imagine_calls = 0
        # filenames written by the CURRENT tool call. The monitor renders images
        # from `frames` on the tool_result event (its data-driven path); without
        # it the fallback groups by the obs_NNNN_ prefix and keeps only ONE rgb
        # per group, so the panorama and all but the last rollout vanish.
        self.last_frames: list[str] = []
        # MapGPT's one distinctive ingredient: the 3D position trace. mini's wp
        # deliberately leaks no pose ("never a pose", wp_bridge.py), so it is
        # rebuilt here and rendered by VLNVerse's own format_mapgpt_history,
        # then delivered with every look alongside the current rollouts.
        self._positions: list = []
        self._collisions: list[bool] = []
        self._history: list[dict] = []

    def _record_pose(self) -> None:
        st = self._call2("env_habitat__observe_camera_pose", {"trigger": "mapgpt"})
        self._positions.append(np.asarray(st.get("position") or [0, 0, 0], dtype=float))

    def _history_text(self) -> str:
        if format_mapgpt_history is None:
            return "(position history unavailable on this host)"
        return format_mapgpt_history({
            "history": self._history,
            "positions": self._positions,
            "collisions": self._collisions,
            "dialogue_history": [""] * (len(self._history) + 1),
        })

    def _pano_name(self) -> str:
        return f"obs_{self._obs_count:04d}_step{self.steps_taken:03d}.png"

    def _tool_observe(self, **kw: Any) -> ToolResult:
        self.last_frames = []
        if not self._positions:
            self._record_pose()          # episode start
        res = super()._tool_observe(**kw)
        if self.live_dir is not None and not (res.info or {}).get("error"):
            self.last_frames.append(self._pano_name())
        return self._attach(res)

    def _tool_goto(self, **kw: Any) -> ToolResult:
        """goto + AUTO-OBSERVE. mini's wp is observe-then-goto: goto clears
        _last_candidates and tells the agent to call observe() next, so a move
        costs two turns. wp_bridge.py has a HABITAT_AUTO_OBSERVE mode that folds
        the fresh look into the goto result; mini's port never got it. This is
        that mode, and it is on for BOTH arms so the interaction pattern is not
        one of the things that differs between them."""
        self.last_frames = []
        res = super()._tool_goto(**kw)
        if self.episode_over or (res.info or {}).get("error"):
            return res

        info = res.info or {}
        # format_mapgpt_history only branches on action being 'INVALID'/'STOP';
        # the distance/position numbers it prints come from _positions.
        self._collisions.append(bool(info.get("collided")))
        self._history.append({"step": len(self._history),
                              "action": info.get("moved_to"),
                              "distance": info.get("distance_m")})
        self._record_pose()
        look = super()._tool_observe()          # re-renders + re-predicts
        if self.live_dir is not None:
            self.last_frames.append(self._pano_name())
        self._tool_calls -= 1                   # the look is free; it was not a turn
        # drop goto's "call observe() to see the new surroundings" — with the
        # look already attached that instruction is now wrong
        moved = {k: v for k, v in res.info.items() if k not in ("kind", "note")}
        merged = ToolResult(
            content=[text_part(json.dumps(moved))] + list(look.content),
            info={**res.info, **{k: v for k, v in look.info.items() if k != "kind"}})
        return self._attach(merged)

    def _attach(self, res: ToolResult) -> ToolResult:
        parts0 = list(res.content) + [text_part(self._history_text())]
        res = ToolResult(content=parts0, info=res.info)
        return _attach_rollouts(self, res)


    def _rollouts(self, cands: list[dict[str, float]]) -> list[dict]:
        import time

        ego = self._call2("env_habitat__observe_egocentric", {"trigger": "wm"})
        rgb = _png_np(ego["rgb"])
        dep = np.squeeze(_as_ndarray(ego["depth"]).astype(np.float32))
        units = ego.get("depth_units") or {}
        if not units.get("normalized", True):        # raw metres on the wire
            dep = dep / float(units.get("scale_m", 10.0))
        dep = np.clip(dep, 0.0, 1.0)                 # world model wants metres/10

        plans = []
        for i, c in enumerate(cands):
            acts = waypoint_to_actions(c["angle"], c["distance"])
            plans.append({"id": i + 1,
                          "direction": WaypointToolSet._direction_of(c["angle"]),
                          "angle_deg": math.degrees(_norm_pi(c["angle"])),
                          "distance_m": c["distance"],
                          "actions": acts, "n_steps": len(acts),
                          "k_eff": min(N_FRAMES, 1 + len(acts))})

        payload = {"want_depth": False, "items": [{
            "rgb": _np_b64(rgb),
            "depth": base64.b64encode(np.ascontiguousarray(dep, dtype=np.float32).tobytes()).decode(),
            "depth_shape": list(dep.shape),
            "poses": actions_to_poses(p["actions"])} for p in plans]}

        t0 = time.time()
        r = requests.post(f"{self.mw_url}/imagine_batch", json=payload, timeout=1800)
        r.raise_for_status()
        j = r.json()
        self._last_ms = int((time.time() - t0) * 1000)
        self.imagine_ms += self._last_ms
        self.imagine_calls += 1

        for p, out in zip(plans, j["results"]):
            frames = [_png_np(s) for s in out["rgb"]]
            p["frames"] = frames[: p["k_eff"]]      # drop the repeated-still tail
        if self.live_dir is not None:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            for p in plans:
                name = f"obs_{self._obs_count:04d}_wp{p['id']}.png"
                (self.live_dir / name).write_bytes(build_rollout_sheet(p))
                self.last_frames.append(name)
        return plans

    _last_ms = 0


def _attach_rollouts(ts: Any, res: ToolResult) -> ToolResult:
    """Append the world model's rollout sheets (one per numbered candidate) to a
    look result that carries candidates. Shared by the wp surface
    (ImagineWaypointToolSet) and the hybrid surface's waypoint lens
    (ImagineHybridToolSet). Layout after this call: image 0 = the panorama
    strip, images 1.. = rollout sheets — the structure imagine_model.py's
    rollout-only pruning relies on."""
    if not ts.imagine or ts.episode_over or not ts._last_candidates:
        return res
    try:
        plans = ts._rollouts(ts._last_candidates)
    except Exception as exc:  # a world-model hiccup must not kill the episode
        return ToolResult(
            content=list(res.content) + [text_part(f"(rollout unavailable: {exc})")],
            info={**res.info, "imagine_error": repr(exc)})

    note = ["**Predicted futures.** A world model was run forward from where you "
            "stand along each waypoint's action sequence. One image per waypoint "
            "follows, in this order:"]
    for p in plans:
        note.append(f"  - Waypoint {p['id']} ({p['direction']}, {p['angle_deg']:+.0f}°, "
                    f"{p['distance_m']:.2f} m): {len(p['frames'])} frames")
    note.append("Within each image frames read left-to-right, top-to-bottom and are "
                "numbered: frame 0 is your current view, the last is arrival. These "
                "are PREDICTIONS — trust layout, geometry and whether a path stays "
                "open more than fine detail or exact colour. Use them to pick the "
                "waypoint whose predicted walk-through best matches the next part of "
                "the instruction; if one runs into a wall or leads away from it, do "
                "not go there. Only the newest look keeps these images.")

    parts = list(res.content) + [text_part("\n".join(note))]
    parts += [png_part(build_rollout_sheet(p)) for p in plans]
    info = {**res.info, "imagine": {
        "n": len(plans), "frames": sum(len(p["frames"]) for p in plans),
        "ms": ts._last_ms}}
    return ToolResult(content=parts, info=info)


# ── phase-2 (sdk_opus5_imagine) design on the hybrid surface ──────────────
# imagine() is an ON-DEMAND tool (the model names the candidates worth
# checking), the rollout starts from the panorama VIEW the candidate falls in
# (residual turns only), the geometry / decoding / sheet drawing are the SDK
# toolset's own functions, loaded from sdk/imagine_tools.py — nothing is
# re-implemented here. No MapGPT position trace, no VLNVerse dependency.
_SDK_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdk", "imagine_tools.py")


def _sdk_tools():
    import importlib.util
    spec = importlib.util.spec_from_file_location("imagine_sdk_tools", _SDK_TOOLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IMAGINE_MARK = "[imagine]"        # first text part of every imagine() result;
                                  # imagine_model.py prunes stale sheets by it
IMAGINE_DESC = (
    "OPTIONAL. Preview walking to chosen waypoints from your LATEST "
    "observe_waypoints() panorama: a learned world model predicts, frame by "
    "frame, what you would see on the way. Name only the waypoints worth "
    "checking, e.g. [1,3] — use it when the choice is genuinely uncertain (a "
    "junction, the instruction and the view not obviously lining up, two "
    "plausible candidates); skip it when the right move is already obvious. "
    "Returns one predicted filmstrip per waypoint. Costs no movement; a few "
    "seconds of compute per waypoint. It does not count as a look: goto() "
    "still follows the observe_waypoints() you already took.")
IMAGINE_SCHEMA = {
    "type": "object", "title": "imagineArguments",
    "properties": {"waypoints": {
        "type": "array", "items": {"type": "integer"},
        "description": "waypoint numbers from the LATEST observe_waypoints() panorama"}},
    "required": ["waypoints"]}


class ImagineHybridToolSet(HybridToolSet):
    """hybrid toolset (primitive step + waypoint goto, two observe lenses, the
    look-then-move gate — all unchanged) + the on-demand imagine(waypoints)
    tool of the phase-2 SDK line (sdk_opus5_imagine). An A/B against the
    plain hybrid cell isolates "imagination available on request" as the only
    delta; the imagine arm's cost is whatever the model chooses to spend."""

    def __init__(self, *args: Any, imagine: bool = True, mw_url: str = MW_URL,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.imagine = imagine
        self.mw_url = mw_url
        self.imagine_calls = 0
        self.imagine_waypoints = 0
        self.imagine_ms = 0
        self.last_frames: list[str] = []
        self._views: list[dict] | None = None      # 12 views of the LATEST panorama
        self._depth_units: dict | None = None
        self._sdk = _sdk_tools()
        if imagine:
            self._register("imagine", IMAGINE_DESC, IMAGINE_SCHEMA, self._tool_imagine)

    # capture the panorama views observe_waypoints() fetched — the rollout's
    # frame 0 is the candidate's own view (rgb + raw depth), no second render
    def _call2(self, function_name: str, inputs: dict[str, Any],
               config: dict[str, Any] | None = None, base: str | None = None) -> dict[str, Any]:
        out = super()._call2(function_name, inputs, config, base)
        if function_name == "env_habitat__observe_panorama":
            self._views = out.get("views") or []
        return out

    def _pano_name(self) -> str:
        return f"obs_{self._obs_count:04d}_step{self.steps_taken:03d}.png"

    def _tool_observe(self, **kw: Any) -> ToolResult:
        self.last_frames = []
        res = super()._tool_observe(**kw)
        if self.live_dir is not None and not (res.info or {}).get("error"):
            self.last_frames.append(self._pano_name())
        return res

    def _tool_observe_waypoints(self, **kw: Any) -> ToolResult:
        self.last_frames = []
        res = super()._tool_observe_waypoints(**kw)
        if self.live_dir is not None and not (res.info or {}).get("error"):
            self.last_frames.append(self._pano_name())
        return res

    def _units(self) -> dict:
        if self._depth_units is None:
            ego = self._call("env_habitat__observe_egocentric", {"trigger": "units"})
            self._depth_units = ego.get("depth_units") or {}
        return self._depth_units

    def _tool_imagine(self, waypoints: Any = None, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        self.last_frames = []
        sdk = self._sdk

        def err(msg: str) -> ToolResult:
            return ToolResult(content=[text_part(json.dumps({"error": msg}))],
                              info={"kind": "imagine", "error": msg})

        if self.episode_over:
            return err(f"episode already over ({self.end_reason})")
        if self._pending != "waypoints" or not self._last_candidates or not self._views:
            return err("imagine() needs a fresh observe_waypoints() with candidates "
                       "— look through the waypoint lens first")
        try:
            ids = sorted({int(w) for w in (waypoints or [])})
        except (TypeError, ValueError):
            return err(f"invalid waypoints {waypoints!r}; expected a list of integers")
        if not ids:
            return err("name at least one waypoint number")
        bad = [i for i in ids if not 1 <= i <= len(self._last_candidates)]
        if bad:
            return err(f"invalid waypoint ids {bad}; valid: 1-{len(self._last_candidates)}")

        try:
            units = self._units()
        except Exception as exc:  # env hiccup — never kills the episode
            return err(f"could not read depth units from the env: {exc}")
        by_dir = {v.get("dir_id"): v for v in self._views}
        plans, items = [], []
        for i in ids:
            c = self._last_candidates[i - 1]
            acts, dir_id, view, res = sdk.residual_actions(c["angle"], c["distance"])
            v = by_dir.get(dir_id)
            if v is None or not v.get("depth_raw_base64"):
                return err("panorama views lack raw depth — the env must serve views_rgbd "
                           "with depth_raw_base64 for imagine()")
            plans.append({"id": i, "view": view, "dir_id": dir_id,
                          "residual_deg": round(math.degrees(res), 1),
                          "angle_deg": round(math.degrees(_norm_pi(c["angle"])), 1),
                          "distance_m": c["distance"], "actions": acts,
                          "n_steps": len(acts),
                          "k_eff": min(sdk.N_FRAMES, 1 + len(acts))})
            rgb = sdk.png_to_np(v["rgb_base64"])
            dep = sdk.metres_to_wm(sdk.decode_depth_raw(v["depth_raw_base64"], units))
            items.append({
                "rgb": sdk.np_to_png_b64(rgb),
                "depth": base64.b64encode(
                    np.ascontiguousarray(dep, dtype=np.float32).tobytes()).decode(),
                "depth_shape": list(dep.shape),
                "poses": sdk.actions_to_poses(acts)})

        import time as _time
        t0 = _time.time()
        try:
            results: list = []
            for lo in range(0, len(items), sdk.MW_BATCH_MAX):
                r = requests.post(f"{self.mw_url}/imagine_batch",
                                  json={"want_depth": False,
                                        "items": items[lo:lo + sdk.MW_BATCH_MAX]},
                                  timeout=1800)
                r.raise_for_status()
                results += r.json()["results"]
        except Exception as exc:  # a world-model hiccup must not kill the episode
            return ToolResult(content=[text_part(json.dumps(
                {"error": f"world model unavailable: {exc}"}))],
                info={"kind": "imagine", "imagine_error": repr(exc)})
        ms = int((_time.time() - t0) * 1000)
        self.imagine_calls += 1
        self.imagine_waypoints += len(plans)
        self.imagine_ms += ms

        parts: list[dict] = []
        for p, out in zip(plans, results):
            p["frames"] = [sdk.png_to_np(x) for x in out["rgb"]][: p["k_eff"]]
            p["sheet"] = sdk.build_rollout_sheet(p)
            if self.live_dir is not None:
                self.live_dir.mkdir(parents=True, exist_ok=True)
                name = f"obs_{self._obs_count:04d}_wp{p['id']}.png"
                (self.live_dir / name).write_bytes(p["sheet"])
                self.last_frames.append(name)
        note = [IMAGINE_MARK + " Predicted walk-throughs (world model), one image per "
                "requested waypoint, in order:"]
        for p in plans:
            note.append(f"  - Waypoint {p['id']}: starts from the {p['view']} view of your "
                        f"panorama, {p['n_steps']} actions "
                        f"({abs(int(round(p['residual_deg'] / 15)))} turn + "
                        f"{int(p['distance_m'] // sdk.FWD_STEP)} forward), "
                        f"{len(p['frames'])} frames")
        note.append("Frames read left-to-right, top-to-bottom; frame 0 is the labeled "
                    "panorama view you already saw, the last frame is predicted arrival. "
                    "These are PREDICTIONS — trust layout, geometry and whether the path "
                    "stays open more than fine detail or colour. Your look is still "
                    "valid: goto() one of these waypoints, or step() after observe().")
        parts.append(text_part("\n".join(note)))
        parts += [png_part(p["sheet"]) for p in plans]
        self._live_log({"imagine": ids, "imagine_ms": ms})
        return ToolResult(content=parts, info={
            "kind": "imagine", "waypoints": ids, "imagine_ms": ms,
            "plans": [{k: v for k, v in p.items() if k not in ("frames", "sheet")}
                      for p in plans]})
