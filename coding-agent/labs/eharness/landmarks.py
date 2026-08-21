"""Landmark grounding — SAM 3 concept masks fused with the depth frame.

The geometry organ (``depthmap.py``) can find a gap between two things but has
no idea WHAT they are. This organ supplies the nouns: it segments the landmark
phrases the instruction split already produced ("the pool", "the bar", "chairs")
and fuses each mask with the same depth frame, so every landmark comes back as
egocentric fact —

    the pool: on your right, 3.4 m away, spanning about 40° of your view

— which is exactly the shape of evidence the harness has been missing.

Two failures it exists to kill, both convicted in EP0 reruns:
  * the 9B milestone judge FALSE-NEGATIVE — it looked at a frame with the pool
    plainly visible and said the pool was not there, vetoed a correct STOP, and
    the agent wandered 300 steps. A detector-backed register turns that
    coin-flip into bookkeeping: the harness KNOWS the pool was seen in frames
    0-7 at 3-9 m on the right, and can say so.
  * the RECEPTION-DESK look-alike — a stone counter that reads as "the bar".
    Geometry plus the route's own order disambiguates: a "bar" seen before the
    gap has been traversed is a look-alike by construction.

Doctrine: an organ. The model never receives masks, boxes, pixel coordinates or
world positions — only sentences in its own egocentric frame. Detection is
advisory; a missing detector, missing weights or a dead server must degrade to
"no landmark evidence", never to a crash.

Serving: model_sam is an auto_host of its own —
    python -m app.server.auto_host --file workspace/nodesets/model/model_sam.py \
        --class SamNodeSet --port 9220
It must run in an env whose torch has kernels for this GPU (an RTX 5090 needs
sm_120: torch >= 2.7/cu128 — ``ac-wp`` qualifies, ``agentcanvas``'s torch 2.5.1
does NOT and fails with "no kernel image is available") and whose transformers
ships ``Sam3Model`` (>= 5.13). SAM 3 weights are HF-gated: request access on the
facebook/sam3 model page first.
"""
from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw
except Exception:  # pragma: no cover
    PILImage = None
    ImageDraw = None

from eharness.depthmap import FOV_DEG, decode_depth, to_metres

DEFAULT_SAM_URL = "http://127.0.0.1:9220"
SCORE_THRESH = 0.5
MAX_INSTANCES = 3          # per phrase; more is noise for a navigation prompt
MIN_MASK_PX = 200          # a 30-pixel blob is not a landmark
MAX_MASK_FRAC = 0.60       # …and a mask that IS the whole picture is not one
                           # either (user report 2026-08-16: the Human tab's
                           # SAM overlay showed one wall-to-wall mask). Region
                           # words the resolver keeps verbatim — "bedroom",
                           # "kitchen area" — segment the ENTIRE view once the
                           # camera stands inside that region: it localises
                           # nothing, and stamped it would paint the semantic
                           # layer across the whole wedge. Skipped, and the
                           # miss note says why.

# An open-vocabulary detector is not a dictionary lookup — it is a match between
# a phrase and pixels, and the SAME piece of furniture answers to some words and
# not others. R2R instructions say "the bar", which in a Matterport house is a
# counter with stools; asked for "bar" the detector finds nothing, asked for
# "bar counter" or "kitchen island" it finds it. The instruction's noun is the
# one thing we cannot change, so the detector gets several ways to hear it.
#
# One entry per SURFACE FORM the instruction might use, mapping to what the
# detector is actually good at. Cheap to extend; each extra word is one more
# SAM call, so keep the lists to what has been seen to matter.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "bar": ("bar counter", "kitchen island", "countertop"),
    "counter": ("bar counter", "countertop", "kitchen island"),
    "pool": ("swimming pool", "pool of water"),
    "chairs": ("chair", "stool", "armchair"),
    "chair": ("chairs", "stool", "armchair"),
    "table": ("dining table", "desk", "coffee table"),
    "sofa": ("couch", "armchair"),
    "couch": ("sofa", "armchair"),
    "stairs": ("staircase", "steps", "stairway"),
    "door": ("doorway", "door frame"),
    "doorway": ("door", "door frame"),
    "sink": ("kitchen sink", "washbasin"),
    "plant": ("potted plant", "houseplant"),
    "rug": ("carpet", "mat"),
    "tv": ("television", "monitor", "screen"),
}


def variants(phrase: str, limit: int = 3) -> list[str]:
    """The phrase itself first, then other ways of saying it.

    Order matters: the word the instruction used is what the model and the
    person are both thinking in, so it gets first refusal and its name is what
    the sighting is reported under. The alternates only run when it finds
    nothing."""
    p = phrase.strip().lower()
    alts = list(SYNONYMS.get(p, ()))
    if p.endswith("s") and p[:-1] not in alts:
        alts.append(p[:-1])                       # chairs → chair
    elif not p.endswith("s") and p + "s" not in alts:
        alts.append(p + "s")
    return [p, *alts][:1 + max(0, limit)]


DEPTH_PCTL = 25            # near quartile of the masked depth = "how far it is"

# Phrases that are never landmarks, so we do not waste a 1-2 s detector call.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "at", "in", "on", "past", "by",
    "your", "you", "it", "that", "this", "there", "where", "when", "then",
    "go", "walk", "stop", "wait", "turn", "left", "right", "straight", "ahead",
    "between", "until", "get", "reach", "toward", "towards", "into", "onto",
    "corner", "side", "front", "back", "way", "step", "steps", "will",
}

# ── generic-phrase filter (user ruling 2026-08-16: 关键词不应该有 "room"
# 这种东西，因为到处都是) ──────────────────────────────────────────────
# A landmark must DISCRIMINATE. "room" / "area" / "small room" name a
# category every indoor scene is made of — SAM grounds them on whatever
# enclosure the camera stands in (measured: a wall-to-wall mask from inside
# the room), the semantic layer gets a label that is true everywhere, and
# the detector ledger tells the judge nothing. A MODIFIED phrase whose
# modifier is itself contentful ("living room", "kitchen area") survives —
# the content word does the discriminating.
_GENERIC_HEADS = {
    "room", "rooms", "area", "areas", "space", "spaces", "place", "places",
    "spot", "spots", "end", "part", "section", "middle", "entrance", "exit",
}
_GENERIC_MODS = {
    "the", "a", "an", "this", "that", "other", "another", "next", "first",
    "second", "third", "last", "same", "new", "old", "small", "little",
    "big", "large", "tiny", "narrow", "wide", "open", "far", "near",
}


def is_generic_phrase(phrase: str) -> bool:
    """True when the phrase carries no discriminating content word —
    every word is a generic head or a bare modifier ("room", "the small
    room", "open area"). "living room" and "kitchen area" are NOT generic:
    "living"/"kitchen" discriminate."""
    words = re.findall(r"[a-zA-Z]+", (phrase or "").lower())
    if not words:
        return True
    heads = [w for w in words if w in _GENERIC_HEADS]
    rest = [w for w in words if w not in _GENERIC_HEADS
            and w not in _GENERIC_MODS]
    return bool(heads) and not rest


def drop_generic(phrases: list[str]) -> list[str]:
    """Filter a landmark list down to phrases that can discriminate."""
    return [p for p in phrases if not is_generic_phrase(p)]


@dataclass
class Sighting:
    """One landmark instance, in the only frame the harness admits."""

    phrase: str
    bearing: float            # radians, counter-clockwise positive (left)
    distance: float           # metres to the near quartile of its surface
    angular_width: float      # radians it spans across the view
    score: float
    pixels: int
    frame_hint: str = ""
    # Which word the detector actually answered to, when it was not the phrase
    # the caller asked for. Diagnostic only — the model is always told the
    # instruction's own noun, so "bar" stays "bar" even when SAM needed
    # "bar counter" to see it.
    matched_as: str = ""
    mask: Any = None          # bool array in RGB space — for the human overlay
                              # only; the model never receives a mask

    def describe(self) -> str:
        deg = math.degrees(self.bearing)
        if abs(deg) < 12:
            side = "straight ahead"
        elif abs(deg) > 75:
            side = f"far to your {'left' if deg > 0 else 'right'}"
        else:
            side = f"to your {'left' if deg > 0 else 'right'}"
        span = math.degrees(self.angular_width)
        size = " (it fills much of the view)" if span > 55 else ""
        return f"the {self.phrase}: {side}, about {self.distance:.1f} m away{size}"


@dataclass
class LandmarkRegister:
    """What the harness has actually SEEN, with its own eyes, and when.

    Monotone: a sighting is never deleted, because "I saw the pool 200 steps
    ago" stays true. This is the bookkeeping the milestone judge could not do
    from pixels."""

    seen: dict[str, list[Sighting]] = field(default_factory=dict)
    first_step: dict[str, int] = field(default_factory=dict)
    last_step: dict[str, int] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)

    def record(self, sightings: list[Sighting], step: int) -> None:
        for s in sightings:
            self.seen.setdefault(s.phrase, []).append(s)
            self.first_step.setdefault(s.phrase, step)
            self.last_step[s.phrase] = step
            self.misses[s.phrase] = 0

    def record_miss(self, phrase: str) -> None:
        self.misses[phrase] = self.misses.get(phrase, 0) + 1

    def ever_seen(self, phrase: str) -> bool:
        return bool(self.seen.get(phrase))

    def closest(self, phrase: str) -> Sighting | None:
        hits = self.seen.get(phrase) or []
        return min(hits, key=lambda s: s.distance) if hits else None

    def evidence_line(self, phrase: str, step: int) -> str:
        """A sentence for the judge — bookkeeping, not perception."""
        hits = self.seen.get(phrase) or []
        if not hits:
            return f"the {phrase} has never been detected in any frame so far"
        near = min(hits, key=lambda s: s.distance)
        first, last = self.first_step.get(phrase, 0), self.last_step.get(phrase, 0)
        span = "" if first == last else f" between step {first} and step {last}"
        tail = ""
        if step - last > 20:
            tail = (f"; it was last seen {step - last} steps ago and is no longer "
                    "in view, i.e. it has been passed")
        return (f"the {phrase} was detected in {len(hits)} frame(s){span}, "
                f"closest approach about {near.distance:.1f} m{tail}")


def landmark_phrases(segments: list[str], terminate: str = "") -> list[str]:
    """Noun phrases worth grounding, pulled from the route the splitter made.

    Deliberately dumb: content words minus a navigation stoplist. A wrong phrase
    costs one detector call and returns nothing; a clever parser costs an LLM
    call per episode and can still be wrong."""
    out: list[str] = []
    for text in [*segments, terminate]:
        for word in re.findall(r"[a-zA-Z]+", (text or "").lower()):
            if (len(word) < 3 or word in _STOPWORDS or word in out
                    # a bare generic head ("room", "area") grounds on
                    # whatever enclosure the camera stands in — never a
                    # single-word landmark (user ruling 2026-08-16)
                    or word in _GENERIC_HEADS or word in _GENERIC_MODS):
                continue
            out.append(word)
    return out[:6]


def upscale_b64(rgb_png_b64: str, target_px: int) -> str:
    """Resize a base64 PNG so its long side is at least ``target_px``.

    SAM 3 rescales its input internally, so a frame handed over below that
    working resolution has already lost the small, distant objects (a bar
    counter at 10 m) before the model runs. Upscaling invents no detail — the
    real lever is rendering the episode larger — but it stops the pipeline from
    downsampling twice. Returns the input unchanged if it is already big
    enough or cannot be decoded."""
    if PILImage is None or not rgb_png_b64:
        return rgb_png_b64
    try:
        img = PILImage.open(BytesIO(base64.b64decode(rgb_png_b64))).convert("RGB")
    except Exception:  # noqa: BLE001
        return rgb_png_b64
    if max(img.size) >= target_px:
        return rgb_png_b64
    scale = target_px / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)),
                     PILImage.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_from_b64(mask_b64: str) -> np.ndarray | None:
    if PILImage is None or not mask_b64:
        return None
    try:
        img = PILImage.open(BytesIO(base64.b64decode(mask_b64)))
        return np.asarray(img.convert("L")) > 127
    except Exception:  # noqa: BLE001
        return None


def _fuse(mask: np.ndarray, depth_m: np.ndarray, hfov_deg: float) -> tuple:
    """Mask + depth → (bearing, distance, angular width).

    The mask is resized to the depth grid by nearest neighbour: SAM sees the RGB
    frame (512²) while depth is 256², and indexing one with the other's
    coordinates is a silent ~12 % shear."""
    dh, dw = depth_m.shape[:2]
    if mask.shape != (dh, dw):
        ys = (np.arange(dh) * mask.shape[0] / dh).astype(int).clip(0, mask.shape[0] - 1)
        xs = (np.arange(dw) * mask.shape[1] / dw).astype(int).clip(0, mask.shape[1] - 1)
        mask = mask[np.ix_(ys, xs)]
    sel = mask & (depth_m > 0.05)
    n = int(sel.sum())
    if n < 20:
        return None
    cols = np.nonzero(sel.any(axis=0))[0]
    f = (dw / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    centre = float(np.average(np.nonzero(sel)[1]))
    bearing = -math.atan2(centre - dw / 2.0, f)          # LEFT positive
    lo = -math.atan2(float(cols.max()) - dw / 2.0, f)
    hi = -math.atan2(float(cols.min()) - dw / 2.0, f)
    distance = float(np.percentile(depth_m[sel], DEPTH_PCTL))
    return bearing, distance, abs(hi - lo), n


class LandmarkOrgan:
    """Calls SAM 3 for a phrase and turns masks into egocentric sightings.

    Never raises: every failure path returns an empty list so a dead detector
    degrades the harness to pure geometry rather than ending an episode."""

    def __init__(self, sam_url: str = DEFAULT_SAM_URL, *, hfov_deg: float = FOV_DEG,
                 score_thresh: float = SCORE_THRESH, timeout: float = 120.0,
                 variant: str = "sam3") -> None:
        self.sam_url = sam_url.rstrip("/")
        self.hfov_deg = hfov_deg
        self.score_thresh = score_thresh
        self.timeout = timeout
        self.variant = variant
        self.available: bool | None = None      # None = not probed yet
        self.last_error = ""
        self.calls = 0
        self.seconds = 0.0
        # phrase → every word tried for it, when none of them matched. This is
        # what tells a human "the detector cannot see your bar under any name",
        # as opposed to "there is no bar here".
        self.misses: dict[str, list[str]] = {}

    def probe(self) -> bool:
        import requests
        try:
            r = requests.get(f"{self.sam_url}/health", timeout=5)
            self.available = r.status_code == 200
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
        return bool(self.available)

    def segment(self, rgb_png_b64: str, phrase: str) -> list[dict[str, Any]]:
        import requests
        if self.available is False:
            return []
        body = {"inputs": {"image_b64": rgb_png_b64, "text": phrase},
                "config": {"variant": self.variant,
                           "score_thresh": str(self.score_thresh)}}
        t0 = time.time()
        try:
            r = requests.post(f"{self.sam_url}/call/model_sam__segment_text",
                              json=body, timeout=self.timeout)
            if r.status_code != 200:
                self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                self.available = False
                return []
            envelope = json.loads(r.json()["outputs"]["masks"] or "{}")
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.available = False
            return []
        finally:
            self.calls += 1
            self.seconds += time.time() - t0
        self.available = True
        return list(envelope.get("masks") or [])[:MAX_INSTANCES]

    def sightings(self, rgb_png_b64: str, depth_field: Any,
                  phrases: list[str], *, scale_m: float | None = None,
                  synonyms: int = 3) -> list[Sighting]:
        """Ground each phrase in the frame, trying other words for it if the
        instruction's own word finds nothing.

        `synonyms` is how many alternates to fall back through; 0 restores the
        single-query behaviour. The fallback only runs on a MISS, so a phrase
        the detector already knows costs exactly one call as before — and a
        phrase it does not costs a few instead of silently returning nothing.

        Whatever word finally matched, the sighting is reported under the word
        the caller asked for: the instruction says "the bar", so the memory and
        the sentences shown to the model have to say "bar" too, even when the
        detector needed to be asked for "bar counter" to find it."""
        depth = (depth_field if isinstance(depth_field, np.ndarray)
                 else decode_depth(depth_field))
        if depth is None or depth.ndim != 2:
            return []
        depth_m, _ = to_metres(depth, scale_m)
        # frame geometry for the floor-expanse filter below (user report
        # 2026-08-16: SAM grounds "hallway" on the ENTIRE walking surface)
        _h, _w = depth_m.shape
        _f = (_w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        _us, _vs = np.meshgrid(np.arange(_w, dtype=np.float32),
                               np.arange(_h, dtype=np.float32))
        _dx = (_us - _w / 2.0) / _f
        _dy = -(_vs - _h / 2.0) / _f
        _yg = _dy * depth_m
        _valid = np.isfinite(depth_m) & (depth_m > 0.05)
        _low = _yg[_valid & (_dy < -0.1)]
        _floor_y = float(np.percentile(_low, 3)) if _low.size > 500 else None
        out: list[Sighting] = []
        for phrase in phrases:
            tried: list[str] = []
            whole_view = False
            floor_expanse = False
            for word in variants(phrase, synonyms):
                found: list[Sighting] = []
                for m in self.segment(rgb_png_b64, word):
                    mask = _mask_from_b64(m.get("mask_b64", ""))
                    if mask is None or int(mask.sum()) < MIN_MASK_PX:
                        continue
                    if float(mask.mean()) > MAX_MASK_FRAC:
                        whole_view = True
                        continue
                    # the floor-expanse filter (user 2026-08-16: SAM 把整个
                    # 地面当 hallway): a mask that IS the walking surface —
                    # ≥70 % of its pixels on the floor plane, spread over
                    # metres — localises nothing and would smear one label
                    # across the whole map. Furniture survives: a counter
                    # or bed top sits well above the floor band.
                    if _floor_y is not None:
                        mm = mask
                        if mm.shape != depth_m.shape:
                            ys_i = (np.arange(_h) * mm.shape[0]
                                    // _h).clip(0, mm.shape[0] - 1)
                            xs_i = (np.arange(_w) * mm.shape[1]
                                    // _w).clip(0, mm.shape[1] - 1)
                            mm = mm[np.ix_(ys_i, xs_i)]
                        sel_m = mm.astype(bool) & _valid
                        if int(sel_m.sum()) >= 50:
                            ymm = _yg[sel_m]
                            if float((np.abs(ymm - _floor_y)
                                      < 0.15).mean()) >= 0.7:
                                xs_m = (_dx * depth_m)[sel_m]
                                zs_m = depth_m[sel_m]
                                if max(float(xs_m.max() - xs_m.min()),
                                       float(zs_m.max() - zs_m.min())) >= 2.2:
                                    floor_expanse = True
                                    continue
                    fused = _fuse(mask, depth_m, self.hfov_deg)
                    if fused is None:
                        continue
                    bearing, distance, width, npx = fused
                    found.append(Sighting(
                        phrase=phrase, bearing=bearing, distance=distance,
                        angular_width=width,
                        score=float(m.get("iou_score") or 0.0),
                        pixels=npx, mask=mask,
                        matched_as="" if word == phrase else word))
                tried.append(word)
                if found:
                    out.extend(found)
                    break
            else:
                notes = []
                if whole_view:
                    notes.append("(only whole-view masks — the camera "
                                 "likely stands INSIDE this region; "
                                 "nothing to localise)")
                if floor_expanse:
                    notes.append("(only floor-expanse masks — SAM grounded "
                                 "this on the walking surface itself; "
                                 "nothing to localise)")
                self.misses[phrase] = tried + notes
        return out


# one stable colour per phrase so the same landmark keeps its colour between
# frames — a viewer tracking "the pool" should not have to re-read the legend
_MASK_COLOURS = ((80, 190, 255), (255, 170, 70), (140, 230, 130),
                 (240, 130, 200), (255, 235, 90), (170, 150, 255))


def render_masks(rgb_png_b64: str, sightings: list[Sighting]) -> bytes:
    """SAM 3's masks painted on the view, each labelled with its distance.

    Purely for a human watching: it is the only way to tell "the detector found
    the pool" apart from "the detector found a blue rectangle and called it the
    pool", and that difference decided several EP0 failures."""
    if PILImage is None or not rgb_png_b64:
        return b""
    try:
        img = PILImage.open(BytesIO(base64.b64decode(rgb_png_b64))).convert("RGB")
    except Exception:  # noqa: BLE001
        return b""
    base = np.asarray(img).astype(np.float32)
    overlay = base.copy()
    phrases: list[str] = []
    boxes: list[tuple[str, int, int, tuple[int, int, int], float]] = []
    for s in sightings:
        if s.mask is None:
            continue
        m = np.asarray(s.mask, dtype=bool)
        if m.shape != base.shape[:2]:
            ys = (np.arange(base.shape[0]) * m.shape[0] / base.shape[0]).astype(int)
            xs = (np.arange(base.shape[1]) * m.shape[1] / base.shape[1]).astype(int)
            m = m[np.ix_(np.clip(ys, 0, m.shape[0] - 1),
                         np.clip(xs, 0, m.shape[1] - 1))]
        if not m.any():
            continue
        if s.phrase not in phrases:
            phrases.append(s.phrase)
        colour = _MASK_COLOURS[phrases.index(s.phrase) % len(_MASK_COLOURS)]
        overlay[m] = 0.45 * base[m] + 0.55 * np.array(colour, dtype=np.float32)
        yy, xx = np.nonzero(m)
        boxes.append((s.phrase, int(xx.mean()), int(yy.min()), colour, s.distance))
    out = PILImage.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for phrase, cx, top, colour, dist in boxes:
        draw.text((max(2, cx - 22), max(2, top - 12)),
                  f"{phrase} {dist:.1f}m", fill=colour)
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def gap_between(a: Sighting, b: Sighting, *, body_m: float = 0.5) -> dict | None:
    """The waypoint that walks BETWEEN two named landmarks.

    "Walk between the bar and chairs" is the segment every EP0 run failed to
    evidence. Two sightings give the gap's bearing and, by the law of cosines,
    its width; if the robot fits, aim just past the midpoint."""
    if a.distance <= 0.1 or b.distance <= 0.1:
        return None
    bearing = 0.5 * (a.bearing + b.bearing)
    reach = 0.5 * (a.distance + b.distance)
    d_ang = abs(a.bearing - b.bearing)
    width = math.sqrt(max(
        a.distance ** 2 + b.distance ** 2
        - 2 * a.distance * b.distance * math.cos(d_ang), 0.0))
    if width < body_m:
        return None
    return {"angle": bearing, "distance": reach + 0.5, "width": width,
            "between": (a.phrase, b.phrase)}


def surroundings_sentence(sightings: list[Sighting]) -> str:
    """One line for the state block's 我周围是什么 — names, sides and distances,
    never pixels or coordinates."""
    if not sightings:
        return ""
    best: dict[str, Sighting] = {}
    for s in sightings:
        cur = best.get(s.phrase)
        if cur is None or s.pixels > cur.pixels:
            best[s.phrase] = s
    parts = [s.describe() for s in sorted(best.values(), key=lambda s: s.distance)]
    return "You can see " + "; ".join(parts) + "."
