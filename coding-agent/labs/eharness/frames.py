"""Frame log, perceptual hashing, and the event-driven keyframe store.

Everything here is coordinate-free by design (the no-coordinates ruling):
a frame is addressed by its index, its step count, its hash, and the *events*
that happened on it — never by a pose.

- ``ahash`` — 8×8 grayscale average hash, hand-rolled on PIL (~15 lines, zero
  new deps). Used by the V0 stall guard and as the novelty signal for keyframe
  promotion. Hamming distance on 64 bits; empirically <6 ≈ "same view".
- ``FrameLog`` — one row per observed frame; persisted to frames.jsonl.
- Keyframe promotion is event-driven (O2 §5.2): decision point / sub-goal
  transition / verification moment / negative fact / visual novelty. NOT
  promoted: frames mid-advance toward an already-chosen target.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image as PILImage


def ahash(png: bytes) -> int:
    """8×8 average hash → 64-bit int."""
    img = PILImage.open(BytesIO(png)).convert("L").resize(
        (8, 8), PILImage.LANCZOS)
    px = list(img.getdata())
    avg = sum(px) / 64
    bits = 0
    for p in px:
        bits = (bits << 1) | (1 if p > avg else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class Frame:
    idx: int                 # monotone frame counter (the recall handle)
    step: int                # env steps taken when captured
    tool: str                # which tool produced it
    hash: int
    events: list[str] = field(default_factory=list)  # e.g. landmark:'stairs'
    keyframe: bool = False
    png_path: str = ""       # under live_dir; pixels live on disk, not here
    motion: str = ""         # motor record since the previous frame ("fwd×3 L×1")
    crisis: bool = False     # interior of a stuck→freed span: history drops it


class FrameLog:
    """Owns every observed frame's metadata + the keyframe subset.

    Pixels are NOT held in memory beyond the most recent few (for the judge's
    evidence and the stall guard) — the live_dir PNGs the toolset already
    writes are the storage layer; we only add addressing.
    """

    RECENT_PNGS = 6  # in-memory tail for judge evidence / forced-look checks

    def __init__(self, live_dir: Path | None) -> None:
        self.live_dir = live_dir
        self.frames: list[Frame] = []
        self.segments: list[dict] = []              # completed-segment reels
        self._recent: list[tuple[int, bytes]] = []  # (idx, png)
        # stage3 P0-3: every detector sighting becomes an indexed landmark
        # event — {phrase, frame, bearing_deg?, distance_m?, score?} — so
        # recall(kind='landmark') answers from the harness's own record
        # instead of hoping a keyframe happened to be tagged
        self.landmark_events: list[dict] = []

    # ── recording ──

    def record(self, png: bytes, *, step: int, tool: str,
               motion: str = "") -> Frame:
        fr = Frame(idx=len(self.frames), step=step, tool=tool,
                   hash=ahash(png), motion=motion)
        # §3.4: EVERY frame gets a unique stable file at record time — the
        # recall handle always resolves, the stored transcript can hold refs
        # instead of base64, and nobody ever again guesses pixels through an
        # obs_*_stepNNN glob that returns whichever same-step frame sorts
        # first. Keyframe promotion becomes a flag, not a copy.
        if self.live_dir is not None:
            try:
                (self.live_dir / "frames").mkdir(parents=True, exist_ok=True)
                fr.png_path = f"frames/frame_{fr.idx:06d}.png"
                (self.live_dir / fr.png_path).write_bytes(png)
            except OSError:
                fr.png_path = ""
        self.frames.append(fr)
        self._recent.append((fr.idx, png))
        del self._recent[:-self.RECENT_PNGS]
        self._persist(fr)
        return fr

    def _save_keyframe_png(self, fr: Frame) -> None:
        """Promotion writes its OWN copy (kf_%04d.png) — never trust the inner
        toolset's live-frame naming to line up with our counter. Keyframes are
        few; the duplicate pixels are the price of a handle that always
        resolves."""
        if self.live_dir is None or fr.png_path:
            return
        for idx, png in self._recent:
            if idx == fr.idx:
                self.live_dir.mkdir(parents=True, exist_ok=True)
                fr.png_path = f"kf_{fr.idx:04d}.png"
                (self.live_dir / fr.png_path).write_bytes(png)
                return

    def tag(self, idx: int, event: str, *, promote: bool = True) -> None:
        """Attach an event to a frame; events are what make keyframes."""
        if 0 <= idx < len(self.frames):
            fr = self.frames[idx]
            fr.events.append(event)
            if promote and not fr.keyframe:
                fr.keyframe = True
                self._save_keyframe_png(fr)
            self._persist(fr)

    def maybe_promote_novel(self, fr: Frame, *, min_dist: int = 24,
                            min_step_gap: int = 12) -> bool:
        """Visual novelty backstop for keyframe promotion. Two hard lessons
        from the first 10-ep run baked in (user observation 2026-08-05: three
        'novel views' that were all the same living room, pool on the right):
        (1) hash distance measures LAYOUT change, not PLACE change — walking
        two meters forward shifts enough pixels to clear a low bar, so the
        bar is high (24/64 bits); (2) a step-gap cooldown: however novel the
        pixels, a frame taken right after the last keyframe is the same
        stretch of walking, not a new place. Event promotions bypass both —
        a decision point is a keyframe no matter what it looks like."""
        kfs = [f for f in self.frames if f.keyframe and f is not fr]
        if not kfs:
            fr.keyframe = True
            fr.events.append("first-view")
        elif (min(hamming(fr.hash, k.hash) for k in kfs) >= min_dist
              and fr.step - max(k.step for k in kfs) >= min_step_gap):
            fr.keyframe = True
            fr.events.append("novel-view")
        else:
            return False
        self._save_keyframe_png(fr)
        self._persist(fr)
        return True

    def _persist(self, fr: Frame) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        with (self.live_dir / "keyframes.jsonl").open("a") as fh:
            fh.write(json.dumps({
                "idx": fr.idx, "step": fr.step, "tool": fr.tool,
                "hash": f"{fr.hash:016x}", "events": fr.events,
                "keyframe": fr.keyframe, "png": fr.png_path,
                "motion": fr.motion, "crisis": fr.crisis}) + "\n")

    # ── reads ──

    def recent_pngs(self, n: int = 2) -> list[bytes]:
        return [png for _idx, png in self._recent[-n:]]

    def last_frame(self) -> Frame | None:
        return self.frames[-1] if self.frames else None

    def png_bytes(self, idx: int) -> bytes | None:
        """Resolve a handle back to pixels — memory tail first, then disk
        (our keyframe copy, then a best-effort match against the inner
        toolset's own live frames by step number)."""
        for i, png in self._recent:
            if i == idx:
                return png
        if not (0 <= idx < len(self.frames)) or self.live_dir is None:
            return None
        fr = self.frames[idx]
        if fr.png_path:
            p = self.live_dir / fr.png_path
            if p.exists():
                return p.read_bytes()
        hits = sorted(self.live_dir.glob(f"obs_*_step{fr.step:03d}.png"))
        return hits[0].read_bytes() if hits else None

    def evidence_pngs(self, max_n: int = 8) -> list[bytes]:
        """Chronological pixel evidence of the walk, for milestone checks —
        did a sub-instruction actually HAPPEN, judged from the harness's own
        records, not the model's narrative. Evenly samples resolvable
        frames; crisis frames excluded; falls back to keyframes + tail when
        sampling resolves thin."""
        idxs = [f.idx for f in self.frames if not f.crisis]
        if not idxs:
            return []
        if len(idxs) > max_n:
            stride = (len(idxs) - 1) / (max_n - 1)
            idxs = sorted({idxs[round(i * stride)] for i in range(max_n)})
        out: list[bytes] = []
        for i in idxs:
            png = self.png_bytes(i)
            if png is not None:
                out.append(png)
        if len(out) < 3:
            for f in self.frames:
                if f.keyframe and not f.crisis:
                    png = self.png_bytes(f.idx)
                    if png is not None and png not in out:
                        out.append(png)
            for png in self.recent_pngs(2):
                if png not in out:
                    out.append(png)
        return out[:max_n]

    # ── stage3 P0-3: the landmark event index ──

    def note_landmark(self, phrase: str, *, frame_idx: int,
                      bearing_deg: float | None = None,
                      distance_m: float | None = None,
                      score: float | None = None) -> None:
        """One detector sighting → one indexed event, persisted to
        landmarks.jsonl. Frames all live on disk (record() writes every
        one), so an event on a non-keyframe still resolves to pixels."""
        if not str(phrase).strip():
            return
        ev: dict[str, Any] = {"phrase": str(phrase).strip(),
                              "frame": int(frame_idx)}
        if bearing_deg is not None:
            ev["bearing_deg"] = round(float(bearing_deg), 1)
        if distance_m is not None:
            ev["distance_m"] = round(float(distance_m), 2)
        if score is not None:
            ev["score"] = round(float(score), 3)
        self.landmark_events.append(ev)
        if self.live_dir is not None:
            try:
                self.live_dir.mkdir(parents=True, exist_ok=True)
                with (self.live_dir / "landmarks.jsonl").open("a") as fh:
                    fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except OSError:
                pass

    @staticmethod
    def _norm_words(s: str) -> set[str]:
        """Normalisation for landmark matching: lowercase, strip
        punctuation, drop trailing plural-s — 'Bars' matches 'bar
        counter'. No synonym table; evidence is returned, never invented."""
        words = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w
                for w in words}

    def landmark_hits(self, query: str, *, top_k: int = 3) -> list[dict]:
        """Newest-first landmark events matching the query (one per frame)."""
        qw = self._norm_words(query)
        if not qw:
            return []
        out: list[dict] = []
        seen: set[int] = set()
        for ev in reversed(self.landmark_events):
            if not (qw & self._norm_words(ev["phrase"])):
                continue
            if ev["frame"] in seen:
                continue
            seen.add(ev["frame"])
            out.append(ev)
            if len(out) >= top_k:
                break
        return out

    def landmark_phrases_seen(self) -> list[str]:
        """Distinct recorded phrases, first-seen order — the honest answer
        to 'what CAN I recall?'."""
        seen: list[str] = []
        for ev in self.landmark_events:
            if ev["phrase"] not in seen:
                seen.append(ev["phrase"])
        return seen

    def recall(self, query: str, *, k: int = 2) -> list[Frame]:
        """Keyframes whose events mention the query (landmark name, 'dead',
        a sub-instruction number as '#3', …). Newest first."""
        q = query.strip().lower()
        hits = [f for f in reversed(self.frames) if f.keyframe and any(
            q in ev.lower() for ev in f.events)]
        if not hits and q.lstrip("#").isdigit():
            n = int(q.lstrip("#"))
            hits = [f for f in reversed(self.frames) if f.keyframe and any(
                ev.startswith(f"subgoal:{n}") for ev in f.events)]
        return hits[:k]

    def keyframes(self) -> list[Frame]:
        return [f for f in self.frames if f.keyframe]

    def check_revisit(self, fr: Frame, *, max_dist: int = 6,
                      min_gap: int = 8) -> "Frame | None":
        """Loop closure, the human way: does the current view closely match a
        keyframe from a while ago? (Recent frames excluded — matching what
        you saw 3 frames ago is just walking, not returning.) Advisory: the
        caller tells the model; nothing is blocked."""
        for k in self.frames:
            if (k.keyframe and k.idx <= fr.idx - min_gap
                    and hamming(fr.hash, k.hash) <= max_dist):
                return k
        return None

    def archive_segment(self, start_idx: int, end_idx: int, *,
                        label: str, route: str) -> str | None:
        """Pack a completed sub-instruction's frames into one filmstrip PNG —
        the episode's long-term memory of "我曾经走过这么一段". The live
        window can drop these frames; the state's done-via line says you
        walked it; the strip is there if you need to see HOW. Returns the
        strip filename (or None if no pixels were recoverable)."""
        if self.live_dir is None or end_idx < start_idx:
            return None
        idxs = list(range(start_idx, end_idx + 1))
        if len(idxs) > 8:   # sample evenly, keep first + last
            stepn = (len(idxs) - 1) / 7
            idxs = [idxs[round(i * stepn)] for i in range(8)]
        tiles = []
        for i in idxs:
            png = self.png_bytes(i)
            if png is None:
                continue
            img = PILImage.open(BytesIO(png)).convert("RGB")
            img = img.resize((max(1, int(img.width * 96 / img.height)), 96))
            tiles.append((i, img))
        if not tiles:
            return None
        strip = PILImage.new("RGB", (sum(t.width for _, t in tiles)
                                     + 2 * (len(tiles) - 1), 96), (24, 24, 24))
        x = 0
        for _i, t in tiles:
            strip.paste(t, (x, 0))
            x += t.width + 2
        seg_n = sum(1 for _ in (self.live_dir / "segments.jsonl").open()) + 1             if (self.live_dir / "segments.jsonl").exists() else 1
        name = f"seg_{seg_n:02d}.png"
        self.live_dir.mkdir(parents=True, exist_ok=True)
        buf = BytesIO()
        strip.save(buf, format="PNG")
        (self.live_dir / name).write_bytes(buf.getvalue())
        rec = {"seg": seg_n, "label": label[:160], "route": route[:200],
               "frames": [start_idx, end_idx],
               "motion": self.motion_trace(start_idx - 1 if start_idx else 0,
                                           end_idx),
               "png": name}
        with (self.live_dir / "segments.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.segments.append(rec)
        return name

    def motion_trace(self, a_idx: int, b_idx: int, *, max_items: int = 8) -> str:
        """The motor record between two frames — what a body knows about the
        route it took, no coordinates: egocentric commands it itself issued.
        Handing this to the model lets IT do the path integration ("I turned
        a big loop and came back") instead of the harness pretending to."""
        moves = [f.motion for f in self.frames[a_idx + 1:b_idx + 1] if f.motion]
        if len(moves) > max_items:
            moves = moves[:max_items - 1] + [f"…(+{len(moves) - max_items + 1} more)"]
        return " · ".join(moves)

    def span_digest(self, *, max_spans: int = 6) -> list[str]:
        """Deterministic one-line summaries of what happened BETWEEN
        consecutive keyframes — built from the event tags on the in-between
        frames (state writes, guard trips), zero model calls. This is the
        "两个关键帧之间做总结" piece: the pixels in a span are gone, but the
        span's events survive as one line."""
        kfs = [f for f in self.frames if f.keyframe]
        lines: list[str] = []
        for a, b in zip(kfs, kfs[1:]):
            evs: list[str] = []
            for f in self.frames[a.idx + 1:b.idx]:
                evs.extend(f.events)
            moved = self.motion_trace(a.idx, b.idx, max_items=5)
            if not evs and not moved:
                continue
            bits = []
            if evs:
                bits.append(" · ".join(evs[:5])
                            + (f" (+{len(evs) - 5} more)" if len(evs) > 5 else ""))
            if moved:
                bits.append(f"moved: {moved}")
            lines.append(f"span frame#{a.idx}→#{b.idx} "
                         f"(steps {a.step}→{b.step}): " + " | ".join(bits))
        return lines[-max_spans:]

    def stub_text(self, idx: int) -> str:
        """L0 elision stub — an index entry, not a hole (O2 §5.1)."""
        if not (0 <= idx < len(self.frames)):
            return "[earlier camera frame elided]"
        fr = self.frames[idx]
        ev = " · ".join(fr.events[-2:]) if fr.events else "no event"
        return (f"[frame#{fr.idx} · step {fr.step} · {ev} · "
                f"recall({fr.idx}) to view]")

    # ── the §12 context contract: uniform history + full action spans ──

    def uniform_history(self, k: int = 8, *, before: int | None = None) -> list[Frame]:
        """Deterministic even sample of the history — §12.3, exactly.

        `before` is the CURRENT frame's idx (excluded — CURRENT has its own
        mandatory slot); history is everything earlier. With more than k
        frames the sample always contains the FIRST and the LAST history
        frame and is near-uniform in between: round(linspace), dedup, order
        kept. No model, no hashes, no scoring — the same episode always
        selects the same frames."""
        hist = self.frames if before is None else self.frames[:max(0, before)]
        if len(hist) <= k:
            return list(hist)
        n = len(hist)
        idxs = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
        return [hist[i] for i in idxs]

    def action_span(self, a_idx: int, b_idx: int) -> str:
        """EVERY action between two frames, in order — §12.4.

        Unlike motion_trace this never omits: the per-frame motion strings
        are already run-length ("fwd×3 L×1"), and all of them are joined.
        Events that must be explicit (goto, guard/stall, veer, STOP attempts,
        segment and landmark changes, crisis) ride along from the frame tags.
        Deterministic — built from the ledger, never from a model."""
        a_idx = max(-1, a_idx)
        frs = self.frames[a_idx + 1:b_idx + 1]
        if not frs:
            return ""
        a_step = self.frames[a_idx].step if 0 <= a_idx < len(self.frames) else 0
        b_step = frs[-1].step
        moves = [f.motion for f in frs if f.motion]
        evs: list[str] = []
        for f in frs:
            evs.extend(f.events)
        head = (f"TRANSITION frame#{a_idx}→#{b_idx} · env steps "
                f"{a_step}–{b_step}" if a_idx >= 0 else
                f"UP TO frame#{b_idx} · env steps 0–{b_step}")
        bits = [head]
        if moves:
            bits.append("actions: " + " → ".join(moves))
        if evs:
            bits.append("events: " + " · ".join(evs))
        return "\n  ".join(bits)
