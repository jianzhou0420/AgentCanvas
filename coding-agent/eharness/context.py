"""context.py — the CYCLE-based multimodal compiler (multiturn revision §3).

The unit of retention is a complete cycle — the assistant's original
reasoning/decision/tool call together with its tool results and the images
those results carried — kept IN PLACE, in time order. History selection picks
whole cycles; a cycle that is not selected leaves the payload entirely (no
reasoning-stripped stumps, no orphan tool results); the images of a selected
cycle stay inside their own tool result, never re-attached to a synthetic
"journey" user message.

Visual slots per request (§3.3):
    Historical RGB   0–12   uniform over the visual cycles, first+last pinned
    CURRENT RGB      1      the newest cycle's action-outcome view
    latest map       1      typed slot; re-attached to the newest cycle if
                            an error turn came back imageless
    hard cap         14     ENFORCED — overflow degrades deterministically

Storage stays light (§3.4): tool results hold {"type": "image_ref"} handles
written by the wrapper at ingest time; only the images that survive assembly
are materialised back into data-URI parts, here, at compile time.
"""
from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MARKER = re.compile(r"^\[frame#(\d+)( auto)?\]$")
_MAP_LABEL = "IMAGE 2 — accumulated top-down memory"
_MAP_STUB = ("[MAP UPDATED — this older accumulated map was superseded; "
             "the latest one rides with the newest observation]")
_EXTRA_FRAME_STUB = "[additional view from this turn elided — recall({}) to view]"


@dataclass
class ContextPolicy:
    history_cycles: int = 12
    images_max: int = 14


def _is_image(p: Any) -> bool:
    return isinstance(p, dict) and p.get("type") in ("image_url", "image_ref")


def _png_part(png: bytes) -> dict:
    b64 = base64.b64encode(png).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def _materialise(part: dict, live_dir: Path | None) -> dict:
    """image_ref → image_url by reading the stable file; a missing file
    becomes an honest stub, never a silent hole."""
    if part.get("type") != "image_ref":
        return part
    ref = str(part.get("ref", ""))
    if live_dir is not None and ref:
        p = live_dir / ref
        if p.exists():
            return _png_part(p.read_bytes())
    return _text(f"[image {ref or '?'} unavailable]")


def _classify(content: list) -> list[tuple[int, str, int | None]]:
    """(part index, kind, frame idx) per image part. Map identity first —
    by the typed flag the wrapper stamps, or by the explaining label."""
    out: list[tuple[int, str, int | None]] = []
    for i, p in enumerate(content):
        if not _is_image(p):
            continue
        if p.get("map") or (i > 0 and isinstance(content[i - 1], dict)
                            and str(content[i - 1].get("text", ""))
                            .startswith(_MAP_LABEL)):
            out.append((i, "map", None))
            continue
        if i + 1 < len(content) and isinstance(content[i + 1], dict):
            mm = _MARKER.match(str(content[i + 1].get("text", "")))
            if mm:
                out.append((i, "frame", int(mm.group(1))))
                continue
        out.append((i, "other", None))
    return out


def _cycle_census(cyc: list[dict]) -> list[tuple[int, int, str, int | None]]:
    """Every image in a cycle: (msg idx, part idx, kind, frame id)."""
    out = []
    for mi, m in enumerate(cyc):
        c = m.get("content")
        if isinstance(c, list):
            for pi, kind, fid in _classify(c):
                out.append((mi, pi, kind, fid))
    return out


def _fix_pairs(msgs: list[dict], start: int) -> None:
    """Provider-legal pairing within msgs[start:] (§14.15): an assistant
    tool_call whose result never arrived (FormatError/retry cut the cycle
    short) is dropped from the emitted copy, and a tool result whose call
    is not in the same region is re-rolled to a plain user message. Every
    provider rejects both shapes outright; the transcript on disk keeps the
    original — only the compiled payload is normalised."""
    result_ids = {m.get("tool_call_id") for m in msgs[start:]
                  if m.get("role") == "tool"}
    call_ids: set[Any] = set()
    for m in msgs[start:]:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                call_ids.add(tc.get("id"))
    for i in range(start, len(msgs)):
        m = msgs[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            kept = [tc for tc in m["tool_calls"]
                    if tc.get("id") in result_ids]
            if len(kept) != len(m["tool_calls"]):
                m2 = {**m, "tool_calls": kept}
                if not kept:
                    m2.pop("tool_calls", None)
                msgs[i] = m2
        elif m.get("role") == "tool" and m.get("tool_call_id") not in call_ids:
            # tagged so the mandatory-slot fallbacks can still recognise the
            # newest cycle's (re-rolled) result as their attach point; the
            # tag is stripped before the payload leaves assemble (§14.15)
            msgs[i] = {"role": "user", "content": m.get("content"),
                       "_reroled_tool": True}


def assemble(messages: list[dict], frames: Any,
             policy: ContextPolicy | None = None,
             live_dir: Path | None = None,
             ) -> tuple[list[dict], dict[str, Any]]:
    t0 = time.time()
    policy = policy or ContextPolicy()

    # head = system + the task message (+ pre-loop user content e.g. survey)
    head_end = 1
    while head_end < len(messages) and messages[head_end].get("role") in (
            "system", "user"):
        head_end += 1
        break
    head = list(messages[:head_end])
    body = messages[head_end:]

    cycles: list[list[dict]] = []
    cur: list[dict] | None = None
    for m in body:
        if m.get("role") == "assistant":
            if cur is not None:
                cycles.append(cur)
            cur = [m]
        elif cur is None:
            head.append(m)
        else:
            cur.append(m)
    if cur is not None:
        cycles.append(cur)

    # §3.3: the opening survey shows on the FIRST decision only; afterwards
    # its images leave the resident payload (words can recall them).
    if cycles:
        new_head = []
        for m in head:
            c = m.get("content")
            if isinstance(c, list) and any(_is_image(p) for p in c):
                m = {**m, "content": [
                    p if not _is_image(p)
                    else _text("[opening survey view shown on the first "
                               "decision — elided from later turns]")
                    for p in c]}
            new_head.append(m)
        head = new_head

    census = [_cycle_census(c) for c in cycles]
    newest_i = len(cycles) - 1 if cycles else -1
    visual_hist = [i for i in range(len(cycles) - 1)
                   if any(k == "frame" for _m, _p, k, _f in census[i])]

    # recall/other images the newest turn carries pay out of the history quota
    recall_kept = sum(1 for _m, _p, k, _f in census[newest_i]
                      if k == "other") if cycles else 0
    quota = max(0, policy.history_cycles - recall_kept)
    if len(visual_hist) <= quota:
        chosen = list(visual_hist)
    elif quota == 0:
        chosen = []
    elif quota == 1:
        chosen = [visual_hist[-1]]
    else:
        n = len(visual_hist)
        idxs = sorted({round(i * (n - 1) / (quota - 1)) for i in range(quota)})
        chosen = [visual_hist[i] for i in idxs]
    chosen_set = set(chosen)

    # the newest map part anywhere (the typed latest-map slot)
    latest_map: tuple[int, int, int] | None = None       # cycle, msg, part
    for ci in range(len(cycles) - 1, -1, -1):
        maps = [(mi, pi) for mi, pi, k, _f in census[ci] if k == "map"]
        if maps:
            latest_map = (ci, *maps[-1])
            break

    out: list[dict] = list(head)
    _fix_pairs(out, 0)      # a stray pre-loop tool result must not reach a provider
    manifest_frames: list[int] = []
    current_frame_id: int | None = None
    reasoning_chars = 0
    dropped: list[tuple[int, int]] = []   # (count, up to cycle index)
    pending_elided = 0
    history_image_slots: list[tuple[int, int]] = []   # (msg idx in out, part idx)
    other_image_slots: list[tuple[int, int]] = []     # newest-cycle recalls
    current_slot: tuple[int, int] | None = None       # the CURRENT frame
    map_slot: tuple[int, int] | None = None           # the one live map
    newest_cycle_start = 0                            # start of newest cycle in out

    for ci, cyc in enumerate(cycles):
        keep = ci == newest_i or ci in chosen_set
        if not keep:
            pending_elided += 1
            continue
        if pending_elided:
            out.append({"role": "user", "content": [
                _text(f"[· {pending_elided} earlier turn"
                      f"{'s' if pending_elided > 1 else ''} not shown — their "
                      "conclusions live in the harness state ·]")]})
            dropped.append((pending_elided, ci))
            pending_elided = 0
        cen = census[ci]
        frame_parts = [(mi, pi, f) for mi, pi, k, f in cen if k == "frame"]
        primary = ((frame_parts[-1] if ci == newest_i else frame_parts[0])
                   if frame_parts else None)
        cycle_start = len(out)
        for mi, m in enumerate(cyc):
            c = m.get("content")
            if m.get("role") == "assistant":
                r = m.get("reasoning_content")
                if isinstance(r, str):
                    reasoning_chars += len(r)
                out.append(m)
                continue
            if not isinstance(c, list):
                out.append(m)
                continue
            info = {(pi): (k, f) for _mi, pi, k, f in cen if _mi == mi}
            parts: list[Any] = []
            for pi, p in enumerate(c):
                kf = info.get(pi)
                if kf is None:
                    parts.append(p)
                    continue
                kind, fid = kf
                if kind == "map":
                    # the ONE live map rides the newest cycle only; an older
                    # retained cycle's map is always a stub, or the fallback
                    # that re-attaches it to the newest turn would deliver
                    # the same map twice (audit P1)
                    if latest_map == (ci, mi, pi) and ci == newest_i:
                        parts.append(_materialise(p, live_dir))
                        map_slot = (len(out), len(parts) - 1)
                    else:
                        parts.append(_text(_MAP_STUB))
                elif kind == "frame":
                    if primary == (mi, pi, fid):
                        parts.append(_materialise(p, live_dir))
                        if ci == newest_i:
                            current_frame_id = fid
                            current_slot = (len(out), len(parts) - 1)
                        else:
                            manifest_frames.append(fid)
                            history_image_slots.append((len(out), len(parts) - 1))
                    else:
                        parts.append(_text(
                            frames.stub_text(fid) if frames is not None
                            and fid is not None else
                            _EXTRA_FRAME_STUB.format(fid)))
                else:   # recall/other
                    if ci == newest_i:
                        parts.append(_materialise(p, live_dir))
                        other_image_slots.append((len(out), len(parts) - 1))
                    else:
                        parts.append(_text("[recalled frame elided — recall "
                                           "it again if needed]"))
            out.append({**m, "content": parts})
        _fix_pairs(out, cycle_start)
        newest_cycle_start = cycle_start   # last emitted cycle IS the newest
    if pending_elided:
        out.append({"role": "user", "content": [
            _text(f"[· {pending_elided} earlier turns not shown ·]")]})
        dropped.append((pending_elided, len(cycles)))

    # ── mandatory-slot fallbacks ──
    # CURRENT: an imageless newest turn (an error result, say) still delivers
    # the newest camera frame from its stable file. The attach point is
    # constrained to the NEWEST cycle — when _fix_pairs re-rolled its orphan
    # tool result to user, the old role=='tool' scan walked past it into a
    # HISTORY turn and the model read its current camera view as the outcome
    # of an old action (review P2). A re-rolled result carries the tag.
    last_tool_at = next(
        (i for i in range(len(out) - 1, newest_cycle_start - 1, -1)
         if out[i].get("role") == "tool" or out[i].get("_reroled_tool")),
        None)
    if (current_frame_id is None and cycles and frames is not None
            and last_tool_at is not None):
        fr = frames.last_frame()
        png = frames.png_bytes(fr.idx) if fr is not None else None
        if png is not None:
            c = out[last_tool_at].get("content")
            c = [_text(str(c))] if isinstance(c, str) else list(c or [])
            c += [_text("[your CURRENT view — this turn's result carried "
                        "no image:]"), _png_part(png),
                  _text(f"[frame#{fr.idx}]")]
            out[last_tool_at] = {**out[last_tool_at], "content": c}
            current_frame_id = fr.idx
            current_slot = (last_tool_at, len(c) - 2)
    # latest map: if the newest cycle had none, the typed slot re-attaches the
    # newest known map file to the newest tool result.
    if latest_map is not None and cycles and latest_map[0] != newest_i \
            and last_tool_at is not None:
        ci, mi, pi = latest_map
        src = cycles[ci][mi].get("content", [])
        if isinstance(src, list) and pi < len(src):
            c = out[last_tool_at].get("content")
            c = [_text(str(c))] if isinstance(c, str) else list(c or [])
            c += [_text(_MAP_LABEL + " (latest known — this turn produced "
                        "none):"), _materialise(src[pi], live_dir)]
            out[last_tool_at] = {**out[last_tool_at], "content": c}
            map_slot = (last_tool_at, len(c) - 1)

    # ── the 14-image hard cap, ENFORCED ──
    def _count() -> int:
        return sum(1 for m in out if isinstance(m.get("content"), list)
                   for p in m["content"] if _is_image(p)
                   or (isinstance(p, dict) and p.get("type") == "image_url"))
    over = _count() - policy.images_max
    degraded: list[int] = []
    while over > 0 and history_image_slots:
        mi, pi = history_image_slots.pop(0)          # oldest history first
        c = list(out[mi]["content"])
        fid = manifest_frames[len(degraded)] if len(degraded) < \
            len(manifest_frames) else None
        c[pi] = _text(frames.stub_text(fid) if frames is not None
                      and fid is not None else "[frame elided — over budget]")
        out[mi] = {**out[mi], "content": c}
        degraded.append(fid if fid is not None else -1)
        over -= 1
    kept_frames = [f for f in manifest_frames if f not in degraded]

    # §14.15: the cap is a CONTRACT. When the history slots are exhausted and
    # the payload is still over (a recall flood on the newest turn can do it
    # with quota already at zero), degradation CONTINUES deterministically:
    # newest-turn recall images oldest-first, then any remaining image except
    # the CURRENT frame and the live map, and only then those two mandatory
    # slots (an absurd images_max < 2 is the only way to reach them).
    degraded_final = 0

    def _strip(slots: list[tuple[int, int]]) -> None:
        nonlocal over, degraded_final
        for mi, pi in slots:
            if over <= 0:
                return
            c = out[mi].get("content")
            if not isinstance(c, list) or pi >= len(c) or not _is_image(c[pi]):
                continue
            c = list(c)
            c[pi] = _text("[image elided — over the image budget]")
            out[mi] = {**out[mi], "content": c}
            degraded_final += 1
            over -= 1

    if over > 0:
        _strip(other_image_slots)
    if over > 0:
        protected = {s for s in (current_slot, map_slot) if s is not None}
        _strip([(mi, pi) for mi, m in enumerate(out)
                if isinstance(m.get("content"), list)
                for pi, p in enumerate(m["content"])
                if _is_image(p) and (mi, pi) not in protected])
    if over > 0:
        _strip([s for s in (map_slot, current_slot) if s is not None])

    n_images = _count()
    if n_images > policy.images_max:
        raise AssertionError(
            f"image cap violated after final degradation: {n_images} > "
            f"{policy.images_max} — the compiler must never return this")
    text_chars = 0
    image_bytes = 0
    for m in out:
        c = m.get("content")
        if isinstance(c, str):
            text_chars += len(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    image_bytes += len(str(
                        (p.get("image_url") or {}).get("url", "")))
                elif isinstance(p, dict):
                    text_chars += len(str(p.get("text", "")))

    manifest = {
        "retained_cycles": len(chosen) + (1 if cycles else 0),
        "history_cycle_ids": chosen,
        "history_frame_ids": kept_frames,
        "current_frame_id": current_frame_id,
        "map_id": (f"map@cycle{latest_map[0]}" if latest_map else None),
        "image_count": n_images,
        "images_degraded_for_cap": degraded,
        "images_degraded_final": degraded_final,
        "reasoning_chars_retained": reasoning_chars,
        "dropped_turns": sum(n for n, _ in dropped),
        "text_chars": text_chars,
        "image_bytes": image_bytes,
        "context_build_ms": round((time.time() - t0) * 1000, 1),
    }
    # internal bookkeeping must not reach a provider
    out = [{k: v for k, v in m.items() if k != "_reroled_tool"} for m in out]
    return out, manifest
