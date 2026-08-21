"""§16.5 — the ONE InstructionResolver.

Route/termination parsing used to live twice — eharness_agent's LLM split
and the bridge's private copy — with different fallbacks, different SAM
phrase extraction and a greedy ``re.search(r"{.*}")`` for "structured"
output. Three executors, three readings of the same sentence. This module
is the only parser now: the driver resolves ONCE per episode, writes a
versioned record into the episode live dir, and every executor (mini
in-process, SDK/Codex via the bridge) reads that same record.

§16.4 — the fallback is CONSERVATIVE. The old one put ``parts[:-1]`` on
the route and ``parts[-1]`` in terminate, so a single-sentence "Walk to
the kitchen." produced ZERO route segments and armed the termination
condition at frame 0 (``cursor >= len(segments)`` holds vacuously). Here:
at least one actionable segment ALWAYS; termination is a separate
condition that never swallows the last movement leg; when parsing fails
the whole instruction becomes the one segment and the terminate clause
only speaks of "after the instruction is completed".
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

RECORD_VERSION = 1
RECORD_NAME = "instruction_record.json"

# the safe terminate: it cannot fire before the one segment is verified,
# and it never renames the endpoint
FALLBACK_TERMINATE = ("only after the whole instruction has been completed, "
                      "stop at its described endpoint")

# The canonical parsing prompt — moved verbatim from eharness_agent (it
# encodes the user's 2026-08-07 ruling: the final clause is a STOP
# condition to verify, not "segment N"), with §16.4's one addition: every
# movement belongs to a segment, and at least one segment always exists.
SPLIT_SYSTEM = (
    "Parse a robot navigation instruction into a STRUCTURED route:\n"
    '- "segments": the route legs, in order — each an ACTIONABLE stretch '
    "(a landmark plus the walk that reaches or passes it). Merge trailing "
    "commentary into the leg it describes; never emit a non-action as its "
    "own item. EVERY movement in the instruction belongs to a segment — "
    "at least ONE segment always; do not move the final walking leg into "
    "the termination clause.\n"
    '- "terminate": the TERMINATION CONDITION — a full conditional STOP '
    'clause stating WHEN/WHERE to stop, e.g. "Stop when you reach the '
    'corner of the bar — the same bar you just walked between — and wait '
    'there". It must keep the condition wording (stop WHEN...) and name '
    "which landmark from the route it belongs to; never a bare noun "
    "phrase. Navigation lives or dies on stopping right, so this is "
    "separate from the legs.\n"
    '- "landmarks": the physical things AND the named rooms/areas on this '
    "route, each COPIED VERBATIM from the instruction's own words (user "
    "ruling 2026-08-16: 不能自由发挥). Never add adjectives, never swap "
    'synonyms, never infer a fancier name: the instruction\'s "chairs" '
    'stays "chairs" (NOT "lounge chairs"), its "pool" stays "pool" (NOT '
    '"swimming pool"), its "hall" stays "hall". The detector has its own '
    "synonym fallback at query time; your job is fidelity. Drop "
    'directions, corners and sides — "the corner of the bar" contributes '
    '"bar", not "corner". Drop bare adjectives. 2-6 entries.\n'
    'Answer with ONLY a JSON object: {"segments": ["...", ...], '
    '"terminate": "...", "landmarks": ["...", ...]}. 1-5 segments; keep the '
    "original wording in segments and terminate."
)


@dataclass
class InstructionRecord:
    """What one episode navigates on. Written once, read everywhere."""
    instruction: str
    segments: list[str]
    terminate: str
    landmarks: list[str] = field(default_factory=list)
    source: str = "fallback"          # "llm" | "fallback"
    fallback_reason: str = ""
    version: int = RECORD_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_transport(model_name: str, model_kwargs: dict | None,
                       system: str, user: str, *,
                       timeout_s: float = 300.0) -> str:
    """One short parsing call. Local ollama goes native (think:false —
    parsing needs no 30-60 s chain); anything else through litellm.
    ``timeout_s`` defaults to the resolver's own 300 s; the Refiner passes
    45 (§20.1 fail-fast) without touching resolver semantics."""
    prefix, _, bare_tag = (model_name or "").partition("/")
    if prefix in ("ollama", "ollama_chat") and bare_tag:
        import requests
        base = (model_kwargs or {}).get("api_base") or "http://127.0.0.1:11434"
        r = requests.post(f"{base.rstrip('/')}/api/chat", json={
            "model": bare_tag, "stream": False, "think": False,
            # temperature 0: parsing is extraction, not writing — sampled
            # runs dropped "kitchen area" from one call and kept it in the
            # next, so the "one record per episode" promise held only by
            # luck of the draw
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }, timeout=timeout_s)
        return (r.json().get("message") or {}).get("content") or ""
    import litellm
    resp = litellm.completion(
        model=model_name,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        timeout=timeout_s,
        **(model_kwargs or {}))
    return resp.choices[0].message.content or ""


def _parse_json_object(text: str) -> dict | None:
    """Strict-ish structured parsing — NOT a greedy regex. Whole-text JSON
    first; otherwise a proper raw_decode from the first '{', which stops
    at the object's real closing brace instead of the last '}' in the
    transcript."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


_FAITH_SKIP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at"}


def _faithful_phrase(phrase: str, instruction: str) -> str | None:
    """The instruction's OWN form of this landmark, or None.

    Enforcement of the verbatim ruling (user 2026-08-16: EP0 said
    "chairs" and the table said "lounge chairs" — 105 of 200 episodes'
    phrases were embellished). Salvage order: the longest contiguous
    sub-phrase found verbatim in the instruction (± plural tail), then a
    split-compound ("bedroom" ↔ the instruction's "bed room"), then a
    shared word stem ("hallway" → the instruction's "hall"). Whatever
    matches is returned in the INSTRUCTION's spelling; no match = None."""
    low = " " + re.sub(r"[^a-z0-9' ]", " ", instruction.lower()) + " "
    words = re.findall(r"[a-z0-9']+", phrase.lower())
    if not words:
        return None
    best = None
    for i in range(len(words)):
        for j in range(len(words), i, -1):
            cand = " ".join(words[i:j])
            forms = [cand, cand + "s", cand + "es"]
            if cand.endswith("s"):
                forms.append(cand[:-1])
            for form in forms:
                if re.search(r"(?<![a-z0-9])" + re.escape(form)
                             + r"(?![a-z0-9])", low):
                    if best is None or len(form) > len(best):
                        best = form
    if best:
        return best
    iw = [w for w in re.findall(r"[a-z0-9']+", low)
          if w not in _FAITH_SKIP and len(w) >= 3]
    joined = "".join(words)
    # split compound: phrase "bedroom" vs instruction "bed room"
    for a, b in zip(iw, iw[1:]):
        if a + b == joined:
            return f"{a} {b}"
    # shared stem: phrase "hallway" vs instruction "hall" (whole word)
    for w in iw:
        for p in words:
            if p != w and (p.startswith(w) or w.startswith(p)) \
                    and min(len(p), len(w)) >= 3:
                return w
    return None


def _validate(data: dict, instruction: str) -> tuple[list[str], str, list[str]] | None:
    """§16.4's contract, enforced: ≥1 non-empty segment, a non-empty
    terminate, both strings; zero segments is a REJECTION, never an
    armed-at-frame-0 route."""
    segs = [str(p).strip() for p in (data.get("segments") or [])
            if str(p).strip()]
    term = str(data.get("terminate") or "").strip()
    from eharness.landmarks import is_generic_phrase
    # two user rulings (2026-08-16) enforced IN CODE, whatever the model
    # returned: no generic phrases ("room"-class), and VERBATIM fidelity —
    # every phrase is mapped back to the instruction's own words
    # (_faithful_phrase) or dropped; embellishments cannot survive.
    marks: list[str] = []
    for p in (data.get("landmarks") or []):
        p = str(p).strip().lower()
        if not p:
            continue
        p = _faithful_phrase(p, instruction) or ""
        if p and not is_generic_phrase(p) and p not in marks:
            marks.append(p)
    # ACCEPTED segments are never sliced: a guard at >6 with a [:5] return
    # silently dropped leg 6 of a 6-leg parse and armed the terminate one
    # leg early (review P2). Over-long parses take the LOSSLESS fallback.
    if not segs or not term or len(segs) > 5:
        return None
    return segs, term, marks[:6]


def _fallback(instruction: str, reason: str) -> InstructionRecord:
    return InstructionRecord(
        instruction=instruction,
        segments=[instruction.strip() or "reach the goal"],
        terminate=FALLBACK_TERMINATE,
        landmarks=[],
        source="fallback",
        fallback_reason=reason,
    )


def resolve(instruction: str, *, model_name: str | None = None,
            model_kwargs: dict | None = None,
            transport: Callable[[str, str], str] | None = None,
            ) -> InstructionRecord:
    """Parse once. Any failure — no model, transport error, invalid JSON,
    zero segments — degrades to the conservative fallback, never to an
    armed terminate."""
    if transport is None and not model_name:
        return _fallback(instruction, "no split model configured")
    try:
        text = (transport(SPLIT_SYSTEM, instruction) if transport is not None
                else _default_transport(model_name or "", model_kwargs,
                                        SPLIT_SYSTEM, instruction))
    except Exception as exc:  # noqa: BLE001
        return _fallback(instruction, f"transport: {str(exc)[:120]}")
    data = _parse_json_object(text)
    if data is None:
        return _fallback(instruction, "invalid JSON")
    parsed = _validate(data, instruction)
    if parsed is None:
        return _fallback(instruction, "rejected (zero segments or no terminate)")
    segs, term, marks = parsed
    return InstructionRecord(instruction=instruction, segments=segs,
                             terminate=term, landmarks=marks, source="llm")


def save_record(live_dir: Path, rec: InstructionRecord) -> None:
    live_dir.mkdir(parents=True, exist_ok=True)
    tmp = live_dir / (RECORD_NAME + ".tmp")
    tmp.write_text(json.dumps(rec.as_dict(), ensure_ascii=False, indent=1))
    import os
    os.replace(tmp, live_dir / RECORD_NAME)


def load_record(live_dir: Path | None) -> InstructionRecord | None:
    if live_dir is None:
        return None
    p = Path(live_dir) / RECORD_NAME
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return InstructionRecord(
            instruction=str(data.get("instruction", "")),
            segments=[str(s) for s in data.get("segments") or []],
            terminate=str(data.get("terminate", "")),
            landmarks=[str(s) for s in data.get("landmarks") or []],
            source=str(data.get("source", "llm")),
            fallback_reason=str(data.get("fallback_reason", "")),
            version=int(data.get("version", 0)),
        )
    except Exception:  # noqa: BLE001
        return None


def resolve_or_load(live_dir: Path | None, instruction: str, *,
                    model_name: str | None = None,
                    model_kwargs: dict | None = None,
                    transport: Callable[[str, str], str] | None = None,
                    ) -> InstructionRecord:
    """The record on disk wins — that is the whole point: whoever resolves
    first (the driver, normally) fixes the reading for every executor of
    the episode. A record for a DIFFERENT instruction (stale live dir) is
    ignored."""
    rec = load_record(live_dir)
    if rec is not None and rec.segments and rec.instruction == instruction:
        return rec
    rec = resolve(instruction, model_name=model_name,
                  model_kwargs=model_kwargs, transport=transport)
    if live_dir is not None:
        try:
            save_record(Path(live_dir), rec)
        except Exception:  # noqa: BLE001 — a read-only dir must not kill a run
            pass
    return rec
