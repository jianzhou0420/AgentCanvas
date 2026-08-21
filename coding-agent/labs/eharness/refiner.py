"""§17 — the conservative instruction Refiner (text-only preprocessor).

NOT a disambiguation decider, NOT visual grounding, NOT a clarification
dialogue. Before navigation starts, the run's OWN nav model reads the raw
instruction once, strips filler that carries no spatial content, smooths
disfluency, and completes a reference only when the sentence's internal
logic forces the answer ("stop at the corner of the bar" after "walk
between the bar and chairs" → the far corner). Everything else is kept
verbatim — no warnings, no blocking, no perceptual clarification goals.

Program-level checks are deliberately dumb (§17.5): can the reply be read,
is `instruction` a non-empty string. Nothing here second-guesses the
model's rewrite with keyword rules — trust it or fall back to the
original, whole.

The canonical prompt lives with the EmbodiedHarness implementation at
``eharness/skills/nav-instruction-refiner/SKILL.md``.  There is deliberately
no vendored duplicate: a missing prompt is recorded and falls back to the
original instruction instead of silently running stale semantics.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from eharness.resolver import _parse_json_object

SKILL_PATH = (
    Path(__file__).resolve().parent
    / "skills"
    / "nav-instruction-refiner"
    / "SKILL.md"
)


@dataclass
class RefinementResult:
    instruction: str          # what navigation runs on (refined, or original)
    clarified: bool = False
    note: str = ""            # audit only — never enters a navigation prompt
    fallback_reason: str = "" # empty = refinement succeeded
    original: str = ""
    duration_ms: int = 0       # wall-clock model-call latency; not nav cost

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_skill_text() -> str:
    text = SKILL_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty refiner skill: {SKILL_PATH}")
    return text


def build_prompt(skill_text: str, instruction: str) -> tuple[str, str]:
    """(system, user). The output-contract pin rides the system prompt even
    if a stale skill text is passed in — the parser depends on it."""
    system = (
        skill_text
        + "\n\nOUTPUT (mandatory): exactly ONE JSON object — "
          '{"instruction": "<cleaned text>", "clarified": <bool>, '
          '"note": "<optional, audit only>"} — no fences, no scratchpad, '
          "no other text. Keep the input language."
    )
    return system, instruction


def parse(text: str | None, original: str) -> RefinementResult:
    """§17.2/§17.5: read the fixed contract; ANY failure — empty reply,
    invalid JSON, empty instruction — is refinement_failed and falls back
    to the original text. No semantic validation happens here, ever."""
    if not text or not str(text).strip():
        return RefinementResult(instruction=original, original=original,
                                fallback_reason="empty output")
    data = _parse_json_object(str(text))
    if data is None:
        return RefinementResult(instruction=original, original=original,
                                fallback_reason="invalid JSON")
    refined = data.get("instruction")
    if not isinstance(refined, str) or not refined.strip():
        return RefinementResult(instruction=original, original=original,
                                fallback_reason="empty instruction field")
    return RefinementResult(
        instruction=refined.strip(),
        clarified=bool(data.get("clarified", False)),
        note=str(data.get("note") or ""),
        original=original,
    )


def refine_episode(instruction: str, *, live_dir: Path | None,
                   call: Callable[[str, str], "str | None"] | None,
                   model: str = "") -> RefinementResult:
    """The one refine of an episode (§17.3): build the prompt, make ONE
    call through the adapter-supplied `call(system, user)`, parse or fall
    back, write the audit record. `call` is None when the adapter has no
    refine capability — that is a recorded fallback, not an error."""
    raw: str | None = None
    result: RefinementResult | None = None
    try:
        system, user = build_prompt(load_skill_text(), instruction)
    except Exception as exc:  # noqa: BLE001
        result = RefinementResult(
            instruction=instruction,
            original=instruction,
            fallback_reason=f"skill: {str(exc)[:120]}",
        )

    if result is None and call is None:
        result = RefinementResult(instruction=instruction,
                                  original=instruction,
                                  fallback_reason="adapter has no refiner")
    elif result is None:
        started = time.perf_counter()
        try:
            raw = call(system, user)
            result = parse(raw, instruction)
            if raw is None and not result.fallback_reason:
                result.fallback_reason = "call returned None"
        except Exception as exc:  # noqa: BLE001 — a preprocessor must never
            raw = None            # cost an episode
            result = RefinementResult(instruction=instruction,
                                      original=instruction,
                                      fallback_reason=f"call: {str(exc)[:120]}")
        finally:
            result.duration_ms = round((time.perf_counter() - started) * 1000)
    if live_dir is not None:
        try:
            live_dir.mkdir(parents=True, exist_ok=True)
            (live_dir / "instruction_refinement.json").write_text(json.dumps(
                {**result.as_dict(), "model": model},
                ensure_ascii=False, indent=1))
        except Exception:  # noqa: BLE001
            pass
    return result
