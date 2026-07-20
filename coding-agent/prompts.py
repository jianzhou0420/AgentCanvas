"""Single source of truth for the std prompt surface (frozen 2026-07-09;
unchanged across std-v1 → std-v2 — the v2 bump touched only resolution and
the LLM-call cap).

The BARE / FULL drafts below are the 2026-07-09 finalized texts, moved here
verbatim from the legacy SDK driver (now frozen for provenance at
legacy/beta-coding-agent/run_episodes.py — the legacy drivers are not edited). Any std run built
through this module records the ledger-nav body md5 and refuses to run a nav
cell whose skill text drifted from the freeze.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# std freeze (07-09): frontmatter-stripped body hash of ledger-nav/SKILL.md
LEDGER_NAV_STD_MD5 = "f7c74272"

SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (RGB image plus \
a clearance readout: meters to the nearest obstacle in the left/center/right \
thirds of the view; 10.0 = open).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.
- look_around(): one call returning four labeled views (ahead / right / \
behind / left); rotates 360 degrees and restores your heading (costs 24 \
turn steps).

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {budget} movement actions; each step() result reports \
roughly how many remain.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

BARE_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

WP_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look around from where you stand. Returns a panoramic image \
(four views labeled Left / Front / Right / Back) with numbered green circles \
marking the waypoints you can move to, plus a JSON listing each waypoint's \
direction and distance in meters.
- goto(waypoint): walk to one numbered waypoint from the LATEST observe(). \
Moving invalidates the old numbers — observe() again after arriving.
- stop(): permanently END the episode, declaring you have reached the goal.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and moving: observe(), then move, then observe() again.
- Before every goto() or stop(), reason out loud in one or two sentences: \
name the part of the instruction you are currently executing, then say which \
numbered waypoint best matches it and why (e.g. "the instruction says turn \
left at the kitchen; waypoint 2 heads left into what looks like a kitchen, so \
I take it"). Do this thinking as visible text, then call the tool.
- You may make at most {wp_max_moves} waypoint moves; each observe() and \
goto() result reports how many remain. When they run out the episode ends, so \
do not wander.
- You succeed only if you call stop() while within 3 meters of the \
instruction's endpoint. stop() is permanent — call it only when you believe \
you are at the goal.
- Work autonomously until you stop; nobody can answer questions.
"""

FIRST_PROMPT = "Begin navigating. Call observe() first to see where you are."


def load_skill(name: str) -> tuple[str, str]:
    """Return (frontmatter-stripped body, md5[:8]) of a skill under
    coding-agent/skills/ — the exact text the drivers feed the model."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    body = text.strip()
    return body, hashlib.md5(body.encode()).hexdigest()[:8]


def build_briefing(
    instruction: str, step_budget: int, *, bare: bool, skill: str | None,
    wp: bool = False, wp_max_moves: int = 30,
) -> tuple[str, str | None]:
    """Render the full task briefing (the SDK cell's system prompt; delivered
    as the first user message on harnesses whose builtin prompt is fixed).
    Returns (briefing, skill_md5)."""
    if wp:  # waypoint action space (wp_bridge.py) — its own tool surface
        briefing = WP_SYSTEM_PROMPT.format(
            instruction=instruction, wp_max_moves=wp_max_moves
        )
        wp_skill_md5: str | None = None
        # wp skills teach waypoint-selection discipline (anti-circling ledger,
        # instruction sub-goal ticking), NOT step() batching — so they append
        # regardless of the bare flag (wp is always its own surface).
        if skill:
            body, wp_skill_md5 = load_skill(skill)
            briefing += (
                "\n\nYou have been equipped with the following navigation skill."
                " Follow its discipline exactly throughout the episode.\n\n"
                f'<skill name="{skill}">\n{body}\n</skill>\n'
            )
        return briefing, wp_skill_md5
    base = BARE_SYSTEM_PROMPT if bare else SYSTEM_PROMPT
    briefing = base.format(instruction=instruction, budget=step_budget)
    skill_md5: str | None = None
    if skill and not bare:
        body, skill_md5 = load_skill(skill)
        briefing += (
            "\n\nYou have been equipped with the following navigation skill."
            " Follow its discipline exactly throughout the episode.\n\n"
            f'<skill name="{skill}">\n{body}\n</skill>\n'
        )
    return briefing, skill_md5


def assert_std_skill_freeze(skill: str) -> str:
    """Std conformance: the nav cell's skill body must match the frozen hash."""
    _, md5 = load_skill(skill)
    if skill == "ledger-nav" and md5 != LEDGER_NAV_STD_MD5:
        raise RuntimeError(
            f"ledger-nav body md5 {md5} != frozen {LEDGER_NAV_STD_MD5} — "
            "the skill drifted; this is std-v2 territory, refusing to run"
        )
    return md5
