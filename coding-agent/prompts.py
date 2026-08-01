"""Single source of truth for the std-v1 prompt surface.

The BARE / FULL drafts below are the 2026-07-09 finalized texts, moved here
verbatim from beta-coding-agent/run_episodes.py (which keeps its own frozen
copy for provenance — the legacy drivers are not edited). Any std run built
through this module records the ledger-nav body md5 and refuses to run a nav
cell whose skill text drifted from the freeze.

Observe/step coupling is toggled by ``auto_observe`` (build_briefing arg):
- separate (default, R2R-CE): classic alternation — observe() after every
  move; step()/goto() return text only. Matches the frozen R2R baselines.
- auto (RxR-CE default): step()/goto() carry the resulting view, so observe()
  is a first-look only — halves the tool-call/API round-trips per move, which
  RxR's long instructions otherwise inflate. The bridge (HABITAT_AUTO_OBSERVE)
  and this prompt must agree, so both are driven off the same flag.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "beta-coding-agent" / "skills"

# std-v1 freeze: frontmatter-stripped body hash of ledger-nav/SKILL.md
LEDGER_NAV_STD_MD5 = "f7c74272"

# ── observe/step guidance fragments, selected by auto_observe ──
# Inserted into the {obs_note}/{step_note}/{goto_line}/{loop_rule} slots below.
_OBS_NOTE_AUTO = " You only need this for your FIRST look."
_OBS_NOTE_SEP = ""
_STEP_NOTE_AUTO = (
    " Its result INCLUDES the resulting camera view, so you never need a "
    "separate observe() after moving."
)
_STEP_NOTE_SEP = ""
_STEP_LOOP_AUTO = (
    "- Call observe() once at the start. After that, every step() returns the "
    "new camera view automatically — do NOT call observe() again; just read the "
    "view step() returns, decide where to go next, and step() again. (Turning "
    "in place, e.g. step([2,2,2,2]), is how you look around.)"
)
_STEP_LOOP_SEP = (
    "- Alternate observing and stepping: look, decide where the instruction "
    "wants you to go next, move, look again."
)
_GOTO_LINE_AUTO = (
    "goto(waypoint): walk to one numbered waypoint from the LATEST result. Its "
    "result AUTOMATICALLY returns the new panorama and freshly numbered "
    "waypoints for your new position, so you never need a separate observe() "
    "after moving."
)
_GOTO_LINE_SEP = (
    "goto(waypoint): walk to one numbered waypoint from the LATEST observe(). "
    "Moving invalidates the old numbers — observe() again after arriving."
)
_WP_LOOP_AUTO = (
    "- Call observe() once at the start. After that, every goto() returns the "
    "new panorama and waypoints automatically — do NOT call observe() again; "
    "just reason and goto() the next waypoint from the numbers in the latest "
    "result."
)
_WP_LOOP_SEP = "- Alternate observing and moving: observe(), then move, then observe() again."

SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (RGB image plus \
a clearance readout: meters to the nearest obstacle in the left/center/right \
thirds of the view; 10.0 = open).{obs_note}
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.{step_note}
- look_around(): one call returning four labeled views (ahead / right / \
behind / left); rotates 360 degrees and restores your heading (costs 24 \
turn steps).

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
{loop_rule}
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
image).{obs_note}
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.{step_note}

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
{loop_rule}
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
direction and distance in meters.{obs_note}
- {goto_line}
- stop(): permanently END the episode, declaring you have reached the goal.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
{loop_rule}
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

HYBRID_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You have TWO navigation interfaces and, at each move, \
you commit to ONE of them:

WAYPOINT interface (cover distance):
- observe_waypoints(): scan a panorama (four views labeled Left / Front / \
Right / Back — this ALSO shows what is ahead of you) with numbered green \
circles marking the waypoints you can jump to, plus a JSON of each waypoint's \
direction, angle and distance in meters.
- goto(waypoint): walk in one call to a numbered waypoint from your LATEST \
observe_waypoints(). One goto travels several meters toward a landmark you can see.

PRIMITIVE interface (precise control):
- observe(): look through the forward-facing camera — a clean egocentric RGB \
view for fine positioning.
- step(actions): execute low-level actions in order. 1 = forward 0.25 m, \
2 = turn left 15 degrees, 3 = turn right 15 degrees, 0 = STOP.

- stop(): permanently END the episode, declaring you have reached the goal. \
This is a SHARED action — always available, from either interface and at any \
point (you never have to look first to stop). In primitive control you may \
also stop with step([0]).

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

How to operate:
- Each move is a strict cycle: LOOK once, then MOVE. Your look chooses your \
interface, and the tools ENFORCE the match: after observe() the only move is \
step(); after observe_waypoints() the only move is goto(). (stop() is the one \
exception — you can end the episode at any time, from either interface, without \
looking first.) You cannot look twice in a row (you must move between looks), \
and you cannot check both the forward camera and the waypoint panorama at the \
same spot. So DECIDE your interface first, then take the single look that \
matches it, then move.
- Choose the interface deliberately — do not default to waypoints. Use the \
WAYPOINT interface to travel: heading toward a visible landmark, down a \
corridor, or across an open room. Switch to the PRIMITIVE interface when you \
need precision: the final approach to the goal, exact placement ("stop at / by \
/ in front of X"), tight turns or doorways, small heading corrections, or when \
no waypoint points where the instruction actually sends you. A good run \
typically uses waypoints to travel and primitives to arrive.
- Switch as often as you like, in EITHER direction and at ANY point — travel by \
waypoint, drop to primitive for a tricky corner or a precise stop, then pick \
waypoints back up to keep moving. Switching is expected and free; it is not a \
last resort, and you are not locked into whichever interface you started with.
- A move invalidates what you last saw, so LOOK again next cycle. goto() needs \
a fresh observe_waypoints() immediately before it.
- You have a budget of {budget} movement actions (a goto spends several, a \
primitive step spends one); each move result reports roughly how many remain.
- You succeed only if you STOP (action 0 or stop()) while within 3 meters of \
the instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Work autonomously until you stop; nobody can answer questions.
"""

FIRST_PROMPT = "Begin navigating. Call observe() first to see where you are."

# Hybrid opens without forcing a forward observe() — the first thing the model
# does is CHOOSE an interface, so its first look is already a committed lens.
HYBRID_FIRST_PROMPT = (
    "Begin navigating. Decide which interface fits your first move, then look "
    "through only that lens — observe_waypoints() to travel by waypoint, or "
    "observe() for primitive control."
)


def load_skill(name: str) -> tuple[str, str]:
    """Return (frontmatter-stripped body, md5[:8]) of a skill under
    beta-coding-agent/skills/ — the exact text the drivers feed the model."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    body = text.strip()
    return body, hashlib.md5(body.encode()).hexdigest()[:8]


def build_briefing(
    instruction: str, step_budget: int, *, bare: bool, skill: str | None,
    wp: bool = False, wp_max_moves: int = 30, auto_observe: bool = False,
    hybrid: bool = False,
) -> tuple[str, str | None]:
    """Render the full task briefing (the SDK cell's system prompt; delivered
    as the first user message on harnesses whose builtin prompt is fixed).

    ``auto_observe`` MUST match the bridge's HABITAT_AUTO_OBSERVE: when True the
    prompt tells the model step()/goto() return the resulting view (observe()
    is first-look only); when False it prescribes the classic alternation.
    Returns (briefing, skill_md5)."""
    if hybrid:  # primitive actions AND waypoint tool in one surface (hybrid_bridge.py)
        # hybrid runs SEPARATE (non-auto-observe) on purpose: the model must
        # choose which lens to look through (forward camera vs waypoint
        # panorama), and that choice is the interface choice — see hybrid_bridge.
        briefing = HYBRID_SYSTEM_PROMPT.format(
            instruction=instruction, budget=step_budget,
        )
        # hybrid is its own bare-style surface (no injected workflow, no skill)
        return briefing, None
    if wp:  # waypoint action space (wp_bridge.py) — its own tool surface
        briefing = WP_SYSTEM_PROMPT.format(
            instruction=instruction, wp_max_moves=wp_max_moves,
            obs_note=(_OBS_NOTE_AUTO if auto_observe else _OBS_NOTE_SEP),
            goto_line=(_GOTO_LINE_AUTO if auto_observe else _GOTO_LINE_SEP),
            loop_rule=(_WP_LOOP_AUTO if auto_observe else _WP_LOOP_SEP),
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
    briefing = base.format(
        instruction=instruction, budget=step_budget,
        obs_note=(_OBS_NOTE_AUTO if auto_observe else _OBS_NOTE_SEP),
        step_note=(_STEP_NOTE_AUTO if auto_observe else _STEP_NOTE_SEP),
        loop_rule=(_STEP_LOOP_AUTO if auto_observe else _STEP_LOOP_SEP),
    )
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
