"""§14.9 — the CapabilityManifest: the ONE place the tool surface is written.

Five files used to hand-write the same facts (bridge registration, SDK
allowed_tools, toolset schemas/descriptions, the system prompt's action-space
section, the first prompt) and they drifted into contradiction. Measured cost
of that drift: the std_sdkeh_opus-5_dwp EP0 record shows the session was
briefed with the BARE system prompt and told to "Call observe() first" while
holding the DWP toolset — the model was graded on a surface its own briefing
denied existed. From now on MCP registration, provider tool lists, prompt
text, monitor expectations and tests all derive from here; hand-writing any
of these facts elsewhere is a review-blocking offence.

Motion-cost facts stated here (and nowhere else):
  * a forward primitive is 0.25 m and costs one env step;
  * a turn primitive is 15° and costs one env step;
  * face()/goto() turn FOR REAL — one env step per 15° notch; only the
    sub-15° remainder is a free micro-alignment (the simulator cannot
    express it as a primitive);
  * looks are free: every action returns the fresh view, there is no look
    tool on the dwp surface.
"""

from __future__ import annotations

STEP_M = 0.25
TURN_NOTCH_DEG = 15

# ── the canonical registry: model-visible tools per surface ──────────────
# "dwp" is the geometry line (multiturn revision); "bare" is the organ-wrapped
# classic surface. These tuples ARE the truth — the MCP server registers
# exactly this set, the SDK allowlist is derived from it, and the monitor and
# tests assert against it.
_VISIBLE: dict[str, tuple[str, ...]] = {
    # face withdrawn from the MODEL surface (user ruling 2026-08-12 evening:
    # the surface is goto + step; step's own 2/3 turns cover direction
    # changes). _tool_face survives as an internal/human verb only.
    "dwp": ("step", "goto", "veer", "recall"),
    "bare": ("observe", "step", "veer", "look_around", "recall"),
}


def mcp_visible_tools(surface: str) -> tuple[str, ...]:
    """The tools the MCP server may expose to the model on this surface."""
    return _VISIBLE[surface]


def sdk_allowed_tools(surface: str) -> list[str]:
    """The Claude Agent SDK allowed_tools list — derived, never hand-written."""
    return [f"mcp__env__{t}" for t in mcp_visible_tools(surface)]


# what the 5173 monitor should expect to see used on the geometry line
MONITOR_TOOLS: tuple[str, ...] = _VISIBLE["dwp"]


# ── tool descriptions (single copies; toolset + bridge import these) ─────
OBSERVE_DESC_DWP = (
    "Look through the robot's forward-facing camera. Returns the current view "
    "with the places you can walk to marked as numbered circles, a short "
    "description of each, and a sentence about the space around you. Pure "
    "read — does not advance the simulator or consume step budget."
)  # INTERNAL on the dwp surface: the harness's own read, never registered

GOTO_DESC = (
    "Walk toward one numbered place from the LATEST view: the robot turns "
    "toward it (real turn steps, one per 15°) and walks the SAFE STRIDE of "
    "the way — the part of the line measured clear and wide enough for your "
    "body. The corridor is re-checked after the turn and at every 0.25 m; "
    "if it closes or something blocks it, the walk stops early and the "
    "result says why. A far place may take a few goto() calls; its distance "
    "counts down as you close in. Every action returns fresh numbers. For "
    "turning, fine alignment or the last metre use step()."
)
GOTO_SCHEMA = {
    "properties": {"place": {"title": "Place", "type": "integer"}},
    "required": ["place"],
    "title": "gotoArguments",
    "type": "object",
}

DWP_STEP_DESC = (
    "Turning, fine alignment and the final metre: a short list of "
    "primitives (1=forward 0.25 m, 2=turn left 15°, 3=turn right 15°, "
    "0=STOP). Turns batch freely — step([2,2,2]) turns 45° left, "
    "step([3,3,3,3,3,3]) faces right — and every movement is executed one "
    "primitive at a time with per-step sensing: the harness brakes early "
    "on a collision, a closing corridor or a perception failure, and "
    "tells you why. Every action returns the new view with fresh numbered "
    "places. At most 2 forward moves per call are carried out — to cover "
    "ground use goto(place)."
)

# the BARE surface's step: no goto/face exist there, so the description may
# not mention them (the bridge's old shared text taught "cover ground with
# goto instead" to a model whose server had no goto — review P1)
STEP_DESC_BARE = (
    "Execute movement actions in order. 0=STOP (ends the episode — a "
    "verifier checks your first STOP; a second STOP always executes), "
    "1=forward 0.25m, 2=turn left 15°, 3=turn right 15°. Use single steps "
    "for the final approach. Your robot HAS A BODY (~0.2m-radius cylinder) "
    "— it needs side clearance and cannot squeeze past furniture; if an "
    "obstacle sits in the CENTER of your view a few steps ahead, veer a "
    "notch around it BEFORE pushing forward. The outcome view is attached; "
    "a stall guard blocks moves that change nothing and suggests veer."
)

FACE_DESC = (
    "Turn on the spot to face a direction in ONE call: face('left') / "
    "face('right') / face('back'), or a signed bearing in degrees (left "
    "positive), e.g. face(35). The robot really turns — each 15° costs one "
    "env turn step; only the sub-15° remainder is a free micro-alignment — "
    "and you get the new view with fresh numbered places. Use it to check a "
    "side opening or a POTENTIAL region before committing to walk."
)
FACE_SCHEMA = {
    "properties": {"direction": {
        "title": "Direction",
        "description": "left | right | back | signed degrees (left positive)"}},
    "required": ["direction"],
    "title": "faceArguments",
    "type": "object",
}


# ── the system prompt's action-space section (dwp) ───────────────────────
def action_space_prompt(surface: str = "dwp") -> str:
    """The action-space paragraphs of the system prompt — same facts as the
    tool descriptions above, phrased for the briefing."""
    if surface != "dwp":
        raise ValueError(f"no action-space text for surface {surface!r}")
    return (
        "- goto(place): walk to one numbered place from the view you were "
        "just given. The robot turns toward it (real turn steps) and walks "
        "there one 0.25 m step at a time, re-checking the way as it goes. "
        "Each place sits in the MIDDLE of open floor and is at most a few "
        "metres away, so a goto is a short, deliberate hop — not a leap "
        "across the room.\n"
        "- step([actions]): the primitives, for TURNING, LINING UP and the "
        "FINAL METRE — not for covering ground. 0 = STOP (ends the "
        "episode), 1 = forward 0.25 m, 2 = turn left 15 deg, 3 = turn "
        "right 15 deg. Turns batch freely (step([2,2,2]) = 45 deg left; "
        "step([3]*6) = a quarter turn right) and every primitive is sensed "
        "as it executes. At most 2 forward moves per call are carried out; "
        "distance is what goto is for."
    )


# ── §20.4: the canonical map contract ────────────────────────────────────
# ONE source for the system prompt's MAP MEMORY input AND the per-turn
# IMAGE 2 label. The colour words quote render_model's actual constants
# (free (30,78,50) green · occupied (186,70,64) red · unknown (26,28,34)
# dark · potential (128,96,38) amber · visited (68,74,118) blue-purple),
# so the legend and the pixels cannot drift apart.
MAP_LEGEND = (
    "your accumulated 2.5D top-down memory, TWO PANELS. LEFT (GLOBAL): "
    "fixed to the WORLD — walls, your trail and the numbered places stay "
    "put while the yellow arrow (you) moves through them and turns with "
    "your real heading; turning your head does NOT rotate this panel, so "
    "loops and left/right relations across the whole walk stay readable. "
    "RIGHT (LOCAL): a small close-up with you at the centre facing UP — "
    "left in this panel is your left, same as the photo. Colours in both: "
    "GREEN = verified FREE floor, RED = OCCUPIED, DARK = UNKNOWN (never "
    "seen — not wall, not floor), AMBER = POTENTIAL (floor glimpsed past "
    "an occluder — worth going to look, never walked blind), BLUE-PURPLE "
    "= ground you already walked (recent trail brighter). Numbered "
    "circles are the places you can walk to (numbers also in the photo "
    "are the same place; a number ONLY on the map is a REMEMBERED place "
    "outside your current view — goto turns you to it), solid dots are "
    "landmarks in view NOW, dashed rings are remembered ones."
)
MAP_USE_RULES = (
    "The numbered places and the geometry lines say what you can execute "
    "RIGHT NOW; the map says where the ROUTE goes long-term. Never treat "
    "an unnumbered green or amber patch as something you can call — only "
    "numbered places are executable. Each turn, check three things on the "
    "map: (1) does the number you are about to take carry you along the "
    "instruction's route; (2) are you heading back onto your own "
    "blue-purple trail — on a LOOP WARNING prefer a direction that leaves "
    "the trail and still fits the instruction; (3) is the landmark you "
    "need in view now, already behind you, or still unverified. When the "
    "map registration reads as rough, trust the current photo and "
    "geometry for local safety and use remembered distances only as "
    "hints."
)


# ── first prompts (dwp) ──────────────────────────────────────────────────
# With observe withdrawn, the opening turn must CARRY the first look — the
# harness injects it (mini: opening_parts; SDK: the bootstrap images). This
# text is only honest when images ride the same message.
DWP_FIRST_PROMPT = (
    "Begin navigating. The pictures in this message are your opening look — "
    "each is labelled: the view you are FACING with the places you can walk "
    "to numbered on it, your accumulated top-down memory, and any extra "
    "labelled views of the turn. Decide your first move from them — every "
    "action returns a fresh view, so you never need to ask to look."
)

# …and the degraded text for an executor whose first message cannot carry
# images (bootstrap failed / unsupported): still no observe, still honest.
DWP_FIRST_PROMPT_NOIMG = (
    "Begin navigating. There is no look action: EVERY action returns the "
    "current view with the places you can walk to numbered on it. If you "
    "need a first look before committing to a direction, a small step "
    "turn (e.g. step([2,2]) — 30° left) is the cheapest probe."
)
