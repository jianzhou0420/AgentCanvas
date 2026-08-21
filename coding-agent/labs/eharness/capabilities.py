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
    # primmap (user 2026-08-17, after jian's slamr2r_sdk_*_instr result):
    # primitives ONLY (0/1/2/3) plus TWO free read-only instruments — the
    # accumulated map as an on-demand tool (no candidate circles anywhere)
    # and an odometry pose readout. No goto, no veer, no recall: the A/B
    # against dwp is "the map as a queried instrument vs the map as a
    # pushed decision surface".
    #
    # SCOPED EXCEPTION to the no-coordinates hard rule (user ruling
    # 2026-08-17: "getPose 和网格线我觉得挺好的……我们可以测一个上限"):
    # get_pose numbers and the gridline labels exist ONLY on this surface,
    # as an upper-bound probe. The main harness line stays coordinate-free.
    "primmap": ("step", "get_map", "get_pose", "get_trajectory"),
    # slamdwp (user 2026-08-17 night): the FULL dwp surface — candidates,
    # goto, remembered, wings, facts all unchanged — but the accumulated
    # map is PULLED via get_map (free) instead of pushed every turn, and
    # the map itself is built by the ported SLAM integrate rule
    # (map_backend=slam). NO coordinates, NO gridlines: the no-coordinates
    # hard rule applies in full on this surface (get_map draws the exact
    # dwp rendering).
    # user 2026-08-17 深夜 trim: veer/recall withdrawn, guards off — the
    # lean surface is step + goto + get_map, nothing else.
    "dwptool": ("step", "goto", "get_map"),
    # slamstep (user 2026-08-17 终裁 + get_pose 追裁): NO DWP — the action
    # space is step (0/1/2/3); instruments are the SLAM+SAM map (get_map)
    # and the pose readout (get_pose, user: "这是我朋友的一个,可以上").
    # Coordinates sanctioned on this surface like primmap.
    "slamstep": ("step", "get_map", "get_pose"),
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


# ── primmap surface: step + get_map, nothing else ────────────────────────
PRIMMAP_STEP_DESC = (
    "Execute movement actions in order: 0=STOP (ends the episode — you "
    "succeed only if you are within 3 m of the instruction's endpoint), "
    "1=forward 0.25 m, 2=turn left 15°, 3=turn right 15°. Actions batch "
    "freely — step([2,2,2]) turns 45° left, step([1]*6) walks 1.5 m — and "
    "every movement is executed one primitive at a time with per-step "
    "sensing: the harness brakes early on a collision or a closing corridor "
    "and tells you why. Every action returns the fresh camera view. Your "
    "robot HAS A BODY (~0.2 m-radius cylinder) — it needs side clearance "
    "and cannot squeeze past furniture."
)

# slamdwp flavor (map_push=0, map_backend=slam): THE SLAM pipeline's own
# map, pulled on demand (user 2026-08-17 深夜: "SLAM 调图的整个流程是不变
# 的,就是用 SLAM 给的这个图"). The dwp menu's numbers ride the renderer's
# numbered-circle channel, so circle N on this map IS place N.
GET_MAP_DESC_DWP = (
    "Read the top-down occupancy map your onboard SLAM system builds "
    "automatically as you move (it already contains the opening ±60° "
    "sweep). UP on the image is your STARTING heading. On it: WHITE = "
    "free space you have seen, BLACK = obstacle, GRAY = unexplored; the "
    "blue ARROW is you pointing your current heading, the blue LINE is "
    "your path so far, the \"S\" circle is your start. Numbered green "
    "circles are the SAME numbered places as WALKABLE PLACES — circle N "
    "here is exactly goto(N). Landmarks the detector has recognised are "
    "tinted patches with their names, plus a \"landmarks\" list (name, "
    "dir_deg relative to your heading, distance). THE MAP IS YOUR "
    "LONG-TERM MEMORY: call this at every junction, to check whether "
    "you are looping back onto your own path line, and to place the "
    "instruction's landmarks. Free — no step cost, call it any time."
)

# slamstep flavor (map_tool + map_backend=slam): THE SLAM pipeline's map,
# no dwp menu — numbered green circles are FRONTIERS (jian's contract),
# tinted patches are SAM landmarks. No coordinates, no pose readout.
GET_MAP_DESC_SLAMSTEP = (
    "Read the top-down occupancy map your onboard SLAM system builds "
    "automatically as you move (it already contains an initial ±60° scan "
    "of your start location). UP on the image is your STARTING heading. "
    "On it: WHITE = free space you have seen, BLACK = obstacle, GRAY = "
    "unexplored; the blue ARROW is you pointing your current heading, "
    "the blue LINE is your path so far, the \"S\" circle is your start. "
    "Numbered green circles are FRONTIERS — openings into unexplored "
    "space — where circle N is frontier \"FN\" in the accompanying JSON "
    "(dir_deg relative to your current heading, positive = to your "
    "right, distance, size); frontier ids are STABLE across calls. "
    "Landmarks a detector has recognised as you moved show as tinted "
    "patches with their names, plus a \"landmarks\" list (name, dir_deg, "
    "distance) — detector estimates, possibly wrong or incomplete. Faint "
    "gridlines every 2 m are labeled with the same x/z coordinates "
    "get_pose() reports, and a scale bar shows 2 m. THE "
    "MAP IS YOUR LONG-TERM MEMORY and it is very important: call this "
    "at every junction, to check whether you are looping back onto your "
    "own path line, and to match the instruction's named objects to "
    "places. Free — no step cost, call it any time."
)

GET_MAP_DESC = (
    "Read your onboard top-down map, built automatically as you move (it "
    "already contains an initial ±60° scan of your start location). "
    "Returns an image fixed to the world — UP on the image is your "
    "STARTING heading (+z), right is +x. On it: GREEN = floor verified "
    "free, RED = obstacle, DARK = never seen, AMBER = floor glimpsed but "
    "not straight-line walkable from where it was seen, BLUE-PURPLE = "
    "ground you already walked; the yellow arrow is you pointing your "
    "current heading, the blue-purple line is your path, and landmarks "
    "the detector has seen are tinted patches labelled with their names "
    "at the remembered spots. Faint gridlines every 2 m are labeled with "
    "the same x/z coordinates get_pose() reports, and a scale bar shows "
    "2 m. Use it for \"where am I / where have I been / what is still "
    "unexplored\" instead of guessing from visual memory. Free — no step "
    "cost, call it any time. It is an odometry estimate built from your "
    "own camera, not ground truth."
)

GET_TRAJECTORY_DESC = (
    "Read your own path so far as (x, z) points in the same fixed frame as "
    "get_pose(), oldest first. Useful to check whether you are circling or "
    "which areas you already covered. Free — no step cost."
)

GET_POSE_DESC = (
    "Read the robot's odometry-estimated pose: position (x, z) in meters "
    "and heading yaw_deg, in a fixed frame anchored at your start pose "
    "(x = right and z = forward OF YOUR STARTING POSE; yaw_deg is 0 at "
    "your starting heading and INCREASES when you turn right). The same "
    "x/z coordinates label the gridlines on get_map(), so pose numbers "
    "place you on the map. Free — no step cost. It is an odometry "
    "estimate (blocked steps are detected and rolled back), not ground "
    "truth."
)

MAP_TOOL_ADDENDUM = (
    "You additionally have two free read-only instruments backed by your "
    "onboard odometry and mapping. They cost no steps and may be called "
    "any time:\n"
    "- get_pose(): your estimated position (x, z, meters) and heading "
    "(yaw_deg) in a fixed frame anchored at your start pose.\n"
    "- get_map(): your accumulated top-down map (see its description). "
    "The same x/z coordinates label its gridlines, so pose numbers place "
    "you on the map. Nothing pushes the map at you — call it whenever "
    "you need \"where am I / where have I been / what is still "
    "unexplored\", and usually before a big direction decision.\n"
    "These are estimates from your own motion and camera, not ground "
    "truth, and can drift slightly."
)

# slamstep: ONE instrument, no coordinates (2026-08-17 终裁)
SLAMSTEP_ADDENDUM = (
    "You additionally have two free read-only instruments backed by an "
    "onboard SLAM system. They cost no steps and may be called any time:\n"
    "- get_pose(): your estimated position (x, z, meters) and heading "
    "(yaw_deg) in a fixed frame anchored at your start pose; the same "
    "x/z coordinates label get_map()'s gridlines.\n"
    "- get_map(): a top-down occupancy map built automatically as you "
    "move (it already contains an initial ±60° scan of your start "
    "location), with your path line, stable-numbered FRONTIERS (openings "
    "into unexplored space, with direction relative to your current "
    "heading and distance), and named landmark patches a detector has "
    "recognised as you moved.\n"
    "THE MAP IS VERY IMPORTANT — it solves \"where am I / where have I "
    "been / what is still unexplored / where were the instruction's "
    "objects\" better than visual memory. Call it at every junction, "
    "before big direction decisions, and to check whether you are "
    "looping back onto your own path line. It is a SLAM estimate, not "
    "ground truth, and can drift slightly."
)

MAP_TOOL_FIRST_PROMPT = (
    "Begin navigating. The pictures in this message are your opening look — "
    "the view you are FACING plus labelled views 60° left and 60° right. "
    "Your onboard map already contains this whole opening sweep — call "
    "get_map() any time (free) to see it. Decide your first move from the "
    "pictures — every action returns a fresh view, so you never need to "
    "ask to look."
)


# ── the system prompt's action-space section (dwp / primmap) ─────────────
def action_space_prompt(surface: str = "dwp") -> str:
    """The action-space paragraphs of the system prompt — same facts as the
    tool descriptions above, phrased for the briefing."""
    if surface in ("primmap", "slamstep"):
        # user 2026-08-18: the instruments (get_map/get_pose) are described
        # ONCE — in the addendum below — not here as well; this section is
        # the movement tool only, with a pointer.
        return (
            "- step([actions]): your ONLY movement tool. 0 = STOP (ends "
            "the episode), 1 = forward 0.25 m, 2 = turn left 15 deg, 3 = "
            "turn right 15 deg. Actions batch freely (step([2,2,2]) = 45 "
            "deg left; step([1]*6) = 1.5 m forward; step([3]*6) = a "
            "quarter turn right) and every primitive is sensed as it "
            "executes — the harness brakes early on a collision or a "
            "closing corridor and tells you why.\n"
            "- get_map() / get_pose(): free read-only instruments, "
            "described below."
        )
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
# IMAGE 2 label. stage3 P0-1: IMAGE 2 is the single anchor-fixed
# AnchorMap.render() — the colour words quote its actual constants
# (free ~(24,66,46)+ green gradient · occupied (176,68,62) red · unknown
# (26,28,34) dark · potential (110,84,36) amber · visited (68,74,118)
# blue-purple), so the legend and the pixels cannot drift apart.
MAP_LEGEND = (
    "your accumulated 2.5D top-down memory, ONE map fixed to the WORLD: "
    "walls, your trail and the numbered places stay put while the yellow "
    "arrow (you) moves through them and turns with your real heading; "
    "turning your head does NOT rotate the map, so loops and left/right "
    "relations across the whole walk stay readable. Colours: GREEN = "
    "verified FREE floor, RED = OCCUPIED, DARK = UNKNOWN (never seen — "
    "not wall, not floor), AMBER = seen-but-not-straight-line (floor "
    "glimpsed past an occluder or too tight on the straight approach — "
    "turn or go around first, it is never walked blind; walking verifies it green), BLUE-PURPLE = "
    "ground you already walked. Numbered circles are the places you can "
    "walk to — a SOLID circle is also in the photo; a DASHED circle is "
    "outside your current view (remembered from the map, or measured in "
    "an opening side view — goto turns you to it and re-checks with "
    "fresh eyes before walking). Landmarks the "
    "detector has seen are tinted patches with their name written at "
    "the remembered spot."
)
MAP_USE_RULES = (
    "The numbered WALKABLE PLACES lines say what you can execute RIGHT "
    "NOW; the map says where the ROUTE goes long-term. Never treat "
    "an unnumbered green or amber patch as something you can call — only "
    "numbered places are executable. Each turn, check three things on the "
    "map: (1) does the number you are about to take carry you along the "
    "instruction's route; (2) are you heading back onto your own "
    "blue-purple trail — on a loop-risk MAP WARNING prefer a direction that leaves "
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
    "to numbered on it, your accumulated top-down memory, and labelled "
    "OPENING VIEWS to your left and right with the SAME numbered places "
    "marked where they fall. It is ONE menu across every image — a number "
    "is the same place everywhere it appears. Decide your first move from "
    "the whole spread — every action returns a fresh view, so you never "
    "need to ask to look."
)

# slamdwp first prompt (map_push=0): the opening spread carries NO map
# image — get_map is where the map lives, and the text says so.
DWP_MAPTOOL_FIRST_PROMPT = (
    "Begin navigating. The pictures in this message are your opening look — "
    "each is labelled: the view you are FACING with the places you can walk "
    "to numbered on it, and labelled OPENING VIEWS to your left and right "
    "with the SAME numbered places marked where they fall. It is ONE menu "
    "across every image. Your accumulated map is NOT attached — call "
    "get_map() (free) whenever you want it; it already contains the whole "
    "opening sweep. Decide your first move from the spread — every action "
    "returns a fresh view, so you never need to ask to look."
)

# …and the degraded text for an executor whose first message cannot carry
# images (bootstrap failed / unsupported): still no observe, still honest.
DWP_FIRST_PROMPT_NOIMG = (
    "Begin navigating. There is no look action: EVERY action returns the "
    "current view with the places you can walk to numbered on it. If you "
    "need a first look before committing to a direction, a small step "
    "turn (e.g. step([2,2]) — 30° left) is the cheapest probe."
)
