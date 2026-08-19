"""wp — frozen briefing copy (exp_workspace contract).

The waypoint-selection surface (wp_bridge): numbered candidates on a
4-view panorama, goto/stop. Classic observe/move alternation only
(auto_observe never applied to this arm's boards). wp_max_moves is a
PER-PROFILE knob (r2r 30 / rxr 45) the driver passes from the run
config, signature-gated.
Forked byte-identically from coding-agent/prompts.py on 2026-08-18; the
builder bakes this arm's knobs — no condition flags cross the folder wall.
"""

_OBS_NOTE_SEP = ""

_GOTO_LINE_SEP = (
    "goto(waypoint): walk to one numbered waypoint from the LATEST observe(). "
    "Moving invalidates the old numbers — observe() again after arriving."
)

_WP_LOOP_SEP = "- Alternate observing and moving: observe(), then move, then observe() again."

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


def build_briefing(instruction: str, step_budget: int, *,
                   wp_max_moves: int = 30) -> str:
    # step_budget is not part of the wp briefing (the move cap binds
    # first); accepted for the uniform folder-builder call shape.
    return WP_SYSTEM_PROMPT.format(
        instruction=instruction, wp_max_moves=wp_max_moves,
        obs_note=_OBS_NOTE_SEP,
        goto_line=_GOTO_LINE_SEP,
        loop_rule=_WP_LOOP_SEP,
    )
