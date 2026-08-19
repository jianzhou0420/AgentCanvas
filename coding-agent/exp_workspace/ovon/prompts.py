"""ovon — frozen briefing copy (exp_workspace contract).

The ObjectNav single-tool briefing (objnav_bridge_singlestep), shared
verbatim by the three OVON val splits; bare surface.
Forked byte-identically from coding-agent/prompts.py on 2026-08-18; the
builder bakes this arm's knobs — no condition flags cross the folder wall.
"""

OBJNAV_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through this tool:

- step(actions): execute movement actions in order, then return the \
robot's forward-facing camera view (an RGB image) after the last action. \
0 = STOP (permanently ends the episode — declares you have found the \
target), 1 = move forward 0.25 m, 2 = turn left 30 degrees, 3 = turn \
right 30 degrees, 4 = tilt the camera up 30 degrees, 5 = tilt the camera \
down 30 degrees (tilt changes the camera pitch only, not your position or \
heading). Calling step([]) with no actions returns the current view \
without moving.

Your task is object-goal navigation: no route is given — search the \
building until you find the target object, walk up to it, and stop there.

Target object: "{goal}"

Rules:
- Call step([]) once at the start to see where you are.
- Every step call shows you the view after your actions: study it, decide \
where to search next, and issue the next step. You never need a separate \
look call.
- Explore efficiently: sweep toward where a "{goal}" is most likely to \
be, and avoid re-walking areas you have already ruled out.
- You have a budget of {budget} movement actions.{budget_note}
- You succeed only if you issue action 0 (STOP) while within 0.5 meters \
of a "{goal}". Any instance counts. STOP is permanent — issue it only \
when you can see the target and are standing right next to it.{stop_note}
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around \
when unsure; if you tilt the camera, level it again before moving on.
- Work autonomously until you stop; nobody can answer questions.
"""


def build_briefing(instruction: str, step_budget: int) -> str:
    # bare surface: no budget broadcast, no STOP gate — both note slots
    # empty (the instruction is the goal-category text).
    return OBJNAV_SYSTEM_PROMPT.format(
        goal=instruction, budget=step_budget,
        budget_note="", stop_note="",
    )
