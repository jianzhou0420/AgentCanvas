"""bare — frozen briefing copy (exp_workspace contract).

The minimal two-tool briefing of the std board (MIP paper §3.2 / §4.2), forked
byte-identically from coding-agent/prompts.py's bare branch on 2026-08-18.
Both profiles of this arm (r2r, rxr) run the classic observe/step alternation
(auto_observe OFF — the paper's protocol on both benchmarks), so only the
separate-alternation variant is frozen here; the AUTO note variants never
applied to this arm's boards.
"""

_OBS_NOTE_SEP = ""
_STEP_NOTE_SEP = ""
_STEP_LOOP_SEP = (
    "- Alternate observing and stepping: look, decide where the instruction "
    "wants you to go next, move, look again."
)

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


def build_briefing(instruction: str, step_budget: int) -> str:
    return BARE_SYSTEM_PROMPT.format(
        instruction=instruction, budget=step_budget,
        obs_note=_OBS_NOTE_SEP,
        step_note=_STEP_NOTE_SEP,
        loop_rule=_STEP_LOOP_SEP,
    )
