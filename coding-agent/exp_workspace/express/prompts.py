"""express — frozen briefing copy (exp_workspace contract).

The EXPRESS-Bench briefing (express_bridge answer(text) surface):
free-form EQA over the discrete action space.
Forked byte-identically from coding-agent/prompts.py on 2026-08-18; the
builder bakes this arm's knobs — no condition flags cross the folder wall.
"""

EXPRESS_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 1 = move forward \
0.25 m, 2 = turn left 30 degrees, 3 = turn right 30 degrees. There is no \
stop action.
- answer(text): permanently END the episode by answering the question in \
free-form natural language (one or two sentences).

Your task is embodied question answering: explore the building until you \
can answer this question about it:

"{question}"

Rules:
- Alternate observing and stepping: look, decide where to go to find the \
evidence the question needs, move, look again.
- You have a budget of {budget} movement actions. If it runs out you can \
still observe and answer from where you stand.
- Your answer is judged on BOTH its correctness and whether your final \
camera view supports it — walk up to the relevant object or place and \
answer while it is in view.
- answer() is permanent — call it once you have seen enough evidence to be \
confident, and always answer before ending: an episode without answer() \
scores zero.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around when \
unsure.
- Work autonomously until you answer; nobody can help you.
"""


def build_briefing(instruction: str, step_budget: int) -> str:
    return EXPRESS_SYSTEM_PROMPT.format(
        question=instruction, budget=step_budget,
    )
